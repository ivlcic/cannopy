import json
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.app import entrypoint
from src.app.args.data import DataArguments
from src.app.args.model import ModelArguments
from src.eval.sequence import iptc_31
from src.eval.sequence.iptc_31 import decision


def test_topic_schema_is_multilabel_and_keeps_string_codes():
    questions = decision._topic_questions([
        {'iptc_code': '20000002', 'label_name': 'arts and entertainment'},
        {'iptc_code': '01000000', 'label_name': 'arts, culture, entertainment and media'},
    ])
    assert list(questions) == ['01000000', '20000002']
    assert all(question['type'] == 'noul' for question in questions.values())
    assert 'arts and entertainment' in questions['20000002']['instructions']
    assert 'broader and narrower' in questions['01000000']['instructions']
    with pytest.raises(ValueError, match='eight-digit'):
        decision._topic_questions([{'iptc_code': 1000000, 'label_name': 'arts'}])


def test_predict_decodes_true_option_by_id_and_batches_without_reference_labels():
    records, batches = [], []

    def encode(tokenizer, record, **kwargs):
        assert record['state'] == {'article': f"Article {record['id']}"}
        assert kwargs['max_length'] == 16384 and kwargs['max_state_tokens'] == 8192
        records.append(record)
        return SimpleNamespace(input_ids=[1, 2, 3], questions=[
            SimpleNamespace(question_id=code, option_ids=('false', 'true'))
            for code in reversed(record['questions'])
        ])

    def collate(encoded, pad_token_id, device):
        assert pad_token_id == 0 and device.type == 'cpu'
        batches.append(len(encoded))
        return encoded

    class Model:
        def __call__(self, encoded):
            assert not torch.is_grad_enabled()
            return [[torch.tensor([0.2, 0.8]).log() if q.question_id == '01000000'
                     else torch.tensor([0.8, 0.2]).log() for q in record.questions] for record in encoded]

    predictor = decision.ClefPredictor.__new__(decision.ClefPredictor)
    predictor.api = SimpleNamespace(encode_record=encode, collate_records=collate)
    predictor.processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    predictor.model = Model()
    predictor.device = torch.device('cpu')
    predictor.max_length, predictor.max_state_tokens, predictor.input_tokens = 16384, 8192, 0
    ids = np.asarray(['sl:1', 'sl:2', 'sl:3'])
    scores = predictor.predict(ids, [f'Article {sample_id}' for sample_id in ids],
                               {'01000000': {'type': 'noul'}, '20000002': {'type': 'noul'}}, 2)
    np.testing.assert_allclose(scores, [[0.8, 0.2]] * 3, atol=1e-7)
    assert batches == [2, 1] and predictor.input_tokens == 9
    assert scores.dtype == np.float32 and len(records) == 3


def test_loader_fails_before_download_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    def no_download(*args, **kwargs):
        pytest.fail('Model download must not start without CUDA')

    monkeypatch.setattr('huggingface_hub.snapshot_download', no_download)
    with pytest.raises(RuntimeError, match='no model weights were downloaded'):
        decision.ClefPredictor(ModelArguments(model_name_or_path='Cloudflare/clef-flash'), {'device': 'cuda'})


@pytest.mark.parametrize('single_language', [False, True])
def test_clef_uses_prepared_test_only_filters_ids_and_loads_once(tmp_path, monkeypatch, single_language):
    paths = entrypoint._ensure_dirs(tmp_path, 'eval', 'sequence', 'iptc-31')
    prepare = paths.get_script_ctx_path('data', 'prepare')
    prepare.mkdir(parents=True)
    merged, _ = entrypoint._load_and_merge_configs(Path('conf'), 'sequence', 'iptc-31', ['clef-flash'])
    model_args, args, _, _ = entrypoint._parse_hf_args(merged)
    args.attributes['sequence']['methods']['clef']['max_samples'] = 2
    if single_language:
        args = replace(args, source=replace(args.source, select=replace(args.source.select, subset='iptc-31.sl')))
    for lang in ('hr', 'sl'):
        # No training files or embeddings are present.
        (prepare / f'iptc-31.{lang}.labels.csv').write_text('id,name\n01000000,arts\n')
        samples = [{'id': f'{lang}:{sample_id}', 'text': f'Article {lang}:{sample_id}', 'label': ['01000000']}
                   for sample_id in ('206', 'keep1', 'keep2', 'keep3')]
        (prepare / f'iptc-31.{lang}.test.jsonl').write_text(''.join(json.dumps(sample) + '\n' for sample in samples))
    source_contents = {path: path.read_bytes() for path in prepare.iterdir()}
    calls, instances = [], []

    class FakePredictor:
        def __init__(self, model_args, parameters):
            assert model_args.model_name_or_path == 'Cloudflare/clef-flash'
            self.input_tokens = 0
            self.device, self.dtype = torch.device('cpu'), torch.float32
            instances.append(self)

        def predict(self, ids, texts, questions, batch_size):
            assert 'sl:206' not in ids
            assert len(questions) == 31 and all(q['type'] == 'noul' for q in questions.values())
            assert texts == [f'Article {sample_id}' for sample_id in ids]
            calls.extend(ids.tolist())
            self.input_tokens += len(ids) * 100
            return np.full((len(ids), len(questions)), 0.1, dtype=np.float32)

    monkeypatch.setattr(decision, 'ClefPredictor', FakePredictor)
    monkeypatch.setattr(iptc_31, 'paths', paths, raising=False)
    monkeypatch.setattr(iptc_31, 'logger', logging.getLogger(__name__), raising=False)
    iptc_31.main(args, model_args)
    assert len(instances) == 1
    subset = 'iptc-31.sl' if single_language else 'iptc-31'
    stem = f'{subset}.limit2.clef-flash.clef'
    metadata = json.loads((paths.context / f'{stem}.metrics.json').read_text())
    assert metadata['num_train'] == 0 and metadata['feature_type'] == 'text'
    assert metadata['supervision'] == 'zero_shot_decision'
    assert metadata['num_labels'] == 31 and metadata['excluded_test_ids'] == ['sl:206']
    assert metadata['num_test_excluded'] == 1
    assert metadata['num_test'] == (2 if single_language else 4)
    assert metadata['num_test_before_limit'] == (3 if single_language else 7)
    assert metadata['input_tokens'] == (200 if single_language else 400)
    if not single_language:
        assert 'own training set' not in metadata['aggregation']
        assert len(metadata['test_data_files']) == 2 and 'test_data' not in metadata
    with np.load(paths.context / f'{stem}.predictions.npz') as saved:
        assert saved['ids'].tolist() == calls
        assert saved['y_score'].shape == saved['y_true'].shape == (metadata['num_test'], 31)
    assert len(list(paths.context.glob('*.metrics.json'))) == 1
    assert all(path.read_bytes() == content for path, content in source_contents.items())


def test_clef_rejects_optuna_before_loading_a_model():
    args = DataArguments(attributes={'sequence': {'optuna': {'trials': 50}}})
    with pytest.raises(ValueError, match='Clef is a zero-shot'):
        decision.evaluate_clef(args, ModelArguments(), None, None)
