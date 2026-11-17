import json
import logging
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.app import entrypoint
from src.app.decider import ClefDecider, TextDecider
from src.app.args.data import DataArguments
from src.app.args.model import ModelArguments
from src.eval.sequence import iptc_31
from src.eval.sequence.iptc_31 import decision


def test_missing_startlux_dependency_is_isolated_and_explains_how_to_install():
    # A fresh process reproduces an unsynced remote environment without
    # uninstalling the SDK from the developer's environment.
    script = '''
import importlib.abc
import sys

class MissingStartLux(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'startlux_decide':
            raise ModuleNotFoundError("No module named 'startlux_decide'", name=fullname)

sys.meta_path.insert(0, MissingStartLux())
from src.eval.sequence.iptc_31 import decision
from src.app.decider import ClefDecider, GlideDecider, TextDecider
from src.app.args.model import ModelArguments

assert callable(ClefDecider) and callable(GlideDecider)
assert 'startlux_decide' not in sys.modules
try:
    TextDecider.create(ModelArguments(model_name_or_path='startlux-models/StartLux-Decision-v1.1-4B-Instant'),
                       {'revision': 'a' * 40, 'device': 'cpu'})
except ImportError as error:
    assert 'uv sync --extra linear-attention' in str(error), str(error)
    assert isinstance(error.__cause__, ModuleNotFoundError)
else:
    raise AssertionError('Missing StartLux dependency was not reported')
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


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


def test_topic_definitions_are_optional_and_require_complete_descriptions():
    labels = [{'iptc_code': '20000038', 'label_name': 'culture',
               'label_description': 'The ideas, customs, arts, traditions of a particular group of persons'}]
    plain = decision._topic_questions(labels)
    described = decision._topic_questions(labels, use_label_descriptions=True)
    assert 'IPTC definition:' not in plain['20000038']['instructions']
    assert described['20000038']['instructions'] == (
        plain['20000038']['instructions'] + '\nIPTC definition: ' + labels[0]['label_description']
    )
    with pytest.raises(ValueError, match='Missing IPTC topic description'):
        decision._topic_questions([{'iptc_code': '20000038', 'label_name': 'culture'}], True)
    with pytest.raises(ValueError, match='must be a boolean'):
        decision._topic_questions(labels, 'true')


def test_predict_decodes_true_option_by_id_and_batches_without_reference_labels():
    records, batches = [], []

    def encode(tokenizer, record, **kwargs):
        assert record['state'] == {'article': f"Article {record['id']}"}
        assert kwargs['max_length'] == 16384 and kwargs['max_state_tokens'] == 8192
        assert 'processor' not in kwargs
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

    predictor = ClefDecider.__new__(ClefDecider)
    predictor.api = SimpleNamespace(encode_record=encode, collate_records=collate)
    predictor.tokenizer = SimpleNamespace(pad_token_id=0)
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
        ClefDecider(ModelArguments(model_name_or_path='Cloudflare/clef-flash'), {'device': 'cuda', 'revision': 'a' * 40})


@pytest.mark.parametrize('pad_token_id', [0, None])
def test_text_loader_restores_head_and_tokenizer_without_image_processor(tmp_path, monkeypatch, pad_token_id):
    from safetensors.torch import save_file
    from transformers import AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration

    class Head(torch.nn.Linear):
        def __init__(self, hidden_size, width):
            super().__init__(hidden_size, width, bias=False)

    class Model(torch.nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone, self.head = backbone, head

    expected = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    (tmp_path / 'joint_head_config.json').write_text(json.dumps({'hidden_size': 3, 'width': 2}))
    save_file({'weight': expected}, tmp_path / 'joint_head.safetensors')
    tokenizer = SimpleNamespace(pad_token_id=pad_token_id, eos_token_id=2)
    backbone = SimpleNamespace(config=SimpleNamespace(use_cache=True))

    def load_tokenizer(path):
        assert path == tmp_path
        return tokenizer

    def load_backbone(path, **kwargs):
        assert path == tmp_path
        assert kwargs == {'dtype': torch.float32, 'device_map': {'': 'cpu'}, 'attn_implementation': 'sdpa'}
        return backbone

    def no_processor(*args, **kwargs):
        pytest.fail('Text-only evaluation must not initialize image/video processors')

    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', load_tokenizer)
    monkeypatch.setattr(Qwen3_5ForConditionalGeneration, 'from_pretrained', load_backbone)
    monkeypatch.setattr(AutoProcessor, 'from_pretrained', no_processor)
    model, loaded_tokenizer = ClefDecider._load_text_model(
        SimpleNamespace(JointSchemaHead=Head, ClefModel=Model), tmp_path,
        torch.device('cpu'), torch.float32, attn_implementation='sdpa',
    )
    assert loaded_tokenizer is tokenizer
    assert tokenizer.pad_token_id == (0 if pad_token_id == 0 else 2)
    assert not backbone.config.use_cache and not model.training
    torch.testing.assert_close(model.head.weight, expected)


@pytest.mark.parametrize('single_language', [False, True])
@pytest.mark.parametrize('model_config', ['clef-flash', 'glide-2b', 'glide-9b', 'starlux1.1-4b'])
@pytest.mark.parametrize('use_descriptions', [False, True])
def test_decision_uses_prepared_test_only_filters_ids_and_loads_once(
    tmp_path, monkeypatch, single_language, model_config, use_descriptions,
):
    paths = entrypoint._ensure_dirs(tmp_path, 'eval', 'sequence', 'iptc-31')
    prepare = paths.get_script_ctx_path('data', 'prepare')
    prepare.mkdir(parents=True)
    merged, _ = entrypoint._load_and_merge_configs(Path('conf'), 'sequence', 'iptc-31', [model_config])
    model_args, args, _, _ = entrypoint._parse_hf_args(merged)
    method = args.attributes['sequence']['method']
    args.attributes['sequence']['methods'][method]['max_samples'] = 2
    args.attributes['sequence']['parameters'] = {'use_label_descriptions': use_descriptions}
    args.attributes['sequence']['exclude_test_ids'] = ['sl:206']
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

    class FakePredictor(TextDecider):
        tracks_truncation = model_config != 'clef-flash'

        def metadata(self):
            result = super().metadata()
            if method in ('glide', 'startlux'):
                result['input_tokens_scope'] = 'sum of per-question prompt tokens'
            if method == 'glide':
                result['mask_mapping_compatibility'] = self.mask_mapping_compatibility
            if method == 'startlux':
                result.update(backend=self.backend_metadata, temperature_by_type=self.temperature_by_type, output_tokens=0)
            return result

        def __init__(self, model_args, parameters):
            assert model_args.model_name_or_path == {
                'clef-flash': 'Cloudflare/clef-flash', 'glide-2b': 'fastino/GLiDE-2B', 'glide-9b': 'fastino/GLiDE-9B',
                'starlux1.1-4b': 'startlux-models/StartLux-Decision-v1.1-4B-Instant',
            }[model_config]
            assert len(parameters['revision']) == 40
            self.model_args, self.max_length = model_args, model_args.max_seq_length
            self.input_tokens = 0
            self.truncated_articles = 0
            self.mask_mapping_compatibility = True
            self.backend_metadata = {'prompt_attention': 'bidirectional', 'images': False}
            self.temperature_by_type = {'noul': 1.2662}
            self.closed = False
            self.device, self.dtype = torch.device('cpu'), torch.float32
            instances.append(self)

        def _predict(self, ids, texts, questions, batch_size):
            assert 'sl:206' not in ids
            assert len(questions) == 31 and all(q['type'] == 'noul' for q in questions.values())
            assert all(('IPTC definition:' in q['instructions']) == use_descriptions for q in questions.values())
            assert texts == [f'Article {sample_id}' for sample_id in ids]
            calls.extend(ids.tolist())
            self.input_tokens += len(ids) * 100
            return np.full((len(ids), len(questions)), 0.1, dtype=np.float32)

        def close(self):
            self.closed = True

    monkeypatch.setattr(TextDecider, 'create', classmethod(lambda cls, model, params: FakePredictor(model, params)))
    monkeypatch.setattr(iptc_31, 'paths', paths, raising=False)
    monkeypatch.setattr(iptc_31, 'logger', logging.getLogger(__name__), raising=False)
    iptc_31.main(args, model_args)
    assert len(instances) == 1
    assert instances[0].closed
    subset = 'iptc-31.sl' if single_language else 'iptc-31'
    variant = '.descriptions' if use_descriptions else ''
    stem = f'{subset}{variant}.limit2.{model_config}.{method}'
    metadata = json.loads((paths.context / f'{stem}.metrics.json').read_text())
    assert metadata['num_train'] == 0 and metadata['feature_type'] == 'text'
    assert metadata['supervision'] == 'zero_shot_decision'
    assert metadata['num_labels'] == 31 and metadata['excluded_test_ids'] == ['sl:206']
    assert metadata['num_test_excluded'] == 1
    assert metadata['num_test'] == (2 if single_language else 4)
    assert metadata['num_test_before_limit'] == (3 if single_language else 7)
    assert metadata['input_tokens'] == (200 if single_language else 400)
    assert metadata['prompt_variant'] == ('name_and_definition' if use_descriptions else 'name_only')
    if use_descriptions:
        assert metadata['label_description_metadata']['language'] == 'en-US'
        assert metadata['label_description_metadata']['license_url'] == 'https://creativecommons.org/licenses/by/4.0/'
    if method == 'glide':
        assert metadata['parameters']['token_budget'] == 16384
        assert metadata['num_test_truncated'] == 0 and metadata['mask_mapping_compatibility']
        assert metadata['input_tokens_scope'] == 'sum of per-question prompt tokens'
    if method == 'startlux':
        assert metadata['backend'] == {'prompt_attention': 'bidirectional', 'images': False}
        assert metadata['temperature_by_type']['noul'] == 1.2662 and metadata['output_tokens'] == 0
        assert metadata['num_test_truncated'] == 0
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
        decision.evaluate_decision(args, ModelArguments(), None, None, method='clef')
