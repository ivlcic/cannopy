import json
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.app import entrypoint
from src.app.labeler import MultilabelLabeler
from src.eval.sequence import iptc_31
from src.eval.sequence.iptc_31 import decision
from src.eval.sequence.iptc_31.methods.bge_m3_clef import RAG_INSTRUCTION, BgeM3ClefRetriever, encode_with_examples


DEFINITIONS = [
    {'iptc_code': '01000000', 'label_name': 'arts and culture'},
    {'iptc_code': '20000002', 'label_name': 'entertainment'},
]


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


def write_sidecar(path, ids, vectors):
    # Accessing this annotation array with allow_pickle=False raises: retrieval
    # must only read IDs and embeddings, even when test labels are available.
    np.savez(path, ids=np.asarray(ids), embeddings=np.asarray(vectors, dtype=np.float32),
             labels=np.asarray([{'must_not_read': True}], dtype=object))


def test_retrieval_aligns_by_id_normalizes_and_reads_training_labels_only(tmp_path):
    write_rows(tmp_path / 'iptc-31.sl.train.jsonl', [
        {'id': 'sl:b', 'text': 'Second article', 'label': ['20000002']},
        {'id': 'sl:a', 'text': 'First article', 'label': ['01000000', '20000002']},
    ])
    write_sidecar(tmp_path / 'iptc-31.sl.bge-m3.train.npz', ['sl:a', 'sl:b'], [[100, 0], [0, 2]])
    write_sidecar(tmp_path / 'iptc-31.sl.bge-m3.test.npz', ['sl:y', 'sl:x'], [[0, 9], [3, 0]])
    labeler = MultilabelLabeler(labels=[item['iptc_code'] for item in DEFINITIONS])
    retriever = BgeM3ClefRetriever(tmp_path, tmp_path, 'iptc-31.sl', labeler, DEFINITIONS)
    first, second = retriever.retrieve(np.asarray(['sl:x', 'sl:y']))
    assert [item['id'] for item in first] == ['sl:a', 'sl:b']
    assert [item['id'] for item in second] == ['sl:b', 'sl:a']
    assert first[0]['text'] == 'First article' and first[0]['similarity'] == pytest.approx(1)
    assert first[0]['topics'] == [{'code': item['iptc_code'], 'name': item['label_name']} for item in DEFINITIONS]
    assert second[0]['text'] == 'Second article'
    assert second[0]['topics'] == [{'code': '20000002', 'name': 'entertainment'}]
    with pytest.raises(ValueError, match='missing IDs: sl:missing'):
        retriever.retrieve(np.asarray(['sl:missing']))

    write_sidecar(tmp_path / 'iptc-31.sl.bge-m3.test.npz', ['sl:a'], [[1, 0]])
    with pytest.raises(ValueError, match='training and query IDs overlap'):
        BgeM3ClefRetriever(tmp_path, tmp_path, 'iptc-31.sl', labeler, DEFINITIONS)
    write_sidecar(tmp_path / 'iptc-31.sl.bge-m3.test.npz', ['hr:x'], [[1, 0]])
    with pytest.raises(ValueError, match='same-language'):
        BgeM3ClefRetriever(tmp_path, tmp_path, 'iptc-31.sl', labeler, DEFINITIONS)


class CharacterTokenizer:
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[ord(char) for char in text])

    def decode(self, ids, **kwargs):
        return ''.join(chr(value) for value in ids)


class TextAPI:
    render = staticmethod(lambda value: value if isinstance(value, str) else
                          json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True))

    @classmethod
    def encode_record(cls, tokenizer, record, max_length=16384, max_state_tokens=None):
        state = tokenizer(cls.render(record['state'])).input_ids
        state = state[:min(max_length - 20, max_state_tokens or max_length)]
        return SimpleNamespace(
            input_ids=[0] * 20 + state, state=record['state'],
            questions=[SimpleNamespace(question_id=code, option_ids=('false', 'true'))
                       for code in reversed(record['questions'])],
        )

    @staticmethod
    def collate_records(encoded, pad_token_id, device):
        return encoded


def test_context_budget_preserves_target_and_complete_topics_and_drops_examples():
    tokenizer, api = CharacterTokenizer(), TextAPI()
    text = 'The complete target article.'
    questions = {'01000000': {'type': 'noul'}}
    topics = [{'code': item['iptc_code'], 'name': item['label_name']} for item in DEFINITIONS]
    example = {'id': 'sl:train', 'text': 'A' * 100, 'topics': topics, 'similarity': 0.8}
    full_state = {'article': text, 'examples': [{'article_excerpt': 'A' * 100, 'assigned_topics': topics}],
                  'instructions': RAG_INSTRUCTION}
    # Exactly enough room for 25 excerpt tokens, including all labels and guidance.
    budget = len(api.render(full_state)) - 75
    encoded, audit = encode_with_examples(api, tokenizer, 'sl:test', text, questions, [example, example],
                                         max_length=budget + 20, max_state_tokens=10000,
                                         max_excerpt_tokens=100, schema_tokens=20)
    assert len(encoded.input_ids) == budget + 20
    assert encoded.state['article'] == text
    assert encoded.state['instructions'] == RAG_INSTRUCTION
    assert encoded.state['examples'] == [{'article_excerpt': 'A' * 25, 'assigned_topics': topics}]
    assert audit['examples'] == [{'id': 'sl:train', 'similarity': 0.8, 'excerpt_tokens': 25}]
    assert audit['examples_dropped'] == 1 and not audit['target_truncated']

    # A full target leaves no space for guidance plus an example: keep the
    # identical state used by plain Clef, with no additional clipping.
    target_budget = len(api.render({'article': text}))
    encoded, audit = encode_with_examples(api, tokenizer, 'sl:test', text, questions, [example],
                                         max_length=10000, max_state_tokens=target_budget,
                                         max_excerpt_tokens=100, schema_tokens=20)
    assert encoded.state == {'article': text}
    assert len(encoded.input_ids) == target_budget + 20
    assert not audit['examples'] and not audit['target_truncated']

    encoded, audit = encode_with_examples(api, tokenizer, 'sl:test', text, questions, [example],
                                         max_length=10000, max_state_tokens=10,
                                         max_excerpt_tokens=100, schema_tokens=20)
    baseline = api.encode_record(tokenizer, {'state': {'article': text}, 'questions': questions},
                                 max_length=10000, max_state_tokens=10)
    assert encoded.input_ids == baseline.input_ids
    assert not audit['examples'] and audit['target_truncated']


@pytest.mark.parametrize('single_language', [False, True])
def test_rag_evaluation_filters_before_retrieval_and_saves_separately(tmp_path, monkeypatch, single_language):
    paths = entrypoint._ensure_dirs(tmp_path, 'eval', 'sequence', 'iptc-31')
    prepare, embed = [paths.get_script_ctx_path('data', name) for name in ('prepare', 'embed')]
    prepare.mkdir(parents=True)
    embed.mkdir(parents=True)
    merged, _ = entrypoint._load_and_merge_configs(Path('conf'), 'sequence', 'iptc-31', ['clef-flash', 'bge-m3_clef'])
    model_args, args, _, _ = entrypoint._parse_hf_args(merged)
    args.attributes['sequence']['parameters'] = {'max_samples': 2}
    args.attributes['sequence']['threshold'] = 0.4
    if single_language:
        args = replace(args, source=replace(args.source, select=replace(args.source.select, subset='iptc-31.sl')))
    for lang in ('hr', 'sl'):
        subset = f'iptc-31.{lang}'
        (prepare / f'{subset}.labels.csv').write_text('id,name\n01000000,arts\n')
        train_ids = [f'{lang}:train{number}' for number in range(3)]
        train = [{'id': sample_id, 'text': f'Training {sample_id}', 'label': ['01000000', '20000002']}
                 for sample_id in reversed(train_ids)]
        write_rows(prepare / f'{subset}.train.jsonl', train)
        write_sidecar(embed / f'{subset}.bge-m3.train.npz', train_ids, np.eye(3))
        test_ids = [f'{lang}:{sample_id}' for sample_id in ('206', 'keep1', 'keep2', 'keep3')]
        write_rows(prepare / f'{subset}.test.jsonl', [
            {'id': sample_id, 'text': f'Target {sample_id}', 'label': ['01000000']} for sample_id in test_ids
        ])
        # No vector for the excluded Slovenian document. Gold labels are poison
        # arrays in the sidecars; retrieval and inference must never access them.
        query_ids = [sample_id for sample_id in reversed(test_ids) if sample_id != 'sl:206']
        write_sidecar(embed / f'{subset}.bge-m3.test.npz', query_ids, [[1, 0.2, 0.1]] * len(query_ids))
    source_contents = {path: path.read_bytes() for root in (prepare, embed) for path in root.iterdir()}
    baseline = paths.context / 'iptc-31.clef-flash.clef.metrics.json'
    baseline.write_text('Existing plain Clef results\n')
    instances, records = [], []

    class Model:
        def __call__(self, encoded):
            assert not torch.is_grad_enabled()
            for record in encoded:
                assert len(record.state['examples']) == 3
                assert all(len(item['assigned_topics']) == 2 for item in record.state['examples'])
                assert all('Target ' not in item['article_excerpt'] for item in record.state['examples'])
            records.extend(encoded)
            return [[torch.tensor([0.2, 0.8]).log() for _ in record.questions] for record in encoded]

    class FakePredictor(decision.ClefPredictor):
        def __init__(self, model_args, parameters):
            assert model_args.model_name_or_path == 'Cloudflare/clef-flash'
            assert parameters['revision'] == '22b75507981345858dd152fbd9a6f8c0c4068b0a'
            assert parameters['device'] == 'cuda' and parameters['max_samples'] == 2
            self.api, self.tokenizer, self.model = TextAPI(), CharacterTokenizer(), Model()
            self.device, self.dtype = torch.device('cpu'), torch.float32
            self.max_length, self.max_state_tokens = 16384, parameters['max_state_tokens']
            self.input_tokens, self.retrieval_audit = 0, []
            instances.append(self)

    monkeypatch.setattr(decision, 'ClefPredictor', FakePredictor)
    monkeypatch.setattr(iptc_31, 'paths', paths, raising=False)
    monkeypatch.setattr(iptc_31, 'logger', logging.getLogger(__name__), raising=False)
    if single_language:
        # Exercise the direct CLI callable as well as main's config dispatch.
        iptc_31.bge_m3_clef(args, model_args)
    else:
        iptc_31.main(args, model_args)
    subset = 'iptc-31.sl' if single_language else 'iptc-31'
    stem = f'{subset}.limit2.clef-flash.bge-m3_clef'
    result = json.loads((paths.context / f'{stem}.metrics.json').read_text())
    assert len(instances) == 1
    assert result['num_test'] == (2 if single_language else 4)
    assert result['num_train'] == (3 if single_language else 6)
    assert result['num_labels'] == 31 and result['threshold'] == 0.4
    assert result['excluded_test_ids'] == ['sl:206']
    assert result['method'] == 'bge-m3_clef' and result['supervision'] == 'retrieval_augmented_decision'
    assert result['parameters']['retrieval_model'] == 'bge-m3' and result['parameters']['top_k'] == 3
    assert result['retrieval_metric'] == 'cosine' and result['retrieval_seconds'] >= 0
    assert result['retrieval_instructions'] == RAG_INSTRUCTION
    assert len(result['retrieval_examples']) == result['num_test'] == len(records)
    for record, audit in zip(records, result['retrieval_examples'], strict=True):
        assert record.state['article'] == f"Target {audit['query_id']}"
        assert audit['query_id'] != 'sl:206'
        assert len(audit['examples']) == 3
        assert all(item['id'].startswith(audit['query_id'].split(':')[0] + ':train') for item in audit['examples'])
        assert not audit['target_truncated'] and audit['examples_dropped'] == 0
    if not single_language:
        assert 'same-language training examples' in result['aggregation']
        assert len(result['train_data_files']) == len(result['train_sidecars']) == len(result['test_sidecars']) == 2
    with np.load(paths.context / f'{stem}.predictions.npz') as saved:
        assert saved['ids'].tolist() == [item['query_id'] for item in result['retrieval_examples']]
        np.testing.assert_allclose(saved['y_score'], 0.8, atol=1e-7)
        assert saved['y_true'].shape == (result['num_test'], 31)
    assert baseline.read_text() == 'Existing plain Clef results\n'
    assert all(path.read_bytes() == original for path, original in source_contents.items())


def test_missing_retrieval_inputs_fail_before_model_load(tmp_path, monkeypatch):
    paths = entrypoint._ensure_dirs(tmp_path, 'eval', 'sequence', 'iptc-31')
    prepare = paths.get_script_ctx_path('data', 'prepare')
    prepare.mkdir(parents=True)
    merged, _ = entrypoint._load_and_merge_configs(Path('conf'), 'sequence', 'iptc-31', ['clef-flash', 'bge-m3_clef'])
    model_args, args, _, _ = entrypoint._parse_hf_args(merged)
    args = replace(args, source=replace(args.source, select=replace(args.source.select, subset='iptc-31.sl')))
    write_rows(prepare / 'iptc-31.sl.test.jsonl', [{'id': 'sl:test', 'text': 'Test', 'label': ['01000000']}])
    write_rows(prepare / 'iptc-31.sl.train.jsonl', [{'id': 'sl:train', 'text': 'Train', 'label': ['01000000']}])

    def no_model(*args, **kwargs):
        pytest.fail('Model loading must not start before retrieval inputs are checked')

    monkeypatch.setattr(decision, 'ClefPredictor', no_model)
    with pytest.raises(FileNotFoundError, match='RAG embeddings not found'):
        decision.evaluate_clef(args, model_args, paths, logging.getLogger(__name__), method='bge-m3_clef')
