import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.app import entrypoint
from src.app.args.data import DataArguments
from src.app.decider import TextDecider
from src.app.labeler import MultilabelLabeler
from src.eval.sequence import iptc_31
from src.eval.sequence.iptc_31.memory import LabelMemory, memory_settings


A, B, C = '01000000', '15000000', '16000000'


def store_memory(directory, language='sl'):
    directory.mkdir(parents=True, exist_ok=True)
    train = directory / f'iptc-31.{language}.bge-m3.train.npz'
    query = directory / f'iptc-31.{language}.bge-m3.test.npz'
    np.savez_compressed(train, ids=np.array([f'{language}:train-a', f'{language}:train-b', f'{language}:train-c']),
                        embeddings=np.array([[1, 0], [.8, .6], [-1, 0]]),
                        labels=np.array([[A], [B, C], [C]], dtype=object))
    # Intentionally reorder IDs and poison gold labels. Retrieval needs only vectors.
    np.savez_compressed(query, ids=np.array([f'{language}:none', f'{language}:a', f'{language}:b']),
                        embeddings=np.array([[0, -1], [2, 0], [-1, 0]]),
                        labels=np.array([['not-an-iptc-code']] * 3, dtype=object))
    return train, query


def make_memory(directory, threshold=.79, top_n=2):
    args = DataArguments(dataset_name='iptc-31')
    args = replace(args, source=replace(args.source, select=replace(args.source.select, subset='iptc-31.sl')))
    return LabelMemory(directory, args, {'model': 'bge-m3', 'threshold': threshold, 'top_n': top_n},
                       MultilabelLabeler(labels=[A, B, C]))


def test_memory_selects_union_by_cosine_aligns_ids_and_ignores_test_labels(tmp_path):
    store_memory(tmp_path)
    memory = make_memory(tmp_path)
    selected, metadata = memory.retrieve(['sl:a', 'sl:none', 'sl:b'])
    assert selected == [[A, B, C], [], [C]]
    assert [row['id'] for row in metadata['queries']] == ['sl:a', 'sl:none', 'sl:b']
    assert [row['id'] for row in metadata['queries'][0]['neighbors']] == ['sl:train-a', 'sl:train-b']
    assert [row['similarity'] for row in metadata['queries'][0]['neighbors']] == pytest.approx([1., .8])
    assert metadata['num_queries_without_candidates'] == 1 and metadata['num_candidate_labels'] == 4
    assert metadata['sources'][0]['num_train'] == 3
    one, _ = make_memory(tmp_path, top_n=1).retrieve(['sl:a'])
    assert one == [[A]]
    all_labels, _ = make_memory(tmp_path, threshold=-1., top_n=100).retrieve(['sl:a'])
    assert all_labels == [[A, B, C]]


def test_memory_fails_for_missing_queries_and_train_test_id_overlap(tmp_path):
    train, query = store_memory(tmp_path)
    memory = make_memory(tmp_path)
    with pytest.raises(ValueError, match='Missing memory query embedding.*sl:missing'):
        memory.retrieve(['sl:missing'])
    np.savez_compressed(query, ids=np.array(['sl:train-a']), embeddings=np.array([[1., 0.]]))
    with pytest.raises(ValueError, match='sample IDs overlap'):
        make_memory(tmp_path)
    query.unlink()
    with pytest.raises(FileNotFoundError, match='./data embed iptc-31 -c sl -c bge-m3'):
        make_memory(tmp_path)


@pytest.mark.parametrize('settings', [
    {'model': '../bge-m3'}, {'model': ''}, {'model': True},
    {'model': 'bge-m3', 'threshold': float('nan')}, {'model': 'bge-m3', 'threshold': 1.1},
    {'model': 'bge-m3', 'threshold': True}, {'model': 'bge-m3', 'top_n': 0},
    {'model': 'bge-m3', 'top_n': 1.5}, {'model': 'bge-m3', 'top_n': True},
    {'model': 'bge-m3', 'unknown': 1}, 'bge-m3',
])
def test_memory_rejects_invalid_switches(settings):
    with pytest.raises(ValueError):
        memory_settings({'memory': settings})


def test_memory_is_opt_in_and_has_explicit_defaults():
    assert memory_settings({}) is None
    assert memory_settings({'memory': {'model': None}}) is None
    assert memory_settings({'memory': {'model': 'bge-m3'}}) == {'model': 'bge-m3', 'threshold': 0., 'top_n': 5}


@pytest.mark.parametrize('model_config', ['clef-flash', 'glide-2b', 'glide-9b', 'starlux1.1-4b'])
@pytest.mark.parametrize('single_language', [False, True])
def test_decision_memory_asks_only_retrieved_labels_and_pools_all_test_rows(tmp_path, monkeypatch, model_config, single_language):
    paths = entrypoint._ensure_dirs(tmp_path, 'eval', 'sequence', 'iptc-31')
    prepare = paths.get_script_ctx_path('data', 'prepare')
    prepare.mkdir(parents=True)
    embedding_dir = paths.get_script_ctx_path('data', 'embed')
    merged, _ = entrypoint._load_and_merge_configs(Path('conf'), 'sequence', 'iptc-31', [model_config])
    merged = entrypoint._apply_cli_overrides(merged, [
        'data.attributes.memory.model=bge-m3', 'data.attributes.memory.threshold=0.95',
        'data.attributes.memory.top_n=5', 'data.attributes.sequence.batch_size=4',
        'data.attributes.sequence.parameters.device=cpu', 'data.attributes.sequence.parameters.max_samples=3',
    ])
    model_args, args, train_args, _ = entrypoint._parse_hf_args(merged)
    if single_language:
        args = replace(args, source=replace(args.source, select=replace(args.source.select, subset='iptc-31.sl')))
    args.attributes['sequence']['exclude_test_ids'] = ['hr:206', 'sl:206']
    for language in ('hr', 'sl'):
        store_memory(embedding_dir, language)
        (prepare / f'iptc-31.{language}.labels.csv').write_text('id,name\n01000000,arts\n')
        samples = [{'id': f'{language}:{suffix}', 'text': f'Article {language}:{suffix}', 'label': labels}
                   for suffix, labels in [('206', [C]), ('a', [A]), ('b', [B]), ('none', [C]), ('after-limit', [A])]]
        (prepare / f'iptc-31.{language}.test.jsonl').write_text(''.join(json.dumps(sample)+'\n' for sample in samples))
    originals = {path: path.read_bytes() for path in paths.base.result.data.rglob('*') if path.is_file()}
    baseline = paths.context / f'iptc-31.{model_config}.{args.attributes["sequence"]["method"]}.metrics.json'
    baseline.write_text('previous full-schema result')
    backend = TextDecider.get_class(model_args)
    calls, instances = [], []

    class Predictor(TextDecider):
        name, method = backend.name, backend.method
        parameter_names = backend.parameter_names
        default_parameters = backend.default_parameters

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            instances.append(self)

        def _predict(self, ids, texts, questions, batch_size):
            assert batch_size == 4
            for sample_id, text in zip(ids, texts, strict=True):
                assert text == f'Article {sample_id}'
                assert tuple(questions) == ((A,) if sample_id.endswith(':a') else (C,))
                calls.append(str(sample_id))
            self.input_tokens += len(ids) * len(questions)
            return np.full((len(ids), len(questions)), .9, dtype=np.float32)

    monkeypatch.setitem(TextDecider._registry, model_args.model_name_or_path.lower(), Predictor)
    monkeypatch.setattr(iptc_31, 'paths', paths, raising=False)
    monkeypatch.setattr(iptc_31, 'logger', logging.getLogger(__name__), raising=False)
    iptc_31.main(args, model_args, train_args)
    [output] = paths.context.glob('*.memory.*.metrics.json')
    metadata = json.loads(output.read_text())
    languages = ['sl'] if single_language else ['hr', 'sl']
    size = len(languages)
    assert metadata['num_labels'] == 31 and metadata['num_test'] == 3*size
    memory = metadata['memory']
    assert memory['num_queries'] == 3*size and memory['num_queries_without_candidates'] == size
    assert memory['num_candidate_labels'] == 2*size and memory['num_gold_candidate_labels'] == size
    assert memory['candidate_label_recall'] == pytest.approx(1/3)
    assert len(memory['sources']) == size and len(memory['queries']) == 3*size
    assert sorted(calls) == [f'{language}:{suffix}' for language in languages for suffix in ('a', 'b')] and len(instances) == 1
    assert metadata['input_tokens'] == 2*size
    assert metadata['metrics']['test/micro.f1'] == pytest.approx(.4)
    with np.load(output.with_name(output.name.replace('.metrics.json', '.predictions.npz'))) as saved:
        assert saved['y_score'].shape == (3*size, 31)
        assert saved['ids'].tolist() == [f'{language}:{suffix}' for language in languages for suffix in ('a', 'b', 'none')]
        assert not saved['y_score'][list(range(2, 3*size, 3))].any()
        for row, code in [(3*language+offset, code) for language in range(size) for offset, code in [(0, A), (1, C)]]:
            assert saved['y_score'][row, list(saved['label_codes']).index(code)] == pytest.approx(.9)
            assert np.count_nonzero(saved['y_score'][row]) == 1
    assert baseline.read_text() == 'previous full-schema result'
    assert all(path.read_bytes() == original for path, original in originals.items())
