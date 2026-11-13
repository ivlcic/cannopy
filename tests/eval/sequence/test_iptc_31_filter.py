import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from sklearn.metrics import f1_score

from src.app.args.data import DataArguments
from src.app.args.model import ModelArguments
from src.app.entrypoint import _ensure_dirs
from src.eval.sequence import iptc_31


def _write_inputs(prepare_dir, embed_dir, lang):
    labeler = iptc_31.MultilabelLabeler(labels=['01000000', '20000002'])
    (prepare_dir / f'iptc-31.{lang}.labels.csv').write_text(
        'id,name,count\n01000000,arts,4\n20000002,culture,4\n', encoding='utf-8',
    )
    for split in ('train', 'test'):
        ids = ([f'{lang}:215', f'{lang}:train2', f'{lang}:train3'] if split == 'train'
               else [f'{lang}:keep1', f'{lang}:206', f'{lang}:keep2'])
        labels = [['01000000'], ['20000002'], ['01000000', '20000002']]
        texts = ['apple orchard', 'banana fruit', 'apple banana orchard fruit']
        records = [dict(id=sample_id, text=text, label=label)
                   for sample_id, text, label in zip(ids, texts, labels)]
        (prepare_dir / f'iptc-31.{lang}.{split}.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in records), encoding='utf-8',
        )
        raw_labels = np.empty(len(labels), dtype=object)
        raw_labels[:] = labels
        np.savez_compressed(
            embed_dir / f'iptc-31.{lang}.fixture.{split}.npz',
            ids=np.asarray(ids), embeddings=np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.float32),
            labels=raw_labels, label_ids=labeler.encode(labels),
        )


@pytest.mark.parametrize('feature_type', ['embeddings', 'tfidf'])
def test_evaluation_filters_only_test_ids_and_pools_audit_metadata(tmp_path, monkeypatch, feature_type):
    paths = _ensure_dirs(tmp_path, 'eval', 'sequence', 'iptc-31')
    prepare_dir = paths.get_script_ctx_path('data', 'prepare')
    embed_dir = paths.get_script_ctx_path('data', 'embed')
    prepare_dir.mkdir(parents=True)
    embed_dir.mkdir(parents=True)
    for lang in ('hr', 'sl'):
        _write_inputs(prepare_dir, embed_dir, lang)
    original_files = {path: path.read_bytes() for directory in (prepare_dir, embed_dir)
                      for path in directory.iterdir()}

    config = yaml.safe_load(Path('conf/data/iptc-31/sequence/iptc-31.yaml').read_text())
    data_args = DataArguments(dataset_name='iptc-31', attributes=config['data']['attributes'])
    options = data_args.attributes['sequence']
    # An excluded ID present only in training must keep its training row.
    options['exclude_test_ids'].append('sl:215')
    if feature_type == 'tfidf':
        options['features'] = {'type': 'tfidf', 'min_df': 1}
    model_args = ModelArguments(short_name='fixture')
    monkeypatch.setattr(iptc_31, 'paths', paths, raising=False)
    monkeypatch.setattr(iptc_31, 'logger', logging.getLogger(__name__), raising=False)

    iptc_31.log_reg(data_args, model_args)

    metadata = json.loads((paths.context / 'iptc-31.fixture.log_reg.metrics.json').read_text())
    assert metadata['num_train'] == 6
    assert metadata['num_test_before_filter'] == 6
    assert metadata['num_test'] == 5
    assert metadata['num_test_excluded'] == 1
    assert metadata['excluded_test_ids'] == ['sl:206']
    with np.load(paths.context / 'iptc-31.fixture.log_reg.predictions.npz') as predictions:
        assert predictions['ids'].tolist() == ['hr:keep1', 'hr:206', 'hr:keep2', 'sl:keep1', 'sl:keep2']
        np.testing.assert_array_equal(predictions['y_true'], [[1, 0], [0, 1], [1, 1], [1, 0], [1, 1]])
        assert predictions['y_score'].shape == (5, 2)
        expected_f1 = f1_score(predictions['y_true'], predictions['y_pred'], average='micro')
        assert metadata['metrics']['test/micro.f1'] == pytest.approx(expected_f1)
    assert all(path.read_bytes() == contents for path, contents in original_files.items())

    # The same filter applies when evaluating one language without pooling.
    single_language = replace(data_args, source=replace(
        data_args.source, select=replace(data_args.source.select, subset='iptc-31.sl'),
    ))
    result = iptc_31._evaluate_subset(single_language, model_args, 'log_reg')
    assert result.ids.tolist() == ['sl:keep1', 'sl:keep2']
    assert result.metadata['num_train'] == 3
    assert result.metadata['num_test_before_filter'] == 3
    assert result.metadata['num_test_excluded'] == 1
    assert result.metadata['num_test'] == 2

    options['exclude_test_ids'] = []
    unfiltered = iptc_31._evaluate_subset(single_language, model_args, 'log_reg')
    assert unfiltered.ids.tolist() == ['sl:keep1', 'sl:206', 'sl:keep2']
    assert unfiltered.metadata['num_test_excluded'] == 0


@pytest.mark.parametrize('excluded_ids', ['sl:206', [206], [''], None])
def test_filter_rejects_invalid_id_lists(excluded_ids):
    test = iptc_31.EmbeddedSplit(np.asarray(['sl:206']), np.ones((1, 2)), np.ones((1, 2)))
    with pytest.raises(ValueError, match='list of non-empty sample ID strings'):
        iptc_31._filter_test_split(test, excluded_ids)


def test_filter_rejects_removing_entire_test_set():
    test = iptc_31.EmbeddedSplit(np.asarray(['sl:206']), np.ones((1, 2)), np.ones((1, 2)))
    with pytest.raises(ValueError, match='excludes every test document'):
        iptc_31._filter_test_split(test, ['sl:206'])
