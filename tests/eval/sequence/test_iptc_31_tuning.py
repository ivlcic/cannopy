import json
import logging
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna
import pytest
import yaml
from scipy.sparse import issparse
from sklearn.metrics import f1_score

from src.app import entrypoint
from src.app.args.data import DataArguments
from src.app.args.model import ModelArguments
from src.eval.sequence import iptc_31
from src.eval.sequence.iptc_31 import tuning


def _training_inputs(paths, lang, size):
    prepare_dir = paths.get_script_ctx_path('data', 'prepare')
    embed_dir = paths.get_script_ctx_path('data', 'embed')
    prepare_dir.mkdir(parents=True, exist_ok=True)
    embed_dir.mkdir(parents=True, exist_ok=True)
    (prepare_dir / f'iptc-31.{lang}.labels.csv').write_text(
        'id,name,count\n01000000,arts,20\n20000002,culture,20\n', encoding='utf-8',
    )
    records = [dict(id=f'{lang}:{row}', text=f'common unique{row}',
                    label=[['01000000'], ['20000002'], ['01000000', '20000002']][row % 3])
               for row in range(size)]
    (prepare_dir / f'iptc-31.{lang}.train.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in records), encoding='utf-8',
    )
    labels = iptc_31.MultilabelLabeler(labels=['01000000', '20000002']).encode(
        [row['label'] for row in records],
    )
    raw_labels = np.empty(size, dtype=object)
    raw_labels[:] = [row['label'] for row in records]
    np.savez_compressed(
        embed_dir / f'iptc-31.{lang}.fixture.train.npz',
        ids=np.asarray([row['id'] for row in records]), embeddings=labels.astype(np.float32),
        labels=raw_labels, label_ids=labels,
    )


@pytest.mark.parametrize(('method', 'feature_type'), [
    ('rae_xmc', 'embeddings'), ('log_reg', 'embeddings'), ('svm', 'embeddings'),
    ('log_reg', 'tfidf'), ('svm', 'tfidf'),
])
def test_optuna_uses_training_only_and_prints_best_settings(tmp_path, monkeypatch, capsys, method, feature_type):
    paths = entrypoint._ensure_dirs(tmp_path, 'eval', 'sequence', 'iptc-31')
    # No test files exist: tuning must neither require nor try to load them.
    _training_inputs(paths, 'hr', 24)
    _training_inputs(paths, 'sl', 36)
    config = yaml.safe_load(Path('conf/data/iptc-31/sequence/iptc-31.yaml').read_text())
    args = DataArguments(dataset_name='iptc-31', attributes=config['data']['attributes'])
    options = args.attributes['sequence']
    options['optuna']['trials'] = 50
    options['methods'][method]['threshold'] = 0.37
    if feature_type == 'tfidf':
        options['features'] = {'type': 'tfidf', 'min_df': 1}
    model_args = ModelArguments(short_name='fixture')
    monkeypatch.setattr(iptc_31, 'paths', paths, raising=False)
    monkeypatch.setattr(iptc_31, 'logger', logging.getLogger(__name__), raising=False)
    studies = []
    create_study = optuna.create_study

    def capture_study(**kwargs):
        study = create_study(**kwargs)
        studies.append(study)
        return study

    monkeypatch.setattr(optuna, 'create_study', capture_study)
    getattr(iptc_31, method)(args, model_args)

    study, = studies
    assert len(study.trials) == 50
    assert study.trials[0].params['threshold'] == 0.37
    assert all(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    assert study.best_value >= study.trials[0].value
    output = capsys.readouterr().out
    assert 'iptc-31.hr, iptc-31.sl' in output
    best = yaml.safe_load(output.split('Best evaluation settings (YAML):\n')[1])
    settings = best['data']['attributes']['sequence']
    assert settings['method'] == method
    assert settings['threshold'] == study.best_params['threshold']
    assert all(settings['parameters'][key] == value for key, value in study.best_params.items()
               if key != 'threshold')
    assert list(paths.context.iterdir()) == []
    assert options['parameters'] == {}

    # Check the pooled objective against sklearn, including unequal language sizes.
    truths, predicted = [], []
    for subset_args in iptc_31._select_subsets(args, model_args, train_only=True):
        split = tuning._load_validation_split(subset_args, model_args, paths, 0.2, 2611)
        assert not set(split.train.ids) & set(split.validation.ids)
        scores = iptc_31.METHODS[method](
            split.train.embeddings, split.train.labels, split.validation.embeddings,
            batch_size=256, **settings['parameters'],
        )
        truths.append(split.validation.labels)
        predicted.append(scores > settings['threshold'])
    expected = f1_score(np.concatenate(truths), np.concatenate(predicted), average='micro', zero_division=0)
    assert study.best_value == pytest.approx(expected)


def test_tfidf_validation_vocabulary_is_training_only_and_split_is_repeatable(tmp_path):
    paths = entrypoint._ensure_dirs(tmp_path, 'eval', 'sequence', 'iptc-31')
    _training_inputs(paths, 'sl', 24)
    args = DataArguments(dataset_name='iptc-31', attributes={'sequence': {'features': {
        'type': 'tfidf', 'min_df': 1,
    }}})
    args = replace(args, source=replace(args.source, select=replace(args.source.select, subset='iptc-31.sl')))
    first = tuning._load_validation_split(args, ModelArguments(short_name='fixture'), paths, 0.2, 2611)
    second = tuning._load_validation_split(args, ModelArguments(short_name='fixture'), paths, 0.2, 2611)
    np.testing.assert_array_equal(first.train.ids, second.train.ids)
    np.testing.assert_array_equal(first.validation.ids, second.validation.ids)
    assert len(first.train.ids) + len(first.validation.ids) == 24
    assert issparse(first.train.embeddings)
    # Every document has one unique token: validation tokens must be absent.
    assert first.train.embeddings.shape[1] == len(first.train.ids) + 1
    np.testing.assert_array_equal(first.train.embeddings.getnnz(axis=1), 2)
    np.testing.assert_array_equal(first.validation.embeddings.getnnz(axis=1), 1)


@pytest.mark.parametrize('trials', [None, 50, 75])
def test_cli_trials_setting_preserves_other_overrides(tmp_path, monkeypatch, trials):
    paths = entrypoint._ensure_dirs(tmp_path, 'eval', 'sequence', 'iptc-31')
    monkeypatch.setenv('APP_SCRIPT', 'eval')
    monkeypatch.setattr(entrypoint, '_ensure_dirs', lambda *args: paths)
    monkeypatch.setattr(entrypoint, '_config_logger', lambda *args: (logging.getLogger(__name__), ''))
    captured = {}

    def capture_call(script, task, context, func_name, module_globals):
        captured.update(module_globals)

    monkeypatch.setattr(entrypoint, '_call_module', capture_call)
    argv = ['sequence', 'iptc-31', 'log_reg', '-c', 'tfidf-word',
            '-s', 'data.attributes.sequence.optuna.validation_size=0.25']
    if trials is not None:
        argv.extend(['-s', f'data.attributes.sequence.optuna.trials={trials}'])
    entrypoint.main(argv)
    settings = captured['data_args'].attributes['sequence']['optuna']
    if trials is None:
        assert 'trials' not in settings
    else:
        assert settings['trials'] == trials
    assert settings['validation_size'] == 0.25


def test_optuna_rejects_unsupported_method(monkeypatch):
    args = DataArguments(attributes={'sequence': {'optuna': {'trials': 50}}})
    with pytest.raises(ValueError, match='Optuna supports rae_xmc, log_reg and svm'):
        iptc_31.mlknn(args, ModelArguments(short_name='fixture'))


@pytest.mark.parametrize('trials', [0, -1, 49, 50.0, '50', True, False, None])
def test_optuna_rejects_invalid_trial_budgets_before_loading_data(monkeypatch, trials):
    args = DataArguments(attributes={'sequence': {'optuna': {'trials': trials}}})
    monkeypatch.setattr(iptc_31, 'paths', None, raising=False)
    monkeypatch.setattr(iptc_31, 'logger', logging.getLogger(__name__), raising=False)
    with pytest.raises(ValueError, match='sequence.optuna.trials must be an integer >= 50'):
        iptc_31.log_reg(args, ModelArguments(short_name='fixture'))


@pytest.mark.parametrize(('model', 'top_k', 'temperature', 'threshold', 'svm_c', 'svm_threshold'), [
    ('bge-m3', 18, 0.07033358828261554, 0.33627343872958393, 0.3960445209361584, 0.33566984547530776),
    ('qwen3-ebd-8b', 16, 0.0906398287085298, 0.34170717322613214, 1.1940483741219816, 0.33308725131333106),
])
def test_model_configs_apply_tuned_settings_per_method(model, top_k, temperature, threshold, svm_c, svm_threshold):
    merged, _ = entrypoint._load_and_merge_configs(Path('conf'), 'sequence', 'iptc-31', [model, 'sl'])
    options = merged['data']['attributes']['sequence']
    parameters, actual_threshold = iptc_31._method_settings(options, 'rae_xmc')
    assert parameters == {'top_k': top_k, 'temperature': temperature, 'metric': 'ip'}
    assert actual_threshold == threshold
    svm_parameters, actual_svm_threshold = iptc_31._method_settings(options, 'svm')
    assert svm_parameters == {'C': svm_c, 'max_iter': -1, 'class_weight': None, 'seed': 2611}
    assert actual_svm_threshold == svm_threshold
    for method in ('zero_shot', 'log_reg', 'mlknn'):
        other_parameters, other_threshold = iptc_31._method_settings(options, method)
        assert other_threshold == 0.5
        assert 'top_k' not in other_parameters
        assert 'temperature' not in other_parameters
    assert 'trials' not in options['optuna']
    assert merged['data']['source']['select']['subset'] == 'iptc-31.sl'

    overridden = entrypoint._apply_cli_overrides(merged, [
        'data.attributes.sequence.threshold=0.4',
        'data.attributes.sequence.parameters.top_k=7',
    ])
    parameters, actual_threshold = iptc_31._method_settings(overridden['data']['attributes']['sequence'], 'rae_xmc')
    assert parameters['top_k'] == 7
    assert actual_threshold == 0.4


def test_cuml_tuning_prepares_features_once_and_prints_selected_backend(tmp_path, monkeypatch, capsys):
    log_reg = import_module('src.eval.sequence.iptc_31.methods.log_reg')

    paths = entrypoint._ensure_dirs(tmp_path, 'eval', 'sequence', 'iptc-31')
    _training_inputs(paths, 'hr', 24)
    _training_inputs(paths, 'sl', 36)
    args = DataArguments(dataset_name='iptc-31', attributes={'sequence': {
        'optuna': {'trials': 50}, 'parameters': {'backend': 'cuml'},
    }})
    monkeypatch.setattr(iptc_31, 'paths', paths, raising=False)
    monkeypatch.setattr(iptc_31, 'logger', logging.getLogger(__name__), raising=False)
    uploaded = []
    predictions = []

    def upload(features):
        device_features = SimpleNamespace(shape=features.shape)
        uploaded.append(device_features)
        return device_features

    def predict(train, labels, query, **options):
        assert options['backend'] == 'cuml'
        assert any(train is features for features in uploaded)
        assert any(query is features for features in uploaded)
        predictions.append(1)
        return np.full((query.shape[0], labels.shape[1]), 0.6, dtype=np.float32)

    monkeypatch.setattr(log_reg, 'prepare_cuml_features', upload)
    monkeypatch.setitem(iptc_31.METHODS, 'log_reg', predict)
    iptc_31.log_reg(args, ModelArguments(short_name='fixture'))
    assert len(uploaded) == 4  # Train/validation matrices, once per language.
    assert len(predictions) == 100  # Two languages for each of 50 trials.
    output = capsys.readouterr().out
    best = yaml.safe_load(output.split('Best evaluation settings (YAML):\n')[1])
    assert best['data']['attributes']['sequence']['parameters']['backend'] == 'cuml'
    assert list(paths.context.iterdir()) == []
