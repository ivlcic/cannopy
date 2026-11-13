"""In-memory Optuna searches using validation rows held out from training only."""

from dataclasses import dataclass
from logging import Logger

import numpy as np
import yaml

from . import EmbeddedSplit, _load_split, _method_settings, _select_subsets
from .features import _read_text_split, vectorize_texts
from .methods import METHODS
from ....app.args.data import DataArguments
from ....app.args.model import ModelArguments
from ....app.args.runtime import Paths
from ....data.embed.newsmon import load_multilabel_labeler
from ....data.prepare.newsmon import get_sidecar_name, get_subset_name


@dataclass
class ValidationSplit:
    subset: str
    train: EmbeddedSplit
    validation: EmbeddedSplit


def _load_validation_split(data_args: DataArguments, model_args: ModelArguments, paths: Paths,
                           validation_size: float, seed: int) -> ValidationSplit:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    subset = get_subset_name(data_args)
    prepare_dir = paths.get_script_ctx_path('data', 'prepare')
    labeler = load_multilabel_labeler(prepare_dir / f'{subset}.labels.csv')
    features = data_args.attributes.get('sequence', {}).get('features', {})
    if features.get('type', 'embeddings') == 'tfidf':
        ids, texts, labels = _read_text_split(prepare_dir / f'{subset}.train.jsonl', labeler)
    else:
        split = _load_split(paths.get_script_ctx_path('data', 'embed') /
                            get_sidecar_name(data_args, model_args, 'train'), labeler)
        ids, labels = split.ids, split.labels
    if len(ids) < 2:
        raise ValueError(f'{subset} requires at least two training documents for Optuna')
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=validation_size, random_state=seed)
    train_rows, validation_rows = next(splitter.split(np.zeros((len(ids), 1)), labels))
    if not len(train_rows) or not len(validation_rows):
        raise ValueError(f'{subset}: Optuna split must contain both training and validation documents')
    if features.get('type', 'embeddings') == 'tfidf':
        # Split raw text first so validation cannot influence vocabulary or IDF.
        train_features, validation_features, _ = vectorize_texts(
            [texts[row] for row in train_rows], [texts[row] for row in validation_rows], features,
        )
    else:
        train_features, validation_features = split.embeddings[train_rows], split.embeddings[validation_rows]
    return ValidationSplit(
        subset,
        EmbeddedSplit(ids[train_rows], train_features, labels[train_rows]),
        EmbeddedSplit(ids[validation_rows], validation_features, labels[validation_rows]),
    )


def tune(data_args: DataArguments, model_args: ModelArguments, method: str,
         paths: Paths, logger: Logger) -> None:
    import optuna

    options = data_args.attributes.get('sequence', {})
    tuning = options.get('optuna', {})
    n_trials = tuning.get('trials')
    validation_size = float(tuning.get('validation_size', 0.2))
    seed = int(tuning.get('seed', 2611))
    batch_size = int(options.get('batch_size', 256))
    if not isinstance(n_trials, int) or isinstance(n_trials, bool) or n_trials < 50:
        raise ValueError('sequence.optuna.trials must be an integer >= 50; omit it to disable Optuna')
    if not 0 < validation_size < 1:
        raise ValueError('sequence.optuna.validation_size must be between 0 and 1')
    if batch_size <= 0:
        raise ValueError('sequence.batch_size must be positive')
    selected = _select_subsets(data_args, model_args, train_only=True)
    splits = [_load_validation_split(args, model_args, paths, validation_size, seed) for args in selected]
    for split in splits:
        logger.info('Optuna %s: %s training and %s validation documents (seed %s)',
                    split.subset, len(split.train.ids), len(split.validation.ids), seed)
    parameters, baseline_threshold = _method_settings(options, method)
    max_k = min(256, min(len(split.train.ids) for split in splits))

    def objective(trial):
        if method == 'rae_xmc':
            suggested = {
                'top_k': trial.suggest_int('top_k', 1, max_k, log=True),
                'temperature': trial.suggest_float('temperature', 0.001, 1.0, log=True),
                'metric': trial.suggest_categorical('metric', ['l2', 'ip']),
            }
        else:
            suggested = {
                'C': trial.suggest_float('C', 0.0001, 1000.0, log=True),
                'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
            }
        threshold = trial.suggest_float('threshold', 0.05, 0.8)
        true_positive = false_positive = false_negative = 0
        for split in splits:
            scores = np.asarray(METHODS[method](
                split.train.embeddings, split.train.labels, split.validation.embeddings,
                batch_size=batch_size, **{**parameters, **suggested},
            ))
            if scores.shape != split.validation.labels.shape or not np.isfinite(scores).all():
                raise ValueError(f'{method} returned invalid validation scores')
            predicted = scores > threshold
            truth = split.validation.labels.astype(bool)
            # Pooled micro-F1 sums counts across languages with their own label sets.
            true_positive += np.count_nonzero(predicted & truth)
            false_positive += np.count_nonzero(predicted & ~truth)
            false_negative += np.count_nonzero(~predicted & truth)
        denominator = 2 * true_positive + false_positive + false_negative
        return float(2 * true_positive / denominator) if denominator else 0.0

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    # Include the configured baseline before sampling the remaining trials.
    if method == 'rae_xmc':
        baseline = {
            'top_k': min(parameters.get('top_k', 50), max_k),
            'temperature': parameters.get('temperature', 0.04),
            'metric': parameters.get('metric', 'l2'),
        }
    else:
        baseline = {'C': parameters.get('C', 1.0), 'class_weight': parameters.get('class_weight')}
    baseline['threshold'] = baseline_threshold
    study.enqueue_trial(baseline)
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    best = dict(study.best_params)
    threshold = best.pop('threshold')
    print(f'\nOptuna: {method} / {model_args.short_name} / {", ".join(split.subset for split in splits)}')
    print(f'Best validation micro-F1: {study.best_value:.6f} ({len(study.trials)} trials)')
    print('Best evaluation settings (YAML):')
    print(yaml.safe_dump({'data': {'attributes': {'sequence': {
        'method': method, 'threshold': threshold, 'parameters': {**parameters, **best},
    }}}}, sort_keys=False), end='')
