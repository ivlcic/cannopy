import json
import time
from dataclasses import dataclass, replace
from logging import Logger
from pathlib import Path

import numpy as np

from .methods import METHODS
from ....app.args.data import DataArguments
from ....app.args.model import ModelArguments
from ....app.args.runtime import Paths
from ....app.labeler import MultilabelLabeler
from ....app.metrics import Metrics
from ....data.embed.newsmon import load_multilabel_labeler, normalize_embeddings
from ....data.prepare.newsmon import get_sidecar_name, get_subset_name

logger: Logger
paths: Paths


@dataclass
class EmbeddedSplit:
    ids: np.ndarray
    embeddings: np.ndarray
    labels: np.ndarray


@dataclass
class EvaluationResult:
    metadata: dict
    ids: np.ndarray
    y_true: np.ndarray
    scores: np.ndarray


def _load_split(path: Path, labeler: MultilabelLabeler) -> EmbeddedSplit:
    if not path.is_file():
        raise FileNotFoundError(f'Embedding sidecar not found: {path}. Run ./data embed iptc-31 first.')
    with np.load(path, allow_pickle=True) as arrays:
        ids = np.asarray(arrays['ids'], dtype=str)
        embeddings = np.asarray(arrays['embeddings'], dtype=np.float32)
        raw_labels = arrays['labels'].tolist()
        encoded_labels = np.asarray(arrays['label_ids'])
    if ids.ndim != 1 or len(ids) == 0 or len(set(ids)) != len(ids):
        raise ValueError(f'Expected non-empty unique sample IDs in {path}')
    if embeddings.ndim != 2 or embeddings.shape[0] != len(ids) or embeddings.shape[1] == 0:
        raise ValueError(f'Embedding shape does not match sample IDs in {path}')
    if not np.isfinite(embeddings).all() or np.any(np.linalg.norm(embeddings, axis=1) == 0):
        raise ValueError(f'Expected finite, non-zero embedding vectors in {path}')
    if len(raw_labels) != len(ids):
        raise ValueError(f'Label count does not match sample IDs in {path}')
    try:
        labels = labeler.encode(raw_labels).astype(np.float32)
    except KeyError as exc:
        raise ValueError(f'Unknown label {exc} in {path}') from exc
    if labels.shape != (len(ids), labeler.num_labels) or not np.array_equal(labels, encoded_labels):
        raise ValueError(f'Encoded labels do not match the language label CSV in {path}')
    return EmbeddedSplit(ids, np.ascontiguousarray(normalize_embeddings(embeddings)), labels)


def _evaluate_subset(data_args: DataArguments, model_args: ModelArguments, method: str) -> EvaluationResult:
    options = data_args.attributes.get('sequence', {})
    batch_size = int(options.get('batch_size', 256))
    threshold = float(options.get('threshold', 0.5))
    if batch_size <= 0 or not 0 <= threshold <= 1:
        raise ValueError('sequence.batch_size must be positive and sequence.threshold must be in [0, 1]')

    subset = get_subset_name(data_args)
    embed_dir = paths.get_script_ctx_path('data', 'embed')
    labels_file = paths.get_script_ctx_path('data', 'prepare') / f'{subset}.labels.csv'
    labeler = load_multilabel_labeler(labels_file)
    train_file = embed_dir / get_sidecar_name(data_args, model_args, 'train')
    test_file = embed_dir / get_sidecar_name(data_args, model_args, 'test')
    train = _load_split(train_file, labeler)
    test = _load_split(test_file, labeler)
    if train.embeddings.shape[1] != test.embeddings.shape[1]:
        raise ValueError('Train and test embedding dimensions differ')
    if set(train.ids) & set(test.ids):
        raise ValueError('Train and test sample IDs overlap')

    logger.info('Evaluating %s on %s: %s reference and %s test samples', method, subset, len(train.ids), len(test.ids))
    parameters = {
        **options.get('methods', {}).get(method, {}),
        **options.get('parameters', {}),
    }
    started = time.perf_counter()
    scores = np.asarray(METHODS[method](
        train.embeddings, train.labels, test.embeddings,
        batch_size=batch_size, **parameters,
    ), dtype=np.float32)
    duration = time.perf_counter() - started
    if scores.shape != test.labels.shape or not np.isfinite(scores).all():
        raise ValueError(f'{method} returned invalid classification scores')
    result = {
        'dataset': data_args.dataset_name,
        'subset': subset,
        'model': model_args.short_name,
        'method': method,
        'num_train': len(train.ids),
        'num_test': len(test.ids),
        'num_labels': labeler.num_labels,
        'label_order': labeler.classes,
        'batch_size': batch_size,
        'threshold': threshold,
        'parameters': parameters,
        'prediction_seconds': duration,
        'train_sidecar': str(train_file),
        'test_sidecar': str(test_file),
    }
    return EvaluationResult(result, test.ids, test.labels, scores)


def _write_evaluation(evaluation: EvaluationResult) -> None:
    result = evaluation.metadata
    threshold = result['threshold']
    metrics = Metrics(model_name=result['model'], prob_type='multilabel').compute_metrics(
        evaluation.y_true, evaluation.scores, prefix='test/', prob_threshold=threshold,
    )
    result['metrics'] = {key: float(value) for key, value in metrics.items()}
    target_name = f"{result['subset']}.{result['model']}.{result['method']}"
    result_file = paths.context / f'{target_name}.metrics.json'
    prediction_file = paths.context / f'{target_name}.predictions.npz'
    np.savez_compressed(
        prediction_file, ids=evaluation.ids, label_codes=np.asarray(result['label_order'], dtype=str),
        y_true=evaluation.y_true, y_score=evaluation.scores,
        y_pred=(evaluation.scores > threshold).astype(np.int64),
    )
    result_file.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    logger.info('Metrics: %s', result['metrics'])
    logger.info('Saved %s and %s', result_file, prediction_file)


def _pool_evaluations(evaluations: list[EvaluationResult], dataset_name: str) -> EvaluationResult:
    label_order = sorted({code for evaluation in evaluations for code in evaluation.metadata['label_order']})
    label_columns = {code: index for index, code in enumerate(label_order)}
    true_rows, score_rows = [], []
    for evaluation in evaluations:
        columns = [label_columns[code] for code in evaluation.metadata['label_order']]
        y_true = np.zeros((len(evaluation.ids), len(label_order)), dtype=np.float32)
        scores = np.zeros_like(y_true)
        y_true[:, columns] = evaluation.y_true
        scores[:, columns] = evaluation.scores
        true_rows.append(y_true)
        score_rows.append(scores)
    ids = np.concatenate([evaluation.ids for evaluation in evaluations])
    if len(set(ids)) != len(ids):
        raise ValueError('Sample IDs overlap across language test sets')
    result = {
        **evaluations[0].metadata,
        'subset': dataset_name,
        'subsets': [evaluation.metadata['subset'] for evaluation in evaluations],
        'num_train': sum(evaluation.metadata['num_train'] for evaluation in evaluations),
        'num_test': len(ids),
        'num_labels': len(label_order),
        'label_order': label_order,
        'prediction_seconds': sum(evaluation.metadata['prediction_seconds'] for evaluation in evaluations),
        'train_sidecars': [evaluation.metadata['train_sidecar'] for evaluation in evaluations],
        'test_sidecars': [evaluation.metadata['test_sidecar'] for evaluation in evaluations],
        'aggregation': 'pooled test samples; retrieval uses each language training set',
    }
    del result['train_sidecar'], result['test_sidecar']
    return EvaluationResult(result, ids, np.concatenate(true_rows), np.concatenate(score_rows))


def _evaluate(data_args: DataArguments, model_args: ModelArguments, method: str) -> None:
    if method not in METHODS:
        raise ValueError(f'Unknown classification method {method!r}; available: {", ".join(METHODS)}')
    if not model_args.short_name:
        raise ValueError('Select the embedding model configuration, for example -c bge-m3')
    if data_args.source.select.subset:
        _write_evaluation(_evaluate_subset(data_args, model_args, method))
        return

    embed_dir = paths.get_script_ctx_path('data', 'embed')
    prepare_dir = paths.get_script_ctx_path('data', 'prepare')
    prefix = f'{data_args.dataset_name}.'
    suffixes = [f'.{model_args.short_name}.{split}.npz' for split in ['train', 'test']]
    languages = {
        file.name[len(prefix):-len(suffix)]
        for file in embed_dir.glob('*.npz')
        for suffix in suffixes
        if file.name.startswith(prefix) and file.name.endswith(suffix)
    }
    selected = []
    for lang in sorted(languages):
        if not lang or '.' in lang:
            continue
        subset = f'{data_args.dataset_name}.{lang}'
        required = [embed_dir / f'{subset}{suffix}' for suffix in suffixes]
        required.append(prepare_dir / f'{subset}.labels.csv')
        missing = [str(file) for file in required if not file.is_file()]
        if missing:
            logger.warning('Skipping %s; missing files: %s', subset, ', '.join(missing))
            continue
        selected.append(replace(data_args, source=replace(
            data_args.source, select=replace(data_args.source.select, subset=subset),
        )))
    if not selected:
        raise FileNotFoundError(
            f'No complete language train/test sidecars with label CSVs for {model_args.short_name} in {embed_dir}'
        )
    logger.info('Evaluating %s for available languages: %s', method, ', '.join(
        get_subset_name(args) for args in selected
    ))
    evaluations = [_evaluate_subset(args, model_args, method) for args in selected]
    _write_evaluation(_pool_evaluations(evaluations, data_args.dataset_name))


def zero_shot(data_args: DataArguments, model_args: ModelArguments) -> None:
    _evaluate(data_args, model_args, 'zero_shot')


def rae_xmc(data_args: DataArguments, model_args: ModelArguments) -> None:
    _evaluate(data_args, model_args, 'rae_xmc')


def main(data_args: DataArguments, model_args: ModelArguments) -> None:
    method = data_args.attributes.get('sequence', {}).get('method', 'zero_shot')
    _evaluate(data_args, model_args, method)
