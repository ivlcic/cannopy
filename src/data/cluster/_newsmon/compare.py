from __future__ import annotations

from decimal import Decimal
from logging import Logger
from statistics import fmean
from typing import Any, Dict, List, Sequence

from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
)
from sklearn.metrics.cluster import contingency_matrix

from .cluster import compute_clusters

DETAIL_FIELDNAMES = [
    'Model',
    'Threshold',
    'Span',
    'ARI',
    'AMI',
    'Homogeneity',
    'Completeness',
    'V-measure',
    'Pairwise Precision',
    'Pairwise Recall',
    'Pairwise F1',
    '# Articles',
    '# Baseline clusters',
    '# Model clusters',
]

AGGREGATE_FIELDNAMES = [
    'Model',
    'Threshold',
    'Mean ARI',
    'Mean AMI',
    'Mean Homogeneity',
    'Mean Completeness',
    'Mean V-measure',
    'Mean Pairwise F1',
    'Mean Precision',
    'Mean Recall',
    'Rank Score',
    'Spans',
]


def _decimal_places(step: float) -> int:
    exponent = Decimal(str(step)).normalize().as_tuple().exponent
    return max(0, -exponent)


def build_thresholds(start_threshold: float, end_threshold: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError(f'fit_threshold_step must be positive, got {step}')
    if end_threshold < start_threshold:
        raise ValueError(
            f'fit_end_threshold must be greater than or equal to fit_start_threshold, got '
            f'{start_threshold}..{end_threshold}'
        )

    places = _decimal_places(step)
    start = Decimal(str(start_threshold))
    end = Decimal(str(end_threshold))
    step_dec = Decimal(str(step))
    thresholds: List[float] = []
    current = start

    while current <= end:
        thresholds.append(round(float(current), places))
        current += step_dec

    if not thresholds:
        thresholds.append(round(float(start), places))
    return thresholds


def _cluster_article_ids(span_clusters: Dict[str, Any]) -> List[List[str]]:
    article_ids: List[List[str]] = []
    for cluster in span_clusters['clusters']:
        ids: List[str] = []
        for article in cluster['articles']:
            ids.append(article['id'])
        article_ids.append(ids)
    return article_ids


def _build_article_assignments(span_clusters: Dict[str, Any]) -> Dict[str, int]:
    assignments: Dict[str, int] = {}
    for cluster_idx, article_ids in enumerate(_cluster_article_ids(span_clusters)):
        for article_id in article_ids:
            if article_id in assignments:
                raise ValueError(
                    f'Article {article_id} appears in more than one cluster for span {span_clusters["key"]}'
                )
            assignments[article_id] = cluster_idx
    return assignments


def _safe_divide(numerator: float, denominator: float, default: float) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def _num_pairs(cluster_size: int) -> float:
    return float(cluster_size * (cluster_size - 1) // 2)


def _compute_pairwise_scores(labels_true: Sequence[int], labels_pred: Sequence[int]) -> Dict[str, float]:
    matrix = contingency_matrix(labels_true, labels_pred, sparse=False)
    true_positive = sum(_num_pairs(int(value)) for value in matrix.ravel())
    predicted_positive = sum(_num_pairs(int(value)) for value in matrix.sum(axis=0).tolist())
    actual_positive = sum(_num_pairs(int(value)) for value in matrix.sum(axis=1).tolist())

    precision = _safe_divide(true_positive, predicted_positive, 1.0)
    recall = _safe_divide(true_positive, actual_positive, 1.0)
    if precision == 0.0 and recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        'Pairwise Precision': precision,
        'Pairwise Recall': recall,
        'Pairwise F1': f1,
    }


def _compare_span(
    baseline_span: Dict[str, Any],
    model_span: Dict[str, Any],
    model_name: str,
    threshold: float,
) -> Dict[str, Any]:
    baseline_assignments = _build_article_assignments(baseline_span)
    model_assignments = _build_article_assignments(model_span)
    shared_article_ids = sorted(baseline_assignments.keys() & model_assignments.keys())

    if not shared_article_ids:
        raise ValueError(f'No shared article ids found for span {baseline_span["key"]}')

    baseline_labels = [baseline_assignments[article_id] for article_id in shared_article_ids]
    model_labels = [model_assignments[article_id] for article_id in shared_article_ids]
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
        baseline_labels, model_labels
    )
    pairwise_scores = _compute_pairwise_scores(baseline_labels, model_labels)

    return {
        'Model': model_name,
        'Threshold': threshold,
        'Span': baseline_span['key'],
        'ARI': adjusted_rand_score(baseline_labels, model_labels),
        'AMI': adjusted_mutual_info_score(baseline_labels, model_labels),
        'Homogeneity': homogeneity,
        'Completeness': completeness,
        'V-measure': v_measure,
        'Pairwise Precision': pairwise_scores['Pairwise Precision'],
        'Pairwise Recall': pairwise_scores['Pairwise Recall'],
        'Pairwise F1': pairwise_scores['Pairwise F1'],
        '# Articles': len(shared_article_ids),
        '# Baseline clusters': len(set(baseline_labels)),
        '# Model clusters': len(set(model_labels)),
    }


def _aggregate_threshold_rows(model_name: str, threshold: float, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    mean_ari = fmean(row['ARI'] for row in rows)
    mean_ami = fmean(row['AMI'] for row in rows)
    mean_homogeneity = fmean(row['Homogeneity'] for row in rows)
    mean_completeness = fmean(row['Completeness'] for row in rows)
    mean_v_measure = fmean(row['V-measure'] for row in rows)
    mean_pairwise_f1 = fmean(row['Pairwise F1'] for row in rows)
    mean_precision = fmean(row['Pairwise Precision'] for row in rows)
    mean_recall = fmean(row['Pairwise Recall'] for row in rows)

    return {
        'Model': model_name,
        'Threshold': threshold,
        'Mean ARI': mean_ari,
        'Mean AMI': mean_ami,
        'Mean Homogeneity': mean_homogeneity,
        'Mean Completeness': mean_completeness,
        'Mean V-measure': mean_v_measure,
        'Mean Pairwise F1': mean_pairwise_f1,
        'Mean Precision': mean_precision,
        'Mean Recall': mean_recall,
        'Rank Score': (0.4 * mean_ari) + (0.3 * mean_pairwise_f1) + (0.2 * mean_ami) + (0.1 * mean_v_measure),
        'Spans': len(rows),
    }


def compare_thresholds(
    baseline_clusters: List[Dict[str, Any]],
    bucketed: Dict[str, List[Dict[str, Any]]],
    model_name: str,
    thresholds: Sequence[float],
    seed: int,
    logger: Logger
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    baseline_by_span = {span['key']: span for span in baseline_clusters}
    detail_rows: List[Dict[str, Any]] = []
    aggregate_rows: List[Dict[str, Any]] = []

    for threshold in thresholds:
        logger.info('Comparing %s threshold %.6f ...', model_name, threshold)
        model_clusters = compute_clusters(bucketed, threshold, seed, logger)
        model_by_span = {span['key']: span for span in model_clusters}
        shared_spans = sorted(baseline_by_span.keys() & model_by_span.keys())
        threshold_rows: List[Dict[str, Any]] = []

        if not shared_spans:
            raise ValueError(f'No shared spans found between baseline and {model_name} clusters')

        missing_baseline_spans = sorted(model_by_span.keys() - baseline_by_span.keys())
        missing_model_spans = sorted(baseline_by_span.keys() - model_by_span.keys())
        if missing_baseline_spans:
            logger.warning(
                'Skipping %d spans missing from baseline for %s threshold %.6f: %s',
                len(missing_baseline_spans), model_name, threshold, ', '.join(missing_baseline_spans)
            )
        if missing_model_spans:
            logger.warning(
                'Skipping %d spans missing from model %s threshold %.6f: %s',
                len(missing_model_spans), model_name, threshold, ', '.join(missing_model_spans)
            )

        for span_key in shared_spans:
            row = _compare_span(baseline_by_span[span_key], model_by_span[span_key], model_name, threshold)
            threshold_rows.append(row)
            detail_rows.append(row)

        aggregate_rows.append(_aggregate_threshold_rows(model_name, threshold, threshold_rows))

    return detail_rows, aggregate_rows


def select_best_aggregate_row(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError('No aggregate rows available')
    return max(
        rows,
        key=lambda row: (
            row['Rank Score'],
            row['Mean ARI'],
            row['Mean Pairwise F1'],
            row['Mean AMI'],
            row['Mean V-measure'],
        ),
    )
