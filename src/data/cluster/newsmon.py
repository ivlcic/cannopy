import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List

from ..embed.newsmon import load_embeddings
from ...app.args.data import DataArguments
from ...app.args.model import ModelArguments
from ...app.args.runtime import Paths
from ._newsmon.cluster import compute_clusters
from ._newsmon.compare import (
    AGGREGATE_FIELDNAMES,
    DETAIL_FIELDNAMES,
    build_thresholds,
    compare_thresholds,
    select_best_aggregate_row,
)

logger: Logger
paths: Paths


def _get_subset_name(data_args: DataArguments) -> str:
    subset = data_args.source.select.subset
    if subset:
        return subset
    return data_args.dataset_name


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone()


def _load_article_buckets(subset: str, model_short_name: str, num_days: int) -> Dict[str, List[Dict[str, Any]]]:
    prepared_dir = paths.get_ctx_path('prepare')  # paths['prepare']['data'] / data_args.dataset_name
    if not prepared_dir.exists():
        logger.error('Source [prepare] %s directory not found: %s', subset, prepared_dir)
        return {}
    embed_dir = paths.get_ctx_path('embed')  # paths['embed']['data'] / data_args.dataset_name
    if not embed_dir.exists():
        logger.error('Source [embed] %s directory not found: %s', subset, embed_dir)
        return {}

    src_file = prepared_dir / f'{subset}.jsonl'
    if not src_file.exists():
        raise FileNotFoundError(f'Prepared data file not found: {src_file}')

    src_ebd_file = embed_dir / f'{subset}.{model_short_name}.jsonl'
    if not src_ebd_file.exists():
        raise FileNotFoundError(f'Embedding file not found: {src_ebd_file}')

    src_ebd = load_embeddings(src_ebd_file)
    collected: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    min_created = None
    max_created = None

    with src_file.open('r', encoding='utf-8') as f_in:
        for line_no, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                article = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Malformed JSON in {src_file} line {line_no}') from exc

            article_id = article.get('id')
            if not article_id:
                logger.warning('Missing id in %s line %d.', src_file, line_no)
                continue
            if article_id not in src_ebd:
                logger.warning('Missing embedding for %s in %s.', article_id, src_ebd_file)
                continue

            created = _parse_datetime(article['created'])
            published = _parse_datetime(article['published'])
            article['created'] = created
            article['published'] = published
            article['date'] = _parse_datetime(article['date'])
            article['embedding'] = src_ebd[article_id]['embedding']

            if min_created is None or created < min_created:
                min_created = created
            if max_created is None or created > max_created:
                max_created = created

            collected[created.strftime('%Y-%m-01')].append(article)

    if min_created is None or max_created is None:
        raise ValueError(f'No clusterable articles found in {src_file}')

    bucketed: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    anchor = min_created.replace(hour=0, minute=0, second=0, microsecond=0)
    for articles in collected.values():
        for article in articles:
            delta = article['created'] - anchor
            bucket_start_date = anchor + timedelta(days=(delta.days // num_days) * num_days)
            bucket_key = bucket_start_date.strftime('%Y-%m-%d')
            bucketed[bucket_key].append(article)
    return bucketed


def _format_csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return f'{value:.6f}'
    return value


def _write_csv_rows(target_file: Path, fieldnames: List[str], rows: List[Dict[str, Any]], append: bool = False) -> None:
    mode = 'a' if append else 'w'
    file_exists = target_file.exists()

    with target_file.open(mode, encoding='utf-8', newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        if not append or not file_exists or target_file.stat().st_size == 0:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_csv_value(row.get(key)) for key in fieldnames})


def fit(data_args: DataArguments, model_args: ModelArguments) -> None:
    logger.info('Clustering fitting %s', data_args.dataset_name)
    target_dir = paths.context
    target_dir.mkdir(parents=True, exist_ok=True)
    subset = _get_subset_name(data_args)
    num_days = data_args.cluster.attributes.get('num_days', 5)
    seed: int = data_args.cluster.attributes.get('seed', 2611)
    baseline_model_name: str = data_args.cluster.attributes.get('fit_baseline_model', 'oai-txt_ebd_3s')
    baseline_threshold: float = data_args.cluster.attributes.get('fit_baseline_threshold', 0.88)
    fit_start_threshold: float = data_args.cluster.attributes.get('fit_start_threshold', 0.85)
    fit_end_threshold: float = data_args.cluster.attributes.get('fit_end_threshold', 0.93)
    fit_threshold_step: float = data_args.cluster.attributes.get('fit_threshold_step', 0.001)

    baseline_bucketed = _load_article_buckets(subset, baseline_model_name, num_days)
    logger.info('Baseline Clustering %s ...', data_args.dataset_name)
    baseline_clusters: List[Dict[str, Any]] = compute_clusters(baseline_bucketed, baseline_threshold, seed, logger)
    logger.info('Baseline Clustering %s done.', data_args.dataset_name)
    bucketed = _load_article_buckets(subset, model_args.short_name, num_days)
    thresholds = build_thresholds(fit_start_threshold, fit_end_threshold, fit_threshold_step)
    logger.info(
        'Clustering fitting %s across %d thresholds from %.6f to %.6f (step %.6f)',
        data_args.dataset_name,
        len(thresholds),
        fit_start_threshold,
        fit_end_threshold,
        fit_threshold_step,
    )

    agg_base_name = f'{subset}_fit_agg-{model_args.short_name}'
    detail_base_name = f'{subset}_fit_detail-{model_args.short_name}'

    fit_detail_file = target_dir / f'{detail_base_name}.csv'
    fit_agg_file = target_dir / f'{agg_base_name}.csv'
    totals_agg_file = target_dir / f'{subset}_fit_agg.csv'
    detail_rows, aggregate_rows = compare_thresholds(
        baseline_clusters,
        bucketed,
        model_args.short_name,
        thresholds,
        seed,
        logger,
    )
    best_row = dict(select_best_aggregate_row(aggregate_rows))
    best_row['Baseline Model'] = baseline_model_name
    best_row['Baseline Threshold'] = baseline_threshold

    _write_csv_rows(fit_detail_file, DETAIL_FIELDNAMES, detail_rows)
    _write_csv_rows(fit_agg_file, AGGREGATE_FIELDNAMES, aggregate_rows)
    _write_csv_rows(
        totals_agg_file,
        AGGREGATE_FIELDNAMES + ['Baseline Model', 'Baseline Threshold'],
        [best_row],
        append=True,
    )
    logger.info(
        'Best threshold for %s is %.6f with rank %.6f',
        model_args.short_name,
        best_row['Threshold'],
        best_row['Rank Score'],
    )


def main(data_args: DataArguments, model_args: ModelArguments) -> None:
    logger.info('Clustering %s', data_args.dataset_name)
    subset = _get_subset_name(data_args)
    num_days = data_args.cluster.attributes.get('num_days', 5)
    bucketed = _load_article_buckets(subset, model_args.short_name, num_days)

    seed: int = data_args.cluster.attributes.get('seed', 2611)
    sim_threshold = data_args.cluster.attributes.get('sim_threshold', 0.80)
    base_name = f'{subset}_clusters-{model_args.short_name}@{sim_threshold}'

    target_dir = paths.context
    target_dir.mkdir(parents=True, exist_ok=True)
    tgt_file = target_dir / f'{base_name}.jsonl'
    clusters: List[Dict[str, Any]] = compute_clusters(bucketed, sim_threshold, seed, logger)

    with tgt_file.open('w', encoding='utf-8') as f_out:
        for k in clusters:
            f_out.write(json.dumps(k, ensure_ascii=False) + '\n')

    output_excel = data_args.cluster.attributes.get('output_excel', False)
    if output_excel:
        from ._newsmon.xlsx import ClusterExcel
        xlsx_file = target_dir / f'{base_name}.xlsx'
        excel = ClusterExcel(xlsx_file)
        excel.write_xlsx(clusters)
