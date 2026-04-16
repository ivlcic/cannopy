import csv
import json
from collections import Counter
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import faiss
import numpy as np
from dateutil.relativedelta import relativedelta

from ...app.args.data import DataArguments
from ...app.args.model import ModelArguments
from ...app.args.runtime import Paths

logger: Logger
paths: Paths

__social_media = {
    '8e3b359f', '3e1c137d', '86f18af6', '1fd92aa0', 'c0953029', '1843f51e',
    '151a2b9a', '05b54365', '0e9d50b8', '9f6a5e6c', 'f789b185'
}


def read_csv_to_dict(path: Path, key_col: str = 'id') -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    with path.open('r', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key = row.get(key_col)
            if not key:
                raise ValueError(f"Missing '{key_col}' value in row from {path}")
            if key in out:
                raise ValueError(f"Duplicate '{key_col}' value '{key}' in {path}")
            out[key] = row
    return out


def get_subset_name(data_args: DataArguments) -> str:
    subset = data_args.source.select.subset
    if subset:
        return subset
    return data_args.dataset_name


def get_sidecar_name(data_args: DataArguments, model_args: ModelArguments, split: Optional[str] = None) -> str:
    subset = get_subset_name(data_args)
    if split:
        return f'{subset}.{model_args.short_name}.{split}.npz'
    return f'{subset}.{model_args.short_name}.npz'


def get_subset_paths(data_args: DataArguments, target_dir: Path, subset: Optional[str] = None) -> Tuple[Path, Path]:
    if not subset:
        subset = get_subset_name(data_args)
    return target_dir / f'{subset}.jsonl', target_dir / f'{subset}.labels.csv'


def _iter_selected_months(data_args: DataArguments) -> List[str]:
    start = datetime.fromisoformat(data_args.source.select.start)
    end = datetime.fromisoformat(data_args.source.select.end)
    if start.year != 2023 or end.year != 2023:
        raise ValueError(
            f"Dates must be in year 2023; got start={start.date().isoformat()}, end={end.date().isoformat()}"
        )
    if end < start:
        raise ValueError(f'end must be >= start; got start={start}, end={end}')

    months: List[str] = []
    cur = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while cur < end:
        months.append(f'{cur.year}_{cur.month:02d}')
        cur = cur + relativedelta(months=1)
    return months


def _filter_out_sample(data_args: DataArguments, sample: Dict[str, Any]) -> bool:
    for key, value in data_args.source.select.filter.items():
        if key == 'min_label_count':
            continue
        if key in sample and sample.get(key) != value:
            return True
    return False


def _join_text(article: Dict[str, Any]) -> str:
    title = article.get('title', {}).get('text', '') or ''
    body = article.get('body', {}).get('text', '') or ''
    if title and body:
        return f'{title}\n\n{body}'
    return title or body


def _collect_labels(article: Dict[str, Any],
                    labels_map: Dict[str, Dict[str, str]]) -> Tuple[List[str], List[Dict[str, str]]]:
    labels: List[str] = []
    labels_info: List[Dict[str, str]] = []
    for tag in article.get('tags', []):
        label_id = tag.get('id', '')
        if not label_id:
            continue
        labels.append(label_id)
        labels_info.append({
            'id': label_id,
            'name': labels_map.get(label_id, {}).get('name', '')
        })
    return labels, labels_info


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_sample(article: Dict[str, Any],
                  article_meta: Dict[str, str],
                  media_map: Dict[str, Dict[str, str]],
                  labels_map: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    article_id = article.get('id', '')
    media_id = article.get('m_id', '')
    labels, labels_info = _collect_labels(article, labels_map)
    media_meta = media_map.get(media_id, {})
    text = _join_text(article)

    sample = {
        'id': article_id,
        'a_id': article_id,
        'uuid': article_meta.get('uuid', article_id),
        'a_uuid': article_meta.get('uuid', article_id),
        'date': article.get('date', ''),
        'created': article_meta.get('created', ''),
        'published': article_meta.get('published', ''),
        'm_id': media_id,
        'public': _coerce_int(article.get('public', 0)),
        'lang': article.get('lang', ''),
        'country': article.get('country', ''),
        'mon_country': article.get('mon_country', ''),
        'reach': _coerce_int(media_meta.get('reach', 0)),
        'type': media_meta.get('type', ''),
        'source': media_meta.get('name', ''),
        'url': article_meta.get('url', ''),
        'title': article.get('title', {}),
        'body': article.get('body', {}),
        'text': text,
        'n_tokens': len(text.split()),
        'label': labels,
        'label_info': labels_info,
        'm_social': 1 if media_id in __social_media else 0,
        'dup': 0,
    }
    return sample


def write_labels_file(labels_file: Path, labels_map: Dict[str, Dict[str, str]], label_counts: Counter) -> None:
    with labels_file.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'name', 'parent_id', 'monitoring_country', 'monitoring_industry', 'count'])
        for label_id, count in sorted(label_counts.items(), key=lambda item: (-item[1], item[0])):
            label_meta = labels_map.get(label_id, {})
            writer.writerow([
                label_id,
                label_meta.get('name', ''),
                label_meta.get('parent_id', ''),
                label_meta.get('monitoring_country', ''),
                label_meta.get('monitoring_industry', ''),
                count,
            ])


def _collect_subset(data_args: DataArguments,
                    source_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, str]], Counter]:
    labels_map = read_csv_to_dict(source_dir / 'map_tags.csv')
    media_map = read_csv_to_dict(source_dir / 'map_media.csv')

    samples: List[Dict[str, Any]] = []
    label_counts: Counter = Counter()

    for postfix in _iter_selected_months(data_args):
        data_file = source_dir / f'data_{postfix}.jsonl'
        map_file = source_dir / f'map_articles_{postfix}.csv'
        if not data_file.exists():
            raise FileNotFoundError(f'Missing data file {data_file}')
        if not map_file.exists():
            raise FileNotFoundError(f'Missing article map file {map_file}')

        article_map = read_csv_to_dict(map_file)
        with data_file.open('r', encoding='utf-8') as f_in:
            for line_no, line in enumerate(f_in, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    article = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'Malformed JSON in {data_file} line {line_no}') from exc

                if _filter_out_sample(data_args, article):
                    continue

                article_id = article.get('id', '')
                if article_id not in article_map:
                    logger.warning('Missing article map for %s in %s line %d', article_id, data_file.name, line_no)
                    continue

                sample = _build_sample(article, article_map[article_id], media_map, labels_map)
                samples.append(sample)
                label_counts.update(sample['label'])

    return samples, labels_map, label_counts


def apply_min_label_count(data_args: DataArguments,
                          samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Counter]:
    min_label_count = data_args.source.select.filter.get('min_label_count')
    if min_label_count in (None, ''):
        label_counts: Counter = Counter()
        for sample in samples:
            label_counts.update(sample['label'])
        return samples, label_counts

    # noinspection PyTypeChecker
    threshold = int(min_label_count)
    label_counts: Counter = Counter()
    for sample in samples:
        label_counts.update(sample['label'])

    valid_labels = {label_id for label_id, count in label_counts.items() if count >= threshold}
    filtered_samples: List[Dict[str, Any]] = []
    filtered_counts: Counter = Counter()
    for sample in samples:
        filtered_labels = [label_id for label_id in sample['label'] if label_id in valid_labels]
        if not filtered_labels:
            continue
        sample['label'] = filtered_labels
        if 'label_info' in sample:
            sample['label_info'] = [item for item in sample['label_info'] if item.get('id') in valid_labels]
        filtered_counts.update(filtered_labels)
        filtered_samples.append(sample)

    return filtered_samples, filtered_counts


def build_hnsw_index(data_args: DataArguments, normalized_embeddings: np.ndarray) -> faiss.IndexHNSWFlat:
    dimension = normalized_embeddings.shape[1]

    attrs = data_args.cluster.attributes
    hnsw_m = attrs['hnsw_m'] if 'hnsw_m' in attrs else 32
    hnsw_ef_construction = attrs['hnsw_ef_construction'] if 'hnsw_ef_construction' in attrs else 200
    hnsw_ef_search = attrs['hnsw_ef_search'] if 'hnsw_ef_search' in attrs else 128
    dist_metric = faiss.METRIC_INNER_PRODUCT
    if 'hnsw_metric' in attrs:
        dist_metric = attrs['dist_metric']
        if dist_metric == 'l2':
            dist_metric = faiss.METRIC_L2
        elif dist_metric == 'ip':
            dist_metric = faiss.METRIC_INNER_PRODUCT
        else:
            raise ValueError(f'Unknown hnsw_metric: {dist_metric}')

    index = faiss.IndexHNSWFlat(dimension, hnsw_m, dist_metric)
    index.hnsw.efConstruction = hnsw_ef_construction
    index.hnsw.efSearch = hnsw_ef_search
    # noinspection PyArgumentList
    index.add(normalized_embeddings)
    return index


def main(data_args: DataArguments) -> None:
    source_dir = paths.get_ctx_path('download')
    if not source_dir.exists():
        logger.error('Download source NewsMon directory not found: %s.', source_dir)
        return

    data_file, labels_file = get_subset_paths(data_args, paths.context)
    samples, labels_map, label_counts = _collect_subset(data_args, source_dir)
    samples, label_counts = apply_min_label_count(data_args, samples)

    with data_file.open('w', encoding='utf-8') as f_out:
        for sample in samples:
            f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
    _write_labels_file(labels_file, labels_map, label_counts)

    logger.info('Prepared %s samples into %s', len(samples), data_file)
    logger.info('Prepared %s labels into %s', len(label_counts), labels_file)
