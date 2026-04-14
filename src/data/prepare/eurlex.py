import csv
import json

from collections import Counter
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List

from ...app.args.data import DataArguments


logger: Logger
paths: Dict[str, Any]


def _join_text(header: str, main_body: List[str]) -> str:
    body = '\n\n'.join(part.strip() for part in main_body if part and part.strip())
    header = header.strip()
    if header and body:
        return f'{header}\n\n{body}'
    return header or body


def _build_sample(item: Dict[str, Any], split: str, lang: str) -> Dict[str, Any]:
    doc_id = item['celex_id']
    header = item.get('header', '') or ''
    main_body = item.get('main_body', []) or []
    title_text = (item.get('title', '') or '').strip()
    body_text = '\n\n'.join(part.strip() for part in main_body if part and part.strip())
    text = _join_text(header, main_body)
    labels = item.get('concepts', []) or []

    return {
        'id': doc_id,
        'a_id': doc_id,
        'uuid': doc_id,
        'a_uuid': doc_id,
        'public': 1,
        'lang': lang,
        'reach': 0,
        'type': item.get('type', '') or '',
        'source': 'EURLEX57K',
        'url': item.get('uri', '') or '',
        'title': {'text': title_text, 'script': '', 'stat': []},
        'body': {'text': body_text, 'script': '', 'stat': []},
        'text': text,
        'n_tokens': len(text.split()),
        'label': labels,
        'split': split,
    }


def _iter_split_files(source_dir: Path, dataset_name: str) -> List[tuple[str, Path]]:
    splits = []
    for split in ['train', 'eval', 'test']:
        split_file = source_dir / f'{dataset_name}.{split}.jsonl'
        if not split_file.exists():
            raise FileNotFoundError(f'Missing EURLEX split file: {split_file}')
        splits.append((split, split_file))
    return splits


def _apply_min_label_count(data_args: DataArguments,
                           samples: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Counter]:
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
        filtered_counts.update(filtered_labels)
        filtered_samples.append(sample)

    return filtered_samples, filtered_counts


def main(data_args: DataArguments) -> None:
    source_dir = paths['base']['data'] / 'download' / 'eurlex'
    if not source_dir.exists():
        logger.error('Source EURLEX57K directory not found: %s.', source_dir)
        return

    target_dir = paths['prepare']['data'] / 'eurlex'
    target_dir.mkdir(parents=True, exist_ok=True)

    data_file = target_dir / f'{data_args.dataset_name}.jsonl'
    labels_file = target_dir / f'{data_args.dataset_name}.labels.csv'
    lang = data_args.source.lang or 'en'

    samples: List[Dict[str, Any]] = []
    for split, split_file in _iter_split_files(source_dir, data_args.dataset_name):
        with split_file.open('r', encoding='utf-8') as f_in:
            for line_no, line in enumerate(f_in, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'Malformed JSON in {split_file} line {line_no}') from exc

                samples.append(_build_sample(item, split, lang))

    samples, label_counts = _apply_min_label_count(data_args, samples)

    with data_file.open('w', encoding='utf-8') as f_out:
        for sample in samples:
            f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')

    with labels_file.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'name', 'parent_id', 'monitoring_country', 'monitoring_industry', 'count'])
        for label_id, count in sorted(label_counts.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow([label_id, '', '', '', '', count])

    logger.info('Prepared %s EURLEX57K samples into %s', len(samples), data_file)
    logger.info('Prepared %s EURLEX57K labels into %s', len(label_counts), labels_file)
