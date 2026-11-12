import json
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List

from .newsmon import apply_min_label_count, write_labels_file
from ...app.args.data import DataArguments
from ...app.args.runtime import Paths

logger: Logger
paths: Paths


def _label_map(data_args: DataArguments) -> Dict[int, Dict[str, Any]]:
    labels = data_args.attributes.get('labels', [])
    if not labels:
        raise ValueError('IPTC-31 requires label mappings in data.attributes.labels')
    mapping = {}
    codes = set()
    for label in labels:
        label_id = label['label_id']
        code = label['iptc_code']
        if not isinstance(code, str) or len(code) != 8 or not code.isdigit():
            raise ValueError(f'IPTC code must be an eight-digit string: {code!r}')
        if label_id in mapping or code in codes:
            raise ValueError(f'Duplicate IPTC-31 label mapping: {label_id}')
        mapping[label_id] = label
        codes.add(code)
    return mapping


def _build_sample(item: Dict[str, Any], split: str, lang: str,
                  labels_map: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    doc_id = item['doc_id']
    if doc_id is None or str(doc_id) == '':
        raise ValueError('Missing document ID')
    text = item['text']
    if not isinstance(text, str):
        raise ValueError('Article text must be a string')
    label_ids = item['labels']
    label_names = item['label_names']
    if not isinstance(label_ids, list) or not isinstance(label_names, list) or len(label_ids) != len(label_names):
        raise ValueError('labels and label_names must be lists of equal length')

    labels_info = []
    for label_id, name in zip(label_ids, label_names):
        if label_id not in labels_map:
            raise ValueError(f'Unknown IPTC-31 label ID: {label_id}')
        label = labels_map[label_id]
        if name != label['label_name']:
            raise ValueError(f'Label name does not match configuration for ID {label_id}: {name!r}')
        labels_info.append({'id': label['iptc_code'], 'name': label['label_name']})

    sample_id = f'{lang}:{doc_id}'
    return {
        **item,
        'id': sample_id,
        'a_id': sample_id,
        'uuid': sample_id,
        'a_uuid': sample_id,
        'public': 1,
        'lang': lang,
        'reach': 0,
        'type': '',
        'source': 'IPTC-31',
        'url': '',
        'title': {'text': '', 'script': '', 'stat': []},
        'body': {'text': text, 'script': '', 'stat': []},
        'n_tokens': len(text.split()),
        'label': [label['id'] for label in labels_info],
        'label_info': labels_info,
        'split': split,
    }


def _iter_split_files(source_dir: Path, languages: List[str]) -> List[tuple[str, str, Path]]:
    splits = []
    for lang in languages:
        for split in ['train', 'test']:
            split_file = source_dir / f'{lang}_{split}.jsonl'
            if not split_file.is_file():
                raise FileNotFoundError(f'Missing IPTC-31 split file: {split_file}')
            splits.append((lang, split, split_file))
    return splits


def main(data_args: DataArguments) -> None:
    source_dir = paths.get_ctx_path('download')
    labels_map = _label_map(data_args)
    languages = data_args.source.attributes.get('languages', ['hr', 'mk', 'sl', 'sr'])
    samples_by_lang = {lang: [] for lang in languages}
    seen_ids = set()
    for lang, split, split_file in _iter_split_files(source_dir, languages):
        with split_file.open(encoding='utf-8') as source:
            for line_no, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    sample = _build_sample(item, split, lang, labels_map)
                    if sample['id'] in seen_ids:
                        raise ValueError(f"Duplicate document ID: {sample['id']}")
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f'Invalid IPTC-31 record in {split_file} line {line_no}: {exc}') from exc
                seen_ids.add(sample['id'])
                samples_by_lang[lang].append(sample)

    csv_labels = {
        label['iptc_code']: {'name': label['label_name']}
        for label in labels_map.values()
    }
    for lang, samples in samples_by_lang.items():
        samples, label_counts = apply_min_label_count(data_args, samples)
        for split in ['train', 'test']:
            data_file = paths.context / f'{data_args.dataset_name}.{lang}.{split}.jsonl'
            split_samples = [sample for sample in samples if sample['split'] == split]
            with data_file.open('w', encoding='utf-8') as output:
                for sample in split_samples:
                    output.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info('Prepared %s IPTC-31 samples into %s', len(split_samples), data_file)

        labels_file = paths.context / f'{data_args.dataset_name}.{lang}.labels.csv'
        write_labels_file(labels_file, csv_labels, label_counts)
        logger.info('Prepared %s IPTC-31 labels into %s', len(label_counts), labels_file)
