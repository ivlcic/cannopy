import json
import random

from collections import Counter, defaultdict
from typing import Any, Dict, List

from ...app.args.data import DataArguments


def _load_samples(source_file) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with source_file.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Malformed JSON in {source_file} line {line_no}') from exc
            samples.append(sample)
    return samples


def _group_by_article_id(samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        article_id = sample.get('a_id') or sample.get('id')
        if not article_id:
            raise ValueError('Sample is missing both a_id and id.')
        # noinspection PyTypeChecker
        grouped[article_id].append(sample)
    return grouped


def _split_ids_random(grouped: Dict[str, List[Dict[str, Any]]], data_args: DataArguments) -> Dict[str, set[str]]:
    article_ids = list(grouped.keys())
    rng = random.Random(data_args.split.seed)
    rng.shuffle(article_ids)

    num_ids = len(article_ids)
    train_count = int(num_ids * data_args.split.train)
    eval_count = int(num_ids * data_args.split.eval)
    test_count = num_ids - train_count - eval_count
    if test_count < 0:
        raise ValueError(
            f'Invalid split ratios train={data_args.split.train} '
            f'eval={data_args.split.eval} test={data_args.split.test}'
        )

    train_ids = set(article_ids[:train_count])
    eval_ids = set(article_ids[train_count:train_count + eval_count])
    test_ids = set(article_ids[train_count + eval_count:])
    return {
        'train': train_ids,
        'eval': eval_ids,
        'test': test_ids,
    }


def _write_split_file(target_file, split_ids: set[str], grouped: Dict[str, List[Dict[str, Any]]]) -> int:
    written = 0
    with target_file.open('w', encoding='utf-8') as f:
        for article_id in split_ids:
            for sample in grouped[article_id]:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                written += 1
    return written
