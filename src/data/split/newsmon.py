import json
import random
from collections import defaultdict
from logging import Logger
from typing import Dict, List, Any

from ...app.helpers import JsonlLoader
from ...app.args.data import DataArguments
from ...app.args.runtime import Paths

logger: Logger
paths: Paths


def _get_subset_name(data_args: DataArguments) -> str:
    subset = data_args.source.select.subset
    if subset:
        return subset
    return data_args.dataset_name


def group_by_article_id(samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        article_id = sample.get('a_id') or sample.get('id')
        if not article_id:
            raise ValueError('Sample is missing both a_id and id.')
        # noinspection PyTypeChecker
        grouped[article_id].append(sample)
    return grouped


def split_ids_random(grouped: Dict[str, List[Dict[str, Any]]], data_args: DataArguments) -> Dict[str, set[str]]:
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


def write_split_file(target_file, split_ids: set[str], grouped: Dict[str, List[Dict[str, Any]]]) -> int:
    written = 0
    with target_file.open('w', encoding='utf-8') as f:
        for article_id in split_ids:
            for sample in grouped[article_id]:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                written += 1
    return written


def main(data_args: DataArguments) -> None:
    subset = _get_subset_name(data_args)

    source_dir = paths.get_ctx_path('prepare')
    source_file = source_dir / f'{subset}.jsonl'
    if not source_file.exists():
        raise FileNotFoundError(f'Prepared subset file not found: {source_file}')

    samples = JsonlLoader.load_samples(source_file)
    grouped = group_by_article_id(samples)
    split_ids = split_ids_random(grouped, data_args)

    split_counts: Dict[str, int] = {}
    for split_name in ['train', 'eval', 'test']:
        target_file = paths.context / f'{subset}.{split_name}.jsonl'
        split_counts[split_name] = write_split_file(target_file, split_ids[split_name], grouped)

    logger.info(
        'Split %s into train/eval/test samples: %s/%s/%s',
        subset,
        split_counts['train'],
        split_counts['eval'],
        split_counts['test'],
    )
