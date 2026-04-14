from logging import Logger
from typing import Any, Dict

from .__common import _load_samples, _group_by_article_id, _split_ids_random, _write_split_file
from ...app.args.data import DataArguments

logger: Logger
paths: Dict[str, Any]


def get_subset_name(data_args: DataArguments) -> str:
    subset = data_args.source.select.subset
    if subset:
        return subset
    return data_args.dataset_name


def main(data_args: DataArguments) -> None:
    subset = get_subset_name(data_args)

    source_dir = paths['base']['data'] / 'prepare' / data_args.dataset_name
    source_file = source_dir / f'{subset}.jsonl'
    if not source_file.exists():
        raise FileNotFoundError(f'Prepared subset file not found: {source_file}')

    target_dir = paths['split']['data'] / data_args.dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)

    samples = _load_samples(source_file)
    grouped = _group_by_article_id(samples)
    split_ids = _split_ids_random(grouped, data_args)

    split_counts: Dict[str, int] = {}
    for split_name in ['train', 'eval', 'test']:
        target_file = target_dir / f'{subset}.{split_name}.jsonl'
        split_counts[split_name] = _write_split_file(target_file, split_ids[split_name], grouped)

    logger.info(
        'Split %s into train/eval/test samples: %s/%s/%s',
        subset,
        split_counts['train'],
        split_counts['eval'],
        split_counts['test'],
    )
