from logging import Logger
from typing import Any, Dict, List

from .newsmon import split_ids_random, group_by_article_id, write_split_file
from ...app.args.data import DataArguments
from ...app.args.runtime import Paths
from ...app.helpers import JsonlLoader

logger: Logger
paths: Paths


def _split_ids_predefined(grouped: Dict[str, List[Dict[str, Any]]]) -> Dict[str, set[str]]:
    split_ids = {
        'train': set(),
        'eval': set(),
        'test': set(),
    }
    for article_id, article_samples in grouped.items():
        split_name = article_samples[0].get('split')
        if split_name not in split_ids:
            raise ValueError(f"Unsupported predefined split '{split_name}' for sample {article_id}")
        # noinspection PyTypeChecker
        split_ids[split_name].add(article_id)
    return split_ids


def main(data_args: DataArguments) -> None:
    source_dir = paths.get_ctx_path('prepare')
    source_file = source_dir / f'{data_args.dataset_name}.jsonl'
    if not source_file.exists():
        raise FileNotFoundError(f'Prepared subset file not found: {source_file}')

    samples = JsonlLoader.load_samples(source_file)
    grouped = group_by_article_id(samples)

    use_eurlex_split = bool(data_args.source.select.filter.get('use_eurlex_split', False))
    if use_eurlex_split:
        split_ids = _split_ids_predefined(grouped)
    else:
        split_ids = split_ids_random(grouped, data_args)

    split_counts: Dict[str, int] = {}
    for split_name in ['train', 'eval', 'test']:
        target_file = paths.context / f'{data_args.dataset_name}.{split_name}.jsonl'
        split_counts[split_name] = write_split_file(target_file, split_ids[split_name], grouped)

    logger.info(
        'Split %s into train/eval/test samples: %s/%s/%s',
        data_args.dataset_name,
        split_counts['train'],
        split_counts['eval'],
        split_counts['test'],
    )
