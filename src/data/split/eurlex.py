from logging import Logger
from typing import Any, Dict, List

from .__common import _load_samples, _split_ids_random, _group_by_article_id, _write_split_file
from ...app.args.data import DataArguments

logger: Logger
paths: Dict[str, Any]


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
        split_ids[split_name].add(article_id)
    return split_ids


def main(data_args: DataArguments) -> None:
    source_dir = paths['base']['data'] / 'prepare' / 'eurlex'
    source_file = source_dir / f'{data_args.dataset_name}.jsonl'
    if not source_file.exists():
        raise FileNotFoundError(f'Prepared subset file not found: {source_file}')

    target_dir = paths['split']['data'] / 'eurlex' / data_args.dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)

    samples = _load_samples(source_file)
    grouped = _group_by_article_id(samples)

    use_eurlex_split = bool(data_args.source.select.filter.get('use_eurlex_split', False))
    if use_eurlex_split:
        split_ids = _split_ids_predefined(grouped)
    else:
        split_ids = _split_ids_random(grouped, data_args)

    split_counts: Dict[str, int] = {}
    for split_name in ['train', 'eval', 'test']:
        target_file = target_dir / f'{data_args.dataset_name}.{split_name}.jsonl'
        split_counts[split_name] = _write_split_file(target_file, split_ids[split_name], grouped)

    logger.info(
        'Split %s into train/eval/test samples: %s/%s/%s',
        data_args.dataset_name,
        split_counts['train'],
        split_counts['eval'],
        split_counts['test'],
    )
