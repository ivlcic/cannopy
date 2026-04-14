from logging import Logger
from typing import Any, Dict

from ...app.args.data import DataArguments
from ...app.args.model import ModelArguments
from .__embed import embed_prepared_subset

logger: Logger
paths: Dict[str, Any]


# noinspection DuplicatedCode
def main(data_args: DataArguments, model_args: ModelArguments) -> None:
    source_dir = paths['base']['data'] / 'prepare' / 'eurlex'
    source_file = source_dir / f'{data_args.dataset_name}.jsonl'
    if not source_file.exists():
        raise FileNotFoundError(f'Prepared subset file not found: {source_file}')

    target_dir = paths['embed']['data'] / data_args.dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)

    embed_prepared_subset(source_file, target_dir, model_args, logger)
