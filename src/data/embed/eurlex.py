from logging import Logger

from .newsmon import (
    store_embedding_array_dict,
    embed_prepared_dataset,
    collect_split_embeddings,
)
from ..prepare.newsmon import get_sidecar_name
from ...app.args.data import DataArguments
from ...app.args.model import ModelArguments
from ...app.args.runtime import Paths

logger: Logger
paths: Paths


# noinspection DuplicatedCode
def main(data_args: DataArguments, model_args: ModelArguments) -> None:
    embeddings = embed_prepared_dataset(paths, data_args, model_args, logger)
    target_name = get_sidecar_name(data_args, model_args)
    target_index_file = paths.context / target_name
    logger.info('Writing embedding array sidecar: %s', target_index_file)
    store_embedding_array_dict(target_index_file, embeddings)

    split_dir = paths.get_ctx_path('split')
    if not split_dir.exists():
        return

    for split_name in ('train', 'eval', 'test'):
        split_source_file = split_dir / f'{data_args.dataset_name}.{split_name}.jsonl'
        if not split_source_file.exists():
            continue
        split_embeddings = collect_split_embeddings(split_source_file, embeddings)
        split_index_file = split_dir / get_sidecar_name(data_args, model_args, split_name)
        store_embedding_array_dict(split_index_file, split_embeddings)
        logger.info('Writing embedding array sidecar: %s', split_index_file)
