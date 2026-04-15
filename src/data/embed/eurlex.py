from logging import Logger

from .newsmon import store_embedding_array_dict, embed_prepared_dataset, collect_split_embeddings
from ...app.args.data import DataArguments
from ...app.args.model import ModelArguments
from ...app.args.runtime import Paths

logger: Logger
paths: Paths


# noinspection DuplicatedCode
def main(data_args: DataArguments, model_args: ModelArguments) -> None:
    source_dir = paths.get_ctx_path('prepare')
    source_file = source_dir / f'{data_args.dataset_name}.jsonl'
    if not source_file.exists():
        raise FileNotFoundError(f'Prepared subset file not found: {source_file}')

    embeddings = embed_prepared_dataset(source_file, paths.context, model_args, logger, data_args.dataset_name)
    target_name = f'{data_args.dataset_name}.{model_args.short_name}'
    target_index_file = paths.context / f'{target_name}.npz'
    if not target_index_file.exists():
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
        split_index_file = split_dir / f'{target_name}.{split_name}.npz'
        store_embedding_array_dict(split_index_file, split_embeddings)
        logger.info('Writing embedding array sidecar: %s', split_index_file)
