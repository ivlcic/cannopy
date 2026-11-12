from logging import Logger

from .newsmon import embed_prepared_dataset, store_embedding_array_dict
from ..prepare.newsmon import get_sidecar_name, get_subset_name
from ...app.args.data import DataArguments
from ...app.args.model import ModelArguments
from ...app.args.runtime import Paths

logger: Logger
paths: Paths


def main(data_args: DataArguments, model_args: ModelArguments) -> None:
    if not data_args.source.select.subset:
        raise ValueError('Select an IPTC-31 language with -c hr, -c mk, -c sl, or -c sr')
    subset = get_subset_name(data_args)
    for split in ['train', 'test']:
        logger.info('Computing embeddings: %s.%s ...', subset, split)
        embeddings = embed_prepared_dataset(paths, data_args, model_args, logger, split=split)
        sidecar_file = paths.context / get_sidecar_name(data_args, model_args, split)
        arrays = store_embedding_array_dict(sidecar_file, embeddings)
        logger.info(
            'Wrote %s embedding vectors with shape %s to %s',
            len(arrays['ids']), arrays['embeddings'].shape, sidecar_file,
        )
