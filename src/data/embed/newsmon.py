from logging import Logger
from typing import Any, Dict

from ...app.args.data import DataArguments
from ...app.args.model import ModelArguments
from ...app.embedder import TextEmbedder

logger: Logger
paths: Dict[str, Any]


def main(data_args: DataArguments, model_args: ModelArguments) -> None:
    # todo write embedder
    embedder = TextEmbedder.create(model_args)
    pass
