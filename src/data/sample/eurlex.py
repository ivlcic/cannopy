from logging import Logger

from . import newsmon as newsmon_
from ...app.args.data import DataArguments
from ...app.args.model import ModelArguments
from ...app.args.runtime import Paths

logger: Logger
paths: Paths


def hard_neg(data_args: DataArguments, model_args: ModelArguments) -> None:
    newsmon_.logger = logger
    newsmon_.paths = paths
    newsmon_.hard_neg(data_args, model_args)
