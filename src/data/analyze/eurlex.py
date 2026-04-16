from logging import Logger

from . import newsmon as newsmon_
from ...app.args.data import DataArguments
from ...app.args.runtime import Paths

logger: Logger
paths: Paths


def main(data_args: DataArguments) -> None:
    newsmon_.logger = logger
    newsmon_.paths = paths
    newsmon_.main(data_args)
