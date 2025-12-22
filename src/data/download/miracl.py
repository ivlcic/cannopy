from logging import Logger
from pathlib import Path
from typing import Any, Dict

from ...app.zip import Zip
from ...app.args.data import DataArguments
from ...app.downloader import Downloader

logger: Logger
paths: Dict[str, Any]


def _download_all(data_args: DataArguments, target_dir: Path) -> None:
    for link in data_args.source.links:
        dest = target_dir / link.lang
        dest.mkdir(parents=True, exist_ok=True)
        local_path = Downloader.download(link.url, dest)
        logger.info('Downloaded %s to %s', link.url, local_path)
        unzipped = Zip.ungzip(local_path)
        logger.info('Decompressed %s to %s', local_path, unzipped)


def main(data_args: DataArguments) -> None:
    logger.info('Downloading %s', data_args.dataset_name)
    if not data_args.source.links:
        logger.error('No dataset_urls provided for %s', data_args.dataset_name)
        return

    target_dir = paths['download']['data'] / data_args.dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)
    _download_all(data_args, target_dir)
