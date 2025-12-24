from logging import Logger
from pathlib import Path
from typing import Any, Dict, Iterable

from ...app.args.data import DataArguments, SourceConfigLink
from ...app.downloader import Downloader

logger: Logger
paths: Dict[str, Any]


def _build_target_dir(dataset_name: str, config_name: str) -> Path:
    target_dir = paths['download']['data'] / dataset_name
    if config_name:
        target_dir = target_dir / config_name
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir.resolve()


def _download_all(links: Iterable[SourceConfigLink], target_dir: Path) -> None:
    for link in links:
        local_path = Downloader.download(link.url, target_dir)
        logger.info('Downloaded %s to %s.', link.url, local_path)


def main(data_args: DataArguments) -> None:
    logger.info('Downloading %s', data_args.dataset_name or 'msmarco...')
    if not data_args.source.links:
        logger.error('No source links provided for %s!', data_args.dataset_name)
        return

    target_dir = _build_target_dir(data_args.dataset_name or 'msmarco', data_args.version)
    _download_all(data_args.source.urls, target_dir)
