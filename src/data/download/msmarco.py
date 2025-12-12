from logging import Logger
from pathlib import Path
from typing import Any, Dict, Iterable

from ...app.args.data import DataArguments
from ...app.downloader import Downloader

logger: Logger
paths: Dict[str, Any]


def _build_target_dir(dataset_name: str, config_name: str) -> Path:
    target_dir = paths['download']['data'] / dataset_name
    if config_name:
        target_dir = target_dir / config_name
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir.resolve()


def _filename_from_url(url: str) -> str:
    url = url.split('?', 1)[0]
    url = url.rstrip('/')
    return url.rsplit('/', 1)[-1]


def _download_all(urls: Iterable[str], target_dir: Path) -> None:
    for url in urls:
        filename = _filename_from_url(url)
        dest = target_dir / filename
        local_path = Downloader.download(url, dest)
        logger.info('Downloaded %s to %s', url, local_path)


def main(data_args: DataArguments) -> None:
    logger.info('Downloading %s', data_args.dataset_name or 'msmarco')
    if not data_args.dataset_urls:
        logger.error('No dataset_urls provided for %s', data_args.dataset_name)
        return

    target_dir = _build_target_dir(data_args.dataset_name or 'msmarco', data_args.dataset_config_name)
    _download_all(data_args.dataset_urls, target_dir)
