import os
from logging import Logger
from typing import Any, Dict

from ...app.zip import Zip
from ...app.args.data import DataArguments
from ...app.downloader import Downloader

logger: Logger
paths: Dict[str, Any]


def main(data_args: DataArguments) -> None:
    # todo write embedder
    logger.info(f'Downloading {data_args.dataset_name}')
    password = os.getenv('NEWSMON_PASSWORD')
    if password is None:
        raise Exception('NEWSMON_PASSWORD environment variable not set')

    download_dir = paths['download']['data']
    for link in data_args.source.links:
        zip_file = Downloader.download(link.url, download_dir)
        logger.info(f'Downloaded {zip_file}')
        logger.info(
            f'Extracting {zip_file} to {download_dir} ...'
        )
        Zip.extract(zip_file, download_dir, password=password)
        logger.info(f'Extracted {zip_file}.')
        zip_file.unlink()
