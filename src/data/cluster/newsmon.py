import os
from logging import Logger
from typing import Any, Dict

from ...app.zip import Zip
from ...app.args.data import DataArguments
from ...app.downloader import Downloader

logger: Logger
paths: Dict[str, Any]


def main(data_args: DataArguments) -> None:
    logger.info(f'Clustering {data_args.dataset_name}')
    password = os.getenv('NEWSMON_PASSWORD')
    if password is None:
        raise Exception('NEWSMON_PASSWORD environment variable not set')

    download_dir = paths['download']['data']
    extract_dir = download_dir / data_args.dataset_name
    extract_dir.mkdir(parents=True, exist_ok=True)
    for link in data_args.source.links:
        zip_file = Downloader.download(link.url, extract_dir)
        logger.info(f'Downloaded {zip_file}')
        logger.info(
            f'Extracting {zip_file} to {extract_dir} ...'
        )
        Zip.extract(zip_file, extract_dir, password=password)
        logger.info(f'Extracted {zip_file}.')
        zip_file.unlink()
