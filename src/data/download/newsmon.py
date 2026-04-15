import os
from logging import Logger

from ...app.args.data import DataArguments
from ...app.args.runtime import Paths
from ...app.downloader import Downloader
from ...app.zip import Zip

logger: Logger
paths: Paths


def main(data_args: DataArguments) -> None:
    logger.info(f'Downloading {data_args.dataset_name}')
    password = os.getenv('NEWSMON_PASSWORD')
    if password is None:
        raise Exception('NEWSMON_PASSWORD environment variable not set')

    for link in data_args.source.links:
        zip_file = Downloader.download(link.url, paths.context)
        logger.info(f'Downloaded {zip_file}')
        logger.info(
            f'Extracting {zip_file} to {paths.context} ...'
        )
        Zip.extract(zip_file, paths.context, password=password)
        logger.info(f'Extracted {zip_file}.')
        zip_file.unlink()
