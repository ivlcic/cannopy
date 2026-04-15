from logging import Logger

from ...app.args.data import DataArguments
from ...app.args.runtime import Paths
from ...app.downloader import Downloader
from ...app.zip import Zip

logger: Logger
paths: Paths


def main(data_args: DataArguments) -> None:
    logger.info(f'Downloading {data_args.dataset_name}.')
    for link in data_args.source.links:
        zip_file = Downloader.download(link.url, paths.context)
        logger.info(f'Downloaded {zip_file}.')
        extract_dir = paths.context / zip_file.stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f'Extracting {zip_file} to {extract_dir}...'
        )
        Zip.extract(zip_file, extract_dir)
        logger.info(f'Extracted {zip_file} to {extract_dir}.')
        zip_file.unlink()
