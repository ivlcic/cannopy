import shutil
from logging import Logger
from typing import Dict, Any

from ...app.downloader import Downloader
from ...app.zip import Zip
from ...app.args.data import DataArguments

logger: Logger
paths: Dict[str, Any]


def main(data_args: DataArguments) -> None:
    logger.info(f'Downloading {data_args.dataset_name}.')

    download_dir = paths['download']['data']
    for link in data_args.source.links:
        zip_file = Downloader.download(link.url, download_dir / link.url.split('/')[-1])
        logger.info(f'Downloaded {zip_file}.')
        name = zip_file.stem
        if 'sample_submission' in name:
            # we rename so we can use it in evaluation as a sample reference
            name = 'sample_reference'

        extract_dir = download_dir / data_args.dataset_name / name
        extract_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f'Extracting {zip_file} to {extract_dir}...'
        )
        Zip.extract(zip_file, extract_dir)
        logger.info(f'Extracted {zip_file} to {extract_dir}.')
        zip_file.unlink()
