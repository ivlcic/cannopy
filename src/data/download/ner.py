from logging import Logger
from typing import Dict, Any

from ...app.downloader import Downloader
from ...app.zip import Zip
from ...app.args.data import DataArguments

logger: Logger
paths: Dict[str, Any]


def main(data_args: DataArguments) -> None:
    logger.info(f'Downloading {data_args.dataset_name}')

    download_dir = paths['download']['data']
    for u in data_args.dataset_urls:
        zip_file = Downloader.download(u, download_dir / u.split('/')[-1])
        logger.info(f'Downloaded {zip_file}')
        extract_dir = download_dir / 'ner' / zip_file.stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f'Extracting {zip_file} to {extract_dir}'
        )
        logger.info(f'Extracted {zip_file} to {extract_dir}')
        Zip.extract(zip_file, extract_dir)
        zip_file.unlink()
