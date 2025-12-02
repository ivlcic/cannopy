import json
import shutil

from pathlib import Path
from typing import Iterable, List

from .common import download_to_file, extract_zip, PathLike


# noinspection PyUnresolvedReferences,PyGlobalUndefined
def main(data_args) -> None:
    global logger, paths

    logger.info(f"Downloading {data_args.dataset_name}")

    download_dir = paths['download']['data']
    for u in data_args.dataset_urls:
        zip_file = download_to_file(u, download_dir / u.split('/')[-1])
        logger.info(f"Downloaded {zip_file}")
        extract_dir = download_dir / 'ner' / zip_file.stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Extracting {zip_file} to {extract_dir}"
        )
        logger.info(f"Extracted {zip_file} to {extract_dir}")
        extract_zip(zip_file, extract_dir)
        zip_file.unlink()
