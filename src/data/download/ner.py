# noinspection PyPackages
from .common import download_to_file, extract_zip


# noinspection PyUnresolvedReferences
def main(data_args) -> None:
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
