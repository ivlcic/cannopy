from logging import Logger
from pathlib import Path
from tempfile import TemporaryDirectory

from ...app.args.data import DataArguments
from ...app.args.runtime import Paths
from ...app.downloader import Downloader
from ...app.zip import Zip

logger: Logger
paths: Paths


def main(data_args: DataArguments) -> None:
    logger.info(f'Downloading {data_args.dataset_name}')
    zip_file = Downloader.download(
        data_args.source.links[0].url, paths.context / 'news_class_dataset.zip'
    )
    languages = data_args.source.attributes.get('languages', ['hr', 'mk', 'sl', 'sr'])

    with TemporaryDirectory(prefix='tmp_iptc_31_', dir=paths.context) as tmp_dir:
        extract_dir = Path(tmp_dir)
        logger.info(f'Extracting {zip_file} to {extract_dir}')
        Zip.extract(zip_file, extract_dir)

        # Validate every split before publishing any files. The archive nests
        # them under news_class_dataset/<language>/.
        split_files = []
        for lang in languages:
            for split in ['train', 'test']:
                filename = f'{lang}_{split}.jsonl'
                matches = sorted(extract_dir.rglob(filename))
                if len(matches) != 1 or not matches[0].is_file() or matches[0].stat().st_size == 0:
                    raise ValueError(f'Expected one non-empty IPTC-31 split file: {filename}')
                split_files.append(matches[0])

        for source in split_files:
            dest = paths.context / source.name
            source.replace(dest)
            logger.info(f'Extracted {dest}')

    zip_file.unlink()
