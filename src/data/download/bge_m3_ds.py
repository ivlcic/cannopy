import shutil
from logging import Logger
from pathlib import Path

from ...app.args.data import DataArguments
from ...app.args.runtime import Paths
from ...app.downloader import Downloader
from ...app.zip import Zip

logger: Logger
paths: Paths


def _download_all(data_args: DataArguments, target_dir: Path) -> None:
    for link in data_args.source.links:
        dest = target_dir / link.url.split('/')[-1]
        if not dest.exists():
            logger.info('Downloading %s...', link.url)
            local_path = Downloader.download(link.url, dest)
            logger.info('Downloaded %s to %s.', link.url, local_path)
        else:
            local_path = dest
        file_name = dest.name
        stem = file_name.split('.')[0]
        if file_name.endswith('.tar.gz'):
            logger.info('Decompressing %s...', local_path)
            unzipped = Zip.untar(local_path)
            logger.info('Decompressed %s to %s.', local_path, unzipped)
            root_dir = dest.parent / stem
            if root_dir.exists():
                new_path = dest.parent / data_args.dataset_name
                shutil.move(root_dir, new_path)
                logger.info('Renamed %s to %s.', root_dir.name, new_path.name)
                local_path.unlink()


def main(data_args: DataArguments) -> None:
    logger.info('Downloading dataset %s', data_args.dataset_name)
    if not data_args.source.links:
        logger.error('No dataset.source.links provided for %s.', data_args.dataset_name)
        return

    _download_all(data_args, paths.context)
