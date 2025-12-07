import json
import shutil

from pathlib import Path
from logging import Logger
from typing import Iterable, List, Dict, Any

from ...app.common import PathLike
from ...app.args.data import DataArguments
from ...app.downloader import Downloader
from ...app.zip import Zip


logger: Logger
paths: Dict[str, Any]


class MergeError(Exception):
    pass


def _gather_json_files(folder: Path) -> List[Path]:
    """Return a sorted list of .json files inside a folder (non-recursive)."""
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".json"])


def _iter_json_records_from_file(fp: Path):
    """
    Yield JSON-serializable objects from a file.
    If the file contains a JSON array, yield each element.
    If it contains a single object (dict, number, string...), yield it once.
    """
    text = fp.read_text(encoding="utf-8")
    obj = json.loads(text)
    if isinstance(obj, list):
        for item in obj:
            yield item
    else:
        yield obj


def _write_jsonl_atomic(records: Iterable, dest: Path) -> None:
    """
    Write records (iterable of JSON-serializable objects) to dest as JSON lines atomically.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    with tmp.open("w", encoding="utf-8") as out:
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=False))
            out.write("\n")

    if not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise MergeError(f"Failed to write non-empty JSONL to {dest}")
    tmp.replace(dest)


def merge_eurlex_jsons_and_remove_dir(eurlex_dir: PathLike, prefix: str, target_dir: PathLike) -> List[Path]:
    eurlex_path = Path(eurlex_dir).resolve()
    if not eurlex_path.exists() or not eurlex_path.is_dir():
        raise MergeError(f"Provided eurlex_dir does not exist or is not a directory: {eurlex_path}")

    dataset_dir = eurlex_path / "dataset"
    splits = ["dev", "test", "train"]
    split_dirs = {s: dataset_dir / s for s in splits}

    # validate structure exists
    missing = [s for s, d in split_dirs.items() if not d.exists() or not d.is_dir()]
    if missing:
        raise MergeError(f"Missing expected split directories under {dataset_dir}: {missing}")

    output_paths: List[Path] = []

    # For each split, collect records and write an atomic JSON lines file
    for split in splits:
        sd = split_dirs[split]
        json_files = _gather_json_files(sd)
        if not json_files:
            raise MergeError(f"No .json files found in {sd} — aborting to avoid producing empty output.")
        dest = target_dir / f"{prefix}.{split}.jsonl"

        # generator that yields records from all files in order
        def records_generator(files: List[Path]):
            for f in files:
                for rec in _iter_json_records_from_file(f):
                    yield rec

        _write_jsonl_atomic(records_generator(json_files), dest)
        output_paths.append(dest)

    # remove eurlex dir
    shutil.rmtree(eurlex_path, ignore_errors=True)

    return output_paths


def main(data_args: DataArguments) -> None:
    logger.info(f"Downloading {data_args.dataset_name}")

    download_dir = paths['download']['data']
    output_dir = paths['base']['data'] / 'prepare'
    zip_file = Downloader.download(data_args.dataset_urls[0],  download_dir / 'eurlex.zip')
    extract_dir = download_dir / 'eurlex'
    extract_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Extracting {zip_file} to {extract_dir}"
    )
    Zip.extract(zip_file, extract_dir)
    merge_eurlex_jsons_and_remove_dir(extract_dir, data_args.dataset_name, output_dir)
    zip_file.unlink()
