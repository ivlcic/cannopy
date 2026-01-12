import os
import shutil
import tempfile

from logging import Logger
from pathlib import Path
from typing import Any, Dict, List

from ...app.args.data import DataArguments

logger: Logger
paths: Dict[str, Any]


def get_files_paths(source_dir: Path) -> List[Path]:
    return [
        source_dir / 'Law-Medical_data' / 'colliee_len-0-500.jsonl',
        source_dir / 'Law-Medical_data' / 'pubmed_qa_labeled_len-0-500.jsonl',
        source_dir / 'en_NLI_data',
        source_dir / 'MIRACL' / 'miracl_en_len-0-500.jsonl',
        source_dir / 'MIRACL' / 'miracl_en_len-500-1000.jsonl',
        source_dir / 'MIRACL' / 'miracl_en_len-1000-2000.jsonl',
        source_dir / 'MIRACL' / 'miracl_en_len-2000-3000.jsonl',
        source_dir / 'MIRACL' / 'miracl_en_len-3000-4000.jsonl',
        source_dir / 'Mr.TyDi' / 'mr-tydi_english_len-0-500.jsonl',
        source_dir / 'Mr.TyDi' / 'mr-tydi_english_len-500-1000.jsonl',
        source_dir / 'Mr.TyDi' / 'mr-tydi_english_len-1000-2000.jsonl',
        source_dir / 'Mr.TyDi' / 'mr-tydi_english_len-2000-3000.jsonl',
        source_dir / 'Mr.TyDi' / 'mr-tydi_english_len-3000-4000.jsonl',
        source_dir / 'Mr.TyDi' / 'mr-tydi_english_len-4000-5000.jsonl',
        source_dir / 'Mr.TyDi' / 'mr-tydi_english_len-5000-6000.jsonl',
        source_dir / 'SQuAD',
        source_dir / 'HotpotQA',
        source_dir / 'MLDR' / 'mldr_en_len-2000-3000.jsonl',
        source_dir / 'MLDR' / 'mldr_en_len-3000-4000.jsonl',
        source_dir / 'MLDR' / 'mldr_en_len-4000-5000.jsonl',
        source_dir / 'MLDR' / 'mldr_en_len-5000-6000.jsonl',
        source_dir / 'MLDR' / 'mldr_en_len-6000-7000.jsonl',
        source_dir / 'MLDR' / 'mldr_en_len-7000-inf.jsonl',
        source_dir / 'NQ',
        source_dir / 'Trivia',
        source_dir / 'MSMARCO',
    ]


def remove_lines_containing_any(path: Path, terms: List[str], enc: str = "utf-8", cs: bool = False) -> int:
    path = Path(path)
    if not cs:
        terms = [t.lower() for t in terms]

    removed = 0
    with tempfile.NamedTemporaryFile(mode="w", delete=False, dir=str(path.parent), encoding=enc, newline="") as tmp:
        tmp_path = Path(tmp.name)

        with path.open("r", encoding=enc, newline="") as f:
            prev = ''
            for line_no, line in enumerate(f, start=1):
                hay = line.lower() if not cs else line
                if any(term in hay for term in terms):
                    removed += 1
                    tmp.write(prev)
                    logger.info(
                        f"Replacing line [%s::%s] offensive [%s] with [%s].", line_no, path, hay, prev
                    )
                    continue
                tmp.write(line)
                prev = line

    os.replace(tmp_path, path)
    return removed


def _copy_downloaded(files_paths: List[Path], target_dir: Path) -> None:
    offensive = ['naked little girl']
    for file in files_paths:
        if file.is_file() and file.suffix == '.jsonl':
            d = target_dir / file.parent.name
            d.mkdir(parents=True, exist_ok=True)
            logger.info(f"Copying %s to %s.", file, d)
            shutil.copy2(file, d)
            remove_lines_containing_any(d / file.name, offensive, enc="utf-8", cs=True)
        if file.is_dir():
            d = target_dir / file.name
            d.mkdir(parents=True, exist_ok=True)
            for child in file.iterdir():
                if child.is_file() and child.suffix == '.jsonl':
                    logger.info(f"Copying %s to %s.", child, d)
                    shutil.copy2(child, d)
                    if 'msmarco' in child.name.lower():
                        continue
                    remove_lines_containing_any(d / child.name, offensive, enc="utf-8", cs=True)


def main(data_args: DataArguments) -> None:
    source_dir = paths["base"]["data"] / "download" / data_args.dataset_name
    if not source_dir.exists():
        logger.error("Source MIRACL directory not found: %s.", source_dir)
        return

    files_paths = get_files_paths(source_dir)
    target_dir = paths["prepare"]["data"] / data_args.dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Copying %s to %s.", source_dir, target_dir)
    _copy_downloaded(files_paths, target_dir)
