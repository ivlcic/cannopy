import shutil

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


def _copy_downloaded(files_paths: List[Path], target_dir: Path) -> None:
    for file in files_paths:
        if file.is_file() and file.suffix == '.jsonl':
            d = target_dir / file.parent.name
            d.mkdir(parents=True, exist_ok=True)
            logger.info(f"Copying %s to %s.", file, d)
            shutil.copy2(file, d)
        if file.is_dir():
            d = target_dir / file.name
            d.mkdir(parents=True, exist_ok=True)
            for child in file.iterdir():
                if child.is_file() and child.suffix == '.jsonl':
                    logger.info(f"Copying %s to %s.", child, d)
                    shutil.copy2(child, d)


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
