import json

from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Optional, IO

from ...app.args.runtime import Paths
from ...app.translator import Translator
from ...app.args.data import DataArguments
from ...data.prepare.bge_m3_ds import get_files_paths

logger: Logger
paths: Paths


# noinspection DuplicatedCode
def main(data_args: DataArguments) -> None:
    raise Exception('Not implemented yet')
    t_cfg = data_args.translate

    source_dir = paths['base']['data'] / 'prepare' / data_args.dataset_name
    if not source_dir.exists():
        logger.error(f'Source [prepare] {data_args.dataset_name} directory not found: %s', source_dir)
        return

    target_dir = paths['translate']['data'] / data_args.dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)
    files_paths = get_files_paths(source_dir)
    files: Dict[Path, Path] = {}
    for file_or_path in files_paths:
        if file_or_path.is_file() and file_or_path.suffix == '.jsonl':
            d = target_dir / file_or_path.parent.name
            d.mkdir(parents=True, exist_ok=True)
            files[file_or_path] = d / (t_cfg.tgt_code + '.' + t_cfg.model.short_name + '.' + file_or_path.name)
        if file_or_path.is_dir():
            d = target_dir / file_or_path.name
            d.mkdir(parents=True, exist_ok=True)
            for child in file_or_path.iterdir():
                if child.is_file() and child.suffix == '.jsonl':
                    files[child] = d / (t_cfg.tgt_code + '.' + t_cfg.model.short_name + '.' + child.name)
    translator: Translator = Translator.create(t_cfg)
    for src, tgt in files.items():
        logger.info('Sampling docs from %s -> %s...', src.name, tgt.name)

        logger.info('Sampled docs from %s -> %s.', src.name, tgt.name)
