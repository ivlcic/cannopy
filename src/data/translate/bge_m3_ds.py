from logging import Logger
from pathlib import Path
from typing import Dict

from ..prepare.bge_m3_ds import get_files_paths
from ...app.args.data import DataArguments
from ...app.args.runtime import Paths
from ...app.helpers import TranslationHelper
from ...app.translator import Translator

logger: Logger
paths: Paths


# noinspection DuplicatedCode
def main(data_args: DataArguments) -> None:
    t_cfg = data_args.translate

    source_dir = paths.get_ctx_path('prepare')
    if not source_dir.exists():
        logger.error(f'Source [prepare] {data_args.dataset_name} directory not found: %s', source_dir)
        return

    files_paths = get_files_paths(source_dir)
    files: Dict[Path, Path] = {}
    for file_or_path in files_paths:
        if file_or_path.is_file() and file_or_path.suffix == '.jsonl':
            d = paths.context / file_or_path.parent.name
            d.mkdir(parents=True, exist_ok=True)
            files[file_or_path] = d / (t_cfg.tgt_code + '.' + t_cfg.model.short_name + '.' + file_or_path.name)
        if file_or_path.is_dir():
            d = paths.context / file_or_path.name
            d.mkdir(parents=True, exist_ok=True)
            for child in file_or_path.iterdir():
                if child.is_file() and child.suffix == '.jsonl':
                    files[child] = d / (t_cfg.tgt_code + '.' + t_cfg.model.short_name + '.' + child.name)
    translator: Translator = Translator.create(t_cfg)
    for src, tgt in files.items():
        logger.info('Translating docs from %s -> %s...', src.name, tgt.name)
        TranslationHelper.translate_file(translator, src, tgt, t_cfg.batch_size)
        logger.info('Translated docs from %s -> %s.', src.name, tgt.name)
