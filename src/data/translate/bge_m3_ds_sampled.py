from logging import Logger
from pathlib import Path
from typing import Any, Dict

from ...app.args.data import DataArguments
from ...app.ir_ds_helper import TranslationHelper
from ...app.translator import Translator

logger: Logger
paths: Dict[str, Any]


# noinspection DuplicatedCode
def main(data_args: DataArguments) -> None:
    t_cfg = data_args.translate

    source_dir = paths['base']['data'] / 'sample' / 'bge-m3-ds'
    if not source_dir.exists():
        logger.error(f'Source [sample] {data_args.dataset_name} directory not found: %s', source_dir)
        return

    target_dir = paths['translate']['data'] / data_args.dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)
    files: Dict[Path, Path] = {}
    for file_or_path in source_dir.iterdir():
        if file_or_path.is_file() and file_or_path.suffix == '.jsonl':
            d = target_dir / file_or_path.parent.name
            d.mkdir(parents=True, exist_ok=True)
            files[file_or_path] = d / (t_cfg.tgt_code + '.' + t_cfg.model.short_name + '.' + file_or_path.name)
    translator: Translator = Translator.create(t_cfg)
    for src, tgt in files.items():
        logger.info('Translating docs from %s -> %s...', src.name, tgt.name)
        TranslationHelper.translate_file(translator, src, tgt, t_cfg.batch_size)
        logger.info('Translated docs from %s -> %s.', src.name, tgt.name)
