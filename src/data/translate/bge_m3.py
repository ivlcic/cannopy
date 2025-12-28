import json

from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Optional, IO

from ...app.translator import Translator
from ...app.args.data import DataArguments
from ..prepare.bge_m3 import get_files_paths

logger: Logger
paths: Dict[str, Any]


def _parse_sample(line: str, line_no: int, source: Path) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        logger.warning('Skipping malformed JSON in %s line %d.', source.name, line_no)
        return None
    if 'query' not in obj:
        logger.warning(
            'Skipping malformed JSON in %s line %d, missing query.', source.name, line_no
        )
        return None
    if 'pos' not in obj:
        logger.warning(
            'Skipping malformed JSON in %s line %d, missing positive samples.', source.name, line_no
        )
        return None
    if 'neg' not in obj:
        logger.warning(
            'Skipping malformed JSON in %s line %d, missing negative samples.', source.name, line_no
        )
        return None
    return obj


def _translate_file(translator: Translator, source: Path, target: Path, batch_size: int = 1) -> None:
    existing = 0
    if target.exists():
        with target.open('r', encoding='utf-8') as f_existing:
            existing = sum(1 for _ in f_existing)

    def write_flush(trans: List[Dict[str, Any]], io: IO[Any]) -> None:
        for item in trans:
            io.write(json.dumps(item, ensure_ascii=False))
            io.write('\n')
        io.flush()

    with source.open('r', encoding='utf-8') as f_in, target.open('a', encoding='utf-8') as f_out:
        chunk: List[Dict[str, Any]] = []
        for line_no, line in enumerate(f_in, start=1):
            if line_no <= existing:
                continue

            obj = _parse_sample(line, line_no, source)
            chunk.append(obj)

            if len(chunk) == batch_size:
                trans_chunk = translator.translate_batch(chunk, ['query', 'pos', 'neg'])
                write_flush(trans_chunk, f_out)
                logger.info(
                    'Translated docs %s:%s from %s -> %s.',
                    line_no, line_no + len(chunk), source.name, target.name
                )
                chunk = []

        if chunk:
            trans_chunk = translator.translate_batch(chunk, ['query', 'pos', 'neg'])
            write_flush(trans_chunk, f_out)
            logger.info(
                'Translated docs %s:%s from %s -> %s.',
                line_no, line_no + len(chunk), source.name, target.name
            )


def main(data_args: DataArguments) -> None:
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
            files[file_or_path] = d / (t_cfg.lang + '-' + file_or_path.name)
        if file_or_path.is_dir():
            d = target_dir / file_or_path.name
            d.mkdir(parents=True, exist_ok=True)
            for child in file_or_path.iterdir():
                if child.is_file() and child.suffix == '.jsonl':
                    files[child] = d / (t_cfg.lang + '-' + child.name)
    translator: Translator = Translator.create(t_cfg)
    for src, tgt in files.items():
        logger.info('Translating docs from %s -> %s...', src.name, tgt.name)
        _translate_file(translator, src, tgt)
        logger.info('Translated docs from %s -> %s.', src.name, tgt.name)
