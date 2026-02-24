from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Optional, IO

from ...app.translator import Translator
from ...app.args.data import DataArguments

logger: Logger
paths: Dict[str, Any]


# noinspection SpellCheckingInspection
def get_files_paths(source_dir: Path) -> List[Path]:
    return [
        source_dir / 'slobench_ensl.en',
    ]


def _parse_sample(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None

    return {'text': line}


def _translate_file(translator: Translator, source: Path, target: Path, batch_size: int = 1) -> None:
    existing = 0
    if target.exists():
        with target.open('r', encoding='utf-8') as f_existing:
            existing = sum(1 for _ in f_existing)

    def write_flush(trans: List[str], io: IO[Any]) -> None:
        for item in trans:
            io.write(item)
            io.write('\n')
        io.flush()

    with source.open('r', encoding='utf-8') as f_in, target.open('a', encoding='utf-8') as f_out:
        chunk: List[Dict[str, Any]] = []
        for line_no, line in enumerate(f_in, start=1):
            if line_no <= existing:
                continue

            obj = _parse_sample(line)
            chunk.append(obj)

            if len(chunk) == batch_size:
                trans_chunk = translator.translate_batch(chunk, ['text'])
                write_flush([sample['text'] for sample in trans_chunk], f_out)
                logger.info(
                    'Translated docs %s:%s from %s -> %s.',
                    line_no, line_no + len(chunk), source.name, target.name
                )
                chunk = []

        if chunk:
            trans_chunk = translator.translate_batch(chunk, ['text'])
            write_flush([sample['text'] for sample in trans_chunk], f_out)
            logger.info(
                'Translated docs %s:%s from %s -> %s.',
                line_no, line_no + len(chunk), source.name, target.name
            )


# noinspection DuplicatedCode
def main(data_args: DataArguments) -> None:
    t_cfg = data_args.translate
    # t_cfg.batch_size = 10 if t_cfg.batch_size <= 1 else t_cfg.batch_size
    # t_cfg.max_batch_threads = 5 if t_cfg.max_batch_threads <= 1 else t_cfg.max_batch_threads
    source_dir = paths['base']['data'] / 'download' / data_args.dataset_name
    if not source_dir.exists():
        logger.error(f'Source [download] {data_args.dataset_name} directory not found: %s', source_dir)
        return

    target_dir = paths['translate']['data'] / data_args.dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)
    files_paths = get_files_paths(source_dir)
    files: Dict[Path, Path] = {}
    for file_or_path in files_paths:
        if file_or_path.is_file() and file_or_path.suffix == '.txt':
            d = target_dir / file_or_path.parent.name
            d.mkdir(parents=True, exist_ok=True)
            files[file_or_path] = d / (t_cfg.tgt_code + '.' + t_cfg.model.short_name + '.' + file_or_path.name)
        if file_or_path.is_dir():
            d = target_dir / file_or_path.name
            d.mkdir(parents=True, exist_ok=True)
            for child in file_or_path.iterdir():
                if child.is_file() and child.suffix == '.txt':
                    new_name = child.name.replace(f'.{t_cfg.src_code}.', f'.{t_cfg.tgt_code}.')
                    files[child] = d / (t_cfg.tgt_code + '.' + t_cfg.model.short_name + '.' + new_name)
    translator: Translator = Translator.create(t_cfg)
    for src, tgt in files.items():
        logger.info('Translating docs from %s -> %s...', src.name, tgt.name)
        _translate_file(translator, src, tgt, t_cfg.batch_size)
        logger.info('Translated docs from %s -> %s.', src.name, tgt.name)
