import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, IO, Tuple

from .translator import Translator

logger = logging.getLogger('json-helper')


class JsonIRHelper:

    @classmethod
    def read_ir_sample(cls, line: str, line_no: int, source: Path) -> Optional[Dict[str, Any]]:
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


class JsonIdHelper:

    @classmethod
    def read_sample(cls, line: str, line_no: int, source: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        line = line.strip()
        if not line:
            return None, None
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'Malformed JSON in {source} line {line_no}') from exc
        if 'id' not in obj:
            logger.warning('Missing id in %s line %d.', source, line_no)
            return None, None
        return obj, obj['id']


class JsonlLoader:

    @classmethod
    def load_samples(cls, source_file) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        with source_file.open('r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'Malformed JSON in {source_file} line {line_no}') from exc
                samples.append(sample)
        return samples

    @classmethod
    def write_samples(cls, data_file: Path, samples: list[dict[str, Any]]) -> None:
        with data_file.open('w', encoding='utf-8') as f_out:
            for sample in samples:
                f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')



class TranslationHelper:

    @classmethod
    def translate_file(cls, translator: Translator, source: Path, target: Path, batch_size: int = 1) -> None:
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

                obj = JsonIRHelper.read_ir_sample(line, line_no, source)
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
