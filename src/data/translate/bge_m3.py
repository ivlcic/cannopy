import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...app.translator import Translator
from ...app.args.data import DataArguments, TranslateConfig
from ..prepare.bge_m3 import get_files_paths

logger: Logger
paths: Dict[str, Any]

__api_clients: Dict[str, Any] = {}


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


def translate_batched(texts: List[str], translate_fn: Translator.fn, max_chars: int = 2_000) -> List[str]:
    out: List[str] = []
    i = 0

    while i < len(texts):
        batch: List[str] = []
        total = 0

        while i < len(texts):
            s = texts[i]
            batch.append(s)
            total += len(s)
            i += 1

            # break AFTER exceeding (or hitting) the limit
            if total >= max_chars:
                break
        translated = translate_fn(batch)
        out.extend(translated)

    return out


def _translate_sample_once(obj: Dict[str, Any], line_no: int, source: Path,
                           t_cfg: TranslateConfig) -> Optional[Dict[str, Any]]:
    query: str = obj['query']
    positives: List[str] = obj['pos']
    negatives: List[str] = obj['neg']

    source_texts: List[str] = [query] + positives + negatives
    translated = translate_batched(
        texts=source_texts,
        translate_fn=lambda batch: Translator.translate(batch, t_cfg.prompt, t_cfg.models),
        max_chars=2000,
    )
    if len(translated) != len(source_texts):
        logger.warning(
            'Invalid translated lines [%s:%s] in %s line %d. Going to a safe mode.',
            len(translated), len(source_texts), source.name, line_no
        )
        translated = []
        for text in [source_texts[i:i+2] for i in range(0, len(source_texts), 2)]:
            translated.extend(
                translate_batched(
                    texts=text,
                    translate_fn=lambda batch: Translator.translate(batch, t_cfg.prompt, t_cfg.models),
                    max_chars=2000,
                )
            )
        if len(translated) != len(source_texts):
            logger.warning(
                'Invalid translated lines [%s:%s] in %s line %d.',
                len(translated), len(source_texts), source.name, line_no
            )
            exit(1)

    out_obj: Dict[str, Any] = {
        'query': translated[0],
        'pos': [translated[1]],
        'neg': translated[2:],
    }
    pos_scores: List[float] = obj.get('pos_scores', [])
    neg_scores: List[float] = obj.get('neg_scores', [])
    if pos_scores and neg_scores:
        out_obj['pos_scores'] = pos_scores
        out_obj['neg_scores'] = neg_scores
    return out_obj


def _translate_chunk(chunk: List[Tuple[int, Dict[str, Any]]], source: Path, t_cfg: TranslateConfig, f_out) -> None:
    attempts = 10
    while attempts > 0:
        attempts -= 1
        results: Dict[int, Optional[Dict[str, Any]]] = {}
        placeholders: Dict[int, Dict[str, Any]] = {
            line_no: {} for line_no, obj in chunk if obj is None
        }
        with ThreadPoolExecutor(max_workers=len(chunk)) as executor:
            future_map = {
                executor.submit(_translate_sample_once, obj, line_no, source, t_cfg): line_no
                for line_no, obj in chunk if obj is not None
            }
            for future in as_completed(future_map):
                line_no = future_map[future]
                try:
                    results[line_no] = future.result()
                except Exception as exc:
                    logger.error('Translation error in %s line %d: %s', source.name, line_no, exc)
                    results[line_no] = None

        if all((results.get(line_no) or placeholders.get(line_no) is not None) for line_no, _ in chunk):
            # All good: write in original order
            for line_no, _ in chunk:
                out_obj = results.get(line_no) or placeholders.get(line_no) or {}
                f_out.write(json.dumps(out_obj, ensure_ascii=False))
                f_out.write('\n')
            f_out.flush()
            return
        else:
            logger.warning(
                'Retrying chunk in %s lines %s (%d attempts left)',
                source.name, [ln for ln, _ in chunk], attempts
            )

    logger.error(
        'Failed to translate chunk in %s lines %s after 10 attempts; skipping.',
        source.name, [ln for ln, _ in chunk]
    )


def _translate_docs(t_cfg: TranslateConfig, source: Path, target: Path) -> None:
    existing = 0
    if target.exists():
        with target.open('r', encoding='utf-8') as f_existing:
            existing = sum(1 for _ in f_existing)

    with source.open('r', encoding='utf-8') as f_in, target.open('a', encoding='utf-8') as f_out:
        chunk: List[Tuple[int, Dict[str, Any]]] = []
        for line_no, line in enumerate(f_in, start=1):
            if line_no <= existing:
                continue

            obj = _parse_sample(line, line_no, source)
            chunk.append((line_no, obj))

            if len(chunk) == 2:
                _translate_chunk(chunk, source, t_cfg, f_out)
                chunk = []

        if chunk:
            _translate_chunk(chunk, source, t_cfg, f_out)


def _translate_file(translator: Translator, source: Path, target: Path) -> None:
    existing = 0
    if target.exists():
        with target.open('r', encoding='utf-8') as f_existing:
            existing = sum(1 for _ in f_existing)

    with source.open('r', encoding='utf-8') as f_in, target.open('a', encoding='utf-8') as f_out:
        for line_no, line in enumerate(f_in, start=1):
            if line_no <= existing:
                continue

            obj = _parse_sample(line, line_no, source)
            out_obj = translator.trans(obj, ['query', 'pos', 'neg'])
            f_out.write(json.dumps(out_obj, ensure_ascii=False))
            f_out.write('\n')
            f_out.flush()


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
            d = target_dir / file_or_path.parent.name / t_cfg.lang
            d.mkdir(parents=True, exist_ok=True)
            files[file_or_path] = d / file_or_path.name
        if file_or_path.is_dir():
            d = target_dir / file_or_path.name / t_cfg.lang
            d.mkdir(parents=True, exist_ok=True)
            for child in file_or_path.iterdir():
                if child.is_file() and child.suffix == '.jsonl':
                    files[child] = d / child.name
    translator: Translator = Translator.create(t_cfg)
    for src, tgt in files.items():
        logger.info('Translating docs from %s -> %s...', src.name, tgt.name)
        _translate_file(translator, src, tgt)
        logger.info('Translated docs from %s -> %s.', src.name, tgt.name)
