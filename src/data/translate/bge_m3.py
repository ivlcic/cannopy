import csv
import json

from logging import Logger
from pathlib import Path
from typing import Any, Dict, List

from ...app.translator import Translator
from ...app.args.data import DataArguments, TranslateConfig
from ..prepare.bge_m3 import get_files_paths

logger: Logger
paths: Dict[str, Any]

__api_clients: Dict[str, Any] = {}


def _translate_docs(t_cfg: TranslateConfig, source: Path, target_dir: Path) -> None:
    doc_file = target_dir.parent / f'docs-qrels-{t_cfg.src_lang}.jsonl'
    if not doc_file.exists():
        logger.warning('No source docs language qrels file found in %s', target_dir)
        return

    out_file = target_dir / f'{t_cfg.lang}-{source.name}.jsonl'
    existing = 0
    if out_file.exists():
        with out_file.open('r', encoding='utf-8') as f_existing:
            existing = sum(1 for _ in f_existing)

    with doc_file.open('r', encoding='utf-8') as f_in, out_file.open('a', encoding='utf-8') as f_out:
        for line_no, line in enumerate(f_in, start=1):
            if line_no <= existing:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning('Skipping malformed JSON in %s line %d', doc_file.name, line_no)
                continue
            query: str = obj['query']
            pos: List[str] = obj['pos']
            neg: List[str] = obj['neg']
            pos_scores: List[float] = obj.get('pos_scores', [])
            neg_scores: List[float] = obj.get('neg_scores', [])
            translated = Translator.translate([query] + pos, t_cfg.prompt, t_cfg.models)
            out_obj = {'query': translated[0], 'pos': translated[1:]}
            translated = Translator.translate(neg, t_cfg.prompt, t_cfg.models)
            out_obj['neg'] = translated
            if pos_scores and neg_scores:
                out_obj['pos_scores'] = pos_scores
                out_obj['neg_scores'] = neg_scores
            f_out.write(json.dumps(out_obj, ensure_ascii=False) + '\n')

        logger.info('Translated docs from %s -> %s', doc_file.name, out_file)


def main(data_args: DataArguments) -> None:
    t_cfg = data_args.translate

    source_dir = paths['base']['data'] / 'prepare' / data_args.dataset_name
    if not source_dir.exists():
        logger.error(f'Source [prepare] {data_args.dataset_name} directory not found: %s', source_dir)
        return

    target_dir = paths['translate']['data'] / data_args.dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)
    files_paths = get_files_paths(source_dir)
    for files_path in files_paths:
        _translate_docs(t_cfg, files_path, target_dir)