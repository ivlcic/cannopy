import csv
import json
import shutil

from logging import Logger
from pathlib import Path
from typing import Any, Dict, List

from ...app.args.data import DataArguments, TranslateConfig, TranslateModelsConfig

logger: Logger
paths: Dict[str, Any]

__api_clients: Dict[str, Any] = {}


def _select_translation_docs(data_args: DataArguments, source_dir: Path, target_dir: Path) -> None:
    t_cfg = data_args.translate
    out_file = target_dir / f'docs-qrels-{t_cfg.src_lang}.jsonl'
    if out_file.exists():
        return

    doc_files = sorted([p for p in source_dir.glob("*.jsonl") if p.is_file()])
    if not doc_files:
        logger.warning("No doc files found in %s", source_dir)
        return

    for doc_file in doc_files:
        with doc_file.open("r", encoding="utf-8") as f_in, out_file.open("a", encoding="utf-8") as f_out:
            logger.info(f"Processing {doc_file.name}")
            for line_no, line in enumerate(f_in, start=1):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON in %s line %d!", doc_file.name, line_no)
                    continue
                doc_id = obj.get("docid", "")
                if doc_id in docids:
                    f_out.write(line)


def main(data_args: DataArguments) -> None:
    t_cfg = data_args.translate

    source_dir = paths['base']['data'] / 'prepare' / 'miracl' / t_cfg.src_lang
    if not source_dir.exists():
        logger.error('Source [prepare] MIRACL directory not found: %s', source_dir)
        return

    target_dir = paths['translate']['data'] / 'miracl' / t_cfg.lang
    target_dir.mkdir(parents=True, exist_ok=True)
    _copy_related(t_cfg, source_dir, target_dir)
    _translate_topics(t_cfg, source_dir, target_dir)
    _translate_docs(t_cfg, target_dir)