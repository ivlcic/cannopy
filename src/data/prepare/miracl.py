import json
import shutil
from collections import defaultdict

from logging import Logger
from pathlib import Path
from typing import Any, Dict

from ...app.args.data import DataArguments

logger: Logger
paths: Dict[str, Any]


def _copy_downloaded(source_dir, target_dir) -> None:
    qrels_files = sorted([p for p in source_dir.glob("qrels.*") if p.is_file()])
    if not qrels_files:
        logger.warning("No query related files found in %s", source_dir)
        return
    topic_files = sorted([p for p in source_dir.glob("topics.*") if p.is_file()])
    if not topic_files:
        logger.warning("No topic/query files found in %s", source_dir)
        return

    for file in qrels_files + topic_files:
        out_file = target_dir / file.name
        if not out_file.exists():
            shutil.copyfile(file, out_file)
            logger.info("Copied %s to %s", file, out_file)


def _load_topic(fn):
    qid2topic = {}
    with open(fn, encoding="utf-8") as f:
        for line in f:
            qid, topic = line.strip().split('\t')
            qid2topic[qid] = topic
    return qid2topic


def _load_qrels(fn):
    if fn is None:
        return None

    qrels = defaultdict(dict)
    with open(fn, encoding="utf-8") as f:
        for line in f:
            qid, _, docid, rel = line.strip().split('\t')
            qrels[qid][docid] = int(rel)
    return qrels


def _select_translation_docs(data_args: DataArguments, source_dir: Path, target_dir: Path) -> None:
    t_cfg = data_args.translate
    out_file = target_dir / f'docs-qrels-{t_cfg.src_lang}.jsonl'
    if out_file.exists():
        return

    docids = set()
    for split in ["dev", "train"]:
        topic_file = source_dir / f"topics.miracl-v{data_args.version}-{t_cfg.src_lang}-{split}.tsv"
        qrels_file = source_dir / f"qrels.miracl-v{data_args.version}-{t_cfg.src_lang}-{split}.tsv"
        qid2topic = _load_topic(topic_file)
        qrels = _load_qrels(qrels_file)
        for qid in qid2topic:
            docids.update(qrels[qid].keys())

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

    source_dir = paths["base"]["data"] / "download" / "miracl" / t_cfg.src_lang
    if not source_dir.exists():
        logger.error("Source MIRACL directory not found: %s.", source_dir)
        return

    target_dir = paths["prepare"]["data"] / "miracl" / t_cfg.lang
    target_dir.mkdir(parents=True, exist_ok=True)
    _copy_downloaded(source_dir, target_dir)
    _select_translation_docs(data_args, source_dir, target_dir)
