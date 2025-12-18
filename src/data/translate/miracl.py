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


def _iter_topic_files(topic_dir: Path) -> List[Path]:
    return sorted([p for p in topic_dir.glob("topics.*") if p.is_file()])


def _translate_text(payload: List[str], prompt: str, models: TranslateModelsConfig) -> List[str]:
    model = models.default
    if 'openai' in model.provider:
        if 'openai' in __api_clients:
            client = __api_clients["openai"]
        else:
            from openai import OpenAI
            client = OpenAI()
            __api_clients["openai"] = client
            logger.debug("Calling OpenAI with model=%s", model.parameters["model"])

        body = {
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": '\n'.join(payload) if len(payload) > 1 else payload[0]},
            ],
        }
        body = body | model.parameters
        response = client.chat.completions.create(**body)
        if len(payload) > 1:
            return response.choices[0].message.content.strip().split("\n")
        else:
            return [response.choices[0].message.content.strip()]
    return []


def _copy_related(t_cfg: TranslateConfig, source_dir, target_dir) -> None:
    qrels_files = sorted([p for p in source_dir.glob("qrels.*") if p.is_file()])
    if not qrels_files:
        logger.warning("No query related files found in %s", source_dir)
        return

    for qrels_file in qrels_files:
        file_name = qrels_file.name.replace(f"-{t_cfg.src_lang}-", f"-{t_cfg.lang}-")
        out_file = target_dir / file_name
        if not out_file.exists():
            shutil.copyfile(qrels_file, out_file)
            logger.info("Copied %s to %s", qrels_file, out_file)


def _translate_topics(t_cfg: TranslateConfig, source_dir, target_dir) -> None:
    topic_files = sorted([p for p in source_dir.glob("topics.*") if p.is_file()])
    if not topic_files:
        logger.warning("No topics files found in %s", source_dir)
        return

    total = 0
    for topic_file in topic_files:
        file_name = topic_file.name.replace(f"-{t_cfg.src_lang}-", f"-{t_cfg.lang}-")
        out_file = target_dir / file_name
        existing = 0
        if out_file.exists():
            with out_file.open("r", encoding="utf-8", newline="") as f_existing:
                existing = sum(1 for _ in f_existing)

        with topic_file.open("r", encoding="utf-8", newline="") as f_in, \
                out_file.open("a", encoding="utf-8", newline="") as f_out:
            reader = csv.reader(f_in, delimiter="\t")
            writer = csv.writer(f_out, delimiter="\t")
            count = 0
            batch: List[str] = []
            idx_batch: List[str] = []
            for line_no, row in enumerate(reader, start=1):
                if len(row) < 2:
                    raise RuntimeError(f"Data in {topic_file} line {line_no} has less than 2 columns!")
                if line_no <= existing:
                    continue  # already translated
                idx, text = row[0], row[1]
                batch.append(text)
                idx_batch.append(idx)
                if len(batch) >= 50:
                    translations = _translate_text(batch, t_cfg.prompt, t_cfg.models) or []
                    for i, translation in enumerate(translations):
                        writer.writerow([idx_batch[i], translation])
                        count += 1
                    batch = []
                    idx_batch = []

            if batch:
                translations = _translate_text(batch, t_cfg.prompt, t_cfg.models) or []
                for i, translation in enumerate(translations):
                    writer.writerow([idx_batch[i], translation])
                    count += 1
            total += count
        if count == 0 and existing > 0:
            logger.info("Skipped %s (already %d rows translated)", topic_file.name, existing)
        else:
            logger.info("Translated %d new rows from %s -> %s", count, topic_file.name, out_file)

    logger.info("Translated %d total rows into %s", total, target_dir)


def _translate_docs(t_cfg: TranslateConfig, source_dir: Path, target_dir: Path) -> None:
    doc_files = sorted([p for p in source_dir.glob("*.jsonl") if p.is_file()])
    if not doc_files:
        logger.warning("No doc files found in %s", source_dir)
        return

    separator = "\n\n------\n\n"
    for doc_file in doc_files:
        out_file = target_dir / doc_file.name
        existing = 0
        if out_file.exists():
            with out_file.open("r", encoding="utf-8") as f_existing:
                existing = sum(1 for _ in f_existing)

        with doc_file.open("r", encoding="utf-8") as f_in, out_file.open("a", encoding="utf-8") as f_out:
            for line_no, line in enumerate(f_in, start=1):
                if line_no <= existing:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON in %s line %d", doc_file.name, line_no)
                    continue
                title = obj.get("title", "")
                text = obj.get("text", "")
                payload = [f"{title}{separator}{text}"]
                translated = _translate_text(payload, t_cfg.prompt, t_cfg.models)
                translated_pair = translated[0] if translated else ""
                translated = translated_pair.split(separator)
                if len(translated) == 2:
                    title = translated[0].strip()
                    text = translated[1].strip()
                else:
                    text = translated[0]
                out_obj = {"title": title, "text": text}
                f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

        logger.info("Translated docs from %s -> %s", doc_file.name, out_file)


def main(data_args: DataArguments) -> None:
    t_cfg = data_args.translate

    source_dir = paths["base"]["data"] / "download" / "miracl" / t_cfg.src_lang
    if not source_dir.exists():
        logger.error("Source MIRACL directory not found: %s", source_dir)
        return

    target_dir = paths["translate"]["data"] / "miracl" / t_cfg.lang
    target_dir.mkdir(parents=True, exist_ok=True)
    _copy_related(t_cfg, source_dir, target_dir)
    _translate_topics(t_cfg, source_dir, target_dir)
    _translate_docs(t_cfg, source_dir, target_dir)
