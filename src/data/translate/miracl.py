import csv
import json

from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from ...app.args.data import DataArguments, TranslateModelsConfig

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


def main(data_args: DataArguments) -> None:
    t_cfg = data_args.translate

    source_dir = paths["base"]["data"] / "download" / "miracl" / t_cfg.src_lang
    if not source_dir.exists():
        logger.error("Source MIRACL directory not found: %s", source_dir)
        return

    target_dir = paths["translate"]["data"] / "miracl" / t_cfg.lang
    target_dir.mkdir(parents=True, exist_ok=True)

    topic_files = _iter_topic_files(source_dir)
    if not topic_files:
        logger.warning("No topics files found in %s", source_dir)
        return

    total = 0
    for topic_file in topic_files:
        out_file = target_dir / topic_file.name
        with topic_file.open("r", encoding="utf-8", newline="") as f_in, \
                out_file.open("w", encoding="utf-8", newline="") as f_out:
            reader = csv.reader(f_in, delimiter="\t")
            writer = csv.writer(f_out, delimiter="\t")
            count = 0
            batch: List[str] = []
            idx_batch: List[str] = []
            for row in reader:
                if len(row) < 2:
                    continue
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
        logger.info("Translated %d rows from %s -> %s", count, topic_file.name, out_file)

    logger.info("Translated %d total rows into %s", total, target_dir)
