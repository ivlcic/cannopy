import csv
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from ...app.args.data import DataArguments, TranslateModelsConfig

logger: Logger
paths: Dict[str, Any]

__api_clients: Dict[str, Any] = {}


def _load_topics(topic_dir: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for path in sorted(topic_dir.glob("topics.*")):
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 2:
                    continue
                rows.append((row[0], row[1]))
    return rows


def _translate_text(text: str, prompt: str, models: TranslateModelsConfig) -> Optional[str]:
    model = models.default
    if 'openai' in model.provider:
        if 'openai' in __api_clients:
            client = __api_clients["openai"]
        else:
            from openai import OpenAI
            client = OpenAI()
            __api_clients["openai"] = client

        body = {
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
        }
        body = body | model.parameters
        logger.debug("Calling OpenAI with model=%s", model.parameters["model"])
        response = client.chat.completions.create(**body)
        return response.choices[0].message.content.strip()
    return None


def main(data_args: DataArguments) -> None:
    t_cfg = data_args.translate

    source_dir = paths["base"]["data"] / "download" / "miracl" / t_cfg.src_lang
    if not source_dir.exists():
        logger.error("Source MIRACL directory not found: %s", source_dir)
        return

    topics = _load_topics(source_dir)
    if not topics:
        logger.warning("No topics found in %s", source_dir)
        return

    target_dir = paths["translate"]["data"] / "miracl" / t_cfg.lang
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_dir / "topics.translated.tsv"

    with out_file.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out, delimiter="\t")
        for idx, text in topics:
            translation = _translate_text(text, t_cfg.prompt, t_cfg.models)
            writer.writerow([idx, translation])

    logger.info("Wrote %d translations to %s", len(topics), out_file)
