import ast
import csv
import json
import sys
from collections import Counter

from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ...app.args.data import DataArguments

logger: Logger
paths: Dict[str, Any]

__social_media = {
    '8e3b359f', '3e1c137d', '86f18af6', '1fd92aa0', 'c0953029', '1843f51e',
    '151a2b9a', '05b54365', '0e9d50b8', '9f6a5e6c', 'f789b185'
}


def _normalize_langs(lang: str) -> List[str]:
    if not lang:
        return []
    return [item.strip() for item in lang.split(",") if item.strip()]


def _set_csv_field_size_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = limit // 10


def _normalize_postfix(value: Any) -> str:
    raw = str(value).strip()
    if len(raw) == 6 and raw.isdigit():
        return f"{raw[:4]}_{raw[4:]}"
    return raw


def _load_labels_map(source_dir: Path) -> Dict[str, Dict[str, str]]:
    labels_map: Dict[str, Dict[str, str]] = {}
    labels_file = source_dir / "map_tags.csv"
    if not labels_file.exists():
        logger.warning("Missing labels map file: %s", labels_file)
        return labels_map

    with labels_file.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            label_id = row.get("id", "")
            if not label_id:
                continue
            labels_map[label_id] = {
                "name": row.get("name", ""),
                "parent_id": row.get("parent_id", ""),
                "monitoring_country": row.get("monitoring_country", ""),
                "monitoring_industry": row.get("monitoring_industry", ""),
            }
    return labels_map


def _load_article_uuid_map(source_dir: Path, postfix: str) -> Dict[str, str]:
    article_map: Dict[str, str] = {}
    article_map_file = source_dir / f"map_articles_{postfix}.csv"
    if not article_map_file.exists():
        logger.warning("Missing map file for chunk %s: %s", postfix, article_map_file)
        return article_map

    with article_map_file.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            article_id = row.get("id", "")
            article_uuid = row.get("uuid", "")
            if article_id and article_uuid:
                article_map[article_id] = article_uuid
    return article_map


def _join_text(article: Dict[str, Any]) -> str:
    title = article.get("title", {}).get("text", "") or ""
    body = article.get("body", {}).get("text", "") or ""
    if title and body:
        return f"{title}\n\n{body}"
    return title or body


def _collect_labels(article: Dict[str, Any], labels_map: Dict[str, Dict[str, str]]) -> Tuple[List[str], List[Dict[str, str]]]:
    labels: List[str] = []
    labels_info: List[Dict[str, str]] = []

    for tag in article.get("tags", []):
        label_id = tag.get("id", "")
        if not label_id:
            continue
        labels.append(label_id)
        label_name = labels_map.get(label_id, {}).get("name", "")
        labels_info.append({"id": label_id, "name": label_name})
    return labels, labels_info


def prep_corpus_extract(
    data_args: DataArguments,
    source_dir: Path,
    target_dir: Path
) -> Tuple[List[Path], Dict[str, Dict[str, str]]]:
    langs = set(_normalize_langs(data_args.lang))
    postfixes = [_normalize_postfix(item) for item in (data_args.subdata_order or [])]
    labels_map = _load_labels_map(source_dir)

    if not postfixes:
        logger.warning("No data chunks configured in data.subdata_order.")
        return [], labels_map

    output_files: List[Path] = []
    for postfix in postfixes:
        article_file = source_dir / f"data_{postfix}.jsonl"
        if not article_file.exists():
            logger.warning("Missing article data file for chunk %s: %s", postfix, article_file)
            continue

        article_uuid_map = _load_article_uuid_map(source_dir, postfix)
        out_file = target_dir / f"{data_args.dataset_name}_{data_args.lang}_{postfix}.csv"
        output_files.append(out_file)

        with (article_file.open("r", encoding="utf-8") as f_in,
              out_file.open("w", encoding="utf-8", newline="") as f_out):
            writer = csv.writer(f_out)
            writer.writerow([
                "a_id",
                "a_uuid",
                "date",
                "m_id",
                "public",
                "lang",
                "n_tokens",
                "text",
                "label",
                "label_info",
                "m_social",
                "dup",
            ])

            written = 0
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                article = json.loads(line)

                if langs and article.get("lang", "") not in langs:
                    continue
                if article.get("public", 0) != 1:
                    continue

                labels, labels_info = _collect_labels(article, labels_map)
                if not labels:
                    continue

                article_id = article.get("id", "")
                media_id = article.get("m_id", "")
                text = _join_text(article)
                writer.writerow([
                    article_id,
                    article_uuid_map.get(article_id, article_id),
                    article.get("date", ""),
                    media_id,
                    article.get("public", 0),
                    article.get("lang", ""),
                    len(text.split()),
                    text,
                    labels,
                    labels_info,
                    1 if media_id in __social_media else 0,
                    0,
                ])
                written += 1

        logger.info("Extracted %s records to %s", written, out_file)

    return output_files, labels_map


def prep_corpus_merge(
    data_args: DataArguments,
    target_dir: Path,
    extracted_files: List[Path],
    labels_map: Dict[str, Dict[str, str]]
) -> None:
    _set_csv_field_size_limit()

    rows: List[Dict[str, Any]] = []
    label_counts: Counter = Counter()

    for file in extracted_files:
        if not file.exists():
            continue
        with file.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                labels = row.get("label", "[]")
                try:
                    labels_list = ast.literal_eval(labels)
                except (SyntaxError, ValueError):
                    labels_list = []
                if not isinstance(labels_list, list):
                    labels_list = []

                row["label"] = labels_list
                rows.append(row)
                label_counts.update(labels_list)

    valid_labels = {label_id for label_id, count in label_counts.items() if count >= 2}
    filtered_rows: List[Dict[str, Any]] = []
    final_counts: Counter = Counter()

    for row in rows:
        filtered_labels = [label_id for label_id in row["label"] if label_id in valid_labels]
        if not filtered_labels:
            continue
        row["label"] = filtered_labels
        try:
            label_info = ast.literal_eval(row.get("label_info", "[]"))
        except (SyntaxError, ValueError):
            label_info = []
        if not isinstance(label_info, list):
            label_info = []
        row["label_info"] = [item for item in label_info if item.get("id", "") in valid_labels]

        final_counts.update(filtered_labels)
        filtered_rows.append(row)

    data_file = target_dir / f"{data_args.dataset_name}_{data_args.lang}.csv"
    with data_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "a_id",
            "a_uuid",
            "date",
            "m_id",
            "public",
            "lang",
            "n_tokens",
            "text",
            "label",
            "label_info",
            "m_social",
            "dup",
        ])
        for row in filtered_rows:
            writer.writerow([
                row.get("a_id", ""),
                row.get("a_uuid", ""),
                row.get("date", ""),
                row.get("m_id", ""),
                row.get("public", 0),
                row.get("lang", ""),
                row.get("n_tokens", 0),
                row.get("text", ""),
                row.get("label", []),
                row.get("label_info", []),
                row.get("m_social", 0),
                row.get("dup", 0),
            ])

    labels_file = target_dir / f"{data_args.dataset_name}_{data_args.lang}_labels.csv"
    with labels_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "parent_id", "monitoring_country", "monitoring_industry", "count"])
        for label_id, count in sorted(final_counts.items(), key=lambda item: (-item[1], item[0])):
            label_meta = labels_map.get(label_id, {})
            writer.writerow([
                label_id,
                label_meta.get("name", ""),
                label_meta.get("parent_id", ""),
                label_meta.get("monitoring_country", ""),
                label_meta.get("monitoring_industry", ""),
                count,
            ])

    logger.info("Merged %s records into %s", len(filtered_rows), data_file)
    logger.info("Wrote %s labels into %s", len(final_counts), labels_file)


def main(data_args: DataArguments) -> None:
    source_dir = paths["base"]["data"] / "download" / "newsmon"
    if not source_dir.exists():
        logger.error("Source Newsmon directory not found: %s.", source_dir)
        return

    target_dir = paths["prepare"]["data"] / "newsmon"
    target_dir.mkdir(parents=True, exist_ok=True)

    extracted_files, labels_map = prep_corpus_extract(data_args, source_dir, target_dir)
    prep_corpus_merge(data_args, target_dir, extracted_files, labels_map)
    for file in extracted_files:
        if file.exists():
            file.unlink()
