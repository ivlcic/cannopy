import csv
from collections import Counter, defaultdict
from logging import Logger
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Tuple

from ...app.args.data import DataArguments

Sentence = Tuple[List[str], List[str]]

logger: Logger
paths: Dict[str, Any]


def _load_sentences(source_dir: Path, split_suffix: str | None = None) -> Dict[str, List[Sentence]]:
    aggregated: DefaultDict[str, List[Sentence]] = defaultdict(list)
    for csv_file in sorted(source_dir.glob("ner-*.csv")):
        stem = csv_file.stem
        if stem.startswith("ner_stats"):
            continue
        if split_suffix:
            if not stem.endswith(split_suffix):
                continue
            lang = stem[: -len(split_suffix)]
            lang = lang.replace("ner-", "")
        else:
            if "." in stem:
                # skip split files when loading base data
                continue
            lang = stem.replace("ner-", "")
        with csv_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            for row in reader:
                if len(row) < 2:
                    continue
                tokens = row[0].split(" ")
                labels = row[1].split(" ")
                aggregated[lang].append((tokens, labels))
    return aggregated


def _collect_tags(aggregated: Dict[str, List[Sentence]]) -> List[str]:
    tags = set()
    for sentences in aggregated.values():
        for _, labels in sentences:
            for label in labels:
                if label != "O":
                    tags.add(label)
    return sorted(tags)


def _compute_stats(aggregated: Dict[str, List[Sentence]], tags: List[str]) -> Dict[str, Any]:
    label_stats: Dict[str, Counter] = {}
    sentence_stats: Counter = Counter()
    token_stats: Counter = Counter()

    for lang, sentences in aggregated.items():
        label_counter: Counter = Counter()
        sent_count = len(sentences)
        tok_count = 0
        for tokens, labels in sentences:
            tok_count += len(tokens)
            for label in labels:
                if label != "O":
                    label_counter[label] += 1
        label_stats[lang] = label_counter
        sentence_stats[lang] = sent_count
        token_stats[lang] = tok_count

    return {"labels": label_stats, "sentences": sentence_stats, "tokens": token_stats, "tags": tags}


def _write_stats(output_dir: Path, stats: Dict[str, Any], file_suffix: str = "") -> None:
    stats_path = output_dir / f"ner-stats{file_suffix}.csv"
    with stats_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["language", "sentences", "tokens", *stats["tags"]])
        for lang in sorted(stats["labels"].keys()):
            counter = stats["labels"][lang]
            row = [lang, stats["sentences"][lang], stats["tokens"][lang]] + \
                  [counter.get(tag, 0) for tag in stats["tags"]]
            writer.writerow(row)


def _format_stats_table(stats: Dict[str, Any]) -> str:
    header = ["lang", "sent", "tok", *stats["tags"]]
    rows: List[List[str]] = []
    widths = [len(col) for col in header]

    for lang, counter in sorted(stats["labels"].items()):
        row = [
            lang,
            str(stats["sentences"][lang]),
            str(stats["tokens"][lang]),
            *[str(counter.get(tag, 0)) for tag in stats["tags"]],
        ]
        rows.append(row)
        widths = [max(w, len(val)) for w, val in zip(widths, row)]

    def _fmt_row(row: List[str]) -> str:
        parts = []
        for idx, val in enumerate(row):
            align = "<" if idx == 0 else ">"
            parts.append(f"{val:{align}{widths[idx]}}")
        return "  ".join(parts)

    return "\n".join([_fmt_row(header)] + [_fmt_row(r) for r in rows])


def _format_split_stats_table(split_stats: Dict[str, Dict[str, Any]], tags: List[str]) -> str:
    header = ["lang", "split", "sent", "tok", *tags]
    rows: List[List[str]] = []
    widths = [len(col) for col in header]

    for split_name in sorted(split_stats.keys()):
        stats = split_stats[split_name]
        for lang, counter in sorted(stats["labels"].items()):
            row = [
                lang,
                split_name,
                str(stats["sentences"][lang]),
                str(stats["tokens"][lang]),
                *[str(counter.get(tag, 0)) for tag in tags],
            ]
            rows.append(row)
            widths = [max(w, len(val)) for w, val in zip(widths, row)]

    def _fmt_row(row: List[str]) -> str:
        parts = []
        for idx, val in enumerate(row):
            align = "<" if idx < 2 else ">"
            parts.append(f"{val:{align}{widths[idx]}}")
        return "  ".join(parts)

    return "\n".join([_fmt_row(header)] + [_fmt_row(r) for r in rows])


def main(data_args: DataArguments) -> None:
    logger.info("Analyzing NER datasets")

    output_dir = paths["analyze"]["data"]
    output_dir.mkdir(parents=True, exist_ok=True)

    base_dir = paths["base"]["data"] / "prepare"
    split_dir = paths["base"]["data"] / "split"

    aggregated = _load_sentences(base_dir)
    if not aggregated:
        logger.warning("No prepared NER data found in %s", base_dir)
        return

    split_suffixes = set()
    if split_dir.exists():
        for csv_file in split_dir.glob("ner-*.csv"):
            stem = csv_file.stem
            if stem.startswith("ner-stats"):
                continue
            parts = stem.split(".")
            if len(parts) >= 2:
                split_suffixes.add("." + parts[-1])

    split_aggregated: Dict[str, Dict[str, List[Sentence]]] = {}
    tags_set = set(_collect_tags(aggregated))
    for suffix in sorted(split_suffixes):
        agg = _load_sentences(split_dir, suffix)
        if agg:
            split_aggregated[suffix] = agg
            tags_set.update(_collect_tags(agg))

    tags = sorted(tags_set)

    base_stats = _compute_stats(aggregated, tags)
    _write_stats(output_dir, base_stats)
    logger.info("NER stats (prepared, matching ner_stats.csv columns):\n%s", _format_stats_table(base_stats))

    if split_aggregated:
        split_stats: Dict[str, Dict[str, Any]] = {}
        for suffix, agg in split_aggregated.items():
            stats = _compute_stats(agg, tags)
            split_name = suffix.lstrip(".")
            split_stats[split_name] = stats
            _write_stats(output_dir, stats, file_suffix=f".{split_name}")
        logger.info(
            "NER split stats (language/split, matching ner_stats.csv columns):\n%s",
            _format_split_stats_table(split_stats, tags)
        )
