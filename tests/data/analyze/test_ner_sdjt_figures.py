import csv
from pathlib import Path

from src.data.analyze._ner_sdjt.figures import _read_harmonized_training_stats


def _write_stats(path: Path, rows: list[dict[str, object]]) -> None:
    columns = [
        "run_name",
        "split",
        "language",
        "tokens",
        "B-LOC",
        "B-ORG",
        "B-PER",
        "I-LOC",
        "I-ORG",
        "I-PER",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_figure_stats_use_full_multi12_harmonized_training_rows(tmp_path: Path) -> None:
    stats_file = tmp_path / "ner-stats.csv"
    _write_stats(stats_file, [
        {
            "run_name": "full-multi12",
            "split": "train",
            "language": "bg",
            "tokens": 100,
            "B-LOC": 2,
            "B-ORG": 3,
            "B-PER": 4,
            "I-LOC": 1,
            "I-ORG": 2,
            "I-PER": 3,
        },
        {
            "run_name": "full-multi12",
            "split": "train",
            "language": "bs",
            "tokens": 80,
            "B-LOC": 5,
            "B-ORG": 6,
            "B-PER": 7,
            "I-LOC": 1,
            "I-ORG": 1,
            "I-PER": 1,
        },
        {
            "run_name": "full-multi12",
            "split": "eval",
            "language": "bg",
            "tokens": 999,
        },
        {
            "run_name": "multi8",
            "split": "train",
            "language": "bg",
            "tokens": 888,
        },
        {
            "run_name": "full-multi12",
            "split": "train",
            "language": "hr-wikiann",
            "tokens": 777,
        },
    ])

    rows = _read_harmonized_training_stats(stats_file)

    assert [row["language"] for row in rows] == ["bg", "bs"]
    assert rows[0]["tokens"] == 100
    assert rows[0]["entity_counts"] == {"PER": 7, "ORG": 5, "LOC": 3}
    assert rows[0]["entity_total"] == 15
    assert rows[0]["is_aux"] is False
    assert rows[1]["is_aux"] is True
