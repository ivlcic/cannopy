import csv
import logging
from pathlib import Path

from src.app.args.data import DataArguments
from src.app.ner import NerSample
from src.data.split import ner as ner_split


class FakePaths:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.context = root / "result" / "data" / "split" / "ner"

    def get_ctx_path(self, script: str) -> Path:
        return self.root / "result" / "data" / script / "ner"


def _sample(sentence: str, split_name: str, labels: list[str]) -> NerSample:
    tokens = sentence.split(" ")
    return NerSample(
        tokens=tokens,
        labels=labels,
        corpus_name="test-corpus",
        doc_id=f"{split_name}-doc",
        sent_id="1",
    )


def _read_samples(path: Path) -> list[NerSample]:
    with path.open(encoding="utf-8", newline="") as input_file:
        return [NerSample.from_csv_row(row) for row in csv.DictReader(input_file)]


def test_main_deduplicates_after_splitting_and_preserves_raw_labels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    test_duplicate = _sample("Isto besedilo", "test", ["O", "B-MISC"])
    eval_duplicate = _sample("isto besedilo", "eval", ["O", "O"])
    train_duplicate = _sample("ISTO BESEDILO", "train", ["O", "B-MISC"])
    unique_train = _sample("Drugo besedilo", "train-unique", ["O", "B-MISC"])
    split_data = {
        "train": {"sl": [train_duplicate, unique_train]},
        "eval": {"sl": [eval_duplicate]},
        "test": {"sl": [test_duplicate]},
    }

    test_paths = FakePaths(tmp_path)
    prepare_dir = test_paths.get_ctx_path("prepare")
    prepare_dir.mkdir(parents=True)
    with (prepare_dir / "ner-sl.csv").open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=NerSample.NER_CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(unique_train.to_csv_row())

    monkeypatch.setattr(ner_split, "paths", test_paths, raising=False)
    monkeypatch.setattr(ner_split, "logger", logging.getLogger(__name__), raising=False)
    monkeypatch.setattr(
        ner_split,
        "_split_language_data",
        lambda aggregated, train_ratio, dev_ratio, test_ratio, seed: split_data,
    )
    data_args = DataArguments()
    data_args.split.dedup = True

    ner_split.main(data_args)

    split_dir = test_paths.context
    assert _read_samples(split_dir / "ner-sl.test.csv") == [test_duplicate]
    assert _read_samples(split_dir / "ner-sl.eval.csv") == []
    assert _read_samples(split_dir / "ner-sl.train.csv") == [unique_train]
    assert test_duplicate.labels == ["O", "B-MISC"]

    analyze_dir = test_paths.get_ctx_path("analyze")
    with (analyze_dir / "ner-duplicates.csv").open(encoding="utf-8", newline="") as input_file:
        duplicate_rows = list(csv.DictReader(input_file))
    assert len(duplicate_rows) == 2
    assert {row["kept_split"] for row in duplicate_rows} == {"test"}
    assert {row["labels_match"] for row in duplicate_rows} == {"True"}
