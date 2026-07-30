from pathlib import Path

import pytest

from src.app.args.runtime import PathSet, Paths, ResultPathSet
from src.data.resample.ner_sdjt import (
    append_seed_suffix,
    parse_seed_suffix,
)
from src.eval.token.ner_sdjt import build_result_rows
from src.train.token.ner_sdjt import init_dirs


def build_paths(root: Path) -> Paths:
    result_root = root / "result"
    return Paths(
        curr_context="ner-sdjt",
        curr_script="train",
        curr_task="token",
        base=PathSet(
            root=root,
            tmp=root / "tmp",
            src=root / "src",
            log=root / "log",
            result=ResultPathSet(
                root=result_root,
                data=result_root / "data",
                test=result_root / "test",
                train=result_root / "train",
                eval=result_root / "eval",
            ),
        ),
        task=result_root / "train" / "token",
        context=result_root / "train" / "token" / "ner-sdjt",
    )


def test_seed_suffix_round_trip() -> None:
    base = Path("/tmp/result/data/split/ner-sdjt")

    seeded = append_seed_suffix(base, 2611)

    assert seeded == Path("/tmp/result/data/split/ner-sdjt.s2611")
    assert parse_seed_suffix(seeded) == 2611
    assert append_seed_suffix(base, None) == base


def test_parse_seed_suffix_rejects_unseeded_path() -> None:
    with pytest.raises(ValueError, match="numeric .sSEED suffix"):
        parse_seed_suffix(Path("ner-sdjt.multi8.mm-bert"))


def test_training_data_root_uses_training_seed(tmp_path: Path) -> None:
    paths = build_paths(tmp_path)
    expected = (
        tmp_path
        / "result"
        / "data"
        / "split"
        / "ner-sdjt.s4760"
        / "multi8"
    )
    expected.mkdir(parents=True)

    data_root, cache_root = init_dirs(paths, "multi8", 4760)

    assert data_root == expected
    assert cache_root == tmp_path / "tmp" / "cache"


def test_aggregate_rows_record_all_matching_model_seeds() -> None:
    train_dirs = [
        Path("ner-sdjt.multi8.mm-bert.b16.lr2e-05.cd05.wr06.wd01.s2611"),
        Path("ner-sdjt.multi8.mm-bert.b16.lr2e-05.cd05.wr06.wd01.s4760"),
    ]
    model_metrics = [
        {"bg": {"p": 0.8, "r": 0.7, "f1": 0.75, "acc": 0.9}},
        {"bg": {"p": 0.9, "r": 0.8, "f1": 0.85, "acc": 0.92}},
    ]

    rows = build_result_rows("multi8", train_dirs, model_metrics)

    assert len(rows) == 1
    assert rows[0]["models_evaluated"] == 2
    assert rows[0]["seeds"] == "2611;4760"
    assert rows[0]["f1"] == pytest.approx(0.8)
    assert rows[0]["f1_std"] == pytest.approx(0.05)
