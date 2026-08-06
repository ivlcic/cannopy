import csv
from pathlib import Path

import pytest

from src.app.args.runtime import PathSet, Paths, ResultPathSet
from src.data.resample.ner_sdjt import (
    append_seed_suffix,
    parse_seed_suffix,
)
from src.eval.token.ner_sdjt import (
    build_result_rows,
    evaluate_language,
    write_results_csv,
)
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
        {
            "bg": {
                "p": 0.8,
                "r": 0.7,
                "f1": 0.75,
                "acc": 0.9,
                "per_p": 0.7,
                "per_r": 0.6,
                "per_f1": 0.65,
                "loc_p": 0.8,
                "loc_r": 0.7,
                "loc_f1": 0.75,
                "org_p": 0.9,
                "org_r": 0.8,
                "org_f1": 0.85,
            }
        },
        {
            "bg": {
                "p": 0.9,
                "r": 0.8,
                "f1": 0.85,
                "acc": 0.92,
                "per_p": 0.8,
                "per_r": 0.7,
                "per_f1": 0.75,
                "loc_p": 0.9,
                "loc_r": 0.8,
                "loc_f1": 0.85,
                "org_p": 1.0,
                "org_r": 0.9,
                "org_f1": 0.95,
            }
        },
    ]

    rows = build_result_rows("multi8", train_dirs, model_metrics)

    assert len(rows) == 1
    assert rows[0]["models_evaluated"] == 2
    assert rows[0]["seeds"] == "2611;4760"
    assert rows[0]["f1"] == pytest.approx(0.8)
    assert rows[0]["f1_std"] == pytest.approx(0.05)
    assert rows[0]["per_f1"] == pytest.approx(0.7)
    assert rows[0]["per_f1_std"] == pytest.approx(0.05)


def test_evaluate_language_retains_entity_metrics() -> None:
    class FakeTrainer:
        def evaluate(self, eval_dataset, metric_key_prefix):
            assert eval_dataset == "dataset"
            assert metric_key_prefix == "test_sl"
            return {
                "test_sl_p": 0.8,
                "test_sl_r": 0.7,
                "test_sl_f1": 0.75,
                "test_sl_acc": 0.9,
                "test_sl_label.PER.p": 0.91,
                "test_sl_label.PER.r": 0.92,
                "test_sl_label.PER.f1": 0.93,
                "test_sl_label.LOC.p": 0.81,
                "test_sl_label.LOC.r": 0.82,
                "test_sl_label.LOC.f1": 0.83,
                "test_sl_label.ORG.p": 0.71,
                "test_sl_label.ORG.r": 0.72,
                "test_sl_label.ORG.f1": 0.73,
            }

    metrics = evaluate_language(FakeTrainer(), "dataset", "sl")

    assert metrics["per_f1"] == pytest.approx(0.93)
    assert metrics["loc_f1"] == pytest.approx(0.83)
    assert metrics["org_f1"] == pytest.approx(0.73)


def test_results_csv_includes_entity_metric_means_and_standard_deviations(
    tmp_path: Path,
) -> None:
    output = tmp_path / "results.csv"
    row = {
        "run_name": "mono-sl",
        "pool_name": "mono",
        "budget_pct": 100,
        "models_evaluated": 1,
        "seeds": "2611",
        "language": "sl",
        "num_models": 1,
        "model_prefix": "model",
    }
    for metric in (
        "p",
        "r",
        "f1",
        "acc",
        "per_p",
        "per_r",
        "per_f1",
        "loc_p",
        "loc_r",
        "loc_f1",
        "org_p",
        "org_r",
        "org_f1",
    ):
        row[metric] = 0.8
        row[f"{metric}_std"] = 0.01

    write_results_csv(output, [row])

    with output.open(encoding="utf-8", newline="") as fp:
        written = next(csv.DictReader(fp))
    assert written["per_f1"] == "0.8"
    assert written["per_f1_std"] == "0.01"
    assert written["loc_f1"] == "0.8"
    assert written["org_f1"] == "0.8"
