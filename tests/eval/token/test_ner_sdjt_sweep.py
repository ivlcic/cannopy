import csv
import json

import pytest

from src.eval.token.ner_sdjt import (
    build_sweep_rows,
    collect_sweep_runs,
    write_sweep_csv,
)


def _write_sweep_run(train_root, learning_rate, dropout_tag, dropout, seed, metric, epoch):
    run_name = (
        f"ner-sdjt.multi8.mm-bert.b16.lr{learning_rate}."
        f"cd{dropout_tag}.wr06.wd01.s{seed}"
    )
    run_dir = train_root / run_name
    run_dir.mkdir()
    step = int(epoch * 100)
    (run_dir / "config.json").write_text(
        json.dumps({"classifier_dropout": dropout}),
        encoding="utf-8",
    )
    (run_dir / "trainer_state.json").write_text(
        json.dumps({
            "best_global_step": step,
            "best_metric": metric,
            "log_history": [
                {
                    "epoch": epoch,
                    "eval_macro_f1": metric,
                    "step": step,
                }
            ],
        }),
        encoding="utf-8",
    )


def test_build_sweep_rows_ranks_complete_configurations_only(tmp_path):
    _write_sweep_run(tmp_path, "1e-05", "05", 0.05, 2611, 0.90, 7)
    _write_sweep_run(tmp_path, "1e-05", "05", 0.05, 4760, 0.92, 8)
    _write_sweep_run(tmp_path, "2e-05", "01", 0.10, 2611, 0.95, 5)

    runs = collect_sweep_runs(tmp_path, "ner-sdjt", "mm-bert")
    rows = build_sweep_rows(runs)

    assert len(rows) == 2
    assert rows[0]["rank"] == 1
    assert rows[0]["selected"] is True
    assert rows[0]["learning_rate"] == pytest.approx(1e-5)
    assert rows[0]["validation_macro_f1_mean"] == pytest.approx(0.91)
    assert rows[0]["validation_macro_f1_std"] == pytest.approx(0.01)
    assert rows[0]["seeds"] == "2611;4760"
    assert rows[1]["rank"] == ""
    assert rows[1]["complete_seed_set"] is False


def test_write_sweep_csv_preserves_per_seed_audit_values(tmp_path):
    _write_sweep_run(tmp_path, "1e-05", "05", 0.05, 2611, 0.90, 7)
    rows = build_sweep_rows(collect_sweep_runs(tmp_path, "ner-sdjt", "mm-bert"))
    output = tmp_path / "ner-sdjt.sweep.mm-bert.csv"

    write_sweep_csv(output, rows)

    with output.open(encoding="utf-8", newline="") as fp:
        written = list(csv.DictReader(fp))
    assert written[0]["validation_macro_f1_by_seed"] == "2611:0.9"
    assert written[0]["best_epochs_by_seed"] == "2611:7"
