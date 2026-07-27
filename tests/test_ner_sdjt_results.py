import pytest

from src.data.analyze._ner_sdjt.results import (
    _exact_sign_test_pvalue,
    _exact_wilcoxon_pvalue,
    compute_rq1,
    compute_rq5,
)


def test_exact_sign_test_matches_expected_two_sided_probability():
    deltas = [1.0] * 7 + [-1.0]

    assert _exact_sign_test_pvalue(deltas) == pytest.approx(0.0703125)


def test_exact_wilcoxon_matches_expected_small_sample_probability():
    deltas = [0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -0.01]

    assert _exact_wilcoxon_pvalue(deltas) == pytest.approx(0.015625)


def test_compute_rq5_compares_full_multilingual_variants():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame.from_records([
        {
            "run_name": "full-multi8",
            "pool_name": "full-multi8",
            "budget_pct": 100,
            "language": "bg",
            "p": 0.80,
            "r": 0.81,
            "f1": 0.82,
            "acc": 0.83,
        },
        {
            "run_name": "full-multi12",
            "pool_name": "full-multi12",
            "budget_pct": 100,
            "language": "bg",
            "p": 0.84,
            "r": 0.85,
            "f1": 0.86,
            "acc": 0.87,
        },
        {
            "run_name": "full-multi12-capaux",
            "pool_name": "full-multi12-capaux",
            "budget_pct": 100,
            "language": "bg",
            "p": 0.83,
            "r": 0.84,
            "f1": 0.85,
            "acc": 0.86,
        },
        {
            "run_name": "full-multi8",
            "pool_name": "full-multi8",
            "budget_pct": 100,
            "language": "sl",
            "p": 0.70,
            "r": 0.71,
            "f1": 0.72,
            "acc": 0.73,
        },
        {
            "run_name": "full-multi12",
            "pool_name": "full-multi12",
            "budget_pct": 100,
            "language": "sl",
            "p": 0.71,
            "r": 0.72,
            "f1": 0.73,
            "acc": 0.74,
        },
        {
            "run_name": "full-multi12-capaux",
            "pool_name": "full-multi12-capaux",
            "budget_pct": 100,
            "language": "sl",
            "p": 0.75,
            "r": 0.76,
            "f1": 0.77,
            "acc": 0.78,
        },
    ])

    result = compute_rq5(df)

    assert list(result["language"]) == ["bg", "sl"]

    bg = result[result["language"] == "bg"].iloc[0]
    assert bool(bg["rq5_complete"]) is True
    assert bg["best_model"] == "full_multi12"
    assert bg["delta_full_multi12_minus_full_multi8"] == pytest.approx(0.04)
    assert bg["delta_full_multi12_capaux_minus_full_multi12"] == pytest.approx(-0.01)

    sl = result[result["language"] == "sl"].iloc[0]
    assert bool(sl["rq5_complete"]) is True
    assert sl["best_model"] == "full_multi12_capaux"
    assert sl["delta_full_multi12_minus_full_multi8"] == pytest.approx(0.01)
    assert sl["delta_full_multi12_capaux_minus_full_multi8"] == pytest.approx(0.05)


def test_compute_rq1_preserves_per_run_f1_std_columns():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame.from_records([
        {
            "run_name": "mono-bg",
            "pool_name": "mono",
            "budget_pct": 100,
            "language": "bg",
            "p": 0.80,
            "p_std": 0.01,
            "r": 0.81,
            "r_std": 0.02,
            "f1": 0.82,
            "f1_std": 0.03,
            "acc": 0.83,
            "acc_std": 0.04,
        },
        {
            "run_name": "multi8",
            "pool_name": "multi8",
            "budget_pct": 100,
            "language": "bg",
            "p": 0.78,
            "p_std": 0.05,
            "r": 0.79,
            "r_std": 0.06,
            "f1": 0.80,
            "f1_std": 0.07,
            "acc": 0.81,
            "acc_std": 0.08,
        },
    ])

    result = compute_rq1(df)

    row = result.iloc[0]
    assert row["mono_f1_std"] == pytest.approx(0.03)
    assert row["multi8_f1_std"] == pytest.approx(0.07)
    assert row["delta_f1_multi8_minus_mono"] == pytest.approx(-0.02)
