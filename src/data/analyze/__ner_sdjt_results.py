from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from ..resample.ner_sdjt import CURVE_BUDGETS, CURVE_LANGUAGES, MAIN_LANGUAGES

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "run_name",
    "pool_name",
    "budget_pct",
    "language",
    "p",
    "r",
    "f1",
    "acc",
}
NUMERIC_COLUMNS = (
    "budget_pct",
    "models_evaluated",
    "num_models",
    "p",
    "p_std",
    "r",
    "r_std",
    "f1",
    "f1_std",
    "acc",
    "acc_std",
)


def _import_pandas():
    import pandas as pd

    return pd


def read_results(csv_path: Path):
    pd = _import_pandas()
    LOGGER.info("Reading CSV: %s", csv_path)
    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="raise")

    for column in ("pool_name", "run_name", "language", "model_prefix"):
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()

    LOGGER.info("Loaded %d rows", len(df))
    return df


def validate_expected_rows(df) -> None:
    LOGGER.info("Validating expected rows")

    for pool_name in ("mono", "multi8", "multi12"):
        for language in MAIN_LANGUAGES:
            mask = (
                (df["pool_name"] == pool_name)
                & (df["budget_pct"] == 100)
                & (df["language"] == language)
            )
            count = int(mask.sum())
            if count == 0:
                LOGGER.warning(
                    "Missing row for pool=%s, language=%s, budget=100",
                    pool_name,
                    language,
                )
            elif count > 1:
                LOGGER.warning(
                    "Duplicate rows for pool=%s, language=%s, budget=100: %d",
                    pool_name,
                    language,
                    count,
                )

    for pool_name in ("multi8-full", "pretrain-multi7-full"):
        for language in MAIN_LANGUAGES:
            mask = (
                (df["pool_name"] == pool_name)
                & (df["budget_pct"] == 100)
                & (df["language"] == language)
            )
            count = int(mask.sum())
            if count == 0:
                LOGGER.warning(
                    "Missing row for pool=%s, language=%s, budget=100",
                    pool_name,
                    language,
                )
            elif count > 1:
                LOGGER.warning(
                    "Duplicate rows for pool=%s, language=%s, budget=100: %d",
                    pool_name,
                    language,
                    count,
                )

    for language in MAIN_LANGUAGES:
        mask = (
            (df["pool_name"] == "full-multi8")
            & (df["budget_pct"] == 100)
            & (df["language"] == language)
        )
        count = int(mask.sum())
        if count == 0:
            LOGGER.warning(
                "Missing row for pool=%s, language=%s, budget=100",
                "full-multi8",
                language,
            )
        elif count > 1:
            LOGGER.warning(
                "Duplicate rows for pool=%s, language=%s, budget=100: %d",
                "full-multi8",
                language,
                count,
            )

    for pool_name in ("full-multi12", "full-multi12-capaux"):
        for language in MAIN_LANGUAGES:
            mask = (
                (df["pool_name"] == pool_name)
                & (df["budget_pct"] == 100)
                & (df["language"] == language)
            )
            count = int(mask.sum())
            if count == 0:
                LOGGER.warning(
                    "Missing row for pool=%s, language=%s, budget=100",
                    pool_name,
                    language,
                )
            elif count > 1:
                LOGGER.warning(
                    "Duplicate rows for pool=%s, language=%s, budget=100: %d",
                    pool_name,
                    language,
                    count,
                )

    for language in sorted(CURVE_LANGUAGES):
        for pool_name in ("mono", "multi8", "multi12"):
            for budget_pct in sorted(CURVE_BUDGETS):
                mask = (
                    (df["pool_name"] == pool_name)
                    & (df["budget_pct"] == budget_pct)
                    & (df["language"] == language)
                )
                count = int(mask.sum())
                if count == 0:
                    LOGGER.warning(
                        "Missing curve row for pool=%s, language=%s, budget=%s",
                        pool_name,
                        language,
                        budget_pct,
                    )
                elif count > 1:
                    LOGGER.warning(
                        "Duplicate curve rows for pool=%s, language=%s, budget=%s: %d",
                        pool_name,
                        language,
                        budget_pct,
                        count,
                    )


def safe_mean(series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.mean())


def select_main_rows(df, pool_name: str):
    out = df[
        (df["pool_name"] == pool_name)
        & (df["budget_pct"] == 100)
        & (df["language"].isin(MAIN_LANGUAGES))
    ].copy()
    out = out.sort_values("language").reset_index(drop=True)
    LOGGER.info("Selected %d main rows for pool=%s", len(out), pool_name)
    return out


def compute_rq1(df):
    LOGGER.info("Computing RQ1: Mono-L vs Multi-8")
    mono = select_main_rows(df, "mono")[["language", "p", "r", "f1", "acc"]].rename(
        columns={
            "p": "mono_p",
            "r": "mono_r",
            "f1": "mono_f1",
            "acc": "mono_acc",
        }
    )
    multi8 = select_main_rows(df, "multi8")[["language", "p", "r", "f1", "acc"]].rename(
        columns={
            "p": "multi8_p",
            "r": "multi8_r",
            "f1": "multi8_f1",
            "acc": "multi8_acc",
        }
    )

    merged = mono.merge(multi8, on="language", how="inner")
    merged["delta_f1_multi8_minus_mono"] = merged["multi8_f1"] - merged["mono_f1"]
    merged["delta_p_multi8_minus_mono"] = merged["multi8_p"] - merged["mono_p"]
    merged["delta_r_multi8_minus_mono"] = merged["multi8_r"] - merged["mono_r"]
    merged["winner"] = merged["delta_f1_multi8_minus_mono"].apply(
        lambda value: "multi8" if value > 0 else "mono"
    )

    mono_macro = safe_mean(merged["mono_f1"])
    multi8_macro = safe_mean(merged["multi8_f1"])
    LOGGER.info("RQ1 mono macro F1   = %.4f", mono_macro)
    LOGGER.info("RQ1 multi8 macro F1 = %.4f", multi8_macro)
    LOGGER.info("RQ1 delta           = %.4f", multi8_macro - mono_macro)

    best = merged.sort_values("delta_f1_multi8_minus_mono", ascending=False)
    if not best.empty:
        LOGGER.info(
            "RQ1 biggest multilingual gain: %s (%.4f F1)",
            best.iloc[0]["language"],
            best.iloc[0]["delta_f1_multi8_minus_mono"],
        )
        LOGGER.info(
            "RQ1 biggest multilingual loss: %s (%.4f F1)",
            best.iloc[-1]["language"],
            best.iloc[-1]["delta_f1_multi8_minus_mono"],
        )

    return merged.sort_values("language").reset_index(drop=True)


def compute_rq2(df):
    LOGGER.info("Computing RQ2: Multi-8 vs Multi-12")
    multi8 = select_main_rows(df, "multi8")[["language", "p", "r", "f1", "acc"]].rename(
        columns={
            "p": "multi8_p",
            "r": "multi8_r",
            "f1": "multi8_f1",
            "acc": "multi8_acc",
        }
    )
    multi12 = select_main_rows(df, "multi12")[["language", "p", "r", "f1", "acc"]].rename(
        columns={
            "p": "multi12_p",
            "r": "multi12_r",
            "f1": "multi12_f1",
            "acc": "multi12_acc",
        }
    )

    merged = multi8.merge(multi12, on="language", how="inner")
    merged["delta_f1_multi12_minus_multi8"] = merged["multi12_f1"] - merged["multi8_f1"]
    merged["delta_p_multi12_minus_multi8"] = merged["multi12_p"] - merged["multi8_p"]
    merged["delta_r_multi12_minus_multi8"] = merged["multi12_r"] - merged["multi8_r"]
    merged["winner"] = merged["delta_f1_multi12_minus_multi8"].apply(
        lambda value: "multi12" if value > 0 else "multi8"
    )

    multi8_macro = safe_mean(merged["multi8_f1"])
    multi12_macro = safe_mean(merged["multi12_f1"])
    LOGGER.info("RQ2 multi8 macro F1  = %.4f", multi8_macro)
    LOGGER.info("RQ2 multi12 macro F1 = %.4f", multi12_macro)
    LOGGER.info("RQ2 delta            = %.4f", multi12_macro - multi8_macro)

    best = merged.sort_values("delta_f1_multi12_minus_multi8", ascending=False)
    if not best.empty:
        LOGGER.info(
            "RQ2 biggest gain from extra languages: %s (%.4f F1)",
            best.iloc[0]["language"],
            best.iloc[0]["delta_f1_multi12_minus_multi8"],
        )
        LOGGER.info(
            "RQ2 biggest loss from extra languages: %s (%.4f F1)",
            best.iloc[-1]["language"],
            best.iloc[-1]["delta_f1_multi12_minus_multi8"],
        )

    return merged.sort_values("language").reset_index(drop=True)


def extract_curve_row(df, pool_name: str, language: str, budget_pct: int):
    rows = df[
        (df["pool_name"] == pool_name)
        & (df["language"] == language)
        & (df["budget_pct"] == budget_pct)
    ]
    if rows.empty:
        return None
    if len(rows) > 1:
        LOGGER.warning(
            "Multiple rows found for pool=%s, language=%s, budget=%s; using first",
            pool_name,
            language,
            budget_pct,
        )
    return rows.iloc[0]


def compute_rq3(df):
    pd = _import_pandas()
    LOGGER.info("Computing RQ3: resource curves for sr and sl")
    records: list[dict[str, Any]] = []
    columns = [
        "language",
        "budget_pct",
        "mono_f1",
        "multi8_f1",
        "multi12_f1",
        "mono_p",
        "multi8_p",
        "multi12_p",
        "mono_r",
        "multi8_r",
        "multi12_r",
        "delta_multi8_minus_mono",
        "delta_multi12_minus_mono",
        "delta_multi12_minus_multi8",
        "best_model",
    ]

    for language in sorted(CURVE_LANGUAGES):
        for budget_pct in sorted(CURVE_BUDGETS):
            mono = extract_curve_row(df, "mono", language, budget_pct)
            multi8 = extract_curve_row(df, "multi8", language, budget_pct)
            multi12 = extract_curve_row(df, "multi12", language, budget_pct)

            if mono is None or multi8 is None or multi12 is None:
                LOGGER.warning(
                    "Skipping incomplete RQ3 point for language=%s budget=%s",
                    language,
                    budget_pct,
                )
                continue

            record = {
                "language": language,
                "budget_pct": budget_pct,
                "mono_f1": float(mono["f1"]),
                "multi8_f1": float(multi8["f1"]),
                "multi12_f1": float(multi12["f1"]),
                "mono_p": float(mono["p"]),
                "multi8_p": float(multi8["p"]),
                "multi12_p": float(multi12["p"]),
                "mono_r": float(mono["r"]),
                "multi8_r": float(multi8["r"]),
                "multi12_r": float(multi12["r"]),
            }
            record["delta_multi8_minus_mono"] = record["multi8_f1"] - record["mono_f1"]
            record["delta_multi12_minus_mono"] = record["multi12_f1"] - record["mono_f1"]
            record["delta_multi12_minus_multi8"] = record["multi12_f1"] - record["multi8_f1"]
            record["best_model"] = max(
                (("mono", record["mono_f1"]), ("multi8", record["multi8_f1"]), ("multi12", record["multi12_f1"])),
                key=lambda item: item[1],
            )[0]
            records.append(record)

    out = pd.DataFrame.from_records(records, columns=columns)
    if out.empty:
        return out

    out = out.sort_values(["language", "budget_pct"]).reset_index(drop=True)
    for language in sorted(CURVE_LANGUAGES):
        sub = out[out["language"] == language].sort_values("budget_pct")
        if sub.empty:
            continue
        LOGGER.info("RQ3 %s curve:", language)
        for _, row in sub.iterrows():
            LOGGER.info(
                "  budget=%3d | mono=%.4f | multi8=%.4f | multi12=%.4f | best=%s",
                int(row["budget_pct"]),
                row["mono_f1"],
                row["multi8_f1"],
                row["multi12_f1"],
                row["best_model"],
            )
    return out


def _rename_metric_columns(frame, prefix: str):
    rename_map = {}
    for column in frame.columns:
        if column == "language":
            continue
        rename_map[column] = f"{prefix}_{column}"
    return frame.rename(columns=rename_map)


def _merge_language_frames(frames):
    merged = None
    for frame in frames:
        if merged is None:
            merged = frame.copy()
        else:
            merged = merged.merge(frame, on="language", how="outer")
    return merged


def _compute_best_model(record: dict[str, Any]) -> str:
    candidates = (
        ("mono", record["mono_f1"]),
        ("multi8", record["multi8_f1"]),
        ("multi8_full", record["multi8_full_f1"]),
        ("pretrain_multi7_full", record["pretrain_multi7_full_f1"]),
    )
    available = [(name, value) for name, value in candidates if value == value]
    if not available:
        return "missing"
    return max(available, key=lambda item: item[1])[0]


def compute_rq4(df):
    pd = _import_pandas()
    LOGGER.info("Computing RQ4: target-specific multilingual training and leave-one-out pretraining")
    mono = _rename_metric_columns(
        select_main_rows(df, "mono")[["language", "p", "r", "f1", "acc"]],
        "mono",
    )
    multi8 = _rename_metric_columns(
        select_main_rows(df, "multi8")[["language", "p", "r", "f1", "acc"]],
        "multi8",
    )
    multi8_full = _rename_metric_columns(
        select_main_rows(df, "multi8-full")[["language", "p", "r", "f1", "acc"]],
        "multi8_full",
    )
    pretrain_multi7_full = _rename_metric_columns(
        select_main_rows(df, "pretrain-multi7-full")[["language", "p", "r", "f1", "acc"]],
        "pretrain_multi7_full",
    )

    merged = _merge_language_frames((mono, multi8, multi8_full, pretrain_multi7_full))
    if merged is None or merged.empty:
        return pd.DataFrame.from_records([], columns=[
            "language",
            "has_mono",
            "has_multi8",
            "has_multi8_full",
            "has_pretrain_multi7_full",
            "rq4_complete",
            "mono_f1",
            "multi8_f1",
            "multi8_full_f1",
            "pretrain_multi7_full_f1",
            "delta_multi8_full_minus_multi8",
            "delta_multi8_full_minus_mono",
            "delta_pretrain_multi7_full_minus_mono",
            "delta_pretrain_multi7_full_minus_multi8_full",
            "delta_pretrain_multi7_full_minus_multi8",
            "best_model",
            "case_a_target_downsampling_hurt",
            "case_b_clean_multilingual_beats_mono",
            "case_c_pretraining_beats_mono",
            "case_d_pretraining_beats_multi8_full",
            "case_e_mono_remains_best",
        ])

    merged["delta_multi8_full_minus_multi8"] = merged["multi8_full_f1"] - merged["multi8_f1"]
    merged["delta_multi8_full_minus_mono"] = merged["multi8_full_f1"] - merged["mono_f1"]
    merged["delta_pretrain_multi7_full_minus_mono"] = (
        merged["pretrain_multi7_full_f1"] - merged["mono_f1"]
    )
    merged["delta_pretrain_multi7_full_minus_multi8_full"] = (
        merged["pretrain_multi7_full_f1"] - merged["multi8_full_f1"]
    )
    merged["delta_pretrain_multi7_full_minus_multi8"] = (
        merged["pretrain_multi7_full_f1"] - merged["multi8_f1"]
    )
    merged["has_mono"] = merged["mono_f1"].notna()
    merged["has_multi8"] = merged["multi8_f1"].notna()
    merged["has_multi8_full"] = merged["multi8_full_f1"].notna()
    merged["has_pretrain_multi7_full"] = merged["pretrain_multi7_full_f1"].notna()
    merged["rq4_complete"] = (
        merged["has_mono"]
        & merged["has_multi8"]
        & merged["has_multi8_full"]
        & merged["has_pretrain_multi7_full"]
    )
    merged["case_a_target_downsampling_hurt"] = merged["delta_multi8_full_minus_multi8"] > 0
    merged["case_b_clean_multilingual_beats_mono"] = merged["delta_multi8_full_minus_mono"] > 0
    merged["case_c_pretraining_beats_mono"] = merged["delta_pretrain_multi7_full_minus_mono"] > 0
    merged["case_d_pretraining_beats_multi8_full"] = merged["delta_pretrain_multi7_full_minus_multi8_full"] > 0
    merged["best_model"] = merged.apply(lambda row: _compute_best_model(row.to_dict()), axis=1)
    merged["case_e_mono_remains_best"] = merged["best_model"] == "mono"

    complete = merged[merged["rq4_complete"]].copy()
    LOGGER.info("RQ4 complete languages            = %d/%d", len(complete), len(merged))
    LOGGER.info("RQ4 macro F1 mono                 = %.4f", safe_mean(complete["mono_f1"]))
    LOGGER.info("RQ4 macro F1 multi8               = %.4f", safe_mean(complete["multi8_f1"]))
    LOGGER.info("RQ4 macro F1 multi8-full          = %.4f", safe_mean(complete["multi8_full_f1"]))
    LOGGER.info(
        "RQ4 macro F1 pretrain-multi7-full = %.4f",
        safe_mean(complete["pretrain_multi7_full_f1"]),
    )

    for _, row in merged.sort_values("language").iterrows():
        LOGGER.info(
            "RQ4 %s | mono=%s | multi8=%s | multi8-full=%s | pretrain-multi7-full=%s | complete=%s | best=%s",
            row["language"],
            f"{row['mono_f1']:.4f}" if row["mono_f1"] == row["mono_f1"] else "NA",
            f"{row['multi8_f1']:.4f}" if row["multi8_f1"] == row["multi8_f1"] else "NA",
            f"{row['multi8_full_f1']:.4f}" if row["multi8_full_f1"] == row["multi8_full_f1"] else "NA",
            f"{row['pretrain_multi7_full_f1']:.4f}" if row["pretrain_multi7_full_f1"] == row["pretrain_multi7_full_f1"] else "NA",
            row["rq4_complete"],
            row["best_model"],
        )

    return merged.sort_values("language").reset_index(drop=True)


def _compute_best_full_model(record: dict[str, Any]) -> str:
    candidates = (
        ("full_multi8", record["full_multi8_f1"]),
        ("full_multi12", record["full_multi12_f1"]),
        ("full_multi12_capaux", record["full_multi12_capaux_f1"]),
    )
    available = [(name, value) for name, value in candidates if value == value]
    if not available:
        return "missing"
    return max(available, key=lambda item: item[1])[0]


def compute_rq5(df):
    pd = _import_pandas()
    LOGGER.info("Computing RQ5: full multilingual training variants")
    full_multi8 = _rename_metric_columns(
        select_main_rows(df, "full-multi8")[["language", "p", "r", "f1", "acc"]],
        "full_multi8",
    )
    full_multi12 = _rename_metric_columns(
        select_main_rows(df, "full-multi12")[["language", "p", "r", "f1", "acc"]],
        "full_multi12",
    )
    full_multi12_capaux = _rename_metric_columns(
        select_main_rows(df, "full-multi12-capaux")[["language", "p", "r", "f1", "acc"]],
        "full_multi12_capaux",
    )

    merged = _merge_language_frames((full_multi8, full_multi12, full_multi12_capaux))
    if merged is None or merged.empty:
        return pd.DataFrame.from_records([], columns=[
            "language",
            "has_full_multi8",
            "has_full_multi12",
            "has_full_multi12_capaux",
            "rq5_complete",
            "full_multi8_f1",
            "full_multi12_f1",
            "full_multi12_capaux_f1",
            "delta_full_multi12_minus_full_multi8",
            "delta_full_multi12_capaux_minus_full_multi8",
            "delta_full_multi12_capaux_minus_full_multi12",
            "best_model",
        ])

    merged["delta_full_multi12_minus_full_multi8"] = (
        merged["full_multi12_f1"] - merged["full_multi8_f1"]
    )
    merged["delta_full_multi12_capaux_minus_full_multi8"] = (
        merged["full_multi12_capaux_f1"] - merged["full_multi8_f1"]
    )
    merged["delta_full_multi12_capaux_minus_full_multi12"] = (
        merged["full_multi12_capaux_f1"] - merged["full_multi12_f1"]
    )
    merged["has_full_multi8"] = merged["full_multi8_f1"].notna()
    merged["has_full_multi12"] = merged["full_multi12_f1"].notna()
    merged["has_full_multi12_capaux"] = merged["full_multi12_capaux_f1"].notna()
    merged["rq5_complete"] = (
        merged["has_full_multi8"]
        & merged["has_full_multi12"]
        & merged["has_full_multi12_capaux"]
    )
    merged["best_model"] = merged.apply(lambda row: _compute_best_full_model(row.to_dict()), axis=1)

    complete = merged[merged["rq5_complete"]].copy()
    LOGGER.info("RQ5 complete languages               = %d/%d", len(complete), len(merged))
    LOGGER.info("RQ5 macro F1 full-multi8             = %.4f", safe_mean(complete["full_multi8_f1"]))
    LOGGER.info("RQ5 macro F1 full-multi12            = %.4f", safe_mean(complete["full_multi12_f1"]))
    LOGGER.info(
        "RQ5 macro F1 full-multi12-capaux     = %.4f",
        safe_mean(complete["full_multi12_capaux_f1"]),
    )

    for _, row in merged.sort_values("language").iterrows():
        LOGGER.info(
            "RQ5 %s | full-multi8=%s | full-multi12=%s | full-multi12-capaux=%s | complete=%s | best=%s",
            row["language"],
            f"{row['full_multi8_f1']:.4f}" if row["full_multi8_f1"] == row["full_multi8_f1"] else "NA",
            f"{row['full_multi12_f1']:.4f}" if row["full_multi12_f1"] == row["full_multi12_f1"] else "NA",
            f"{row['full_multi12_capaux_f1']:.4f}" if row["full_multi12_capaux_f1"] == row["full_multi12_capaux_f1"] else "NA",
            row["rq5_complete"],
            row["best_model"],
        )

    return merged.sort_values("language").reset_index(drop=True)


def _exact_sign_test_pvalue(deltas: list[float]) -> float:
    nonzero = [delta for delta in deltas if delta != 0]
    if not nonzero:
        return 1.0
    positives = sum(1 for delta in nonzero if delta > 0)
    negatives = len(nonzero) - positives
    if positives == negatives:
        return 1.0
    tail = max(positives, negatives)
    probability = sum(math.comb(len(nonzero), value) for value in range(tail, len(nonzero) + 1))
    return min(1.0, 2.0 * probability / (2 ** len(nonzero)))


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        for pos in range(start, end):
            original_index = indexed[pos][0]
            ranks[original_index] = avg_rank
        start = end
    return ranks


def _exact_wilcoxon_pvalue(deltas: list[float]) -> float:
    nonzero = [delta for delta in deltas if delta != 0]
    if not nonzero:
        return 1.0

    ranks = _average_ranks([abs(delta) for delta in nonzero])
    scaled_ranks = [int(round(rank * 2)) for rank in ranks]
    total_rank = sum(scaled_ranks)
    observed_positive = sum(rank for delta, rank in zip(nonzero, scaled_ranks) if delta > 0)
    observed_stat = min(observed_positive, total_rank - observed_positive)

    extreme = 0
    assignments = 1 << len(scaled_ranks)
    for mask in range(assignments):
        positive_rank_sum = 0
        for idx, rank in enumerate(scaled_ranks):
            if mask & (1 << idx):
                positive_rank_sum += rank
        stat = min(positive_rank_sum, total_rank - positive_rank_sum)
        if stat <= observed_stat:
            extreme += 1
    return extreme / assignments


def _format_pvalue(value: float) -> str:
    return f"p = {value:.3f}"


def _build_comparison_record(comparison: str, favored_label: str, left_values: list[float],
                             right_values: list[float], interpretation: str) -> dict[str, str]:
    paired = [
        (float(left), float(right))
        for left, right in zip(left_values, right_values)
        if left == left and right == right
    ]
    deltas = [left - right for left, right in paired]
    total = len(deltas)
    favored_count = sum(1 for delta in deltas if delta > 0)
    mean_delta_points = 100.0 * (sum(deltas) / total) if total else float("nan")
    return {
        "Comparison": comparison,
        "Mean delta, F1 points": f"{mean_delta_points:+.2f} for {favored_label}",
        "Direction count": f"{favored_count}/{total} languages",
        "Exact sign test": _format_pvalue(_exact_sign_test_pvalue(deltas)),
        "Exact Wilcoxon": _format_pvalue(_exact_wilcoxon_pvalue(deltas)),
        "Interpretation": interpretation,
    }


def compute_statistical_summary(df, rq1, rq2, rq4, rq5):
    pd = _import_pandas()
    mono_main = select_main_rows(df, "mono")[["language", "f1"]].rename(columns={"f1": "mono_f1"})
    full_multi8_main = select_main_rows(df, "full-multi8")[["language", "f1"]].rename(
        columns={"f1": "full_multi8_f1"}
    )
    full_vs_mono = full_multi8_main.merge(mono_main, on="language", how="inner").sort_values("language")

    records = [
        _build_comparison_record(
            "Mono-L vs Multi-8",
            "Mono",
            rq1["mono_f1"].tolist(),
            rq1["multi8_f1"].tolist(),
            "Strong evidence that balanced Multi-8 underperforms Mono overall, though the sign test is conservative.",
        ),
        _build_comparison_record(
            "Multi-8 vs Multi-12",
            "Multi-8",
            rq2["multi8_f1"].tolist(),
            rq2["multi12_f1"].tolist(),
            "Statistically robust: adding lower-confidence auxiliary languages hurts balanced training.",
        ),
        _build_comparison_record(
            "Multi8-full-L vs Multi-8",
            "Multi8-full-L",
            rq4["multi8_full_f1"].tolist(),
            rq4["multi8_f1"].tolist(),
            "Strong evidence that the balanced Multi-8 losses are largely due to target downsampling.",
        ),
        _build_comparison_record(
            "Multi8-full-L vs Mono-L",
            "Multi8-full-L",
            rq4["multi8_full_f1"].tolist(),
            rq4["mono_f1"].tolist(),
            "No reliable evidence that full-target multilingual training beats monolingual overall. Better to say it recovers the gap.",
        ),
        _build_comparison_record(
            "Pretrain-Multi7-full-L vs Mono-L",
            "Pretrain",
            rq4["pretrain_multi7_full_f1"].tolist(),
            rq4["mono_f1"].tolist(),
            "No overall significant advantage for leave-one-out pretraining.",
        ),
        _build_comparison_record(
            "Pretrain-Multi7-full-L vs Multi8-full-L",
            "Pretrain",
            rq4["pretrain_multi7_full_f1"].tolist(),
            rq4["multi8_full_f1"].tolist(),
            "No evidence that pretraining-then-adaptation is better than joint full-target multilingual training.",
        ),
        _build_comparison_record(
            "Full-Multi8 vs Mono-L",
            "Full-Multi8",
            full_vs_mono["full_multi8_f1"].tolist(),
            full_vs_mono["mono_f1"].tolist(),
            "Suggestive but not significant. Treat as broadly competitive, not clearly better.",
        ),
        _build_comparison_record(
            "Full-Multi12 vs Full-Multi8",
            "Full-Multi12",
            rq5["full_multi12_f1"].tolist(),
            rq5["full_multi8_f1"].tolist(),
            "Not significant; effectively a tie.",
        ),
        _build_comparison_record(
            "Full-Multi12-CapAux vs Full-Multi12",
            "CapAux",
            rq5["full_multi12_capaux_f1"].tolist(),
            rq5["full_multi12_f1"].tolist(),
            "No clear evidence that capping auxiliary languages helps in the full-pool setting.",
        ),
    ]
    return pd.DataFrame.from_records(records)


def write_summary(outdir: Path, rq1, rq2, rq3, rq4, rq5, statistical_summary) -> Path:
    summary_path = outdir / "summary.txt"
    mono_macro = safe_mean(rq1["mono_f1"])
    multi8_macro = safe_mean(rq1["multi8_f1"])
    multi12_macro = safe_mean(rq2["multi12_f1"])

    with summary_path.open("w", encoding="utf-8") as file:
        file.write("NER RESULTS SUMMARY\n")
        file.write("===================\n\n")

        file.write("RQ1: Mono-L vs Multi-8\n")
        file.write("----------------------\n")
        file.write(f"Mono macro F1:   {mono_macro:.6f}\n")
        file.write(f"Multi-8 macro F1:{multi8_macro:.6f}\n")
        file.write(f"Delta:           {multi8_macro - mono_macro:.6f}\n\n")
        file.write(
            rq1[["language", "mono_f1", "multi8_f1", "delta_f1_multi8_minus_mono", "winner"]].to_string(index=False)
        )
        file.write("\n\n")

        file.write("RQ2: Multi-8 vs Multi-12\n")
        file.write("------------------------\n")
        file.write(f"Multi-8 macro F1:  {safe_mean(rq2['multi8_f1']):.6f}\n")
        file.write(f"Multi-12 macro F1: {multi12_macro:.6f}\n")
        file.write(f"Delta:             {multi12_macro - safe_mean(rq2['multi8_f1']):.6f}\n\n")
        file.write(
            rq2[
                ["language", "multi8_f1", "multi12_f1", "delta_f1_multi12_minus_multi8", "winner"]
            ].to_string(index=False)
        )
        file.write("\n\n")

        file.write("RQ3: Resource curves\n")
        file.write("--------------------\n")
        for language in sorted(CURVE_LANGUAGES):
            file.write(f"\nLanguage: {language}\n")
            sub = rq3[rq3["language"] == language][
                [
                    "budget_pct",
                    "mono_f1",
                    "multi8_f1",
                    "multi12_f1",
                    "delta_multi8_minus_mono",
                    "delta_multi12_minus_mono",
                    "best_model",
                ]
            ]
            if sub.empty:
                file.write("No complete curve points available.\n")
                continue
            file.write(sub.to_string(index=False))
            file.write("\n")

        file.write("\nRQ4: Target-specific multilingual training and leave-one-out pretraining\n")
        file.write("-----------------------------------------------------------------------\n")
        rq4_complete = rq4[rq4["rq4_complete"]] if "rq4_complete" in rq4.columns else rq4
        file.write(f"Complete languages:            {len(rq4_complete)}/{len(rq4)}\n")
        file.write(f"Mono macro F1:                 {safe_mean(rq4_complete['mono_f1']):.6f}\n")
        file.write(f"Multi-8 macro F1:              {safe_mean(rq4_complete['multi8_f1']):.6f}\n")
        file.write(f"Multi8-full-L macro F1:        {safe_mean(rq4_complete['multi8_full_f1']):.6f}\n")
        file.write(f"Pretrain-Multi7-full-L macro F1:{safe_mean(rq4_complete['pretrain_multi7_full_f1']):.6f}\n\n")
        file.write(
            rq4[
                [
                    "language",
                    "has_mono",
                    "has_multi8",
                    "has_multi8_full",
                    "has_pretrain_multi7_full",
                    "rq4_complete",
                    "mono_f1",
                    "multi8_f1",
                    "multi8_full_f1",
                    "pretrain_multi7_full_f1",
                    "delta_multi8_full_minus_multi8",
                    "delta_multi8_full_minus_mono",
                    "delta_pretrain_multi7_full_minus_mono",
                    "delta_pretrain_multi7_full_minus_multi8_full",
                    "best_model",
                    "case_a_target_downsampling_hurt",
                    "case_b_clean_multilingual_beats_mono",
                    "case_c_pretraining_beats_mono",
                    "case_d_pretraining_beats_multi8_full",
                    "case_e_mono_remains_best",
                ]
            ].to_string(index=False)
        )
        file.write("\n")

        file.write("\nRQ5: Full multilingual training variants\n")
        file.write("----------------------------------------\n")
        rq5_complete = rq5[rq5["rq5_complete"]] if "rq5_complete" in rq5.columns else rq5
        file.write(f"Complete languages:               {len(rq5_complete)}/{len(rq5)}\n")
        file.write(f"Full-Multi8 macro F1:             {safe_mean(rq5_complete['full_multi8_f1']):.6f}\n")
        file.write(f"Full-Multi12 macro F1:            {safe_mean(rq5_complete['full_multi12_f1']):.6f}\n")
        file.write(
            f"Full-Multi12-CapAux macro F1:     {safe_mean(rq5_complete['full_multi12_capaux_f1']):.6f}\n\n"
        )
        file.write(
            rq5[
                [
                    "language",
                    "has_full_multi8",
                    "has_full_multi12",
                    "has_full_multi12_capaux",
                    "rq5_complete",
                    "full_multi8_f1",
                    "full_multi12_f1",
                    "full_multi12_capaux_f1",
                    "delta_full_multi12_minus_full_multi8",
                    "delta_full_multi12_capaux_minus_full_multi8",
                    "delta_full_multi12_capaux_minus_full_multi12",
                    "best_model",
                ]
            ].to_string(index=False)
        )
        file.write("\n")

        file.write("\nStatistical Comparison Summary\n")
        file.write("------------------------------\n")
        file.write("Using the current language-level mean F1s, the important comparisons look like this:\n\n")
        file.write(statistical_summary.to_string(index=False))
        file.write("\n")

    LOGGER.info("Wrote summary: %s", summary_path)
    return summary_path


def write_outputs(outdir: Path, rq1, rq2, rq3, rq4, rq5, statistical_summary) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "rq1": outdir / "rq1_mono_vs_multi8.csv",
        "rq2": outdir / "rq2_multi8_vs_multi12.csv",
        "rq3": outdir / "rq3_resource_curves.csv",
        "rq4": outdir / "rq4_target_specific_vs_pretrain.csv",
        "rq5": outdir / "rq5_full_multilingual_variants.csv",
    }
    rq1.to_csv(outputs["rq1"], index=False)
    rq2.to_csv(outputs["rq2"], index=False)
    rq3.to_csv(outputs["rq3"], index=False)
    rq4.to_csv(outputs["rq4"], index=False)
    rq5.to_csv(outputs["rq5"], index=False)
    outputs["summary"] = write_summary(outdir, rq1, rq2, rq3, rq4, rq5, statistical_summary)
    return outputs


def analyze_results(csv_path: Path, outdir: Path) -> dict[str, Path]:
    df = read_results(csv_path)
    validate_expected_rows(df)
    rq1 = compute_rq1(df)
    rq2 = compute_rq2(df)
    rq3 = compute_rq3(df)
    rq4 = compute_rq4(df)
    rq5 = compute_rq5(df)
    statistical_summary = compute_statistical_summary(df, rq1, rq2, rq4, rq5)
    return write_outputs(outdir, rq1, rq2, rq3, rq4, rq5, statistical_summary)
