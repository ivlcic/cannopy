from __future__ import annotations

import logging
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


def write_summary(outdir: Path, rq1, rq2, rq3) -> Path:
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

    LOGGER.info("Wrote summary: %s", summary_path)
    return summary_path


def write_outputs(outdir: Path, rq1, rq2, rq3) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "rq1": outdir / "rq1_mono_vs_multi8.csv",
        "rq2": outdir / "rq2_multi8_vs_multi12.csv",
        "rq3": outdir / "rq3_resource_curves.csv",
    }
    rq1.to_csv(outputs["rq1"], index=False)
    rq2.to_csv(outputs["rq2"], index=False)
    rq3.to_csv(outputs["rq3"], index=False)
    outputs["summary"] = write_summary(outdir, rq1, rq2, rq3)
    return outputs


def analyze_results(csv_path: Path, outdir: Path) -> dict[str, Path]:
    df = read_results(csv_path)
    validate_expected_rows(df)
    rq1 = compute_rq1(df)
    rq2 = compute_rq2(df)
    rq3 = compute_rq3(df)
    return write_outputs(outdir, rq1, rq2, rq3)
