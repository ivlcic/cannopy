#!/usr/bin/env python3
"""
Analyze multilingual NER results for the current paper RQs.

Expected input CSV columns:
run_name,pool_name,budget_pct,language,num_languages,p,r,f1,acc

Example:
python analyze_ner_results.py results.csv --outdir analysis_out --log-level INFO

What it computes
----------------
RQ1: Mono-L vs Multi-8 on the 8 main evaluation languages at budget 100
RQ2: Multi-8 vs Multi-12 on the 8 main evaluation languages at budget 100
RQ3: Resource curves for sr and sl at budgets 10/25/50/100

Outputs
-------
- Console logs
- analysis_out/rq1_mono_vs_multi8.csv
- analysis_out/rq2_multi8_vs_multi12.csv
- analysis_out/rq3_resource_curves.csv
- analysis_out/summary.txt
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd


MAIN_LANGS = ["bg", "cs", "hr", "pl", "ru", "sl", "sr", "uk"]
CURVE_LANGS = ["sr", "sl"]
BUDGETS = [10, 25, 50, 100]
REQUIRED_COLUMNS = {
    "run_name",
    "pool_name",
    "budget_pct",
    "language",
    "num_languages",
    "p",
    "r",
    "f1",
    "acc",
}


def setup_logging(log_level: str, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "analysis.log"

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


def read_results(csv_path: Path) -> pd.DataFrame:
    logging.info("Reading CSV: %s", csv_path)
    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    numeric_cols = ["budget_pct", "num_languages", "p", "r", "f1", "acc"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")

    df["pool_name"] = df["pool_name"].astype(str).str.strip()
    df["run_name"] = df["run_name"].astype(str).str.strip()
    df["language"] = df["language"].astype(str).str.strip()

    logging.info("Loaded %d rows", len(df))
    return df


def validate_expected_rows(df: pd.DataFrame) -> None:
    logging.info("Validating expected rows")

    # Main full-budget rows
    for pool in ["mono", "multi8", "multi12"]:
        for lang in MAIN_LANGS:
            mask = (
                (df["pool_name"] == pool)
                & (df["budget_pct"] == 100)
                & (df["language"] == lang)
            )
            count = int(mask.sum())
            if count == 0:
                logging.warning("Missing row for pool=%s, language=%s, budget=100", pool, lang)
            elif count > 1:
                logging.warning(
                    "Duplicate rows for pool=%s, language=%s, budget=100: %d",
                    pool,
                    lang,
                    count,
                )

    # Curve rows
    for lang in CURVE_LANGS:
        for pool in ["mono", "multi8", "multi12"]:
            for budget in BUDGETS:
                if pool == "mono":
                    run_prefix = f"mono-{lang}"
                else:
                    run_prefix = f"{pool}-{lang}"

                mask = (
                    (df["pool_name"] == pool)
                    & (df["budget_pct"] == budget)
                    & (df["language"] == lang)
                )
                count = int(mask.sum())
                if count == 0:
                    logging.warning(
                        "Missing curve row for pool=%s, language=%s, budget=%s",
                        pool,
                        lang,
                        budget,
                    )
                elif count > 1:
                    logging.warning(
                        "Duplicate curve rows for pool=%s, language=%s, budget=%s: %d",
                        pool,
                        lang,
                        budget,
                        count,
                    )


def safe_mean(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.mean())


def select_main_rows(df: pd.DataFrame, pool: str) -> pd.DataFrame:
    out = df[
        (df["pool_name"] == pool)
        & (df["budget_pct"] == 100)
        & (df["language"].isin(MAIN_LANGS))
    ].copy()

    out = out.sort_values("language").reset_index(drop=True)
    logging.info("Selected %d main rows for pool=%s", len(out), pool)
    return out


def compute_rq1(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Computing RQ1: Mono-L vs Multi-8")
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
        lambda x: "multi8" if x > 0 else "mono"
    )

    mono_macro = safe_mean(merged["mono_f1"])
    multi8_macro = safe_mean(merged["multi8_f1"])
    logging.info("RQ1 mono macro F1   = %.4f", mono_macro)
    logging.info("RQ1 multi8 macro F1 = %.4f", multi8_macro)
    logging.info("RQ1 delta           = %.4f", multi8_macro - mono_macro)

    best = merged.sort_values("delta_f1_multi8_minus_mono", ascending=False)
    if not best.empty:
        logging.info(
            "RQ1 biggest multilingual gain: %s (%.4f F1)",
            best.iloc[0]["language"],
            best.iloc[0]["delta_f1_multi8_minus_mono"],
        )
        logging.info(
            "RQ1 biggest multilingual loss: %s (%.4f F1)",
            best.iloc[-1]["language"],
            best.iloc[-1]["delta_f1_multi8_minus_mono"],
        )

    return merged.sort_values("language").reset_index(drop=True)


def compute_rq2(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Computing RQ2: Multi-8 vs Multi-12")
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
        lambda x: "multi12" if x > 0 else "multi8"
    )

    multi8_macro = safe_mean(merged["multi8_f1"])
    multi12_macro = safe_mean(merged["multi12_f1"])
    logging.info("RQ2 multi8 macro F1  = %.4f", multi8_macro)
    logging.info("RQ2 multi12 macro F1 = %.4f", multi12_macro)
    logging.info("RQ2 delta            = %.4f", multi12_macro - multi8_macro)

    best = merged.sort_values("delta_f1_multi12_minus_multi8", ascending=False)
    if not best.empty:
        logging.info(
            "RQ2 biggest gain from extra languages: %s (%.4f F1)",
            best.iloc[0]["language"],
            best.iloc[0]["delta_f1_multi12_minus_multi8"],
        )
        logging.info(
            "RQ2 biggest loss from extra languages: %s (%.4f F1)",
            best.iloc[-1]["language"],
            best.iloc[-1]["delta_f1_multi12_minus_multi8"],
        )

    return merged.sort_values("language").reset_index(drop=True)


def extract_curve_row(df: pd.DataFrame, pool: str, lang: str, budget: int) -> pd.Series | None:
    rows = df[
        (df["pool_name"] == pool)
        & (df["language"] == lang)
        & (df["budget_pct"] == budget)
    ]
    if rows.empty:
        return None
    if len(rows) > 1:
        logging.warning(
            "Multiple rows found for pool=%s, language=%s, budget=%s; using first",
            pool,
            lang,
            budget,
        )
    return rows.iloc[0]


def compute_rq3(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Computing RQ3: resource curves for sr and sl")
    records: List[dict] = []

    for lang in CURVE_LANGS:
        for budget in BUDGETS:
            mono = extract_curve_row(df, "mono", lang, budget)
            m8 = extract_curve_row(df, "multi8", lang, budget)
            m12 = extract_curve_row(df, "multi12", lang, budget)

            if mono is None or m8 is None or m12 is None:
                logging.warning(
                    "Skipping incomplete RQ3 point for language=%s budget=%s",
                    lang,
                    budget,
                )
                continue

            rec = {
                "language": lang,
                "budget_pct": budget,
                "mono_f1": float(mono["f1"]),
                "multi8_f1": float(m8["f1"]),
                "multi12_f1": float(m12["f1"]),
                "mono_p": float(mono["p"]),
                "multi8_p": float(m8["p"]),
                "multi12_p": float(m12["p"]),
                "mono_r": float(mono["r"]),
                "multi8_r": float(m8["r"]),
                "multi12_r": float(m12["r"]),
            }
            rec["delta_multi8_minus_mono"] = rec["multi8_f1"] - rec["mono_f1"]
            rec["delta_multi12_minus_mono"] = rec["multi12_f1"] - rec["mono_f1"]
            rec["delta_multi12_minus_multi8"] = rec["multi12_f1"] - rec["multi8_f1"]
            rec["best_model"] = max(
                [("mono", rec["mono_f1"]), ("multi8", rec["multi8_f1"]), ("multi12", rec["multi12_f1"])],
                key=lambda x: x[1],
            )[0]
            records.append(rec)

    out = pd.DataFrame.from_records(records).sort_values(["language", "budget_pct"]).reset_index(drop=True)

    for lang in CURVE_LANGS:
        sub = out[out["language"] == lang].sort_values("budget_pct")
        if sub.empty:
            continue
        logging.info("RQ3 %s curve:", lang)
        for _, row in sub.iterrows():
            logging.info(
                "  budget=%3d | mono=%.4f | multi8=%.4f | multi12=%.4f | best=%s",
                int(row["budget_pct"]),
                row["mono_f1"],
                row["multi8_f1"],
                row["multi12_f1"],
                row["best_model"],
            )

    return out


def write_summary(
    outdir: Path,
    rq1: pd.DataFrame,
    rq2: pd.DataFrame,
    rq3: pd.DataFrame,
) -> None:
    summary_path = outdir / "summary.txt"
    mono_macro = safe_mean(rq1["mono_f1"])
    multi8_macro = safe_mean(rq1["multi8_f1"])
    multi12_macro = safe_mean(rq2["multi12_f1"])

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("NER RESULTS SUMMARY\n")
        f.write("===================\n\n")

        f.write("RQ1: Mono-L vs Multi-8\n")
        f.write("----------------------\n")
        f.write(f"Mono macro F1:   {mono_macro:.6f}\n")
        f.write(f"Multi-8 macro F1:{multi8_macro:.6f}\n")
        f.write(f"Delta:           {multi8_macro - mono_macro:.6f}\n\n")
        f.write(rq1[["language", "mono_f1", "multi8_f1", "delta_f1_multi8_minus_mono", "winner"]].to_string(index=False))
        f.write("\n\n")

        f.write("RQ2: Multi-8 vs Multi-12\n")
        f.write("------------------------\n")
        f.write(f"Multi-8 macro F1:  {safe_mean(rq2['multi8_f1']):.6f}\n")
        f.write(f"Multi-12 macro F1: {multi12_macro:.6f}\n")
        f.write(f"Delta:             {multi12_macro - safe_mean(rq2['multi8_f1']):.6f}\n\n")
        f.write(rq2[["language", "multi8_f1", "multi12_f1", "delta_f1_multi12_minus_multi8", "winner"]].to_string(index=False))
        f.write("\n\n")

        f.write("RQ3: Resource curves\n")
        f.write("--------------------\n")
        for lang in CURVE_LANGS:
            sub = rq3[rq3["language"] == lang][
                ["budget_pct", "mono_f1", "multi8_f1", "multi12_f1",
                 "delta_multi8_minus_mono", "delta_multi12_minus_mono", "best_model"]
            ]
            f.write(f"\nLanguage: {lang}\n")
            f.write(sub.to_string(index=False))
            f.write("\n")

    logging.info("Wrote summary: %s", summary_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path, help="Path to the input CSV")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis_out"),
        help="Directory for outputs",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()

    setup_logging(args.log_level, args.outdir)

    try:
        df = read_results(args.csv_path)
        validate_expected_rows(df)

        rq1 = compute_rq1(df)
        rq2 = compute_rq2(df)
        rq3 = compute_rq3(df)

        rq1.to_csv(args.outdir / "rq1_mono_vs_multi8.csv", index=False)
        rq2.to_csv(args.outdir / "rq2_multi8_vs_multi12.csv", index=False)
        rq3.to_csv(args.outdir / "rq3_resource_curves.csv", index=False)
        write_summary(args.outdir, rq1, rq2, rq3)

        logging.info("Done.")
        logging.info("Saved outputs to: %s", args.outdir.resolve())

    except Exception as exc:
        logging.exception("Analysis failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()