from __future__ import annotations

import json
from logging import Logger
from pathlib import Path
from typing import Any, Dict

from ...app.args.data import DataArguments
from ...app.args.runtime import Paths
from ..resample.ner_sdjt import (
    SplitSamples,
    count_tokens,
    available_run_names,
    resolve_run_spec_from_name,
    load_run_corpora,
)
from .ner import _collect_tags, _compute_stats, _format_stats_table, _load_sentences, _write_stats

logger: Logger
paths: Paths


def summarize_corpora(samples_by_split: SplitSamples) -> Dict[str, Dict[str, Dict[str, int]]]:
    summary: Dict[str, Dict[str, Dict[str, int]]] = {}
    for split, split_samples in samples_by_split.items():
        summary[split] = {}
        for lang, sentences in split_samples.items():
            summary[split][lang] = {
                "sentences": len(sentences),
                "tokens": count_tokens(sentences),
            }
    return summary


def load_generated_run(run_dir: Path, run_name: str) -> SplitSamples:
    if not run_dir.exists():
        raise FileNotFoundError(f"Generated SDJT run split not found at {run_dir}. Run `./data resample ner-sdjt` first.")

    split_names = ("train", "eval", "test")
    snapshot: SplitSamples = {split: {} for split in split_names}
    for split in split_names:
        snapshot[split] = _load_sentences(run_dir, f".{split}")
        if not snapshot[split]:
            logger.warning("No %s samples found for run %s at %s", split, run_name, run_dir)
    return snapshot


def write_run_analysis(output_dir: Path, run_name: str, snapshot: SplitSamples, metadata: Dict[str, Any]) -> None:
    run_output_dir = output_dir / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    split_names = ("train", "eval", "test")

    tags_set = set()
    for split in split_names:
        tags_set.update(_collect_tags(snapshot[split]))
    tags = sorted(tags_set)

    for split in split_names:
        stats = _compute_stats(snapshot[split], tags)
        _write_stats(run_output_dir, stats, file_suffix=f".{split}")
        logger.info("SDJT %s %s stats:\n%s", run_name, split, _format_stats_table(stats))

    metadata_path = run_output_dir / "manifest.json"
    with metadata_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2, sort_keys=True)


def main(data_args: DataArguments) -> None:
    logger.info("Analyzing SDJT NER datasets")

    split_dir = paths.get_ctx_path("split")
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Split data not found at {split_dir}. Run `./data resample {paths.curr_context}` first."
        )

    snapshot_epoch = int(data_args.attributes.get("snapshot_epoch", 0))

    all_metadata: Dict[str, Dict[str, Any]] = {}
    run_names = available_run_names()
    run_specs = [resolve_run_spec_from_name(run_name) for run_name in run_names]
    for run_spec in run_specs:
        run_dir = split_dir / run_spec.run_name
        snapshot = load_generated_run(run_dir, run_spec.run_name)
        source_corpora = load_run_corpora(run_dir, run_spec)
        token_budgets = {
            lang: count_tokens(sentences)
            for lang, sentences in snapshot["train"].items()
        }
        metadata: Dict[str, Any] = {
            "run_name": run_spec.run_name,
            "pool_name": run_spec.pool_name,
            "train_languages": list(run_spec.train_languages),
            "eval_languages": list(run_spec.eval_languages),
            "target_language": run_spec.target_language,
            "budget_pct": run_spec.budget_pct,
            "uses_macro_eval": run_spec.uses_macro_eval,
            "snapshot_epoch": snapshot_epoch,
            "sampling_seed": int(data_args.sampling.seed or data_args.split.seed or 2611),
            "token_budgets": token_budgets,
            "snapshot_summary": summarize_corpora(snapshot),
            "source_summary": summarize_corpora(source_corpora),
            "dynamic_epoch_resampling": run_spec.is_multilingual,
        }
        write_run_analysis(paths.context, run_spec.run_name, snapshot, metadata)
        all_metadata[run_spec.run_name] = metadata

    runs_path = paths.context / "runs.json"
    with runs_path.open("w", encoding="utf-8") as fp:
        json.dump(all_metadata, fp, ensure_ascii=False, indent=2, sort_keys=True)
    logger.info("Wrote SDJT NER analysis for %d runs to %s", len(all_metadata), paths.context)
