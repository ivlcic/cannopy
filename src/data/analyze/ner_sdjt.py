from __future__ import annotations

import csv
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List

from app.pip import Pip
from .__ner_sdjt import write_dataset_shift_figures
from .ner import _collect_tags, _compute_stats, _load_sentences
from ..resample.ner_sdjt import SplitSamples, available_run_names, resolve_run_spec_from_name
from ...app.args.data import DataArguments
from ...app.args.runtime import Paths

logger: Logger
paths: Paths


def load_generated_run(run_dir: Path, run_name: str) -> SplitSamples:
    if not run_dir.exists():
        raise FileNotFoundError(f"Generated SDJT run split not found at {run_dir}. "
                                f"Run `./data resample ner-sdjt` first.")

    split_names = ("train", "eval", "test")
    snapshot: SplitSamples = {split: {} for split in split_names}
    for split in split_names:
        snapshot[split] = _load_sentences(run_dir, f".{split}")
        if not snapshot[split]:
            logger.warning("No %s samples found for run %s at %s", split, run_name, run_dir)
    return snapshot


def build_stats_rows(snapshot: SplitSamples, run_metadata: Dict[str, Any], tags: List[str]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for split in ("train", "eval", "test"):
        stats = _compute_stats(snapshot[split], tags)
        for lang in sorted(stats["labels"].keys()):
            counter = stats["labels"][lang]
            label_count = sum(counter.values())
            row = [
                run_metadata["run_name"],
                run_metadata["pool_name"],
                run_metadata["budget_pct"],
                run_metadata["seed"],
                split,
                lang,
                stats["sentences"][lang],
                stats["tokens"][lang],
                label_count,
            ]
            row.extend(counter.get(tag, 0) for tag in tags)
            rows.append(row)
    return rows


def _write_stats_csv(output_file: Path, rows: List[List[Any]], tags: List[str]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "run_name",
            "pool_name",
            "budget_pct",
            "seed",
            "split",
            "language",
            "sentences",
            "tokens",
            "label_count",
            *tags,
        ])
        writer.writerows(rows)


def results():
    logger.info("Analyzing SDJT NER Results")
    Pip.install_packages("pandas", "3.0.2")
    pass


def main(data_args: DataArguments) -> None:
    logger.info("Analyzing SDJT NER datasets")

    split_dir = paths.get_ctx_path("split")
    if not split_dir.exists():
        raise FileNotFoundError(f"Split data not found at {split_dir}. "
                                f"Run `./data resample {paths.curr_context}` first.")

    output_dir = paths.get_ctx_path("analyze")
    output_file = output_dir / "ner-stats.csv"
    base_stats_file = paths.base.root / "result" / "data" / "analyze" / "ner" / "ner-stats.csv"
    density_figure = output_dir / "entity-density-by-language.svg"
    composition_figure = output_dir / "label-composition-by-language.svg"
    seed = int(data_args.sampling.seed or data_args.split.seed or 2611)
    run_specs = [resolve_run_spec_from_name(run_name) for run_name in available_run_names()]

    snapshots: Dict[str, SplitSamples] = {}
    tags_set = set()
    for run_spec in run_specs:
        run_dir = split_dir / run_spec.run_name
        snapshot = load_generated_run(run_dir, run_spec.run_name)
        snapshots[run_spec.run_name] = snapshot
        for split in ("train", "eval", "test"):
            tags_set.update(_collect_tags(snapshot[split]))

    tags = sorted(tags_set)
    rows: List[List[Any]] = []
    for run_spec in run_specs:
        rows.extend(build_stats_rows(
            snapshots[run_spec.run_name],
            {
                "run_name": run_spec.run_name,
                "pool_name": run_spec.pool_name,
                "budget_pct": run_spec.budget_pct,
                "seed": seed,
            },
            tags,
        ))

    _write_stats_csv(output_file, rows, tags)
    logger.info("Wrote %d SDJT NER stats rows to %s", len(rows), output_file)

    write_dataset_shift_figures(base_stats_file, density_figure, composition_figure)
    logger.info("Wrote entity density figure to %s", density_figure)
    logger.info("Wrote label composition figure to %s", composition_figure)
