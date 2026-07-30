from __future__ import annotations

import csv
import json
import math
import re
from logging import Logger
from pathlib import Path
from statistics import pstdev
from typing import Any, Dict, Iterable, List, Sequence

from torch.utils.data import Dataset
from transformers import (
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from ...app.args.data import DataArguments
from ...app.args.model import ModelArguments
from ...app.args.runtime import Paths
from ...app.dataset import NerSamplesLoader
from ...app.metrics import TokenClassificationMetrics
from ...data.resample.ner_sdjt import (
    available_run_names,
    parse_seed_suffix,
    resolve_run_spec_from_name,
)
from ...train.token.ner_sdjt import (
    build_split_datasets,
    compute_model_prefix,
    init_dirs,
    load_model_and_tokenizer,
)

logger: Logger
paths: Paths

SWEEP_COLUMNS = [
    "rank",
    "selected",
    "model",
    "batch_size",
    "learning_rate",
    "classifier_dropout",
    "classifier_dropout_tag",
    "warmup_ratio_tag",
    "weight_decay_tag",
    "num_seeds",
    "discovered_num_seeds",
    "complete_seed_set",
    "seeds",
    "validation_macro_f1_mean",
    "validation_macro_f1_std",
    "validation_macro_f1_min",
    "validation_macro_f1_max",
    "best_epoch_mean",
    "best_epochs_by_seed",
    "validation_macro_f1_by_seed",
    "run_directories",
]


def compute_train_dirs(m_args: ModelArguments, d_args: DataArguments, t_args: TrainingArguments, run_name: str) -> List[Path]:
    run_spec = resolve_run_spec_from_name(run_name)
    model_prefix = compute_model_prefix(m_args, d_args, t_args, run_spec)
    train_root = paths.get_script_path("train")
    train_dirs = sorted(
        path for path in train_root.glob(f"{model_prefix}.s*")
        if path.is_dir()
    )
    if not train_dirs:
        raise FileNotFoundError(train_root / f"{model_prefix}.s*")
    return train_dirs


def compute_output(m_args: ModelArguments, d_args: DataArguments, t_args: TrainingArguments) -> Path:
    output = paths.context / f"{compute_model_prefix(m_args, d_args, t_args)}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def resolve_requested_run_names(data_args: DataArguments) -> List[str]:
    attrs = data_args.attributes or {}
    run_name = str(attrs.get("run_name", "")).strip()
    if run_name:
        return [run_name]
    return available_run_names()


def evaluate_language(trainer: Trainer, dataset: Dataset, lang: str) -> Dict[str, float]:
    metrics = trainer.evaluate(eval_dataset=dataset, metric_key_prefix=f"test_{lang}")
    return {
        "p": float(metrics.get(f"test_{lang}_p", 0.0)),
        "r": float(metrics.get(f"test_{lang}_r", 0.0)),
        "f1": float(metrics.get(f"test_{lang}_f1", 0.0)),
        "acc": float(metrics.get(f"test_{lang}_acc", 0.0)),
    }


def aggregate_metric_values(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(sum(values) / len(values)), float(pstdev(values))


def aggregate_language_metrics(metric_rows: Sequence[Dict[str, Any]], base_row: Dict[str, Any], language: str) -> Dict[str, Any]:
    row = dict(base_row)
    row["language"] = language
    row["num_models"] = len(metric_rows)
    for metric in ("p", "r", "f1", "acc"):
        mean_value, std_value = aggregate_metric_values([float(metric_row[metric]) for metric_row in metric_rows])
        row[metric] = mean_value
        row[f"{metric}_std"] = std_value
    return row


def build_result_rows(run_name: str, train_dirs: Sequence[Path],
                      metric_rows_by_model: Sequence[Dict[str, Dict[str, float]]]) -> List[Dict[str, Any]]:
    run_spec = resolve_run_spec_from_name(run_name)
    model_prefix = train_dirs[0].name.rsplit(".s", 1)[0]
    model_seeds = [parse_seed_suffix(train_dir) for train_dir in train_dirs]
    base_row = {
        "run_name": run_name,
        "pool_name": run_spec.pool_name,
        "budget_pct": run_spec.budget_pct,
        "model_prefix": model_prefix,
        "models_evaluated": len(train_dirs),
        "seeds": ";".join(str(seed) for seed in model_seeds),
    }
    rows: List[Dict[str, Any]] = []
    if not metric_rows_by_model:
        return rows

    languages = sorted(metric_rows_by_model[0].keys())
    for lang in languages:
        lang_rows = [metric_rows[lang] for metric_rows in metric_rows_by_model if lang in metric_rows]
        rows.append(aggregate_language_metrics(lang_rows, base_row, lang))
    if len(languages) > 1:
        macro_rows = []
        for metric_rows in metric_rows_by_model:
            metric_names = ("p", "r", "f1", "acc")
            macro_row = {
                metric: sum(float(metric_rows[lang][metric]) for lang in languages) / len(languages)
                for metric in metric_names
            }
            macro_rows.append(macro_row)
        rows.append(aggregate_language_metrics(macro_rows, base_row, "macro"))
    return rows


def write_results_csv(output: Path, rows: Iterable[Dict[str, Any]]) -> None:
    columns = [
        "run_name",
        "pool_name",
        "budget_pct",
        "models_evaluated",
        "seeds",
        "language",
        "num_models",
        "p",
        "p_std",
        "r",
        "r_std",
        "f1",
        "f1_std",
        "acc",
        "acc_std",
        "model_prefix",
    ]
    with output.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_sweep_run_dir(run_dir: Path, dataset_name: str, model_name: str) -> Dict[str, Any] | None:
    prefix = re.escape(f"{dataset_name}.multi8.{model_name}.")
    pattern = re.compile(
        rf"^{prefix}b(?P<batch_size>\d+)\.lr(?P<learning_rate>.+?)"
        rf"\.cd(?P<classifier_dropout_tag>[^.]+)"
        rf"\.wr(?P<warmup_ratio_tag>[^.]+)"
        rf"\.wd(?P<weight_decay_tag>[^.]+)"
        rf"\.s(?P<seed>\d+)$"
    )
    match = pattern.fullmatch(run_dir.name)
    if match is None:
        return None
    values = match.groupdict()
    try:
        learning_rate = float(values["learning_rate"])
    except ValueError:
        logger.warning("Skipping sweep directory with invalid learning rate: %s", run_dir)
        return None
    return {
        "model": model_name,
        "batch_size": int(values["batch_size"]),
        "learning_rate": learning_rate,
        "classifier_dropout_tag": values["classifier_dropout_tag"],
        "warmup_ratio_tag": values["warmup_ratio_tag"],
        "weight_decay_tag": values["weight_decay_tag"],
        "seed": int(values["seed"]),
        "run_directory": run_dir.name,
    }


def load_classifier_dropout(run_dir: Path) -> float | None:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return None
    with config_path.open(encoding="utf-8") as fp:
        config = json.load(fp)
    classifier_dropout = config.get("classifier_dropout")
    if classifier_dropout is None:
        classifier_dropout = config.get("hidden_dropout_prob")
    return float(classifier_dropout) if classifier_dropout is not None else None


def load_best_validation_result(state_path: Path) -> tuple[float, float, int]:
    with state_path.open(encoding="utf-8") as fp:
        state = json.load(fp)

    best_metric = state.get("best_metric")
    if best_metric is None or not math.isfinite(float(best_metric)):
        raise ValueError(f"Missing finite best_metric in {state_path}")
    best_metric = float(best_metric)

    best_step = state.get("best_global_step")
    if best_step is None:
        checkpoint = str(state.get("best_model_checkpoint") or "")
        checkpoint_match = re.search(r"checkpoint-(\d+)$", checkpoint)
        best_step = int(checkpoint_match.group(1)) if checkpoint_match else None

    evaluation_entries = [
        entry
        for entry in state.get("log_history", [])
        if "eval_macro_f1" in entry
    ]
    best_entry = next(
        (entry for entry in evaluation_entries if best_step is not None and entry.get("step") == best_step),
        None,
    )
    if best_entry is None:
        best_entry = next(
            (
                entry
                for entry in evaluation_entries
                if math.isclose(float(entry["eval_macro_f1"]), best_metric, rel_tol=0.0, abs_tol=1e-12)
            ),
            None,
        )
    if best_entry is None:
        raise ValueError(f"Unable to resolve the best validation epoch in {state_path}")

    return best_metric, float(best_entry["epoch"]), int(best_entry["step"])


def collect_sweep_runs(train_root: Path, dataset_name: str, model_name: str) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    if not train_root.exists():
        raise FileNotFoundError(train_root)

    for run_dir in sorted(path for path in train_root.iterdir() if path.is_dir()):
        run = parse_sweep_run_dir(run_dir, dataset_name, model_name)
        if run is None:
            continue
        state_path = run_dir / "trainer_state.json"
        if not state_path.exists():
            logger.warning("Skipping incomplete sweep run without trainer_state.json: %s", run_dir)
            continue
        try:
            classifier_dropout = load_classifier_dropout(run_dir)
            best_metric, best_epoch, best_step = load_best_validation_result(state_path)
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            logger.warning("Skipping invalid sweep run %s: %s", run_dir, exc)
            continue
        run.update({
            "classifier_dropout": classifier_dropout,
            "best_validation_macro_f1": best_metric,
            "best_epoch": best_epoch,
            "best_step": best_step,
        })
        runs.append(run)

    if not runs:
        expected = train_root / f"{dataset_name}.multi8.{model_name}.*.s*/trainer_state.json"
        raise FileNotFoundError(expected)
    return runs


def build_sweep_rows(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    discovered_seeds = sorted({int(run["seed"]) for run in runs})
    discovered_seed_set = set(discovered_seeds)
    grouped: Dict[tuple[Any, ...], List[Dict[str, Any]]] = {}
    for run in runs:
        key = (
            run["model"],
            run["batch_size"],
            run["learning_rate"],
            run["classifier_dropout"],
            run["classifier_dropout_tag"],
            run["warmup_ratio_tag"],
            run["weight_decay_tag"],
        )
        grouped.setdefault(key, []).append(run)

    rows: List[Dict[str, Any]] = []
    for key, candidate_runs in grouped.items():
        candidate_runs = sorted(candidate_runs, key=lambda item: int(item["seed"]))
        seeds = [int(run["seed"]) for run in candidate_runs]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate sweep seeds for configuration {key}: {seeds}")
        metrics = [float(run["best_validation_macro_f1"]) for run in candidate_runs]
        epochs = [float(run["best_epoch"]) for run in candidate_runs]
        metric_mean, metric_std = aggregate_metric_values(metrics)
        rows.append({
            "rank": "",
            "selected": False,
            "model": key[0],
            "batch_size": key[1],
            "learning_rate": key[2],
            "classifier_dropout": key[3],
            "classifier_dropout_tag": key[4],
            "warmup_ratio_tag": key[5],
            "weight_decay_tag": key[6],
            "num_seeds": len(seeds),
            "discovered_num_seeds": len(discovered_seeds),
            "complete_seed_set": set(seeds) == discovered_seed_set,
            "seeds": ";".join(str(seed) for seed in seeds),
            "validation_macro_f1_mean": metric_mean,
            "validation_macro_f1_std": metric_std,
            "validation_macro_f1_min": min(metrics),
            "validation_macro_f1_max": max(metrics),
            "best_epoch_mean": sum(epochs) / len(epochs),
            "best_epochs_by_seed": ";".join(
                f"{run['seed']}:{run['best_epoch']:g}"
                for run in candidate_runs
            ),
            "validation_macro_f1_by_seed": ";".join(
                f"{run['seed']}:{run['best_validation_macro_f1']:.12g}"
                for run in candidate_runs
            ),
            "run_directories": ";".join(str(run["run_directory"]) for run in candidate_runs),
        })

    complete_rows = sorted(
        (row for row in rows if row["complete_seed_set"]),
        key=lambda row: (
            -float(row["validation_macro_f1_mean"]),
            float(row["validation_macro_f1_std"]),
            float(row["learning_rate"]),
            float(row["classifier_dropout"]) if row["classifier_dropout"] is not None else math.inf,
        ),
    )
    for rank, row in enumerate(complete_rows, start=1):
        row["rank"] = rank
        row["selected"] = rank == 1

    incomplete_rows = sorted(
        (row for row in rows if not row["complete_seed_set"]),
        key=lambda row: -float(row["validation_macro_f1_mean"]),
    )
    return complete_rows + incomplete_rows


def write_sweep_csv(output: Path, rows: Iterable[Dict[str, Any]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=SWEEP_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def sweep(data_args: DataArguments, model_args: ModelArguments) -> None:
    model_name = str(model_args.short_name or "").strip()
    if not model_name:
        raise ValueError("model.short_name is required to collect SDJT sweep results.")

    train_root = paths.get_script_path("train")
    runs = collect_sweep_runs(train_root, data_args.dataset_name, model_name)
    rows = build_sweep_rows(runs)
    output = paths.context / f"{data_args.dataset_name}.sweep.{model_name}.csv"
    write_sweep_csv(output, rows)

    complete_count = sum(bool(row["complete_seed_set"]) for row in rows)
    logger.info(
        "Wrote %d SDJT Multi-8 sweep configurations (%d complete) from %d runs to %s",
        len(rows),
        complete_count,
        len(runs),
        output,
    )
    if rows and rows[0]["selected"]:
        logger.info(
            "Selected %s sweep configuration: learning_rate=%s classifier_dropout=%s "
            "validation_macro_f1_mean=%.6f validation_macro_f1_std=%.6f seeds=%s",
            model_name,
            rows[0]["learning_rate"],
            rows[0]["classifier_dropout"],
            rows[0]["validation_macro_f1_mean"],
            rows[0]["validation_macro_f1_std"],
            rows[0]["seeds"],
        )


def main(data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments) -> None:
    logger.info("Evaluating SDJT NER")

    requested_run_names = resolve_requested_run_names(data_args)
    output = compute_output(model_args, data_args, train_args)
    all_rows: List[Dict[str, Any]] = []

    for run_name in requested_run_names:
        run_spec = resolve_run_spec_from_name(run_name)
        try:
            train_dirs = compute_train_dirs(model_args, data_args, train_args, run_name)
        except FileNotFoundError:
            if len(requested_run_names) == 1:
                raise
            logger.warning("Skipping %s because trained model dir is missing.", run_name)
            continue

        train_languages = list(run_spec.train_languages)
        evaluation_languages = list(run_spec.eval_languages)
        test_metric_rows: List[Dict[str, Dict[str, float]]] = []
        for train_dir in train_dirs:
            model_seed = parse_seed_suffix(train_dir)
            data_root, cache_root = init_dirs(paths, run_name, model_seed)
            ner_samples = NerSamplesLoader(
                data_root,
                train_languages,
                split_languages={
                    "train": train_languages,
                    "eval": evaluation_languages,
                    "test": evaluation_languages,
                },
            )
            metrics = TokenClassificationMetrics(id2label=ner_samples.labeler.id2label)
            model, tokenizer = load_model_and_tokenizer(model_args, cache_root, ner_samples.labeler, train_dir)
            collator = DataCollatorForTokenClassification(tokenizer, padding="longest")
            test_datasets = build_split_datasets(tokenizer, model_args.max_seq_length, ner_samples, "test")

            trainer = Trainer(
                model=model,
                args=train_args,
                eval_dataset=next(iter(test_datasets.values())),
                data_collator=collator,
                processing_class=tokenizer,
                compute_metrics=metrics,
            )
            test_metric_rows.append({
                lang: evaluate_language(trainer, dataset, lang)
                for lang, dataset in test_datasets.items()
            })
            logger.info(
                "Evaluated %s model seed %d on matching split %s",
                run_name,
                model_seed,
                data_root,
            )

        run_rows = build_result_rows(run_name, train_dirs, test_metric_rows)
        all_rows.extend(run_rows)
        logger.info("Evaluated %s across %d model seeds: %s", run_name, len(train_dirs), run_rows)

    if not all_rows:
        raise FileNotFoundError(
            f"No SDJT NER models found for evaluation under {paths.get_script_path('train')}."
        )

    write_results_csv(output, all_rows)
    logger.info("Wrote %d SDJT NER result rows to %s", len(all_rows), output)
