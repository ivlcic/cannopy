from __future__ import annotations

import csv
from statistics import pstdev
from logging import Logger
from pathlib import Path
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
from ...data.resample.ner_sdjt import available_run_names, resolve_run_spec_from_name
from ...train.token.ner_sdjt import (
    build_split_datasets,
    compute_model_prefix,
    init_dirs,
    load_model_and_tokenizer,
)

logger: Logger
paths: Paths


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
    base_row = {
        "run_name": run_name,
        "pool_name": run_spec.pool_name,
        "budget_pct": run_spec.budget_pct,
        "model_prefix": model_prefix,
        "models_evaluated": len(train_dirs),
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


def main(data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments) -> None:
    logger.info("Evaluating SDJT NER")

    requested_run_names = resolve_requested_run_names(data_args)
    output = compute_output(model_args, data_args, train_args)
    all_rows: List[Dict[str, Any]] = []

    for run_name in requested_run_names:
        run_spec = resolve_run_spec_from_name(run_name)
        try:
            data_root, cache_root = init_dirs(paths, run_name)
            train_dirs = compute_train_dirs(model_args, data_args, train_args, run_name)
        except FileNotFoundError:
            if len(requested_run_names) == 1:
                raise
            logger.warning("Skipping %s because trained model dir is missing.", run_name)
            continue

        train_languages = list(run_spec.train_languages)
        evaluation_languages = list(run_spec.eval_languages)
        ner_samples = NerSamplesLoader(
            data_root,
            train_languages,
            split_languages={
                "train": train_languages,
                "eval": evaluation_languages,
                "test": evaluation_languages,
            },
        )
        test_metric_rows: List[Dict[str, Dict[str, float]]] = []
        for train_dir in train_dirs:
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

        run_rows = build_result_rows(run_name, train_dirs, test_metric_rows)
        all_rows.extend(run_rows)
        logger.info("Evaluated %s across %d model seeds: %s", run_name, len(train_dirs), run_rows)

    if not all_rows:
        raise FileNotFoundError(
            f"No SDJT NER models found for evaluation under {paths.get_script_path('train')}."
        )

    write_results_csv(output, all_rows)
    logger.info("Wrote %d SDJT NER result rows to %s", len(all_rows), output)
