from __future__ import annotations

import csv
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
from ...train.token.ner_sdjt import build_split_datasets, compute_model_name, init_dirs, load_model_and_tokenizer

logger: Logger
paths: Paths


def compute_train_dir(m_args: ModelArguments, d_args: DataArguments, t_args: TrainingArguments, run_name: str) -> Path:
    run_spec = resolve_run_spec_from_name(run_name)
    model_name = compute_model_name(m_args, d_args, t_args, run_spec)
    output_dir = paths.get_script_path("train") / model_name
    if not output_dir.exists():
        raise FileNotFoundError(output_dir)
    return output_dir


def compute_output(m_args: ModelArguments, d_args: DataArguments, t_args: TrainingArguments) -> Path:
    model_name = (
        f"{d_args.dataset_name}.{m_args.short_name}.b{t_args.train_batch_size}.lr{t_args.learning_rate}.csv"
    )
    output = paths.context / model_name
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


def compute_macro_row(language_rows: Sequence[Dict[str, Any]], run_row: Dict[str, Any]) -> Dict[str, Any]:
    metrics = ("p", "r", "f1", "acc")
    row = dict(run_row)
    row["language"] = "macro"
    row["num_languages"] = len(language_rows)
    for metric in metrics:
        row[metric] = sum(float(lang_row[metric]) for lang_row in language_rows) / len(language_rows)
    return row


def build_result_rows(run_name: str, train_dir: Path, language_metrics: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    run_spec = resolve_run_spec_from_name(run_name)
    base_row = {
        "run_name": run_name,
        "pool_name": run_spec.pool_name,
        "budget_pct": run_spec.budget_pct,
        "train_dir": str(train_dir),
    }
    rows: List[Dict[str, Any]] = []
    for lang in sorted(language_metrics):
        row = dict(base_row)
        row["language"] = lang
        row["num_languages"] = 1
        row.update(language_metrics[lang])
        rows.append(row)
    if len(rows) > 1:
        rows.append(compute_macro_row(rows, base_row))
    return rows


def write_results_csv(output: Path, rows: Iterable[Dict[str, Any]]) -> None:
    columns = [
        "run_name",
        "pool_name",
        "budget_pct",
        "language",
        "num_languages",
        "p",
        "r",
        "f1",
        "acc",
        "train_dir",
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
            train_dir = compute_train_dir(model_args, data_args, train_args, run_name)
        except FileNotFoundError:
            if len(requested_run_names) == 1:
                raise
            logger.warning("Skipping %s because trained model dir is missing.", run_name)
            continue

        ner_samples = NerSamplesLoader(data_root, list(run_spec.train_languages))
        metrics = TokenClassificationMetrics(id2label=ner_samples.labeler.id2label)
        model_args.model_name_or_path = str(train_dir)
        model, tokenizer = load_model_and_tokenizer(model_args, cache_root, ner_samples.labeler)
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

        language_metrics = {
            lang: evaluate_language(trainer, dataset, lang)
            for lang, dataset in test_datasets.items()
        }
        run_rows = build_result_rows(run_name, train_dir, language_metrics)
        all_rows.extend(run_rows)
        logger.info("Evaluated %s: %s", run_name, run_rows)

    if not all_rows:
        raise FileNotFoundError(
            f"No SDJT NER models found for evaluation under {paths.get_script_path('train')}."
        )

    write_results_csv(output, all_rows)
    logger.info("Wrote %d SDJT NER result rows to %s", len(all_rows), output)
