from __future__ import annotations

from pathlib import Path
from logging import Logger
from typing import Dict, Mapping, Optional

from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from ...app.args.data import DataArguments
from ...app.dataset import NerDataset, NerSamplesLoader
from ...app.args.model import ModelArguments
from ...app.args.runtime import Paths
from ...app.metrics import TokenClassificationMetrics
from ...data.resample.ner_sdjt import RunSpec, available_run_names, resolve_run_spec_from_name

logger: Logger
paths: Paths


def _format_language_counts(samples_by_lang: Mapping[str, list]) -> str:
    if not samples_by_lang:
        return "none"
    return ", ".join(f"{lang}={len(samples)}" for lang, samples in sorted(samples_by_lang.items()))


def _log_run_configuration(run_spec: RunSpec, data_root: Path, cache_root: Path,
                           output_dir: str, model_args: ModelArguments,
                           train_args: TrainingArguments) -> None:
    logger.info(
        "Resolved SDJT NER run configuration: run=%s pool=%s budget_pct=%d train_langs=%s eval_langs=%s "
        "macro_eval=%s seed=%s data_root=%s cache_root=%s output_dir=%s",
        run_spec.run_name,
        run_spec.pool_name,
        run_spec.budget_pct,
        ",".join(run_spec.train_languages),
        ",".join(run_spec.eval_languages),
        run_spec.uses_macro_eval,
        train_args.seed,
        data_root,
        cache_root,
        output_dir,
    )
    logger.info(
        "Model configuration: model=%s tokenizer=%s max_seq_length=%s attn_implementation=%s dtype=%s",
        model_args.model_name_or_path,
        model_args.tokenizer_name or model_args.model_name_or_path,
        model_args.max_seq_length,
        model_args.attn_implementation,
        model_args.dtype,
    )
    logger.info(
        "Training arguments: epochs=%s train_batch=%s eval_batch=%s lr=%s warmup_ratio=%s "
        "eval_strategy=%s save_strategy=%s logging_strategy=%s load_best_model_at_end=%s metric_for_best_model=%s",
        train_args.num_train_epochs,
        train_args.per_device_train_batch_size,
        train_args.per_device_eval_batch_size,
        train_args.learning_rate,
        train_args.warmup_ratio,
        train_args.eval_strategy,
        train_args.save_strategy,
        train_args.logging_strategy,
        train_args.load_best_model_at_end,
        train_args.metric_for_best_model,
    )


def _log_dataset_summary(ner_samples: NerSamplesLoader, datasets: Mapping[str, Dataset], run_spec: RunSpec) -> None:
    for split in ner_samples.splits:
        samples_by_lang = ner_samples.samples_by_lang[split]
        logger.info(
            "Loaded SDJT %s split for %s: total_samples=%d languages=%d [%s]",
            split,
            run_spec.run_name,
            len(datasets[split]),
            len(samples_by_lang),
            _format_language_counts(samples_by_lang),
        )
    logger.info(
        "Resolved NER labels for %s: num_labels=%d labels=%s",
        run_spec.run_name,
        ner_samples.labeler.num_labels,
        ",".join(ner_samples.labeler.label_list),
    )


def init_dirs(p: Paths, run_name: str) -> tuple[Path, Path]:
    data_root = p.get_script_ctx_path("data", "split") / run_name
    if not data_root.exists():
        raise EnvironmentError(f"Split data not found at {data_root}. Run `./data resample {p.curr_context}` first.")
    cache_root = p.base.tmp / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    if not cache_root.exists():
        raise EnvironmentError(f"Unable to initialize cache dir at {cache_root}.")
    return data_root, cache_root


def compute_model_name(m_args: ModelArguments, d_args: DataArguments,
                       t_args: TrainingArguments, run_spec: RunSpec) -> str:
    return (f"{d_args.dataset_name}.{run_spec.run_name}.{m_args.short_name}"
            f".b{t_args.train_batch_size}.lr{t_args.learning_rate}.s{t_args.seed}")


def compute_output_dir(m_args: ModelArguments, d_args: DataArguments, t_args: TrainingArguments,
                       run_spec: RunSpec) -> Path:
    model_name = compute_model_name(m_args, d_args, t_args, run_spec)
    output_dir = paths.context / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


class MacroEvalTrainer(Trainer):
    def __init__(self, *args, eval_datasets_by_name: Optional[Mapping[str, Dataset]] = None, **kwargs):
        # noinspection PyArgumentList
        super().__init__(*args, **kwargs)
        self.eval_datasets_by_name = dict(eval_datasets_by_name or {})

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        if eval_dataset is not None or not self.eval_datasets_by_name:
            return super().evaluate(
                eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
            )

        aggregated_metrics: Dict[str, float] = {}
        macro_f1_values = []
        for lang, lang_dataset in self.eval_datasets_by_name.items():
            lang_metrics = super().evaluate(
                eval_dataset=lang_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=f"{metric_key_prefix}_{lang}",
            )
            aggregated_metrics.update(lang_metrics)
            lang_f1 = lang_metrics.get(f"{metric_key_prefix}_{lang}_f1")
            if lang_f1 is not None:
                macro_f1_values.append(float(lang_f1))

        if macro_f1_values:
            macro_f1 = sum(macro_f1_values) / len(macro_f1_values)
            aggregated_metrics[f"{metric_key_prefix}_macro_f1"] = macro_f1
            self.log({f"{metric_key_prefix}_macro_f1": macro_f1})
        return aggregated_metrics


def build_split_datasets(tokenizer, max_seq_length: int,
                         ner_samples: NerSamplesLoader, split: str) -> Dict[str, Dataset]:
    return {
        lang: NerDataset(tokenizer, max_seq_length, ner_samples.labeler, list(sentences))
        for lang, sentences in ner_samples.samples_by_lang[split].items()
    }


def build_eval_datasets(tokenizer, max_seq_length: int, ner_samples: NerSamplesLoader) -> Dict[str, Dataset]:
    return build_split_datasets(tokenizer, max_seq_length, ner_samples, "eval")


def load_model_and_tokenizer(model_args: ModelArguments, cache_root: Path, labeler):
    tokenizer_name = model_args.tokenizer_name or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_root)
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=cache_root,
        num_labels=labeler.num_labels,
        id2label=labeler.id2label,
        label2id=labeler.label2id,
    )
    return model, tokenizer


def main(data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments) -> None:
    run_names = available_run_names()
    attrs = data_args.attributes or {}
    run_name = str(attrs.get("run_name", "")).strip()
    if not run_name:
        raise ValueError(
            f"run_name attribute is required for SDJT NER resampling an d should be one of {run_names}!"
        )
    # base_seed = int(train_args.seed or train_args.seed or 2611)
    run_spec = resolve_run_spec_from_name(run_name)
    logger.info("Training SDJT NER run %s", run_spec.run_name)

    data_root, cache_root = init_dirs(paths, run_name)
    train_args.output_dir = str(compute_output_dir(model_args, data_args, train_args, run_spec))
    train_args.metric_for_best_model = run_spec.metric_name
    if not data_root.exists():
        raise FileNotFoundError(f"Run split not found at {data_root}. Run `./data split {paths.curr_context}` first.")

    languages = list(run_spec.train_languages)
    ner_samples = NerSamplesLoader(data_root, languages)
    metrics = TokenClassificationMetrics(id2label=ner_samples.labeler.id2label)
    _log_run_configuration(run_spec, data_root, cache_root, train_args.output_dir, model_args, train_args)
    model, tokenizer = load_model_and_tokenizer(model_args, cache_root, ner_samples.labeler)
    collator = DataCollatorForTokenClassification(tokenizer, padding="longest")
    datasets = ner_samples.create_split_datasets(tokenizer, model_args.max_seq_length)
    eval_datasets = build_eval_datasets(tokenizer, model_args.max_seq_length, ner_samples)
    _log_dataset_summary(ner_samples, datasets, run_spec)

    trainer_cls = MacroEvalTrainer if run_spec.uses_macro_eval else Trainer
    trainer_kwargs = {
        "model": model,
        "args": train_args,
        "train_dataset": datasets["train"],
        "eval_dataset": datasets["eval"],
        "data_collator": collator,
        "processing_class": tokenizer,
        "compute_metrics": metrics,
    }
    if run_spec.uses_macro_eval:
        trainer_kwargs["eval_datasets_by_name"] = eval_datasets
        logger.info(
            "Using %s with macro evaluation across languages: %s",
            trainer_cls.__name__,
            ",".join(sorted(eval_datasets.keys())),
        )
    else:
        target_lang = run_spec.eval_languages[0]
        trainer_kwargs["eval_dataset"] = eval_datasets[target_lang]
        logger.info(
            "Using %s with target-language evaluation: %s",
            trainer_cls.__name__,
            target_lang,
        )

    trainer = trainer_cls(**trainer_kwargs)
    logger.info("Starting training for %s", run_spec.run_name)
    train_result = trainer.train()
    trainer.save_model(train_args.output_dir)

    # noinspection PyTypeChecker
    state_path = Path(train_args.output_dir) / "trainer_state.json"
    trainer.state.save_to_json(str(state_path))
    logger.info("Saved trainer state to %s", state_path)
    logger.info(
        "Training complete for %s; global steps=%s, training_loss=%.6f",
        run_spec.run_name,
        train_result.global_step,
        train_result.training_loss,
    )

    logger.info("Starting evaluation for %s", run_spec.run_name)
    eval_metrics = trainer.evaluate()
    logger.info("Evaluation metrics for %s: %s", run_spec.run_name, eval_metrics)
