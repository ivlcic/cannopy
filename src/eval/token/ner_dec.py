"""Generate and score BIO sequences for the decoder NER test split."""

import json
from logging import Logger

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from ...app.args.data import DataArguments
from ...app.args.model import ModelArguments
from ...app.args.runtime import Paths
from ...app.metrics import TokenClassificationMetrics
from ...app.ner_decoder import generate_sentence_labels
from ...train.token.ner_dec import compute_run_name, init_dirs, load_samples


logger: Logger
paths: Paths


def main(data_args: DataArguments, model_args: ModelArguments,
         train_args: TrainingArguments) -> None:
    run_name = compute_run_name(model_args, data_args, train_args)
    train_dir = paths.get_script_ctx_path("train", "token") / run_name
    if not train_dir.exists():
        raise FileNotFoundError(f"Trained ner-dec checkpoint not found at {train_dir}")
    data_root, _ = init_dirs(paths)
    loader = load_samples(data_root, data_args.subdata_order)
    tokenizer = AutoTokenizer.from_pretrained(train_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = {"dtype": torch.float32} if device.type == "cpu" else {}
    if model_args.attn_implementation and device.type == "cuda":
        model_kwargs["attn_implementation"] = model_args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(train_dir, **model_kwargs).to(device)
    valid_labels = set(loader.labeler.label2id)
    references: list[list[str]] = []
    predictions: list[list[str]] = []
    invalid_count = 0
    for language_samples in loader.samples_by_lang["test"].values():
        for sample in language_samples:
            labels, raw_output = generate_sentence_labels(
                model, tokenizer, sample.tokens, model_args.max_seq_length,
                model_args.use_chat_template, valid_labels,
            )
            if labels is None:
                invalid_count += 1
                logger.warning("Invalid BIO output for %s: %r", sample.tokens, raw_output)
                labels = ["O"] * len(sample.tokens)
            references.append(sample.labels)
            predictions.append(labels)

    result = TokenClassificationMetrics(loader.labeler.id2label).seqeval.compute(
        predictions=predictions, references=references,
    )
    metrics = {
        "p": result["overall_precision"],
        "r": result["overall_recall"],
        "f1": result["overall_f1"],
        "acc": result["overall_accuracy"],
        "invalid_sequences": invalid_count,
        "sentences": len(references),
    }
    output_path = paths.context / f"{run_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Generative NER test metrics: %s; saved to %s", metrics, output_path)
