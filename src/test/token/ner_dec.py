"""Generate BIO labels for an unlabeled sentence."""

import json
import re
from logging import Logger

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from ...app.args.data import DataArguments
from ...app.args.model import ModelArguments
from ...app.args.runtime import Paths
from ...app.ner_decoder import generate_sentence_labels
from ...train.token.ner_dec import compute_run_name


logger: Logger
paths: Paths


def main(data_args: DataArguments, model_args: ModelArguments,
         train_args: TrainingArguments) -> None:
    text = str((data_args.attributes or {}).get("text") or "Janez Novak živi v Ljubljani.")
    matches = list(re.finditer(r"\w+|[^\w\s]", text, flags=re.UNICODE))
    if not matches:
        print("[]")
        return
    tokens = [match.group(0) for match in matches]
    run_name = compute_run_name(model_args, data_args, train_args)
    train_dir = paths.get_script_ctx_path("train", "token") / run_name
    if not train_dir.exists():
        raise FileNotFoundError(f"Trained ner-dec checkpoint not found at {train_dir}")
    tokenizer = AutoTokenizer.from_pretrained(train_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = {"dtype": torch.float32} if device.type == "cpu" else {}
    if model_args.attn_implementation and device.type == "cuda":
        model_kwargs["attn_implementation"] = model_args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(train_dir, **model_kwargs).to(device)
    valid_labels = set(getattr(model.config, "ner_labels", []))
    if not valid_labels:
        raise ValueError(f"Checkpoint {train_dir} does not store its NER label set")
    labels, raw_output = generate_sentence_labels(
        model, tokenizer, tokens, model_args.max_seq_length,
        model_args.use_chat_template, valid_labels,
    )
    if labels is None:
        raise ValueError(
            f"Model generated {raw_output!r}; expected {len(tokens)} token/BIO rows"
        )
    predictions = [
        {"word": match.group(0), "start": match.start(), "end": match.end(), "label": label}
        for match, label in zip(matches, labels, strict=True)
    ]
    logger.info("Classified %d words with %s", len(predictions), train_dir)
    print(json.dumps(predictions, ensure_ascii=False, indent=2))
