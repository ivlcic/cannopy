"""Generate token/BIO rows with a causal language model."""

from __future__ import annotations

import re
from typing import Sequence

import torch
from torch.utils.data import Dataset

from .ner import NerSample


def format_sentence(tokens: Sequence[str]) -> str:
    sentence = " ".join(tokens)
    sentence = re.sub(r"\s+([.,!?;:%)\]}])", r"\1", sentence)
    return re.sub(r"([([{])\s+", r"\1", sentence)


def format_generation_prompt(tokens: Sequence[str]) -> str:
    return f"BIO-NER:\n{format_sentence(tokens)}\n"


def format_generation_answer(tokens: Sequence[str], labels: Sequence[str]) -> str:
    if len(tokens) != len(labels):
        raise ValueError("NER tokens and labels must have the same length")
    width = max(map(len, tokens), default=0) + 2
    return "\n".join(f"{token:<{width}}{label}" for token, label in zip(tokens, labels, strict=True))


def encode_generation_prompt(tokenizer, tokens: Sequence[str], use_chat_template: bool) -> list[int]:
    prompt = format_generation_prompt(tokens)
    if use_chat_template:
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("Generative NER requires a tokenizer with a chat template")
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
    return tokenizer(prompt, add_special_tokens=False)["input_ids"]


def split_generation_tokens(tokenizer, tokens: Sequence[str], max_seq_length: int,
                            use_chat_template: bool,
                            labels: Sequence[str] | None = None) -> list[tuple[int, int]]:
    """Keep each prompt and its potential BIO answer within the token budget."""
    if labels is not None and len(tokens) != len(labels):
        raise ValueError("NER tokens and labels must have the same length")

    def fits(start: int, end: int) -> bool:
        prompt_length = len(encode_generation_prompt(
            tokenizer, tokens[start:end], use_chat_template,
        ))
        # Reserve room for each generated word, BIO label, and line break.
        answer_budget = max(16, (end - start) * 12)
        if labels is not None:
            answer = format_generation_answer(tokens[start:end], labels[start:end])
            answer_budget = max(
                answer_budget,
                len(tokenizer(answer, add_special_tokens=False)["input_ids"]) + 1,
            )
        return prompt_length + answer_budget <= max_seq_length

    chunks: list[tuple[int, int]] = []
    start = 0
    while start < len(tokens):
        lo, hi = start + 1, len(tokens)
        best = start
        while lo <= hi:
            mid = (lo + hi) // 2
            if fits(start, mid):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if best == start:
            raise ValueError(
                f"NER word {start}={tokens[start]!r} cannot fit max_seq_length={max_seq_length}"
            )
        chunks.append((start, best))
        start = best
    return chunks


class NerGenerationDataset(Dataset):
    def __init__(self, tokenizer, max_seq_length: int, samples: Sequence[NerSample],
                 use_chat_template: bool = True) -> None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Generative NER requires an EOS token")
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.samples: list[NerSample] = []
        self.use_chat_template = use_chat_template
        for sample in samples:
            for start, end in split_generation_tokens(
                tokenizer, sample.tokens, max_seq_length, use_chat_template, sample.labels,
            ):
                self.samples.append(NerSample(
                    tokens=sample.tokens[start:end],
                    labels=sample.labels[start:end],
                    corpus_name=sample.corpus_name,
                    doc_id=sample.doc_id,
                    sent_id=sample.sent_id,
                ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        sample = self.samples[index]
        prompt_ids = encode_generation_prompt(
            self.tokenizer, sample.tokens, self.use_chat_template,
        )
        answer = format_generation_answer(sample.tokens, sample.labels)
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + answer_ids + [self.tokenizer.eos_token_id]
        if len(input_ids) > self.max_seq_length:
            raise ValueError(
                f"NER sentence {index} needs {len(input_ids)} tokens, above "
                f"max_seq_length={self.max_seq_length}; increase the config limit"
            )
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt_ids) + answer_ids + [self.tokenizer.eos_token_id],
        }


class CausalNerCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(item["input_ids"]) for item in features)
        left_pad = self.tokenizer.padding_side == "left"
        padded = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            count = width - len(item["input_ids"])
            for name, pad_value in (
                ("input_ids", self.pad_token_id),
                ("attention_mask", 0),
                ("labels", -100),
            ):
                pad = [pad_value] * count
                padded[name].append(
                    pad + item[name] if left_pad else item[name] + pad
                )
        return {name: torch.tensor(values, dtype=torch.long) for name, values in padded.items()}


def parse_generation(text: str, tokens: Sequence[str], valid_labels: set[str]) -> list[str] | None:
    rows = text.strip().splitlines()
    if len(rows) != len(tokens):
        return None
    labels: list[str] = []
    for row, token in zip(rows, tokens, strict=True):
        columns = row.strip().rsplit(maxsplit=1)
        if len(columns) != 2 or columns[0] != token or columns[1] not in valid_labels:
            return None
        labels.append(columns[1])
    return labels


def generate_labels(model, tokenizer, tokens: Sequence[str], max_seq_length: int,
                    use_chat_template: bool, valid_labels: set[str]
                    ) -> tuple[list[str] | None, str]:
    prompt_ids = encode_generation_prompt(tokenizer, tokens, use_chat_template)
    if len(prompt_ids) >= max_seq_length:
        raise ValueError(f"NER prompt needs {len(prompt_ids)} tokens, above max_seq_length={max_seq_length}")
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    model.eval()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=min(max_seq_length - len(prompt_ids), max(16, len(tokens) * 12)),
            do_sample=False,
            pad_token_id=(
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            ),
        )
    response = tokenizer.decode(output_ids[0, len(prompt_ids):], skip_special_tokens=True).strip()
    return parse_generation(response, tokens, valid_labels), response


def generate_sentence_labels(model, tokenizer, tokens: Sequence[str], max_seq_length: int,
                             use_chat_template: bool, valid_labels: set[str]
                             ) -> tuple[list[str] | None, str]:
    labels: list[str] = []
    responses: list[str] = []
    for start, end in split_generation_tokens(
        tokenizer, tokens, max_seq_length, use_chat_template,
    ):
        chunk_labels, response = generate_labels(
            model, tokenizer, tokens[start:end], max_seq_length,
            use_chat_template, valid_labels,
        )
        responses.append(response)
        if chunk_labels is None:
            return None, " | ".join(responses)
        labels.extend(chunk_labels)
    return labels, " | ".join(responses)
