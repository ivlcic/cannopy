import logging
import torch

from typing import List, Optional
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .args.model import ModelArguments

logger = logging.getLogger("core.ner_tagger")


class EncoderTokenClassifier:

    def __init__(self, model_name_or_path: str, model_args: ModelArguments) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_kwargs = {}
        self.autocast = False

        if self.device == "cuda":
            if model_args.attn_implementation:
                self.model_kwargs["attn_implementation"] = model_args.attn_implementation
            if model_args.dtype:
                self.autocast = True
                self.model_kwargs["dtype"] = getattr(torch, model_args.dtype)

        tokenizer_name = model_args.tokenizer_name or model_args.model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path,
            **self.model_kwargs,
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info("Loaded token classifier model=%s on %s", model_name_or_path, self.device)

    @classmethod
    def align_subwords_to_words(cls, labels: List[str], word_ids: List[Optional[int]]) -> List[str]:
        aligned_labels: List[str] = []
        current_word = None
        current_label = None

        for label, word_id in zip(labels, word_ids):
            if word_id is None:
                continue

            if word_id != current_word:
                if current_word is not None:
                    aligned_labels.append(current_label)
                current_word = word_id
                current_label = label

        if current_word is not None:
            aligned_labels.append(current_label)

        return aligned_labels

    def classify_tokens(self, tokens: List[str]) -> List[str]:
        inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = inputs.to(self.device)
        with torch.no_grad():
            if self.autocast:
                with torch.autocast(device_type="cuda", dtype=self.model_kwargs["dtype"]):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

        predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()
        if isinstance(predictions, int):
            predictions = [predictions]
        labels = [self.model.config.id2label[pred] for pred in predictions]
        word_ids = inputs.word_ids()[1:-1]  # Exclude [CLS] and [SEP] tokens
        return self.align_subwords_to_words(labels[1:-1], word_ids)

    def classify_text(self, text) -> List[List[str]]:
        return [self.classify_tokens(sentence_tokens) for sentence_tokens in text]
