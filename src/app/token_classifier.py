import re
import logging
import torch

from typing import List, Optional, Tuple, Dict, Any
from transformers import AutoModelForTokenClassification, AutoTokenizer, BatchEncoding

from .args.model import ModelArguments

logger = logging.getLogger("core.token_classifier")

WordList = List[Dict[str, Any]]


class EncoderTokenClassifier:
    WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

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

        tokenizer_name = model_args.tokenizer_name or model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path,
            **self.model_kwargs,
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info("Loaded token classifier model=%s on %s", model_name_or_path, self.device)

    def _predict(self, inputs: BatchEncoding) -> Tuple[List[int], List[str]]:
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
        return predictions, labels

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

    @staticmethod
    def _normalize_label(label: Optional[str], none_label: Optional[str] = None) -> Optional[str]:
        if label is None:
            return None
        if none_label is not None and label == none_label:
            return None
        return label

    def classify_tokens(self, tokens: List[str]) -> List[str]:
        inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        predictions, labels = self._predict(inputs)
        word_ids = inputs.word_ids(batch_index=0)
        return self.align_subwords_to_words(labels, word_ids)

    @classmethod
    def _split_with_spans(cls, text: str) -> List[Tuple[str, int, int]]:
        return [(m.group(0), m.start(), m.end()) for m in cls.WORD_RE.finditer(text)]

    @classmethod
    def _split_sentences(cls, text: str) -> List[str]:
        if not text:
            return []
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p for p in parts if p]

    def _max_content_len(self) -> int:
        max_len = self.tokenizer.model_max_length
        if max_len is None or max_len <= 0 or max_len > 100000:
            max_len = 512
        special = self.tokenizer.num_special_tokens_to_add(pair=False)
        return max(8, max_len - special)

    def _chunk_sentences(self, sentences: List[str]) -> List[str]:
        if not sentences:
            return []
        max_len = self._max_content_len()
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        def tok_len(s: str) -> int:
            return len(self.tokenizer.encode(s, add_special_tokens=False))

        for sent in sentences:
            sent_len = tok_len(sent)
            if sent_len > max_len:
                if current:
                    chunks.append(" ".join(current))
                    current = []
                    current_len = 0
                # hard split long sentence by tokens
                ids = self.tokenizer.encode(sent, add_special_tokens=False)
                for i in range(0, len(ids), max_len):
                    part_ids = ids[i:i + max_len]
                    chunks.append(self.tokenizer.decode(part_ids, skip_special_tokens=True))
                continue
            if current_len + sent_len > max_len and current:
                chunks.append(" ".join(current))
                current = [sent]
                current_len = sent_len
            else:
                current.append(sent)
                current_len += sent_len

        if current:
            chunks.append(" ".join(current))
        return chunks

    @classmethod
    def map_by_offsets(cls, text: str, tokens: List[str], labels: List[str], offsets: List[Tuple[int, int]],
                       none_label: Optional[str] = None) -> WordList:
        """
        Returns a list of word-level items:
          { "text": <substring>, "start": i, "end": j, "tokens": [...], "labels": [...] }
        """
        words = cls._split_with_spans(text)

        # init per-word buckets
        out: WordList = [
            {
                "word": w,
                "span": (ws, we),
                # "token_indices": [],
                # "tokens": [],
                "labels": []
            }
            for (w, ws, we) in words
        ]

        def normalize_label(label: Optional[str]) -> Optional[str]:
            if label is None:
                return None
            if none_label is not None and label == none_label:
                return None
            return label

        wi = 0  # current word index
        for ti, (tok, lab, (ts, te)) in enumerate(zip(tokens, labels, offsets)):
            if (ts, te) == (0, 0) or ts == te:
                continue  # specials/pad

            # advance word pointer until a word ends and after a token starts
            while wi < len(out) and out[wi]["span"][1] <= ts:
                wi += 1

            # token may overlap multiple words (rare), so walk forward while overlapping
            wj = wi
            while wj < len(out):
                ws, we = out[wj]["span"]
                if ws >= te:
                    break  # no more overlaps

                # overlap condition
                if ts < we and te > ws:
                    n_label = normalize_label(lab)
                    # out[wj]["token_indices"].append(ti)
                    # out[wj]["tokens"].append(tok)
                    if n_label is not None:
                        out[wj]["labels"].append(n_label)

                wj += 1

        return out

    def classify_sentence(self, sentence: str, none_label: Optional[str] = None) -> WordList:
        words = self._split_with_spans(sentence)
        if not words:
            return []

        word_texts = [word for word, _, _ in words]
        inputs = self.tokenizer(
            word_texts,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
        )
        predictions, labels = self._predict(inputs)
        word_ids = inputs.word_ids(batch_index=0)
        word_labels = self.align_subwords_to_words(labels, word_ids)

        result: WordList = []
        for (word, start, end), label in zip(words, word_labels):
            normalized = self._normalize_label(label, none_label=none_label)
            result.append({
                "word": word,
                "span": (start, end),
                "labels": [] if normalized is None else [normalized],
            })
        return result

    def classify_sentences(self, sentences: List[str], none_label: Optional[str] = None) -> List[WordList]:
        result: List[WordList] = []
        for sentence in sentences:
            result.append(self.classify_sentence(sentence, none_label=none_label))
        return result

    def classify_text(self, text: str, none_label: Optional[str] = None) -> WordList:
        sentences = self._split_sentences(text)
        result: WordList = []
        [result.extend(x) for x in self.classify_sentences(sentences, none_label=none_label)]
        return result

    def count_labels(self, text: str, none_label: Optional[str] = None) -> int:
        count = 0
        for x in self.classify_text(text, none_label=none_label):
            if x["labels"]:
                count += len(set(x["labels"]))
        return count
