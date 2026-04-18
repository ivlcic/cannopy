import csv
import json
import logging

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch

from torch.utils.data import Dataset
from transformers.utils import ExplicitEnum

from app.labeler import MulticlassLabeler, MultilabelLabeler

logger = logging.getLogger('core.dataset')

TextSample = Tuple[List[str], List[str]]
SequenceSample = Tuple[str, List[str]]


class SubtokenLabelingStrategy(ExplicitEnum):
    """All the valid subtoken labeling strategies"""
    NEXT_NONE = "next_none"
    NEXT_FIRST = "next_first"
    NEXT_INSIDE = "next_inside"


class ClassificationDataset(Dataset):
    def __init__(self, tokenizer, max_seq_len: int, labeler: MulticlassLabeler):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.labeler = labeler


class NerDataset(ClassificationDataset):

    def __init__(self, tokenizer, max_seq_len: int, labeler: MulticlassLabeler, samples: List[TextSample],
                 subtoken_labeling_strategy: SubtokenLabelingStrategy = SubtokenLabelingStrategy.NEXT_NONE):
        super().__init__(tokenizer, max_seq_len, labeler)
        self.samples = samples
        self.subtoken_labeling_strategy = subtoken_labeling_strategy

    def _encode_label(self, label: str) -> int:
        if label in self.labeler.label2id:
            return self.labeler.encode(label)
        if self.labeler.default_label is not None:
            return self.labeler.encode(self.labeler.default_label)
        raise KeyError(f'Unknown label without configured default label: {label}')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens, labels = self.samples[idx]
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_attention_mask=True,
        )
        word_ids: List[Optional[int]] = encoding.word_ids()
        label_ids: List[int] = []
        previous_word_id = None

        if self.subtoken_labeling_strategy == SubtokenLabelingStrategy.NEXT_NONE:
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                # Only label the first token of a given word.
                elif word_id != previous_word_id and word_id is not None:  #
                    label: str = labels[word_id]
                    label_ids.append(self._encode_label(label))
                else:
                    label_ids.append(-100)
                previous_word_id = word_id
        elif self.subtoken_labeling_strategy == SubtokenLabelingStrategy.NEXT_INSIDE:
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                else:
                    label: str = labels[word_id]
                    if word_id != previous_word_id:
                        label_ids.append(self._encode_label(label))
                    else:
                        # mark a continuation of a previous label on a SUB!!!-token as I-XYZ if it was B-XYZ
                        if label.startswith("B-"):
                            label = "I-" + label[2:]
                        label_ids.append(self._encode_label(label))
                previous_word_id = word_id
        else:
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                else:
                    label: str = labels[word_id]
                    label_ids.append(self._encode_label(label))

        encoding['labels'] = torch.tensor(label_ids, dtype=torch.long)
        encoding['input_ids'] = torch.tensor(encoding['input_ids'], dtype=torch.long)
        encoding['attention_mask'] = torch.tensor(encoding['attention_mask'], dtype=torch.long)
        return encoding


class MultilabelSequenceDataset(ClassificationDataset):

    def __init__(self, tokenizer, max_seq_len: int, labeler: MultilabelLabeler, samples: List[SequenceSample]):
        super().__init__(tokenizer, max_seq_len, labeler)
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text, labels = self.samples[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_len,
            return_attention_mask=True,
        )
        encoding['labels'] = torch.tensor(self.labeler.encode(labels), dtype=torch.float32)
        encoding['input_ids'] = torch.tensor(encoding['input_ids'], dtype=torch.long)
        encoding['attention_mask'] = torch.tensor(encoding['attention_mask'], dtype=torch.long)
        return encoding


# noinspection PyMethodMayBeStatic
class NerSamplesLoader:

    def _load_split_file(self, path: Path) -> List[TextSample]:
        samples: List[TextSample] = []
        if not path.exists():
            return samples
        with path.open('r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            for row in reader:
                if len(row) < 2:
                    continue
                tokens = row[0].split(' ')
                labels = row[1].split(' ')
                samples.append((tokens, labels))
        return samples

    def _collect_labels(self, samples_by_lang: Dict[str, Dict[str, List[TextSample]]]) -> List[str]:
        labels = {'O'}
        for split in samples_by_lang:
            for sentences in samples_by_lang[split].values():
                for _, labs in sentences:
                    labels.update(labs)
        return list(labels)

    def __init__(self, path: Path, languages: List[str]) -> None:
        def sort_ner_label(label: str) -> Tuple[str, str]:
            if label == 'O':
                return '', label
            if '-' in label:
                prefix, postfix = label.split('-', 1)
                return postfix, prefix
            return label, ''

        self.samples_by_lang: Dict[str, Dict[str, List[TextSample]]] = {}
        self.path: Path = path
        self.splits: List[str] = ['train', 'eval', 'test']
        self.languages: List[str] = languages
        for split in self.splits:
            self.samples_by_lang[split]: Dict[str, List[TextSample]] = {}
            for lang in languages:
                file_path = path / f'ner-{lang}.{split}.csv'
                lang_samples = self._load_split_file(file_path)
                if not lang_samples:
                    logger.warning('No %s samples found for language %s at %s', split, lang, file_path)
                    continue
                logger.info('Loaded %s %d samples for %s', split, len(lang_samples), lang)
                self.samples_by_lang[split][lang] = lang_samples

        for split in self.splits:
            if not self.samples_by_lang[split]:
                raise ValueError(f'No {split} samples loaded from {path}')

        self.label_list = self._collect_labels(self.samples_by_lang)
        self.labeler = MulticlassLabeler(
            self.label_list,
            default_label='O',
            sorter=sort_ner_label,
        )

    def create_split_datasets(self, tokenizer, max_seq_length: int = 512) -> Dict[str, Dataset]:
        datasets: Dict[str, Dataset] = {}
        for split in self.splits:
            samples = [sample for sentences in self.samples_by_lang[split].values() for sample in sentences]
            dataset = NerDataset(tokenizer, max_seq_length, self.labeler, samples)
            logger.info(
                'Prepared %d samples across %d languages for %s',
                len(dataset),
                len(self.samples_by_lang[split]),
                split
            )
            datasets[split] = dataset
        return datasets


class NewsmonSamplesLoader:

    @staticmethod
    def _sample_text(sample: Dict[str, Any]) -> str:
        text = sample.get('text', '')
        if text:
            return str(text)
        title = sample.get('title', {}).get('text', '') or ''
        body = sample.get('body', {}).get('text', '') or ''
        if title and body:
            return f'{title}\n\n{body}'
        return title or body

    def _load_split_file(self, path: Path) -> List[SequenceSample]:
        samples: List[SequenceSample] = []
        if not path.exists():
            return samples
        with path.open('r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'Malformed JSON in {path} line {line_no}') from exc
                text = self._sample_text(sample)
                labels = sample.get('label', []) or []
                if not text or not labels:
                    continue
                samples.append((text, labels))
        return samples

    @staticmethod
    def _collect_labels(samples: List[SequenceSample]) -> List[str]:
        labels = set()
        for _, sample_labels in samples:
            labels.update(sample_labels)
        return list(labels)

    def _filter_unknown_labels(self, split: str) -> None:
        filtered_samples: List[SequenceSample] = []
        dropped_labels = 0
        skipped_samples = 0
        known_labels = set(self.labeler.classes)
        for text, labels in self.samples_by_split[split]:
            filtered = [label for label in labels if label in known_labels]
            dropped_labels += len(labels) - len(filtered)
            if not filtered:
                skipped_samples += 1
                continue
            filtered_samples.append((text, filtered))
        self.samples_by_split[split] = filtered_samples
        if dropped_labels > 0 or skipped_samples > 0:
            logger.warning(
                'Filtered %d unseen labels and skipped %d %s samples with no train-known labels left',
                dropped_labels,
                skipped_samples,
                split,
            )

    def __init__(self, path: Path, subset: str) -> None:
        self.path = path
        self.splits = ['train', 'eval', 'test']
        self.subset = subset
        self.samples_by_split: Dict[str, List[SequenceSample]] = {}

        for split in self.splits:
            file_path = path / f'{subset}.{split}.jsonl'
            split_samples = self._load_split_file(file_path)
            if not split_samples:
                logger.warning('No %s samples found for subset %s at %s', split, subset, file_path)
                continue
            logger.info('Loaded %s %d samples for %s', split, len(split_samples), subset)
            self.samples_by_split[split] = split_samples

        for split in self.splits:
            if split not in self.samples_by_split:
                raise ValueError(f'No {split} samples loaded from {path} for subset {subset}')

        self.label_list = self._collect_labels(self.samples_by_split['train'])
        self.labeler = MultilabelLabeler(self.label_list)
        for split in self.splits:
            if split != 'train':
                self._filter_unknown_labels(split)

    def create_split_datasets(self, tokenizer, max_seq_length: int = 512) -> Dict[str, Dataset]:
        datasets: Dict[str, Dataset] = {}
        for split in self.splits:
            dataset = MultilabelSequenceDataset(
                tokenizer,
                max_seq_length,
                self.labeler,
                self.samples_by_split[split]
            )
            logger.info('Prepared %d samples for %s split', len(dataset), split)
            datasets[split] = dataset
        return datasets
