import csv
import torch
import logging

from pathlib import Path
from typing import Dict, List, Tuple

from torch.utils.data import Dataset
from transformers.utils import ExplicitEnum

logger = logging.getLogger('core.dataset')

Sentence = Tuple[List[str], List[str]]



class SubtokenLabelingStrategy(ExplicitEnum):
    """All the valid subtoken labeling strategies"""

    NEXT_NONE = "next_none"
    NEXT_FIRST = "next_first"
    NEXT_INSIDE = "next_inside"


class NerDataset(Dataset):

    def __init__(self, samples: List[Sentence], tokenizer, label2id: Dict[str, int], max_length: int,
                 subtoken_labeling_strategy: SubtokenLabelingStrategy = SubtokenLabelingStrategy.NEXT_NONE):
        self.samples = samples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.subtoken_labeling_strategy = subtoken_labeling_strategy

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens, labels = self.samples[idx]
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        word_ids = encoding.word_ids()
        label_ids: List[int] = []
        previous_word_id = None

        if self.subtoken_labeling_strategy == SubtokenLabelingStrategy.NEXT_NONE:
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != previous_word_id:  # Only label the first token of a given word.
                    label = labels[word_id]
                    label_ids.append(self.label2id.get(label, self.label2id["O"]))
                else:
                    label_ids.append(-100)
                previous_word_id = word_id
        elif self.subtoken_labeling_strategy == SubtokenLabelingStrategy.NEXT_INSIDE:
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                else:
                    label = labels[word_id]
                    if word_id != previous_word_id:
                        label_ids.append(self.label2id.get(label, self.label2id["O"]))
                    else:
                        # mark a continuation of a previous label on a SUB!!!-token as I-XYZ if it was B-XYZ
                        if label.startswith("B-"):
                            label = "I-" + label[2:]
                        label_ids.append(self.label2id.get(label, self.label2id["O"]))
                previous_word_id = word_id
        else:
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                else:
                    label = labels[word_id]
                    label_ids.append(self.label2id.get(label, self.label2id["O"]))

        encoding['labels'] = torch.tensor(label_ids, dtype=torch.long)
        encoding['input_ids'] = torch.tensor(encoding['input_ids'], dtype=torch.long)
        encoding['attention_mask'] = torch.tensor(encoding['attention_mask'], dtype=torch.long)
        return encoding


# noinspection PyMethodMayBeStatic
class NerSamplesLoader:

    def _load_split_file(self, path: Path) -> List[Sentence]:
        samples: List[Sentence] = []
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

    def _collect_labels(self, samples_by_lang: Dict[str, Dict[str, List[Sentence]]]) -> List[str]:
        labels = {'O'}
        for split in samples_by_lang:
            for sentences in samples_by_lang[split].values():
                for _, labs in sentences:
                    labels.update(labs)

        def _label_key(label: str) -> Tuple[str, str]:
            if label == 'O':
                return '', label
            if '-' in label:
                prefix, postfix = label.split('-', 1)
                return postfix, prefix
            return label, ''

        return sorted(labels, key=_label_key)

    def __init__(self, path: Path, languages: List[str]) -> None:
        self.samples_by_lang: Dict[str, Dict[str, List[Sentence]]] = {}
        self.path: Path = path
        self.splits: List[str] = ['train', 'eval', 'test']
        self.languages: List[str] = languages
        for split in self.splits:
            self.samples_by_lang[split]: Dict[str, List[Sentence]] = {}
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
        self.label2id = {label: idx for idx, label in enumerate(self.label_list)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def create_split_datasets(self, tokenizer, max_seq_length: int = 512) -> Dict[str, Dataset]:
        datasets: Dict[str, Dataset] = {}
        for split in self.splits:
            samples = [sample for sentences in self.samples_by_lang[split].values() for sample in sentences]
            dataset = NerDataset(samples, tokenizer, self.label2id, max_seq_length)
            logger.info(
                'Prepared %d samples across %d languages for %s',
                len(dataset),
                len(self.samples_by_lang[split]),
                split
            )
            datasets[split] = dataset
        return datasets
