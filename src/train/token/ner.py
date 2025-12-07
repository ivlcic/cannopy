from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                          TrainingArguments)

from ...app.args.model import ModelArguments
from ...app.args.data import DataArguments

Sentence = Tuple[List[str], List[str]]

logger: Logger
paths: Dict[str, Any]


def _load_split_file(path: Path) -> List[Sentence]:
    samples: List[Sentence] = []
    if not path.exists():
        return samples
    with path.open('r', encoding='utf-8') as f:
        next(f, None)  # header
        for line in f:
            parts = line.rstrip('\n').split(',')
            if len(parts) < 2:
                continue
            tokens = parts[0].split(' ')
            labels = parts[1].split(' ')
            samples.append((tokens, labels))
    return samples


def _collect_labels(samples_by_lang: Dict[str, Dict[str, List[Sentence]]]) -> List[str]:
    labels = {'O'}
    for split in samples_by_lang:
        for sentences in samples_by_lang[split].values():
            for _, labs in sentences:
                labels.update(labs)
    return sorted(labels)


class NerDataset(Dataset):

    def __init__(self, samples: List[Sentence], tokenizer, label2id: Dict[str, int], max_length: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

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
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            else:
                label_ids.append(self.label2id.get(labels[word_id], self.label2id['O']))

        encoding['labels'] = torch.tensor(label_ids, dtype=torch.long)
        encoding['input_ids'] = torch.tensor(encoding['input_ids'], dtype=torch.long)
        encoding['attention_mask'] = torch.tensor(encoding['attention_mask'], dtype=torch.long)
        return encoding


def main(data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments) -> None:
    logger.info('Training NER')

    data_root = paths['base']['data'] / 'split'
    if not data_root.exists():
        logger.error('Split data not found at %s. Run `./data split ner` first.', data_root)
        return

    languages = data_args.subdata_order or []
    if not languages:
        languages = [p.stem.split('.')[0] for p in data_root.glob('ner-*.train.csv')]

    samples_by_lang: Dict[str, Dict[str, List[Sentence]]] = {}
    for split in ['train', 'eval']:
        samples_by_lang[split]: Dict[str, List[Sentence]] = {}
        for lang in languages:
            file_path = data_root / f'ner-{lang}.{split}.csv'
            lang_samples = _load_split_file(file_path)
            if not lang_samples:
                logger.warning('No %s samples found for language %s at %s', split, lang, file_path)
                continue
            logger.info('Loaded %s %d samples for %s', split, len(lang_samples), lang)
            samples_by_lang[split][lang] = lang_samples

    for split in ['train', 'eval']:
        if not samples_by_lang[split]:
            logger.error('No %s samples loaded from %s', split, data_root)
            return

    label_list = _collect_labels(samples_by_lang)
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}

    tokenizer_name = model_args.tokenizer_name or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    collator = DataCollatorForTokenClassification(tokenizer, padding='longest')
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    datasets: Dict[str, Dataset] = {}
    dataloaders: Dict[str, DataLoader] = {}
    for split in ['train', 'eval']:
        samples = [sample for sentences in samples_by_lang[split].values() for sample in sentences]
        dataset = NerDataset(samples, tokenizer, label2id, model_args.max_seq_length)
        loader = DataLoader(
            dataset,
            batch_size=train_args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collator,
        )
        logger.info(
            'Prepared %d batches (%d samples) across %d languages for training',
            len(loader),
            len(dataset),
            len(samples_by_lang[split]),
        )
        datasets[split] = dataset
        dataloaders[split] = loader

    # This is where a training loop or Trainer would be invoked; for now we only prepare loaders/model.
    return
