import csv
import random
from collections import defaultdict
from logging import Logger
from pathlib import Path
from typing import Any, DefaultDict, Dict, List

from ..prepare.ner import Sentence, _write_outputs
from ...app.args.data import DataArguments

logger: Logger
paths: Dict[str, Any]


def _load_prepared_sentences(source_dir: Path) -> Dict[str, List[Sentence]]:
    aggregated: DefaultDict[str, List[Sentence]] = defaultdict(list)
    for csv_file in sorted(source_dir.glob('ner-*.csv')):
        stem = csv_file.stem
        if stem.startswith('ner_stats'):
            continue
        if '.' in stem:
            # skip already split files such as lang.train.csv
            continue
        lang = stem.replace('ner-', '')
        with csv_file.open('r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            for row in reader:
                if len(row) < 2:
                    continue
                tokens = row[0].split(' ')
                labels = row[1].split(' ')
                aggregated[lang].append((tokens, labels))
    return aggregated


def _split_language_data(aggregated: Dict[str, List[Sentence]], train_ratio: float, dev_ratio: float,
                         test_ratio: float, seed: int) -> Dict[str, DefaultDict[str, List[Sentence]]]:
    ratios_sum = train_ratio + dev_ratio + test_ratio
    if ratios_sum <= 0:
        ratios_sum = 1.0
        train_ratio, dev_ratio, test_ratio = 0.8, 0.1, 0.1
    rng = random.Random(seed)
    splits: Dict[str, DefaultDict[str, List[Sentence]]] = {
        'train': defaultdict(list),
        'eval': defaultdict(list),
        'test': defaultdict(list),
    }
    for lang, sentences in aggregated.items():
        shuffled = list(sentences)
        rng.shuffle(shuffled)
        total = len(shuffled)
        train_n = int(total * train_ratio / ratios_sum)
        dev_n = int(total * dev_ratio / ratios_sum)
        test_n = total - train_n - dev_n
        splits['train'][lang] = shuffled[:train_n]
        splits['eval'][lang] = shuffled[train_n:train_n + dev_n]
        splits['test'][lang] = shuffled[train_n + dev_n:train_n + dev_n + test_n]
    return splits


def main(data_args: DataArguments) -> None:
    logger.info('Splitting NER datasets')

    source_dir = paths['base']['data'] / 'prepare'
    target_dir = paths['split']['data']
    target_dir.mkdir(parents=True, exist_ok=True)

    aggregated = _load_prepared_sentences(source_dir)
    if not aggregated:
        logger.warning('No prepared NER data found in %s', source_dir)
        return

    train_ratio = data_args.split['train']
    dev_ratio = data_args.split['eval']
    test_ratio = data_args.split['test']
    seed = data_args.split['seed']

    split_data = _split_language_data(aggregated, train_ratio, dev_ratio, test_ratio, seed)
    for split_name, sentences_by_lang in split_data.items():
        suffix = f'.{split_name}'
        _write_outputs(target_dir, sentences_by_lang, suffix)

    logger.info(
        'Wrote split files (train/dev/test) to %s using seed=%s and ratios train=%.3f dev=%.3f test=%.3f',
        target_dir, seed, train_ratio, dev_ratio, test_ratio
    )
