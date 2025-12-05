import csv
import random
import shutil

from collections import Counter, defaultdict
from logging import Logger
from pathlib import Path
from typing import DefaultDict, Dict, List, Any

from .parser import (Sentence, NerDatasetParser, CnecParser, BsnlpParser, SukParser, SetimesParser, Hr500kParser,
                     WannParser, NerUkParser)
from ....app.args.data import DataArguments

logger: Logger
paths: Dict[str, Any]


def _write_outputs(output_dir: Path, aggregated: Dict[str, List[Sentence]],
                   tags: List[str] | None = None, file_suffix: str = "") -> Dict[str, Any]:
    """
    Write per-language CSVs and gather stats: label counts, sentence counts, token counts.
    Returns a dict with keys: labels (Dict[str, Counter]), sentences (Counter), tokens (Counter), tags (List[str]).
    """
    label_stats: Dict[str, Counter] = {}
    sentence_stats: Counter = Counter()
    token_stats: Counter = Counter()
    for lang, sentences in aggregated.items():
        target = output_dir / f'{lang}{file_suffix}.csv'
        label_counter: Counter = Counter()
        sent_count = len(sentences)
        tok_count = 0
        with target.open('w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sentence', 'labels'])
            for tokens, labels in sentences:
                tok_count += len(tokens)
                writer.writerow([' '.join(tokens), ' '.join(labels)])
                for label in labels:
                    if label != 'O':
                        label_counter[label] += 1
        label_stats[lang] = label_counter
        sentence_stats[lang] = sent_count
        token_stats[lang] = tok_count

    all_tags = list(tags) if tags else sorted({tag for counter in label_stats.values() for tag in counter})

    return {'labels': label_stats, 'sentences': sentence_stats, 'tokens': token_stats, 'tags': all_tags}


def _write_stats(output_dir: Path, stats: Dict[str, Any], file_suffix: str = "") -> None:
    stats_path = output_dir / f'ner_stats{file_suffix}.csv'
    with stats_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['language', 'sentences', 'tokens', *stats['tags']])
        for lang in sorted(stats['labels'].keys()):
            counter = stats['labels'][lang]
            row = [lang, stats['sentences'][lang], stats['tokens'][lang]] + \
                  [counter.get(tag, 0) for tag in stats['tags']]
            writer.writerow(row)


def _format_stats_table(stats: Dict[str, Any]) -> str:
    header = ['lang', 'sent', 'tok', *stats['tags']]
    rows: List[List[str]] = []
    widths = [len(col) for col in header]

    for lang, counter in sorted(stats['labels'].items()):
        row = [
            lang,
            str(stats['sentences'][lang]),
            str(stats['tokens'][lang]),
            *[str(counter.get(tag, 0)) for tag in stats['tags']],
        ]
        rows.append(row)
        widths = [max(w, len(val)) for w, val in zip(widths, row)]

    def _fmt_row(_row: List[str]) -> str:
        parts = []
        for idx, val in enumerate(_row):
            align = '<' if idx == 0 else '>'
            parts.append(f'{val:{align}{widths[idx]}}')
        return '  '.join(parts)

    return '\n'.join([_fmt_row(header)] + [_fmt_row(r) for r in rows])


def _split_language_data(aggregated: Dict[str, List[Sentence]], train_ratio: float, dev_ratio: float,
                         test_ratio: float, seed: int) -> Dict[str, DefaultDict[str, List[Sentence]]]:
    ratios_sum = train_ratio + dev_ratio + test_ratio
    if ratios_sum <= 0:
        ratios_sum = 1.0
        train_ratio, dev_ratio, test_ratio = 0.8, 0.1, 0.1
    rng = random.Random(seed)
    splits: Dict[str, DefaultDict[str, List[Sentence]]] = {
        'train': defaultdict(list),
        'dev': defaultdict(list),
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
        splits['dev'][lang] = shuffled[train_n:train_n + dev_n]
        splits['test'][lang] = shuffled[train_n + dev_n:train_n + dev_n + test_n]
    return splits


def _format_split_stats_table(split_stats: Dict[str, Dict[str, Any]], tags: List[str]) -> str:
    header = ['lang', 'split', 'sent', 'tok', *tags]
    rows: List[List[str]] = []
    widths = [len(col) for col in header]

    for split_name in sorted(split_stats.keys()):
        stats = split_stats[split_name]
        for lang, counter in sorted(stats['labels'].items()):
            row = [
                lang,
                split_name,
                str(stats['sentences'][lang]),
                str(stats['tokens'][lang]),
                *[str(counter.get(tag, 0)) for tag in tags],
            ]
            rows.append(row)
            widths = [max(w, len(val)) for w, val in zip(widths, row)]

    def _fmt_row(_row: List[str]) -> str:
        parts = []
        for idx, val in enumerate(_row):
            align = '<' if idx < 2 else '>'
            parts.append(f'{val:{align}{widths[idx]}}')
        return '  '.join(parts)

    return '\n'.join([_fmt_row(header)] + [_fmt_row(r) for r in rows])


# noinspection
def main(data_args: DataArguments) -> None:
    logger.info('Preparing NER datasets')

    download_root = paths['base']['data'] / 'download' / 'ner'
    output_dir = paths['prepare']['data'] / 'ner'

    label_remap = data_args.label_remap
    parsers: List[NerDatasetParser] = [
        BsnlpParser(
            logger, download_root / 'bsnlp-2017-21' / 'bsnlp', label_remap.get('bsnlp', {})
        ),
        CnecParser(
            logger, download_root / 'CNEC_2.0_konkol' / 'CNEC_2.0_konkol', label_remap.get('cnec', {})
        ),
        SetimesParser(
            logger, download_root / 'setimes-sr.conll' / 'setimes-sr.conll', label_remap.get('setimes', {})
        ),
        Hr500kParser(
            logger, download_root / 'hr500k-1.0' / 'hr500k.conll', label_remap.get('hr500k', {})
        ),
        SukParser(
            logger, download_root / 'SUK.CoNLL-U' / 'SUK.CoNLL-U', label_remap.get('suk', {})
        ),
        NerUkParser(
            logger, download_root / 'ner-uk' / 'ner-uk', label_remap.get('ner-uk', {})
        ),
        WannParser(
            logger,
            download_root,
            {
                'bs-wann': 'bs',
                'mk-wann': 'mk',
                'sk-wann': 'sk',
                'sq-wann': 'sq',
            },
            label_remap.get('wann', {})
        ),
    ]

    aggregated: DefaultDict[str, List[Sentence]] = defaultdict(list)
    for _parser in parsers:
        parsed = _parser.parse()
        for lang, sentences in parsed.items():
            aggregated[lang].extend(sentences)
            logger.info('Parsed %d sentences for %s', len(sentences), lang)

    if not aggregated:
        logger.warning('No NER sentences parsed; nothing to write')
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = _write_outputs(output_dir, aggregated)
    _write_stats(output_dir, stats)

    logger.info('Wrote %d language files to %s', len(aggregated), output_dir)
    logger.info('Wrote stats to %s', output_dir / 'ner_stats.csv')
    logger.info('NER stats (matching ner_stats.csv columns):\n%s', _format_stats_table(stats))

    train_ratio = data_args.split["train"]
    dev_ratio = data_args.split["validation"]
    test_ratio = data_args.split["test"]
    seed = data_args.split["seed"]

    split_data = _split_language_data(aggregated, train_ratio, dev_ratio, test_ratio, seed)
    split_stats: Dict[str, Dict[str, Any]] = {}
    for split_name, sentences_by_lang in split_data.items():
        suffix = f'.{split_name}'
        split_stats[split_name] = _write_outputs(
            output_dir, sentences_by_lang, tags=stats['tags'], file_suffix=suffix
        )
        _write_stats(output_dir, split_stats[split_name], file_suffix=suffix)

    logger.info('Wrote split files (train/dev/test) to %s using seed=%s and ratios train=%.3f dev=%.3f test=%.3f',
                output_dir, seed, train_ratio, dev_ratio, test_ratio)
    logger.info('NER split stats (language/split, matching ner_stats.csv columns):\n%s',
                _format_split_stats_table(split_stats, stats['tags']))

    # clean up downloaded folders
    for child in download_root.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
