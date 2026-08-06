import csv
import random
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import DefaultDict, Dict, List, Sequence, Tuple

from ..analyze.ner import aggregate_csv_file
from ..prepare.ner import Sentence, write_outputs
from ...app.args.data import DataArguments
from ...app.args.runtime import Paths

DEDUP_SPLIT_PRIORITY: Tuple[str, ...] = ('test', 'eval', 'train')
SPLIT_NAMES: Tuple[str, ...] = ('train', 'eval', 'test')

SplitSamples = Dict[str, Dict[str, List[Sentence]]]

logger: Logger
paths: Paths


@dataclass
class DedupCounts:
    before: int = 0
    removed: int = 0
    label_conflicts: int = 0


def _stable_int(text: str) -> int:
    value = 0
    for char in text:
        value = ((value * 131) + ord(char)) % 2_147_483_647
    return value


def _load_prepared_sentences(source_dir: Path) -> Dict[str, List[Sentence]]:
    aggregated: DefaultDict[str, List[Sentence]] = defaultdict(list)
    for csv_file in sorted(source_dir.glob('ner-*.csv')):
        stem: str = csv_file.stem
        if stem.startswith('ner_stats'):
            continue
        if '.' in stem:
            # skip already split files such as lang.train.csv
            continue
        lang = stem.replace('ner-', '')
        aggregate_csv_file(csv_file, lang, aggregated)
    return aggregated


def _split_language_data(aggregated: Dict[str, List[Sentence]], train_ratio: float, dev_ratio: float,
                         test_ratio: float, seed: int) -> Dict[str, DefaultDict[str, List[Sentence]]]:
    ratios_sum = train_ratio + dev_ratio + test_ratio
    if ratios_sum <= 0:
        ratios_sum = 1.0
        train_ratio, dev_ratio, test_ratio = 0.8, 0.1, 0.1
    splits: Dict[str, DefaultDict[str, List[Sentence]]] = {
        'train': defaultdict(list),
        'eval': defaultdict(list),
        'test': defaultdict(list),
    }
    for lang, sentences in aggregated.items():
        shuffled = list(sentences)
        rng = random.Random(seed + _stable_int(lang))
        rng.shuffle(shuffled)
        total = len(shuffled)
        train_n = int(total * train_ratio / ratios_sum)
        dev_n = int(total * dev_ratio / ratios_sum)
        test_n = total - train_n - dev_n
        splits['train'][lang] = shuffled[:train_n]
        splits['eval'][lang] = shuffled[train_n:train_n + dev_n]
        splits['test'][lang] = shuffled[train_n + dev_n:train_n + dev_n + test_n]
    return splits


def _normalized_sentence_key(sample: Sentence) -> str:
    text = ' '.join(sample.tokens)
    return unicodedata.normalize('NFKC', text).casefold()


def _has_aligned_tokens_and_labels(sample: Sentence) -> bool:
    return (
        bool(sample.tokens and sample.labels)
        and len(sample.tokens) == len(sample.labels)
    )


def deduplicate_corpora(source_corpora: SplitSamples) -> Tuple[SplitSamples, List[Dict[str, object]], List[Dict[str, object]]]:
    deduplicated: SplitSamples = {split_name: {} for split_name in SPLIT_NAMES}
    counts: Dict[Tuple[str, str, str], DedupCounts] = {}
    duplicate_rows: List[Dict[str, object]] = []
    languages = sorted({
        lang
        for split_samples in source_corpora.values()
        for lang in split_samples
    })

    for lang in languages:
        seen: Dict[str, Tuple[str, Sentence]] = {}
        for split_name in DEDUP_SPLIT_PRIORITY:
            kept: List[Sentence] = []
            for sample in source_corpora.get(split_name, {}).get(lang, []):
                if not _has_aligned_tokens_and_labels(sample):
                    kept.append(sample)
                    continue

                corpus_name = sample.corpus_name or 'unknown'
                count_key = (lang, split_name, corpus_name)
                corpus_counts = counts.setdefault(count_key, DedupCounts())
                corpus_counts.before += 1

                sentence_key = _normalized_sentence_key(sample)
                survivor = seen.get(sentence_key)
                if survivor is None:
                    seen[sentence_key] = (split_name, sample)
                    kept.append(sample)
                    continue

                survivor_split, survivor_sample = survivor
                labels_match = sample.labels == survivor_sample.labels
                corpus_counts.removed += 1
                if not labels_match:
                    corpus_counts.label_conflicts += 1
                removed_row = sample.to_csv_row()
                kept_row = survivor_sample.to_csv_row()
                duplicate_rows.append({
                    'language': lang,
                    'removed_split': split_name,
                    'removed_corpus_name': corpus_name,
                    'removed_doc_id': sample.doc_id,
                    'removed_sent_id': sample.sent_id,
                    'removed_sentence': removed_row['sentence'],
                    'removed_labels': removed_row['labels'],
                    'kept_split': survivor_split,
                    'kept_corpus_name': survivor_sample.corpus_name or 'unknown',
                    'kept_doc_id': survivor_sample.doc_id,
                    'kept_sent_id': survivor_sample.sent_id,
                    'kept_sentence': kept_row['sentence'],
                    'kept_labels': kept_row['labels'],
                    'labels_match': labels_match,
                })
            deduplicated[split_name][lang] = kept

    stats_rows: List[Dict[str, object]] = []
    for (lang, split_name, corpus_name), corpus_counts in sorted(counts.items()):
        stats_rows.append({
            'language': lang,
            'split': split_name,
            'corpus_name': corpus_name,
            'before': corpus_counts.before,
            'duplicates_removed': corpus_counts.removed,
            'after': corpus_counts.before - corpus_counts.removed,
            'label_conflicts': corpus_counts.label_conflicts,
        })
    return deduplicated, stats_rows, duplicate_rows


def write_dedup_reports(
    output_dir: Path,
    stats_rows: Sequence[Dict[str, object]],
    duplicate_rows: Sequence[Dict[str, object]],
) -> Tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / 'ner-dedup-stats.csv'
    duplicates_path = output_dir / 'ner-duplicates.csv'
    duplicates_data_path = output_dir / 'ner-duplicates-data.csv'
    stats_columns = [
        'language',
        'split',
        'corpus_name',
        'before',
        'duplicates_removed',
        'after',
        'label_conflicts',
    ]
    duplicate_columns = [
        'language',
        'removed_split',
        'removed_corpus_name',
        'removed_doc_id',
        'removed_sent_id',
        'kept_split',
        'kept_corpus_name',
        'kept_doc_id',
        'kept_sent_id',
        'labels_match',
    ]
    duplicate_data_columns = [
        'language',
        'labels_match',
        'removed_split',
        'removed_corpus_name',
        'removed_doc_id',
        'removed_sent_id',
        'removed_sentence',
        'removed_labels',
        'kept_split',
        'kept_corpus_name',
        'kept_doc_id',
        'kept_sent_id',
        'kept_sentence',
        'kept_labels',
    ]
    with stats_path.open('w', encoding='utf-8', newline='') as stats_file:
        writer = csv.DictWriter(stats_file, fieldnames=stats_columns)
        writer.writeheader()
        writer.writerows(stats_rows)
    with duplicates_path.open('w', encoding='utf-8', newline='') as duplicates_file:
        writer = csv.DictWriter(
            duplicates_file,
            fieldnames=duplicate_columns,
            extrasaction='ignore',
        )
        writer.writeheader()
        writer.writerows(duplicate_rows)
    with duplicates_data_path.open(
        'w',
        encoding='utf-8',
        newline='',
    ) as duplicates_data_file:
        writer = csv.DictWriter(
            duplicates_data_file,
            fieldnames=duplicate_data_columns,
        )
        writer.writeheader()
        writer.writerows(duplicate_rows)
    return stats_path, duplicates_path, duplicates_data_path


def main(data_args: DataArguments) -> None:
    logger.info('Splitting NER datasets...')

    source_dir = paths.get_ctx_path('prepare')
    target_dir = paths.context
    target_dir.mkdir(parents=True, exist_ok=True)

    aggregated = _load_prepared_sentences(source_dir)
    if not aggregated:
        logger.warning('No prepared NER data found in %s!', source_dir)
        return

    train_ratio = data_args.split.train
    dev_ratio = data_args.split.eval
    test_ratio = data_args.split.test
    seed = data_args.split.seed

    split_data = _split_language_data(aggregated, train_ratio, dev_ratio, test_ratio, seed)
    if data_args.split.dedup:
        split_data, stats_rows, duplicate_rows = deduplicate_corpora(split_data)
        stats_path, duplicates_path, duplicates_data_path = write_dedup_reports(
            paths.get_ctx_path('analyze'),
            stats_rows,
            duplicate_rows,
        )
        logger.info(
            'Removed %d duplicate NER samples, including %d label conflicts; '
            'wrote reports to %s, %s, and %s.',
            len(duplicate_rows),
            sum(int(row['label_conflicts']) for row in stats_rows),
            stats_path,
            duplicates_path,
            duplicates_data_path,
        )
    else:
        logger.info('NER sentence deduplication is disabled.')

    for split_name, sentences_by_lang in split_data.items():
        suffix = f'.{split_name}'
        write_outputs(target_dir, sentences_by_lang, suffix)

    logger.info(
        'Wrote split files (train/dev/test) to %s using seed=%s and ratios train=%.3f dev=%.3f test=%.3f.',
        target_dir, seed, train_ratio, dev_ratio, test_ratio
    )
