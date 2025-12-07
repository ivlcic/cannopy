import csv
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


def _write_outputs(output_dir: Path, aggregated: Dict[str, List[Sentence]], file_suffix: str = ""):
    """
    Write per-language CSVs.
    """
    for lang, sentences in aggregated.items():
        target = output_dir / f'ner-{lang}{file_suffix}.csv'
        label_counter: Counter = Counter()
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


# noinspection
def main(data_args: DataArguments) -> None:
    logger.info('Preparing NER datasets')

    download_root = paths['base']['data'] / 'download' / 'ner'
    output_dir = paths['prepare']['data']

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
    _write_outputs(output_dir, aggregated)

    logger.info('Wrote %d language files to %s', len(aggregated), output_dir)

    # clean up download folder
    shutil.rmtree(download_root, ignore_errors=True)
