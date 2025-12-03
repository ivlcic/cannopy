from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, DefaultDict, Dict, Iterable, List, Optional, Tuple, Any

from syntok.segmenter import process as syntok_process

from ...app.args.data import DataArguments

Sentence = Tuple[List[str], List[str]]

LABEL_RE = re.compile(r'([BIO])[-_]?(.+)', re.IGNORECASE)


# noinspection PyMethodMayBeStatic
class NerDatasetParser:

    def __init__(self, root: Path, label_remap: Dict[Any, Any]):
        self.root = root
        self.label_remap = label_remap

    def parse(self) -> Dict[str, List[Sentence]]:
        raise NotImplementedError

    def _normalize_label(self, raw: str) -> str:
        raw = raw.strip()
        if raw.startswith('NER='):
            raw = raw.split('=', 1)[1]
        if raw.lower() == 'o':
            return 'O'
        match = LABEL_RE.match(raw)
        if match:
            return f'{match.group(1)}-{match.group(2)}'
        return raw

    def _map_label(self, token: str, label: str) -> str:
        if self.label_remap:
            return self.label_remap.get(label, label)
        return label


# noinspection PyGlobalUndefined
class ConllDatasetParser(NerDatasetParser):

    def _iter_sources(self) -> Iterable[Tuple[Path, str, int, Callable[[List[str]], str]]]:
        raise NotImplementedError

    def _parse_conll_file(self, path: Path, token_idx: int, label_selector,
                          keep_comment: bool = False) -> Tuple[List[Sentence], bool]:
        sentences: List[Sentence] = []
        tokens: List[str] = []
        labels: List[str] = []
        file_has_labels = False

        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                if tokens:
                    sentences.append((tokens, labels))
                    tokens, labels = [], []
                continue
            if line.startswith('#') and not keep_comment:
                continue
            parts = line.split('\t')
            if len(parts) <= token_idx:
                continue
            token = parts[token_idx]
            label = self._map_label(token, self._normalize_label(label_selector(parts)))
            if label != 'O':
                file_has_labels = True
            tokens.append(token)
            labels.append(label)

        if tokens:
            sentences.append((tokens, labels))

        return sentences, file_has_labels

    def parse(self) -> Dict[str, List[Sentence]]:
        global logger
        output: DefaultDict[str, List[Sentence]] = defaultdict(list)
        for path, lang, token_idx, label_selector in self._iter_sources():
            if not path.exists():
                continue
            sentences, has_labels = self._parse_conll_file(path, token_idx, label_selector)
            if not has_labels:
                continue
            output[lang].extend(sentences)
            logger.info('%s: %s -> %d sentences', lang, path.name, len(sentences))
        return output


class CnecParser(ConllDatasetParser):
    def _iter_sources(self) -> Iterable[Tuple[Path, str, int, Callable[[List[str]], str]]]:
        if not self.root.exists():
            return []
        logger.info('CNEC: scanning %s', self.root)
        files = ['train.conll', 'dtest.conll', 'etest.conll']
        return [
            (self.root / fname, 'cs', 0, lambda parts: parts[-1])
            for fname in files
        ]


class SetimesParser(ConllDatasetParser):
    def _iter_sources(self) -> Iterable[Tuple[Path, str, int, Callable[[List[str]], str]]]:
        path = self.root / 'set.sr.conll'
        if not path.exists():
            return []
        logger.info('SETimes: %s', path)
        label_idx = 10
        return [(path, 'sr', 1, lambda parts: parts[label_idx] if len(parts) > label_idx else 'O')]


class Hr500kParser(ConllDatasetParser):
    def _iter_sources(self) -> Iterable[Tuple[Path, str, int, Callable[[List[str]], str]]]:
        path = self.root / 'hr500k.conll' / 'hr500k.conll'
        if not path.exists():
            return []
        logger.info('hr500k: %s', path)
        label_idx = 10
        return [(path, 'hr', 1, lambda parts: parts[label_idx] if len(parts) > label_idx else 'O')]


class SukParser(ConllDatasetParser):
    def _iter_sources(self) -> Iterable[Tuple[Path, str, int, Callable[[List[str]], str]]]:
        if not self.root.exists():
            return []
        label_idx = 9
        ds_paths = []
        for path in sorted(self.root.glob('*.ud.conllu')):
            stem = path.stem
            if stem.startswith('ssj500k-tag') or stem.startswith('ambiga'):
                continue
            ds_paths.append((path, 'sl', 1, lambda parts, idx=label_idx: parts[idx] if len(parts) > idx else 'O'))
        return ds_paths


# noinspection PyGlobalUndefined
class WannParser(NerDatasetParser):
    def __init__(self, root: Path, mapping: Dict[str, str], label_remap: Dict[str, str]):
        NerDatasetParser.__init__(self, root, label_remap)
        self.base = root
        self.mapping = mapping

    def parse(self) -> Dict[str, List[Sentence]]:
        global logger
        output: DefaultDict[str, List[Sentence]] = defaultdict(list)
        for folder, lang in self.mapping.items():
            dataset_dir = self.base / folder
            if not dataset_dir.exists():
                continue
            for split in ('train', 'dev', 'test', 'extra'):
                split_path = dataset_dir / split
                if not split_path.exists():
                    continue
                sentences = self._parse_split(split_path)
                output[lang].extend(sentences)
                logger.info('WANN %s %s: %d sentences', lang, split_path.name, len(sentences))
        return output

    def _parse_split(self, path: Path) -> List[Sentence]:
        sentences: List[Sentence] = []
        tokens: List[str] = []
        labels: List[str] = []
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append((tokens, labels))
                    tokens, labels = [], []
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            token_raw, label_raw = parts[0], parts[1]
            token = token_raw.split(':', 1)[1] if ':' in token_raw else token_raw
            tokens.append(token)
            labels.append(self._map_label(token, self._normalize_label(label_raw)))
        if tokens:
            sentences.append((tokens, labels))
        return sentences


# noinspection PyMethodMayBeStatic, PyGlobalUndefined
class BsnlpParser(NerDatasetParser):

    def __init__(self, root: Path, label_remap: Dict[Any, Any]):
        NerDatasetParser.__init__(self, root, label_remap)
        self.raw_root = root / 'raw'
        self.ann_root = root / 'annotated'

    def _map_label(self, token: str, label: str) -> str:
        label = super()._map_label(token, label)
        if '@' in token:
            return 'O'
        return label

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        tokens: List[str] = []
        for paragraph in syntok_process(text):
            for sentence in paragraph:
                tokens.extend([tok.value for tok in sentence])
        return tokens

    def _parse_annotation_file(self, path: Path) -> Tuple[str, List[Tuple[List[str], str]]]:
        lines = path.read_text(encoding='utf-8').splitlines()
        if not lines:
            return '', []
        doc_id = lines[0].strip()
        entities: List[Tuple[List[str], str]] = []
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            surface, ent_type = parts[0].strip(), parts[2].strip()
            tokens = self._tokenize(surface)
            if not tokens:
                continue
            entities.append((tokens, ent_type.upper()))
        return doc_id, entities

    def _load_annotations(self) -> Dict[Tuple[str, str, str], List[Tuple[List[str], str]]]:
        index: Dict[Tuple[str, str, str], List[Tuple[List[str], str]]] = {}
        for topic_dir in sorted(self.ann_root.iterdir()):
            if not topic_dir.is_dir():
                continue
            for lang_dir in sorted(topic_dir.iterdir()):
                if not lang_dir.is_dir():
                    continue
                lang = lang_dir.name
                for ann_file in lang_dir.glob('*.out'):
                    doc_id, entities = self._parse_annotation_file(ann_file)
                    if doc_id and entities:
                        index.setdefault((topic_dir.name, lang, doc_id), []).extend(entities)
        return index

    def _process_doc(
        self,
        raw_file: Path,
        annos: Dict[Tuple[str, str, str], List[Tuple[List[str], str]]],
        topic: str,
        lang: str
    ) -> Tuple[str, List[Sentence]]:
        global logger
        lines = raw_file.read_text(encoding='utf-8').splitlines()
        if len(lines) < 5:
            return '', []
        doc_id = lines[0].strip()
        entities = annos.get((topic, lang, doc_id), [])
        if not entities:
            return doc_id, []

        text = '\n'.join(lines[4:])
        sentences: List[Sentence] = []
        for paragraph in syntok_process(text):
            for sentence in paragraph:
                tokens = [tok.value for tok in sentence]
                labels = ['O'] * len(tokens)
                lower_tokens = [t.lower() for t in tokens]
                for ent_tokens, ent_type in entities:
                    if not ent_tokens:
                        continue
                    pattern = [t.lower() for t in ent_tokens]
                    idx = 0
                    while idx <= len(tokens) - len(pattern):
                        if all(label == 'O' for label in labels[idx:idx + len(pattern)]) and \
                                lower_tokens[idx:idx + len(pattern)] == pattern:
                            labels[idx] = self._map_label(tokens[idx], self._normalize_label(f'B-{ent_type}'))
                            for j in range(1, len(pattern)):
                                labels[idx + j] = self._map_label(
                                    tokens[idx + j], self._normalize_label(f'I-{ent_type}')
                                )
                            idx += len(pattern)
                        else:
                            idx += 1
                if tokens:
                    sentences.append((tokens, labels))
        return doc_id, sentences

    def parse(self) -> Dict[str, List[Sentence]]:
        global logger
        output: DefaultDict[str, List[Sentence]] = defaultdict(list)
        if not self.raw_root.exists() or not self.ann_root.exists():
            return output

        annos = self._load_annotations()
        topic_counts: Dict[str, int] = defaultdict(int)
        for topic_dir in sorted(self.raw_root.iterdir()):
            if not topic_dir.is_dir():
                continue
            for lang_dir in sorted(topic_dir.iterdir()):
                if not lang_dir.is_dir():
                    continue
                lang = lang_dir.name
                for raw_file in lang_dir.glob('*.txt'):
                    doc_id, sentences = self._process_doc(raw_file, annos, topic_dir.name, lang)
                    if doc_id and sentences:
                        output[lang].extend(sentences)
                        topic_counts[topic_dir.name] += len(sentences)
        for topic, count in topic_counts.items():
            logger.info('BSNLP %s: %d sentences', topic, count)
        return output


# noinspection PyUnresolvedReferences, PyGlobalUndefined
def main(data_args: DataArguments) -> None:
    global logger, paths

    logger.info('Preparing NER datasets')

    download_root = paths['base']['data'] / 'download' / 'ner'
    output_dir = paths['prepare']['data'] / 'ner'

    parsers: List[NerDatasetParser] = [
        BsnlpParser(download_root / 'bsnlp-2017-21' / 'bsnlp', data_args.label_remap.get('bsnlp', {})),
        CnecParser(download_root / 'CNEC_2.0_konkol' / 'CNEC_2.0_konkol', data_args.label_remap.get('cnec', {})),
        SetimesParser(download_root / 'setimes-sr.conll' / 'setimes-sr.conll', data_args.label_remap.get('setimes', {})),
        Hr500kParser(download_root / 'hr500k-1.0', data_args.label_remap.get('hr500k', {})),
        SukParser(download_root / 'SUK.CoNLL-U' / 'SUK.CoNLL-U', data_args.label_remap.get('suk', {})),
        WannParser(
            download_root,
            {
                'bs-wann': 'bs',
                'mk-wann': 'mk',
                'sk-wann': 'sk',
                'sq-wann': 'sq',
            },
            data_args.label_remap.get('wann', {})
        ),
    ]

    aggregated: DefaultDict[str, List[Sentence]] = defaultdict(list)
    for parser in parsers:
        parsed = parser.parse()
        for lang, sentences in parsed.items():
            aggregated[lang].extend(sentences)
            logger.info('Parsed %d sentences for %s', len(sentences), lang)

    if not aggregated:
        logger.warning('No NER sentences parsed; nothing to write')
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for lang, sentences in aggregated.items():
        target = output_dir / f'{lang}.csv'
        with target.open('w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sentence', 'labels'])
            for tokens, labels in sentences:
                writer.writerow([' '.join(tokens), ' '.join(labels)])
        logger.info('Wrote %s with %d sentences', target, len(sentences))
    logger.info('Wrote %d language files to %s', len(aggregated), output_dir)
