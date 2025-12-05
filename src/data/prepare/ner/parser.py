import re
from collections import defaultdict

from logging import Logger
from pathlib import Path
from typing import Tuple, List, Dict, Any, Iterable, Callable, DefaultDict

from syntok.segmenter import process as syntok_process

Sentence = Tuple[List[str], List[str]]
LABEL_RE = re.compile(r'([BI])-(.+)', re.IGNORECASE)


# noinspection PyMethodMayBeStatic
class NerDatasetParser:

    def __init__(self, logger: Logger, root: Path, label_remap: Dict[Any, Any]):
        self.logger = logger
        self.root = root
        self.label_remap = label_remap

    def parse(self) -> Dict[str, List[Sentence]]:
        raise NotImplementedError

    def _normalize_label(self, raw: str) -> str:
        raw = raw.strip()
        raw = raw.split("|", 1)[0]
        if raw.startswith('NER='):
            raw = raw.split('=', 1)[1]
        if raw.lower() == 'o':
            return 'O'
        if raw == 'O':
            return raw
        match = LABEL_RE.match(raw)
        if match:
            return f'{match.group(1)}-{match.group(2)}'
        return raw

    def _map_label(self, token: str, label: str) -> str:
        if self.label_remap:
            return self.label_remap.get(label, label)
        return label


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
        output: DefaultDict[str, List[Sentence]] = defaultdict(list)
        for path, lang, token_idx, label_selector in self._iter_sources():
            if not path.exists():
                continue
            sentences, has_labels = self._parse_conll_file(path, token_idx, label_selector)
            if not has_labels:
                continue
            output[lang].extend(sentences)
            self.logger.info('%s: %s -> %d sentences', lang, path.name, len(sentences))
        return output


class CnecParser(ConllDatasetParser):
    def _iter_sources(self) -> Iterable[Tuple[Path, str, int, Callable[[List[str]], str]]]:
        if not self.root.exists():
            return []
        self.logger.info('CNEC: scanning %s', self.root)
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
        self.logger.info('SETimes: %s', path)
        label_idx = 10
        return [(path, 'sr', 1, lambda parts: parts[label_idx] if len(parts) > label_idx else 'O')]


class Hr500kParser(ConllDatasetParser):
    def _iter_sources(self) -> Iterable[Tuple[Path, str, int, Callable[[List[str]], str]]]:
        path = self.root / 'hr500k.conll'
        if not path.exists():
            return []
        self.logger.info('hr500k: %s', path)
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


class WannParser(NerDatasetParser):
    def __init__(self, logger: Logger, root: Path, mapping: Dict[str, str], label_remap: Dict[str, str]):
        NerDatasetParser.__init__(self, logger, root, label_remap)
        self.base = root
        self.mapping = mapping

    def parse(self) -> Dict[str, List[Sentence]]:
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
                self.logger.info('WANN %s %s: %d sentences', lang, split_path.name, len(sentences))
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


# noinspection PyMethodMayBeStatic
class BsnlpParser(NerDatasetParser):

    def __init__(self, logger: Logger, root: Path, label_remap: Dict[Any, Any]):
        NerDatasetParser.__init__(self, logger, root, label_remap)
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
            self.logger.info('BSNLP %s: %d sentences', topic, count)
        return output


# noinspection PyMethodMayBeStatic
class NerUkParser(NerDatasetParser):

    def __init__(self, logger: Logger, root: Path, label_remap: Dict[Any, Any]):
        NerDatasetParser.__init__(self, logger, root, label_remap)

    def _iter_pairs(self) -> Iterable[Tuple[Path, Path]]:
        data_dir = self.root / 'v2.0' / 'data'
        for subset in ('bruk', 'ng'):
            subset_dir = data_dir / subset
            if not subset_dir.exists():
                continue
            for txt_file in subset_dir.glob('*.txt'):
                ann_file = txt_file.with_suffix('.ann')
                if ann_file.exists():
                    yield txt_file, ann_file

    def _load_spans(self, ann_file: Path) -> List[Tuple[int, int, str]]:
        spans: List[Tuple[int, int, str]] = []
        for line in ann_file.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            parts = line.split('\t')
            label = ''
            start = end = None
            if len(parts) == 3:
                span_bits = parts[1].split()
                if len(span_bits) >= 3:
                    label = span_bits[0]
                    try:
                        start = int(span_bits[1])
                        end = int(span_bits[2])
                    except ValueError:
                        continue
            elif len(parts) >= 4:
                label = parts[1]
                try:
                    start = int(parts[2])
                    end = int(parts[3])
                except ValueError:
                    continue
            if label and start is not None and end is not None:
                spans.append((start, end, label))
        return sorted(spans, key=lambda x: x[0])

    def _token_offsets(self, text: str, tokens: List[str]) -> List[Tuple[int, int]]:
        offsets: List[Tuple[int, int]] = []
        cursor = 0
        for tok in tokens:
            pos = text.find(tok, cursor)
            if pos == -1:
                pos = cursor
            end = pos + len(tok)
            offsets.append((pos, end))
            cursor = end
        return offsets

    def _parse_pair(self, txt_file: Path, ann_file: Path) -> List[Sentence]:
        lines = txt_file.read_text(encoding='utf-8').splitlines()
        spans = self._load_spans(ann_file)
        sentences: List[Sentence] = []

        # Build global token offsets across the whole file
        full_text = txt_file.read_text(encoding='utf-8')
        all_tokens = full_text.split()
        all_offsets = self._token_offsets(full_text, all_tokens)

        # Walk lines and build per-line sentences
        cursor = 0
        token_index = 0
        for line in lines:
            line_tokens = line.split()
            if not line_tokens:
                cursor += len(line) + 1  # include newline
                continue
            line_start = full_text.find(line, cursor)
            if line_start == -1:
                line_start = cursor
            line_end = line_start + len(line)

            tokens = []
            labels = []
            while token_index < len(all_tokens):
                tok_start, tok_end = all_offsets[token_index]
                if tok_start >= line_end:
                    break
                if tok_end <= line_start:
                    token_index += 1
                    continue
                tokens.append(all_tokens[token_index])
                label = 'O'
                for start, end, raw_label in spans:
                    if tok_end <= start or tok_start >= end:
                        continue
                    prefix = 'B' if label == 'O' else 'I'
                    normalized = self._normalize_label(raw_label)
                    normalized = self._map_label(tokens[-1], normalized)
                    if normalized == 'O':
                        break
                    label = f'{prefix}-{normalized}'
                    break
                labels.append(label)
                token_index += 1

            if tokens:
                sentences.append((tokens, labels))

            cursor = line_end + 1  # assume newline separator

        return sentences

    def parse(self) -> Dict[str, List[Sentence]]:
        output: DefaultDict[str, List[Sentence]] = defaultdict(list)
        count_files = 0
        for txt_file, ann_file in self._iter_pairs():
            sentences = self._parse_pair(txt_file, ann_file)
            if sentences:
                output['uk'].extend(sentences)
                count_files += 1
        self.logger.info('ner-uk: %d sentences from %d files', len(output['uk']), count_files)
        return output
