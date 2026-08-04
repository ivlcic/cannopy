import csv
import re
import shutil

from collections import Counter, defaultdict
from logging import Logger
from pathlib import Path
from typing import Tuple, List, Dict, Any, Iterable, Callable, DefaultDict

from syntok.segmenter import process as syntok_process

from ...app.args.runtime import Paths
from ...app.args.data import DataArguments
from ...app.ner import NerSample

logger: Logger
paths: Paths

Sentence = NerSample
LABEL_RE = re.compile(r'([BI])-(.+)', re.IGNORECASE)


# noinspection PyMethodMayBeStatic
class NerDatasetParser:

    def __init__(
        self,
        root: Path,
        label_remap: Dict[Any, Any],
        corpus: str = '',
        label_remap_exact: Dict[Any, bool] | None = None,
    ):
        self.root = root
        self.label_remap = label_remap
        self.corpus = corpus or root.name
        self.label_remap_exact = label_remap_exact or {}
        self.label_remap_counts: Counter[Tuple[str, str]] = Counter()
        self._validate_label_remap_exact()

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
        mapped_label = self.label_remap.get(label, label)
        if label != 'O' or mapped_label != 'O':
            self.label_remap_counts[(label, mapped_label)] += 1
        return mapped_label

    def _validate_label_remap_exact(self) -> None:
        marked_labels = set(self.label_remap_exact)
        graded_labels = {
            source_label
            for source_label, target_label in self.label_remap.items()
            if target_label != 'O'
        }
        missing = sorted(graded_labels - marked_labels)
        unused = sorted(marked_labels - graded_labels)
        invalid = sorted(
            source_label
            for source_label, is_exact in self.label_remap_exact.items()
            if not isinstance(is_exact, bool)
        )
        errors = []
        if missing:
            errors.append(f'missing markers for {missing}')
        if unused:
            errors.append(f'unused markers for {unused}')
        if invalid:
            errors.append(f'non-boolean markers for {invalid}')
        if errors:
            raise ValueError(
                f'Invalid exact-meaning markers for {self.corpus}: '
                + '; '.join(errors)
            )

    def remap_stats(self) -> List[Dict[str, Any]]:
        mappings = set(self.label_remap_counts)
        mappings.update(
            (str(source_label), str(target_label))
            for source_label, target_label in self.label_remap.items()
        )

        rows: List[Dict[str, Any]] = []
        for source_label, target_label in sorted(mappings):
            if target_label == 'O':
                meaning_match = 'not_applicable'
            elif source_label not in self.label_remap:
                meaning_match = 'exact'
            elif self.label_remap_exact.get(source_label, False):
                meaning_match = 'exact'
            else:
                meaning_match = 'inexact'

            rows.append({
                'corpus': self.corpus,
                'source_label': source_label,
                'target_label': target_label,
                'token_count': self.label_remap_counts[(source_label, target_label)],
                'configured_remap': (
                    source_label in self.label_remap
                    and self.label_remap[source_label] == target_label
                ),
                'meaning_match': meaning_match,
            })
        return rows


class ConllDatasetParser(NerDatasetParser):

    def _iter_sources(self) -> Iterable[Tuple[Path, str, int, Callable[[List[str]], str]]]:
        raise NotImplementedError

    def _source_corpus_name(self, path: Path) -> str:
        return self.corpus

    def _parse_conll_file(
        self,
        path: Path,
        token_idx: int,
        label_selector: Callable[[List[str]], str],
    ) -> Tuple[List[Sentence], bool]:
        sentences: List[Sentence] = []
        tokens: List[str] = []
        labels: List[str] = []
        file_has_labels = False
        corpus_name = self._source_corpus_name(path)
        current_doc_id = path.stem
        current_sent_id = ''
        sentence_index = 0

        def append_sentence() -> None:
            nonlocal tokens, labels, current_sent_id, sentence_index
            if not tokens:
                return
            sentence_index += 1
            sentences.append(NerSample(
                tokens=tokens,
                labels=labels,
                corpus_name=corpus_name,
                doc_id=current_doc_id,
                sent_id=current_sent_id or str(sentence_index),
            ))
            tokens, labels = [], []
            current_sent_id = ''

        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                append_sentence()
                continue
            if line.startswith('#'):
                metadata = line[1:].strip()
                if metadata.startswith('newdoc id ='):
                    current_doc_id = metadata.split('=', 1)[1].strip() or current_doc_id
                elif metadata.startswith('sent_id ='):
                    current_sent_id = metadata.split('=', 1)[1].strip()
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

        append_sentence()

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
            logger.info(
                '%s: %s -> %d sentences (%d tokens)',
                lang,
                path.name,
                len(sentences),
                sum(len(sample.tokens) for sample in sentences),
            )
        return output


# noinspection SpellCheckingInspection
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


# noinspection SpellCheckingInspection
class SetimesParser(ConllDatasetParser):
    def _iter_sources(self) -> Iterable[Tuple[Path, str, int, Callable[[List[str]], str]]]:
        path = self.root / 'set.sr.conll'
        if not path.exists():
            return []
        logger.info('SETimes: %s', path)
        label_idx = 10
        return [(path, 'sr', 1, lambda parts: parts[label_idx] if len(parts) > label_idx else 'O')]


# noinspection SpellCheckingInspection
class Hr500kParser(ConllDatasetParser):
    def _iter_sources(self) -> Iterable[Tuple[Path, str, int, Callable[[List[str]], str]]]:
        path = self.root / 'hr500k.conll'
        if not path.exists():
            return []
        logger.info('hr500k: %s', path)
        label_idx = 10
        return [(path, 'hr', 1, lambda parts: parts[label_idx] if len(parts) > label_idx else 'O')]


# noinspection SpellCheckingInspection
class SukParser(ConllDatasetParser):
    CORPUS_NAMES = {
        'elexiswsd': 'elexis-wsd',
        'senticoref': 'senticoref',
        'ssj500k-syn': 'ssj500k',
    }

    def _source_corpus_name(self, path: Path) -> str:
        source_name = path.name.removesuffix('.ud.conllu')
        return self.CORPUS_NAMES.get(source_name, source_name)

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


# noinspection SpellCheckingInspection
class WannParser(NerDatasetParser):
    def __init__(
        self,
        root: Path,
        mapping: Dict[str, str],
        label_remap: Dict[str, str],
        corpus: str = '',
        label_remap_exact: Dict[Any, bool] | None = None,
    ):
        NerDatasetParser.__init__(
            self,
            root,
            label_remap,
            corpus=corpus,
            label_remap_exact=label_remap_exact,
        )
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
                logger.info('WANN %s %s: %d sentences', lang, split_path.name, len(sentences))
        return output

    def _parse_split(self, path: Path) -> List[Sentence]:
        sentences: List[Sentence] = []
        tokens: List[str] = []
        labels: List[str] = []
        doc_id = path.relative_to(self.base).as_posix()

        def append_sentence() -> None:
            nonlocal tokens, labels
            if not tokens:
                return
            sentences.append(NerSample(
                tokens=tokens,
                labels=labels,
                corpus_name=self.corpus,
                doc_id=doc_id,
                sent_id=str(len(sentences) + 1),
            ))
            tokens, labels = [], []

        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                append_sentence()
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            token_raw, label_raw = parts[0], parts[1]
            token = token_raw.split(':', 1)[1] if ':' in token_raw else token_raw
            tokens.append(token)
            labels.append(self._map_label(token, self._normalize_label(label_raw)))
        append_sentence()
        return sentences


# noinspection PyMethodMayBeStatic, SpellCheckingInspection
class BsnlpParser(NerDatasetParser):

    def __init__(
        self,
        root: Path,
        label_remap: Dict[Any, Any],
        corpus: str = '',
        label_remap_exact: Dict[Any, bool] | None = None,
    ):
        NerDatasetParser.__init__(
            self,
            root,
            label_remap,
            corpus=corpus,
            label_remap_exact=label_remap_exact,
        )
        self.raw_root = root / 'raw'
        self.ann_root = root / 'annotated'

    def _map_label(self, token: str, label: str) -> str:
        if '@' in token:
            if label != 'O':
                self.label_remap_counts[(label, 'O')] += 1
            return 'O'
        return super()._map_label(token, label)

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
        sentence_index = 0
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
                    sentence_index += 1
                    sentences.append(NerSample(
                        tokens=tokens,
                        labels=labels,
                        corpus_name=self.corpus,
                        doc_id=f'{topic}/{doc_id}',
                        sent_id=str(sentence_index),
                    ))
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
            logger.info('BSNLP %s: %d sentences', topic, count)
        return output


# noinspection PyMethodMayBeStatic, SpellCheckingInspection
class NerUkParser(NerDatasetParser):

    def _doc_id(self, txt_file: Path) -> str:
        data_root = self.root / 'v2.0' / 'data'
        try:
            relative_path = txt_file.relative_to(data_root)
        except ValueError:
            relative_path = txt_file.relative_to(self.root)
        return relative_path.with_suffix('').as_posix()

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
        for line_no, line in enumerate(lines, start=1):
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
                    prefix = 'B' if tok_start <= start < tok_end else 'I'
                    normalized = self._normalize_label(raw_label)
                    normalized = self._map_label(tokens[-1], normalized)
                    if normalized == 'O':
                        break
                    label = f'{prefix}-{normalized}'
                    break
                labels.append(label)
                token_index += 1

            if tokens:
                sentences.append(NerSample(
                    tokens=tokens,
                    labels=labels,
                    corpus_name=self.corpus,
                    doc_id=self._doc_id(txt_file),
                    sent_id=str(line_no),
                ))

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
        logger.info('ner-uk: %d sentences from %d files', len(output['uk']), count_files)
        return output


def write_remap_stats(output_dir: Path, parsers: List[NerDatasetParser]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'ner-remap-stats.csv'
    columns = [
        'corpus',
        'source_label',
        'target_label',
        'token_count',
        'configured_remap',
        'meaning_match',
    ]
    rows = [
        row
        for parser in parsers
        for row in parser.remap_stats()
    ]
    with output_path.open('w', encoding='utf-8', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_outputs(
    output_dir: Path,
    aggregated: Dict[str, List[Sentence]],
    file_suffix: str = "",
) -> None:
    """
    Write per-language CSVs.
    """
    for lang, sentences in aggregated.items():
        target = output_dir / f'ner-{lang}{file_suffix}.csv'
        label_counter: Counter = Counter()
        tok_count = 0
        with target.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=NerSample.NER_CSV_COLUMNS)
            writer.writeheader()
            for sample in sentences:
                tok_count += len(sample.tokens)
                writer.writerow(sample.to_csv_row())
                for label in sample.labels:
                    if label != 'O':
                        label_counter[label] += 1


# noinspection SpellCheckingInspection
def main(data_args: DataArguments) -> None:
    logger.info(f'Preparing {data_args.dataset_name} datasets')

    download_root = paths.get_ctx_path('download')
    output_dir = paths.context

    label_remap = data_args.label_remap
    label_remap_exact = data_args.label_remap_exact
    parsers: List[NerDatasetParser] = [
        BsnlpParser(
            download_root / 'bsnlp-2017-21' / 'bsnlp',
            label_remap.get('bsnlp', {}),
            corpus='bsnlp',
            label_remap_exact=label_remap_exact.get('bsnlp', {}),
        ),
        CnecParser(
            download_root / 'CNEC_2.0_konkol' / 'CNEC_2.0_konkol',
            label_remap.get('cnec', {}),
            corpus='cnec',
            label_remap_exact=label_remap_exact.get('cnec', {}),
        ),
        SetimesParser(
            download_root / 'setimes-sr.conll' / 'setimes-sr.conll',
            label_remap.get('setimes', {}),
            corpus='setimes',
            label_remap_exact=label_remap_exact.get('setimes', {}),
        ),
        Hr500kParser(
            download_root / 'hr500k-1.0' / 'hr500k.conll',
            label_remap.get('hr500k', {}),
            corpus='hr500k',
            label_remap_exact=label_remap_exact.get('hr500k', {}),
        ),
        SukParser(
            download_root / 'SUK.CoNLL-U' / 'SUK.CoNLL-U',
            label_remap.get('suk', {}),
            corpus='suk',
            label_remap_exact=label_remap_exact.get('suk', {}),
        ),
        NerUkParser(
            download_root / 'ner-uk' / 'ner-uk',
            label_remap.get('ner-uk', {}),
            corpus='ner-uk',
            label_remap_exact=label_remap_exact.get('ner-uk', {}),
        ),
        WannParser(
            download_root,
            {
                'bs-wann': 'bs',
                'mk-wann': 'mk',
                'sk-wann': 'sk',
                'sq-wann': 'sq',
            },
            label_remap.get('wann', {}),
            corpus='wann',
            label_remap_exact=label_remap_exact.get('wann', {}),
        ),
        WannParser(
            download_root,
            {
                'hr-wann': 'hr-wikiann',
            },
            label_remap.get('wann', {}),
            corpus='wikiann-hr',
            label_remap_exact=label_remap_exact.get('wann', {}),
        ),
    ]

    aggregated: DefaultDict[str, List[Sentence]] = defaultdict(list)
    for _parser in parsers:
        parsed = _parser.parse()
        for lang, sentences in parsed.items():
            aggregated[lang].extend(sentences)
            logger.info(
                'Parsed %d sentences (%d tokens) for %s',
                len(sentences),
                sum(len(sample.tokens) for sample in sentences),
                lang,
            )

    remap_stats_path = write_remap_stats(paths.get_ctx_path('analyze'), parsers)
    logger.info('Wrote label-remapping statistics to %s', remap_stats_path)

    if not aggregated:
        logger.warning(f'No {data_args.dataset_name} sentences parsed; nothing to write')
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(output_dir, aggregated)

    logger.info('Wrote %d language files to %s', len(aggregated), output_dir)

    # clean up download folder
    # shutil.rmtree(download_root, ignore_errors=True)
