from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple

from syntok.segmenter import process as syntok_process

from ...app.args.data import DataArguments

Sentence = Tuple[List[str], List[str]]

LABEL_RE = re.compile(r"^[BIO]([-_].+)?", re.IGNORECASE)


def _normalize_label(raw: Optional[str]) -> str:
    if not raw:
        return "O"
    raw = raw.strip()
    if raw.startswith("NER="):
        raw = raw.split("=", 1)[1]
    if raw.lower() == "o":
        return "O"
    match = re.match(r"([BIO])[-_]?(.+)", raw, re.IGNORECASE)
    if match:
        return f"{match.group(1).upper()}-{match.group(2).upper()}"
    return raw.upper()


def _find_iob_label(parts: List[str]) -> str:
    for part in reversed(parts):
        if LABEL_RE.match(part):
            return part
    return "O"


def _parse_conll_file(path: Path, token_idx: int, label_selector, keep_comment: bool = False) \
        -> Tuple[List[Sentence], bool]:
    sentences: List[Sentence] = []
    tokens: List[str] = []
    labels: List[str] = []
    file_has_labels = False

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if tokens:
                sentences.append((tokens, labels))
                tokens, labels = [], []
            continue
        if line.startswith("#") and not keep_comment:
            continue
        parts = line.split("\t")
        if len(parts) <= token_idx:
            continue
        token = parts[token_idx]
        label = _normalize_label(label_selector(parts))
        if label != "O":
            file_has_labels = True
        tokens.append(token)
        labels.append(label)

    if tokens:
        sentences.append((tokens, labels))

    return sentences, file_has_labels


class NerDatasetParser:
    def parse(self) -> Dict[str, List[Sentence]]:
        raise NotImplementedError


class CnecParser(NerDatasetParser):
    def __init__(self, root: Path):
        self.root = root

    def parse(self) -> Dict[str, List[Sentence]]:
        output: DefaultDict[str, List[Sentence]] = defaultdict(list)
        if not self.root.exists():
            return output
        for fname in ("train.conll", "dtest.conll", "etest.conll"):
            path = self.root / fname
            if not path.exists():
                continue
            sentences, _ = _parse_conll_file(path, 0, lambda parts: parts[-1])
            output["cs"].extend(sentences)
        return output


class SetimesParser(NerDatasetParser):
    def __init__(self, root: Path):
        self.root = root

    def parse(self) -> Dict[str, List[Sentence]]:
        output: DefaultDict[str, List[Sentence]] = defaultdict(list)
        path = self.root / "set.sr.conll"
        if not path.exists():
            return output
        sentences, _ = _parse_conll_file(path, 1, _find_iob_label)
        output["sr"].extend(sentences)
        return output


class Hr500kParser(NerDatasetParser):
    def __init__(self, root: Path):
        self.root = root

    def parse(self) -> Dict[str, List[Sentence]]:
        output: DefaultDict[str, List[Sentence]] = defaultdict(list)
        path = self.root / "hr500k.conll" / "hr500k.conll"
        if not path.exists():
            return output
        sentences, _ = _parse_conll_file(path, 1, _find_iob_label)
        output["hr"].extend(sentences)
        return output


class WannParser(NerDatasetParser):
    def __init__(self, base: Path, mapping: Dict[str, str]):
        self.base = base
        self.mapping = mapping

    def parse(self) -> Dict[str, List[Sentence]]:
        output: DefaultDict[str, List[Sentence]] = defaultdict(list)
        for folder, lang in self.mapping.items():
            dataset_dir = self.base / folder
            if not dataset_dir.exists():
                continue
            for split in ("train", "dev", "test", "extra"):
                split_path = dataset_dir / split
                if not split_path.exists():
                    continue
                output[lang].extend(self._parse_split(split_path))
        return output

    @staticmethod
    def _parse_split(path: Path) -> List[Sentence]:
        sentences: List[Sentence] = []
        tokens: List[str] = []
        labels: List[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
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
            token = token_raw.split(":", 1)[1] if ":" in token_raw else token_raw
            tokens.append(token)
            labels.append(_normalize_label(label_raw))
        if tokens:
            sentences.append((tokens, labels))
        return sentences


class SukParser(NerDatasetParser):
    def __init__(self, root: Path):
        self.root = root

    def parse(self) -> Dict[str, List[Sentence]]:
        output: DefaultDict[str, List[Sentence]] = defaultdict(list)
        if not self.root.exists():
            return output

        for path in sorted(self.root.glob("*.conllu")):
            sentences, has_labels = _parse_conll_file(path, 1, self._extract_label)
            if not has_labels:
                continue
            output["sl"].extend(sentences)
        return output

    @staticmethod
    def _extract_label(parts: List[str]) -> str:
        for part in parts:
            if "NER=" in part:
                for feature in part.split("|"):
                    if feature.startswith("NER="):
                        return feature.split("=", 1)[1]
        return _find_iob_label(parts)


class BsnlpParser(NerDatasetParser):
    def __init__(self, root: Path):
        self.raw_root = root / "raw"
        self.ann_root = root / "annotated"

    def parse(self) -> Dict[str, List[Sentence]]:
        output: DefaultDict[str, List[Sentence]] = defaultdict(list)
        if not self.raw_root.exists() or not self.ann_root.exists():
            return output

        annotations = self._load_annotations()
        for topic_dir in sorted(self.raw_root.iterdir()):
            if not topic_dir.is_dir():
                continue
            for lang_dir in sorted(topic_dir.iterdir()):
                if not lang_dir.is_dir():
                    continue
                lang = lang_dir.name
                for raw_file in lang_dir.glob("*.txt"):
                    doc_id, sentences = self._process_doc(raw_file, annotations, lang)
                    if doc_id and sentences:
                        output[lang].extend(sentences)
        return output

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        tokens: List[str] = []
        for paragraph in syntok_process(text):
            for sentence in paragraph:
                tokens.extend([tok.value for tok in sentence])
        return tokens

    def _load_annotations(self) -> Dict[Tuple[str, str], List[Tuple[List[str], str]]]:
        index: Dict[Tuple[str, str], List[Tuple[List[str], str]]] = {}
        for topic_dir in sorted(self.ann_root.iterdir()):
            if not topic_dir.is_dir():
                continue
            for lang_dir in sorted(topic_dir.iterdir()):
                if not lang_dir.is_dir():
                    continue
                lang = lang_dir.name
                for ann_file in lang_dir.glob("*.*"):
                    doc_id, entities = self._parse_annotation_file(ann_file)
                    if doc_id and entities:
                        index[(lang, doc_id)] = entities
        return index

    def _parse_annotation_file(self, path: Path) -> Tuple[str, List[Tuple[List[str], str]]]:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return "", []
        doc_id = lines[0].strip()
        entities: List[Tuple[List[str], str]] = []
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            surface, ent_type = parts[0], parts[2]
            tokens = self._tokenize(surface)
            if not tokens:
                continue
            entities.append((tokens, ent_type.upper()))
        return doc_id, entities

    def _process_doc(
        self,
        raw_file: Path,
        annotations: Dict[Tuple[str, str], List[Tuple[List[str], str]]],
        lang: str
    ) -> Tuple[str, List[Sentence]]:
        lines = raw_file.read_text(encoding="utf-8").splitlines()
        if len(lines) < 5:
            return "", []
        doc_id = lines[0].strip()
        entities = annotations.get((lang, doc_id), [])
        if not entities:
            return doc_id, []

        text = "\n".join(lines[4:])
        sentences: List[Sentence] = []
        for paragraph in syntok_process(text):
            for sentence in paragraph:
                tokens = [tok.value for tok in sentence]
                labels = ["O"] * len(tokens)
                lower_tokens = [t.lower() for t in tokens]
                for ent_tokens, ent_type in entities:
                    if not ent_tokens:
                        continue
                    pattern = [t.lower() for t in ent_tokens]
                    idx = 0
                    while idx <= len(tokens) - len(pattern):
                        if all(label == "O" for label in labels[idx:idx + len(pattern)]) and \
                                lower_tokens[idx:idx + len(pattern)] == pattern:
                            labels[idx] = f"B-{ent_type}"
                            for j in range(1, len(pattern)):
                                labels[idx + j] = f"I-{ent_type}"
                            idx += len(pattern)
                        else:
                            idx += 1
                if tokens:
                    sentences.append((tokens, labels))
        return doc_id, sentences


def _write_outputs(output_dir: Path, sentences_by_lang: Dict[str, List[Sentence]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for lang, sentences in sentences_by_lang.items():
        target = output_dir / f"{lang}.csv"
        with target.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sentence", "labels"])
            for tokens, labels in sentences:
                writer.writerow([" ".join(tokens), " ".join(labels)])


# noinspection PyUnresolvedReferences
def main(data_args: DataArguments) -> None:
    global logger, paths

    logger.info("Preparing NER datasets")

    download_root = paths["base"]["data"] / "download" / "ner"
    output_dir = paths["prepare"]["data"] / "ner"

    parsers: List[NerDatasetParser] = [
        BsnlpParser(download_root / "bsnlp-2017-21" / "bsnlp"),
        CnecParser(download_root / "CNEC_2.0_konkol" / "CNEC_2.0_konkol"),
        SetimesParser(download_root / "setimes-sr.conll" / "setimes-sr.conll"),
        Hr500kParser(download_root / "hr500k-1.0"),
        SukParser(download_root / "SUK.CoNLL-U" / "SUK.CoNLL-U"),
        WannParser(
            download_root,
            {
                "bs-wann": "bs",
                "mk-wann": "mk",
                "sk-wann": "sk",
                "sq-wann": "sq",
            },
        ),
    ]

    aggregated: DefaultDict[str, List[Sentence]] = defaultdict(list)
    for parser in parsers:
        parsed = parser.parse()
        for lang, sentences in parsed.items():
            aggregated[lang].extend(sentences)
            logger.info("Parsed %d sentences for %s", len(sentences), lang)

    if not aggregated:
        logger.warning("No NER sentences parsed; nothing to write")
        return

    _write_outputs(output_dir, aggregated)
    logger.info("Wrote %d language files to %s", len(aggregated), output_dir)
