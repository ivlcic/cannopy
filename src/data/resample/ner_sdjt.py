from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ...app.args.data import DataArguments
from ...app.args.runtime import Paths

logger: Logger
paths: Paths

Sentence = Tuple[List[str], List[str]]
SplitSamples = Dict[str, Dict[str, List[Sentence]]]

L8: Tuple[str, ...] = ("bg", "cs", "hr", "pl", "ru", "sl", "sr", "uk")
AUX: Tuple[str, ...] = ("bs", "mk", "sk", "sq")
L12: Tuple[str, ...] = L8 + AUX
CURVE_LANGUAGES = frozenset({"sr", "sl"})
CURVE_BUDGETS = frozenset({10, 25, 50, 100})
CORE_ENTITY_TYPES = frozenset({"PER", "ORG", "LOC"})


@dataclass(frozen=True)
class RunSpec:
    run_name: str
    pool_name: str
    train_languages: Tuple[str, ...]
    eval_languages: Tuple[str, ...]
    target_language: Optional[str] = None
    budget_pct: int = 100
    uses_macro_eval: bool = False

    @property
    def is_monolingual(self) -> bool:
        return self.pool_name == "mono"

    @property
    def is_multilingual(self) -> bool:
        return self.pool_name in {"multi8", "multi12"}

    @property
    def metric_name(self) -> str:
        return "eval_macro_f1" if self.uses_macro_eval else "eval_f1"


def available_run_names() -> List[str]:
    names = [f"mono-{lang}" for lang in L8]
    names.extend(["multi8", "multi12"])
    for lang in sorted(CURVE_LANGUAGES):
        for budget in sorted(b for b in CURVE_BUDGETS if b < 100):
            names.append(f"mono-{lang}-p{budget}")
            names.append(f"multi8-{lang}-p{budget}")
            names.append(f"multi12-{lang}-p{budget}")
    return names


def _normalize_procedure(value: str) -> str:
    return value.strip().lower().replace("_", "").replace("-", "")


def _normalize_lang(value: str) -> str:
    return value.strip().lower()


def _stable_int(text: str) -> int:
    value = 0
    for ch in text:
        value = ((value * 131) + ord(ch)) % 2_147_483_647
    return value


def _normalize_budget(value: int) -> int:
    budget = int(value)
    if budget <= 0 or budget > 100:
        raise ValueError(f"Budget percentage must be in 1..100, got {budget}.")
    return budget


def _parse_budget_suffix(suffix: str) -> int:
    if not suffix.startswith("p"):
        raise ValueError(f"Invalid budget suffix {suffix!r}. Expected p10, p25, p50, or p100.")
    return _normalize_budget(int(suffix[1:]))


def resolve_run_spec_from_name(run_name: str) -> RunSpec:
    normalized = run_name.strip().lower()
    if not normalized:
        raise ValueError("Run name is empty.")
    if normalized == "multi8":
        return RunSpec(
            run_name="multi8",
            pool_name="multi8",
            train_languages=L8,
            eval_languages=L8,
            uses_macro_eval=True,
        )
    if normalized == "multi12":
        return RunSpec(
            run_name="multi12",
            pool_name="multi12",
            train_languages=L12,
            eval_languages=L8,
            uses_macro_eval=True,
        )

    parts = normalized.split("-")
    if len(parts) not in {2, 3}:
        raise ValueError(
            f"Unsupported run name {run_name!r}. Expected one of {', '.join(available_run_names())}."
        )

    procedure = _normalize_procedure(parts[0])
    lang = _normalize_lang(parts[1])
    budget = 100 if len(parts) == 2 else _parse_budget_suffix(parts[2])

    if procedure == "mono":
        if lang not in L8:
            raise ValueError(f"Monolingual runs are supported only for L8 languages, got {lang!r}.")
        if budget == 100:
            return RunSpec(
                run_name=f"mono-{lang}",
                pool_name="mono",
                train_languages=(lang,),
                eval_languages=(lang,),
                target_language=lang,
            )
        if lang not in CURVE_LANGUAGES or budget not in CURVE_BUDGETS:
            raise ValueError(f"Monolingual budgeted runs are supported only for sr/sl with 10/25/50/100.")
        return RunSpec(
            run_name=f"mono-{lang}-p{budget}",
            pool_name="mono",
            train_languages=(lang,),
            eval_languages=(lang,),
            target_language=lang,
            budget_pct=budget,
        )

    if procedure not in {"multi8", "multi12"}:
        raise ValueError(f"Unsupported procedure {parts[0]!r}.")
    if lang not in CURVE_LANGUAGES:
        raise ValueError(f"Target-specific multilingual variants are supported only for sr/sl, got {lang!r}.")
    if budget == 100:
        return resolve_run_spec_from_name(procedure)
    if budget not in CURVE_BUDGETS:
        raise ValueError(f"Budgeted multilingual runs are supported only for 10/25/50/100.")
    train_languages = L8 if procedure == "multi8" else L12
    return RunSpec(
        run_name=f"{procedure}-{lang}-p{budget}",
        pool_name=procedure,
        train_languages=train_languages,
        eval_languages=(lang,),
        target_language=lang,
        budget_pct=budget,
    )


def harmonize_label(label: str) -> str:
    value = label.strip()
    if not value or value.upper() == "O":
        return "O"
    if "-" not in value:
        entity = value.upper()
        return f"B-{entity}" if entity in CORE_ENTITY_TYPES else "O"
    prefix, entity = value.split("-", 1)
    prefix = prefix.upper()
    entity = entity.upper()
    if prefix in {"S", "U"}:
        prefix = "B"
    elif prefix in {"E", "L"}:
        prefix = "I"
    if prefix not in {"B", "I"}:
        return "O"
    if entity not in CORE_ENTITY_TYPES:
        return "O"
    return f"{prefix}-{entity}"


def _read_split_file(path: Path) -> List[Sentence]:
    samples: List[Sentence] = []
    if not path.exists():
        return samples
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.reader(fp)
        next(reader, None)
        for line_no, row in enumerate(reader, start=2):
            if len(row) < 2:
                continue
            tokens = row[0].split(" ")
            labels = [harmonize_label(label) for label in row[1].split(" ")]
            if len(tokens) != len(labels):
                logger.warning(
                    "Skipping malformed row %d from %s due to token/label mismatch (%d != %d).",
                    line_no,
                    path,
                    len(tokens),
                    len(labels),
                )
                continue
            samples.append((tokens, labels))
    return samples


def load_run_corpora(source_dir: Path, run_spec: RunSpec) -> SplitSamples:
    split_names = ("train", "eval", "test")
    samples_by_split: SplitSamples = {split: {} for split in split_names}
    for split in split_names:
        languages = run_spec.train_languages if split == "train" else run_spec.eval_languages
        for lang in languages:
            file_path = source_dir / f"ner-{lang}.{split}.csv"
            samples = _read_split_file(file_path)
            if not samples:
                raise FileNotFoundError(f"No {split} samples found for language {lang} at {file_path}.")
            samples_by_split[split][lang] = samples
    return samples_by_split


def count_tokens(sentences: Sequence[Sentence]) -> int:
    return sum(len(tokens) for tokens, _ in sentences)


def _shuffle_sentences(sentences: Sequence[Sentence], seed: int) -> List[Sentence]:
    shuffled = list(sentences)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def _take_until_token_budget(sentences: Sequence[Sentence], token_budget: int) -> List[Sentence]:
    selected: List[Sentence] = []
    seen_tokens = 0
    for sentence in sentences:
        selected.append(sentence)
        seen_tokens += len(sentence[0])
        if seen_tokens >= token_budget:
            break
    return selected


def sample_sentences_by_token_budget(sentences: Sequence[Sentence], token_budget: int, seed: int) -> List[Sentence]:
    if not sentences or token_budget <= 0:
        return []
    shuffled = _shuffle_sentences(sentences, seed)
    total_tokens = count_tokens(shuffled)
    if total_tokens > token_budget:
        return _take_until_token_budget(shuffled, token_budget)

    selected = list(shuffled)
    seen_tokens = total_tokens
    if seen_tokens >= token_budget:
        return selected

    index = 0
    while seen_tokens < token_budget:
        sentence = shuffled[index % len(shuffled)]
        selected.append(sentence)
        seen_tokens += len(sentence[0])
        index += 1
    return selected


def build_monolingual_subset(sentences: Sequence[Sentence], budget_pct: int, seed: int) -> List[Sentence]:
    normalized_budget = _normalize_budget(budget_pct)
    if normalized_budget >= 100:
        return list(sentences)
    total_tokens = count_tokens(sentences)
    subset_tokens = max(1, math.ceil(total_tokens * normalized_budget / 100.0))
    return sample_sentences_by_token_budget(sentences, subset_tokens, seed)


def compute_multilingual_token_budget(train_by_lang: Mapping[str, Sequence[Sentence]],
                                      data_args: DataArguments) -> int:
    attrs = data_args.attributes or {}
    override = attrs.get(
        "token_budget_per_language",
        data_args.sampling.attributes.get("token_budget_per_language"),
    )
    if override:
        # noinspection PyTypeChecker
        return int(override)
    token_totals = [count_tokens(sentences) for sentences in train_by_lang.values()]
    if not token_totals:
        raise ValueError("No multilingual training corpora available to compute token budget.")
    return min(token_totals)


def compute_language_token_budgets(train_by_lang: Mapping[str, Sequence[Sentence]],
                                   run_spec: RunSpec, data_args: DataArguments) -> Dict[str, int]:
    if run_spec.is_monolingual:
        target_lang = run_spec.target_language or run_spec.train_languages[0]
        total_tokens = count_tokens(train_by_lang[target_lang])
        if run_spec.budget_pct >= 100:
            return {target_lang: total_tokens}
        return {target_lang: max(1, math.ceil(total_tokens * run_spec.budget_pct / 100.0))}

    base_budget = compute_multilingual_token_budget(train_by_lang, data_args)
    budgets = {lang: base_budget for lang in run_spec.train_languages}
    if run_spec.target_language and run_spec.budget_pct < 100:
        budgets[run_spec.target_language] = max(1, math.ceil(base_budget * run_spec.budget_pct / 100.0))
    return budgets


def build_multilingual_epoch_samples(train_by_lang: Mapping[str, Sequence[Sentence]],
                                     language_token_budgets: Mapping[str, int],
                                     seed: int) -> Tuple[List[Sentence], Dict[str, List[Sentence]]]:
    sampled_by_lang: Dict[str, List[Sentence]] = {}
    pooled: List[Sentence] = []
    for lang in sorted(language_token_budgets):
        lang_seed = seed + _stable_int(lang)
        sampled = sample_sentences_by_token_budget(train_by_lang[lang], language_token_budgets[lang], lang_seed)
        sampled_by_lang[lang] = sampled
        pooled.extend(sampled)
    random.Random(seed + _stable_int("pooled")).shuffle(pooled)
    return pooled, sampled_by_lang


def create_run_snapshot(source_dir: Path, run_spec: RunSpec, data_args: DataArguments,
                        epoch: int = 0) -> SplitSamples:
    corpora = load_run_corpora(source_dir, run_spec)
    snapshot: SplitSamples = {
        "train": {},
        "eval": {lang: list(sentences) for lang, sentences in corpora["eval"].items()},
        "test": {lang: list(sentences) for lang, sentences in corpora["test"].items()},
    }
    base_seed = int(data_args.sampling.seed or data_args.split.seed or 2611)

    if run_spec.is_multilingual:
        token_budgets = compute_language_token_budgets(corpora["train"], run_spec, data_args)
        _, sampled_by_lang = build_multilingual_epoch_samples(
            corpora["train"],
            token_budgets,
            base_seed + (epoch * 10_003),
        )
        snapshot["train"] = sampled_by_lang
    else:
        target_lang = run_spec.target_language or run_spec.train_languages[0]
        subset = build_monolingual_subset(
            corpora["train"][target_lang],
            run_spec.budget_pct,
            base_seed + _stable_int(target_lang),
        )
        snapshot["train"] = {target_lang: subset}
    return snapshot


def _write_split_csv(path: Path, sentences: Iterable[Sentence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["sentence", "labels"])
        for tokens, labels in sentences:
            writer.writerow([" ".join(tokens), " ".join(labels)])


def write_run_snapshot(target_dir: Path, snapshot: SplitSamples) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for split, split_samples in snapshot.items():
        for lang, sentences in split_samples.items():
            _write_split_csv(target_dir / f"ner-{lang}.{split}.csv", sentences)


def main(data_args: DataArguments) -> None:
    source_dir = paths.get_ctx_path("split").parent / 'ner'
    target_dir = paths.get_ctx_path("split")
    snapshot_epoch = int(data_args.attributes.get("snapshot_epoch", 0))

    run_names = available_run_names()
    run_specs = [resolve_run_spec_from_name(run_name) for run_name in run_names]
    for run_spec in run_specs:
        snapshot = create_run_snapshot(source_dir, run_spec, data_args, epoch=snapshot_epoch)
        output_dir = target_dir / run_spec.run_name
        write_run_snapshot(output_dir, snapshot)
        logger.info("Prepared SDJT NER split for %s at %s", run_spec.run_name, output_dir)
    logger.info("Prepared %d SDJT NER runs under %s", len(run_specs), target_dir)
