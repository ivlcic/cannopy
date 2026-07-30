from __future__ import annotations

import csv
import math
import random
import unicodedata
from dataclasses import dataclass, replace
from logging import Logger
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ...app.args.data import DataArguments
from ...app.args.runtime import Paths
from ...app.ner import NER_CSV_COLUMNS, NerSample

logger: Logger
paths: Paths

Sentence = NerSample
SplitSamples = Dict[str, Dict[str, List[Sentence]]]

MAIN_LANGUAGES: Tuple[str, ...] = ("bg", "cs", "hr", "pl", "ru", "sl", "sr", "uk")
AUX_LANGUAGES: Tuple[str, ...] = ("bs", "mk", "sk", "sq")
ALL_LANGUAGES: Tuple[str, ...] = MAIN_LANGUAGES + AUX_LANGUAGES
HR_WIKIANN_SOURCE = "hr-wikiann"
CROATIAN_ABLATION_EVAL_LANGUAGES: Tuple[str, ...] = tuple(
    lang for lang in MAIN_LANGUAGES if lang != "hr"
)
CROATIAN_ABLATION_RUN_NAMES: Tuple[str, ...] = (
    "multi7-no-hr",
    "multi7-plus-hr500k",
    "multi7-plus-hr-wikiann",
)
SOURCE_KEYS: Tuple[str, ...] = ALL_LANGUAGES + (HR_WIKIANN_SOURCE,)
CURVE_LANGUAGES = frozenset({"sr", "sl"})
CURVE_BUDGETS = frozenset({10, 25, 50, 100})
CORE_ENTITY_TYPES: Tuple[str, ...] = ("PER", "ORG", "LOC")
SPLIT_NAMES: Tuple[str, ...] = ("train", "eval", "test")
DEDUP_SPLIT_PRIORITY: Tuple[str, ...] = ("test", "eval", "train")


def append_seed_suffix(path: Path, seed: int | None) -> Path:
    if seed is None:
        return path
    seed_value = int(seed)
    if seed_value < 0:
        raise ValueError(f"Seed must be non-negative, got {seed_value}.")
    return path.with_name(f"{path.name}.s{seed_value}")


def parse_seed_suffix(path: Path) -> int:
    marker = path.name.rsplit(".s", 1)
    if len(marker) != 2 or not marker[1].isdigit():
        raise ValueError(f"Path does not end in a numeric .sSEED suffix: {path}")
    return int(marker[1])


@dataclass(frozen=True)
class RunSpec:
    run_name: str
    pool_name: str
    train_languages: Tuple[str, ...]
    eval_languages: Tuple[str, ...]
    target_language: Optional[str] = None
    budget_pct: int = 100
    uses_macro_eval: bool = False
    init_from_run_name: Optional[str] = None

    @property
    def is_monolingual(self) -> bool:
        return self.pool_name == "mono"

    @property
    def is_croatian_source_ablation(self) -> bool:
        return self.run_name in CROATIAN_ABLATION_RUN_NAMES

    @property
    def is_multilingual(self) -> bool:
        return self.is_croatian_source_ablation or self.pool_name in {
            "multi8",
            "multi12",
            "multi8-full",
            "full-multi8",
            "full-multi12",
            "full-multi12-capaux",
        }

    @property
    def metric_name(self) -> str:
        return "eval_macro_f1" if self.uses_macro_eval else "eval_f1"


@dataclass
class DedupCounts:
    before: int = 0
    removed: int = 0
    label_conflicts: int = 0


def available_run_names() -> List[str]:
    names = [f"mono-{lang}" for lang in MAIN_LANGUAGES]
    names.extend(["multi8", "multi12", "full-multi8", "full-multi12", "full-multi12-capaux"])
    names.extend(CROATIAN_ABLATION_RUN_NAMES)
    names.extend([f"multi8-full-{lang}" for lang in MAIN_LANGUAGES])
    names.extend([f"pretrain-multi7-full-{lang}" for lang in MAIN_LANGUAGES])
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
            train_languages=MAIN_LANGUAGES,
            eval_languages=MAIN_LANGUAGES,
            uses_macro_eval=True,
        )
    if normalized == "multi12":
        return RunSpec(
            run_name="multi12",
            pool_name="multi12",
            train_languages=ALL_LANGUAGES,
            eval_languages=MAIN_LANGUAGES,
            uses_macro_eval=True,
        )
    if normalized == "full-multi8":
        return RunSpec(
            run_name="full-multi8",
            pool_name="full-multi8",
            train_languages=MAIN_LANGUAGES,
            eval_languages=MAIN_LANGUAGES,
            uses_macro_eval=True,
        )
    if normalized == "full-multi12":
        return RunSpec(
            run_name="full-multi12",
            pool_name="full-multi12",
            train_languages=ALL_LANGUAGES,
            eval_languages=MAIN_LANGUAGES,
            uses_macro_eval=True,
        )
    if normalized == "full-multi12-capaux":
        return RunSpec(
            run_name="full-multi12-capaux",
            pool_name="full-multi12-capaux",
            train_languages=ALL_LANGUAGES,
            eval_languages=MAIN_LANGUAGES,
            uses_macro_eval=True,
        )
    if normalized == "multi7-no-hr":
        return RunSpec(
            run_name=normalized,
            pool_name=normalized,
            train_languages=CROATIAN_ABLATION_EVAL_LANGUAGES,
            eval_languages=CROATIAN_ABLATION_EVAL_LANGUAGES,
            uses_macro_eval=True,
        )
    if normalized == "multi7-plus-hr500k":
        return RunSpec(
            run_name=normalized,
            pool_name=normalized,
            train_languages=MAIN_LANGUAGES,
            eval_languages=CROATIAN_ABLATION_EVAL_LANGUAGES,
            uses_macro_eval=True,
        )
    if normalized == "multi7-plus-hr-wikiann":
        return RunSpec(
            run_name=normalized,
            pool_name=normalized,
            train_languages=tuple(
                HR_WIKIANN_SOURCE if lang == "hr" else lang
                for lang in MAIN_LANGUAGES
            ),
            eval_languages=CROATIAN_ABLATION_EVAL_LANGUAGES,
            uses_macro_eval=True,
        )

    parts = normalized.split("-")
    if len(parts) not in {2, 3, 4}:
        raise ValueError(
            f"Unsupported run name {run_name!r}. Expected one of {', '.join(available_run_names())}."
        )

    if len(parts) == 3 and parts[0] == "multi8" and parts[1] == "full":
        lang = _normalize_lang(parts[2])
        if lang not in MAIN_LANGUAGES:
            raise ValueError(f"Multi8-full runs are supported only for main languages, got {lang!r}.")
        return RunSpec(
            run_name=f"multi8-full-{lang}",
            pool_name="multi8-full",
            train_languages=MAIN_LANGUAGES,
            eval_languages=(lang,),
            target_language=lang,
        )

    if len(parts) == 4 and parts[0] == "pretrain" and parts[1] == "multi7" and parts[2] == "full":
        lang = _normalize_lang(parts[3])
        if lang not in MAIN_LANGUAGES:
            raise ValueError(f"Pretrain-Multi7-Full runs are supported only for main languages, got {lang!r}.")
        return RunSpec(
            run_name=f"pretrain-multi7-full-{lang}",
            pool_name="pretrain-multi7-full",
            train_languages=(lang,),
            eval_languages=(lang,),
            target_language=lang,
            init_from_run_name="multi8",
        )

    procedure = _normalize_procedure(parts[0])
    lang = _normalize_lang(parts[1])
    budget = 100 if len(parts) == 2 else _parse_budget_suffix(parts[2])

    if procedure == "mono":
        if lang not in MAIN_LANGUAGES:
            raise ValueError(f"Monolingual runs are supported only for main languages, got {lang!r}.")
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
    train_languages = MAIN_LANGUAGES if procedure == "multi8" else ALL_LANGUAGES
    return RunSpec(
        run_name=f"{procedure}-{lang}-p{budget}",
        pool_name=procedure,
        train_languages=train_languages,
        eval_languages=(lang,),
        target_language=lang,
        budget_pct=budget,
    )


def resolve_pretrain_multi7_spec(run_spec: RunSpec) -> RunSpec:
    if run_spec.pool_name != "pretrain-multi7-full":
        raise ValueError(
            f"Pretrain multi7 stage is supported only for pretrain-multi7-full runs, got {run_spec.run_name}."
        )
    target_lang = run_spec.target_language or run_spec.eval_languages[0]
    pretrain_languages = tuple(lang for lang in MAIN_LANGUAGES if lang != target_lang)
    return RunSpec(
        run_name=f"{run_spec.run_name}-pretrain",
        pool_name="multi8",
        train_languages=pretrain_languages,
        eval_languages=pretrain_languages,
        uses_macro_eval=True,
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
        reader = csv.DictReader(fp)
        for line_no, row in enumerate(reader, start=2):
            sample = NerSample.from_csv_row(row)
            if not sample.tokens or not sample.labels:
                continue
            labels = [harmonize_label(label) for label in sample.labels]
            if len(sample.tokens) != len(labels):
                logger.warning(
                    "Skipping malformed row %d from %s due to token/label mismatch (%d != %d).",
                    line_no,
                    path,
                    len(sample.tokens),
                    len(labels),
                )
                continue
            samples.append(replace(sample, labels=labels))
    return samples


def load_source_corpora(source_dir: Path, languages: Sequence[str]) -> SplitSamples:
    samples_by_split: SplitSamples = {split: {} for split in SPLIT_NAMES}
    for split in SPLIT_NAMES:
        for lang in languages:
            file_path = source_dir / f"ner-{lang}.{split}.csv"
            samples = _read_split_file(file_path)
            if not samples:
                raise FileNotFoundError(f"No {split} samples found for language {lang} at {file_path}.")
            samples_by_split[split][lang] = samples
    return samples_by_split


def _normalized_sentence_key(sample: Sentence) -> str:
    text = " ".join(sample.tokens)
    return unicodedata.normalize("NFKC", text).casefold()


def deduplicate_corpora(
    source_corpora: SplitSamples,
) -> Tuple[SplitSamples, List[Dict[str, object]], List[Dict[str, object]]]:
    deduplicated: SplitSamples = {split: {} for split in SPLIT_NAMES}
    counts: Dict[Tuple[str, str, str], DedupCounts] = {}
    duplicate_rows: List[Dict[str, object]] = []
    languages = sorted({
        lang
        for split_samples in source_corpora.values()
        for lang in split_samples
    })

    for lang in languages:
        seen: Dict[str, Tuple[str, Sentence]] = {}
        for split in DEDUP_SPLIT_PRIORITY:
            kept: List[Sentence] = []
            for sample in source_corpora.get(split, {}).get(lang, []):
                corpus_name = sample.corpus_name or "unknown"
                count_key = (lang, split, corpus_name)
                corpus_counts = counts.setdefault(count_key, DedupCounts())
                corpus_counts.before += 1

                sentence_key = _normalized_sentence_key(sample)
                survivor = seen.get(sentence_key)
                if survivor is None:
                    seen[sentence_key] = (split, sample)
                    kept.append(sample)
                    continue

                survivor_split, survivor_sample = survivor
                labels_match = sample.labels == survivor_sample.labels
                corpus_counts.removed += 1
                if not labels_match:
                    corpus_counts.label_conflicts += 1
                duplicate_rows.append({
                    "language": lang,
                    "removed_split": split,
                    "removed_corpus_name": corpus_name,
                    "removed_doc_id": sample.doc_id,
                    "removed_sent_id": sample.sent_id,
                    "kept_split": survivor_split,
                    "kept_corpus_name": survivor_sample.corpus_name or "unknown",
                    "kept_doc_id": survivor_sample.doc_id,
                    "kept_sent_id": survivor_sample.sent_id,
                    "labels_match": labels_match,
                })
            deduplicated[split][lang] = kept

    stats_rows: List[Dict[str, object]] = []
    for (lang, split, corpus_name), corpus_counts in sorted(counts.items()):
        stats_rows.append({
            "language": lang,
            "split": split,
            "corpus_name": corpus_name,
            "before": corpus_counts.before,
            "duplicates_removed": corpus_counts.removed,
            "after": corpus_counts.before - corpus_counts.removed,
            "label_conflicts": corpus_counts.label_conflicts,
        })
    return deduplicated, stats_rows, duplicate_rows


def write_dedup_reports(
    output_dir: Path,
    stats_rows: Sequence[Dict[str, object]],
    duplicate_rows: Sequence[Dict[str, object]],
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "ner-dedup-stats.csv"
    duplicates_path = output_dir / "ner-duplicates.csv"
    stats_columns = [
        "language",
        "split",
        "corpus_name",
        "before",
        "duplicates_removed",
        "after",
        "label_conflicts",
    ]
    duplicate_columns = [
        "language",
        "removed_split",
        "removed_corpus_name",
        "removed_doc_id",
        "removed_sent_id",
        "kept_split",
        "kept_corpus_name",
        "kept_doc_id",
        "kept_sent_id",
        "labels_match",
    ]
    with stats_path.open("w", encoding="utf-8", newline="") as stats_file:
        writer = csv.DictWriter(stats_file, fieldnames=stats_columns)
        writer.writeheader()
        writer.writerows(stats_rows)
    with duplicates_path.open("w", encoding="utf-8", newline="") as duplicates_file:
        writer = csv.DictWriter(duplicates_file, fieldnames=duplicate_columns)
        writer.writeheader()
        writer.writerows(duplicate_rows)
    return stats_path, duplicates_path


def select_run_corpora(source_corpora: SplitSamples, run_spec: RunSpec) -> SplitSamples:
    selected: SplitSamples = {split: {} for split in SPLIT_NAMES}
    for split in SPLIT_NAMES:
        languages = run_spec.train_languages if split == "train" else run_spec.eval_languages
        for lang in languages:
            samples = source_corpora.get(split, {}).get(lang, [])
            if not samples:
                raise FileNotFoundError(f"No {split} samples available for language {lang}.")
            selected[split][lang] = samples
    return selected


def load_run_corpora(source_dir: Path, run_spec: RunSpec) -> SplitSamples:
    languages = tuple(dict.fromkeys(run_spec.train_languages + run_spec.eval_languages))
    source_corpora = load_source_corpora(source_dir, languages)
    return select_run_corpora(source_corpora, run_spec)


def count_tokens(sentences: Sequence[Sentence]) -> int:
    return sum(len(sentence.tokens) for sentence in sentences)


def _shuffle_sentences(sentences: Sequence[Sentence], seed: int) -> List[Sentence]:
    shuffled = list(sentences)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def _take_until_token_budget(sentences: Sequence[Sentence], token_budget: int) -> List[Sentence]:
    selected: List[Sentence] = []
    seen_tokens = 0
    for sentence in sentences:
        selected.append(sentence)
        seen_tokens += len(sentence.tokens)
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
        seen_tokens += len(sentence.tokens)
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


def compute_croatian_ablation_token_budget(
    train_by_lang: Mapping[str, Sequence[Sentence]],
    run_spec: RunSpec,
    data_args: DataArguments,
) -> int:
    missing_base_languages = [
        lang
        for lang in CROATIAN_ABLATION_EVAL_LANGUAGES
        if lang not in train_by_lang
    ]
    if missing_base_languages:
        raise ValueError(
            f"Croatian source ablation {run_spec.run_name} is missing common base languages: "
            f"{missing_base_languages}."
        )

    attrs = data_args.attributes or {}
    override = attrs.get(
        "token_budget_per_language",
        data_args.sampling.attributes.get("token_budget_per_language"),
    )
    if override:
        token_budget = int(override)
    else:
        token_budget = min(
            count_tokens(train_by_lang[lang])
            for lang in CROATIAN_ABLATION_EVAL_LANGUAGES
        )

    source_token_counts = {
        source: count_tokens(sentences)
        for source, sentences in train_by_lang.items()
    }
    insufficient_sources = {
        source: total_tokens
        for source, total_tokens in source_token_counts.items()
        if total_tokens < token_budget
    }
    if insufficient_sources:
        raise ValueError(
            f"Croatian source ablation {run_spec.run_name} requires {token_budget} unique "
            f"tokens per source, but these sources are smaller: {insufficient_sources}. "
            "Lower the common token_budget_per_language for all three ablation runs; "
            "do not oversample this comparison."
        )
    return token_budget


def compute_language_token_budgets(train_by_lang: Mapping[str, Sequence[Sentence]],
                                   run_spec: RunSpec, data_args: DataArguments) -> Dict[str, int]:
    if run_spec.is_monolingual:
        target_lang = run_spec.target_language or run_spec.train_languages[0]
        total_tokens = count_tokens(train_by_lang[target_lang])
        if run_spec.budget_pct >= 100:
            return {target_lang: total_tokens}
        return {target_lang: max(1, math.ceil(total_tokens * run_spec.budget_pct / 100.0))}

    if run_spec.is_croatian_source_ablation:
        base_budget = compute_croatian_ablation_token_budget(
            train_by_lang,
            run_spec,
            data_args,
        )
    else:
        base_budget = compute_multilingual_token_budget(train_by_lang, data_args)
    budgets = {lang: base_budget for lang in run_spec.train_languages}
    if run_spec.pool_name == "full-multi8":
        return {
            lang: count_tokens(train_by_lang[lang])
            for lang in run_spec.train_languages
        }
    if run_spec.pool_name == "full-multi12":
        return {
            lang: count_tokens(train_by_lang[lang])
            for lang in run_spec.train_languages
        }
    if run_spec.pool_name == "full-multi12-capaux":
        return {
            lang: count_tokens(train_by_lang[lang]) if lang in MAIN_LANGUAGES else base_budget
            for lang in run_spec.train_languages
        }
    if run_spec.pool_name == "multi8-full":
        target_lang = run_spec.target_language or run_spec.train_languages[0]
        budgets[target_lang] = count_tokens(train_by_lang[target_lang])
        return budgets
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


def create_run_snapshot(
    source_dir: Path,
    run_spec: RunSpec,
    data_args: DataArguments,
    epoch: int = 0,
    source_corpora: Optional[SplitSamples] = None,
) -> SplitSamples:
    corpora = (
        select_run_corpora(source_corpora, run_spec)
        if source_corpora is not None
        else load_run_corpora(source_dir, run_spec)
    )
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


def create_pretrain_snapshot(
    source_dir: Path,
    run_spec: RunSpec,
    data_args: DataArguments,
    epoch: int = 0,
    source_corpora: Optional[SplitSamples] = None,
) -> SplitSamples:
    pretrain_spec = resolve_pretrain_multi7_spec(run_spec)
    corpora = (
        select_run_corpora(source_corpora, pretrain_spec)
        if source_corpora is not None
        else load_run_corpora(source_dir, pretrain_spec)
    )
    snapshot: SplitSamples = {
        "train": {},
        "eval": {lang: list(sentences) for lang, sentences in corpora["eval"].items()},
        "test": {lang: list(sentences) for lang, sentences in corpora["test"].items()},
    }
    base_seed = int(data_args.sampling.seed or data_args.split.seed or 2611)
    token_budgets = compute_language_token_budgets(corpora["train"], pretrain_spec, data_args)
    _, sampled_by_lang = build_multilingual_epoch_samples(
        corpora["train"],
        token_budgets,
        base_seed + (epoch * 10_003),
    )
    snapshot["train"] = sampled_by_lang
    return snapshot


def _write_split_csv(path: Path, sentences: Iterable[Sentence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=NER_CSV_COLUMNS)
        writer.writeheader()
        for sentence in sentences:
            writer.writerow(sentence.to_csv_row())


def write_run_snapshot(target_dir: Path, snapshot: SplitSamples, file_prefix: str = "ner") -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for split, split_samples in snapshot.items():
        for lang, sentences in split_samples.items():
            _write_split_csv(target_dir / f"{file_prefix}-{lang}.{split}.csv", sentences)


def main(data_args: DataArguments) -> None:
    source_dir = paths.get_ctx_path("split").parent / 'ner'
    split_seed = data_args.split.seed
    target_dir = append_seed_suffix(paths.get_ctx_path("split"), split_seed)
    analyze_dir = append_seed_suffix(paths.get_ctx_path("analyze"), split_seed)
    snapshot_epoch = int(data_args.attributes.get("snapshot_epoch", 0))
    source_corpora = load_source_corpora(source_dir, SOURCE_KEYS)

    if data_args.sampling.dedup:
        source_corpora, stats_rows, duplicate_rows = deduplicate_corpora(source_corpora)
        stats_path, duplicates_path = write_dedup_reports(
            analyze_dir,
            stats_rows,
            duplicate_rows,
        )
        logger.info(
            "Removed %d duplicate NER samples, including %d label conflicts; "
            "wrote reports to %s and %s.",
            len(duplicate_rows),
            sum(int(row["label_conflicts"]) for row in stats_rows),
            stats_path,
            duplicates_path,
        )
    else:
        logger.info("NER sentence deduplication is disabled.")

    run_names = available_run_names()
    run_specs = [resolve_run_spec_from_name(run_name) for run_name in run_names]
    for run_spec in run_specs:
        output_dir = target_dir / run_spec.run_name
        snapshot = create_run_snapshot(
            source_dir,
            run_spec,
            data_args,
            epoch=snapshot_epoch,
            source_corpora=source_corpora,
        )
        write_run_snapshot(output_dir, snapshot)
        if run_spec.pool_name == "pretrain-multi7-full":
            pretrain_snapshot = create_pretrain_snapshot(
                source_dir,
                run_spec,
                data_args,
                epoch=snapshot_epoch,
                source_corpora=source_corpora,
            )
            write_run_snapshot(output_dir, pretrain_snapshot, file_prefix="pretrain-ner")
        logger.info("Prepared SDJT NER split for %s at %s", run_spec.run_name, output_dir)
    logger.info("Prepared %d SDJT NER runs under %s", len(run_specs), target_dir)
