import csv

import pytest

from src.app.args.data import DataArguments
from src.app.ner import NerSample
from src.data.resample.ner_sdjt import (
    CROATIAN_ABLATION_EVAL_LANGUAGES,
    HR_WIKIANN_SOURCE,
    compute_language_token_budgets,
    harmonize_label,
    resolve_run_spec_from_name,
)
from src.data.split.ner import deduplicate_corpora, write_dedup_reports


def _sample(
    sentence: str,
    corpus_name: str,
    doc_id: str,
    sent_id: str,
    labels: list[str] | None = None,
) -> NerSample:
    tokens = sentence.split(' ')
    return NerSample(
        tokens=tokens,
        labels=labels or ['O'] * len(tokens),
        corpus_name=corpus_name,
        doc_id=doc_id,
        sent_id=sent_id,
    )


@pytest.mark.parametrize(
    ('label', 'expected'),
    [
        ('O', 'O'),
        ('per', 'B-PER'),
        ('U-ORG', 'B-ORG'),
        ('L-LOC', 'I-LOC'),
        ('B-MISC', 'O'),
        ('X-PER', 'O'),
    ],
)
def test_harmonize_label_maps_to_sdjt_label_space(label: str, expected: str) -> None:
    assert harmonize_label(label) == expected


def test_deduplication_prefers_test_then_eval_then_train_per_language() -> None:
    kept_test = _sample(
        'Isto besedilo',
        'ssj500k',
        'test-doc',
        '1',
        labels=['O', 'B-PER'],
    )
    duplicate_eval = _sample(
        'isto besedilo',
        'senticoref',
        'eval-doc',
        '2',
    )
    duplicate_train = _sample(
        'ISTO BESEDILO',
        'elexis-wsd',
        'train-doc',
        '3',
        labels=['O', 'B-PER'],
    )
    unique_train = _sample('Drugo besedilo', 'ssj500k', 'train-doc', '4')
    same_text_other_language = _sample(
        'isto besedilo',
        'setimes',
        'sr-doc',
        '1',
    )
    source = {
        'train': {
            'sl': [duplicate_train, unique_train],
            'sr': [same_text_other_language],
        },
        'eval': {'sl': [duplicate_eval], 'sr': []},
        'test': {'sl': [kept_test], 'sr': []},
    }

    deduplicated, stats_rows, duplicate_rows = deduplicate_corpora(source)

    assert deduplicated['test']['sl'] == [kept_test]
    assert deduplicated['eval']['sl'] == []
    assert deduplicated['train']['sl'] == [unique_train]
    assert deduplicated['train']['sr'] == [same_text_other_language]
    assert len(duplicate_rows) == 2
    assert {row['kept_split'] for row in duplicate_rows} == {'test'}
    assert sum(not row['labels_match'] for row in duplicate_rows) == 1

    stats = {
        (row['language'], row['split'], row['corpus_name']): row
        for row in stats_rows
    }
    assert stats[('sl', 'eval', 'senticoref')]['duplicates_removed'] == 1
    assert stats[('sl', 'eval', 'senticoref')]['label_conflicts'] == 1
    assert stats[('sl', 'train', 'elexis-wsd')]['duplicates_removed'] == 1


def test_deduplication_normalizes_unicode_and_writes_audit_reports(tmp_path) -> None:
    composed = _sample('Čas Café', 'corpus-a', 'doc-1', '1')
    decomposed = _sample('čas Cafe\u0301', 'corpus-b', 'doc-2', '1')
    source = {
        'train': {'sl': [decomposed]},
        'eval': {'sl': []},
        'test': {'sl': [composed]},
    }

    _, stats_rows, duplicate_rows = deduplicate_corpora(source)
    stats_path, duplicates_path, duplicates_data_path = write_dedup_reports(
        tmp_path,
        stats_rows,
        duplicate_rows,
    )

    assert len(duplicate_rows) == 1
    with stats_path.open(encoding='utf-8', newline='') as stats_file:
        assert len(list(csv.DictReader(stats_file))) == 2
    with duplicates_path.open(encoding='utf-8', newline='') as duplicates_file:
        rows = list(csv.DictReader(duplicates_file))
    assert rows[0]['removed_corpus_name'] == 'corpus-b'
    assert rows[0]['kept_corpus_name'] == 'corpus-a'
    with duplicates_data_path.open(
        encoding='utf-8',
        newline='',
    ) as duplicates_data_file:
        data_rows = list(csv.DictReader(duplicates_data_file))
    assert data_rows[0]['removed_sentence'] == 'čas Cafe\u0301'
    assert data_rows[0]['removed_labels'] == 'O O'
    assert data_rows[0]['kept_sentence'] == 'Čas Café'
    assert data_rows[0]['kept_labels'] == 'O O'


def test_deduplication_ignores_malformed_samples_seen_before_valid_duplicates() -> None:
    malformed_test = _sample(
        'Isto besedilo',
        'corpus-a',
        'test-doc',
        '1',
        labels=['O'],
    )
    valid_train = _sample(
        'isto besedilo',
        'corpus-b',
        'train-doc',
        '2',
    )
    source = {
        'train': {'sl': [valid_train]},
        'eval': {'sl': []},
        'test': {'sl': [malformed_test]},
    }

    deduplicated, _, duplicate_rows = deduplicate_corpora(source)

    assert deduplicated['test']['sl'] == [malformed_test]
    assert deduplicated['train']['sl'] == [valid_train]
    assert duplicate_rows == []


def test_croatian_ablation_run_specs_share_evaluation_languages() -> None:
    base = resolve_run_spec_from_name('multi7-no-hr')
    manual = resolve_run_spec_from_name('multi7-plus-hr500k')
    wikiann = resolve_run_spec_from_name('multi7-plus-hr-wikiann')

    assert base.eval_languages == CROATIAN_ABLATION_EVAL_LANGUAGES
    assert manual.eval_languages == CROATIAN_ABLATION_EVAL_LANGUAGES
    assert wikiann.eval_languages == CROATIAN_ABLATION_EVAL_LANGUAGES
    assert 'hr' not in base.train_languages
    assert 'hr' in manual.train_languages
    assert HR_WIKIANN_SOURCE in wikiann.train_languages
    assert 'hr' not in wikiann.train_languages
    assert base.uses_macro_eval is True
    assert manual.uses_macro_eval is True
    assert wikiann.uses_macro_eval is True


def test_croatian_ablation_uses_common_base_quota_without_oversampling() -> None:
    def sentences(count: int, corpus: str) -> list[NerSample]:
        return [
            _sample(f'token-{index}', corpus, f'doc-{index}', '1')
            for index in range(count)
        ]

    base_train = {
        lang: sentences(3 if lang == 'sr' else 5, lang)
        for lang in CROATIAN_ABLATION_EVAL_LANGUAGES
    }
    data_args = DataArguments()

    for run_name, extra_source in (
        ('multi7-no-hr', None),
        ('multi7-plus-hr500k', 'hr'),
        ('multi7-plus-hr-wikiann', HR_WIKIANN_SOURCE),
    ):
        run_spec = resolve_run_spec_from_name(run_name)
        train_by_lang = dict(base_train)
        if extra_source:
            train_by_lang[extra_source] = sentences(8, extra_source)

        budgets = compute_language_token_budgets(
            train_by_lang,
            run_spec,
            data_args,
        )

        assert set(budgets) == set(run_spec.train_languages)
        assert set(budgets.values()) == {3}

    wikiann_spec = resolve_run_spec_from_name('multi7-plus-hr-wikiann')
    too_small = dict(base_train)
    too_small[HR_WIKIANN_SOURCE] = sentences(2, HR_WIKIANN_SOURCE)
    with pytest.raises(ValueError, match='do not oversample'):
        compute_language_token_budgets(too_small, wikiann_spec, data_args)
