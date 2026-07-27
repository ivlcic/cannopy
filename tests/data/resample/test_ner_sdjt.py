import csv

import pytest

from src.app.args.data import DataArguments
from src.app.ner import NerSample
from src.data.resample.ner_sdjt import (
    CROATIAN_ABLATION_EVAL_LANGUAGES,
    HR_WIKIANN_SOURCE,
    compute_language_token_budgets,
    deduplicate_corpora,
    resolve_run_spec_from_name,
    write_dedup_reports,
)


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
    stats_path, duplicates_path = write_dedup_reports(
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
