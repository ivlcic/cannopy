import csv
import logging

import pytest

from src.data.prepare import ner as ner_prepare
from src.app.dataset import NerSamplesLoader
from src.app.ner import NER_CSV_COLUMNS, NerSample
from src.data.prepare.ner import (
    ConllDatasetParser,
    NerDatasetParser,
    NerUkParser,
    WannParser,
    write_outputs,
    write_remap_stats,
)
from src.data.resample.ner_sdjt import _read_split_file, _write_split_csv


def test_remap_stats_include_exact_inexact_o_and_zero_count(tmp_path) -> None:
    parser = NerDatasetParser(
        tmp_path,
        {
            'B-place': 'B-LOC',
            'B-event': 'B-MISC',
            'B-time': 'O',
            'I-time': 'O',
        },
        corpus='sample',
        label_remap_exact={
            'B-place': True,
            'B-event': False,
        },
    )

    parser._map_label('Ljubljana', 'B-place')
    parser._map_label('festival', 'B-event')
    parser._map_label('today', 'B-time')
    parser._map_label('Acme', 'B-ORG')
    parser._map_label('word', 'O')

    output_path = write_remap_stats(tmp_path, [parser])
    with output_path.open(encoding='utf-8', newline='') as stats_file:
        rows = {
            (row['source_label'], row['target_label']): row
            for row in csv.DictReader(stats_file)
        }

    assert output_path.name == 'ner-remap-stats.csv'
    assert rows[('B-place', 'B-LOC')]['meaning_match'] == 'exact'
    assert rows[('B-event', 'B-MISC')]['meaning_match'] == 'inexact'
    assert rows[('B-time', 'O')]['meaning_match'] == 'not_applicable'
    assert rows[('I-time', 'O')]['token_count'] == '0'
    assert rows[('B-ORG', 'B-ORG')]['configured_remap'] == 'False'
    assert ('O', 'O') not in rows


def test_remap_requires_a_marker_for_each_non_o_target(tmp_path) -> None:
    with pytest.raises(ValueError, match='missing markers'):
        NerDatasetParser(
            tmp_path,
            {'B-place': 'B-LOC'},
            corpus='sample',
        )


def test_ner_uk_assigns_i_prefix_to_entity_continuations(tmp_path) -> None:
    txt_path = tmp_path / 'sample.txt'
    ann_path = tmp_path / 'sample.ann'
    txt_path.write_text('Alpha Beta outside', encoding='utf-8')
    ann_path.write_text('T1\tORG\t0\t10\tAlpha Beta\n', encoding='utf-8')
    parser = NerUkParser(
        tmp_path,
        {'ORG': 'ORG'},
        corpus='ner-uk',
        label_remap_exact={'ORG': True},
    )

    sentences = parser._parse_pair(txt_path, ann_path)

    assert sentences == [
        NerSample(
            tokens=['Alpha', 'Beta', 'outside'],
            labels=['B-ORG', 'I-ORG', 'O'],
            corpus_name='ner-uk',
            doc_id='sample',
            sent_id='1',
        )
    ]
    assert parser.label_remap_counts[('ORG', 'ORG')] == 2


def test_conll_parser_uses_source_document_and_sentence_ids(tmp_path) -> None:
    source_path = tmp_path / 'sample.conllu'
    source_path.write_text(
        '# newdoc id = article-7\n'
        '# sent_id = article-7.3\n'
        'Ljubljana\tB-loc\n'
        'today\tO\n',
        encoding='utf-8',
    )
    parser = ConllDatasetParser(
        tmp_path,
        {'B-loc': 'B-LOC'},
        corpus='sample-corpus',
        label_remap_exact={'B-loc': True},
    )

    sentences, has_labels = parser._parse_conll_file(
        source_path,
        token_idx=0,
        label_selector=lambda parts: parts[-1],
    )

    assert has_labels is True
    assert sentences == [
        NerSample(
            tokens=['Ljubljana', 'today'],
            labels=['B-LOC', 'O'],
            corpus_name='sample-corpus',
            doc_id='article-7',
            sent_id='article-7.3',
        )
    ]


def test_wikiann_croatian_is_kept_as_a_separate_training_source(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        ner_prepare,
        'logger',
        logging.getLogger(__name__),
        raising=False,
    )
    source_dir = tmp_path / 'hr-wann'
    source_dir.mkdir()
    (source_dir / 'train').write_text(
        'hr:Zagreb\tB-LOC\n'
        'hr:je\tO\n'
        '\n',
        encoding='utf-8',
    )
    parser = WannParser(
        tmp_path,
        {'hr-wann': 'hr-wikiann'},
        {},
        corpus='wikiann-hr',
    )

    parsed = parser.parse()

    assert list(parsed) == ['hr-wikiann']
    assert parsed['hr-wikiann'] == [
        NerSample(
            tokens=['Zagreb', 'je'],
            labels=['B-LOC', 'O'],
            corpus_name='wikiann-hr',
            doc_id='hr-wann/train',
            sent_id='1',
        )
    ]


def test_writer_and_loader_preserve_ner_metadata(tmp_path) -> None:
    sample = NerSample(
        tokens=['Janez', 'Novak'],
        labels=['B-PER', 'I-PER'],
        corpus_name='ssj500k',
        doc_id='ssj1',
        sent_id='ssj1.1.1',
    )
    write_outputs(tmp_path, {'sl': [sample]}, file_suffix='.train')

    csv_path = tmp_path / 'ner-sl.train.csv'
    with csv_path.open(encoding='utf-8', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        assert reader.fieldnames == NER_CSV_COLUMNS
        assert next(reader) == sample.to_csv_row()

    loader = NerSamplesLoader.__new__(NerSamplesLoader)
    assert loader._load_split_file(csv_path) == [sample]


def test_loader_accepts_distinct_training_and_evaluation_languages(tmp_path) -> None:
    training_sample = NerSample(
        tokens=['Zagreb'],
        labels=['B-LOC'],
        corpus_name='wikiann-hr',
    )
    evaluation_sample = NerSample(
        tokens=['Sofija'],
        labels=['B-LOC'],
        corpus_name='bsnlp',
    )
    write_outputs(tmp_path, {'hr-wikiann': [training_sample]}, file_suffix='.train')
    write_outputs(tmp_path, {'bg': [evaluation_sample]}, file_suffix='.eval')
    write_outputs(tmp_path, {'bg': [evaluation_sample]}, file_suffix='.test')

    loader = NerSamplesLoader(
        tmp_path,
        ['hr-wikiann'],
        split_languages={
            'train': ['hr-wikiann'],
            'eval': ['bg'],
            'test': ['bg'],
        },
    )

    assert loader.samples_by_lang['train'] == {'hr-wikiann': [training_sample]}
    assert loader.samples_by_lang['eval'] == {'bg': [evaluation_sample]}
    assert loader.samples_by_lang['test'] == {'bg': [evaluation_sample]}


def test_resampling_preserves_metadata_while_harmonizing_labels(tmp_path) -> None:
    target_path = tmp_path / 'target.csv'
    sample = NerSample(
        tokens=['festival', 'Ljubljana'],
        labels=['B-MISC', 'B-LOC'],
        corpus_name='sample-corpus',
        doc_id='article-1',
        sent_id='4',
    )
    write_outputs(tmp_path, {'sl': [sample]}, file_suffix='.eval')
    source_path = tmp_path / 'ner-sl.eval.csv'

    loaded = _read_split_file(source_path)
    _write_split_csv(target_path, loaded)

    assert loaded == [
        NerSample(
            tokens=sample.tokens,
            labels=['O', 'B-LOC'],
            corpus_name=sample.corpus_name,
            doc_id=sample.doc_id,
            sent_id=sample.sent_id,
        )
    ]
    loader = NerSamplesLoader.__new__(NerSamplesLoader)
    assert loader._load_split_file(target_path) == loaded
