import numpy as np
import pytest

from src.app.labeler import BinaryLabeler, MulticlassLabeler, MultilabelLabeler


def test_binary_labeler_round_trip():
    labeler = BinaryLabeler(labels=['yes', 'no'])

    assert labeler.computed is True
    assert labeler.encode('no') == 0
    assert labeler.encode('yes') == 1
    assert labeler.decode(0) == 'no'
    assert labeler.decode(1) == 'yes'
    assert labeler.label2id == {'no': 0, 'yes': 1}
    assert labeler.id2label == {0: 'no', 1: 'yes'}
    assert labeler.labels2ids() == {'no': 0, 'yes': 1}
    assert labeler.ids2labels() == {0: 'no', 1: 'yes'}
    assert labeler.num_labels == 1


def test_binary_labeler_requires_two_values():
    labeler = BinaryLabeler(labels=['yes'])

    with pytest.raises(ValueError):
        labeler.fit()


def test_multiclass_labeler_round_trip():
    labeler = MulticlassLabeler(labels=['dog', 'bird', 'cat'])

    assert labeler.encode('bird') == 0
    assert labeler.encode('cat') == 1
    assert labeler.encode('dog') == 2
    assert labeler.decode(0) == 'bird'
    assert labeler.decode(1) == 'cat'
    assert labeler.decode(2) == 'dog'


def test_multiclass_labeler_fit_uses_unique_sorted_classes():
    labeler = MulticlassLabeler(labels=['dog', 'bird', 'dog', 'cat', 'bird'])

    assert labeler.classes == ['bird', 'cat', 'dog']
    assert labeler.label2id == {'bird': 0, 'cat': 1, 'dog': 2}
    assert labeler.id2label == {0: 'bird', 1: 'cat', 2: 'dog'}
    assert labeler.labels2ids() == {'bird': 0, 'cat': 1, 'dog': 2}
    assert labeler.num_labels == 3


def test_multiclass_labeler_collect_updates_classes_only():
    labeler = MulticlassLabeler()

    labeler.collect(['dog', ['bird', 'cat'], 'dog'])
    assert labeler.computed is False
    labeler.fit()

    assert labeler.classes == ['bird', 'cat', 'dog']


def test_multiclass_labeler_keeps_default_label():
    labeler = MulticlassLabeler(labels=['dog', 'bird', 'cat'], default_label='bird')

    assert labeler.default_label == 'bird'
    assert labeler.encode(labeler.default_label) == 0


def test_multiclass_labeler_uses_custom_sorter():
    labeler = MulticlassLabeler(
        labels=['B-ORG', 'O', 'I-ORG', 'B-PER'],
        sorter=lambda label: (
            '',
            label,
        ) if label == 'O' else (
            label.split('-', 1)[1],
            label.split('-', 1)[0],
        ),
    )

    assert labeler.classes == ['O', 'B-ORG', 'I-ORG', 'B-PER']
    assert labeler.label2id == {'O': 0, 'B-ORG': 1, 'I-ORG': 2, 'B-PER': 3}


def test_multilabel_labeler_round_trip():
    labeler = MultilabelLabeler(labels=['news', 'sports', 'tech'])

    encoded = labeler.encode([['news', 'tech'], ['sports'], []])

    assert encoded.tolist() == [[1, 0, 1], [0, 1, 0], [0, 0, 0]]
    assert labeler.decode(encoded) == [('news', 'tech'), ('sports',), ()]
    assert labeler.decode(np.array([1, 0, 1])) == ['news', 'tech']
