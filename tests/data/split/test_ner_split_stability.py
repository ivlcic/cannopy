from src.app.ner import NerSample
from src.data.split.ner import _split_language_data


def _sentences(prefix: str) -> list[NerSample]:
    return [
        NerSample(
            tokens=[f'{prefix}-{index}'],
            labels=['O'],
            corpus_name=prefix,
        )
        for index in range(20)
    ]


def test_adding_a_source_does_not_change_other_language_splits() -> None:
    base = {
        'bg': _sentences('bg'),
        'pl': _sentences('pl'),
    }
    extended = {
        'bg': list(base['bg']),
        'hr-wikiann': _sentences('hr-wikiann'),
        'pl': list(base['pl']),
    }

    base_splits = _split_language_data(base, 0.8, 0.1, 0.1, 2611)
    extended_splits = _split_language_data(extended, 0.8, 0.1, 0.1, 2611)

    for split in ('train', 'eval', 'test'):
        assert base_splits[split]['bg'] == extended_splits[split]['bg']
        assert base_splits[split]['pl'] == extended_splits[split]['pl']
