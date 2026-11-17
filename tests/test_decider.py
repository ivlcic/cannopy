import numpy as np
import pytest
import torch

from src.app.args.model import ModelArguments
from src.app.decider import ClefDecider, GlideDecider, StartLuxDecider, TextDecider


@pytest.mark.parametrize('model, backend', [
    (' Cloudflare/CLEF-FLASH ', ClefDecider),
    ('fastino/GLiDE-2B', GlideDecider),
    ('fastino/GLiDE-9B', GlideDecider),
    ('startlux-models/StartLux-Decision-v1.1-4B-Instant', StartLuxDecider),
])
def test_model_registry_selects_backend_without_loading_weights(model, backend):
    assert TextDecider.get_class(ModelArguments(model_name_or_path=model)) is backend
    assert TextDecider.for_method(backend.method) is backend


def test_decider_factory_lifecycle_and_common_prediction_contract(monkeypatch):
    monkeypatch.setattr(TextDecider, '_registry', dict(TextDecider._registry))

    @TextDecider.register('test/decision')
    class Backend(TextDecider):
        name, method = 'Test', 'test'

        def _predict(self, ids, texts, questions, batch_size):
            self.input_tokens += len(texts) * 10
            self.observed_batch_size = batch_size
            return np.asarray([[.8, .2]] * len(texts))

    model = ModelArguments(model_name_or_path='test/decision', batch_size=2)
    questions = {'topic_a': {'type': 'noul'}, 'topic_b': {'type': 'noul'}}
    with TextDecider.create(model, {'revision': 'a' * 40, 'device': 'cpu'}) as decider:
        assert isinstance(decider, Backend) and decider.dtype == torch.float32
        np.testing.assert_allclose(decider.decide('Article', questions), [.8, .2])
        np.testing.assert_allclose(decider.decide(['One', 'Two'], questions), [[.8, .2], [.8, .2]])
        assert decider.decide([], questions).shape == (0, 2)
        assert decider.counters() == {'input_tokens': 30} and decider.observed_batch_size == 2
        assert decider.metadata()['supervision'] == 'zero_shot_decision'
        with pytest.raises(ValueError, match='sample IDs'):
            decider.predict(['1'], [], questions)
        with pytest.raises(ValueError, match='batch_size'):
            decider.predict(['1'], ['Article'], questions, 0)
        with pytest.raises(ValueError, match='noul questions'):
            decider.decide('Article', {'topic': {'type': 'choice'}})
    with pytest.raises(RuntimeError, match='closed'):
        decider.decide('Article', questions)


@pytest.mark.parametrize('scores', [np.asarray([[np.nan]]), np.asarray([[1.1]]), np.zeros((2, 1))])
def test_common_predict_rejects_invalid_backend_probabilities(scores):
    class Backend(TextDecider):
        name, method = 'Test', 'test'

        def _predict(self, *args):
            return scores

    decider = Backend(ModelArguments(), {'revision': 'a' * 40, 'device': 'cpu'})
    with pytest.raises(ValueError, match='invalid classification probabilities'):
        decider.decide('Article', {'topic': {'type': 'noul'}})


def test_factory_rejects_unknown_models_and_options():
    with pytest.raises(ValueError, match='Unknown decision model'):
        TextDecider.create(ModelArguments(model_name_or_path='unknown/model'))
    with pytest.raises(ValueError, match='pinned model revision'):
        ClefDecider.validate_parameters({})
    with pytest.raises(ValueError, match='Unknown Clef parameters'):
        ClefDecider.validate_parameters({'revision': 'a' * 40, 'typo': 1})


def test_candidate_questions_are_batched_and_mapped_to_full_label_columns():
    calls = []

    class Backend(TextDecider):
        def _predict(self, ids, texts, questions, batch_size):
            calls.append((ids.tolist(), tuple(questions), batch_size))
            return np.asarray([[.8 if code == 'a' else .2 for code in questions]] * len(ids))

    decider = Backend(ModelArguments(), {'revision': 'a'*40, 'device': 'cpu'})
    questions = {code: {'type': 'noul'} for code in ('a', 'b', 'c')}
    scores = decider.predict_candidates(['1', '2', '3', '4'], ['One', 'Two', 'Three', 'Four'], questions,
                                        [['b', 'a'], [], ['a', 'b', 'a'], ['c']], batch_size=4)
    np.testing.assert_allclose(scores, [[.8, .2, 0], [0, 0, 0], [.8, .2, 0], [0, 0, .2]])
    assert calls == [(['1', '3'], ('a', 'b'), 4), (['4'], ('c',), 4)]
    with pytest.raises(ValueError, match='known question IDs'):
        decider.predict_candidates(['1'], ['Article'], questions, [['unknown']])
    with pytest.raises(ValueError, match='match the sample IDs'):
        decider.predict_candidates(['1'], ['Article'], questions, [])
