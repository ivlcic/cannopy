import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from startlux_decide.backends.fake import FakeBackend, FakeTokenizer
from startlux_decide.pipeline import Decision

from src.app import entrypoint
from src.app.args.model import ModelArguments
from src.eval.sequence import iptc_31
from src.eval.sequence.iptc_31 import decision
from src.eval.sequence.iptc_31.methods import startlux


@pytest.fixture
def release(tmp_path, monkeypatch):
    import huggingface_hub
    from startlux_decide import modelcfg

    (tmp_path / 'config.json').write_text(json.dumps({'jev_prompt_attention': 'bidirectional'}))
    (tmp_path / 'decision_config.json').write_text(json.dumps({
        'prompt_attention': 'bidirectional', 'temperature_by_type': {'noul': 1.2662},
    }))

    def download(repo_id, **kwargs):
        assert repo_id == 'startlux-models/StartLux-Decision-v1.1-4B-Instant'
        assert kwargs['revision'] == 'a' * 40
        assert '*.safetensors' in kwargs['allow_patterns']
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, 'snapshot_download', download)
    monkeypatch.setattr(modelcfg, 'load_tokenizer', lambda path: FakeTokenizer())
    model = ModelArguments(model_name_or_path='startlux-models/StartLux-Decision-v1.1-4B-Instant',
                           max_seq_length=16384, dtype='bfloat16', attn_implementation='sdpa')
    return model, {'revision': 'a' * 40, 'device': 'cpu', 'max_state_tokens': 8192}


def test_startlux_uses_official_prompts_calibration_and_instant_readout(release, monkeypatch):
    class Backend(FakeBackend):
        def __init__(self, path, tokenizer, bidirectional, device, dtype, max_length):
            assert bidirectional and device == 'cpu' and dtype == 'float32' and max_length == 16384
            super().__init__(tokenizer)
            self.closed = False

        async def readout(self, prompt, letter_ids):
            assert list(letter_ids) == [ord('A'), ord('B')]
            assert prompt.endswith('<think>\n\n</think>\n\n')
            assert 'A) yes\nB) no' in prompt
            self.calls.append(prompt)
            return [2., 0.] if 'arts' in prompt else [0., 2.]

        def describe(self):
            return {'prompt_attention': 'bidirectional'}

        async def close(self):
            self.closed = True

    monkeypatch.setattr(startlux, '_TextBackend', Backend)
    predictor = startlux.StartLuxPredictor(*release)
    questions = {'01000000': {'type': 'noul', 'instructions': 'arts'},
                 '20000002': {'type': 'noul', 'instructions': 'sport'}}
    try:
        scores = predictor.predict(np.asarray(['sl:1', 'sl:2']), ['Članek 1', 'Članek 2'], questions, 2)
        true_probability = 1 / (1 + np.exp(-2 / 1.2662))
        np.testing.assert_allclose(scores, [[true_probability, 1 - true_probability]] * 2)
        assert len(predictor.backend.calls) == 4
        assert predictor.input_tokens == sum(len(predictor.tokenizer.encode(p)) for p in predictor.backend.calls)
        assert predictor.decider.defaults.think_mode == 'never' and not predictor.decider.reads_images
        assert all('"article": "Članek' in p and 'label' not in p for p in predictor.backend.calls)
        predictor.max_state_tokens = 5
        predictor.predict(np.asarray(['sl:3']), ['Long article'], questions, 1)
        assert predictor.truncated_articles == 1
        assert all('Evidence:\n{"art\n\nQuestion:' in p for p in predictor.backend.calls[-2:])
        assert 'Question: sport' in predictor.backend.calls[-1]
    finally:
        predictor.close()
    predictor.close()  # Idempotent cleanup also covers caller error paths.
    assert predictor.backend.closed and predictor._runner is None


def test_startlux_closes_worker_when_loading_fails(release, monkeypatch):
    instances = []

    class BrokenBackend(startlux._TextBackend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

        def _load(self):
            raise RuntimeError('load failed')

    monkeypatch.setattr(startlux, '_TextBackend', BrokenBackend)
    with pytest.raises(RuntimeError, match='load failed'):
        startlux.StartLuxPredictor(*release)
    assert instances[0]._executor._shutdown
    assert all(not thread.is_alive() for thread in instances[0]._executor._threads)


def test_startlux_text_loader_uses_sdpa_and_release_attention_checks_without_processor(monkeypatch):
    from startlux_decide.backends import bidirectional
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    body = SimpleNamespace(config=SimpleNamespace(use_cache=True))
    model = SimpleNamespace(model=SimpleNamespace(language_model=body),
                            get_output_embeddings=lambda: SimpleNamespace(weight='letter weights'))
    model.eval = lambda: model

    def load(path, **kwargs):
        assert kwargs == {'dtype': torch.float32, 'device_map': {'': 'cpu'}, 'attn_implementation': 'sdpa'}
        return model

    monkeypatch.setattr(Qwen3_5ForConditionalGeneration, 'from_pretrained', load)
    monkeypatch.setattr(AutoProcessor, 'from_pretrained', lambda *args, **kwargs: pytest.fail('Text only'))
    monkeypatch.setattr(bidirectional, 'enable', lambda loaded: {'checked': loaded is body})
    backend = startlux._TextBackend('checkpoint', FakeTokenizer(), True, device='cpu', dtype='float32', max_length=8)
    backend._load()
    assert backend.check == {'checked': True} and not body.config.use_cache and not backend.reads_images
    assert backend._inputs('short', None)['input_ids'].shape == (1, 5)
    with pytest.raises(ValueError, match='exceeding model.max_seq_length=8'):
        backend._inputs('overlong prompt', None)
    with pytest.raises(ValueError, match='text only'):
        backend._inputs('short', object())
    backend._executor.shutdown()


def test_startlux_release_attention_checks_on_real_tiny_qwen_decoder():
    from startlux_decide.backends import bidirectional
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

    config = Qwen3_5TextConfig(
        vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
        layer_types=['linear_attention', 'full_attention'],
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000,
                         'partial_rotary_factor': 0.5, 'mrope_section': [1, 1, 2]},
    )
    config._attn_implementation = 'sdpa'
    body = Qwen3_5TextModel(config).eval()
    report = bidirectional.enable(body)
    assert report['full_attention_layers'] == [1] and report['layers'] == 2
    assert report['late_token_reach'] > 0 and report['padding_invariance_max_diff'] < 1e-3
    assert body.config._attn_implementation == 'sdpa'


def test_startlux_requires_cuda_before_downloading(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', lambda *args, **kwargs: pytest.fail('No download'))
    with pytest.raises(RuntimeError, match='no model weights were downloaded'):
        startlux.StartLuxPredictor(ModelArguments(), {'revision': 'a' * 40})


@pytest.mark.parametrize('answers, usage, message', [
    ({}, {'input_tokens': 1, 'output_tokens': 0}, 'incomplete or misaligned'),
    ({'01000000': {'type': 'noul', 'noul': float('nan')}}, {}, 'invalid topic probability'),
    ({'01000000': {'type': 'noul', 'noul': 1.2}}, {}, 'invalid topic probability'),
    ({'01000000': {'type': 'choice', 'noul': .7}}, {}, 'invalid topic probability'),
    ({'01000000': {'type': 'noul', 'noul': .7}}, {'input_tokens': -1, 'output_tokens': 0}, 'invalid token usage'),
    ({'01000000': {'type': 'noul', 'noul': .7}}, {'input_tokens': 1, 'output_tokens': 1}, 'generated text'),
])
def test_startlux_rejects_misaligned_scores_and_generated_answers(answers, usage, message):
    import asyncio

    async def decide(*args):
        return Decision(answers=answers, usage=usage, timing_ms={})

    predictor = startlux.StartLuxPredictor.__new__(startlux.StartLuxPredictor)
    predictor.decider = SimpleNamespace(decide=decide)
    predictor.max_state_tokens = None
    predictor.input_tokens = predictor.truncated_articles = 0
    with asyncio.Runner() as runner:
        predictor._runner = runner
        with pytest.raises(ValueError, match=message):
            predictor.predict(np.asarray(['sl:1']), ['Article'], {'01000000': {'type': 'noul'}}, 1)


def test_startlux_evaluation_closes_worker_after_prediction_error(tmp_path, monkeypatch):
    paths = entrypoint._ensure_dirs(tmp_path, 'eval', 'sequence', 'iptc-31')
    merged, _ = entrypoint._load_and_merge_configs(Path('conf'), 'sequence', 'iptc-31', ['starlux1.1-4b', 'sl'])
    model, args, _, _ = entrypoint._parse_hf_args(merged)
    monkeypatch.setattr(iptc_31, 'paths', paths, raising=False)
    retained = iptc_31.EmbeddedSplit(np.asarray(['sl:1']), np.asarray(['Article']), np.zeros((1, 31)))
    monkeypatch.setattr(decision, '_read_text_split', lambda *args: (retained.ids, ['Article'], retained.labels))
    monkeypatch.setattr(decision, '_filter_test_split', lambda *args: (retained, []))
    closed = []

    class Predictor:
        input_tokens = 0

        def __init__(self, *args):
            pass

        def predict(self, *args):
            raise ValueError('prediction failed')

        def close(self):
            closed.append(True)

    monkeypatch.setattr(decision, 'StartLuxPredictor', Predictor)
    with pytest.raises(ValueError, match='prediction failed'):
        decision.evaluate_decision(args, model, paths, logging.getLogger(__name__), method='startlux')
    assert closed == [True]
