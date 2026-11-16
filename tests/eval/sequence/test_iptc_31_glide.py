import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.app.args.data import DataArguments
from src.app.args.model import ModelArguments
from src.eval.sequence.iptc_31 import decision
from src.app import decider as glide


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[ord(char) for char in text])

    def decode(self, ids, **kwargs):
        return ''.join(chr(value) for value in ids)


class RequestError(ValueError):
    pass


def test_glide_batches_articles_maps_topic_ids_and_reads_noul_not_confidence():
    batches = []

    class Model:
        tokenizer = CharacterTokenizer()

        def decide_batch(self, requests):
            assert not torch.is_grad_enabled()
            batches.append(requests)
            for request in requests:
                assert set(request) == {'state', 'questions'}
                assert 'label' not in json.loads(request['state'])
            return [{'answers': {'20000002': {'type': 'noul', 'noul': 0.2, 'confidence': 0.6},
                                 '01000000': {'type': 'noul', 'noul': 0.8, 'confidence': 0.6}},
                     'usage': {'input_tokens': 200}} for _ in requests]

    predictor = glide.GlideDecider.__new__(glide.GlideDecider)
    predictor.api = SimpleNamespace(RequestError=RequestError)
    predictor.model = Model()
    predictor.max_state_tokens = 1000
    predictor.input_tokens, predictor.truncated_articles = 0, 0
    ids = np.asarray(['sl:1', 'sl:2', 'sl:3'])
    texts = [f'Članek {sample_id}' for sample_id in ids]
    questions = {code: {'type': 'noul'} for code in ('01000000', '20000002')}
    scores = predictor.predict(ids, texts, questions, batch_size=2)
    np.testing.assert_allclose(scores, [[0.8, 0.2]] * 3, atol=1e-7)
    assert [len(batch) for batch in batches] == [2, 1]
    assert [json.loads(request['state'])['article'] for batch in batches for request in batch] == texts
    assert predictor.input_tokens == 600 and predictor.truncated_articles == 0

    predictor.max_state_tokens = 10
    assert predictor._article_state('Članek, daljši od omejitve', predictor.model.tokenizer, skip_special_tokens=True) == json.dumps(
        {'article': 'Članek, daljši od omejitve'}, ensure_ascii=False,
    )[:10]
    assert predictor.truncated_articles == 1


@pytest.mark.parametrize('result, message', [
    (RequestError('prompt exceeds the maximum context length'), 'could not score sl:1'),
    ({'answers': {}, 'usage': {'input_tokens': 1}}, 'incomplete or misaligned'),
    ({'answers': {'01000000': {'type': 'noul', 'noul': float('nan')}}, 'usage': {'input_tokens': 1}}, 'invalid topic probability'),
    ({'answers': {'01000000': {'type': 'noul', 'noul': 1.2}}, 'usage': {'input_tokens': 1}}, 'invalid topic probability'),
    ({'answers': {'01000000': {'type': 'noul', 'noul': 0.7}}, 'usage': {'input_tokens': -1}}, 'invalid input token usage'),
])
def test_glide_rejects_per_article_errors_and_invalid_scores(result, message):
    predictor = glide.GlideDecider.__new__(glide.GlideDecider)
    predictor.api = SimpleNamespace(RequestError=RequestError)
    predictor.model = SimpleNamespace(decide_batch=lambda requests: [result], tokenizer=CharacterTokenizer())
    predictor.max_state_tokens = None
    predictor.input_tokens, predictor.truncated_articles = 0, 0
    with pytest.raises(ValueError, match=message):
        predictor.predict(np.asarray(['sl:1']), ['Article'], {'01000000': {'type': 'noul'}}, 1)


def test_glide_loader_uses_pinned_text_only_api_without_processor(tmp_path, monkeypatch):
    calls = []
    model_args = ModelArguments(short_name='glide-2b', model_name_or_path='fastino/GLiDE-2B',
                                max_seq_length=16384, dtype='bfloat16', attn_implementation='sdpa')
    parameters = {'revision': 'a' * 40, 'device': 'cuda', 'token_budget': 2048, 'max_state_tokens': 8192}

    def download(repo_id, **kwargs):
        assert repo_id == 'fastino/GLiDE-2B' and kwargs['revision'] == parameters['revision']
        assert kwargs['allow_patterns'] == ['glide/*.py', '*.json', '*.safetensors', 'chat_template.jinja']
        return str(tmp_path)

    def load(path, **kwargs):
        assert path == str(tmp_path)
        assert kwargs == {'device': 'cuda', 'dtype': torch.bfloat16, 'vision': False,
                          'max_len': 16384, 'token_budget': 2048, 'name': 'glide-2b'}
        calls.append(path)
        return SimpleNamespace(network=SimpleNamespace(recurrence=None, backbone=SimpleNamespace()))

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr('huggingface_hub.snapshot_download', download)
    monkeypatch.setattr(glide.GlideDecider, '_load_api', staticmethod(lambda path: SimpleNamespace(Glide=SimpleNamespace(from_pretrained=load))))
    predictor = glide.GlideDecider(model_args, parameters)
    assert calls == [str(tmp_path)] and predictor.device.type == 'cuda'
    assert predictor.dtype == torch.bfloat16 and not predictor.mask_mapping_compatibility

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='no model weights were downloaded'):
        glide.GlideDecider(model_args, parameters)
    assert calls == [str(tmp_path)]


def test_glide_compatibility_preserves_bidirectional_and_linear_masks_with_real_qwen():
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
    backbone = Qwen3_5TextModel(config).eval()
    model = SimpleNamespace(network=SimpleNamespace(backbone=backbone, recurrence=None))
    glide.GlideDecider._enable_mask_mapping(model)
    ids = torch.tensor([[1, 2, 3], [0, 4, 5]])
    padding = torch.tensor([[1, 1, 1], [0, 1, 1]])
    masks = {'full_attention': padding[:, None, None, :].bool(), 'linear_attention': padding}
    observed = []

    def capture(module, args, kwargs):
        observed.append(kwargs['attention_mask'])

    hooks = [layer.register_forward_pre_hook(capture, with_kwargs=True) for layer in backbone.layers]
    with torch.inference_mode():
        positions = (padding.cumsum(-1) - 1).clamp_min(0)
        together = backbone(input_ids=ids, position_ids=positions, attention_mask=masks, use_cache=False).last_hidden_state
        assert observed[0] is masks['linear_attention'] and observed[1] is masks['full_attention']
        # Noncausal full attention can see future tokens. A generic causal
        # language-model forward would silently change this model's semantics.
        assert observed[1][0, 0, 0, 2]
        separate = backbone(input_ids=ids[1:, 1:], position_ids=torch.tensor([[0, 1]]),
                            attention_mask={'full_attention': torch.ones(1, 1, 1, 2, dtype=torch.bool),
                                            'linear_attention': None}, use_cache=False).last_hidden_state
    for hook in hooks:
        hook.remove()
    torch.testing.assert_close(together[1, -1], separate[0, -1], rtol=1e-4, atol=1e-5)


def test_glide_rejects_optuna_before_model_loading():
    args = DataArguments(attributes={'sequence': {'optuna': {'trials': 50}}})
    with pytest.raises(ValueError, match='GLiDE is a zero-shot'):
        decision.evaluate_decision(args, ModelArguments(), None, None, method='glide')
