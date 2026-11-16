"""Pinned GLiDE decision API for independent IPTC topic probabilities."""

import hashlib
import importlib.util
import json
import sys
from numbers import Real
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from .....app.args.model import ModelArguments


def _load_glide_api(checkpoint: Path):
    # Keep each checkpoint's relative imports in its own package namespace.
    name = '_cannopy_glide_' + hashlib.sha256(str(checkpoint).encode()).hexdigest()[:16]
    if name not in sys.modules:
        package = checkpoint / 'glide'
        spec = importlib.util.spec_from_file_location(
            name, package / '__init__.py', submodule_search_locations=[str(package)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f'Cannot load GLiDE inference package from {checkpoint}')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            for key in list(sys.modules):
                if key == name or key.startswith(name + '.'):
                    del sys.modules[key]
            raise
    return sys.modules[name]


def _mask_mapping_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                          past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
    """GLiDE's supplied full/linear masks on the project's Transformers 5.8.

    Qwen3.5 in 5.8 rebuilds a causal mask and rejects a mask mapping. GLiDE
    requires bidirectional full attention, with causal linear-attention layers.
    This is the uncached text-only layer loop, retaining those supplied masks.
    """
    from transformers.modeling_outputs import BaseModelOutputWithPast

    if past_key_values is not None or use_cache or not isinstance(attention_mask, dict):
        raise ValueError('GLiDE compatibility forward requires explicit masks and uncached inference')
    hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
    if position_ids is None:
        position_ids = torch.arange(hidden.shape[1], device=hidden.device).expand(hidden.shape[0], -1)
    if position_ids.ndim != 2:
        raise ValueError('GLiDE text-only inference requires two-dimensional position IDs')
    rotary_positions = position_ids[None].expand(3, -1, -1)
    position_embeddings = self.rotary_emb(hidden, rotary_positions)
    for layer, kind in zip(self.layers[:self.config.num_hidden_layers], self.config.layer_types, strict=True):
        hidden = layer(hidden, position_embeddings=position_embeddings, attention_mask=attention_mask[kind],
                       position_ids=position_ids, past_key_values=None, use_cache=False, **kwargs)
    return BaseModelOutputWithPast(last_hidden_state=self.norm(hidden), past_key_values=None)


def _enable_mask_mapping(model):
    # Newer Qwen implementations accept GLiDE's mask mapping natively. The
    # pinned 5.8 implementation exposes _update_linear_attn_mask instead.
    # Recurrent GLiDE releases already install their own mask-aware forward.
    backbone = model.network.backbone
    if not model.network.recurrence and hasattr(backbone, '_update_linear_attn_mask'):
        backbone.forward = MethodType(_mask_mapping_forward, backbone)
        return True
    return False


class GlidePredictor:
    def __init__(self, model_args: ModelArguments, parameters: dict):
        from huggingface_hub import snapshot_download

        self.device = torch.device(parameters.get('device', 'cuda'))
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('GLiDE requires an available CUDA device; no model weights were downloaded')
        dtype_name = model_args.dtype or 'bfloat16'
        if dtype_name not in ('bfloat16', 'float16', 'float32'):
            raise ValueError('GLiDE model.dtype must be bfloat16, float16 or float32')
        if model_args.attn_implementation not in (None, 'sdpa'):
            raise ValueError('GLiDE uses SDPA attention; set model.attn_implementation=sdpa')
        if not isinstance(model_args.max_seq_length, int) or isinstance(model_args.max_seq_length, bool) or model_args.max_seq_length <= 0:
            raise ValueError('GLiDE model.max_seq_length must be a positive integer')
        self.dtype = torch.float32 if self.device.type == 'cpu' else getattr(torch, dtype_name)
        self.max_state_tokens = parameters.get('max_state_tokens', 8192)
        self.input_tokens, self.truncated_articles = 0, 0
        checkpoint = Path(snapshot_download(
            model_args.model_name_or_path, revision=parameters['revision'], cache_dir=model_args.cache_dir,
            allow_patterns=['glide/*.py', '*.json', '*.safetensors', 'chat_template.jinja'],
        ))
        self.api = _load_glide_api(checkpoint)
        self.model = self.api.Glide.from_pretrained(
            str(checkpoint), device=str(self.device), dtype=self.dtype, vision=False,
            max_len=model_args.max_seq_length, token_budget=parameters.get('token_budget', 16384),
            name=model_args.short_name,
        )
        self.mask_mapping_compatibility = _enable_mask_mapping(self.model)

    def _state(self, text: str) -> str:
        # Match the published API's JSON rendering. A truncated state remains
        # text, preserving the article prefix and leaving every question intact.
        state = json.dumps({'article': text}, ensure_ascii=False)
        if self.max_state_tokens is not None:
            ids = self.model.tokenizer(state, add_special_tokens=False).input_ids
            if len(ids) > self.max_state_tokens:
                state = self.model.tokenizer.decode(ids[:self.max_state_tokens], skip_special_tokens=True)
                self.truncated_articles += 1
        return state

    def predict(self, ids: np.ndarray, texts: list[str], questions: dict, batch_size: int) -> np.ndarray:
        if len(ids) != len(texts):
            raise ValueError('GLiDE article texts must match the sample IDs')
        codes = list(questions)
        scores = np.empty((len(ids), len(codes)), dtype=np.float32)
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start:start + batch_size]
            requests = [{'state': self._state(text), 'questions': questions} for text in texts[start:start + batch_size]]
            with torch.inference_mode():
                results = self.model.decide_batch(requests)
            if not isinstance(results, list) or len(results) != len(batch_ids):
                raise ValueError('GLiDE returned an unexpected number of article outputs')
            for offset, (sample_id, result) in enumerate(zip(batch_ids, results, strict=True)):
                if isinstance(result, self.api.RequestError):
                    raise ValueError(f'GLiDE could not score {sample_id}: {result}') from result
                answers = result.get('answers') if isinstance(result, dict) else None
                if not isinstance(answers, dict) or set(answers) != set(codes):
                    raise ValueError(f'GLiDE returned an incomplete or misaligned topic schema for {sample_id}')
                for column, code in enumerate(codes):
                    answer = answers[code]
                    value = answer.get('noul') if isinstance(answer, dict) else None
                    if (not isinstance(answer, dict) or answer.get('type') != 'noul'
                            or isinstance(value, bool) or not isinstance(value, Real)
                            or not np.isfinite(value) or not 0 <= value <= 1):
                        raise ValueError(f'GLiDE returned an invalid topic probability for {sample_id}, {code}')
                    scores[start + offset, column] = value
                usage = result.get('usage', {})
                tokens = usage.get('input_tokens') if isinstance(usage, dict) else None
                if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
                    raise ValueError(f'GLiDE returned invalid input token usage for {sample_id}')
                self.input_tokens += tokens
        return scores
