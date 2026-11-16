"""StartLux Instant decisions using the release's Transformers reference engine."""

import asyncio
from contextlib import suppress
from numbers import Real

import numpy as np
import torch
from startlux_decide.backends.base import BackendRequestError
from startlux_decide.backends.transformers import TransformersBackend

from .....app.args.model import ModelArguments


class _TextBackend(TransformersBackend):
    """Keep the release readout and attention checks, without image processors."""

    def __init__(self, *args, max_length: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_length = max_length

    def _load(self):
        from startlux_decide.backends import bidirectional
        from transformers import Qwen3_5ForConditionalGeneration

        self.device = torch.device(self.device_name)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.path, dtype=getattr(torch, self.dtype_name),
            device_map={'': str(self.device)}, attn_implementation='sdpa',
        ).eval()
        self.body = self.model.model.language_model
        self.body.config.use_cache = False
        self.head = self.model.get_output_embeddings().weight
        # This verifies causal equivalence, padding invariance and that later
        # prompt tokens reach earlier ones in the full-attention layers.
        self.check = bidirectional.enable(self.body)

    def _inputs(self, prompt, media):
        if media is not None:
            raise BackendRequestError('IPTC StartLux evaluation accepts text only')
        inputs = super()._inputs(prompt, media)
        length = inputs['input_ids'].shape[1]
        if length > self.context_length:
            raise BackendRequestError(
                f'StartLux question prompt has {length} tokens, exceeding model.max_seq_length={self.context_length}; '
                'reduce max_state_tokens or increase max_seq_length'
            )
        return inputs


class StartLuxPredictor:
    def __init__(self, model_args: ModelArguments, parameters: dict):
        from huggingface_hub import snapshot_download
        from startlux_decide.modelcfg import load_model_folder, load_tokenizer
        from startlux_decide.pipeline import Decider, default_controls

        self.device = torch.device(parameters.get('device', 'cuda'))
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('StartLux requires an available CUDA device; no model weights were downloaded')
        dtype_name = model_args.dtype or 'bfloat16'
        if dtype_name not in ('bfloat16', 'float16', 'float32'):
            raise ValueError('StartLux model.dtype must be bfloat16, float16 or float32')
        self.dtype = torch.float32 if self.device.type == 'cpu' else getattr(torch, dtype_name)
        if model_args.attn_implementation not in (None, 'sdpa'):
            raise ValueError('StartLux bidirectional prompt attention requires model.attn_implementation=sdpa')
        max_length = model_args.max_seq_length
        if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length <= 0:
            raise ValueError('StartLux model.max_seq_length must be a positive integer')
        self.max_state_tokens = parameters.get('max_state_tokens', 8192)
        self.input_tokens = self.truncated_articles = 0
        checkpoint = snapshot_download(
            model_args.model_name_or_path, revision=parameters['revision'], cache_dir=model_args.cache_dir,
            allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt'],
        )
        folder = load_model_folder(checkpoint)
        if not folder.bidirectional or folder.auto_think is not None:
            raise ValueError('StartLux evaluation requires a bidirectional Instant checkpoint')
        self.tokenizer = load_tokenizer(checkpoint)
        self.backend = _TextBackend(
            checkpoint, self.tokenizer, bidirectional=True, device=str(self.device),
            dtype=str(self.dtype).removeprefix('torch.'), max_length=max_length,
        )
        self._runner = asyncio.Runner()
        try:
            self._runner.run(self.backend.start())
            self.decider = Decider(
                folder, self.backend, defaults=default_controls(folder, think_mode='never'),
                context_length=max_length, max_images=0,
            )
            self.backend_metadata = self.backend.describe()
            self.temperature_by_type = dict(folder.temperature_by_type)
        except BaseException:
            with suppress(Exception):
                self.close()
            raise

    def close(self):
        if self._runner is not None:
            try:
                self._runner.run(self.backend.close())
            finally:
                self._runner.close()
                self._runner = None

    def predict(self, ids: np.ndarray, texts: list[str], questions: dict, batch_size: int) -> np.ndarray:
        from startlux_decide.controls import RequestError
        from startlux_decide.prompt import state_text

        scores = np.empty((len(ids), len(questions)), dtype=np.float32)
        # The reference engine runs one question per forward pass on one GPU
        # worker. Decider schedules the independent topic questions internally.
        for row, (sample_id, text) in enumerate(zip(ids, texts, strict=True)):
            state = state_text({'article': text})
            if self.max_state_tokens is not None:
                tokens = self.tokenizer.encode(state, add_special_tokens=False)
                if len(tokens) > self.max_state_tokens:
                    state = self.tokenizer.decode(tokens[:self.max_state_tokens], skip_special_tokens=False)
                    self.truncated_articles += 1
            try:
                response = self._runner.run(self.decider.decide(state, questions))
            except RequestError as error:
                raise ValueError(f'StartLux article {sample_id}: {error}') from error
            if set(response.answers) != set(questions):
                raise ValueError(f'StartLux returned an incomplete or misaligned topic schema for {sample_id}')
            for column, code in enumerate(questions):
                answer = response.answers[code]
                probability = answer.get('noul')
                if (answer.get('type') != 'noul' or isinstance(probability, bool) or not isinstance(probability, Real)
                        or not np.isfinite(probability) or not 0 <= probability <= 1):
                    raise ValueError(f'StartLux returned an invalid topic probability for {sample_id}/{code}')
                scores[row, column] = probability
            input_tokens = response.usage.get('input_tokens')
            if (not isinstance(input_tokens, int) or isinstance(input_tokens, bool) or input_tokens < 0
                    or response.usage.get('output_tokens') != 0):
                raise ValueError('StartLux Instant returned invalid token usage or generated text')
            self.input_tokens += input_tokens
        return scores
