"""Registered text decision models and their local, independent yes/no readouts.

Like TextEmbedder, TextDecider selects a backend by model_name_or_path. Questions
are ordered mappings of IDs to {type: 'noul', instructions: ...}; predict returns
P(true) columns in that order. Dataset preparation and training stay in task code.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import logging
import sys
from abc import ABC, abstractmethod
from contextlib import suppress
from numbers import Real
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from .args.model import ModelArguments

logger = logging.getLogger('core.decider')


class TextDecider(ABC):
    _registry: dict[str, type[TextDecider]] = {}
    name = 'Decision model'
    method: str
    supports_batching = True
    tracks_truncation = False
    parameter_names = {'revision', 'device', 'max_state_tokens'}
    default_parameters: dict = {}

    @classmethod
    def register(cls, *names: str):
        def decorator(subclass):
            for name in names:
                cls._registry[name.strip().lower()] = subclass
            return subclass
        return decorator

    @classmethod
    def get_class(cls, model_args: ModelArguments) -> type[TextDecider]:
        name = (model_args.model_name_or_path or '').strip()
        if not name:
            raise ValueError('model_name_or_path is empty; select a decision model configuration')
        try:
            return cls._registry[name.lower()]
        except KeyError:
            raise ValueError(f'Unknown decision model {name!r}. Available: {sorted(cls._registry)}') from None

    @classmethod
    def create(cls, model_args: ModelArguments, parameters: dict | None = None) -> TextDecider:
        return cls.get_class(model_args)(model_args, parameters or {})

    @classmethod
    def methods(cls) -> set[str]:
        return {backend.method for backend in cls._registry.values()}

    @classmethod
    def for_method(cls, method: str) -> type[TextDecider]:
        for backend in cls._registry.values():
            if backend.method == method:
                return backend
        raise ValueError(f'Unknown decision method: {method}')

    @classmethod
    def validate_parameters(cls, parameters: dict) -> None:
        unknown = set(parameters) - cls.parameter_names
        if unknown:
            raise ValueError(f'Unknown {cls.name} parameters: {", ".join(sorted(unknown))}')
        if not isinstance(parameters.get('revision'), str) or not parameters['revision']:
            raise ValueError(f'{cls.name} requires a pinned model revision')
        limit = parameters.get('max_state_tokens', 8192)
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0):
            raise ValueError(f'{cls.name} max_state_tokens must be a positive integer or null')

    def __init__(self, model_args: ModelArguments, parameters: dict):
        self.validate_parameters(parameters)
        self.model_args, self.parameters = model_args, dict(parameters)
        self.device = torch.device(parameters.get('device', 'cuda'))
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError(f'{self.name} requires an available CUDA device; no model weights were downloaded')
        dtype_name = model_args.dtype or 'bfloat16'
        if dtype_name not in ('bfloat16', 'float16', 'float32'):
            raise ValueError(f'{self.name} model.dtype must be bfloat16, float16 or float32')
        self.dtype = torch.float32 if self.device.type == 'cpu' else getattr(torch, dtype_name)
        self.max_length = model_args.max_seq_length
        if not isinstance(self.max_length, int) or isinstance(self.max_length, bool) or self.max_length <= 0:
            raise ValueError(f'{self.name} model.max_seq_length must be a positive integer')
        self.max_state_tokens = parameters.get('max_state_tokens', 8192)
        self.input_tokens = self.truncated_articles = 0
        self._closed = self._batching_notice = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        self._closed = True

    def counters(self) -> dict:
        result = {'input_tokens': self.input_tokens}
        if self.tracks_truncation:
            result['truncated_articles'] = self.truncated_articles
        return result

    def metadata(self) -> dict:
        return {'device': str(self.device), 'dtype': str(self.dtype), 'max_seq_length': self.max_length,
                'supervision': 'zero_shot_decision', 'num_train': 0}

    def decide(self, texts: str | list[str], questions: dict, batch_size: int | None = None) -> np.ndarray:
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        scores = self.predict(np.asarray([str(row) for row in range(len(texts))]), texts, questions,
                              batch_size if batch_size is not None else self.model_args.batch_size)
        return scores[0] if single else scores

    def predict(self, ids, texts: list[str], questions: dict, batch_size: int = 1) -> np.ndarray:
        if getattr(self, '_closed', False):
            raise RuntimeError(f'{self.name} decider is closed')
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError('Decision batch_size must be a positive integer')
        ids = np.asarray(ids, dtype=str)
        if ids.ndim != 1 or len(ids) != len(texts) or any(not isinstance(text, str) for text in texts):
            raise ValueError('Decision article texts must match the sample IDs')
        if (not isinstance(questions, dict) or not questions
                or any(not isinstance(code, str) or not code or not isinstance(question, dict)
                       or question.get('type') != 'noul' for code, question in questions.items())):
            raise ValueError('Text decisions require a non-empty mapping of string IDs to noul questions')
        if not len(ids):
            return np.empty((0, len(questions)), dtype=np.float32)
        if not self.supports_batching and batch_size > 1 and not getattr(self, '_batching_notice', False):
            logger.info('%s uses sequential per-question forwards; batch_size=%s does not enable GPU batching',
                        self.name, batch_size)
            self._batching_notice = True
        scores = np.asarray(self._predict(ids, texts, questions, batch_size), dtype=np.float32)
        if scores.shape != (len(ids), len(questions)) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError(f'{self.name} returned invalid classification probabilities')
        return scores

    def _answer_scores(self, answers, questions, sample_id) -> list[float]:
        if not isinstance(answers, dict) or set(answers) != set(questions):
            raise ValueError(f'{self.name} returned an incomplete or misaligned topic schema for {sample_id}')
        scores = []
        for code in questions:
            answer = answers[code]
            value = answer.get('noul') if isinstance(answer, dict) else None
            if (not isinstance(answer, dict) or answer.get('type') != 'noul' or isinstance(value, bool)
                    or not isinstance(value, Real) or not np.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f'{self.name} returned an invalid topic probability for {sample_id}/{code}')
            scores.append(value)
        return scores

    def predict_candidates(self, ids, texts: list[str], questions: dict, candidates: list[list[str]],
                           batch_size: int = 1) -> np.ndarray:
        """Score only each article's selected questions; omitted columns remain zero."""
        # Validate the shared schema/batch settings even if every selection is empty.
        self.predict([], [], questions, batch_size)
        ids = np.asarray(ids, dtype=str)
        if ids.ndim != 1 or len(ids) != len(texts) or len(ids) != len(candidates) or any(not isinstance(text, str) for text in texts):
            raise ValueError('Decision candidates and article texts must match the sample IDs')
        columns = {code: column for column, code in enumerate(questions)}
        groups = {}
        for row, selected in enumerate(candidates):
            if not isinstance(selected, (list, tuple)) or any(not isinstance(code, str) or code not in questions for code in selected):
                raise ValueError('Decision candidates must contain only known question IDs')
            selected = set(selected)
            codes = tuple(code for code in questions if code in selected)
            if codes:
                groups.setdefault(codes, []).append(row)
        scores = np.zeros((len(ids), len(questions)), dtype=np.float32)
        for codes, rows in groups.items():
            chosen = {code: questions[code] for code in codes}
            values = self.predict(ids[rows], [texts[row] for row in rows], chosen, batch_size)
            scores[np.ix_(rows, [columns[code] for code in codes])] = values
        return scores

    def _count_tokens(self, usage, sample_id) -> None:
        tokens = usage.get('input_tokens') if isinstance(usage, dict) else None
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
            raise ValueError(f'{self.name} returned invalid input token usage for {sample_id}')
        self.input_tokens += tokens

    def _article_state(self, text, tokenizer, *, skip_special_tokens=False) -> str:
        state = json.dumps({'article': text}, ensure_ascii=False)
        if self.max_state_tokens is not None:
            tokens = tokenizer.encode(state, add_special_tokens=False)
            if len(tokens) > self.max_state_tokens:
                state = tokenizer.decode(tokens[:self.max_state_tokens], skip_special_tokens=skip_special_tokens)
                self.truncated_articles += 1
        return state

    @abstractmethod
    def _predict(self, ids: np.ndarray, texts: list[str], questions: dict, batch_size: int) -> np.ndarray:
        raise NotImplementedError


@TextDecider.register('Cloudflare/clef-flash')
class ClefDecider(TextDecider):
    name, method = 'Clef', 'clef'
    parameter_names = TextDecider.parameter_names | {'head_checkpoint'}
    HEAD_WEIGHTS = 'joint_head.safetensors'
    HEAD_CONFIG = 'joint_head_config.json'
    HEAD_MANIFEST = 'clef_head_training.json'

    @classmethod
    def validate_parameters(cls, parameters):
        super().validate_parameters(parameters)
        if 'head_checkpoint' in parameters and (not isinstance(parameters['head_checkpoint'], str)
                                                 or not parameters['head_checkpoint']):
            raise ValueError('Clef head_checkpoint must be a directory')

    def __init__(self, model_args: ModelArguments, parameters: dict):
        from huggingface_hub import snapshot_download

        super().__init__(model_args, parameters)
        self.head_metadata = None
        if directory := parameters.get('head_checkpoint'):
            self.head_metadata = self._head_manifest(directory)
        self.checkpoint = Path(snapshot_download(
            model_args.model_name_or_path, revision=parameters['revision'], cache_dir=model_args.cache_dir,
        ))
        self.api = self._load_helper(self.checkpoint)
        self.head_config = json.loads((self.checkpoint / self.HEAD_CONFIG).read_text())
        kwargs = {'attn_implementation': model_args.attn_implementation} if model_args.attn_implementation else {}
        self.model, self.tokenizer = self._load_text_model(self.api, self.checkpoint, self.device, self.dtype, **kwargs)
        if directory:
            self.restore_head(directory)

    @staticmethod
    def _load_helper(checkpoint: Path):
        name = '_cannopy_clef_' + hashlib.sha256(str(checkpoint).encode()).hexdigest()[:16]
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, checkpoint / 'joint_schema_model.py')
            if spec is None or spec.loader is None:
                raise ImportError(f'Cannot load Clef inference helper from {checkpoint}')
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                del sys.modules[name]
                raise
        return sys.modules[name]

    @staticmethod
    def _load_text_model(api, checkpoint: Path, device: torch.device, dtype: torch.dtype, **kwargs):
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if tokenizer.pad_token_id is None:
            raise ValueError('Clef tokenizer requires a padding or EOS token for batching')
        backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
            checkpoint, dtype=dtype, device_map={'': str(device)}, **kwargs,
        )
        backbone.config.use_cache = False
        head = api.JointSchemaHead(**json.loads((checkpoint / ClefDecider.HEAD_CONFIG).read_text()))
        head.load_state_dict(load_file(checkpoint / ClefDecider.HEAD_WEIGHTS), strict=True)
        return api.ClefModel(backbone, head.to(device=device, dtype=dtype)).eval(), tokenizer

    @staticmethod
    def default_head_directory(root: Path, subset: str, model_name: str, limit: int | None = None,
                               *, seed: int) -> Path:
        return root / f'{subset}{f".limit{limit}" if limit is not None else ""}.{model_name}.s{seed}'

    @classmethod
    def read_head_manifest(cls, directory, model_name: str, revision: str) -> dict:
        directory = Path(directory)
        for name in (cls.HEAD_WEIGHTS, cls.HEAD_CONFIG, cls.HEAD_MANIFEST):
            if not (directory / name).is_file():
                raise FileNotFoundError(f'Missing Clef head checkpoint file: {directory / name}')
        metadata = json.loads((directory / cls.HEAD_MANIFEST).read_text(encoding='utf-8'))
        if metadata.get('format') != 'cannopy-clef-head-v1':
            raise ValueError('Unsupported Clef head checkpoint format')
        if metadata.get('base_model') != model_name or metadata.get('revision') != revision:
            raise ValueError('Clef head checkpoint requires its original base model and pinned revision')
        if not metadata.get('train_ids') or not metadata.get('validation_ids') or not metadata.get('topic_questions'):
            raise ValueError('Clef head checkpoint is missing training provenance')
        return metadata

    def _head_manifest(self, directory) -> dict:
        metadata = self.read_head_manifest(directory, self.model_args.model_name_or_path, self.parameters['revision'])
        if metadata['max_seq_length'] != self.max_length or metadata['max_state_tokens'] != self.max_state_tokens:
            raise ValueError('Clef head requires the same max_seq_length and max_state_tokens as training')
        return metadata

    def save_head(self, directory, metadata: dict) -> None:
        from safetensors.torch import save_file

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_file({key: value.detach().cpu().contiguous() for key, value in self.model.head.state_dict().items()},
                  directory / self.HEAD_WEIGHTS)
        (directory / self.HEAD_CONFIG).write_text(json.dumps(self.head_config, indent=2) + '\n', encoding='utf-8')
        (directory / self.HEAD_MANIFEST).write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        self.head_metadata = metadata

    def restore_head(self, directory) -> None:
        from safetensors.torch import load_file

        metadata = self._head_manifest(directory)
        self.model.head.load_state_dict(load_file(Path(directory) / self.HEAD_WEIGHTS), strict=True)
        self.head_metadata = metadata

    def encode(self, text: str, questions: dict, sample_id: str = 'unknown'):
        return self.api.encode_record(
            self.tokenizer, {'id': sample_id, 'state': {'article': text}, 'questions': questions},
            max_length=self.max_length, max_state_tokens=self.max_state_tokens,
        )

    def collate(self, records, device='cpu') -> dict:
        return self.api.collate_records(records, self.tokenizer.pad_token_id, torch.device(device))

    def metadata(self):
        result = super().metadata()
        if self.head_metadata:
            result.update(supervision='supervised_decision_head', num_train=len(self.head_metadata['train_ids']),
                          num_validation=len(self.head_metadata['validation_ids']), head_training=self.head_metadata)
        return result

    def _predict(self, ids, texts, questions, batch_size):
        if (getattr(self, 'head_metadata', None)
                and any(self.head_metadata['topic_questions'].get(code) != question for code, question in questions.items())):
            raise ValueError('Clef head requires the same topic questions/descriptions as training')
        columns = {code: index for index, code in enumerate(questions)}
        scores = np.empty((len(ids), len(questions)), dtype=np.float32)
        for start in range(0, len(ids), batch_size):
            encoded = [self.encode(text, questions, str(sample_id))
                       for sample_id, text in zip(ids[start:start + batch_size], texts[start:start + batch_size], strict=True)]
            with torch.inference_mode():
                outputs = self.model(self.collate(encoded, self.device))
            if len(outputs) != len(encoded):
                raise ValueError('Clef returned an unexpected number of article outputs')
            for offset, (record, logits) in enumerate(zip(encoded, outputs, strict=True)):
                if len(logits) != len(questions) or {question.question_id for question in record.questions} != set(questions):
                    raise ValueError('Clef returned an incomplete or misaligned topic schema')
                for question, values in zip(record.questions, logits, strict=True):
                    if values.ndim != 1 or len(values) != len(question.option_ids):
                        raise ValueError('Clef returned invalid option logits')
                    scores[start + offset, columns[question.question_id]] = values.float().softmax(-1)[question.option_ids.index('true')].item()
                self.input_tokens += len(record.input_ids)
        return scores


@TextDecider.register('fastino/GLiDE-2B', 'fastino/GLiDE-9B')
class GlideDecider(TextDecider):
    name, method = 'GLiDE', 'glide'
    tracks_truncation = True
    parameter_names = TextDecider.parameter_names | {'token_budget'}
    default_parameters = {'token_budget': 16384}

    @classmethod
    def validate_parameters(cls, parameters):
        super().validate_parameters(parameters)
        budget = parameters.get('token_budget', 16384)
        if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
            raise ValueError('GLiDE token_budget must be a positive integer')

    def __init__(self, model_args, parameters):
        from huggingface_hub import snapshot_download

        super().__init__(model_args, parameters)
        if model_args.attn_implementation not in (None, 'sdpa'):
            raise ValueError('GLiDE uses SDPA attention; set model.attn_implementation=sdpa')
        self.checkpoint = Path(snapshot_download(
            model_args.model_name_or_path, revision=parameters['revision'], cache_dir=model_args.cache_dir,
            allow_patterns=['glide/*.py', '*.json', '*.safetensors', 'chat_template.jinja'],
        ))
        self.api = self._load_api(self.checkpoint)
        self.model = self.api.Glide.from_pretrained(
            str(self.checkpoint), device=str(self.device), dtype=self.dtype, vision=False,
            max_len=self.max_length, token_budget=parameters.get('token_budget', 16384), name=model_args.short_name,
        )
        self.mask_mapping_compatibility = self._enable_mask_mapping(self.model)

    @staticmethod
    def _load_api(checkpoint: Path):
        # Isolate each pinned release's relative imports without changing sys.path.
        name = '_cannopy_glide_' + hashlib.sha256(str(checkpoint).encode()).hexdigest()[:16]
        if name not in sys.modules:
            package = checkpoint / 'glide'
            spec = importlib.util.spec_from_file_location(name, package / '__init__.py',
                                                        submodule_search_locations=[str(package)])
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

    @staticmethod
    def _mask_mapping_forward(backbone, input_ids=None, attention_mask=None, position_ids=None,
                              past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
        """Preserve GLiDE's bidirectional full/causal linear masks on Transformers 5.8."""
        from transformers.modeling_outputs import BaseModelOutputWithPast

        if past_key_values is not None or use_cache or not isinstance(attention_mask, dict):
            raise ValueError('GLiDE compatibility forward requires explicit masks and uncached inference')
        hidden = backbone.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if position_ids is None:
            position_ids = torch.arange(hidden.shape[1], device=hidden.device).expand(hidden.shape[0], -1)
        if position_ids.ndim != 2:
            raise ValueError('GLiDE text-only inference requires two-dimensional position IDs')
        position_embeddings = backbone.rotary_emb(hidden, position_ids[None].expand(3, -1, -1))
        for layer, kind in zip(backbone.layers[:backbone.config.num_hidden_layers], backbone.config.layer_types, strict=True):
            hidden = layer(hidden, position_embeddings=position_embeddings, attention_mask=attention_mask[kind],
                           position_ids=position_ids, past_key_values=None, use_cache=False, **kwargs)
        return BaseModelOutputWithPast(last_hidden_state=backbone.norm(hidden), past_key_values=None)

    @staticmethod
    def _enable_mask_mapping(model):
        backbone = model.network.backbone
        # Recurrent releases install their own compatible forward.
        if not model.network.recurrence and hasattr(backbone, '_update_linear_attn_mask'):
            backbone.forward = MethodType(GlideDecider._mask_mapping_forward, backbone)
            return True
        return False

    def metadata(self):
        return {**super().metadata(), 'input_tokens_scope': 'sum of per-question prompt tokens',
                'mask_mapping_compatibility': self.mask_mapping_compatibility}

    def _predict(self, ids, texts, questions, batch_size):
        scores = np.empty((len(ids), len(questions)), dtype=np.float32)
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start:start + batch_size]
            requests = [{'state': self._article_state(text, self.model.tokenizer, skip_special_tokens=True),
                         'questions': questions} for text in texts[start:start + batch_size]]
            with torch.inference_mode():
                results = self.model.decide_batch(requests)
            if not isinstance(results, list) or len(results) != len(batch_ids):
                raise ValueError('GLiDE returned an unexpected number of article outputs')
            for offset, (sample_id, result) in enumerate(zip(batch_ids, results, strict=True)):
                if isinstance(result, self.api.RequestError):
                    raise ValueError(f'GLiDE could not score {sample_id}: {result}') from result
                answers = result.get('answers') if isinstance(result, dict) else None
                scores[start + offset] = self._answer_scores(answers, questions, sample_id)
                self._count_tokens(result.get('usage', {}), sample_id)
        return scores


@TextDecider.register('startlux-models/StartLux-Decision-v1.1-4B-Instant')
class StartLuxDecider(TextDecider):
    name, method = 'StartLux', 'startlux'
    tracks_truncation = True
    supports_batching = False

    def __init__(self, model_args, parameters):
        from huggingface_hub import snapshot_download

        super().__init__(model_args, parameters)
        try:
            from startlux_decide.modelcfg import load_model_folder, load_tokenizer
            from startlux_decide.pipeline import Decider, default_controls
        except ModuleNotFoundError as error:
            if not error.name or error.name.split('.')[0] != 'startlux_decide':
                raise
            raise ImportError('StartLux inference requires startlux-decide. Run `uv sync --extra linear-attention` '
                              'in this checkout, then retry the evaluation.') from error
        if model_args.attn_implementation not in (None, 'sdpa'):
            raise ValueError('StartLux bidirectional prompt attention requires model.attn_implementation=sdpa')
        self.checkpoint = Path(snapshot_download(
            model_args.model_name_or_path, revision=parameters['revision'], cache_dir=model_args.cache_dir,
            allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt'],
        ))
        folder = load_model_folder(str(self.checkpoint))
        if not folder.bidirectional or folder.auto_think is not None:
            raise ValueError('StartLux decisions require a bidirectional Instant checkpoint')
        self.tokenizer = load_tokenizer(str(self.checkpoint))
        self.backend = _load_startlux_backend()(
            str(self.checkpoint), self.tokenizer, bidirectional=True, device=str(self.device),
            dtype=str(self.dtype).removeprefix('torch.'), max_length=self.max_length,
        )
        self._runner = asyncio.Runner()
        try:
            self._runner.run(self.backend.start())
            self.decider = Decider(folder, self.backend, defaults=default_controls(folder, think_mode='never'),
                                   context_length=self.max_length, max_images=0)
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
                super().close()

    def metadata(self):
        return {**super().metadata(), 'input_tokens_scope': 'sum of per-question prompt tokens',
                'backend': self.backend_metadata, 'temperature_by_type': self.temperature_by_type, 'output_tokens': 0}

    def _predict(self, ids, texts, questions, batch_size):
        from startlux_decide.controls import RequestError

        scores = np.empty((len(ids), len(questions)), dtype=np.float32)
        for row, (sample_id, text) in enumerate(zip(ids, texts, strict=True)):
            state = self._article_state(text, self.tokenizer)
            try:
                response = self._runner.run(self.decider.decide(state, questions))
            except RequestError as error:
                raise ValueError(f'StartLux article {sample_id}: {error}') from error
            scores[row] = self._answer_scores(response.answers, questions, sample_id)
            if response.usage.get('output_tokens') != 0:
                raise ValueError('StartLux Instant returned invalid token usage or generated text')
            self._count_tokens(response.usage, sample_id)
        return scores


def _load_startlux_backend():
    """Import the optional SDK only when StartLux is selected."""
    from startlux_decide.backends.base import BackendRequestError
    from startlux_decide.backends.transformers import TransformersBackend

    class TextBackend(TransformersBackend):
        def __init__(self, *args, max_length: int, **kwargs):
            super().__init__(*args, **kwargs)
            self.context_length = max_length

        def _load(self):
            from startlux_decide.backends import bidirectional
            from transformers import Qwen3_5ForConditionalGeneration

            self.device = torch.device(self.device_name)
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                self.path, dtype=getattr(torch, self.dtype_name), device_map={'': str(self.device)},
                attn_implementation='sdpa',
            ).eval()
            self.body = self.model.model.language_model
            self.body.config.use_cache = False
            self.head = self.model.get_output_embeddings().weight
            self.check = bidirectional.enable(self.body)

        def _inputs(self, prompt, media):
            if media is not None:
                raise BackendRequestError('StartLux text decisions accept text only')
            inputs = super()._inputs(prompt, media)
            length = inputs['input_ids'].shape[1]
            if length > self.context_length:
                raise BackendRequestError(
                    f'StartLux question prompt has {length} tokens, exceeding model.max_seq_length={self.context_length}; '
                    'reduce max_state_tokens or increase max_seq_length'
                )
            return inputs

    return TextBackend
