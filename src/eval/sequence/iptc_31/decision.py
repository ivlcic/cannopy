"""Local decision models over prepared IPTC text and independent topic questions."""

import hashlib
import importlib.util
import json
import sys
import time
from contextlib import ExitStack
from logging import Logger
from pathlib import Path

import numpy as np
import torch

from . import (
    EmbeddedSplit, EvaluationResult, _filter_test_split, _method_settings,
    _pool_evaluations, _select_subsets, _write_evaluation,
)
from .features import _read_text_split
from .methods.glide import GlidePredictor
from .methods.startlux import StartLuxPredictor
from ....app.args.data import DataArguments
from ....app.args.model import ModelArguments
from ....app.args.runtime import Paths
from ....app.labeler import MultilabelLabeler
from ....data.prepare.newsmon import get_subset_name


QUESTION_TEMPLATE = (
    "Does this news article belong to the IPTC topic '{name}' ({code})? "
    "Evaluate this topic independently. Multiple topics, including both broader "
    "and narrower topics, may apply."
)
DEFINITION_TEMPLATE = '\nIPTC definition: {description}'


def _topic_questions(definitions: list[dict], use_label_descriptions: bool = False) -> dict:
    if not isinstance(use_label_descriptions, bool):
        raise ValueError('use_label_descriptions must be a boolean')
    questions = {}
    for label in definitions:
        code, name = label['iptc_code'], label['label_name']
        if not isinstance(code, str) or len(code) != 8 or not code.isdigit() or code in questions:
            raise ValueError('Decision evaluation requires unique eight-digit IPTC code strings')
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f'Missing IPTC topic name for {code}')
        instructions = QUESTION_TEMPLATE.format(name=name, code=code)
        if use_label_descriptions:
            description = label.get('label_description')
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f'Missing IPTC topic description for {code}')
            instructions += DEFINITION_TEMPLATE.format(description=description)
        questions[code] = {'type': 'noul', 'instructions': instructions}
    if not questions:
        raise ValueError('Decision evaluation requires topic definitions in data.attributes.labels')
    return dict(sorted(questions.items()))


def _load_helper(checkpoint: Path):
    # Use a revision-specific module name, without changing sys.path or mixing
    # helpers from other downloaded models. Dataclasses need a registered module.
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


def _load_text_model(api, checkpoint: Path, device: torch.device, dtype: torch.dtype, **kwargs):
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    # The release loader always builds an AutoProcessor, including image/video
    # processors. IPTC evaluation only uses text, so load its tokenizer directly.
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        raise ValueError('Clef tokenizer requires a padding or EOS token for batching')
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
        checkpoint, dtype=dtype, device_map={'': str(device)}, **kwargs,
    )
    backbone.config.use_cache = False
    head = api.JointSchemaHead(**json.loads((checkpoint / 'joint_head_config.json').read_text()))
    head.load_state_dict(load_file(checkpoint / 'joint_head.safetensors'), strict=True)
    head = head.to(device=device, dtype=dtype)
    return api.ClefModel(backbone, head).eval(), tokenizer


class ClefPredictor:
    def __init__(self, model_args: ModelArguments, parameters: dict):
        from huggingface_hub import snapshot_download

        self.device = torch.device(parameters.get('device', 'cuda'))
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Clef requires an available CUDA device; no model weights were downloaded')
        dtype_name = model_args.dtype or 'bfloat16'
        if dtype_name not in ('bfloat16', 'float16', 'float32'):
            raise ValueError('Clef model.dtype must be bfloat16, float16 or float32')
        self.dtype = torch.float32 if self.device.type == 'cpu' else getattr(torch, dtype_name)
        self.max_length = model_args.max_seq_length
        if self.max_length <= 0:
            raise ValueError('Clef model.max_seq_length must be positive')
        self.max_state_tokens = parameters.get('max_state_tokens', 8192)
        self.input_tokens = 0
        checkpoint = Path(snapshot_download(
            model_args.model_name_or_path, revision=parameters['revision'], cache_dir=model_args.cache_dir,
        ))
        self.api = _load_helper(checkpoint)
        kwargs = {'attn_implementation': model_args.attn_implementation} if model_args.attn_implementation else {}
        self.model, self.tokenizer = _load_text_model(
            self.api, checkpoint, device=self.device, dtype=self.dtype, **kwargs,
        )

    def predict(self, ids: np.ndarray, texts: list[str], questions: dict, batch_size: int) -> np.ndarray:
        codes = list(questions)
        columns = {code: index for index, code in enumerate(codes)}
        scores = np.empty((len(ids), len(codes)), dtype=np.float32)
        for start in range(0, len(ids), batch_size):
            encoded = [self.api.encode_record(
                self.tokenizer,
                {'id': str(sample_id), 'state': {'article': text}, 'questions': questions},
                max_length=self.max_length, max_state_tokens=self.max_state_tokens,
            ) for sample_id, text in zip(ids[start:start + batch_size], texts[start:start + batch_size], strict=True)]
            batch = self.api.collate_records(encoded, self.tokenizer.pad_token_id, self.device)
            with torch.inference_mode():
                outputs = self.model(batch)
            if len(outputs) != len(encoded):
                raise ValueError('Clef returned an unexpected number of article outputs')
            for offset, (record, logits) in enumerate(zip(encoded, outputs, strict=True)):
                if len(logits) != len(codes) or {question.question_id for question in record.questions} != set(codes):
                    raise ValueError('Clef returned an incomplete or misaligned topic schema')
                for question, question_logits in zip(record.questions, logits, strict=True):
                    if question_logits.ndim != 1 or len(question_logits) != len(question.option_ids):
                        raise ValueError('Clef returned invalid option logits')
                    probability = question_logits.float().softmax(-1)[question.option_ids.index('true')].item()
                    scores[start + offset, columns[question.question_id]] = probability
                self.input_tokens += len(record.input_ids)
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError('Clef returned invalid topic probabilities')
        return scores


def evaluate_decision(data_args: DataArguments, model_args: ModelArguments, paths: Paths, logger: Logger,
                      method: str = 'clef') -> None:
    predictors = {'clef': ('Clef', ClefPredictor), 'glide': ('GLiDE', GlidePredictor),
                  'startlux': ('StartLux', StartLuxPredictor)}
    if method not in predictors:
        raise ValueError(f'Unknown decision method: {method}')
    name, factory = predictors[method]
    options = data_args.attributes.get('sequence', {})
    if 'trials' in options.get('optuna', {}):
        raise ValueError(f'Optuna supports rae_xmc, log_reg and svm; {name} is a zero-shot decision method')
    if not model_args.short_name or not model_args.model_name_or_path:
        raise ValueError('Select a decision model configuration, for example -c clef-flash, -c glide-2b or -c starlux1.1-4b')
    if options.get('features', {}).get('type') != 'text':
        raise ValueError(f'{name} requires text features; select its model configuration')
    parameters, threshold = _method_settings(options, method)
    parameters.setdefault('use_label_descriptions', False)
    allowed = {'revision', 'device', 'max_state_tokens', 'max_samples', 'use_label_descriptions'}
    if method == 'glide':
        allowed.add('token_budget')
        parameters.setdefault('token_budget', 16384)
    unknown = set(parameters) - allowed
    if unknown:
        raise ValueError(f'Unknown {name} parameters: {", ".join(sorted(unknown))}')
    if not isinstance(parameters.get('revision'), str) or not parameters['revision']:
        raise ValueError(f'{name} requires a pinned model revision')
    batch_size = options.get('batch_size', 1)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0 or not 0 <= threshold <= 1:
        raise ValueError(f'{name} requires a positive integer batch_size and threshold in [0, 1]')
    max_samples = parameters.get('max_samples')
    state_limit = parameters.get('max_state_tokens')
    for field, value in (('max_samples', max_samples), ('max_state_tokens', state_limit)):
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
            raise ValueError(f'{name} {field} must be a positive integer or null')
    if method == 'glide' and (not isinstance(parameters['token_budget'], int)
                              or isinstance(parameters['token_budget'], bool) or parameters['token_budget'] <= 0):
        raise ValueError('GLiDE token_budget must be a positive integer')
    use_descriptions = parameters['use_label_descriptions']
    questions = _topic_questions(data_args.attributes.get('labels', []), use_descriptions)
    labeler = MultilabelLabeler(labels=list(questions))
    prepare_dir = paths.get_script_ctx_path('data', 'prepare')
    datasets = []
    for args in _select_subsets(data_args, model_args, test_only=True):
        subset = get_subset_name(args)
        test_file = prepare_dir / f'{subset}.test.jsonl'
        ids, texts, labels = _read_text_split(test_file, labeler)
        before_filter = len(ids)
        # The common row filter works on text arrays as well as feature matrices.
        retained, excluded = _filter_test_split(EmbeddedSplit(ids, np.asarray(texts), labels), options.get('exclude_test_ids', []))
        datasets.append((subset, test_file, before_filter, excluded, retained))

    logger.info('Loading %s model %s at revision %s', name, model_args.model_name_or_path, parameters['revision'])
    loaded = time.perf_counter()
    with ExitStack() as resources:
        predictor = factory(model_args, parameters)
        if close := getattr(predictor, 'close', None):
            resources.callback(close)
        load_seconds = time.perf_counter() - loaded
        evaluations = []
        suffix = ('.descriptions' if use_descriptions else '') + (f'.limit{max_samples}' if max_samples is not None else '')
        for subset, test_file, before_filter, excluded, retained in datasets:
            ids, texts, labels = retained.ids[:max_samples], retained.embeddings[:max_samples].tolist(), retained.labels[:max_samples]
            logger.info('%s %s: scoring %s articles after excluding %s test IDs', method, subset, len(ids), len(excluded))
            input_tokens = predictor.input_tokens
            truncated_articles = getattr(predictor, 'truncated_articles', 0)
            started = time.perf_counter()
            scores = np.asarray(predictor.predict(ids, texts, questions, batch_size), dtype=np.float32)
            duration = time.perf_counter() - started
            if scores.shape != labels.shape or not np.isfinite(scores).all():
                raise ValueError(f'{name} returned invalid classification scores')
            result = {
                'dataset': data_args.dataset_name, 'subset': subset + suffix,
                'model': model_args.short_name, 'model_name_or_path': model_args.model_name_or_path,
                'method': method, 'supervision': 'zero_shot_decision', 'num_train': 0,
                'num_test': len(ids), 'num_test_before_filter': before_filter,
                'num_test_excluded': len(excluded), 'excluded_test_ids': excluded,
                'num_test_before_limit': len(retained.ids), 'num_labels': labeler.num_labels,
                'label_order': labeler.classes, 'batch_size': batch_size, 'threshold': threshold,
                'parameters': parameters, 'question_template': QUESTION_TEMPLATE + (DEFINITION_TEMPLATE if use_descriptions else ''),
                'prompt_variant': 'name_and_definition' if use_descriptions else 'name_only',
                'topic_questions': questions, 'feature_type': 'text', 'test_data': str(test_file),
                'method_seconds': duration, 'model_load_seconds': load_seconds,
                'input_tokens': predictor.input_tokens - input_tokens,
                'device': str(predictor.device), 'dtype': str(predictor.dtype),
            }
            if use_descriptions:
                result['label_description_metadata'] = data_args.attributes.get('label_description_metadata', {})
            if method in ('glide', 'startlux'):
                result.update(max_seq_length=model_args.max_seq_length,
                              input_tokens_scope='sum of per-question prompt tokens',
                              num_test_truncated=predictor.truncated_articles - truncated_articles)
            if method == 'glide':
                result['mask_mapping_compatibility'] = predictor.mask_mapping_compatibility
            if method == 'startlux':
                result.update(backend=predictor.backend_metadata, temperature_by_type=predictor.temperature_by_type,
                              output_tokens=0)
            evaluations.append(EvaluationResult(result, ids, labels, scores))
        if data_args.source.select.subset:
            final = evaluations[0]
        else:
            final = _pool_evaluations(evaluations, data_args.dataset_name + suffix)
            for field in ('input_tokens', 'num_test_before_limit'):
                final.metadata[field] = sum(evaluation.metadata[field] for evaluation in evaluations)
            if method in ('glide', 'startlux'):
                final.metadata['num_test_truncated'] = sum(evaluation.metadata['num_test_truncated'] for evaluation in evaluations)
        _write_evaluation(final)


def evaluate_clef(data_args: DataArguments, model_args: ModelArguments, paths: Paths, logger: Logger) -> None:
    evaluate_decision(data_args, model_args, paths, logger, method='clef')
