"""Local Clef inference over prepared IPTC text and independent topic questions."""

import hashlib
import importlib.util
import json
import sys
import time
from logging import Logger
from pathlib import Path

import numpy as np
import torch

from . import (
    EmbeddedSplit, EvaluationResult, _filter_test_split, _method_settings,
    _pool_evaluations, _select_subsets, _write_evaluation,
)
from .features import _read_text_split
from .methods.bge_m3_clef import RAG_INSTRUCTION, BgeM3ClefRetriever, encode_with_examples
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


def _topic_questions(definitions: list[dict]) -> dict:
    names = {}
    for label in definitions:
        code, name = label['iptc_code'], label['label_name']
        if not isinstance(code, str) or len(code) != 8 or not code.isdigit() or code in names:
            raise ValueError('Clef requires unique eight-digit IPTC code strings')
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f'Missing IPTC topic name for {code}')
        names[code] = name
    if not names:
        raise ValueError('Clef requires topic definitions in data.attributes.labels')
    return {code: {
        'type': 'noul', 'instructions': QUESTION_TEMPLATE.format(name=name, code=code),
    } for code, name in sorted(names.items())}


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
        self.retrieval_audit = []
        checkpoint = Path(snapshot_download(
            model_args.model_name_or_path, revision=parameters['revision'], cache_dir=model_args.cache_dir,
        ))
        self.api = _load_helper(checkpoint)
        kwargs = {'attn_implementation': model_args.attn_implementation} if model_args.attn_implementation else {}
        self.model, self.tokenizer = _load_text_model(
            self.api, checkpoint, device=self.device, dtype=self.dtype, **kwargs,
        )

    def predict(self, ids: np.ndarray, texts: list[str], questions: dict, batch_size: int,
                *, examples: list[list[dict]] | None = None, max_excerpt_tokens: int = 256) -> np.ndarray:
        codes = list(questions)
        columns = {code: index for index, code in enumerate(codes)}
        scores = np.empty((len(ids), len(codes)), dtype=np.float32)
        if examples is not None:
            if len(examples) != len(ids):
                raise ValueError('RAG contexts must match the target article IDs')
            schema_tokens = len(self.api.encode_record(
                self.tokenizer, {'state': '', 'questions': questions}, max_length=self.max_length,
            ).input_ids)
        for start in range(0, len(ids), batch_size):
            encoded = []
            for row in range(start, min(start + batch_size, len(ids))):
                if examples is None:
                    record = self.api.encode_record(
                        self.tokenizer, {'id': str(ids[row]), 'state': {'article': texts[row]}, 'questions': questions},
                        max_length=self.max_length, max_state_tokens=self.max_state_tokens,
                    )
                else:
                    record, audit = encode_with_examples(
                        self.api, self.tokenizer, str(ids[row]), texts[row], questions, examples[row],
                        max_length=self.max_length, max_state_tokens=self.max_state_tokens,
                        max_excerpt_tokens=max_excerpt_tokens, schema_tokens=schema_tokens,
                    )
                    self.retrieval_audit.append(audit)
                encoded.append(record)
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


def evaluate_clef(data_args: DataArguments, model_args: ModelArguments, paths: Paths, logger: Logger,
                   method: str = 'clef') -> None:
    options = data_args.attributes.get('sequence', {})
    if 'trials' in options.get('optuna', {}):
        raise ValueError('Optuna supports rae_xmc, log_reg and svm; Clef is a zero-shot decision method')
    if not model_args.short_name or not model_args.model_name_or_path:
        raise ValueError('Select a Clef model configuration, for example -c clef-flash')
    if options.get('features', {}).get('type') != 'text':
        raise ValueError('Clef requires text features; select -c clef-flash')
    if method == 'bge-m3_clef':
        methods = options.get('methods', {})
        settings = {**methods.get('clef', {}), **methods.get('bge-m3_clef', {})}
        parameters, threshold = _method_settings({**options, 'methods': {method: settings}}, method)
        for name, value in {'retrieval_model': 'bge-m3', 'top_k': 3, 'max_excerpt_tokens': 256}.items():
            parameters.setdefault(name, value)
    else:
        parameters, threshold = _method_settings(options, 'clef')
    allowed = {'revision', 'device', 'max_state_tokens', 'max_samples'}
    if method == 'bge-m3_clef':
        allowed |= {'retrieval_model', 'top_k', 'max_excerpt_tokens'}
    unknown = set(parameters) - allowed
    if unknown:
        raise ValueError(f'Unknown Clef parameters: {", ".join(sorted(unknown))}')
    if not isinstance(parameters.get('revision'), str) or not parameters['revision']:
        raise ValueError('Clef requires a pinned model revision')
    batch_size = options.get('batch_size', 1)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0 or not 0 <= threshold <= 1:
        raise ValueError('Clef requires a positive integer batch_size and threshold in [0, 1]')
    max_samples = parameters.get('max_samples')
    state_limit = parameters.get('max_state_tokens')
    for name, value in (('max_samples', max_samples), ('max_state_tokens', state_limit)):
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
            raise ValueError(f'Clef {name} must be a positive integer or null')
    if method == 'bge-m3_clef' and (
        not isinstance(parameters['max_excerpt_tokens'], int) or isinstance(parameters['max_excerpt_tokens'], bool)
        or parameters['max_excerpt_tokens'] <= 0
    ):
        raise ValueError('bge-m3_clef.max_excerpt_tokens must be a positive integer')
    questions = _topic_questions(data_args.attributes.get('labels', []))
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
        retriever = None
        if method == 'bge-m3_clef':
            retriever = BgeM3ClefRetriever(prepare_dir, paths.get_script_ctx_path('data', 'embed'), subset,
                                       labeler, data_args.attributes['labels'],
                                       parameters['retrieval_model'], parameters['top_k'])
            # Check every retained query ID before downloading/loading Clef.
        retrieval_started = time.perf_counter()
        examples = None if retriever is None else retriever.retrieve(retained.ids[:max_samples])
        retrieval_seconds = time.perf_counter() - retrieval_started
        datasets.append((subset, test_file, before_filter, excluded, retained, retriever, examples, retrieval_seconds))

    logger.info('Loading Clef model %s at revision %s', model_args.model_name_or_path, parameters['revision'])
    loaded = time.perf_counter()
    predictor = ClefPredictor(model_args, parameters)
    load_seconds = time.perf_counter() - loaded
    evaluations = []
    suffix = f'.limit{max_samples}' if max_samples is not None else ''
    for subset, test_file, before_filter, excluded, retained, retriever, examples, retrieval_seconds in datasets:
        ids, texts, labels = retained.ids[:max_samples], retained.embeddings[:max_samples].tolist(), retained.labels[:max_samples]
        logger.info('%s %s: scoring %s articles after excluding %s test IDs', method, subset, len(ids), len(excluded))
        input_tokens = predictor.input_tokens
        started = time.perf_counter()
        retrieval_options = {} if retriever is None else {
            'examples': examples, 'max_excerpt_tokens': parameters['max_excerpt_tokens'],
        }
        scores = np.asarray(predictor.predict(ids, texts, questions, batch_size, **retrieval_options), dtype=np.float32)
        duration = time.perf_counter() - started
        if scores.shape != labels.shape or not np.isfinite(scores).all():
            raise ValueError('Clef returned invalid classification scores')
        result = {
            'dataset': data_args.dataset_name, 'subset': subset + suffix,
            'model': model_args.short_name, 'model_name_or_path': model_args.model_name_or_path,
            'method': method, 'supervision': 'zero_shot_decision' if retriever is None else 'retrieval_augmented_decision',
            'num_train': 0 if retriever is None else retriever.num_train,
            'num_test': len(ids), 'num_test_before_filter': before_filter,
            'num_test_excluded': len(excluded), 'excluded_test_ids': excluded,
            'num_test_before_limit': len(retained.ids), 'num_labels': labeler.num_labels,
            'label_order': labeler.classes, 'batch_size': batch_size, 'threshold': threshold,
            'parameters': parameters, 'question_template': QUESTION_TEMPLATE,
            'topic_questions': questions, 'feature_type': 'text', 'test_data': str(test_file),
            'method_seconds': duration, 'model_load_seconds': load_seconds,
            'input_tokens': predictor.input_tokens - input_tokens,
            'device': str(predictor.device), 'dtype': str(predictor.dtype),
        }
        if retriever is not None:
            result.update(train_data=str(retriever.train_file), train_sidecar=str(retriever.train_sidecar),
                          test_sidecar=str(retriever.test_sidecar), retrieval_metric='cosine',
                          retrieval_instructions=RAG_INSTRUCTION, retrieval_seconds=retrieval_seconds,
                          retrieval_examples=predictor.retrieval_audit[-len(ids):])
        evaluations.append(EvaluationResult(result, ids, labels, scores))
    if data_args.source.select.subset:
        final = evaluations[0]
    else:
        final = _pool_evaluations(evaluations, data_args.dataset_name + suffix)
        for field in ('input_tokens', 'num_test_before_limit'):
            final.metadata[field] = sum(evaluation.metadata[field] for evaluation in evaluations)
        if method == 'bge-m3_clef':
            final.metadata['retrieval_seconds'] = sum(evaluation.metadata['retrieval_seconds'] for evaluation in evaluations)
            final.metadata['retrieval_examples'] = [record for evaluation in evaluations
                                                    for record in evaluation.metadata['retrieval_examples']]
    _write_evaluation(final)
