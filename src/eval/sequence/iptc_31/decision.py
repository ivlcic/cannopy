"""Local decision models over prepared IPTC text and independent topic questions."""

import time
from logging import Logger

import numpy as np

from . import (
    EmbeddedSplit, EvaluationResult, _filter_test_split, _method_settings,
    _pool_evaluations, _select_subsets, _write_evaluation,
)
from .features import _read_text_split
from ....app.args.data import DataArguments
from ....app.args.model import ModelArguments
from ....app.args.runtime import Paths
from ....app.decider import ClefDecider, TextDecider
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


def evaluate_decision(data_args: DataArguments, model_args: ModelArguments, paths: Paths, logger: Logger,
                      method: str = 'clef') -> None:
    backend = TextDecider.for_method(method)
    name = backend.name
    options = data_args.attributes.get('sequence', {})
    if 'trials' in options.get('optuna', {}):
        raise ValueError(f'Optuna supports rae_xmc, log_reg and svm; {name} is a zero-shot decision method')
    if not model_args.short_name or not model_args.model_name_or_path:
        raise ValueError('Select a decision model configuration, for example -c clef-flash, -c glide-2b or -c starlux1.1-4b')
    if options.get('features', {}).get('type') != 'text':
        raise ValueError(f'{name} requires text features; select its model configuration')
    backend = TextDecider.get_class(model_args)
    if backend.method != method:
        raise ValueError(f'{backend.name} model requires sequence.method={backend.method}')
    parameters, threshold = _method_settings(options, method)
    parameters = {**backend.default_parameters, **parameters}
    parameters.setdefault('use_label_descriptions', False)
    if parameters.get('head_checkpoint') == 'auto':
        parameters['head_checkpoint'] = str(ClefDecider.default_head_directory(
            paths.get_script_ctx_path('train', 'sequence'), get_subset_name(data_args), model_args.short_name,
            options.get('head_training', {}).get('max_samples_per_language'),
        ))
    model_parameters = {key: value for key, value in parameters.items()
                        if key not in ('max_samples', 'use_label_descriptions')}
    backend.validate_parameters(model_parameters)
    batch_size = options.get('batch_size', 1)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0 or not 0 <= threshold <= 1:
        raise ValueError(f'{name} requires a positive integer batch_size and threshold in [0, 1]')
    max_samples = parameters.get('max_samples')
    if max_samples is not None and (not isinstance(max_samples, int) or isinstance(max_samples, bool) or max_samples <= 0):
        raise ValueError(f'{name} max_samples must be a positive integer or null')
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
    with TextDecider.create(model_args, model_parameters) as predictor:
        load_seconds = time.perf_counter() - loaded
        evaluations = []
        suffix = ('.descriptions' if use_descriptions else '') + (f'.limit{max_samples}' if max_samples is not None else '')
        for subset, test_file, before_filter, excluded, retained in datasets:
            ids, texts, labels = retained.ids[:max_samples], retained.embeddings[:max_samples].tolist(), retained.labels[:max_samples]
            logger.info('%s %s: scoring %s articles after excluding %s test IDs', method, subset, len(ids), len(excluded))
            before = predictor.counters()
            started = time.perf_counter()
            scores = np.asarray(predictor.predict(ids, texts, questions, batch_size), dtype=np.float32)
            duration = time.perf_counter() - started
            if scores.shape != labels.shape or not np.isfinite(scores).all():
                raise ValueError(f'{name} returned invalid classification scores')
            result = {
                'dataset': data_args.dataset_name, 'subset': subset + suffix,
                'model': model_args.short_name, 'model_name_or_path': model_args.model_name_or_path,
                'method': method,
                'num_test': len(ids), 'num_test_before_filter': before_filter,
                'num_test_excluded': len(excluded), 'excluded_test_ids': excluded,
                'num_test_before_limit': len(retained.ids), 'num_labels': labeler.num_labels,
                'label_order': labeler.classes, 'batch_size': batch_size, 'threshold': threshold,
                'parameters': parameters, 'question_template': QUESTION_TEMPLATE + (DEFINITION_TEMPLATE if use_descriptions else ''),
                'prompt_variant': 'name_and_definition' if use_descriptions else 'name_only',
                'topic_questions': questions, 'feature_type': 'text', 'test_data': str(test_file),
                'method_seconds': duration, 'model_load_seconds': load_seconds,
                **predictor.metadata(),
            }
            if use_descriptions:
                result['label_description_metadata'] = data_args.attributes.get('label_description_metadata', {})
            for key, value in predictor.counters().items():
                result['num_test_truncated' if key == 'truncated_articles' else key] = value - before[key]
            evaluations.append(EvaluationResult(result, ids, labels, scores))
        if data_args.source.select.subset:
            final = evaluations[0]
        else:
            final = _pool_evaluations(evaluations, data_args.dataset_name + suffix)
            for field in ('input_tokens', 'num_test_before_limit', 'num_test_truncated'):
                if field in final.metadata:
                    final.metadata[field] = sum(evaluation.metadata[field] for evaluation in evaluations)
        _write_evaluation(final)


def evaluate_clef(data_args: DataArguments, model_args: ModelArguments, paths: Paths, logger: Logger) -> None:
    evaluate_decision(data_args, model_args, paths, logger, method='clef')
