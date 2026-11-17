"""Supervised Clef joint-head training with a frozen, deterministic backbone."""

import hashlib
from dataclasses import dataclass
from logging import Logger
from pathlib import Path

import numpy as np
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, set_seed, EarlyStoppingCallback

from ....app.args.data import DataArguments
from ....app.args.model import ModelArguments
from ....app.args.runtime import Paths
from ....app.decider import ClefDecider
from ....app.labeler import MultilabelLabeler
from ....app.metrics import MultilabelSequenceMetrics
from ....eval.sequence.iptc_31 import _method_settings
from ....eval.sequence.iptc_31.decision import _topic_questions
from ....eval.sequence.iptc_31.features import _read_text_split


@dataclass
class Sample:
    sample_id: str
    text: str
    labels: np.ndarray
    subset: str


def load_training_samples(data_args, questions, prepare_dir: Path, limit: int | None, seed: int):
    """Read original train files only; tests and test-ID exclusions play no role."""
    labeler = MultilabelLabeler(labels=list(questions))
    selected = data_args.source.select.subset
    files = ([prepare_dir / f'{selected}.train.jsonl'] if selected else
             sorted(prepare_dir.glob(f'{data_args.dataset_name}.*.train.jsonl')))
    if not files:
        raise FileNotFoundError(f'No prepared IPTC training text in {prepare_dir}; run ./data prepare iptc-31')
    samples, sources, seen_ids = [], [], set()
    rng = np.random.default_rng(seed)
    for file in files:
        ids, texts, labels = _read_text_split(file, labeler)
        subset = file.name.removesuffix('.train.jsonl')
        rows = np.arange(len(ids))
        if limit is not None and len(rows) > limit:
            rows = np.sort(rng.choice(rows, size=limit, replace=False))
        for row in rows:
            if ids[row] in seen_ids:
                raise ValueError(f'Duplicate training sample ID: {ids[row]}')
            if not texts[row].strip():
                raise ValueError(f'Empty article text for {ids[row]}')
            seen_ids.add(ids[row])
            samples.append(Sample(str(ids[row]), texts[row], labels[row], subset))
        sources.append({'path': str(file), 'sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
                        'num_original': len(ids), 'num_selected': len(rows)})
    return samples, sources


def validation_split(samples: list[Sample], size: float, seed: int):
    # Exact texts (after whitespace normalization) form indivisible groups,
    # including repeats across languages. Language indicators also enter the
    # multilabel stratification so the pooled holdout represents each language.
    groups = {}
    languages = sorted({sample.subset for sample in samples})
    for row, sample in enumerate(samples):
        key = hashlib.sha256(' '.join(sample.text.split()).encode()).hexdigest()
        groups.setdefault(key, []).append(row)
    if len(groups) < 2:
        raise ValueError('Head training requires at least two distinct article texts')
    rows_by_group = list(groups.values())
    strata = np.zeros((len(groups), len(samples[0].labels) + len(languages)))
    for group, rows in enumerate(rows_by_group):
        strata[group, :len(samples[0].labels)] = np.max([samples[row].labels for row in rows], axis=0)
        for row in rows:
            strata[group, len(samples[0].labels) + languages.index(samples[row].subset)] = 1
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=size, random_state=seed)
    train_groups, val_groups = next(splitter.split(np.zeros((len(groups), 1)), strata))
    train = [samples[row] for group in train_groups for row in rows_by_group[group]]
    validation = [samples[row] for group in val_groups for row in rows_by_group[group]]
    if not train or not validation:
        raise ValueError('Head training requires non-empty training and validation partitions')
    return train, validation


class HeadDataset(Dataset):
    def __init__(self, samples, decider: ClefDecider, questions):
        self.items = []
        for sample in samples:
            record = decider.encode(sample.text, questions, sample.sample_id)
            self.items.append({'record': record, 'labels': sample.labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class HeadCollator:
    def __init__(self, decider: ClefDecider):
        self.decider = decider

    def __call__(self, items):
        batch = self.decider.collate([item['record'] for item in items])
        batch['labels'] = torch.from_numpy(np.stack([item['labels'] for item in items]))
        return batch


class FrozenBackboneHead(torch.nn.Module):
    def __init__(self, clef_model, label_order):
        super().__init__()
        self.backbone = clef_model.language_model
        self.backbone.requires_grad_(False)
        # Keep optimizer parameters in FP32; Trainer supplies BF16 autocast on CUDA.
        self.head = clef_model.head.float()
        self.head.requires_grad_(True)
        self.label_order = list(label_order)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, input_ids, attention_mask, records, media=None, labels=None):
        if media:
            raise ValueError('Clef head training accepts text only')
        text_model = self.backbone.model
        if hasattr(text_model, 'language_model'):
            text_model = text_model.language_model
        # no_grad creates ordinary tensors that the trainable head can save for
        # backward; inference_mode tensors would not be safe at this boundary.
        with torch.no_grad():
            hidden = text_model(input_ids=input_ids, attention_mask=attention_mask,
                                use_cache=False, return_dict=True).last_hidden_state
        options = self.head(hidden, input_ids, attention_mask, records, self.backbone.get_output_embeddings().weight)
        rows = []
        for record, option_logits in zip(records, options, strict=True):
            if len(option_logits) != len(record.questions):
                raise ValueError('Clef head returned incomplete option logits')
            by_code = {}
            for question, values in zip(record.questions, option_logits, strict=True):
                if set(question.option_ids) != {'true', 'false'} or len(values) != 2:
                    raise ValueError('IPTC head training requires independent binary topic decisions')
                values = values.float()
                by_code[question.question_id] = values[question.option_ids.index('true')] - values[question.option_ids.index('false')]
            if set(by_code) != set(self.label_order):
                raise ValueError('Clef head returned an incomplete or misaligned topic schema')
            rows.append(torch.stack([by_code[code] for code in self.label_order]))
        logits = torch.stack(rows).float()
        result = {'logits': logits}
        if labels is not None:
            result['loss'] = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
        return result


class HeadTrainer(Trainer):
    """Keep the standard optimizer/resume loop, saving only trainable head weights."""

    def __init__(self, *args, decider: ClefDecider, provenance, **kwargs):
        self.decider, self.provenance = decider, provenance
        super().__init__(*args, **kwargs)

    def _save(self, output_dir=None, state_dict=None):
        self.decider.save_head(output_dir or self.args.output_dir, self.provenance)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        saved = self.decider.read_head_manifest(resume_from_checkpoint, self.provenance['base_model'], self.provenance['revision'])
        for field in ('topic_questions', 'train_ids', 'validation_ids', 'training_sources', 'max_seq_length', 'max_state_tokens'):
            if saved[field] != self.provenance[field]:
                raise ValueError(f'Cannot resume Clef head training with different {field}')
        for name in ('optimizer.pt', 'scheduler.pt', 'trainer_state.json'):
            if not (Path(resume_from_checkpoint) / name).is_file():
                raise ValueError('Resume requires a Trainer checkpoint-N directory with optimizer/scheduler state')
        self.decider.restore_head(resume_from_checkpoint)

    def _load_best_model(self):
        self.decider.restore_head(self.state.best_model_checkpoint)


def train_head(data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments,
               paths: Paths, logger: Logger) -> None:
    options = data_args.attributes.get('sequence', {})
    if options.get('method') != 'clef' or not model_args.model_name_or_path:
        raise ValueError('Select the Clef head training configuration: -c clef-flash -c clef-flash-head')
    if 'trials' in options.get('optuna', {}):
        raise ValueError('Optuna is not supported for Clef head training')
    if (train_args.world_size != 1 or train_args.n_gpu > 1 or train_args.deepspeed
            or train_args.fsdp or train_args.gradient_checkpointing):
        raise ValueError('Clef head training uses one device; select one visible GPU and disable distributed/gradient checkpointing')
    parameters, threshold = _method_settings(options, 'clef')
    # Initialize from the released head; head_checkpoint selects a trained head
    # for evaluation, while train.resume_from_checkpoint resumes training state.
    parameters.pop('head_checkpoint', None)
    if parameters.get('max_samples') is not None:
        raise ValueError('Use head_training.max_samples_per_language to limit training data; max_samples limits evaluation only')
    parameters['device'] = str(train_args.device)
    if (train_args.device.type != 'cpu' and (model_args.dtype or 'bfloat16') != 'float32'
            and not (train_args.bf16 or train_args.fp16)):
        raise ValueError('BF16/FP16 backbones require train.bf16=true or train.fp16=true for head training')
    settings = options.get('head_training', {})
    unknown = set(settings) - {'validation_size', 'max_samples_per_language'}
    if unknown:
        raise ValueError(f'Unknown head_training settings: {", ".join(sorted(unknown))}')
    validation_size = settings.get('validation_size', .2)
    if isinstance(validation_size, bool) or not isinstance(validation_size, (int, float)) or not 0 < validation_size < 1:
        raise ValueError('head_training.validation_size must be between 0 and 1')
    limit = settings.get('max_samples_per_language')
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 2):
        raise ValueError('head_training.max_samples_per_language must be an integer >= 2 or null')
    if not isinstance(parameters.get('revision'), str) or not parameters['revision']:
        raise ValueError('Clef head training requires a pinned revision')
    state_limit = parameters.get('max_state_tokens')
    if state_limit is not None and (not isinstance(state_limit, int) or isinstance(state_limit, bool) or state_limit < 1):
        raise ValueError('max_state_tokens must be a positive integer or null')
    if not 0 <= threshold <= 1:
        raise ValueError('Clef head threshold must be in [0, 1]')
    use_descriptions = parameters.get('use_label_descriptions', False)
    questions = _topic_questions(data_args.attributes.get('labels', []), use_descriptions)
    output = Path(train_args.output_dir) if train_args.output_dir != 'trainer_output' else ClefDecider.default_head_directory(
        paths.get_script_ctx_path('train', 'sequence'), data_args.source.select.subset or data_args.dataset_name,
        model_args.short_name, limit, seed=train_args.seed,
    )
    if output.exists() and any(output.iterdir()) and not train_args.resume_from_checkpoint:
        raise FileExistsError(f'Training output is not empty: {output}; choose train.output_dir or train.resume_from_checkpoint')
    train_args.output_dir = str(output)
    train_args.remove_unused_columns = False
    train_args.label_names = ['labels']
    set_seed(train_args.seed)
    samples, sources = load_training_samples(data_args, questions, paths.get_script_ctx_path('data', 'prepare'), limit, train_args.seed)
    training, validation = validation_split(samples, validation_size, train_args.seed)
    logger.info('Training one shared Clef head on %d articles; validating on %d; original test files are untouched',
                len(training), len(validation))
    model_parameters = {key: value for key, value in parameters.items() if key not in ('max_samples', 'use_label_descriptions')}
    ClefDecider.validate_parameters(model_parameters)
    decider = ClefDecider(model_args, model_parameters)
    model = FrozenBackboneHead(decider.model, list(questions))
    model_args.validate_training_parameter_dtypes(model)
    provenance = {
        'format': 'cannopy-clef-head-v1', 'base_model': model_args.model_name_or_path, 'revision': parameters['revision'],
        'dataset': data_args.dataset_name, 'label_order': list(questions), 'topic_questions': questions,
        'use_label_descriptions': use_descriptions, 'max_seq_length': model_args.max_seq_length,
        'max_state_tokens': state_limit, 'threshold': threshold, 'seed': train_args.seed,
        'validation_size': validation_size, 'training_sources': sources,
        'train_ids': [sample.sample_id for sample in training], 'validation_ids': [sample.sample_id for sample in validation],
        'trainable_parameters': sum(parameter.numel() for parameter in model.head.parameters()),
        'training_arguments': train_args.to_dict(),
    }
    if use_descriptions:
        provenance['label_description_metadata'] = data_args.attributes.get('label_description_metadata', {})
    datasets = [HeadDataset(part, decider, questions) for part in (training, validation)]
    metrics = MultilabelSequenceMetrics(model_name=model_args.short_name)
    trainer = HeadTrainer(
        model=model, args=train_args, train_dataset=datasets[0], eval_dataset=datasets[1],
        data_collator=HeadCollator(decider),
        compute_metrics=lambda prediction: metrics.compute_metrics(prediction.label_ids, prediction.predictions, prob_threshold=threshold),
        preprocess_logits_for_metrics=metrics.preprocess_logits,
        decider=decider, provenance=provenance, callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=5,
                early_stopping_threshold=0.0,
            ),
        ],
    )
    trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
    provenance['validation_metrics'] = trainer.evaluate()
    provenance['best_validation_metric'] = trainer.state.best_metric
    provenance['best_checkpoint'] = trainer.state.best_model_checkpoint
    trainer.save_model(str(output))
    trainer.state.save_to_json(str(output / 'trainer_state.json'))
    logger.info('Saved trained Clef head to %s; validation metrics: %s', output, provenance['validation_metrics'])
