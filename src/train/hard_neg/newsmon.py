import os
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, PreTrainedModel

from ...app.args.data import DataArguments
from ...app.args.model import ModelArguments
from ...app.args.runtime import Paths
from ...app.helpers import JsonIRHelper
from ...app.model.bge_m3 import BGEM3Model

logger: Logger
paths: Paths


class HardNegativeDataset(Dataset):
    def __init__(self, source_file: Path, max_examples: int = 0):
        self.samples: list[dict[str, Any]] = []
        with source_file.open('r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                sample = JsonIRHelper.read_ir_sample(line, line_no, source_file)
                if sample is None:
                    continue
                if not sample['pos'] or not sample['neg']:
                    continue
                self.samples.append(sample)
                if max_examples > 0 and len(self.samples) >= max_examples:
                    break
        if not self.samples:
            raise ValueError(f'No hard-negative samples loaded from {source_file}')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        return {
            'query': sample['query'],
            'passages': [sample['pos'][0], *sample['neg']],
        }


class HardNegativeCollator:
    def __init__(self, tokenizer, query_max_length: int, passage_max_length: int):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.passage_max_length = passage_max_length

    def __call__(self, features: list[dict[str, Any]]) -> Dict[str, Dict[str, torch.Tensor]]:
        queries = [feature['query'] for feature in features]
        passages = [passage for feature in features for passage in feature['passages']]
        query_batch: Dict[str, torch.Tensor] = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.query_max_length,
            return_tensors='pt',
        )
        passage_batch: Dict[str, torch.Tensor] = self.tokenizer(
            passages,
            padding=True,
            truncation=True,
            max_length=self.passage_max_length,
            return_tensors='pt',
        )
        return {
            'query': dict(query_batch),
            'passage': dict(passage_batch),
        }


class BGEM3Trainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None) -> None:
        del state_dict
        # noinspection PyTypeChecker, PyUnresolvedReferences
        output_dir: Union[str, Path] = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        # noinspection PyTypeChecker
        if hasattr(self, 'accelerator'):
            model: torch.nn.Module = self.accelerator.unwrap_model(self.model)
        else:
            # noinspection PyTypeChecker
            model: Union[PreTrainedModel, torch.nn.Module] = self.model
        model.save(output_dir)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))


def init_dirs(p: Paths) -> Path:
    data_root = p.base.result.data / 'split'
    if not data_root.exists():
        raise EnvironmentError(f'Split data root not found at {data_root}.')
    return data_root


def resolve_train_file(data_root: Path,
                       data_args: DataArguments,
                       model_args: ModelArguments,
                       extra_args: Dict[str, Any]) -> Path:
    hard_neg_cfg = extra_args.get('hard_neg', {})
    source_file = hard_neg_cfg.get('train_file', '')
    if source_file:
        file_path = Path(source_file)
        if not file_path.is_absolute():
            file_path = paths.base.root / file_path
        return file_path

    split_name = hard_neg_cfg.get('split', 'train')
    file_name = f'{data_args.dataset_name}.{model_args.short_name}.{split_name}.hn.jsonl'
    return data_root / paths.curr_context / file_name


def resolve_eval_file(extra_args: Dict[str, Any]) -> Optional[Path]:
    hard_neg_cfg = extra_args.get('hard_neg', {})
    source_file = hard_neg_cfg.get('eval_file', '')
    if not source_file:
        return None
    file_path = Path(source_file)
    if not file_path.is_absolute():
        file_path = paths.base.root / file_path
    return file_path


def compute_output_dir(m_args: ModelArguments, d_args: DataArguments, t_args: TrainingArguments) -> Path:
    model_name = f'{d_args.dataset_name}.{m_args.short_name}.b{t_args.train_batch_size}.lr{t_args.learning_rate}'
    output_dir = paths.context / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_model(model_args: ModelArguments, extra_args: Dict[str, Any]) -> BGEM3Model:
    bge_m3_cfg = extra_args.get('bge_m3', {})
    return BGEM3Model(
        model_name=model_args.model_name_or_path,
        normlized=bool(bge_m3_cfg.get('normlized', True)),
        sentence_pooling_method=bge_m3_cfg.get('sentence_pooling_method', 'cls'),
        negatives_cross_device=bool(bge_m3_cfg.get('negatives_cross_device', False)),
        temperature=float(bge_m3_cfg.get('temperature', 1.0)),
        enable_sub_batch=bool(bge_m3_cfg.get('enable_sub_batch', True)),
        unified_finetuning=bool(bge_m3_cfg.get('unified_finetuning', True)),
        use_self_distill=bool(bge_m3_cfg.get('use_self_distill', False)),
        colbert_dim=int(bge_m3_cfg.get('colbert_dim', -1)),
        ensemble_distill_start_step=int(bge_m3_cfg.get('ensemble_distill_start_step', -1)),
    )


def maybe_enable_gradient_checkpointing(model: BGEM3Model, extra_args: Dict[str, Any]) -> None:
    hard_neg_cfg = extra_args.get('hard_neg', {})
    if bool(hard_neg_cfg.get('gradient_checkpointing', False)):
        model.gradient_checkpointing_enable()


def main(data_args: DataArguments,
         model_args: ModelArguments,
         train_args: TrainingArguments,
         extra_args: Dict[str, Any]) -> None:
    logger.info('Training BGE-M3 hard-negative retriever')

    data_root = init_dirs(paths)
    train_file = resolve_train_file(data_root, data_args, model_args, extra_args)
    if not train_file.exists():
        raise FileNotFoundError(f'Hard-negative train file not found: {train_file}')
    eval_file = resolve_eval_file(extra_args)
    if eval_file is not None and not eval_file.exists():
        raise FileNotFoundError(f'Hard-negative eval file not found: {eval_file}')

    hard_neg_cfg = extra_args.get('hard_neg', {})
    train_args.output_dir = str(compute_output_dir(model_args, data_args, train_args))
    train_args.remove_unused_columns = False

    model = load_model(model_args, extra_args)
    maybe_enable_gradient_checkpointing(model, extra_args)
    tokenizer = model.tokenizer

    query_max_length = int(hard_neg_cfg.get('query_max_length', min(512, model_args.max_seq_length)))
    passage_max_length = int(hard_neg_cfg.get('passage_max_length', model_args.max_seq_length))
    max_examples = int(hard_neg_cfg.get('max_examples', 0))
    train_dataset = HardNegativeDataset(train_file, max_examples=max_examples)
    eval_dataset = HardNegativeDataset(eval_file, max_examples=max_examples) if eval_file is not None else None
    collator = HardNegativeCollator(tokenizer, query_max_length, passage_max_length)

    trainer = BGEM3Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    logger.info(
        'Loaded %d train samples%s from %s',
        len(train_dataset),
        f' and {len(eval_dataset)} eval samples' if eval_dataset is not None else '',
        train_file,
    )
    train_result = trainer.train()
    trainer.save_model(train_args.output_dir)
    state_path = Path(train_args.output_dir) / 'trainer_state.json'
    trainer.state.save_to_json(str(state_path))
    logger.info('Saved trainer state to %s', state_path)
    logger.info(
        'Training complete; global steps=%s, training_loss=%.6f',
        train_result.global_step,
        train_result.training_loss,
    )

    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        logger.info('Evaluation metrics: %s', eval_metrics)
