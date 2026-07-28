from logging import Logger
from pathlib import Path
from typing import Dict, Tuple

from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments
)

from ...app.args.data import DataArguments
from ...app.args.model import ModelArguments
from ...app.args.runtime import Paths
from ...app.dataset import NewsmonSamplesLoader
from ...app.metrics import MultilabelSequenceMetrics
from ...data.prepare.newsmon import get_subset_data_path, get_subset_name

logger: Logger
paths: Paths


def init_dirs(p: Paths, data_args: DataArguments) -> Tuple[Path, Path]:
    data_root = p.get_script_ctx_path('data', 'split')
    train_file = get_subset_data_path(data_args, data_root, 'train')
    if not train_file.exists():
        raise EnvironmentError(f'Split training data not found at {train_file}. Run `./data split newsmon` first.')

    cache_root = p.base.tmp / 'cache'
    cache_root.mkdir(parents=True, exist_ok=True)
    if not cache_root.exists():
        raise EnvironmentError(f'Unable to init cache data dir at {cache_root}.')
    return data_root, cache_root


def compute_output_dir(m_args: ModelArguments, d_args: DataArguments, t_args: TrainingArguments) -> Path:
    model_name = f'{d_args.dataset_name}.{m_args.short_name}.b{t_args.train_batch_size}.lr{t_args.learning_rate}'
    output_dir = paths.context / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main(data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments) -> None:
    logger.info('Training NewsMon multilabel sequence classifier')

    data_root, cache_root = init_dirs(paths, data_args)
    train_args.output_dir = str(compute_output_dir(model_args, data_args, train_args))
    subset = get_subset_name(data_args)
    newsmon_samples = NewsmonSamplesLoader(data_root, subset)
    metrics = MultilabelSequenceMetrics(model_name=model_args.short_name or model_args.model_name_or_path)

    tokenizer_name = model_args.tokenizer_name or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_root)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=cache_root,
        num_labels=newsmon_samples.labeler.num_labels,
        id2label=newsmon_samples.labeler.id2label,
        label2id=newsmon_samples.labeler.label2id,
        problem_type='multi_label_classification',
        dtype=model_args.dtype,
    )
    model_args.validate_training_parameter_dtypes(model)

    datasets: Dict[str, Dataset] = newsmon_samples.create_split_datasets(tokenizer, model_args.max_seq_length)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['eval'],
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=metrics,
        preprocess_logits_for_metrics=metrics.preprocess_logits,
    )

    train_result = trainer.train()
    trainer.save_model(train_args.output_dir)
    state_path = Path(str(train_args.output_dir)) / 'trainer_state.json'
    trainer.state.save_to_json(str(state_path))
    logger.info('Saved trainer state to %s', state_path)
    logger.info(
        'Training complete; global steps=%s, training_loss=%.6f',
        train_result.global_step,
        train_result.training_loss
    )

    eval_metrics = trainer.evaluate()
    logger.info('Evaluation metrics: %s', eval_metrics)
