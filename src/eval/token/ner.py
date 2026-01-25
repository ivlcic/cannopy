from logging import Logger
from pathlib import Path
from typing import Any, Dict

from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from ...app.dataset import NerSamplesLoader
from ...app.metrics import TokenClassificationMetrics
from ...app.args.model import ModelArguments
from ...app.args.data import DataArguments
from ...train.token.ner import init_dirs

logger: Logger
paths: Dict[str, Any]


def compute_output_dir(m_args: ModelArguments, d_args: DataArguments, t_args: TrainingArguments) -> Path:
    model_name = f'{d_args.dataset_name}.{m_args.short_name}.b{t_args.train_batch_size}.lr{t_args.learning_rate}'
    output_dir = paths['token']['train'] / model_name
    if not output_dir.exists():
        raise FileNotFoundError(output_dir)
    return output_dir


def main(data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments) -> None:
    logger.info('Evaluating NER')

    data_root, cache_root = init_dirs(paths)
    train_args.output_dir = str(compute_output_dir(model_args, data_args, train_args))

    languages = data_args.subdata_order or []
    if not languages:
        languages = [p.stem.split('.')[0] for p in data_root.glob('ner-*.train.csv')]


    ner_samples = NerSamplesLoader(data_root, languages)
    metrics = TokenClassificationMetrics(id2label=ner_samples.id2label)

    tokenizer_name = model_args.tokenizer_name or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_root)
    collator = DataCollatorForTokenClassification(tokenizer, padding='longest')
    model = AutoModelForTokenClassification.from_pretrained(
        train_args.output_dir,
        cache_dir=cache_root,
        num_labels=len(ner_samples.label_list),
        id2label=ner_samples.id2label,
        label2id=ner_samples.label2id,
    )

    datasets: Dict[str, Dataset] = ner_samples.create_split_datasets(tokenizer, model_args.max_seq_length)
    trainer = Trainer(
        model=model,
        args=train_args,
        eval_dataset=datasets['test'],
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=metrics.compute_metrics,
    )
    metrics = trainer.evaluate()
    logger.info('Test metrics: %s', metrics)
