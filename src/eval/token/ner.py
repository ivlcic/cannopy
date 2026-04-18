import json
from logging import Logger
from pathlib import Path
from typing import Dict

from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from ...app.args.data import DataArguments
from ...app.args.model import ModelArguments
from ...app.args.runtime import Paths
from ...app.dataset import NerSamplesLoader
from ...app.metrics import TokenClassificationMetrics
from ...train.token.ner import init_dirs

logger: Logger
paths: Paths


def compute_train_dir(m_args: ModelArguments, d_args: DataArguments, t_args: TrainingArguments) -> Path:
    model_name = f'{d_args.dataset_name}.{m_args.short_name}.b{t_args.train_batch_size}.lr{t_args.learning_rate}'
    output_dir = paths.get_script_path('train') / model_name
    print(output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(output_dir)
    return output_dir


def compute_output(m_args: ModelArguments, d_args: DataArguments, t_args: TrainingArguments) -> Path:
    model_name = f'{d_args.dataset_name}.{m_args.short_name}.b{t_args.train_batch_size}.lr{t_args.learning_rate}.json'
    output = paths.context / model_name
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


# noinspection DuplicatedCode
def main(data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments) -> None:
    logger.info('Evaluating NER')

    train_args.output_dir = str(compute_train_dir(model_args, data_args, train_args))
    output = compute_output(model_args, data_args, train_args)

    data_root, cache_root = init_dirs(paths)
    languages = data_args.subdata_order or []
    if not languages:
        languages = [p.stem.split('.')[0] for p in data_root.glob('ner-*.train.csv')]

    ner_samples = NerSamplesLoader(data_root, languages)
    # ner_samples = NerSamplesLoader(data_root, ['sl'])
    metrics = TokenClassificationMetrics(id2label=ner_samples.labeler.id2label)

    tokenizer_name = model_args.tokenizer_name or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_root)
    collator = DataCollatorForTokenClassification(tokenizer, padding='longest')
    # noinspection PyTypeChecker
    model = AutoModelForTokenClassification.from_pretrained(
        train_args.output_dir,
        cache_dir=cache_root,
        num_labels=ner_samples.labeler.num_labels,
        id2label=ner_samples.labeler.id2label,
        label2id=ner_samples.labeler.label2id,
        dtype=model_args.dtype,
        attn_implementation=model_args.attn_implementation
    )

    datasets: Dict[str, Dataset] = ner_samples.create_split_datasets(tokenizer, model_args.max_seq_length)
    trainer = Trainer(
        model=model,
        args=train_args,
        eval_dataset=datasets['test'],
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=metrics,
    )
    metrics = trainer.evaluate()
    log = trainer.state.log_history[0]
    with open(output, 'w', encoding='utf-8') as fp:
        json.dump(log, fp, ensure_ascii=False, indent=2, sort_keys=False)
    logger.info('Test metrics: %s', metrics)
