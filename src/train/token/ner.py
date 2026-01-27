from logging import Logger
from pathlib import Path
from typing import Any, Dict, Tuple

from torch.utils.data import Dataset
from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                          Trainer, TrainingArguments)

from ...app.dataset import NerSamplesLoader
from ...app.metrics import TokenClassificationMetrics
from ...app.args.model import ModelArguments
from ...app.args.data import DataArguments

logger: Logger
paths: Dict[str, Any]


def init_dirs(paths_dict: Dict[str, Any]) -> Tuple[Path, Path]:
    data_root = paths_dict['base']['data'] / 'split'
    if not data_root.exists():
        raise EnvironmentError(f'Split data not found at %s. Run `./data split ner` first.', data_root)

    cache_root = paths_dict['base']['tmp'] / 'cache'
    cache_root.mkdir(parents=True, exist_ok=True)
    if not cache_root.exists():
        raise EnvironmentError(f'Unable to init cache data dir at {cache_root}. Run `./data split ner` first.')
    return data_root, cache_root


def compute_output_dir(m_args: ModelArguments, d_args: DataArguments, t_args: TrainingArguments) -> Path:
    model_name = f'{d_args.dataset_name}.{m_args.short_name}.b{t_args.train_batch_size}.lr{t_args.learning_rate}'
    output_dir = paths['token']['train'] / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# noinspection DuplicatedCode
def main(data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments) -> None:
    logger.info('Training NER')

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
        model_args.model_name_or_path,
        cache_dir=cache_root,
        num_labels=len(ner_samples.label_list),
        id2label=ner_samples.id2label,
        label2id=ner_samples.label2id,
    )

    datasets: Dict[str, Dataset] = ner_samples.create_split_datasets(tokenizer, model_args.max_seq_length)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['eval'],
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=metrics.compute_metrics
    )

    train_result = trainer.train()
    trainer.save_model(train_args.output_dir)
    state_path = Path(train_args.output_dir) / "trainer_state.json"
    trainer.state.save_to_json(str(state_path))
    logger.info("Saved trainer state to %s", state_path)
    logger.info('Training complete; global steps=%s, training_loss=%.6f',
                train_result.global_step, train_result.training_loss)

    metrics = trainer.evaluate()
    logger.info('Evaluation metrics: %s', metrics)
