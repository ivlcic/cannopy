from logging import Logger
from typing import Any, Dict

from transformers import TrainingArguments

from ...app.args.model import ModelArguments
from ...app.args.data import DataArguments

logger: Logger
paths: Dict[str, Any]

def main(data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments) -> None:
    logger.info("Training NER")
