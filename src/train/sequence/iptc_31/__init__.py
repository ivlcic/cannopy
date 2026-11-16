"""IPTC-31 decision-head training entry points."""

from logging import Logger

from transformers import TrainingArguments

from ....app.args.data import DataArguments
from ....app.args.model import ModelArguments
from ....app.args.runtime import Paths

logger: Logger
paths: Paths


def clef_flash_head(data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments) -> None:
    from .clef_head import train_head

    train_head(data_args, model_args, train_args, paths, logger)


def main(data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments) -> None:
    clef_flash_head(data_args, model_args, train_args)
