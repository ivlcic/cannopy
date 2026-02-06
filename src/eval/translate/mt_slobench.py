import os
import shutil

import torch
import zipfile

from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Optional

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
)

from ...app.args.model import ModelArguments
from ...app.args.data import DataArguments

logger: Logger
paths: Dict[str, Any]


def main(data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments) -> None:
    logger.info('Evaluating NER')

    train_args.output_dir = compute_train_dir(model_args, train_args)
    output_dir = compute_output(model_args, data_args, train_args)

    sub_dir = data_args.attributes.get('use_subdir', 'sample_reference')
    input_dir = paths['base']['data'] / 'download' / data_args.dataset_name / sub_dir