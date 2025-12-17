import csv
import random
from collections import defaultdict
from logging import Logger
from pathlib import Path
from typing import Any, DefaultDict, Dict, List

from ...app.args.data import DataArguments

logger: Logger
paths: Dict[str, Any]


def main(data_args: DataArguments) -> None:
    logger.info('Translating MIRACL dataset')

    data_args.translate_config

    source_dir = paths['base']['data'] / 'download' / 'miracl'
    target_dir = paths['translate']['data']
    target_dir.mkdir(parents=True, exist_ok=True)

    #aggregated = _load_prepared_sentences(source_dir)
    #if not aggregated:
    #    logger.warning('No downloaded miracl data found in %s', source_dir)
    #    return
