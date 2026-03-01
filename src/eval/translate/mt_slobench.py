from logging import Logger
from pathlib import Path
from typing import Any, Dict, Optional, List

from app.pip import Pip
from ...app.args.data import DataArguments

logger: Logger
paths: Dict[str, Any]


def _parse_sample(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None

    return {'text': line}


def eval_ref(source_file, trans_file, ref_file):
    logger.info('Installing evaluation packages')
    Pip.install_packages('nltk', '3.9.3')
    Pip.install_packages('sacrebleu', '2.6.0')
    Pip.install_packages('bert-score', '0.3.13')
    logger.info('Installed evaluation packages')
    pass


def eval_ref_free(source_file, trans_file):
    pass


def main(data_args: DataArguments) -> None:
    logger.info('Evaluating MT Slobench translations')
    ds_name = 'mt-slobench'
    translate_ds_dir = paths['base']['data'] / 'translate' / ds_name
    download_ds_dir = paths['base']['data'] / 'download' / ds_name

    t_cfg = data_args.translate
    translation_dir = translate_ds_dir / f'slobench_ensl.{t_cfg.src_code}'
    if not translation_dir.exists() or not translation_dir.is_dir():
        logger.error(
            f'Translation [translate] {ds_name} directory not found: %s', translation_dir
        )
        return

    ref_file = download_ds_dir / f'slobench_ensl.{t_cfg.src_code}' / f'slobench_ensl.{t_cfg.src_code}.txt'
    if not ref_file.exists() or not ref_file.is_file():
        logger.error(
            f'Reference [translate] {ds_name} directory not found: %s', ref_file
        )
        return

    src_file = download_ds_dir / f'slobench_ensl.{t_cfg.tgt_code}' / f'slobench_ensl.{t_cfg.tgt_code}.txt'
    if not src_file.exists() or not src_file.is_file():
        logger.error(
            f'Source [translate] {ds_name} directory not found: %s', src_file
        )
        return

    test_prefix = None  # eval specific model only
    if hasattr(t_cfg, "model"):
        test_prefix = t_cfg.tgt_code + '.' + t_cfg.model.short_name + '.'

    target_dir = paths['translate']['data'] / ds_name
    target_dir.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []
    for child in translation_dir.iterdir():
        if not child.is_file() and not child.suffix == '.txt':
            continue
        if test_prefix is None:
            files.append(child)
        elif child.name.startswith(test_prefix):
            files.append(child)

    for trans_file in files:
        logger.info('Evaluating translations from %s...', trans_file)
        eval_ref(src_file, trans_file, ref_file)
        eval_ref_free(src_file, trans_file)
        logger.info('Evaluated translations from %s...', trans_file)
