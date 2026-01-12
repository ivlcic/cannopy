import json
import shutil

from logging import Logger
from typing import Any, Dict, List
from datetime import datetime
from dateutil.relativedelta import relativedelta

from ...app.args.data import DataArguments
from ...app.args.model import ModelArguments
from ...app.embedder import TextEmbedder

logger: Logger
paths: Dict[str, Any]


def _load_embeddings(file_name) -> Dict[str, List[float]]:
    embeddings: Dict[str, List[float]] = {}
    with file_name.open('r', encoding='utf-8') as f_in:
        for line_no, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if 'id' not in obj:
                    logger.warning('Missing id in %s line %d.', file_name, line_no)
                    continue
                if 'embeddings' not in obj:
                    logger.warning('Missing embeddings in %s line %d.', file_name, line_no)
                    continue
                embeddings[obj['id']] = obj['embeddings']
            except json.JSONDecodeError:
                logger.warning('Skipping malformed JSON in %s line %d.', file_name.name, line_no)
                raise
    return embeddings


def _filter_out_sample(data_args: DataArguments, sample: Dict[str, Any]) -> bool:
    if data_args.source.select.filter:
        for k, v in data_args.source.select.filter.items():
            if k in sample and sample[k] != v:
                return True
    return False


def _get_text(sample: Dict[str, Any]) -> str:
    title = sample['title']['text']
    body = sample['body']['text']
    return title + '\n' + body


def main(data_args: DataArguments, model_args: ModelArguments) -> None:
    source_dir = paths['base']['data'] / 'download' / data_args.dataset_name
    if not source_dir.exists():
        logger.error(f'Source [download] {data_args.dataset_name} directory not found: %s', source_dir)
        return

    target_dir = paths['embed']['data'] / data_args.dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)

    start = datetime.fromisoformat(data_args.source.select.start)
    end = datetime.fromisoformat(data_args.source.select.end)
    # Validate year == 2023 as Newsmon is 2023 only
    if start.year != 2023 or end.year != 2023:
        raise ValueError(
            f"Dates must be in year 2023; got start={start.date().isoformat()}, end={end.date().isoformat()}"
        )
    if end < start:
        raise ValueError(f"end must be >= start; got start={start}, end={end}")

    embedder = TextEmbedder.create(model_args)

    # Iterate month-by-month with per-month clipped ranges
    cur = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    # is data only a subset
    subset = ''
    if data_args.source.select.subset:
        subset = f'{data_args.source.select.subset}_'

    while cur <= end:
        next_month = cur + relativedelta(months=1)
        src_file = source_dir / f'data_{start.year}_{cur.month:02d}.jsonl'
        if not src_file.exists():
            raise FileNotFoundError(
                f"Data file not found: {src_file}, check data.source.select.start and data.source.select.end"
            )
        tgt_file = target_dir / f'{subset}data_{start.year}_{cur.month:02d}.jsonl'

        src_ebd_file = source_dir / f'data_{start.year}_{cur.month:02d}-{model_args.short_name}.jsonl'
        src_ebd: Dict[str, List[float]] = {}
        if src_ebd_file.exists():
            logger.info(
                f"Source embedding data {src_ebd_file} for model {model_args.short_name} exists. Will use that!"
            )
            src_ebd = _load_embeddings(src_ebd_file)

        tgt_ebd_file = target_dir / f'{subset}data_{start.year}_{cur.month:02d}-{model_args.short_name}.jsonl'
        tmp_tgt_ebd_file = target_dir / f'tmp.{subset}data_{start.year}_{cur.month:02d}-{model_args.short_name}.jsonl'
        tgt_ebd: Dict[str, List[float]] = {}
        if tgt_ebd_file.exists():
            logger.info(
                f"Target embedding data {tgt_ebd_file} for model {model_args.short_name} exists. Will use that!"
            )
            tgt_ebd = _load_embeddings(tgt_ebd_file)

        with (src_file.open('r', encoding='utf-8') as f_in,
              tgt_file.open('w', encoding='utf-8') as f_out,
              tmp_tgt_ebd_file.open('w', encoding='utf-8') as f_ebd_out):
            for line_no, line in enumerate(f_in, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if 'id' not in obj:
                        logger.warning('Missing id in %s line %d.', src_file, line_no)
                        continue
                    if _filter_out_sample(data_args, obj):
                        continue

                    ebd = src_ebd.get(obj['id'], None)
                    if not ebd:
                        ebd = tgt_ebd.get(obj['id'], None)
                        if not ebd:
                            text = _get_text(obj)
                            ebd = {
                                'id': obj['id'],
                                'embedding': embedder.embed(text)
                            }
                    f_ebd_out.write(json.dumps(ebd, ensure_ascii=False) + '\n')
                    f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                except json.JSONDecodeError:
                    logger.warning('Skipping malformed JSON in %s line %d.', src_file.name, line_no)
                    raise

        shutil.move(tmp_tgt_ebd_file, tgt_ebd_file)
        cur = next_month
