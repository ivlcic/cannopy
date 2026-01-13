import csv
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

def read_csv_to_dict(path: str, key_col: str = "id") -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = row.get(key_col)
            if not key:
                raise ValueError(f"Missing '{key_col}' value in row: {row}")
            if key in out:
                raise ValueError(f"Duplicate '{key_col}' value: {key}")
            out[key] = row
    return out


def load_embeddings(file_name) -> Dict[str, List[float]]:
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
                # accept both embeddings/embedding keys
                vec = obj.get('embeddings', obj.get('embedding'))
                if vec is None:
                    logger.warning('Missing embeddings in %s line %d.', file_name, line_no)
                    continue
                embeddings[obj['id']] = vec
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

    map_media = read_csv_to_dict(source_dir / f'map_media.csv')

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

    while cur < end:
        next_month = cur + relativedelta(months=1)
        src_file = source_dir / f'data_{start.year}_{cur.month:02d}.jsonl'
        if not src_file.exists():
            raise FileNotFoundError(
                f"Data file not found: {src_file}, check data.source.select.start and data.source.select.end"
            )
        map_file = source_dir / f'map_articles_{start.year}_{cur.month:02d}.csv'
        if not map_file.exists():
            raise FileNotFoundError(
                f"Map file not found: {src_file}, check data.source.select.start and data.source.select.end"
            )
        map_articles = read_csv_to_dict(map_file)

        tgt_file = target_dir / f'{subset}data_{start.year}_{cur.month:02d}.jsonl'

        src_ebd_file = source_dir / f'data_{start.year}_{cur.month:02d}-{model_args.short_name}.jsonl'
        src_ebd: Dict[str, List[float]] = {}
        if src_ebd_file.exists():
            logger.info(
                f"Source embedding data {src_ebd_file} for model {model_args.short_name} exists. Will use that!"
            )
            src_ebd = load_embeddings(src_ebd_file)

        tgt_ebd_file = target_dir / f'{subset}data_{start.year}_{cur.month:02d}-{model_args.short_name}.jsonl'
        tmp_tgt_ebd_file = target_dir / f'tmp.{subset}data_{start.year}_{cur.month:02d}-{model_args.short_name}.jsonl'
        tgt_ebd: Dict[str, List[float]] = {}
        if tgt_ebd_file.exists():
            logger.info(
                f"Target embedding data {tgt_ebd_file} for model {model_args.short_name} exists. Will use that!"
            )
            tgt_ebd = load_embeddings(tgt_ebd_file)

        with (src_file.open('r', encoding='utf-8') as f_in,
              #tgt_file.open('w', encoding='utf-8') as f_out,
              tmp_tgt_ebd_file.open('w', encoding='utf-8') as f_ebd_out):
            batch_ids: List[str] = []
            batch_texts: List[str] = []

            def flush_batch() -> None:
                if not batch_ids:
                    return
                vectors = embedder.embed(batch_texts)
                if len(vectors) != len(batch_ids):
                    raise RuntimeError(
                        f'Embedding count mismatch (got {len(vectors)} vectors for {len(batch_ids)} ids)'
                    )
                for sid, vec in zip(batch_ids, vectors):
                    f_ebd_out.write(json.dumps({'id': sid, 'embedding': vec}, ensure_ascii=False) + '\n')
                batch_ids.clear()
                batch_texts.clear()

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

                    article = map_articles[obj['id']]
                    reach = 0
                    source_type = None
                    source_name = None
                    url = None
                    if obj['m_id'] in map_media:
                        source = map_media[obj['m_id']]
                        if 'reach' in source:
                            reach = int(source['reach'])
                        if 'type' in source:
                            source_type = source['type']
                        if 'name' in source:
                            source_name = source['name']
                    else:
                        logger.warning(
                            'Missing %s article media in %s line %d.', obj['m_id'], src_file, line_no
                        )
                    if 'url' in article:
                        url = article['url']
                    created = datetime.fromisoformat(article['created'].replace('Z', '+00:00'))
                    published = datetime.fromisoformat(article['published'].replace('Z', '+00:00'))
                    obj['reach'] = reach
                    obj['type'] = source_type
                    obj['source'] = source_name
                    obj['url'] = url
                    obj['uuid'] = article['uuid']
                    obj['published'] = published.isoformat().replace("+00:00", "Z")
                    obj['created'] = created.isoformat().replace("+00:00", "Z")

                    sample_id = obj['id']
                    cached_vec = src_ebd.get(sample_id) or tgt_ebd.get(sample_id)
                    if cached_vec:
                        f_ebd_out.write(json.dumps({'id': sample_id, 'embedding': cached_vec}, ensure_ascii=False) + '\n')
                    else:
                        batch_ids.append(sample_id)
                        batch_texts.append(_get_text(obj))
                        if len(batch_ids) >= model_args.batch_size:
                            flush_batch()

                    #f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                except json.JSONDecodeError:
                    logger.warning('Skipping malformed JSON in %s line %d.', src_file.name, line_no)
                    raise
            flush_batch()

        shutil.move(tmp_tgt_ebd_file, tgt_ebd_file)
        cur = next_month
