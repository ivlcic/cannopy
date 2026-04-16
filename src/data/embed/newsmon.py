import csv
import json
import shutil
import statistics
import time
from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..prepare.newsmon import get_subset_name, get_sidecar_name
from ...app.args.data import DataArguments
from ...app.args.model import ModelArguments
from ...app.args.runtime import Paths
from ...app.embedder import TextEmbedder
from ...app.helpers import JsonIdHelper
from ...app.labeler import MultilabelLabeler

logger: Logger
paths: Paths

EmbeddingRecord = Dict[str, Any]


@dataclass
class EmbedMeasurements:
    subset: Optional[str]
    model: str
    batch_size: int
    startup_vram_bytes: int = 0
    model_vram_bytes: int = 0
    peak_vram_bytes: int = 0
    total_runtime_seconds: float = 0.0
    batch_durations: List[float] = field(default_factory=list)
    batch_sample_counts: List[int] = field(default_factory=list)


def load_embeddings(file_name: Path, labeler: Optional[MultilabelLabeler] = None) -> Dict[str, EmbeddingRecord]:
    embeddings: Dict[str, EmbeddingRecord] = {}
    with file_name.open('r', encoding='utf-8') as f_in:
        for line_no, line in enumerate(f_in, start=1):
            obj, _ = JsonIdHelper.read_sample(line, line_no, file_name)
            if not obj:
                continue
            vec = obj.get('embeddings', obj.get('embedding'))
            if vec is None:
                logger.warning('Missing embeddings in %s line %d.', file_name, line_no)
                continue
            labels = obj.get('labels', [])
            encoded_labels = obj.get('label_ids')
            if encoded_labels is None and labeler is not None:
                encoded_labels = labeler.encode([labels])[0].tolist()
            embeddings[obj['id']] = {
                'embedding': vec,
                'labels': labels,
                'label_ids': encoded_labels,
                'text': obj.get('text', ''),
            }
    return embeddings


def load_multilabel_labeler(source_labels_file: Path) -> MultilabelLabeler:
    labels: List[str] = []
    with source_labels_file.open('r', encoding='utf-8', newline='') as f_in:
        reader = csv.reader(f_in)
        next(reader, None)
        for row in reader:
            if not row or not row[0]:
                continue
            labels.append(row[0])
    return MultilabelLabeler(labels=labels)


def _get_sample_text(sample: Dict[str, Any]) -> str:
    text = sample.get('text', '')
    if text:
        return text
    title = sample.get('title', {}).get('text', '') or ''
    body = sample.get('body', {}).get('text', '') or ''
    if title and body:
        return f'{title}\n\n{body}'
    return title or body


def _get_vram_bytes() -> int:
    if not torch.cuda.is_available():
        return 0
    try:
        return int(torch.cuda.memory_allocated())
    except RuntimeError:
        return 0


def _stddev(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values))


def _build_stats_dict(measurements: EmbedMeasurements) -> Dict[str, Any]:
    per_sample_durations = [
        batch_duration / sample_count
        for batch_duration, sample_count in zip(measurements.batch_durations, measurements.batch_sample_counts)
        if sample_count > 0
    ]
    model_delta = max(0, measurements.model_vram_bytes - measurements.startup_vram_bytes)
    runtime_delta = max(0, measurements.peak_vram_bytes - measurements.startup_vram_bytes - model_delta)
    total_vram = (measurements.startup_vram_bytes
                  + max(0, measurements.peak_vram_bytes - measurements.startup_vram_bytes))
    return {
        'subset': measurements.subset,
        'model': measurements.model,
        'batch_size': measurements.batch_size,
        'vram_total_bytes': total_vram,
        'vram_model_bytes': model_delta,
        'vram_runtime_bytes': runtime_delta,
        'total_runtime_seconds': measurements.total_runtime_seconds,
        'num_batches': len(measurements.batch_durations),
        'num_samples_embedded': sum(measurements.batch_sample_counts),
        'batch_runtime_seconds_avg': (
            float(statistics.fmean(measurements.batch_durations)) if measurements.batch_durations else 0.0
        ),
        'batch_runtime_seconds_std': _stddev(measurements.batch_durations),
        'sample_runtime_seconds_avg': (
            float(statistics.fmean(per_sample_durations)) if per_sample_durations else 0.0
        ),
        'sample_runtime_seconds_std': _stddev(per_sample_durations),
    }


def embed_prepared_dataset(path: Paths, data_args: DataArguments, model_args: ModelArguments,
                           log: Logger) -> Dict[str, EmbeddingRecord]:
    subset = get_subset_name(data_args)

    source_dir = paths.get_ctx_path('prepare')
    source_file = source_dir / f'{subset}.jsonl'
    if not source_file.exists():
        raise FileNotFoundError(f'Prepared subset file not found: {source_file}')

    source_labels_file = source_dir / f'{subset}.labels.csv'
    if not source_labels_file.exists():
        raise FileNotFoundError(f'Prepared subset labels file not found: {source_labels_file}')

    labeler = load_multilabel_labeler(source_labels_file)
    target_name = f'{subset}.{model_args.short_name}'
    target_file = path.context / f'{target_name}.jsonl'
    target_stats_file = path.context / f'{target_name}.stats.jsonl'
    tmp_target_file = path.context / f'tmp.{target_name}'
    write_stats = not target_file.exists()

    cached_embeddings: Dict[str, EmbeddingRecord] = {}
    if target_file.exists():
        log.info('Target embedding data %s exists. Will reuse cached vectors.', target_file)
        cached_embeddings = load_embeddings(target_file, labeler)

    measurements = EmbedMeasurements(
        subset=subset,
        model=model_args.short_name,
        batch_size=model_args.batch_size,
        startup_vram_bytes=_get_vram_bytes(),
    )
    embedder = TextEmbedder.create(model_args)
    measurements.model_vram_bytes = _get_vram_bytes()
    embeddings: Dict[str, EmbeddingRecord] = dict(cached_embeddings)
    if write_stats and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except RuntimeError:
            log.warning('Unable to reset CUDA peak memory stats.')

    run_started = time.perf_counter()
    with (source_file.open('r', encoding='utf-8') as f_in,
          tmp_target_file.open('w', encoding='utf-8') as f_out):
        batch_ids: List[str] = []
        batch_texts: List[str] = []
        batch_labels: List[List[str]] = []

        def flush_batch() -> None:
            if not batch_ids:
                return
            batch_started = time.perf_counter()
            vectors = embedder.embed(batch_texts)
            batch_duration = time.perf_counter() - batch_started
            if len(vectors) != len(batch_ids):
                raise RuntimeError(
                    f'Embedding count mismatch (got {len(vectors)} vectors for {len(batch_ids)} ids)'
                )
            measurements.batch_durations.append(batch_duration)
            measurements.batch_sample_counts.append(len(batch_ids))
            enc_labels = labeler.encode(batch_labels).tolist()
            for s_id, ebd, lbl, lbl_ids in zip(batch_ids, vectors, batch_labels, enc_labels):
                embeddings[s_id] = {
                    'id': s_id,
                    'embedding': ebd,
                    'labels': lbl,
                    'label_ids': lbl_ids
                }
                f_out.write(
                    json.dumps(
                        {'id': s_id, 'embedding': ebd}, ensure_ascii=False
                    ) + '\n'
                )
            batch_ids.clear()
            batch_texts.clear()
            batch_labels.clear()

        for line_no, line in enumerate(f_in, start=1):
            sample, sample_id = JsonIdHelper.read_sample(line, line_no, source_file)
            if not sample_id or not sample:
                continue
            text = _get_sample_text(sample)
            labels = sample.get('label', []) or []
            encoded_labels = labeler.encode([labels]).tolist()
            cached_record = cached_embeddings.get(sample_id)
            if cached_record is not None:
                embeddings[sample_id] = {
                            'id': sample_id,
                            'embedding': cached_record['embedding'],
                            'labels': labels,
                            'label_ids': encoded_labels
                        }
                f_out.write(
                    json.dumps(
                        {'id': sample_id, 'embedding': cached_record['embedding']}, ensure_ascii=False
                    ) + '\n'
                )
                continue

            batch_ids.append(sample_id)
            batch_texts.append(text)
            batch_labels.append(labels)
            if len(batch_ids) >= model_args.batch_size:
                flush_batch()

        flush_batch()

    shutil.move(tmp_target_file, target_file)
    measurements.total_runtime_seconds = time.perf_counter() - run_started
    measurements.peak_vram_bytes = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0

    if write_stats:
        with target_stats_file.open('w', encoding='utf-8') as f_stats:
            json.dump(_build_stats_dict(measurements), f_stats, ensure_ascii=False, indent=2)

    log.info('Embedded prepared data %s into %s', source_file, target_file)
    return embeddings


def collect_split_embeddings(split_source_file: Path,
                             embeddings: Dict[str, EmbeddingRecord]) -> Dict[str, EmbeddingRecord]:
    split_embeddings: Dict[str, EmbeddingRecord] = {}
    with split_source_file.open('r', encoding='utf-8') as f_in:
        for line_no, line in enumerate(f_in, start=1):
            sample, sample_id = JsonIdHelper.read_sample(line, line_no, split_source_file)
            if not sample_id:
                continue
            embedding = embeddings.get(sample_id)
            if embedding is None:
                raise KeyError(f'Missing top-level embedding for split sample id: {sample_id}')
            split_embeddings[sample_id] = embedding
    return split_embeddings


def _build_embedding_array_dict(embeddings: Dict[str, EmbeddingRecord]) -> Dict[str, np.ndarray]:
    ids: List[str] = list(embeddings.keys())
    if not ids:
        return {
            'ids': np.asarray([], dtype=str),
            'embeddings': np.empty((0, 0), dtype=np.float32),
            'labels': np.asarray([], dtype=object),
            'label_ids': np.empty((0, 0), dtype=np.int64),
        }

    vectors = np.asarray([embeddings[sample_id]['embedding'] for sample_id in ids], dtype=np.float32)
    labels = np.asarray([embeddings[sample_id].get('labels', []) for sample_id in ids], dtype=object)
    label_ids = np.asarray([embeddings[sample_id]['label_ids'] for sample_id in ids], dtype=np.int64)
    return {
        'ids': np.asarray(ids, dtype=str),
        'embeddings': vectors,
        'labels': labels,
        'label_ids': label_ids,
    }


def store_embedding_array_dict(target_file: Path, embeddings: Dict[str, EmbeddingRecord]) -> Dict[str, np.ndarray]:
    embedding_array_dict = _build_embedding_array_dict(embeddings)
    np.savez_compressed(target_file, **embedding_array_dict)
    return embedding_array_dict


# noinspection DuplicatedCode
def main(data_args: DataArguments, model_args: ModelArguments) -> None:
    subset = get_subset_name(data_args)
    logger.info('Computing embeddings: %s ...', subset)
    embeddings = embed_prepared_dataset(paths, data_args, model_args, logger)
    target_index_file = paths.context / get_sidecar_name(data_args, model_args)
    logger.info('Writing embedding array sidecar: %s ...', target_index_file)
    store_embedding_array_dict(target_index_file, embeddings)

    split_dir = paths.get_ctx_path('split')
    if not split_dir.exists():
        return

    for split_name in ('train', 'eval', 'test'):
        split_source_file = split_dir / f'{subset}.{split_name}.jsonl'
        if not split_source_file.exists():
            continue
        split_embeddings = collect_split_embeddings(split_source_file, embeddings)
        split_index_file = split_dir / get_sidecar_name(data_args, model_args, split_name)
        logger.info('Writing embedding array sidecar: %s', split_index_file)
        store_embedding_array_dict(split_index_file, split_embeddings)
