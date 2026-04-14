import json
import shutil
import statistics
import time

from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ...app.helpers import JsonIdHelper
from ...app.args.model import ModelArguments
from ...app.embedder import TextEmbedder

logger: Logger
paths: Dict[str, Any]


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


def load_embeddings(file_name: Path) -> Dict[str, List[float]]:
    embeddings: Dict[str, List[float]] = {}
    with file_name.open('r', encoding='utf-8') as f_in:
        for line_no, line in enumerate(f_in, start=1):
            obj, _ = JsonIdHelper.read_sample(line, line_no, file_name)
            if not obj:
                continue
            vec = obj.get('embeddings', obj.get('embedding'))
            if vec is None:
                logger.warning('Missing embeddings in %s line %d.', file_name, line_no)
                continue
            # noinspection PyTypeChecker
            embeddings[obj['id']] = vec
    return embeddings


def get_sample_text(sample: Dict[str, Any]) -> str:
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


def build_stats_dict(measurements: EmbedMeasurements) -> Dict[str, Any]:
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


def embed_prepared_dataset(source_file: Path, target_dir: Path, model_args: ModelArguments,
                           log: Logger, subset: Optional[str] = None) -> Dict[str, List[float]]:
    target_name = f'{subset}.{model_args.short_name}'
    target_file = target_dir / f'{target_name}.jsonl'
    target_stats_file = target_dir / f'{target_name}.stats.jsonl'
    tmp_target_file = target_dir / f'tmp.{target_name}'
    write_stats = not target_file.exists()

    cached_embeddings: Dict[str, List[float]] = {}
    if target_file.exists():
        log.info('Target embedding data %s exists. Will reuse cached vectors.', target_file)
        cached_embeddings = load_embeddings(target_file)

    measurements = EmbedMeasurements(
        subset=subset,
        model=model_args.short_name,
        batch_size=model_args.batch_size,
        startup_vram_bytes=_get_vram_bytes(),
    )
    embedder = TextEmbedder.create(model_args)
    measurements.model_vram_bytes = _get_vram_bytes()
    embeddings: Dict[str, List[float]] = dict(cached_embeddings)
    if write_stats and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except RuntimeError:
            log.warning('Unable to reset CUDA peak memory stats.')

    run_started = time.perf_counter()
    with source_file.open('r', encoding='utf-8') as f_in, tmp_target_file.open('w', encoding='utf-8') as f_out:
        batch_ids: List[str] = []
        batch_texts: List[str] = []

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
            for s_id, vec in zip(batch_ids, vectors):
                # noinspection PyTypeChecker
                embeddings[s_id] = vec
                f_out.write(json.dumps({'id': s_id, 'embedding': vec}, ensure_ascii=False) + '\n')
            batch_ids.clear()
            batch_texts.clear()

        for line_no, line in enumerate(f_in, start=1):
            sample, sample_id = JsonIdHelper.read_sample(line, line_no, source_file)
            if not sample_id or not sample:
                continue
            cached_vec = cached_embeddings.get(sample_id)
            if cached_vec is not None:
                embeddings[sample_id] = cached_vec
                f_out.write(json.dumps({'id': sample_id, 'embedding': cached_vec}, ensure_ascii=False) + '\n')
                continue

            batch_ids.append(sample_id)
            batch_texts.append(get_sample_text(sample))
            if len(batch_ids) >= model_args.batch_size:
                flush_batch()

        flush_batch()

    shutil.move(tmp_target_file, target_file)
    measurements.total_runtime_seconds = time.perf_counter() - run_started
    measurements.peak_vram_bytes = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0

    if write_stats:
        with target_stats_file.open('w', encoding='utf-8') as f_stats:
            json.dump(build_stats_dict(measurements), f_stats, ensure_ascii=False, indent=2)

    log.info('Embedded prepared data %s into %s', source_file, target_file)
    return embeddings


def collect_split_embeddings(split_source_file: Path, embeddings: Dict[str, List[float]]) -> Dict[str, List[float]]:
    split_embeddings: Dict[str, List[float]] = {}
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
