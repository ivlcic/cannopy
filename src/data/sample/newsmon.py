import json
from logging import Logger
from pathlib import Path
from typing import Any

import numpy as np

from app.args.model import ModelArguments
from ..prepare.newsmon import get_subset_name, get_subset_paths, get_sidecar_name, build_hnsw_index
from ...app.args.data import DataArguments
from ...app.args.runtime import Paths
from ...app.helpers import JsonlLoader

logger: Logger
paths: Paths

HARD_NEGATIVE_COUNT = 15
HNSW_SEARCH_K = 128


def _load_embedding_sidecar(sidecar_file: Path) -> dict[str, np.ndarray]:
    with np.load(sidecar_file, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return embeddings / norms


def _labels_overlap(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.any(np.logical_and(left != 0, right != 0)))


def _sample_text(sample: dict[str, Any]) -> str:
    text = sample.get('text', '')
    if text:
        return str(text)
    title = sample.get('title', {}).get('text', '') or ''
    body = sample.get('body', {}).get('text', '') or ''
    if title and body:
        return f'{title}\n\n{body}'
    return title or body


def _target_file_name(data_args: DataArguments, model_args: ModelArguments) -> str:
    subset = get_subset_name(data_args)
    return f'{subset}.{model_args.short_name}.hard-negatives.jsonl'


# noinspection DuplicatedCode
def main(data_args: DataArguments, model_args: ModelArguments) -> None:
    prepare_dir = paths.get_ctx_path('prepare')
    source_file, _ = get_subset_paths(data_args, prepare_dir)
    if not source_file.exists():
        raise FileNotFoundError(f'Prepared subset file not found: {source_file}')

    sidecar_file = paths.get_ctx_path('embed') / get_sidecar_name(data_args, model_args)
    if not sidecar_file.exists():
        raise FileNotFoundError(f'Embedding sidecar file not found: {sidecar_file}')

    samples = JsonlLoader.load_samples(source_file)
    sidecar = _load_embedding_sidecar(sidecar_file)
    if 'ids' not in sidecar or 'embeddings' not in sidecar or 'label_ids' not in sidecar:
        raise ValueError(f'Invalid embedding sidecar file: {sidecar_file}')

    ids = np.asarray(sidecar['ids'])
    embeddings = np.asarray(sidecar['embeddings'], dtype=np.float32)
    label_ids = np.asarray(sidecar['label_ids'], dtype=np.int64)
    if embeddings.ndim != 2:
        raise ValueError(f'Invalid embeddings array in sidecar .npz file: expected 2D array.')
    if label_ids.ndim != 2 or label_ids.shape[0] != embeddings.shape[0]:
        raise ValueError(f'Invalid label_ids array in sidecar .npz file: shape mismatch.')
    if len(samples) != len(ids):
        raise ValueError(
            f'Sample count mismatch between prepared data ({len(samples)}) and sidecar ids ({len(ids)}).'
        )

    samples_by_id = {str(sample['id']): sample for sample in samples if 'id' in sample}
    ordered_samples: list[dict[str, Any]] = []
    for sample_id in ids.tolist():
        sample = samples_by_id.get(str(sample_id))
        if sample is None:
            raise KeyError(f'Missing prepared sample for sidecar id: {sample_id}')
        ordered_samples.append(sample)

    normalized = _normalize_embeddings(embeddings)
    index = build_hnsw_index(data_args, normalized)
    top_k = min(
        int(data_args.sampling.attributes.get('top_k', HNSW_SEARCH_K)),
        len(ids),
    )
    negative_k = int(data_args.sampling.attributes.get('hard_negative_k', HARD_NEGATIVE_COUNT))
    # noinspection PyArgumentList
    _, neighbors = index.search(normalized, top_k)

    target_file = paths.context / _target_file_name(data_args, model_args)
    written = 0
    skipped = 0
    with target_file.open('w', encoding='utf-8') as f_out:
        for row_index, row_neighbors in enumerate(neighbors):
            query_sample = ordered_samples[row_index]
            query_text = _sample_text(query_sample)
            if not query_text:
                skipped += 1
                continue

            positive_text: str | None = None
            negative_texts: list[str] = []
            query_labels = label_ids[row_index]

            for neighbor_index in row_neighbors.tolist():
                if neighbor_index < 0 or neighbor_index == row_index:
                    continue
                neighbor_sample = ordered_samples[neighbor_index]
                neighbor_text = _sample_text(neighbor_sample)
                if not neighbor_text:
                    continue
                if _labels_overlap(query_labels, label_ids[neighbor_index]):
                    if positive_text is None:
                        positive_text = neighbor_text
                    continue
                if len(negative_texts) < negative_k:
                    negative_texts.append(neighbor_text)
                if positive_text is not None and len(negative_texts) >= negative_k:
                    break

            if positive_text is None or len(negative_texts) < negative_k:
                skipped += 1
                continue

            record = {
                'query': query_text,
                'pos': [positive_text],
                'neg': negative_texts,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
            written += 1

    logger.info(
        'Wrote %d hard-negative IR samples to %s; skipped %d samples without enough positives/negatives',
        written,
        target_file,
        skipped,
    )
