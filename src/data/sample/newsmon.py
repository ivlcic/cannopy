import json
from logging import Logger
from typing import Any, Optional

import numpy as np

from app.args.model import ModelArguments
from ..embed.newsmon import load_embedding_sidecar, build_hnsw_index
from ..prepare.newsmon import get_subset_name, get_subset_data_path, get_sidecar_name
from ...app.args.data import DataArguments
from ...app.args.runtime import Paths
from ...app.helpers import JsonlLoader

logger: Logger
paths: Paths


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


def _target_file_name(data_args: DataArguments, model_args: ModelArguments, split: Optional[str] = None) -> str:
    subset = get_subset_name(data_args)
    if split:
        return f'{subset}.{model_args.short_name}.{split}.hn.jsonl'
    return f'{subset}.{model_args.short_name}.hn.jsonl'


# noinspection DuplicatedCode
def hard_neg(data_args: DataArguments, model_args: ModelArguments) -> None:
    logger.info('Preparing NewsMon hard-negative dataset')

    split = data_args.sampling.attributes.get('split', None)

    if split:
        split_dir = paths.get_ctx_path('split')
        source_file = get_subset_data_path(data_args, split_dir, split)
        if not source_file.exists():
            raise FileNotFoundError(f'Prepared {split} subset file not found: {source_file}')

        sidecar_file = split_dir / get_sidecar_name(data_args, model_args, 'train')
        if not sidecar_file.exists():
            raise FileNotFoundError(f'Embedding {split} sidecar file not found: {sidecar_file}')
    else:
        embed_dir = paths.get_ctx_path('embed')
        source_file = get_subset_data_path(data_args, embed_dir)
        if not source_file.exists():
            raise FileNotFoundError(f'Prepared subset file not found: {source_file}')

        sidecar_file = embed_dir / get_sidecar_name(data_args, model_args)
        if not sidecar_file.exists():
            raise FileNotFoundError(f'Embedding sidecar file not found: {sidecar_file}')

    logger.info('Loading training samples from %s', source_file)
    samples = JsonlLoader.load_samples(source_file)
    logger.info('Loading embedding sidecar from %s', sidecar_file)
    sidecar = load_embedding_sidecar(sidecar_file)
    if 'ids' not in sidecar or 'embeddings' not in sidecar or 'label_ids' not in sidecar:
        raise ValueError(f'Invalid embedding sidecar file: {sidecar_file}')

    ids = np.asarray(sidecar['ids'])
    embeddings = np.asarray(sidecar['embeddings'], dtype=np.float32)
    label_ids = np.asarray(sidecar['label_ids'], dtype=np.int64)
    if embeddings.ndim != 2:
        raise ValueError(f'Invalid embeddings array in sidecar .npz file: expected 2D array.')
    if label_ids.ndim != 2:
        raise ValueError(f'Invalid label_ids array in sidecar .npz file: expected 2D array.')
    if label_ids.shape[0] != embeddings.shape[0]:
        raise ValueError(f'Invalid label_ids array in sidecar .npz file: shape mismatch.')
    if len(samples) != len(ids):
        raise ValueError(
            f'Sample count mismatch between prepared data ({len(samples)}) and sidecar ids ({len(ids)}).'
        )
    logger.info(
        'Loaded %d samples with embedding dim=%d and label dim=%d',
        len(samples),
        embeddings.shape[1],
        label_ids.shape[1],
    )

    samples_by_id = {str(sample['id']): sample for sample in samples if 'id' in sample}
    ordered_samples: list[dict[str, Any]] = []
    for sample_id in ids.tolist():
        sample = samples_by_id.get(str(sample_id))
        if sample is None:
            raise KeyError(f'Missing prepared sample for sidecar id: {sample_id}')
        ordered_samples.append(sample)

    index = build_hnsw_index(data_args, embeddings)
    top_k = min(
        int(data_args.sampling.attributes.get('top_k', 1000)),
        len(ids),
    )
    negative_k = int(data_args.sampling.attributes.get('hard_neg_k', 15))
    progress_every = int(data_args.sampling.attributes.get('progress_every', 1000))
    logger.info(
        'Building hard negatives with top_k=%d, hard_neg_k=%d, progress_every=%d',
        top_k,
        negative_k,
        progress_every,
    )
    # noinspection PyArgumentList
    _, neighbors = index.search(embeddings, top_k)
    logger.info('Neighbor search completed for %d queries', len(ids))

    if split:
        target_file = paths.get_ctx_path('split') / _target_file_name(data_args, model_args, split)
    else:
        target_file = paths.get_ctx_path('embed') / _target_file_name(data_args, model_args)

    written = 0
    skipped = 0
    skipped_missing_text = 0
    skipped_missing_matches = 0
    with target_file.open('w', encoding='utf-8') as f_out:
        for row_index, row_neighbors in enumerate(neighbors):
            query_sample = ordered_samples[row_index]
            query_text = _sample_text(query_sample)
            if not query_text:
                skipped += 1
                skipped_missing_text += 1
                continue

            positive_text: str | None = None
            negative_texts: list[str] = []
            query_labels = label_ids[row_index]

            for neighbor_index in row_neighbors.tolist():
                if neighbor_index < 0 or neighbor_index == row_index:
                    continue
                neighbor_sample = ordered_samples[neighbor_index]
                # noinspection PyTypeChecker
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
                skipped_missing_matches += 1
                continue

            record = {
                'query': query_text,
                'pos': [positive_text],
                'neg': negative_texts,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
            written += 1
            if progress_every > 0 and (row_index + 1) % progress_every == 0:
                logger.info(
                    'Processed %d/%d queries; written=%d skipped=%d',
                    row_index + 1,
                    len(ids),
                    written,
                    skipped,
                )

    logger.info(
        'Wrote %d hard-negative IR samples to %s; skipped=%d (missing_text=%d, insufficient_matches=%d)',
        written,
        target_file,
        skipped,
        skipped_missing_text,
        skipped_missing_matches,
    )
