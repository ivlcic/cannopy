import csv
import shutil
from logging import Logger
from pathlib import Path
from typing import Any

# noinspection PyPackageRequirements
import faiss
import numpy as np

from ..embed.newsmon import load_embedding_sidecar, build_hnsw_index
from ..prepare.newsmon import (
    get_subset_name,
    get_sidecar_name,
    get_subset_paths,
    apply_min_label_count,
    write_labels_file,
    read_csv_to_dict,
)
from ...app.args.data import DataArguments
from ...app.args.model import ModelArguments
from ...app.args.runtime import Paths
from ...app.helpers import JsonlLoader

logger: Logger
paths: Paths


def _label_string(labels: Any) -> str:
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()
    if isinstance(labels, (list, tuple)):
        return "|".join(str(label) for label in labels)
    return str(labels)


def _find(parent: list[int], item: int) -> int:
    while parent[item] != item:
        parent[item] = parent[parent[item]]
        item = parent[item]
    return item


def _union(parent: list[int], left: int, right: int) -> bool:
    left_root = _find(parent, left)
    right_root = _find(parent, right)
    if left_root == right_root:
        return False
    parent[right_root] = left_root
    return True


def _write_duplicates_csv(target_file: Path, sidecar: dict[str, np.ndarray], embeddings: np.ndarray, index: faiss.Index,
                          top_k: int = 10, sim_threshold: float = 0.98) -> tuple[int, set[str]]:
    ids = sidecar["ids"]
    labels = sidecar.get("labels")

    seen_pairs: set[tuple[int, int]] = set()
    candidate_pairs: list[tuple[float, int, int]] = []
    k = min(top_k, len(ids))
    # noinspection PyArgumentList
    similarities, neighbors = index.search(embeddings, k)
    for row_index, (row_similarities, row_neighbors) in enumerate(zip(similarities, neighbors)):
        for similarity, col_index in zip(row_similarities.tolist(), row_neighbors.tolist()):
            if col_index < 0 or col_index == row_index:
                continue
            left_index = min(row_index, col_index)
            right_index = max(row_index, col_index)
            pair = (left_index, right_index)
            if pair in seen_pairs:
                continue
            exact_similarity = float(np.dot(embeddings[left_index], embeddings[right_index]))
            exact_similarity = min(1.0, max(-1.0, exact_similarity))
            if exact_similarity < sim_threshold:
                continue
            seen_pairs.add(pair)
            candidate_pairs.append((exact_similarity, left_index, right_index))

    candidate_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    parent = list(range(len(ids)))
    kept_pairs: list[tuple[float, int, int]] = []
    for exact_similarity, left_index, right_index in candidate_pairs:
        if _union(parent, left_index, right_index):
            kept_pairs.append((exact_similarity, left_index, right_index))

    with target_file.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(
            [
                "id_left",
                "id_right",
                "similarity",
                "labels_left",
                "labels_right"
            ]
        )
        for exact_similarity, left_index, right_index in kept_pairs:
            writer.writerow(
                [
                    ids[left_index],
                    ids[right_index],
                    exact_similarity,
                    _label_string(labels[left_index]) if labels is not None else "",
                    _label_string(labels[right_index]) if labels is not None else ""
                ]
            )

    removed_ids = {str(ids[right_index]) for _, _, right_index in kept_pairs}
    return len(kept_pairs), removed_ids


# noinspection DuplicatedCode
def dedup(data_args: DataArguments, model_args: ModelArguments) -> None:
    subset = get_subset_name(data_args)

    embed_dir = paths.get_ctx_path('embed')
    if not embed_dir.exists():
        raise FileNotFoundError(f"Embedding directory not found: {embed_dir}")
    sidecar_file = embed_dir / get_sidecar_name(data_args, model_args)
    if not sidecar_file.exists():
        raise FileNotFoundError(f"Embedding sidecar file not found: {sidecar_file}")
    logger.info("Loading embedding sidecar: %s ...", sidecar_file)
    sidecar = load_embedding_sidecar(sidecar_file)
    if "ids" not in sidecar or "embeddings" not in sidecar or "labels" not in sidecar or "label_ids" not in sidecar:
        raise ValueError(f"Invalid embedding sidecar file: {sidecar_file}")

    embeddings = np.asarray(sidecar["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"Invalid embeddings array in sidecar .npz file: expected 2D array.")
    logger.info("Loaded embedding sidecar: %s.", sidecar_file)
    logger.info("Building HNSW index ...")
    index = build_hnsw_index(data_args, embeddings)
    logger.info("Built HNSW index.")

    attrs = data_args.sampling.attributes
    sim_threshold: float = attrs['sim_threshold'] if 'sim_threshold' in attrs else 0.99
    top_k: int = attrs['top_k'] if 'top_k' in attrs else 10

    prepare_dir = paths.get_ctx_path('prepare')
    duplicates_file = prepare_dir / f"{subset}.duplicates.csv"
    logger.info(
        "Writing near-duplicate pairs at similarity >= %.2f and top-k %s to %s ...",
        sim_threshold,
        top_k,
        duplicates_file
    )
    duplicate_count, removed_ids = _write_duplicates_csv(
        duplicates_file, sidecar, embeddings, index, top_k, sim_threshold
    )
    logger.info(
        "Wrote %d near-duplicate pairs at similarity >= %.2f and top-k %s to %s",
        duplicate_count,
        sim_threshold,
        top_k,
        duplicates_file,
    )

    src_data_file, src_labels_file = get_subset_paths(data_args, prepare_dir)
    tgt_data_file, tgt_labels_file = get_subset_paths(data_args, paths.context)
    if not src_data_file.exists():
        raise FileNotFoundError(f'Prepared subset file not found: {src_data_file}')
    if not src_labels_file.exists():
        raise FileNotFoundError(f'Prepared subset labels file not found: {src_labels_file}')

    logger.info("Removing duplicate-linked samples %s ...", src_data_file)
    samples = JsonlLoader.load_samples(src_data_file)
    filtered_samples = [sample for sample in samples if str(sample.get('id', '')) not in removed_ids]
    filtered_samples, label_counts = apply_min_label_count(data_args, filtered_samples)

    labels_map = read_csv_to_dict(src_labels_file)
    JsonlLoader.write_samples(tgt_data_file, filtered_samples)
    write_labels_file(tgt_labels_file, labels_map, label_counts)
    logger.info("Removed duplicate-linked samples %s.", src_data_file)
    logger.info(
        'Wrote %d resampled samples after removing %d duplicate-linked samples to %s',
        len(filtered_samples),
        len(removed_ids),
        tgt_data_file,
    )
    logger.info('Wrote %d labels to %s', len(label_counts), tgt_labels_file)
    orig_data_file, orig_labels_file = get_subset_paths(data_args, prepare_dir, subset=f'{subset}.orig')
    logger.info('Moving original samples from %s to %s', src_data_file, orig_data_file)
    shutil.move(src_data_file, orig_data_file)
    logger.info('Moving original labels from %s to %s', src_labels_file, orig_labels_file)
    shutil.move(src_labels_file, orig_labels_file)
    logger.info('Moving resampled samples from %s to %s', src_data_file, tgt_data_file)
    shutil.move(tgt_data_file, src_data_file)
    logger.info('Moving resampled labels from %s to %s', src_data_file, tgt_data_file)
    shutil.move(tgt_labels_file, src_labels_file)
