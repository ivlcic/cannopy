"""Retrieve candidate topic labels from saved, language-specific training embeddings."""

import re
import time
from numbers import Real
from pathlib import Path

import faiss
import numpy as np

from ....app.args.model import ModelArguments
from ....app.labeler import MultilabelLabeler
from ....data.prepare.newsmon import get_sidecar_name, get_subset_name


def memory_settings(attributes: dict) -> dict | None:
    options = attributes.get('memory')
    if options is None:
        return None
    if not isinstance(options, dict):
        raise ValueError('data.attributes.memory must be a mapping')
    unknown = set(options) - {'model', 'threshold', 'top_n'}
    if unknown:
        raise ValueError(f'Unknown memory settings: {", ".join(sorted(unknown))}')
    model = options.get('model')
    if model is None:
        return None
    if not isinstance(model, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', model):
        raise ValueError('memory.model must be a saved embedding model short name, such as bge-m3')
    threshold = options.get('threshold', 0.0)
    if isinstance(threshold, bool) or not isinstance(threshold, Real) or not np.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError('memory.threshold must be a finite cosine similarity in [-1, 1]')
    top_n = options.get('top_n', 5)
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError('memory.top_n must be a positive integer')
    return {'model': model, 'threshold': float(threshold), 'top_n': top_n}


def _vectors(arrays, path: Path):
    ids = np.asarray(arrays['ids'], dtype=str)
    vectors = np.asarray(arrays['embeddings'], dtype=np.float32)
    if ids.ndim != 1 or not len(ids) or len(set(ids)) != len(ids) or any(not value for value in ids):
        raise ValueError(f'Expected non-empty unique sample IDs in {path}')
    if vectors.ndim != 2 or vectors.shape[0] != len(ids) or vectors.shape[1] == 0:
        raise ValueError(f'Embedding shape does not match sample IDs in {path}')
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError(f'Expected finite, non-zero embedding vectors in {path}')
    return ids, np.ascontiguousarray(vectors / norms)


class LabelMemory:
    """One training-only cosine index; query sidecars supply vectors, never labels."""

    def __init__(self, directory: Path, data_args, settings: dict, labeler: MultilabelLabeler):
        self.started = time.perf_counter()
        self.settings, self.labeler = settings, labeler
        model_args = ModelArguments(short_name=settings['model'])
        self.train_file = directory / get_sidecar_name(data_args, model_args, 'train')
        self.query_file = directory / get_sidecar_name(data_args, model_args, 'test')
        for path in (self.train_file, self.query_file):
            if not path.is_file():
                language = get_subset_name(data_args).rsplit('.', 1)[-1]
                raise FileNotFoundError(
                    f'Memory embedding sidecar not found: {path}. '
                    f'Run ./data embed {data_args.dataset_name} -c {language} -c {settings["model"]} first.'
                )
        with np.load(self.train_file, allow_pickle=True) as arrays:
            self.ids, train_vectors = _vectors(arrays, self.train_file)
            try:
                self.labels = labeler.encode(arrays['labels'].tolist()).astype(bool)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f'Invalid memory training labels in {self.train_file}: {error}') from error
        if self.labels.shape != (len(self.ids), labeler.num_labels):
            raise ValueError(f'Training label shape does not match memory samples in {self.train_file}')
        # Do not access labels/label_ids from the test sidecar.
        with np.load(self.query_file, allow_pickle=False) as arrays:
            query_ids, self.query_vectors = _vectors(arrays, self.query_file)
        if set(self.ids) & set(query_ids):
            raise ValueError('Memory training and test sample IDs overlap')
        if train_vectors.shape[1] != self.query_vectors.shape[1]:
            raise ValueError('Memory training and query embedding dimensions differ')
        self.query_rows = {sample_id: row for row, sample_id in enumerate(query_ids)}
        self.index = faiss.IndexFlatIP(train_vectors.shape[1])
        self.index.add(train_vectors)

    def retrieve(self, ids) -> tuple[list[list[str]], dict]:
        try:
            rows = [self.query_rows[sample_id] for sample_id in ids]
        except KeyError as error:
            raise ValueError(f'Missing memory query embedding for sample {error.args[0]} in {self.query_file}') from error
        similarities, neighbors = self.index.search(self.query_vectors[rows], min(self.settings['top_n'], len(self.ids)))
        candidates, queries = [], []
        for sample_id, scores, indices in zip(ids, similarities, neighbors, strict=True):
            accepted = [(int(index), float(np.clip(score, -1, 1))) for index, score in zip(indices, scores, strict=True)
                        if index >= 0 and np.clip(score, -1, 1) >= self.settings['threshold']]
            present = np.any(self.labels[[index for index, _ in accepted]], axis=0) if accepted else np.zeros(self.labeler.num_labels, bool)
            codes = [code for code, keep in zip(self.labeler.classes, present, strict=True) if keep]
            candidates.append(codes)
            queries.append({'id': str(sample_id), 'candidate_labels': codes,
                            'neighbors': [{'id': str(self.ids[index]), 'similarity': score} for index, score in accepted]})
        return candidates, {
            **self.settings, 'metric': 'cosine', 'selection': 'union of retrieved training samples labels',
            'sources': [{'train_sidecar': str(self.train_file), 'query_sidecar': str(self.query_file), 'num_train': len(self.ids)}],
            'num_queries': len(ids), 'num_queries_without_candidates': sum(not codes for codes in candidates),
            'num_candidate_labels': sum(map(len, candidates)), 'retrieval_seconds': time.perf_counter() - self.started,
            'queries': queries,
        }


def pool_memory(evaluations) -> dict:
    """Pool retrieval records and counts without averaging language-level rates."""
    memories = [evaluation.metadata['memory'] for evaluation in evaluations]
    result = {**memories[0]}
    for key in ('sources', 'queries'):
        result[key] = [item for memory in memories for item in memory[key]]
    for key in ('num_queries', 'num_queries_without_candidates', 'num_candidate_labels', 'retrieval_seconds',
                'num_gold_labels', 'num_gold_candidate_labels'):
        result[key] = sum(memory[key] for memory in memories)
    result['candidate_label_recall'] = (
        result['num_gold_candidate_labels'] / result['num_gold_labels'] if result['num_gold_labels'] else 0.0
    )
    return result
