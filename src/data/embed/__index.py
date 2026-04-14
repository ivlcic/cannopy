from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from ...app.args.data import DataArguments


def must_build_index(data_args: DataArguments) -> bool:
    index = data_args.source.attributes.get('index')
    if index:
        return True
    return False


def build_embedding_array_dict(embeddings: Dict[str, Sequence[float]]) -> Dict[str, np.ndarray]:
    ids: List[str] = list(embeddings.keys())
    if not ids:
        return {
            'ids': np.asarray([], dtype=str),
            'embeddings': np.empty((0, 0), dtype=np.float32),
        }

    vectors = np.asarray([embeddings[sample_id] for sample_id in ids], dtype=np.float32)
    return {
        'ids': np.asarray(ids, dtype=str),
        'embeddings': vectors,
    }


def store_embedding_array_dict(target_file: Path, embeddings: Dict[str, Sequence[float]]) -> Dict[str, np.ndarray]:
    embedding_array_dict = build_embedding_array_dict(embeddings)
    np.savez_compressed(target_file, **embedding_array_dict)
    return embedding_array_dict
