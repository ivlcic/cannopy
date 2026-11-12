import faiss
import numpy as np


def predict(train_embeddings: np.ndarray, train_labels: np.ndarray,
            test_embeddings: np.ndarray, *, batch_size: int,
            top_k: int = 50, temperature: float = 0.04, metric: str = 'l2') -> np.ndarray:
    """Aggregate retrieved label vectors with temperature-scaled softmax weights.

    L2 uses negative Euclidean distance (FAISS returns squared L2 distances).
    Inner product uses cosine similarity on the normalized embedding sidecars.
    Only training-document labels contribute; the reference's lambda is fixed at 1.
    """
    if not isinstance(top_k, (int, np.integer)) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError('rae_xmc.top_k must be a positive integer')
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('rae_xmc.temperature must be finite and positive')
    if metric not in ('l2', 'ip'):
        raise ValueError("rae_xmc.metric must be 'l2' or 'ip'")
    if len(train_embeddings) == 0:
        raise ValueError('rae_xmc requires non-empty training embeddings')

    index_type = faiss.IndexFlatL2 if metric == 'l2' else faiss.IndexFlatIP
    index = index_type(train_embeddings.shape[1])
    index.add(train_embeddings)
    neighbors_count = min(top_k, len(train_embeddings))
    scores = np.empty((len(test_embeddings), train_labels.shape[1]), dtype=np.float32)
    for start in range(0, len(test_embeddings), batch_size):
        end = min(start + batch_size, len(test_embeddings))
        similarities, neighbors = index.search(test_embeddings[start:end], neighbors_count)
        if metric == 'l2':
            similarities = -np.sqrt(np.maximum(similarities, 0.0))
        logits = (similarities - similarities.max(axis=1, keepdims=True)) / temperature
        weights = np.exp(logits)
        weights /= weights.sum(axis=1, keepdims=True)
        scores[start:end] = np.einsum('bk,bkl->bl', weights, train_labels[neighbors], optimize=True)
    return np.clip(scores, 0.0, 1.0)
