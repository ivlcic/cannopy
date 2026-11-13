import numpy as np

from .....app.model.mlknn import MLKNN


def predict(train_embeddings: np.ndarray, train_labels: np.ndarray,
            test_embeddings: np.ndarray, *, batch_size: int,
            n_neighbors: int = 10, smoothing: float = 1.0,
            metric: str = 'euclidean', device: str = 'cpu') -> np.ndarray:
    """Estimate ML-kNN label posteriors from training-neighbor label counts."""
    if not isinstance(n_neighbors, (int, np.integer)) or isinstance(n_neighbors, bool) or n_neighbors <= 0:
        raise ValueError('mlknn.n_neighbors must be a positive integer')
    if not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError('mlknn.smoothing must be finite and positive')
    classifier = MLKNN(n_neighbors=n_neighbors, smoothing=smoothing, metric=metric, device=device)
    classifier.fit(train_embeddings, train_labels)
    scores = np.empty((len(test_embeddings), train_labels.shape[1]), dtype=np.float32)
    for start in range(0, len(test_embeddings), batch_size):
        end = min(start + batch_size, len(test_embeddings))
        scores[start:end] = classifier.predict_proba(test_embeddings[start:end]).cpu().numpy()
    return scores
