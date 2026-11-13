import faiss
import numpy as np


def predict(train_embeddings: np.ndarray, train_labels: np.ndarray,
            test_embeddings: np.ndarray, *, batch_size: int) -> np.ndarray:
    """Transfer the nearest training document's labels, as in EMMA fa_test_zshot."""
    index = faiss.IndexFlatL2(train_embeddings.shape[1])
    index.add(train_embeddings)
    scores = np.empty((len(test_embeddings), train_labels.shape[1]), dtype=np.float32)
    for start in range(0, len(test_embeddings), batch_size):
        end = min(start + batch_size, len(test_embeddings))
        _, neighbors = index.search(test_embeddings[start:end], 1)
        scores[start:end] = train_labels[neighbors[:, 0]]
    return scores
