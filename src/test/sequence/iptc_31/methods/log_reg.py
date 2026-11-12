import numpy as np
from scipy.sparse import spmatrix
from sklearn.linear_model import LogisticRegression


def predict(train_embeddings: np.ndarray | spmatrix, train_labels: np.ndarray,
            test_embeddings: np.ndarray | spmatrix, *, batch_size: int,
            C: float = 1.0, max_iter: int = 1000,
            class_weight: str | None = None, seed: int = 2611) -> np.ndarray:
    """Fit an independent L2-regularized binary classifier for each label."""
    if not np.isfinite(C) or C <= 0:
        raise ValueError('log_reg.C must be finite and positive')
    if not isinstance(max_iter, (int, np.integer)) or isinstance(max_iter, bool) or max_iter <= 0:
        raise ValueError('log_reg.max_iter must be a positive integer')
    if class_weight not in (None, 'balanced'):
        raise ValueError("log_reg.class_weight must be null or 'balanced'")
    if train_embeddings.shape[0] == 0 or not np.isin(train_labels, [0, 1]).all():
        raise ValueError('log_reg requires non-empty training data with binary labels')

    scores = np.empty((test_embeddings.shape[0], train_labels.shape[1]), dtype=np.float32)
    for column in range(train_labels.shape[1]):
        targets = train_labels[:, column]
        # Some labels have no positive training examples. Keep their columns
        # aligned and use the observed constant without inventing examples.
        if np.all(targets == targets[0]):
            scores[:, column] = targets[0]
            continue
        classifier = LogisticRegression(
            C=C, solver='liblinear', max_iter=max_iter,
            class_weight=class_weight, random_state=seed,
        )
        classifier.fit(train_embeddings, targets)
        for start in range(0, test_embeddings.shape[0], batch_size):
            end = min(start + batch_size, test_embeddings.shape[0])
            scores[start:end, column] = classifier.predict_proba(test_embeddings[start:end])[:, 1]
    return scores
