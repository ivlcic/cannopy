import numpy as np
from scipy.sparse import issparse, spmatrix
from sklearn.metrics.pairwise import linear_kernel
from sklearn.svm import SVC


def predict(train_embeddings: np.ndarray | spmatrix, train_labels: np.ndarray,
            test_embeddings: np.ndarray | spmatrix, *, batch_size: int,
            C: float = 1.0, max_iter: int = -1,
            class_weight: str | None = None, seed: int = 2611) -> np.ndarray:
    """Fit independent linear SVMs with training-only Platt calibration."""
    if not np.isfinite(C) or C <= 0:
        raise ValueError('svm.C must be finite and positive')
    if not isinstance(max_iter, (int, np.integer)) or isinstance(max_iter, bool) or (max_iter != -1 and max_iter <= 0):
        raise ValueError('svm.max_iter must be -1 or a positive integer')
    if class_weight not in (None, 'balanced'):
        raise ValueError("svm.class_weight must be null or 'balanced'")
    if train_embeddings.shape[0] == 0 or not np.isin(train_labels, [0, 1]).all():
        raise ValueError('svm requires non-empty training data with binary labels')

    # Reuse the linear kernel across labels and calibration folds, rather
    # than recomputing high-dimensional dot products for every binary SVM.
    train = train_embeddings.astype(np.float64) if issparse(train_embeddings) else np.asarray(train_embeddings, dtype=np.float64)
    train_kernel = np.ascontiguousarray(linear_kernel(train))
    scores = np.empty((test_embeddings.shape[0], train_labels.shape[1]), dtype=np.float32)
    classifiers = []
    for column in range(train_labels.shape[1]):
        targets = train_labels[:, column]
        if np.all(targets == targets[0]):
            scores[:, column] = targets[0]
            continue
        classifier = SVC(
            C=C, kernel='precomputed', probability=True, max_iter=max_iter,
            class_weight=class_weight, random_state=seed,
        )
        classifier.fit(train_kernel, targets)
        classifiers.append((column, classifier))

    if classifiers:
        for start in range(0, test_embeddings.shape[0], batch_size):
            end = min(start + batch_size, test_embeddings.shape[0])
            query_kernel = np.ascontiguousarray(linear_kernel(test_embeddings[start:end], train))
            for column, classifier in classifiers:
                scores[start:end, column] = classifier.predict_proba(query_kernel)[:, 1]
    return scores
