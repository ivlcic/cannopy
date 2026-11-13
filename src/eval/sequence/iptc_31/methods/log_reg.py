import numpy as np
from scipy.sparse import issparse, spmatrix
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression


def _cuml_modules():
    try:
        import cupy as cp
        from cupyx.scipy import sparse
        from cuml.linear_model import LogisticRegression
    except ImportError as exc:
        raise ImportError('log_reg backend=cuml requires RAPIDS; install with uv sync --extra cuml') from exc
    return cp, sparse, LogisticRegression


def prepare_cuml_features(features):
    """Upload once; inputs already on the device stay there between trials."""
    cp, sparse, _ = _cuml_modules()
    if issparse(features) or sparse.issparse(features):
        return sparse.csr_matrix(features, dtype=cp.float32)
    return cp.asarray(features, dtype=cp.float32, order='F')


def predict(train_embeddings: np.ndarray | spmatrix, train_labels: np.ndarray,
            test_embeddings: np.ndarray | spmatrix, *, batch_size: int,
            C: float = 1.0, max_iter: int = 1000,
            class_weight: str | None = None, seed: int = 2611,
            backend: str = 'sklearn') -> np.ndarray:
    """Fit an independent L2-regularized binary classifier for each label."""
    if not np.isfinite(C) or C <= 0:
        raise ValueError('log_reg.C must be finite and positive')
    if not isinstance(max_iter, (int, np.integer)) or isinstance(max_iter, bool) or max_iter <= 0:
        raise ValueError('log_reg.max_iter must be a positive integer')
    if class_weight not in (None, 'balanced'):
        raise ValueError("log_reg.class_weight must be null or 'balanced'")
    if train_embeddings.shape[0] == 0 or not np.isin(train_labels, [0, 1]).all():
        raise ValueError('log_reg requires non-empty training data with binary labels')
    if backend not in ('sklearn', 'cuml'):
        raise ValueError("log_reg.backend must be 'sklearn' or 'cuml'")

    if backend == 'cuml':
        cp, _, classifier_type = _cuml_modules()
        train_embeddings = prepare_cuml_features(train_embeddings)
        test_embeddings = prepare_cuml_features(test_embeddings)
        scores = cp.empty((test_embeddings.shape[0], train_labels.shape[1]), dtype=cp.float32)
        classifier_options = {'solver': 'qn', 'output_type': 'cupy'}
    else:
        classifier_type = SklearnLogisticRegression
        scores = np.empty((test_embeddings.shape[0], train_labels.shape[1]), dtype=np.float32)
        classifier_options = {'solver': 'liblinear', 'random_state': seed}

    for column in range(train_labels.shape[1]):
        targets = train_labels[:, column]
        # Some labels have no positive training examples. Keep their columns
        # aligned and use the observed constant without inventing examples.
        if np.all(targets == targets[0]):
            scores[:, column] = targets[0]
            continue
        classifier = classifier_type(
            C=C, max_iter=max_iter, class_weight=class_weight, **classifier_options,
        )
        classifier.fit(train_embeddings, cp.asarray(targets, dtype=cp.float32) if backend == 'cuml' else targets)
        for start in range(0, test_embeddings.shape[0], batch_size):
            end = min(start + batch_size, test_embeddings.shape[0])
            scores[start:end, column] = classifier.predict_proba(test_embeddings[start:end])[:, 1]
    return cp.asnumpy(scores) if backend == 'cuml' else scores
