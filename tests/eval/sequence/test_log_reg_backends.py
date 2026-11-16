import sys
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse
from sklearn.linear_model import LogisticRegression

log_reg = import_module('src.eval.sequence.iptc_31.methods.log_reg')


def _data(as_sparse):
    train = np.asarray([[1, 0], [2, 0], [0, 1], [0, 2]], dtype=np.float32)
    query = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    labels = np.asarray([[1, 0, 1, 0], [1, 0, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0]], dtype=np.float32)
    if as_sparse:
        train, query = sparse.csr_matrix(train), sparse.csr_matrix(query)
    return train, labels, query


@pytest.mark.parametrize('as_sparse', [False, True])
def test_cuml_adapter_forwards_supported_options_and_preserves_constant_labels(monkeypatch, as_sparse):
    instances = []

    class FakeCumlClassifier:
        def __init__(self, **options):
            self.options = options
            self.cpu = LogisticRegression(C=options['C'], max_iter=options['max_iter'],
                                          class_weight=options['class_weight'], solver='lbfgs')
            instances.append(self)

        def fit(self, features, targets):
            assert features.dtype == np.float32
            assert targets.dtype == np.float32
            self.cpu.fit(features, targets)

        def predict_proba(self, features):
            return self.cpu.predict_proba(features)

    fake_cupy = SimpleNamespace(asarray=np.asarray, asnumpy=np.asarray, empty=np.empty, float32=np.float32)
    monkeypatch.setattr(log_reg, '_cuml_modules', lambda: (fake_cupy, sparse, FakeCumlClassifier))
    train, labels, query = _data(as_sparse)
    scores = log_reg.predict(train, labels, query, batch_size=2, C=2.0,
                             max_iter=300, class_weight='balanced', backend='cuml')
    assert type(scores) is np.ndarray
    assert scores.dtype == np.float32
    assert scores.shape == (3, 4)
    assert len(instances) == 2
    for classifier in instances:
        assert classifier.options == {
            'C': 2.0, 'max_iter': 300, 'class_weight': 'balanced', 'solver': 'qn', 'output_type': 'cupy',
        }
    assert scores[0, 0] > scores[1, 0]
    assert scores[1, 1] > scores[0, 1]
    np.testing.assert_array_equal(scores[:, 2], 1)
    np.testing.assert_array_equal(scores[:, 3], 0)


def test_missing_cuml_is_actionable_and_does_not_break_sklearn(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cupy', None)
    train, labels, query = _data(False)
    with pytest.raises(ImportError, match='uv sync --extra cuml'):
        log_reg.predict(train, labels, query, batch_size=2, backend='cuml')
    scores = log_reg.predict(train, labels, query, batch_size=2)
    assert scores.shape == (3, 4)


def test_invalid_backend_is_rejected():
    train, labels, query = _data(False)
    with pytest.raises(ValueError, match='log_reg.backend'):
        log_reg.predict(train, labels, query, batch_size=2, backend='unknown')


@pytest.mark.parametrize('as_sparse', [False, True])
def test_real_cuml_on_cuda(as_sparse):
    cp = pytest.importorskip('cupy')
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA device unavailable')
    if not device_count:
        pytest.skip('CUDA device unavailable')
    pytest.importorskip('cuml')
    train, labels, query = _data(as_sparse)
    scores = log_reg.predict(train, labels, query, batch_size=2, C=2.0, backend='cuml')
    assert type(scores) is np.ndarray
    assert scores.dtype == np.float32
    assert np.isfinite(scores).all() and ((0 <= scores) & (scores <= 1)).all()
    assert scores[0, 0] > scores[1, 0]
    assert scores[1, 1] > scores[0, 1]
    np.testing.assert_array_equal(scores[:, 2], 1)
    np.testing.assert_array_equal(scores[:, 3], 0)
