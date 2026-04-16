import pytest


np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from src.app.model.mlknn import MLKNN


def build_separable_dataset():
    features = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
    labels = np.array(
        [
            [1, 0],
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
            [0, 1],
        ],
        dtype=np.int8,
    )
    return features, labels


def test_mlknn_fit_estimates_smoothed_priors_and_posteriors():
    features, labels = build_separable_dataset()
    classifier = MLKNN(n_neighbors=2, smoothing=1.0)

    classifier.fit(features, labels)

    expected_prior_positive = np.array([0.5, 0.5])
    expected_prior_negative = np.array([0.5, 0.5])
    expected_posterior_positive = np.array(
        [
            [1 / 6, 1 / 6, 4 / 6],
            [4 / 6, 1 / 6, 1 / 6],
        ]
    )
    expected_posterior_negative = np.array(
        [
            [4 / 6, 1 / 6, 1 / 6],
            [1 / 6, 1 / 6, 4 / 6],
        ]
    )

    np.testing.assert_allclose(classifier.prior_positive.cpu().numpy(), expected_prior_positive)
    np.testing.assert_allclose(classifier.prior_negative.cpu().numpy(), expected_prior_negative)
    np.testing.assert_allclose(classifier.posterior_positive_.cpu().numpy(), expected_posterior_positive)
    np.testing.assert_allclose(classifier.posterior_negative_.cpu().numpy(), expected_posterior_negative)


def test_mlknn_predict_prefers_labels_supported_by_neighbor_counts():
    features, labels = build_separable_dataset()
    classifier = MLKNN(n_neighbors=2, smoothing=1.0)
    query = np.array([[1.5], [10.5], [6.0]])

    classifier.fit(features, labels)
    probabilities = classifier.predict_proba(query)
    predictions = classifier.predict(query)

    expected_probabilities = np.array(
        [
            [0.8, 0.2],
            [0.2, 0.8],
            [0.2, 0.8],
        ]
    )
    expected_predictions = np.array(
        [
            [1, 0],
            [0, 1],
            [0, 1],
        ],
        dtype=np.int8,
    )

    np.testing.assert_allclose(probabilities.cpu().numpy(), expected_probabilities)
    np.testing.assert_array_equal(predictions.cpu().numpy(), expected_predictions)


def test_mlknn_rejects_non_binary_targets():
    features, _ = build_separable_dataset()
    labels = np.array([[0], [1], [2], [0], [1], [0]])
    classifier = MLKNN(n_neighbors=2)

    with pytest.raises(ValueError, match="binary indicator matrix"):
        classifier.fit(features, labels)
