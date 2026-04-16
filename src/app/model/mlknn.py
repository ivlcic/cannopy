from __future__ import annotations

import numpy as np

import torch


class MLKNN:
    """
    PyTorch implementation of ML-KNN from Zhang and Zhou (2007).

    The model estimates Laplace-smoothed priors for each label together with
    Laplace-smoothed conditional probabilities over the count of relevant
    labels among the k nearest neighbors.
    """

    InputArray = torch.Tensor | np.ndarray

    def __init__(self, n_neighbors: int = 10, smoothing: float = 1.0, metric: str = "euclidean", p: int = 2,
                 device: str | torch.device | None = None) -> None:
        self.n_neighbors = n_neighbors
        self.smoothing = smoothing
        self.metric = metric
        self.p = p
        self.device = torch.device(device) if device is not None else None
        self.x_train: torch.Tensor = torch.empty(0, 0, dtype=torch.float32, device=self.device)
        self.y_train: torch.Tensor = torch.empty(0, 0, dtype=torch.int64, device=self.device)
        self.prior_positive: torch.Tensor = torch.empty(0, dtype=torch.float32, device=self.device)
        self.prior_negative: torch.Tensor = torch.empty(0, dtype=torch.float32, device=self.device)
        self.posterior_positive: torch.Tensor = torch.empty(0, dtype=torch.float32, device=self.device)
        self.posterior_negative: torch.Tensor = torch.empty(0, dtype=torch.float32, device=self.device)
        self.label_count = 0
        self.is_fitted = False

    def _as_feature_tensor(self, value: InputArray) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if tensor.ndim != 2:
            raise ValueError("X must be a 2D feature matrix.")
        return tensor

    def _as_target_tensor(self, value: InputArray, n_samples: int) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=self.device).to(torch.int64)
        if tensor.ndim != 2:
            raise ValueError("y must be a 2D binary indicator matrix.")
        if tensor.shape[0] != n_samples:
            raise ValueError("X and y must contain the same number of samples.")
        if tensor.shape[1] == 0:
            raise ValueError("y must contain at least one label column.")
        if not torch.all((tensor == 0) | (tensor == 1)):
            raise ValueError("y must be a binary indicator matrix.")
        return tensor.to(torch.float32)

    def _validate_hyperparameters(self, n_samples: int) -> None:
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1.")
        if self.smoothing <= 0.0:
            raise ValueError("smoothing must be greater than 0.")
        if n_samples <= self.n_neighbors:
            raise ValueError("n_neighbors must be smaller than the number of training samples.")
        if self.metric == "minkowski" and self.p < 1:
            raise ValueError("p must be at least 1 for the Minkowski metric.")

    def _pairwise_distances(self, x_left: torch.Tensor, x_right: torch.Tensor) -> torch.Tensor:
        if self.metric == "euclidean":
            return torch.cdist(x_left, x_right, p=2)
        if self.metric == "manhattan":
            return torch.cdist(x_left, x_right, p=1)
        if self.metric == "minkowski":
            return torch.cdist(x_left, x_right, p=self.p)
        raise ValueError("metric must be one of: 'euclidean', 'manhattan', 'minkowski'.")

    def _training_neighbor_indices(self, x_train: torch.Tensor) -> torch.Tensor:
        distances = self._pairwise_distances(x_train, x_train)
        distances.fill_diagonal_(torch.inf)
        return torch.topk(distances, k=self.n_neighbors, largest=False, dim=1).indices

    def fit(self, x: InputArray, y: InputArray) -> "MLKNN":
        self.x_train = self._as_feature_tensor(x)
        self.y_train = self._as_target_tensor(y, n_samples=self.x_train.shape[0])
        self._validate_hyperparameters(n_samples=self.x_train.shape[0])
        self.label_count = self.y_train.shape[1]

        neighbor_indices = self._training_neighbor_indices(self.x_train)
        label_neighbor_counts = self.y_train[neighbor_indices].sum(dim=1).to(torch.long)

        positive_label_counts = self.y_train.sum(dim=0)
        self.prior_positive = (
            self.smoothing + positive_label_counts
        ) / (2.0 * self.smoothing + self.x_train.shape[0])
        self.prior_negative = 1.0 - self.prior_positive

        self.posterior_positive = torch.zeros(
            (self.label_count, self.n_neighbors + 1),
            dtype=torch.float32,
            device=self.x_train.device,
        )
        self.posterior_negative = torch.zeros_like(self.posterior_positive)

        for label_index in range(self.label_count):
            positive_mask = self.y_train[:, label_index] == 1
            negative_mask = ~positive_mask

            positive_histogram = torch.bincount(
                label_neighbor_counts[positive_mask, label_index],
                minlength=self.n_neighbors + 1,
            ).to(torch.float32)
            negative_histogram = torch.bincount(
                label_neighbor_counts[negative_mask, label_index],
                minlength=self.n_neighbors + 1,
            ).to(torch.float32)

            self.posterior_positive[label_index] = (
                self.smoothing + positive_histogram
            ) / (self.smoothing * (self.n_neighbors + 1) + positive_histogram.sum())
            self.posterior_negative[label_index] = (
                self.smoothing + negative_histogram
            ) / (self.smoothing * (self.n_neighbors + 1) + negative_histogram.sum())

        return self

    def _query_neighbor_indices(self, x_query: torch.Tensor) -> torch.Tensor:
        distances = self._pairwise_distances(x_query, self.x_train)
        return torch.topk(distances, k=self.n_neighbors, largest=False, dim=1).indices

    def predict_proba(self, x: InputArray) -> torch.Tensor:
        if not self.is_fitted:
            raise ValueError("Call fit() before predict() or predict_proba().")
        x_query = self._as_feature_tensor(x)

        neighbor_indices = self._query_neighbor_indices(x_query)
        neighbor_label_counts = self.y_train[neighbor_indices].sum(dim=1).to(torch.long)

        positive_posterior = torch.zeros(
            (x_query.shape[0], self.label_count),
            dtype=torch.float32,
            device=x_query.device,
        )
        negative_posterior = torch.zeros_like(positive_posterior)

        for label_index in range(self.label_count):
            positive_posterior[:, label_index] = (
                self.prior_positive[label_index]
                * self.posterior_positive[label_index, neighbor_label_counts[:, label_index]]
            )
            negative_posterior[:, label_index] = (
                self.prior_negative[label_index]
                * self.posterior_negative[label_index, neighbor_label_counts[:, label_index]]
            )

        normalization = positive_posterior + negative_posterior
        probabilities = torch.full_like(positive_posterior, 0.5)
        valid_mask = normalization > 0.0
        probabilities[valid_mask] = positive_posterior[valid_mask] / normalization[valid_mask]
        return probabilities

    def predict(self, x: InputArray) -> torch.Tensor:
        probabilities = self.predict_proba(x)
        return (probabilities >= 0.5).to(torch.int8)

