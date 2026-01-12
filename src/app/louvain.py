from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, List, Optional

import networkx as nx


@dataclass
class LouvainResult:
    communities: List[set]
    labels: Dict[Hashable, int]


class LouvainClustering:
    """
    Thin wrapper around networkx Louvain implementation.
    """

    def __init__(
        self,
        resolution: float = 1.0,
        threshold: float = 0.0000001,
        weight: str = "weight",
        seed: Optional[int] = None,
    ) -> None:
        self.resolution = resolution
        self.threshold = threshold
        self.weight = weight
        self.seed = seed
        self._result: Optional[LouvainResult] = None

    def fit(self, graph: nx.Graph) -> LouvainResult:
        communities = nx.community.louvain_communities(
            graph,
            weight=self.weight,
            resolution=self.resolution,
            threshold=self.threshold,
            seed=self.seed,
        )
        labels = self._labels_from_communities(communities)
        self._result = LouvainResult(communities=communities, labels=labels)
        return self._result

    def fit_from_edges(self, edges: Iterable[tuple]) -> LouvainResult:
        graph = nx.Graph()
        graph.add_edges_from(edges)
        return self.fit(graph)

    def labels(self) -> Dict[Hashable, int]:
        if self._result is None:
            raise ValueError("Call fit() before requesting labels.")
        return self._result.labels

    def communities(self) -> List[set]:
        if self._result is None:
            raise ValueError("Call fit() before requesting communities.")
        return self._result.communities

    def _labels_from_communities(self, communities: List[set]) -> Dict[Hashable, int]:
        labels: Dict[Hashable, int] = {}
        for idx, nodes in enumerate(communities):
            for node in nodes:
                labels[node] = idx
        return labels
