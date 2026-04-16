import logging

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Union, Dict, Callable

import numpy as np
from numpy import ndarray

logger = logging.getLogger('core.labeler')


class Labeler(ABC):

    valid_type_codes: List[str] = []

    def __init__(self, labels: Optional[List] = None, default_label: Any = None,
                 sorter: Optional[Callable[[Any], Any]] = None):
        self.default_label = default_label
        self.sorter = sorter
        self.computed = False
        self.classes: List[Any] = list(labels) if labels else []
        self.label2id: Dict[Any, int] = {}
        self.id2label: Dict[int, Any] = {}
        self.num_labels = len(self.classes)
        if self.classes:
            self.fit()

    def collect(self, labels: List[Any]):
        self.computed = False
        for x in labels:
            if isinstance(x, (list, set, tuple)):
                [self.classes.append(a) for a in x if a not in self.classes]
            elif x not in self.classes:
                self.classes.append(x)

    def fit(self):
        sorter = self.sorter or self._sort_key
        self.classes = sorted(set(self.classes), key=sorter)
        self.label2id = {label: idx for idx, label in enumerate(self.classes)}
        self.id2label = {idx: label for idx, label in enumerate(self.classes)}
        self.computed = True
        self.num_labels = len(self.classes)

    @staticmethod
    def _sort_key(label: Any) -> tuple[str, str]:
        return type(label).__name__, str(label)

    def _ensure_fitted(self) -> None:
        if not self.computed:
            self.fit()

    @abstractmethod
    def get_type_code(self):
        pass

    @abstractmethod
    def encode(self, value):
        pass

    @abstractmethod
    def decode(self, value):
        pass

    def labels2ids(self):
        self._ensure_fitted()
        return dict(self.label2id)

    def ids2labels(self):
        self._ensure_fitted()
        return dict(self.id2label)


class BinaryLabeler(Labeler):
    def __init__(self, labels: Optional[List] = None, default_label: Any = None,
                 sorter: Optional[Callable[[Any], Any]] = None):
        super().__init__(labels, default_label, sorter)

    def fit(self):
        super().fit()
        if self.num_labels != 2:
            raise ValueError('Invalid data was passed into Labeler collect. Must have at least two values for label!')
        self.num_labels = 1  # for binary classification we have a single label with two values
        logger.debug('Total number of labels: %s', self.num_labels)

    def get_type_code(self):
        return 'binary'

    def encode(self, label: Any) -> int:
        self._ensure_fitted()
        return self.label2id[label]

    def decode(self, label_id: int) -> Any:
        self._ensure_fitted()
        return self.classes[int(label_id)]


class MulticlassLabeler(Labeler):

    def __init__(self, labels: Optional[List] = None, default_label: Any = None,
                 sorter: Optional[Callable[[Any], Any]] = None):
        super().__init__(labels, default_label, sorter)

    def fit(self):
        super().fit()
        logger.debug('Total number of labels: %s', self.num_labels)

    def get_type_code(self):
        return 'multiclass'

    def encode(self, label: Any) -> int:
        self._ensure_fitted()
        return self.label2id[label]

    def decode(self, label_id: int) -> Any:
        self._ensure_fitted()
        return self.classes[int(label_id)]


class MultilabelLabeler(Labeler):

    def __init__(self, labels: Optional[List] = None, default_label: Any = None,
                 sorter: Optional[Callable[[Any], Any]] = None):
        super().__init__(labels, default_label, sorter)

    def fit(self):
        super().fit()
        logger.debug('Total number of labels: %s', self.num_labels)

    def get_type_code(self):
        return 'multilabel'

    def encode(self, labels: Union[List, ndarray]) -> ndarray:
        self._ensure_fitted()
        rows = []
        for label_set in labels:
            row = np.zeros(len(self.classes), dtype=np.int64)
            for label in label_set:
                row[self.label2id[label]] = 1
            rows.append(row)
        return np.asarray(rows, dtype=np.int64)

    def decode(self, vector: ndarray) -> List:
        self._ensure_fitted()
        values = np.asarray(vector)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        result = []
        for row in values:
            labels = tuple(
                self.classes[index]
                for index, value in enumerate(row)
                if int(value) != 0
            )
            result.append(labels)

        if len(result) == 1:  # Check if there's only one tuple in the list the return single list
            return list(result[0])
        return result


Labeler.valid_type_codes = [
    BinaryLabeler().get_type_code(),
    MultilabelLabeler().get_type_code(),
    MulticlassLabeler().get_type_code()
]
