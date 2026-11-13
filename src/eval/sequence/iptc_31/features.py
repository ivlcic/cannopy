import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ....app.labeler import MultilabelLabeler


def _read_text_split(path: Path, labeler: MultilabelLabeler):
    ids, texts, labels = [], [], []
    with path.open(encoding='utf-8') as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
                sample_id = str(sample['id'])
                text = sample['text']
                if not sample_id or not isinstance(text, str):
                    raise ValueError('Expected a sample ID and string text')
                encoded = labeler.encode([sample['label']])[0]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'Invalid prepared record in {path} line {line_no}: {exc}') from exc
            ids.append(sample_id)
            texts.append(text)
            labels.append(encoded)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f'Expected non-empty unique sample IDs in {path}')
    return np.asarray(ids, dtype=str), texts, np.asarray(labels, dtype=np.float32)


def vectorize_texts(train_texts: list[str], query_texts: list[str], options: dict):
    parameters = {key: value for key, value in options.items() if key != 'type'}
    if 'ngram_range' in parameters:
        parameters['ngram_range'] = tuple(parameters['ngram_range'])
    vectorizer = TfidfVectorizer(**parameters)
    # Vocabulary, IDF and feature selection are learned exclusively on train.
    train_features = vectorizer.fit_transform(train_texts).tocsr()
    query_features = vectorizer.transform(query_texts).tocsr()
    return train_features, query_features, vectorizer


def load_tfidf_splits(train_file: Path, test_file: Path, labeler: MultilabelLabeler, options: dict):
    train_ids, train_texts, train_labels = _read_text_split(train_file, labeler)
    test_ids, test_texts, test_labels = _read_text_split(test_file, labeler)
    train_features, test_features, vectorizer = vectorize_texts(train_texts, test_texts, options)
    return (
        (train_ids, train_features, train_labels),
        (test_ids, test_features, test_labels),
        vectorizer,
    )
