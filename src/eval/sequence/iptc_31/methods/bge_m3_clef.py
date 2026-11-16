"""Retrieve labeled training excerpts and fit them around the target article."""

from pathlib import Path

import faiss
import numpy as np

from ..features import _read_text_split


RAG_INSTRUCTION = (
    "The state contains the target article and labeled training examples. "
    "Decide about the target article only. Use examples to interpret topic "
    "boundaries; their labels need not apply to the target article."
)


def _embedding_rows(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f'RAG embeddings not found: {path}. Run ./data embed iptc-31 for the retrieval model and language.')
    # Test annotations in sidecars are deliberately not read for retrieval.
    with np.load(path, allow_pickle=False) as values:
        ids = np.asarray(values['ids'], dtype=str)
        vectors = np.asarray(values['embeddings'], dtype=np.float32)
    if ids.ndim != 1 or not len(ids) or len(set(ids)) != len(ids):
        raise ValueError(f'Expected non-empty unique embedding IDs in {path}')
    if vectors.ndim != 2 or vectors.shape[0] != len(ids) or not vectors.shape[1]:
        raise ValueError(f'Embedding dimensions do not match sample IDs in {path}')
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError(f'Expected finite, non-zero embedding vectors in {path}')
    return ids, np.ascontiguousarray(vectors / norms)


class BgeM3ClefRetriever:
    def __init__(self, prepare_dir: Path, embed_dir: Path, subset: str, labeler,
                 definitions: list[dict], retrieval_model: str = 'bge-m3', top_k: int = 3):
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError('bge-m3_clef.top_k must be a positive integer')
        if not isinstance(retrieval_model, str) or not retrieval_model.strip():
            raise ValueError('bge-m3_clef.retrieval_model must be a non-empty model name')
        self.top_k = top_k
        self.train_file = prepare_dir / f'{subset}.train.jsonl'
        self.train_sidecar = embed_dir / f'{subset}.{retrieval_model}.train.npz'
        self.test_sidecar = embed_dir / f'{subset}.{retrieval_model}.test.npz'
        train_ids, texts, labels = _read_text_split(self.train_file, labeler)
        ids, vectors = _embedding_rows(self.train_sidecar)
        self.query_ids, self.query_vectors = _embedding_rows(self.test_sidecar)
        if set(ids) != set(train_ids):
            raise ValueError(f'RAG prepared training IDs and embedding IDs differ for {subset}')
        if vectors.shape[1] != self.query_vectors.shape[1]:
            raise ValueError(f'RAG train/test embedding dimensions differ for {subset}')
        if set(ids) & set(self.query_ids):
            raise ValueError(f'RAG training and query IDs overlap for {subset}; held-out documents must not enter the index')
        prefix = subset.rsplit('.', 1)[-1] + ':'
        if any(not sample_id.startswith(prefix) for sample_id in [*ids, *self.query_ids]):
            raise ValueError(f'RAG requires same-language training and query documents for {subset}')
        names = {definition['iptc_code']: definition['label_name'] for definition in definitions}
        training_rows = {sample_id: row for row, sample_id in enumerate(train_ids)}
        self.examples = []
        for sample_id in ids:
            row = training_rows[sample_id]
            self.examples.append({
                'id': str(sample_id), 'text': texts[row],
                'topics': [{'code': labeler.classes[column], 'name': names[labeler.classes[column]]}
                           for column in np.flatnonzero(labels[row])],
            })
        self.num_train = len(ids)
        self.query_rows = {sample_id: row for row, sample_id in enumerate(self.query_ids)}
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def retrieve(self, query_ids: np.ndarray) -> list[list[dict]]:
        missing = [str(sample_id) for sample_id in query_ids if sample_id not in self.query_rows]
        if missing:
            raise ValueError(f'RAG query embeddings are missing IDs: {", ".join(missing[:5])}')
        vectors = self.query_vectors[[self.query_rows[sample_id] for sample_id in query_ids]]
        similarities, rows = self.index.search(vectors, min(self.top_k, self.num_train))
        return [[{**self.examples[int(row)], 'similarity': float(similarity)}
                 for row, similarity in zip(neighbors, scores, strict=True)]
                for neighbors, scores in zip(rows, similarities, strict=True)]


def encode_with_examples(api, tokenizer, sample_id: str, text: str, questions: dict,
                         examples: list[dict], *, max_length: int, max_state_tokens: int | None,
                         max_excerpt_tokens: int, schema_tokens: int):
    """Shrink or omit examples, preserving the target article's baseline budget."""
    budget = max_length - schema_tokens
    if max_state_tokens is not None:
        budget = min(budget, max_state_tokens)
    state = {'article': text}

    def token_count(candidate):
        return len(tokenizer(api.render(candidate), add_special_tokens=False).input_ids)

    def context(examples):
        return {'article': text, 'examples': examples, 'instructions': RAG_INSTRUCTION}

    target_truncated = token_count(state) > budget
    included, audit = [], []
    if not target_truncated:
        for example in examples:
            excerpt_ids = tokenizer(example['text'], add_special_tokens=False).input_ids[:max_excerpt_tokens]
            best, best_tokens = None, 0
            lower, upper = 1, len(excerpt_ids)
            while lower <= upper:
                # Most articles leave enough room for the full excerpt.
                count = upper if best is None and upper == len(excerpt_ids) else (lower + upper) // 2
                excerpt = tokenizer.decode(excerpt_ids[:count], skip_special_tokens=True)
                candidate = {'article_excerpt': excerpt, 'assigned_topics': example['topics']}
                if token_count(context([*included, candidate])) <= budget:
                    best, best_tokens, lower = candidate, count, count + 1
                else:
                    upper = count - 1
            if best is not None:
                included.append(best)
                audit.append({'id': example['id'], 'similarity': example['similarity'], 'excerpt_tokens': best_tokens})
        if included:
            state = context(included)
    encoded = api.encode_record(tokenizer, {'id': str(sample_id), 'state': state, 'questions': questions},
                                max_length=max_length, max_state_tokens=max_state_tokens)
    return encoded, {'query_id': str(sample_id), 'examples': audit,
                     'examples_dropped': len(examples) - len(included), 'target_truncated': target_truncated}
