from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys

from abc import ABC, abstractmethod
from typing import Dict, List, Sequence, Union
from sentence_transformers import SentenceTransformer

from .args.model import ModelArguments

logger = logging.getLogger('core.embedder')

Vector = List[float]
EmbeddingInput = Union[str, List[str]]


class TextEmbedder(ABC):
    _registry: Dict[str, "type[TextEmbedder]"] = {}

    @classmethod
    def register(cls, *names: str):
        keys = [n.strip().lower() for n in names if n and n.strip()]

        def decorator(subclass: "type[TextEmbedder]") -> "type[TextEmbedder]":
            for key in keys:
                cls._registry[key] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, model_args: ModelArguments) -> "TextEmbedder":
        name = model_args.model_name_or_path.strip()
        key = name.lower()
        embedder_cls = cls._registry.get(key)
        if embedder_cls is None:
            raise ValueError(f"Unknown embedder model '{name}'. Available: {sorted(cls._registry)}")
        return embedder_cls(model_args)

    def __init__(self, model_args: ModelArguments) -> None:
        self.model_args = model_args

    def embed(self, texts: EmbeddingInput) -> Union[Vector, List[Vector]]:
        if isinstance(texts, str):
            vectors = self._embed_many([texts])
            return vectors[0] if vectors else []
        vectors = self._embed_many(texts)
        return vectors

    @abstractmethod
    def _embed_many(self, texts: Sequence[str]) -> List[Vector]:
        raise NotImplementedError


@TextEmbedder.register("BAAI/bge-m3")
class BgeM3Embedder(TextEmbedder):
    def __init__(self, model_args: ModelArguments) -> None:
        super().__init__(model_args)
        model_name = model_args.model_name_or_path or "BAAI/bge-m3"
        self.model = SentenceTransformer(model_name)
        if model_args.max_seq_length:
            self.model.max_seq_length = model_args.max_seq_length
        logger.info('Loaded SentenceTransformer model=%s', model_name)

    def _embed_many(self, texts: Sequence[str]) -> List[Vector]:
        if not texts:
            return []
        vectors = self.model.encode(list(texts), normalize_embeddings=True)
        return [v.tolist() for v in vectors]


@TextEmbedder.register("OpenAI/text-embedding-ada-002", "OpenAI/text-embedding-3-small")
class OpenaiTextEmbedder(TextEmbedder):
    def __init__(self, model_args: ModelArguments) -> None:
        super().__init__(model_args)
        pkg = "openai"
        ver = "2.14.0"
        if importlib.util.find_spec(pkg) is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", f'{pkg}=={ver}'])
        from openai import OpenAI

        if not model_args.model_name_or_path:
            raise ValueError("OpenAI embedder requires model_name_or_path to be set.")
        self.client = OpenAI()
        logger.info('Creating OpenAI client with model=%s', model_args.model_name_or_path)

    def _embed_many(self, texts: Sequence[str]) -> List[Vector]:
        if not texts:
            return []
        response = self.client.embeddings.create(
            model=self.model_args.model_name_or_path,
            input=list(texts),
        )
        return [item.embedding for item in response.data]
