from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys

from abc import ABC, abstractmethod
from typing import Dict, List, Union

import numpy as np
import torch
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

    @abstractmethod
    def embed(self, texts: EmbeddingInput) -> Union[Vector, List[Vector]]:
        raise NotImplementedError

    @abstractmethod
    def embed2np(self, texts: EmbeddingInput) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def embed2pt(self, texts: EmbeddingInput) -> torch.Tensor:
        raise NotImplementedError


class STEmbedder(TextEmbedder):
    def __init__(self, model_args: ModelArguments) -> None:
        super().__init__(model_args)
        model_name = model_args.model_name_or_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = getattr(model_args, "batch_size", 32)
        self.model = SentenceTransformer(model_name, device=self.device, trust_remote_code=True)
        if model_args.max_seq_length:
            self.model.max_seq_length = model_args.max_seq_length
        logger.info('Loaded SentenceTransformer model=%s on %s', model_name, self.device)

    def _embed(self, texts: EmbeddingInput, pt: bool = True) -> np.ndarray | torch.Tensor:
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        if not batch:
            if pt:
                return torch.empty((0, 0), device='cpu')
            else:
                return np.empty((0, 0))
        with torch.inference_mode():
            vectors = self.model.encode(
                batch,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=not pt,
                convert_to_tensor=pt,
                device=self.device
            )
        if pt:
            vectors = vectors.detach().to("cpu")
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return vectors[0] if single else vectors

    def embed2pt(self, texts: EmbeddingInput) -> torch.Tensor:
        return self._embed(texts)

    def embed(self, texts: EmbeddingInput) -> Union[Vector, List[Vector]]:
        arr = self._embed(texts)
        return arr.tolist()

    def embed2np(self, texts: EmbeddingInput) -> np.ndarray:
        return self._embed(texts, pt=False)


@TextEmbedder.register("BAAI/bge-m3")
class BgeM3Embedder(STEmbedder):
    def __init__(self, model_args: ModelArguments) -> None:
        super().__init__(model_args)


@TextEmbedder.register("Qwen/Qwen3-Embedding-0.6B")
class Qwen3Embedder(STEmbedder):
    def __init__(self, model_args: ModelArguments) -> None:
        super().__init__(model_args)


@TextEmbedder.register("jinaai/jina-embeddings-v3")
class Qwen3Embedder(STEmbedder):
    def __init__(self, model_args: ModelArguments) -> None:
        super().__init__(model_args)


@TextEmbedder.register("Alibaba-NLP/gte-multilingual-base")
class Qwen3Embedder(STEmbedder):
    def __init__(self, model_args: ModelArguments) -> None:
        super().__init__(model_args)


@TextEmbedder.register("OpenAI/text-embedding-ada-002", "OpenAI/text-embedding-3-small")
class OpenaiTextEmbedder(TextEmbedder):
    def __init__(self, model_args: ModelArguments) -> None:
        super().__init__(model_args)
        # intentional inline install and import
        pkg = "openai"
        ver = "2.14.0"
        if importlib.util.find_spec(pkg) is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", f'{pkg}=={ver}'])
        from openai import OpenAI

        if not model_args.model_name_or_path:
            raise ValueError("OpenAI embedder requires model_name_or_path to be set.")
        self.client = OpenAI()
        logger.info('Creating OpenAI client with model=%s', model_args.model_name_or_path)

    def embed(self, texts: EmbeddingInput) -> Union[Vector, List[Vector]]:
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        if not batch:
            return [] if single else []
        response = self.client.embeddings.create(
            model=self.model_args.model_name_or_path,
            input=batch,
        )
        data = [item.embedding for item in response.data]
        return data[0] if single else data

    def embed2pt(self, texts: EmbeddingInput) -> torch.Tensor:
        out = self.embed(texts)
        single = isinstance(texts, str)
        data = [out] if single else out
        tensor = torch.tensor(data, dtype=torch.float32)
        return tensor[0] if single else tensor

    def embed2np(self, texts: EmbeddingInput) -> np.ndarray:
        out = self.embed(texts)
        single = isinstance(texts, str)
        data = [out] if single else out
        return np.array(data, dtype=np.float32 if data else float)
