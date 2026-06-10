from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Dict, List, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .args.model import ModelArguments
from .package import Package

logger = logging.getLogger('core.embedder')

Vector = List[float]
EmbeddingInput = Union[str, List[str]]


class EmbeddingMode(StrEnum):
    DOCUMENT = "document"
    QUERY = "query"


class TextEmbedder(ABC):
    _registry: Dict[str, "type[TextEmbedder]"] = {}
    valid_modes = {EmbeddingMode.DOCUMENT, EmbeddingMode.QUERY}

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
        if not name:
            raise ValueError(
                "model_name_or_path is empty. Check the loaded config files (e.g. -c jina-ebd-v3.yaml)."
            )
        key = name.lower()
        embedder_cls = cls._registry.get(key)
        if embedder_cls is None:
            raise ValueError(f"Unknown embedder model '{name}'. Available: {sorted(cls._registry)}")
        return embedder_cls(model_args)

    def __init__(self, model_args: ModelArguments) -> None:
        self.model_args = model_args
        self.mode = EmbeddingMode.DOCUMENT

    def set_mode(self, mode: str | EmbeddingMode) -> None:
        try:
            normalized_mode = EmbeddingMode(str(mode).strip().lower())
        except ValueError as exc:
            raise ValueError(f"Unsupported embedder mode '{mode}'. Expected one of {sorted(self.valid_modes)}")
        if normalized_mode not in self.valid_modes:
            raise ValueError(f"Unsupported embedder mode '{mode}'. Expected one of {sorted(self.valid_modes)}")
        self.mode = normalized_mode

    def embed_query2pt(self, texts: EmbeddingInput) -> torch.Tensor:
        previous_mode = self.mode
        self.set_mode(EmbeddingMode.QUERY)
        try:
            return self.embed2pt(texts)
        finally:
            self.mode = previous_mode

    def embed_query(self, texts: EmbeddingInput) -> Union[Vector, List[Vector]]:
        previous_mode = self.mode
        self.set_mode(EmbeddingMode.QUERY)
        try:
            return self.embed(texts)
        finally:
            self.mode = previous_mode

    def embed_query2np(self, texts: EmbeddingInput) -> np.ndarray:
        previous_mode = self.mode
        self.set_mode(EmbeddingMode.QUERY)
        try:
            return self.embed2np(texts)
        finally:
            self.mode = previous_mode

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
        self.truncate_dim = getattr(model_args, "truncate_dim", None)
        try:
            self.model = SentenceTransformer(
                model_name,
                device=self.device,
                trust_remote_code=True,
                model_kwargs=self._sentence_transformer_model_kwargs(),
            )
        except FileNotFoundError as exc:
            msg = (
                "Failed to load HF dynamic module; cache might be corrupted. "
                "Try removing ~/.cache/huggingface/modules/transformers_modules/jinaai "
                "and re-run the command."
            )
            raise FileNotFoundError(msg) from exc
        if model_args.max_seq_length:
            self.model.max_seq_length = model_args.max_seq_length
        self.tokenizer = self.model.tokenizer
        self.truncate = getattr(model_args, "truncate", False)
        self.task = None
        self.prompt_name = None
        logger.info('Loaded SentenceTransformer model=%s on %s', model_name, self.device)

    def _sentence_transformer_model_kwargs(self) -> Dict[str, object]:
        return {}

    def _encode_batch(self, batch: List[str], pt: bool) -> np.ndarray | torch.Tensor:
        encode_kwargs = {
            "batch_size": self.batch_size,
            "normalize_embeddings": True,
            "convert_to_numpy": not pt,
            "convert_to_tensor": pt,
            "task": self.task,
            "prompt_name": self.prompt_name,
            "device": self.device
        }
        if self.truncate_dim is not None:
            encode_kwargs["truncate_dim"] = self.truncate_dim
        # noinspection PyTypeChecker
        return self.model.encode(
            batch,
            **encode_kwargs
        )

    def _truncate_text(self, text: str) -> str:
        tok = self.tokenizer

        # Reserve space for special tokens the model will add
        special = tok.num_special_tokens_to_add(pair=False)
        budget = max(0, self.model.max_seq_length - special)

        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) <= budget:
            return text

        return tok.decode(ids[:budget], skip_special_tokens=True)

    def _ret_empty(self, batch: List[str], single: bool = False, pt: bool = True):
        dim = self.model.get_embedding_dimension()
        shape = (dim,) if single else (len(batch), dim)
        if pt:
            return torch.empty(shape, device="cpu")
        return np.empty(shape)

    def _embed(self, texts: EmbeddingInput, pt: bool = True) -> np.ndarray | torch.Tensor:
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        if not batch:
            return self._ret_empty(batch, single, pt)
        if self.truncate:
            batch = [self._truncate_text(b) for b in batch]
        try:
            vectors = self._encode_batch(batch, pt)
        except torch.OutOfMemoryError:
            logger.warning("Hitting memory problems")
            return self._ret_empty(batch, single, pt)
        if pt:
            # noinspection PyUnresolvedReferences
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

    def embed_query2pt(self, texts: EmbeddingInput) -> torch.Tensor:
        previous_mode = self.mode
        self.set_mode(EmbeddingMode.QUERY)
        try:
            return self.embed2pt(texts)
        finally:
            self.mode = previous_mode

    def embed_query(self, texts: EmbeddingInput) -> Union[Vector, List[Vector]]:
        previous_mode = self.mode
        self.set_mode(EmbeddingMode.QUERY)
        try:
            return self.embed(texts)
        finally:
            self.mode = previous_mode

    def embed_query2np(self, texts: EmbeddingInput) -> np.ndarray:
        previous_mode = self.mode
        self.set_mode(EmbeddingMode.QUERY)
        try:
            return self.embed2np(texts)
        finally:
            self.mode = previous_mode


# noinspection SpellCheckingInspection
@TextEmbedder.register("BAAI/bge-m3")
class BgeM3Embedder(STEmbedder):
    def __init__(self, model_args: ModelArguments) -> None:
        super().__init__(model_args)


@TextEmbedder.register("Qwen/Qwen3-Embedding-0.6B")
class Qwen3Embedder(STEmbedder):
    def __init__(self, model_args: ModelArguments) -> None:
        super().__init__(model_args)
        # self.truncate = True


@TextEmbedder.register("codefuse-ai/F2LLM-v2-0.6B")
class F2llmV2Embedder(STEmbedder):
    QUERY_PROMPT = "Instruct: Given a question, retrieve passages that can help answer the question.\nQuery: "

    def __init__(self, model_args: ModelArguments) -> None:
        super().__init__(model_args)

    def _sentence_transformer_model_kwargs(self) -> Dict[str, object]:
        if self.device == "cuda":
            return {"torch_dtype": "bfloat16"}
        return {}

    def _encode_batch(self, batch: List[str], pt: bool) -> np.ndarray | torch.Tensor:
        common_kwargs = {
            "batch_size": self.batch_size,
            "normalize_embeddings": True,
            "convert_to_numpy": not pt,
            "convert_to_tensor": pt,
            "device": self.device,
        }
        if self.truncate_dim is not None:
            common_kwargs["truncate_dim"] = self.truncate_dim
        if self.mode == EmbeddingMode.QUERY:
            encode_query = getattr(self.model, "encode_query", None)
            if callable(encode_query):
                # noinspection PyTypeChecker
                return encode_query(batch, **common_kwargs)
            # sentence-transformers 3.4.0 lacks encode_query; emulate the official prompt path.
            prompted_batch = [self.QUERY_PROMPT + text for text in batch]
            # noinspection PyTypeChecker
            return self.model.encode(prompted_batch, **common_kwargs)
        encode_document = getattr(self.model, "encode_document", None)
        if callable(encode_document):
            # noinspection PyTypeChecker
            return encode_document(batch, **common_kwargs)
        # Older sentence-transformers releases lack encode_document; plain encode is the document path.
        # noinspection PyTypeChecker
        return self.model.encode(batch, **common_kwargs)


@TextEmbedder.register("codefuse-ai/ML-Embed-0.6B")
class MlEmbedV06BEmbedder(F2llmV2Embedder):
    pass


@TextEmbedder.register("jinaai/jina-embeddings-v3")
class JinaV3Embedder(STEmbedder):
    def __init__(self, model_args: ModelArguments) -> None:
        Package.install_packages("einops", "0.8.2")
        super().__init__(model_args)


@TextEmbedder.register("Alibaba-NLP/gte-multilingual-base")
class GteMultilingualEmbedder(STEmbedder):
    def __init__(self, model_args: ModelArguments) -> None:
        super().__init__(model_args)


@TextEmbedder.register("Snowflake/snowflake-arctic-embed-l-v2.0")
class SnowflakeArcticEmbedder(STEmbedder):
    def __init__(self, model_args: ModelArguments) -> None:
        super().__init__(model_args)

    def _encode_batch(self, batch: List[str], pt: bool) -> np.ndarray | torch.Tensor:
        self.prompt_name = "query" if self.mode == EmbeddingMode.QUERY else None
        return super()._encode_batch(batch, pt)


@TextEmbedder.register("OpenAI/text-embedding-ada-002", "OpenAI/text-embedding-3-small")
class OpenaiTextEmbedder(TextEmbedder):
    def __init__(self, model_args: ModelArguments) -> None:
        super().__init__(model_args)
        # intentional inline install and import
        Package.install_packages("openai", "2.41.0")
        Package.install_packages("tiktoken", "0.13.0")
        # noinspection PyUnresolvedReferences,PyPackageRequirements
        from openai import OpenAI

        if not model_args.model_name_or_path:
            raise ValueError("OpenAI embedder requires model_name_or_path to be set.")
        self.client = OpenAI()
        self.model_name = model_args.model_name_or_path.replace("OpenAI/", '')
        self.max_seq_length = model_args.max_seq_length
        # noinspection PyUnresolvedReferences,PyPackageRequirements
        import tiktoken
        # noinspection PyUnresolvedReferences
        self.encoder = tiktoken.encoding_for_model(self.model_name)
        logger.info('Creating OpenAI client with model=%s', model_args.model_name_or_path)

    def _truncate_text(self, text: str) -> str:
        tokens = self.encoder.encode(text)
        if len(tokens) <= self.max_seq_length:
            return text
        return self.encoder.decode(tokens[:self.max_seq_length])

    def embed(self, texts: EmbeddingInput) -> Union[Vector, List[Vector]]:
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        if not batch:
            return [] if single else []
        batch = [self._truncate_text(b) for b in batch]
        response = self.client.embeddings.create(
            model=self.model_name,
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
