import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.app.args.model import ModelArguments
from src.app.embedder import LlamaCppQwen3GgufEmbedder, TextEmbedder


class FakeLlama:
    init_kwargs = None

    @classmethod
    def from_pretrained(cls, **kwargs):
        cls.init_kwargs = kwargs
        return cls()

    def embed(self, text, normalize=False):
        assert normalize is False
        offset = 1 if text == "first story" else 2
        return [float(index + offset) for index in range(4096)]


@pytest.mark.parametrize(
    ("model_name_or_path", "repo_id", "filename"),
    [
        (
            "Qwen/Qwen3-Embedding-8B-GGUF/Qwen3-Embedding-8B-Q8_0.gguf",
            "Qwen/Qwen3-Embedding-8B-GGUF",
            "Qwen3-Embedding-8B-Q8_0.gguf",
        ),
        (
            "Qwen/Qwen3-Embedding-4B-GGUF/Qwen3-Embedding-4B-Q8_0.gguf",
            "Qwen/Qwen3-Embedding-4B-GGUF",
            "Qwen3-Embedding-4B-Q8_0.gguf",
        ),
    ],
)
@pytest.mark.parametrize("truncate_dim", [256, 128, 64])
def test_llama_cpp_qwen3_gguf_embedder_truncates_and_normalizes(
    monkeypatch, truncate_dim, model_name_or_path, repo_id, filename
):
    llama_cpp = SimpleNamespace(
        Llama=FakeLlama,
        LLAMA_POOLING_TYPE_LAST=42,
    )
    monkeypatch.setitem(sys.modules, "llama_cpp", llama_cpp)
    model_args = ModelArguments(
        model_name_or_path=model_name_or_path,
        max_seq_length=8192,
        truncate_dim=truncate_dim,
    )

    embedder = TextEmbedder.create(model_args)
    vectors = embedder.embed2np(["first story", "second story"])

    assert isinstance(embedder, LlamaCppQwen3GgufEmbedder)
    assert vectors.shape == (2, truncate_dim)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0])
    assert FakeLlama.init_kwargs == {
        "repo_id": repo_id,
        "filename": filename,
        "n_ctx": 8192,
        "n_gpu_layers": -1,
        "n_batch": 8192,
        "embedding": True,
        "pooling_type": 42,
        "verbose": False,
    }


def test_llama_cpp_qwen3_gguf_embedder_requires_dimension():
    model_args = ModelArguments(
        model_name_or_path=(
            "Qwen/Qwen3-Embedding-8B-GGUF/Qwen3-Embedding-8B-Q8_0.gguf"
        ),
    )

    with pytest.raises(ValueError, match="requires truncate_dim"):
        TextEmbedder.create(model_args)


def test_llama_cpp_qwen3_gguf_embedder_requires_hugging_face_file_path():
    model_args = ModelArguments(
        model_name_or_path="Qwen3-Embedding-8B-Q8_0.gguf",
        truncate_dim=64,
    )

    with pytest.raises(ValueError, match="owner/repository/model.gguf"):
        LlamaCppQwen3GgufEmbedder(model_args)
