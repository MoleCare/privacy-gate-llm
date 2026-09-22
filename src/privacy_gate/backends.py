# SPDX-License-Identifier: Apache-2.0
"""Where the embedding comes from. Three backends, one head.

    ollama   Ollama's /api/embed (the default; the head was fitted on its output)
    openai   any OpenAI-compatible /v1/embeddings: a gateway, vLLM, LM Studio, llama.cpp, a cloud service
    local    sentence-transformers in this process (`pip install 'privacy-gate[local]'`), CPU is enough

The first two are standard library only. Every backend must return unit vectors: the gate refuses a vector
that is not L2-normalised rather than score it, because an un-normalised vector produces a confidently wrong
answer, which is the worst failure a privacy gate can have.

Whether the three agree on the gold set is not assumed; `scripts/backends_agree.py` measures it, and the
result is in docs/JOURNAL.md.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from typing import Protocol

from . import ollama

DEFAULT_MODEL = "bge-m3"
DEFAULT_MODEL_HF = "BAAI/bge-m3"
DEFAULT_OPENAI_URL = "http://127.0.0.1:11434"  # Ollama serves /v1/embeddings too
BACKENDS = ("ollama", "openai", "local")


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbedder:
    name = "ollama"

    def __init__(
        self,
        url: str = ollama.DEFAULT_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 900.0,
        options: dict | None = None,
    ) -> None:
        """`options` go to Ollama unchanged. `{"num_gpu": 0}` keeps the encoder on the CPU, so that on a box
        with one GPU slot the gate never evicts the chat model somebody is using. `num_ctx` is never sent: it
        makes Ollama load the model under a second cache entry."""
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout
        self.options = {k: v for k, v in (options or {}).items() if k != "num_ctx"}

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload: dict = {"model": self.model, "input": texts, "keep_alive": "30m"}
        if self.options:
            payload["options"] = self.options
        body = ollama._post(self.url, "/api/embed", payload, self.timeout)
        vectors = body.get("embeddings")
        if not vectors or len(vectors) != len(texts):
            raise ollama.OllamaError(f"expected {len(texts)} embeddings, got {0 if not vectors else len(vectors)}")
        return vectors


class OpenAIEmbedder:
    """POST /v1/embeddings, the shape every OpenAI-compatible server speaks. The key, if any, comes from
    `api_key` or the environment (`PRIVACY_GATE_API_KEY`, then `OPENAI_API_KEY`); it is sent only as a header."""

    name = "openai"

    def __init__(
        self,
        url: str = DEFAULT_OPENAI_URL,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        timeout: float = 900.0,
    ) -> None:
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout
        self.api_key = api_key or os.environ.get("PRIVACY_GATE_API_KEY") or os.environ.get("OPENAI_API_KEY")

    def embed(self, texts: list[str]) -> list[list[float]]:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        payload = json.dumps({"model": self.model, "input": texts, "encoding_format": "float"}).encode()
        request = urllib.request.Request(self.url + "/v1/embeddings", payload, headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read()[:200].decode("utf-8", "replace")
            raise ollama.OllamaError(f"{self.url}/v1/embeddings: HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, OSError, ValueError) as error:
            raise ollama.OllamaError(f"{self.url}/v1/embeddings: {error}") from error
        rows = body.get("data")
        if not isinstance(rows, list) or len(rows) != len(texts):
            raise ollama.OllamaError(f"expected {len(texts)} embeddings, got {0 if not rows else len(rows)}")
        rows = sorted(rows, key=lambda row: int(row.get("index", 0)))
        return [normalise(row["embedding"]) for row in rows]


class LocalEmbedder:
    """sentence-transformers, in this process. The encoder loads once, on first use. `normalize_embeddings`
    is always on: without it the vectors are not unit length and the gate refuses them."""

    name = "local"

    def __init__(self, model: str = DEFAULT_MODEL_HF, device: str | None = None) -> None:
        self.model_id, self.device, self._model = model, device, None

    def _load(self):  # noqa: ANN202 - the type lives in an optional dependency
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:  # pragma: no cover - depends on the environment
                raise ollama.OllamaError(
                    "the local backend needs sentence-transformers: pip install 'privacy-gate[local]'"
                ) from error
            self._model = SentenceTransformer(self.model_id, device=self.device)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._load().encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return [[float(value) for value in vector] for vector in vectors]


def normalise(vector: list[float]) -> list[float]:
    """Unit length. OpenAI-compatible servers usually return unit vectors already; some do not, and the
    difference is invisible until a score is wrong."""
    norm = math.sqrt(sum(float(v) * float(v) for v in vector))
    if norm == 0.0:
        raise ollama.OllamaError("embedding is the zero vector")
    return [float(v) / norm for v in vector]


def make_embedder(
    backend: str = "ollama", url: str | None = None, model: str | None = None, api_key: str | None = None
) -> Embedder:
    if backend == "ollama":
        return OllamaEmbedder(url or ollama.DEFAULT_URL, model or DEFAULT_MODEL)
    if backend == "openai":
        return OpenAIEmbedder(url or DEFAULT_OPENAI_URL, model or DEFAULT_MODEL, api_key)
    if backend == "local":
        return LocalEmbedder(model or DEFAULT_MODEL_HF)
    raise ValueError(f"unknown backend {backend!r}; one of {', '.join(BACKENDS)}")
