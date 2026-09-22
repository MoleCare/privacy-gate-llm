# SPDX-License-Identifier: Apache-2.0
"""The gate itself: text in, hold or send out.

    from privacy_gate.gate import Gate
    gate = Gate.load()                       # the head shipped with this version, Ollama on 127.0.0.1:11434
    gate.decide("the woman from Tuesday's clinic has a 7mm lesion on her shoulder")
    # Decision(hold=True, score=6.72, threshold=0.1209)

    from privacy_gate.backends import make_embedder
    gate = Gate.load(embedder=make_embedder("openai", url="http://127.0.0.1:11500"))   # any /v1/embeddings
    gate = Gate.load(embedder=make_embedder("local"))                                   # sentence-transformers

Two things this deliberately does not do.

It does not replace `llm-harness/rules/v1.yaml`. The rules run first and keep
precedence, and this may only ever **add** a hold. A `send` from here is not an
assurance that the text is safe; it is the absence of a second opinion.

It does not pick its own threshold. The threshold is the catch-against-friction
dial, it is stored in the head file, and moving it is a policy decision with a
measured cost in `docs/JOURNAL.md`.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from . import backends, ollama
from .head import Model

BUNDLED_HEAD = "head-v1.json"


@dataclass(frozen=True)
class Decision:
    hold: bool
    score: float
    threshold: float

    @property
    def margin(self) -> float:
        """How far past the threshold. Near zero means the gate is unsure, which
        is worth logging: those are the examples worth adding to the gold set."""
        return self.score - self.threshold


def bundled_head_path() -> Path:
    """The head this version ships: inside the wheel, or model/ in a checkout run with PYTHONPATH=src."""
    packaged = resources.files("privacy_gate") / "models" / BUNDLED_HEAD
    try:
        if packaged.is_file():
            return Path(str(packaged))
    except (OSError, TypeError):  # pragma: no cover - an exotic importer
        pass
    checkout = Path(__file__).resolve().parents[2] / "model" / BUNDLED_HEAD
    if checkout.is_file():
        return checkout
    raise FileNotFoundError(f"no bundled head: neither the package's models/{BUNDLED_HEAD} nor {checkout}")


class Gate:
    def __init__(
        self,
        model: Model,
        threshold: float,
        embedding_model: str = backends.DEFAULT_MODEL,
        url: str = ollama.DEFAULT_URL,
        embedder: backends.Embedder | None = None,
        head_path: Path | None = None,
        head_sha256: str | None = None,
        meta: dict | None = None,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.embedding_model = embedding_model
        self.url = url
        self.embedder = embedder or backends.OllamaEmbedder(url, embedding_model)
        self.head_path = head_path
        self.head_sha256 = head_sha256
        self.meta = meta or {}

    @staticmethod
    def load(
        path: str | Path | None = None,
        url: str = ollama.DEFAULT_URL,
        embedder: backends.Embedder | None = None,
    ) -> Gate:
        """`path` None means the head shipped with this version. `url` keeps the old call shape (Ollama);
        `embedder` overrides it with any backend."""
        head_path = Path(path) if path is not None else bundled_head_path()
        text = head_path.read_text()
        raw = json.loads(text)
        embedding_model = str(raw.get("embedding_model", backends.DEFAULT_MODEL))
        if embedder is None:
            embedder = backends.OllamaEmbedder(url, embedding_model)
        elif isinstance(embedder, backends.LocalEmbedder) and embedder.model_id == backends.DEFAULT_MODEL_HF:
            embedder.model_id = str(raw.get("embedding_model_hf", backends.DEFAULT_MODEL_HF))
        return Gate(
            model=Model.from_json(json.dumps(raw)),
            threshold=float(raw.get("threshold", 0.0)),
            embedding_model=embedding_model,
            url=url,
            embedder=embedder,
            head_path=head_path,
            head_sha256=hashlib.sha256(text.encode()).hexdigest(),
            meta={k: v for k, v in raw.items() if k not in ("weights", "mean", "stdev")},
        )

    def decide_many(self, texts: list[str]) -> list[Decision]:
        """One embedding call for the batch, then arithmetic."""
        vectors = self.embedder.embed(texts)
        return [self._decide_vector(v) for v in vectors]

    def decide(self, text: str) -> Decision:
        return self.decide_many([text])[0]

    def _decide_vector(self, vector: list[float]) -> Decision:
        if len(vector) != len(self.model.weights):
            raise ValueError(
                f"embedding has {len(vector)} dimensions, head expects "
                f"{len(self.model.weights)}; wrong embedding model?"
            )
        norm = math.sqrt(sum(value * value for value in vector))
        if abs(norm - 1.0) > 0.01:
            # The head was fitted on Ollama's /api/embed output, which is
            # L2-normalised. sentence-transformers does NOT normalise by
            # default, and an un-normalised vector does not fail — it produces a
            # confidently wrong score. That is the worst possible failure for a
            # privacy gate, so it is refused rather than scored.
            raise ValueError(
                f"embedding is not L2-normalised (norm {norm:.4f}). The head was "
                "fitted on unit vectors. With sentence-transformers pass "
                "normalize_embeddings=True; Ollama already does this."
            )
        score = self.model.score(vector)
        return Decision(hold=score > self.threshold, score=score, threshold=self.threshold)
