# SPDX-License-Identifier: Apache-2.0
"""The gate itself: text in, hold or send out.

    from privacy_gate.gate import Gate
    gate = Gate.load("model/head-v1.json")
    gate.decide("the woman from Tuesday's clinic has a 7mm lesion on her shoulder")
    # Decision(hold=True, score=6.72, threshold=0.1209)

Two things this deliberately does not do.

It does not replace `llm-harness/rules/v1.yaml`. The rules run first and keep
precedence, and this may only ever **add** a hold. A `send` from here is not an
assurance that the text is safe; it is the absence of a second opinion.

It does not pick its own threshold. The threshold is the catch-against-friction
dial, it is stored in the head file, and moving it is a policy decision with a
measured cost in `docs/JOURNAL.md`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from . import embed as embedding
from . import ollama
from .head import Model


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


class Gate:
    def __init__(
        self,
        model: Model,
        threshold: float,
        embedding_model: str = embedding.DEFAULT_MODEL,
        url: str = ollama.DEFAULT_URL,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.embedding_model = embedding_model
        self.url = url

    @staticmethod
    def load(path: str | Path, url: str = ollama.DEFAULT_URL) -> "Gate":
        raw = json.loads(Path(path).read_text())
        return Gate(
            model=Model.from_json(json.dumps(raw)),
            threshold=float(raw.get("threshold", 0.0)),
            embedding_model=str(raw.get("embedding_model", embedding.DEFAULT_MODEL)),
            url=url,
        )

    def decide_many(self, texts: list[str]) -> list[Decision]:
        """One embedding call for the batch, then arithmetic."""
        vectors = embedding.embed(texts, self.embedding_model, url=self.url)
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
