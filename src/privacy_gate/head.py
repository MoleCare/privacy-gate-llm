# SPDX-License-Identifier: Apache-2.0
"""The logistic head the gate scores with: weights, bias and the training statistics.

This is the inference half only. Fitting, cross-validation and the threshold sweep are maintainer tools
and live in `training/head.py` in the repository, outside the wheel.

Pure Python on purpose. The inference host has neither numpy nor scikit-learn, and the
problem is 127 examples by 1024 dimensions, which is nothing. Adding a
dependency to multiply a small matrix would buy a version conflict, not speed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

Vector = list[float]


@dataclass
class Model:
    weights: Vector
    bias: float
    #: Per-dimension mean and standard deviation of the training fold. Held with
    #: the model because scoring must standardise with the *training* statistics;
    #: recomputing them on the evaluation fold leaks it into the fit.
    mean: Vector
    stdev: Vector

    def score(self, vector: Vector) -> float:
        """Log-odds that this text must be held back."""
        total = self.bias
        for value, weight, mu, sigma in zip(vector, self.weights, self.mean, self.stdev):
            total += weight * (value - mu) / sigma
        return total

    def to_json(self, **extra: object) -> str:
        return json.dumps(
            {
                "weights": self.weights,
                "bias": self.bias,
                "mean": self.mean,
                "stdev": self.stdev,
                **extra,
            }
        )

    @staticmethod
    def from_json(text: str) -> Model:
        raw = json.loads(text)
        return Model(
            weights=raw["weights"],
            bias=raw["bias"],
            mean=raw["mean"],
            stdev=raw["stdev"],
        )
