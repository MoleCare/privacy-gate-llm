# SPDX-License-Identifier: Apache-2.0
"""A logistic head over embeddings, and an honest way to score it.

Pure Python on purpose. The inference host has neither numpy nor scikit-learn, and the
problem is 127 examples by 1024 dimensions, which is nothing. Adding a
dependency to multiply a small matrix would buy a version conflict, not speed.

The whole point of this module is the *cross-validated* number. Fitting a
1024-parameter model to 127 examples and reporting its accuracy on those same
127 would produce a beautiful figure and mean nothing at all: with more
parameters than examples, a linear model can separate almost any labelling,
including a random one. `cross_validate` is therefore the only function here
whose output should ever be quoted, and `test_random_labels_score_near_chance`
in the test suite is what keeps that honest.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

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
    def from_json(text: str) -> "Model":
        raw = json.loads(text)
        return Model(
            weights=raw["weights"],
            bias=raw["bias"],
            mean=raw["mean"],
            stdev=raw["stdev"],
        )


def _sigmoid(x: float) -> float:
    # Split by sign so neither branch can overflow exp().
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _standardise(rows: list[Vector]) -> tuple[Vector, Vector]:
    n = len(rows)
    width = len(rows[0])
    mean = [sum(row[j] for row in rows) / n for j in range(width)]
    stdev = []
    for j in range(width):
        variance = sum((row[j] - mean[j]) ** 2 for row in rows) / n
        # A dimension that never varies carries no information; 1.0 makes it
        # contribute zero after centring instead of dividing by zero.
        stdev.append(math.sqrt(variance) or 1.0)
    return mean, stdev


def fit(
    rows: list[Vector],
    labels: list[int],
    *,
    l2: float = 1.0,
    epochs: int = 400,
    learning_rate: float = 0.5,
) -> Model:
    """Logistic regression by full-batch gradient descent with L2.

    L2 matters more than the optimiser here. With 1024 dimensions and ~100
    training rows the unregularised fit is degenerate, so the penalty is what
    makes the cross-validated number mean anything.
    """
    if not rows:
        raise ValueError("no rows to fit")
    if len(rows) != len(labels):
        raise ValueError(f"{len(rows)} rows but {len(labels)} labels")

    mean, stdev = _standardise(rows)
    scaled = [
        [(value - mu) / sigma for value, mu, sigma in zip(row, mean, stdev)] for row in rows
    ]
    n, width = len(scaled), len(scaled[0])
    weights = [0.0] * width
    bias = 0.0

    for _ in range(epochs):
        gradient = [0.0] * width
        bias_gradient = 0.0
        for row, label in zip(scaled, labels):
            total = bias + sum(w * v for w, v in zip(weights, row))
            error = _sigmoid(total) - label
            bias_gradient += error
            for j, value in enumerate(row):
                gradient[j] += error * value
        step = learning_rate / n
        bias -= step * bias_gradient
        for j in range(width):
            weights[j] -= step * (gradient[j] + l2 * weights[j])

    return Model(weights=weights, bias=bias, mean=mean, stdev=stdev)


def auc(scores: list[float], labels: list[int]) -> float:
    """Probability a random positive outranks a random negative; ties count half."""
    positives = [s for s, y in zip(scores, labels) if y == 1]
    negatives = [s for s, y in zip(scores, labels) if y == 0]
    if not positives or not negatives:
        return float("nan")
    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives
    )
    return wins / (len(positives) * len(negatives))


def folds(n: int, k: int, seed: int = 0) -> list[list[int]]:
    """A deterministic partition of range(n) into k folds. Every index appears
    exactly once, which `test_folds_partition_exactly` asserts."""
    if not 2 <= k <= n:
        raise ValueError(f"k must be between 2 and {n}, got {k}")
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return [order[i::k] for i in range(k)]


@dataclass
class CrossValidated:
    auc: float
    held_out_scores: dict[int, float]
    k: int


def cross_validate(
    rows: list[Vector], labels: list[int], *, k: int = 5, seed: int = 0, l2: float = 1.0
) -> CrossValidated:
    """Score every example from a model that never saw it.

    This is the only number from this module worth quoting.
    """
    held_out: dict[int, float] = {}
    for fold in folds(len(rows), k, seed):
        holdout = set(fold)
        train_rows = [r for i, r in enumerate(rows) if i not in holdout]
        train_labels = [y for i, y in enumerate(labels) if i not in holdout]
        if len(set(train_labels)) < 2:
            raise ValueError("a training fold has only one class; use fewer folds")
        model = fit(train_rows, train_labels, l2=l2)
        for i in fold:
            held_out[i] = model.score(rows[i])
    ordered = [held_out[i] for i in range(len(rows))]
    return CrossValidated(auc=auc(ordered, labels), held_out_scores=held_out, k=k)


def sweep(scores: list[float], labels: list[int]) -> list[dict]:
    """Catch and friction at every distinct threshold."""
    positives = sum(labels)
    negatives = len(labels) - positives
    cuts = sorted(set(scores))
    candidates = (
        [cuts[0] - 1.0] + [(a + b) / 2 for a, b in zip(cuts, cuts[1:])] + [cuts[-1] + 1.0]
    )
    points = []
    for threshold in candidates:
        caught = sum(1 for s, y in zip(scores, labels) if y == 1 and s > threshold)
        friction = sum(1 for s, y in zip(scores, labels) if y == 0 and s > threshold)
        points.append(
            {
                "threshold": round(threshold, 4),
                "catch_rate": round(caught / positives, 4) if positives else 0.0,
                "friction_rate": round(friction / negatives, 4) if negatives else 0.0,
                "leaks": positives - caught,
                "friction": friction,
            }
        )
    return points


def main(argv: list[str] | None = None) -> int:
    """Cross-validate a logistic head over cached embeddings."""
    from .labels import holds

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--embeddings", required=True, help="output of privacy_gate.embed")
    parser.add_argument("--rules", help="JSON map of example id to whether the rules fire")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--min-catch", type=float, default=0.98)
    parser.add_argument("--out")
    parser.add_argument(
        "--export",
        help="after reporting, refit on ALL examples and write the head here",
    )
    parser.add_argument("--embedding-model", default="bge-m3", help="Ollama tag")
    parser.add_argument("--embedding-model-hf", default="BAAI/bge-m3", help="Hugging Face id")
    args = parser.parse_args(argv)

    rows_in = [
        json.loads(line)
        for line in Path(args.embeddings).read_text().splitlines()
        if line.strip()
    ]
    vectors = [r["vector"] for r in rows_in]
    labels = [1 if holds(r["label"]) else 0 for r in rows_in]
    ids = [r["id"] for r in rows_in]

    result = cross_validate(vectors, labels, k=args.k, l2=args.l2)
    scores = [result.held_out_scores[i] for i in range(len(vectors))]
    points = sweep(scores, labels)

    rules = json.loads(Path(args.rules).read_text()) if args.rules else None
    if rules is not None:
        positives = sum(labels)
        negatives = len(labels) - positives
        for point in points:
            caught = sum(
                1
                for i, (s, y) in enumerate(zip(scores, labels))
                if y == 1 and (s > point["threshold"] or rules.get(ids[i], False))
            )
            friction = sum(
                1
                for i, (s, y) in enumerate(zip(scores, labels))
                if y == 0 and (s > point["threshold"] or rules.get(ids[i], False))
            )
            point["composed_catch_rate"] = round(caught / positives, 4)
            point["composed_friction_rate"] = round(friction / negatives, 4)
            point["composed_leaks"] = positives - caught

    ckey = "composed_catch_rate" if rules is not None else "catch_rate"
    fkey = "composed_friction_rate" if rules is not None else "friction_rate"
    eligible = [p for p in points if p[ckey] >= args.min_catch]
    chosen = min(eligible, key=lambda p: p[fkey]) if eligible else None

    print(f"### logistic head over embeddings, {args.k}-fold cross-validated\n")
    print(f"**AUC {result.auc:.4f}** over {len(vectors)} examples "
          f"({sum(labels)} sensitive). Every score comes from a model that never "
          f"saw that example.\n")
    if chosen:
        print(f"At catch >= {args.min_catch:.0%}: threshold {chosen['threshold']:+.3f}, "
              f"catch {chosen[ckey]:.1%}, **friction {chosen[fkey]:.1%}**, "
              f"leaks {chosen.get('composed_leaks', chosen['leaks'])}.\n")
    else:
        print(f"No threshold reaches catch >= {args.min_catch:.0%}.\n")

    print("| threshold | catch | friction |")
    print("|---|---|---|")
    seen: set[tuple] = set()
    for point in points:
        pair = (point[ckey], point[fkey])
        if pair in seen:
            continue
        seen.add(pair)
        print(f"| {point['threshold']:+.3f} | {point[ckey]:.1%} | {point[fkey]:.1%} |")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(
                {
                    "auc": result.auc,
                    "k": args.k,
                    "l2": args.l2,
                    "chosen": chosen,
                    "points": points,
                    "scores": dict(zip(ids, scores)),
                },
                indent=2,
            )
        )

    if args.export:
        # The shipped head sees every example. Its quality is the
        # cross-validated number above, which was measured without them; this
        # refit only uses the extra data, it does not re-measure anything.
        final = fit(vectors, labels, l2=args.l2)
        Path(args.export).parent.mkdir(parents=True, exist_ok=True)
        Path(args.export).write_text(
            final.to_json(
                embedding_model=args.embedding_model,
                # The Ollama tag and the Hugging Face id name the same weights
                # but are not interchangeable strings, and a consumer needs
                # whichever matches its runtime.
                embedding_model_hf=args.embedding_model_hf,
                # The head is fitted on L2-normalised vectors. Recorded because
                # an un-normalised input scores confidently wrong instead of
                # failing, and a consumer has no other way to know.
                embeddings_l2_normalised=True,
                dimensions=len(final.weights),
                trained_on=len(vectors),
                l2=args.l2,
                cross_validated_auc=round(result.auc, 4),
                threshold=chosen["threshold"] if chosen else 0.0,
                note=(
                    "score(text) > threshold means hold. Cross-validated AUC and "
                    "the operating point are recorded in docs/JOURNAL.md."
                ),
            )
            + "\n"
        )
        print(f"\nHead written to {args.export}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
