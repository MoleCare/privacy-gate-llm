#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The no-model baseline, and the gold set's length audit.

Needs nothing but Python. Run it before quoting any neural number:

    PYTHONPATH=src python3 scripts/lexical_baseline.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from privacy_gate import head  # noqa: E402
from privacy_gate.dataset import load  # noqa: E402
from privacy_gate.features import length_report, vectorise  # noqa: E402
from privacy_gate.labels import holds  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/gold.jsonl")
    parser.add_argument("--rules", default="data/rules-baseline.json")
    parser.add_argument("--dimensions", type=int, default=512)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out")
    args = parser.parse_args()

    examples = load(args.data)
    labels = [1 if holds(e.label) else 0 for e in examples]

    print("## Gold set length audit\n")
    lengths = length_report(
        {
            "sensitive": [e.text for e in examples if holds(e.label)],
            "clean": [e.text for e in examples if not holds(e.label)],
        }
    )
    print("| class | n | chars (median) | chars (mean) | tokens (median) | tokens (mean) |")
    print("|---|---|---|---|---|---|")
    for name, row in lengths.items():
        print(
            f"| {name} | {row['n']:.0f} | {row['chars_median']:.0f} | {row['chars_mean']} "
            f"| {row['tokens_median']:.0f} | {row['tokens_mean']} |"
        )

    print("\n## Hashed unigrams, logistic head, cross-validated\n")
    vectors = [vectorise(e.text, args.dimensions) for e in examples]
    result = head.cross_validate(vectors, labels, k=args.k)
    scores = [result.held_out_scores[i] for i in range(len(vectors))]
    points = head.sweep(scores, labels)

    rules = json.loads(Path(args.rules).read_text()) if args.rules else None
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if rules:
        for point in points:
            caught = sum(
                1
                for e, s, y in zip(examples, scores, labels)
                if y == 1 and (s > point["threshold"] or rules.get(e.id, False))
            )
            friction = sum(
                1
                for e, s, y in zip(examples, scores, labels)
                if y == 0 and (s > point["threshold"] or rules.get(e.id, False))
            )
            point["composed_catch_rate"] = round(caught / positives, 4)
            point["composed_friction_rate"] = round(friction / negatives, 4)

    ckey = "composed_catch_rate" if rules else "catch_rate"
    fkey = "composed_friction_rate" if rules else "friction_rate"
    print(f"**AUC {result.auc:.4f}** ({args.dimensions} dimensions, {args.k}-fold).\n")
    for target in (0.98, 0.95, 0.90):
        eligible = [p for p in points if p[ckey] >= target]
        if eligible:
            best = min(eligible, key=lambda p: p[fkey])
            print(f"- catch >= {target:.0%}: friction {best[fkey]:.1%} "
                  f"(threshold {best['threshold']:+.3f})")

    print("\n### By slice, at the catch >= 98% operating point\n")
    eligible = [p for p in points if p[ckey] >= 0.98]
    if eligible:
        cut = min(eligible, key=lambda p: p[fkey])["threshold"]
        by_slice: dict[str, list[int]] = {}
        for example, score, label in zip(examples, scores, labels):
            correct = (score > cut) == bool(label)
            by_slice.setdefault(example.slice, []).append(1 if correct else 0)
        print("| slice | n | gate accuracy |")
        print("|---|---|---|")
        for name in sorted(by_slice):
            hits = by_slice[name]
            print(f"| {name} | {len(hits)} | {sum(hits) / len(hits):.1%} |")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(
                {"auc": result.auc, "lengths": lengths, "points": points}, indent=2
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
