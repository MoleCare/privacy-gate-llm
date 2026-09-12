# SPDX-License-Identifier: Apache-2.0
"""Score a model on a labelled set, and score it *in composition with the rules*.

Two numbers matter, and they pull against each other:

  leak      a sensitive text the gate let through. This is the incident.
  friction  an ordinary text the gate held back. This is what gets a guard
            switched off, which then causes the incident.

A third number is the reason this project exists at all: **residual catch**, the
share of sensitive texts that `llm-harness`'s regex ruleset misses and the model
finds. The model can only ever add a verdict, never clear one, so that figure is
the entire value on offer. Accuracy in isolation would flatter it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from . import ollama, prompt
from .dataset import Example, load
from .labels import BINARY, LABELS, holds, normalise


@dataclass
class Prediction:
    id: str
    slice: str
    gold: str
    raw: str
    predicted: str | None
    seconds: float


def classify(
    examples: list[Example],
    model: str,
    *,
    url: str,
    constrain: bool,
    binary: bool = False,
    progress: bool = True,
    checkpoint: Path | None = None,
) -> list[Prediction]:
    """Classify every example, writing each result as it arrives.

    The checkpoint is a plain JSONL of predictions. A run that is killed part
    way — which happens on a shared box — resumes from it instead of paying for
    the same tokens twice. Delete the file to force a fresh run.
    """
    vocabulary = BINARY if binary else LABELS
    build = prompt.build_binary if binary else prompt.build
    schema = {"type": "string", "enum": list(vocabulary)} if constrain else None

    done: dict[str, Prediction] = {}
    if checkpoint and checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["id"]] = Prediction(**row)
        if done and progress:
            print(f"  resuming, {len(done)} already done", file=sys.stderr, flush=True)

    out: list[Prediction] = []
    for index, example in enumerate(examples, 1):
        if example.id in done:
            out.append(done[example.id])
            continue
        reply = ollama.chat(model, build(example.text), url=url, schema=schema)
        result = Prediction(
            id=example.id,
            slice=example.slice,
            gold=example.label,
            raw=reply.content,
            predicted=normalise(reply.content, vocabulary),
            seconds=reply.seconds,
        )
        out.append(result)
        if checkpoint:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            with checkpoint.open("a") as handle:
                handle.write(json.dumps(asdict(result)) + "\n")
        if progress and index % 10 == 0:
            print(f"  {index}/{len(examples)}", file=sys.stderr, flush=True)
    return out


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def score(predictions: list[Prediction], rules: dict[str, bool] | None = None) -> dict:
    """Metrics for the model alone, and for rules-then-model as deployed.

    An unparseable answer counts as CLEAN, because that is what a gate that
    cannot read the answer has to do: it has no verdict to add. Counting it as a
    catch would score the model on its own failures.
    """
    sensitive = [p for p in predictions if holds(p.gold)]
    clean = [p for p in predictions if not holds(p.gold)]

    def model_holds(p: Prediction) -> bool:
        return p.predicted is not None and holds(p.predicted)

    leaks = [p for p in sensitive if not model_holds(p)]
    friction = [p for p in clean if model_holds(p)]
    caught = [p for p in sensitive if model_holds(p)]
    # In binary mode there is no category to be right about, so this is reported
    # as None rather than as a score of zero, which would read as a failure.
    binary_mode = any(p.predicted in BINARY for p in predictions if p.predicted)
    exact = None if binary_mode else [p for p in caught if p.predicted == p.gold]

    seconds = [p.seconds for p in predictions]
    result: dict = {
        "n": len(predictions),
        "sensitive": len(sensitive),
        "clean": len(clean),
        "unparseable": sum(1 for p in predictions if p.predicted is None),
        "model": {
            "catch_rate": round(_rate(len(caught), len(sensitive)), 4),
            "friction_rate": round(_rate(len(friction), len(clean)), 4),
            "leaks": len(leaks),
            "friction": len(friction),
            "label_accuracy_on_caught": (
                None if exact is None else round(_rate(len(exact), len(caught)), 4)
            ),
        },
        "latency_seconds": {
            "mean": round(statistics.mean(seconds), 3),
            "median": round(statistics.median(seconds), 3),
            "max": round(max(seconds), 3),
        },
        "leak_ids": [p.id for p in leaks],
        "friction_ids": [p.id for p in friction],
    }

    by_slice: dict[str, dict] = defaultdict(lambda: {"n": 0, "correct_gate": 0})
    for p in predictions:
        bucket = by_slice[p.slice]
        bucket["n"] += 1
        if model_holds(p) == holds(p.gold):
            bucket["correct_gate"] += 1
    result["by_slice"] = {
        name: {**data, "gate_accuracy": round(_rate(data["correct_gate"], data["n"]), 4)}
        for name, data in sorted(by_slice.items())
    }

    if rules is not None:
        missed_by_rules = [p for p in sensitive if not rules.get(p.id, False)]
        rescued = [p for p in missed_by_rules if model_holds(p)]
        rules_friction = [p for p in clean if rules.get(p.id, False)]
        composed_friction = [p for p in clean if rules.get(p.id, False) or model_holds(p)]
        composed_caught = [p for p in sensitive if rules.get(p.id, False) or model_holds(p)]
        result["rules"] = {
            "catch_rate": round(_rate(len(sensitive) - len(missed_by_rules), len(sensitive)), 4),
            "friction_rate": round(_rate(len(rules_friction), len(clean)), 4),
            "missed": len(missed_by_rules),
        }
        result["composed"] = {
            "catch_rate": round(_rate(len(composed_caught), len(sensitive)), 4),
            "friction_rate": round(_rate(len(composed_friction), len(clean)), 4),
            "leaks": len(sensitive) - len(composed_caught),
        }
        result["residual"] = {
            "missed_by_rules": len(missed_by_rules),
            "rescued_by_model": len(rescued),
            "residual_catch_rate": round(_rate(len(rescued), len(missed_by_rules)), 4),
            "still_leaking_ids": [p.id for p in missed_by_rules if not model_holds(p)],
        }
    return result


def render(model: str, result: dict) -> str:
    m = result["model"]
    lines = [
        f"### {model}",
        "",
        f"{result['n']} examples: {result['sensitive']} sensitive, {result['clean']} clean. "
        f"{result['unparseable']} unparseable.",
        "",
        "| | catch rate | friction rate | leaks |",
        "|---|---|---|---|",
        f"| model alone | {m['catch_rate']:.1%} | {m['friction_rate']:.1%} | {m['leaks']} |",
    ]
    if "rules" in result:
        r, c = result["rules"], result["composed"]
        lines += [
            f"| regex rules alone | {r['catch_rate']:.1%} | {r['friction_rate']:.1%} | {r['missed']} |",
            f"| **rules then model** | **{c['catch_rate']:.1%}** | {c['friction_rate']:.1%} | **{c['leaks']}** |",
        ]
        res = result["residual"]
        lines += [
            "",
            f"**Residual catch: {res['rescued_by_model']}/{res['missed_by_rules']} "
            f"({res['residual_catch_rate']:.1%})** of what the rules miss.",
        ]
    accuracy = m["label_accuracy_on_caught"]
    lines += [
        "",
        (
            "Binary gate, no category asked for. "
            if accuracy is None
            else f"Label accuracy on what it caught: {accuracy:.1%}. "
        )
        + f"Median {result['latency_seconds']['median']:.2f}s, max {result['latency_seconds']['max']:.2f}s.",
        "",
        "| slice | n | gate accuracy |",
        "|---|---|---|",
    ]
    for name, data in result["by_slice"].items():
        lines.append(f"| {name} | {data['n']} | {data['gate_accuracy']:.1%} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="data/gold.jsonl")
    parser.add_argument("--rules", help="JSON map of example id to whether the regex rules fire")
    parser.add_argument("--url", default=ollama.DEFAULT_URL)
    parser.add_argument("--out", help="write predictions and metrics here")
    parser.add_argument(
        "--free-text",
        action="store_true",
        help="do not constrain decoding to the labels, to measure format failures",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="ask only the gate question (KEEP or SEND), not the category",
    )
    args = parser.parse_args(argv)

    examples = load(args.data)
    rules = json.loads(Path(args.rules).read_text()) if args.rules else None
    predictions = classify(
        examples,
        args.model,
        url=args.url,
        constrain=not args.free_text,
        binary=args.binary,
        checkpoint=Path(f"{args.out}.partial.jsonl") if args.out else None,
    )
    result = score(predictions, rules)
    print(render(args.model, result))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(
                {
                    "model": args.model,
                    "constrained": not args.free_text,
                    "metrics": result,
                    "predictions": [asdict(p) for p in predictions],
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
