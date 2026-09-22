#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Do the embedding backends agree? The head was fitted on Ollama's /api/embed output. Another server, or
sentence-transformers with the Hugging Face weights, may embed the same text a little differently, and a
little difference near the threshold is a different verdict. This measures it instead of assuming it.

    PYTHONPATH=src python3 scripts/backends_agree.py \
        --reference ollama=http://127.0.0.1:11434 \
        --against openai=http://127.0.0.1:11500 --against local \
        --data data/gold.jsonl --out runs/backends-agree.json

Prints, per backend against the reference: the share of examples with the same verdict, the examples that
differ (id, label, both scores), the largest score difference, the cosine similarity of the vectors, and the
catch and friction each backend gets on its own. Standard library only; the `local` backend needs
`pip install 'privacy-gate[local]'`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from privacy_gate import backends  # noqa: E402
from privacy_gate.dataset import load  # noqa: E402
from privacy_gate.gate import Gate  # noqa: E402


def parse_backend(spec: str) -> tuple[str, str | None, str | None]:
    """'ollama=http://host:11434', 'openai=http://gw:11500', 'local', 'local=BAAI/bge-m3'."""
    name, _, rest = spec.partition("=")
    if name == "local":
        return name, None, rest or None
    return name, rest or None, None


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def scores_for(gate: Gate, texts: list[str], batch: int) -> tuple[list[list[float]], float]:
    vectors, started = [], time.time()
    for start in range(0, len(texts), batch):
        vectors += gate.embedder.embed(texts[start:start + batch])
    return vectors, time.time() - started


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", default="ollama=http://127.0.0.1:11434")
    parser.add_argument("--against", action="append", default=[], help="repeatable")
    parser.add_argument("--data", default="data/gold.jsonl")
    parser.add_argument("--head", default=None, help="default: the bundled head")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--ollama-cpu", action="store_true",
                        help="ollama backends: options.num_gpu = 0, so the run never evicts a chat model on a shared box")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    examples = load(args.data)
    texts = [e.text for e in examples]
    sensitive = [e.label != "CLEAN" for e in examples]

    def run(spec: str) -> dict:
        name, url, model = parse_backend(spec)
        embedder = backends.make_embedder(name, url, model)
        if name == "ollama" and args.ollama_cpu:
            embedder = backends.OllamaEmbedder(url or backends.ollama.DEFAULT_URL, model or backends.DEFAULT_MODEL,
                                               options={"num_gpu": 0})
        gate = Gate.load(args.head, embedder=embedder)
        vectors, seconds = scores_for(gate, texts, args.batch)
        decisions = [gate._decide_vector(v) for v in vectors]
        holds = [d.hold for d in decisions]
        catch = sum(1 for h, s in zip(holds, sensitive) if h and s) / max(1, sum(sensitive))
        friction = sum(1 for h, s in zip(holds, sensitive) if h and not s) / max(1, sum(1 for s in sensitive if not s))
        return {"spec": spec, "backend": name, "seconds": round(seconds, 1), "per_text_ms": round(1000 * seconds / len(texts), 1),
                "vectors": vectors, "scores": [d.score for d in decisions], "holds": holds,
                "catch": round(catch, 4), "friction": round(friction, 4), "threshold": gate.threshold}

    reference = run(args.reference)
    print(f"reference {reference['spec']}: catch {reference['catch']}, friction {reference['friction']}, "
          f"{reference['per_text_ms']} ms per text")
    report = {"data": args.data, "n": len(examples), "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "reference": {k: v for k, v in reference.items() if k != "vectors"}, "against": []}
    for spec in args.against:
        other = run(spec)
        same = sum(1 for a, b in zip(reference["holds"], other["holds"]) if a == b)
        differ = [{"id": e.id, "label": e.label, "reference_score": round(a, 4), "score": round(b, 4)}
                  for e, a, b, ha, hb in zip(examples, reference["scores"], other["scores"], reference["holds"], other["holds"]) if ha != hb]
        deltas = [abs(a - b) for a, b in zip(reference["scores"], other["scores"])]
        cosines = [cosine(a, b) for a, b in zip(reference["vectors"], other["vectors"])]
        summary = {**{k: v for k, v in other.items() if k != "vectors"},
                   "same_verdict": same, "same_verdict_share": round(same / len(examples), 4), "differ": differ,
                   "max_score_delta": round(max(deltas), 4), "median_score_delta": round(sorted(deltas)[len(deltas) // 2], 4),
                   "min_cosine": round(min(cosines), 5), "median_cosine": round(sorted(cosines)[len(cosines) // 2], 5)}
        report["against"].append(summary)
        print(f"{spec}: same verdict on {same} of {len(examples)} ({summary['same_verdict_share']:.1%}); "
              f"catch {other['catch']}, friction {other['friction']}; score delta median {summary['median_score_delta']} "
              f"max {summary['max_score_delta']}; cosine min {summary['min_cosine']} median {summary['median_cosine']}; "
              f"{other['per_text_ms']} ms per text")
        for d in differ:
            print(f"   differs: {d['id']} {d['label']}: reference {d['reference_score']} vs {d['score']}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
