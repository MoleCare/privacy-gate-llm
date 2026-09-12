# SPDX-License-Identifier: Apache-2.0
"""Embeddings, as the other way to build this gate.

Prompting a small chat model asks it to *reason* to a word. An embedding model
does one forward pass, emits a vector, and a linear head decides. For a gate that
must run inline on every request that is the better shape:

- no generation step at all, so latency is one forward pass rather than one pass
  plus sampling
- the head is a few thousand parameters and trains on a laptop in seconds
- the output is a calibrated probability, so the catch/friction trade-off is a
  dial rather than whatever the model felt like saying
- `bge-m3` is 1.16 GB, smaller than the 0.8B chat model that failed at this

`bge-m3` produces 1024 dimensions and handles long inputs, which matters because
the buried examples in the gold set are whole prompts, not sentences.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import ollama
from .dataset import load

DEFAULT_MODEL = "bge-m3"


def embed(texts: list[str], model: str = DEFAULT_MODEL, *, url: str = ollama.DEFAULT_URL,
          timeout: float = 900.0) -> list[list[float]]:
    """Embed a batch. Ollama's /api/embed takes a list and returns them in order."""
    body = ollama._post(
        url, "/api/embed", {"model": model, "input": texts, "keep_alive": "30m"}, timeout
    )
    vectors = body.get("embeddings")
    if not vectors or len(vectors) != len(texts):
        raise ollama.OllamaError(
            f"expected {len(texts)} embeddings, got {0 if not vectors else len(vectors)}"
        )
    return vectors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Embed a labelled set and cache the vectors.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default="data/gold.jsonl")
    parser.add_argument("--url", default=ollama.DEFAULT_URL)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args(argv)

    examples = load(args.data)
    out = Path(args.out)

    # Cache per example id, so a run interrupted by a busy box resumes for free.
    cached: dict[str, list[float]] = {}
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                cached[row["id"]] = row["vector"]
        print(f"  resuming, {len(cached)} already embedded", file=sys.stderr)

    todo = [e for e in examples if e.id not in cached]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as handle:
        for start in range(0, len(todo), args.batch):
            chunk = todo[start : start + args.batch]
            vectors = embed([e.text for e in chunk], args.model, url=args.url)
            for example, vector in zip(chunk, vectors):
                handle.write(
                    json.dumps(
                        {
                            "id": example.id,
                            "label": example.label,
                            "slice": example.slice,
                            "vector": vector,
                        }
                    )
                    + "\n"
                )
            handle.flush()
            print(f"  {min(start + args.batch, len(todo))}/{len(todo)}", file=sys.stderr)

    print(f"{len(examples)} embeddings in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
