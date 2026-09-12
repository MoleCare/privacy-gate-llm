#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""How long does the gate actually take?

This is a blocking question, not a curiosity. A gate that runs inline on every
request and adds a second to each one gets switched off, and a guard that is off
protects nothing — the same failure as too much friction, arrived at from the
other side.

What is measured is the whole decision: embed the text, standardise, dot,
compare. The arithmetic is 1024 multiply-adds and is far below the noise floor;
in practice this measures the embedding call and the network hop to it.

    PYTHONPATH=src python3 scripts/latency.py --head model/head-v0.json

Report the host's load alongside the numbers. On a shared box the same call can
differ by an order of magnitude, and a percentile without that context is not a
measurement, it is an anecdote.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from privacy_gate.dataset import load  # noqa: E402
from privacy_gate.gate import Gate  # noqa: E402

#: Sizes that matter, taken from the gold set rather than invented. A one-line
#: prompt and a whole support queue are both real inputs, and they do not cost
#: the same.
BUCKETS = [
    ("short", 0, 80),
    ("typical", 80, 200),
    ("long", 200, 10_000),
]


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile. No numpy, and n is small enough that the
    interpolation method would be a false precision."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q / 100 * len(ordered) + 0.5) - 1))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", default="model/head-v0.json")
    parser.add_argument("--data", default="data/gold.jsonl")
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--repeats", type=int, default=5, help="passes over each bucket")
    parser.add_argument("--batch", type=int, default=16, help="size for the batched figure")
    parser.add_argument("--out")
    args = parser.parse_args()

    gate = Gate.load(args.head, url=args.url)
    examples = load(args.data)

    try:
        load_avg = Path("/proc/loadavg").read_text().split()[0]
    except OSError:
        load_avg = "unknown"

    # The first call may have to evict another model and load this one, which is
    # a real cost but not a per-request one. Measured separately and excluded.
    started = time.monotonic()
    gate.decide("warm up the model so the first timed call is not a load")
    cold = time.monotonic() - started

    results: dict[str, dict] = {}
    for name, low, high in BUCKETS:
        texts = [e.text for e in examples if low <= len(e.text) < high]
        if not texts:
            continue
        timings: list[float] = []
        for _ in range(args.repeats):
            for text in texts:
                started = time.monotonic()
                gate.decide(text)
                timings.append((time.monotonic() - started) * 1000)
        results[name] = {
            "n": len(timings),
            "chars_median": statistics.median(len(t) for t in texts),
            "p50_ms": round(statistics.median(timings), 1),
            "p95_ms": round(percentile(timings, 95), 1),
            "max_ms": round(max(timings), 1),
        }

    # One embedding call for many texts is the shape most callers should use:
    # a gate on a batch of tool results, or a whole support queue at once.
    batch = [e.text for e in examples[: args.batch]]
    started = time.monotonic()
    gate.decide_many(batch)
    batched_ms = (time.monotonic() - started) * 1000

    report = {
        "host_load_1min": load_avg,
        "cold_start_seconds": round(cold, 2),
        "by_size": results,
        "batched": {
            "n": len(batch),
            "total_ms": round(batched_ms, 1),
            "per_text_ms": round(batched_ms / len(batch), 1),
        },
    }

    print(f"Host 1-minute load average: {load_avg}")
    print(f"Cold start (may include evicting another model): {cold:.2f}s\n")
    print("| text size | n | median chars | p50 | p95 | max |")
    print("|---|---|---|---|---|---|")
    for name, row in results.items():
        print(f"| {name} | {row['n']} | {row['chars_median']:.0f} | "
              f"{row['p50_ms']:.0f} ms | {row['p95_ms']:.0f} ms | {row['max_ms']:.0f} ms |")
    print(f"\nBatched: {len(batch)} texts in one call, "
          f"{batched_ms:.0f} ms total, {batched_ms / len(batch):.0f} ms each.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
