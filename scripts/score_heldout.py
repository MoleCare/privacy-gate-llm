#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Score a held-out set with a shipped head, exactly as the gate would.

Held-out sets (`data/fresh.jsonl`, `data/short-probe.jsonl`) are never part of
the data a head is fitted on, so this is the honest check on a new head: it uses
the head file's own threshold and reports which rows it would hold.

    PYTHONPATH=src python3 scripts/score_heldout.py --head model/head-v1.json \
        --data data/short-probe.jsonl --out runs/probe-v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from privacy_gate.dataset import load  # noqa: E402
from privacy_gate.gate import Gate  # noqa: E402
from privacy_gate.labels import holds  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--out")
    args = parser.parse_args()

    gate = Gate.load(args.head, url=args.url)
    examples = load(args.data)
    decisions = gate.decide_many([e.text for e in examples])

    rows = []
    for e, d in zip(examples, decisions):
        rows.append({"id": e.id, "label": e.label, "slice": e.slice, "text": e.text,
                     "score": round(d.score, 4), "hold": d.hold})
    sensitive = [r for r in rows if holds(r["label"])]
    clean = [r for r in rows if not holds(r["label"])]
    caught = sum(r["hold"] for r in sensitive)
    friction = sum(r["hold"] for r in clean)

    print(f"{Path(args.head).name} on {Path(args.data).name} (threshold {gate.threshold:+.4f})")
    for r in sorted(rows, key=lambda r: -r["score"]):
        wrong = r["hold"] != holds(r["label"])
        print(f"  {'HOLD' if r['hold'] else 'send'} {r['score']:+7.2f}  {r['label']:6}  {r['text'][:60]}{'   <-- wrong' if wrong else ''}")
    print(f"catch {caught}/{len(sensitive)} | friction {friction}/{len(clean)}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "head": args.head, "data": args.data, "threshold": gate.threshold,
            "caught": caught, "sensitive": len(sensitive),
            "friction": friction, "clean": len(clean), "rows": rows,
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
