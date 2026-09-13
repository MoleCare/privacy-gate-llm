#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Record what the existing regex ruleset already catches.

`llm-harness explain` decides locally, so this is safe to run over a file of
fixtures that contains fake credentials.

It is run with `HARNESS_GATE=0` for two reasons. This measurement is of the
*patterns* and nothing else, and llm-harness now has an optional semantic step
that would otherwise make an embedding call per example — slower, and no longer
strictly true that nothing leaves the process. Pinning it off keeps the baseline
what it says it is.

It has to run on the machine where llm-harness is installed. The result is
committed as `data/rules-baseline.json`, so the evaluation on the server needs
no Node.

Usage:  python3 scripts/rules_oracle.py [--data data/gold.jsonl]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from privacy_gate.dataset import load  # noqa: E402


def privacy_verdict(text: str) -> tuple[bool, str | None]:
    """Run one text through the ruleset and read the privacy step out of the trace."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
        handle.write(text)
        path = handle.name
    try:
        done = subprocess.run(
            ["llm-harness", "explain", "--file", path],
            capture_output=True,
            text=True,
            timeout=60,
            # Measure the patterns alone. See the module docstring.
            env={**os.environ, "HARNESS_GATE": "0"},
        )
    finally:
        Path(path).unlink(missing_ok=True)
    if done.returncode != 0:
        raise RuntimeError(f"llm-harness failed: {done.stderr[:300]}")
    for line in done.stdout.splitlines():
        line = line.strip()
        if not line.startswith('{"step":"privacy"'):
            continue
        step = json.loads(line)
        return bool(step.get("matched")), step.get("rule") or step.get("id")
    raise RuntimeError("no privacy step in the trace; has the CLI output changed?")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/gold.jsonl")
    parser.add_argument("--out", default="data/rules-baseline.json")
    args = parser.parse_args()

    matched: dict[str, bool] = {}
    rule_of: dict[str, str] = {}
    for example in load(args.data):
        hit, rule = privacy_verdict(example.text)
        matched[example.id] = hit
        if hit and rule:
            rule_of[example.id] = rule

    Path(args.out).write_text(json.dumps(matched, indent=2, sort_keys=True) + "\n")
    Path(args.out).with_suffix(".rules.json").write_text(
        json.dumps(rule_of, indent=2, sort_keys=True) + "\n"
    )
    print(f"{sum(matched.values())}/{len(matched)} matched a privacy rule -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
