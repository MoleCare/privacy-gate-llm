# SPDX-License-Identifier: Apache-2.0
"""Write test/differential.json: random texts and the spans the Python engine gives for every rule.

test/differential.test.mjs then checks that the JavaScript engine gives the same spans. The texts are the
conformance fixtures, spliced together and seasoned with the characters where Python's `re.ASCII` and a
JavaScript RegExp could disagree (no-break space, long s, Kelvin sign, astral characters, line separators).

    PYTHONPATH=src python js/scripts/differential.py > js/test/differential.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from privacy_gate.guard.rules import Ruleset

ROOT = Path(__file__).resolve().parents[2]
EXTRA = [
    "\u00a0",
    "\u017f",
    "\u212a",
    "\U0001f600",
    "\n",
    "\r",
    "\u2028",
    "\u00e9",
    "\u2168",
    "  ",
    "-",
    "0",
    "@",
]


def main() -> None:
    rules = Ruleset.load()
    texts = [c["text"] for c in json.loads((ROOT / "rules" / "conformance.v2.json").read_text())["cases"]]
    rng = random.Random(5)
    cases = []
    for _ in range(3000):
        text = rng.choice(texts)
        for _ in range(rng.randint(0, 4)):
            at = rng.randint(0, len(text))
            text = text[:at] + rng.choice([*EXTRA, rng.choice(texts)[:20]]) + text[at:]
        if rng.random() < 0.3:
            text = text.upper()
        elif rng.random() < 0.3:
            text = text.lower()
        cases.append({"text": text, "spans": {r.id: [list(s) for s in r.spans(text)] for r in rules.rules}})
    json.dump({"ruleset": rules.sha256, "cases": cases}, sys.stdout)


if __name__ == "__main__":
    main()
