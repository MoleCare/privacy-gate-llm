# SPDX-License-Identifier: Apache-2.0
"""Loading and checking the labelled sets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .labels import LABELS


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    label: str
    slice: str


def load(path: str | Path) -> list[Example]:
    """Read a JSONL set, refusing anything malformed.

    A silently dropped row would change a denominator, and every number in
    `docs/JOURNAL.md` is a ratio, so this raises instead.
    """
    examples: list[Example] = []
    seen: set[str] = set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number} is not JSON: {error}") from error
        for field in ("id", "text", "label", "slice"):
            if field not in row:
                raise ValueError(f"{path}:{number} has no {field!r}")
        if row["label"] not in LABELS:
            raise ValueError(f"{path}:{number} label {row['label']!r} is not one of {LABELS}")
        if row["id"] in seen:
            raise ValueError(f"{path}:{number} repeats id {row['id']!r}")
        seen.add(row["id"])
        examples.append(
            Example(id=row["id"], text=row["text"], label=row["label"], slice=row["slice"])
        )
    if not examples:
        raise ValueError(f"{path} is empty")
    return examples
