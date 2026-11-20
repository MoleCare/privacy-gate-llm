# SPDX-License-Identifier: Apache-2.0
"""Detection: run the layers over one text and report findings by span, never by value.

L1 (the rules) reads the text itself and, when L0 is on, every view from `normalize.views`. A finding from a
view is mapped back to the span of the original text it came from and marked ``encoded`` when it came from a
decoded view, because then its span is the whole encoded run rather than the value.

L2 and L3 (an entity model, a prose classifier) are callables passed in by the caller; they arrive in later
phases. Each returns ``(category, start, end)`` triples over the original text.

Overlaps: block findings are all kept, since each is a reason to refuse. A mask finding overlapping a block
finding is dropped (the block wins); of two overlapping mask findings the longer is kept, then the earlier.
What is left for masking is sorted and disjoint.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .normalize import views
from .rules import Ruleset

#: An L2 or L3 detector: text in, (category, start, end) out.
Detector = Callable[[str], Iterable[tuple[str, int, int]]]


@dataclass(frozen=True)
class Finding:
    category: str
    rule_id: str
    aliases: tuple[str, ...]
    start: int
    end: int
    layer: str
    #: Found inside a decoded view; the span is the whole encoded run.
    encoded: bool = False


def detect(text: str, ruleset: Ruleset, layers: Iterable[str] = ("L0", "L1"),
           detectors: dict[str, Detector] | None = None,
           timings: dict[str, float] | None = None) -> list[Finding]:
    """Every finding in `text`, overlaps resolved. Detector exceptions propagate: the caller holds the text."""
    layers = tuple(layers)
    found: list[Finding] = []
    started = time.perf_counter()
    if "L1" in layers:
        for rule in ruleset.rules:
            for start, end in rule.spans(text):
                found.append(Finding(rule.category, rule.id, rule.aliases, start, end, "L1"))
    _add(timings, "L1", started)
    if "L0" in layers and "L1" in layers:
        started = time.perf_counter()
        for view in views(text):
            for rule in ruleset.rules:
                for start, end in rule.spans(view.text):
                    lo, hi = view.original(start, end)
                    found.append(Finding(rule.category, rule.id, rule.aliases, lo, hi, "L0", view.depth > 0))
        _add(timings, "L0", started)
    for layer in ("L2", "L3"):
        if layer in layers and detectors and layer in detectors:
            started = time.perf_counter()
            for category, start, end in detectors[layer](text):
                if category not in ruleset.categories or not 0 <= start < end <= len(text):
                    raise ValueError(f"detector {layer} returned an unknown category or a bad span")
                found.append(Finding(category, f"PG-{layer}-{category.upper()}", (), start, end, layer))
            _add(timings, layer, started)
    return resolve(found, ruleset)


def _add(timings: dict[str, float] | None, layer: str, started: float) -> None:
    if timings is not None:
        timings[layer] = timings.get(layer, 0.0) + (time.perf_counter() - started) * 1000


def resolve(findings: list[Finding], ruleset: Ruleset) -> list[Finding]:
    def is_block(f: Finding) -> bool:
        return ruleset.categories[f.category].action == "block"

    # The same span found on the text and on a view is one finding; the plain L1 one is kept (sorted last).
    unique = {(f.category, f.rule_id, f.start, f.end): f for f in sorted(findings, key=lambda f: f.layer == "L1")}
    blocks = sorted((f for f in unique.values() if is_block(f)), key=lambda f: (f.start, f.end, f.rule_id))
    masks = [f for f in unique.values() if not is_block(f)
             and not any(b.start < f.end and f.start < b.end for b in blocks)]
    masks.sort(key=lambda f: (-(f.end - f.start), f.start, f.rule_id))
    kept: list[Finding] = []
    for f in masks:
        if not any(k.start < f.end and f.start < k.end for k in kept):
            kept.append(f)
    kept.sort(key=lambda f: f.start)
    return blocks + kept
