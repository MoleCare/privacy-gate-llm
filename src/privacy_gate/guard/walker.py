# SPDX-License-Identifier: Apache-2.0
"""The body walker: what in a request body the rules read.

Most leaks fixed in llm-harness pull request 32 were in the *shape* of the body, not in the regexes: a label
that is a JSON key was never seen beside its value, a number was never seen at all, and a small image or PDF
block was too short for the long-base64 rule. This module is a port of that walker (`scanUnits` and
`binaryKind` in llm-harness `src/egress.ts`), with the same semantics, so that the package and the harness
cannot drift where the last leaks were (architecture review R1):

- every leaf is read as ``key: value``, so a label and its value can match one rule; a top-level leaf, or one
  in a top-level array, has no key and is read bare;
- numbers and booleans are read too, written as JavaScript writes them (``40219``, ``true``); null is not;
- an array's items keep the key the array sits under: ``{"mrn": [98765]}`` reads ``mrn: 98765``;
- an object that is an image, document, audio or file block is reported as a binary unit (and its readable
  fields are still read); a document whose source is plain text or content blocks is readable, not binary;
- past the depth or node cap, on a cycle, or on a value that is not JSON, it raises `ScanLimitError`, and the
  caller holds the whole body. Nothing it did not finish reading may leave.

Python addition: each unit carries the path of the value (``messages.0.content.1.text``) so that a decision
can say where a finding was without saying what it was. The walk is iterative, so depth is bounded by the cap
and not by the interpreter's stack.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

#: Rule ids for what the walker itself holds. The PRIV names are llm-harness's, kept as aliases (review R3).
BINARY_RULE = "PG-BINARY-001"
BINARY_ALIAS = "PRIV-BINARY"
SCAN_RULE = "PG-SCAN-001"
SCAN_ALIAS = "PRIV-SCAN"

OPAQUE_SOURCES = frozenset({"base64", "url", "file"})


@dataclass(frozen=True)
class ScanLimits:
    max_depth: int = 64
    max_nodes: int = 500_000
    #: Characters of text read in total. Larger bodies are refused rather than half read (design, section 7).
    max_chars: int = 2_000_000


DEFAULT_LIMITS = ScanLimits()


class ScanLimitError(Exception):
    """The body could not be read in full. The message names the limit, never the content."""


@dataclass(frozen=True)
class TextUnit:
    path: tuple[str | int, ...]
    key: str | None
    #: What the rules read: ``key: value`` or the bare value.
    text: str
    #: Where the value starts in `text` (after ``key: ``), 0 when bare.
    value_offset: int
    #: True when the value is a string (it can be masked in place); False for a number or boolean.
    is_string: bool


@dataclass(frozen=True)
class BinaryUnit:
    path: tuple[str | int, ...]
    what: str


Unit = TextUnit | BinaryUnit


def binary_kind(obj: dict[str, Any]) -> str | None:
    """What kind of unreadable content an object is, or None. Anthropic image and document blocks (a document
    with a text or content source is readable), any block with an opaque source, OpenAI image, audio and file
    parts, and container uploads."""
    kind = obj.get("type")
    source = obj.get("source")
    source_type = source.get("type") if isinstance(source, dict) else None
    if kind in ("image", "image_url", "input_image"):
        return "image"
    if kind == "document":
        return None if source_type in ("text", "content") else "document"
    if kind == "input_audio":
        return "audio"
    if kind in ("file", "input_file") and ("file" in obj or "file_data" in obj or "file_id" in obj):
        return "file"
    if kind == "container_upload":
        return "file"
    if isinstance(source_type, str) and source_type in OPAQUE_SOURCES:
        return kind if isinstance(kind, str) else "binary"
    return None


def js_string(value: bool | float) -> str:
    """A number or boolean as JavaScript's String() writes it, so both engines read the same text."""
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    if value.is_integer() and abs(value) < 1e21:
        return str(int(value))
    text = repr(value)
    if "e" in text:  # 1e-07 in Python, 1e-7 in JavaScript
        mantissa, exponent = text.split("e")
        sign = "-" if exponent.startswith("-") else "+"
        text = f"{mantissa}e{sign}{int(exponent.lstrip('+-'))}"
    return text


def scan_units(value: Any, limits: ScanLimits = DEFAULT_LIMITS, pair_keys: bool = True) -> Iterator[Unit]:
    """Every unit of a JSON-like value in document order. Raises ScanLimitError past a cap or on a value
    that is not JSON. With ``pair_keys=False`` only strings are read, bare: the text a model reads."""
    stack: list[tuple[Any, str | None, int, tuple[str | int, ...]]] = [(value, None, 0, ())]
    nodes = 0
    chars = 0
    while stack:
        v, key, depth, path = stack.pop()
        nodes += 1
        if nodes > limits.max_nodes:
            raise ScanLimitError(f"more than {limits.max_nodes} values")
        if isinstance(v, str):
            text, offset = (f"{key}: {v}", len(key) + 2) if pair_keys and key is not None else (v, 0)
            chars += len(text)
            if chars > limits.max_chars:
                raise ScanLimitError(f"more than {limits.max_chars} characters")
            yield TextUnit(path, key, text, offset, True)
        elif isinstance(v, (bool, int, float)):
            if pair_keys:
                bare = js_string(v)
                text, offset = (f"{key}: {bare}", len(key) + 2) if key is not None else (bare, 0)
                chars += len(text)
                if chars > limits.max_chars:
                    raise ScanLimitError(f"more than {limits.max_chars} characters")
                yield TextUnit(path, key, text, offset, False)
        elif v is None:
            continue
        elif isinstance(v, (list, tuple)):
            if depth >= limits.max_depth:
                raise ScanLimitError(f"nested deeper than {limits.max_depth}")
            for i in range(len(v) - 1, -1, -1):
                stack.append((v[i], key, depth + 1, (*path, i)))
        elif isinstance(v, dict):
            if depth >= limits.max_depth:
                raise ScanLimitError(f"nested deeper than {limits.max_depth}")
            kind = binary_kind(v)
            if kind:
                yield BinaryUnit(path, kind)
            items = list(v.items())
            for k, x in reversed(items):
                if not isinstance(k, str):
                    raise ScanLimitError("an object key that is not a string")
                stack.append((x, k, depth + 1, (*path, k)))
        else:
            # Bytes, a set, an object: not JSON, so not something the walker can promise to have read.
            raise ScanLimitError(f"a value of type {type(v).__name__} is not JSON")


def path_text(path: tuple[str | int, ...]) -> str:
    """A path as a dotted string for decisions and audit lines: ``messages.0.content``. Keys are kept as they
    are; a key can itself carry data, so callers that log paths should treat them like rule ids, not values."""
    return ".".join(str(p) for p in path) or "$"
