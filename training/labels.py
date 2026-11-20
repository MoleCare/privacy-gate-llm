# SPDX-License-Identifier: Apache-2.0
"""The four labels, and the one thing the gate actually does with them."""

from __future__ import annotations

CLEAN = "CLEAN"
HEALTH = "HEALTH"
SECRET = "SECRET"
PII = "PII"

LABELS: tuple[str, ...] = (CLEAN, HEALTH, SECRET, PII)

#: The gate itself only ever needs this much. Asking a very small model for the
#: category as well turned out to cost it the decision (JOURNAL, run 1), so the
#: two questions are separable and are measured separately. The words are verbs
#: rather than names because the model has to pick an action, not classify.
KEEP = "KEEP"
SEND = "SEND"
BINARY: tuple[str, ...] = (SEND, KEEP)

#: Order used when a text carries more than one kind of sensitive content.
#: It decides the logged label only; the gate itself is binary.
PRECEDENCE: tuple[str, ...] = (HEALTH, SECRET, PII)


def holds(label: str) -> bool:
    """True when this label means "do not let the text leave the machine"."""
    return label not in (CLEAN, SEND)


def normalise(raw: str, vocabulary: tuple[str, ...] = LABELS) -> str | None:
    """Read a label out of whatever a model actually returned.

    Small models add punctuation, quotes, a stray full stop, or wrap the word in
    JSON even when asked for one word. Anything that is not exactly one of the
    labels returns None, and the caller counts it as unparseable rather than
    guessing. Guessing here would quietly turn a format bug into a privacy result.
    """
    text = raw.strip().strip("\"'`.,:;*").upper()
    if text in vocabulary:
        return text
    # A single label somewhere in a short reply is still unambiguous; two are not.
    found = [label for label in vocabulary if label in text]
    return found[0] if len(found) == 1 else None
