# SPDX-License-Identifier: Apache-2.0
"""Cheap lexical features, as a baseline and as an audit of the gold set.

This exists for two reasons, and the second is the important one.

**As a baseline.** Word features cost nothing: no model, no GPU, no network. If a
logistic head over hashed unigrams already separates the set, then any embedding
or fine-tuned model has to beat *that*, not the regex ruleset. Reporting a
neural result against a strawman would overstate it.

**As an audit.** The gold set was written by one person in one sitting. If the
clean examples are short imperatives about code and the sensitive ones are long
narrative sentences about people, a classifier can score well by learning the
author's habits rather than the boundary. A high lexical score is therefore
ambiguous — it may mean the task has genuine lexical signal ("passphrase",
"lesion", "biopsy" really are informative), or it may mean the set is
contaminated by style. Either way it must be known before any neural number is
quoted, and `docs/JOURNAL.md` reports the length statistics alongside it so the
two readings can be told apart.

Hashing rather than a fitted vocabulary is deliberate: a vocabulary built over
the whole set before cross-validation leaks the evaluation fold into the
features. A hash needs no fitting, so each fold is clean by construction.
"""

from __future__ import annotations

import hashlib
import re
import statistics

#: Words, numbers, and the punctuation that carries meaning here. `patient_id`
#: splits into two tokens on purpose: the model should not key on one identifier
#: spelling.
TOKEN = re.compile(r"[a-z0-9]+")

DEFAULT_DIMENSIONS = 512


def tokenise(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def _bucket(token: str, dimensions: int) -> int:
    digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimensions


def vectorise(text: str, dimensions: int = DEFAULT_DIMENSIONS) -> list[float]:
    """A hashed bag of words, L2-normalised.

    Normalising matters for the audit: without it the vector's magnitude encodes
    length, and a length difference between the classes would show up as a
    "lexical" result when it is really a style result.
    """
    vector = [0.0] * dimensions
    for token in tokenise(text):
        vector[_bucket(token, dimensions)] += 1.0
    norm = sum(value * value for value in vector) ** 0.5
    if norm:
        vector = [value / norm for value in vector]
    return vector


def length_report(texts_by_class: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    """Characters and tokens per class, so a length artefact cannot hide.

    If the two classes differ sharply here, a good lexical score says more about
    how the examples were written than about what they mean.
    """
    report: dict[str, dict[str, float]] = {}
    for name, texts in texts_by_class.items():
        characters = [len(t) for t in texts]
        tokens = [len(tokenise(t)) for t in texts]
        report[name] = {
            "n": len(texts),
            "chars_median": statistics.median(characters),
            "chars_mean": round(statistics.mean(characters), 1),
            "tokens_median": statistics.median(tokens),
            "tokens_mean": round(statistics.mean(tokens), 1),
        }
    return report
