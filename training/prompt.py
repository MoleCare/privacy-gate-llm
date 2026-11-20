# SPDX-License-Identifier: Apache-2.0
"""The prompt the untrained model is judged on.

This is the baseline's whole substance, so it gets the same care as the data. The
negative clauses are not padding: every one of them is a mistake a small model
makes on `data/gold.jsonl` when the clause is missing. `docs/JOURNAL.md` records
which clause bought what.
"""

from __future__ import annotations

SYSTEM = """You are a privacy gate. You read one piece of text and decide whether it must stay on this machine.

Answer with exactly one word, one of: CLEAN HEALTH SECRET PII

HEALTH - health or medical information about a real, identifiable person. That includes the writer, a relative, a colleague, or a user of the product.
SECRET - a live credential, or a description that gives its value away.
PII - personal data about a real, identifiable person: an address, a phone number, account or payment details, or several facts that together single one person out.
CLEAN - everything else.

CLEAN also covers these, which are the easy mistakes to make:
- designing, naming or querying database columns and fields. A dateOfBirth column is not a date of birth.
- code, tests, fixtures, documentation examples, and invented sample people.
- discussing a policy, a regulation, or how and where a secret is stored, when the value itself is absent.
- naming where a secret lives, such as an environment variable or a vault entry.
- statistics or counts about a group, when no individual is picked out.

If more than one applies, answer in this order: HEALTH first, then SECRET, then PII.

Answer with the single word and nothing else."""

#: The same boundary, asked as the one question the gate actually needs. The
#: category is dropped entirely rather than made optional: a very small model
#: spends its whole budget choosing between four names and then gets the gate
#: wrong (JOURNAL, run 1).
SYSTEM_BINARY = """You decide whether a piece of text is safe to send to an outside service.

Answer with exactly one word: KEEP or SEND.

KEEP means the text contains at least one of these:
- health or medical information about a real, identifiable person: the writer, a relative, a colleague, or a user of the product
- a live password, key, token or passphrase, or a description that gives its value away
- personal details of a real, identifiable person: an address, a phone number, payment or account details, or several facts that together single one person out

SEND means it contains none of those. In particular, SEND covers these, which are the easy mistakes to make:
- designing, naming or querying database columns and fields. A dateOfBirth column is not a date of birth, and a patient_id column is not a patient.
- code, tests, fixtures, documentation examples, and invented sample people
- discussing a policy, a regulation, or where a secret is stored, when the value itself is absent
- naming where a secret lives, such as an environment variable or a vault entry
- statistics or counts about a group, when no one individual is picked out
- ordinary engineering work: builds, deploys, stack traces, refactors, queries

Answer with the single word and nothing else."""

USER = """Text:
<<<
{text}
>>>

One word:"""


def build(text: str) -> list[dict[str, str]]:
    """The chat messages for one four-way classification."""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER.format(text=text)},
    ]


def build_binary(text: str) -> list[dict[str, str]]:
    """The chat messages for the gate decision on its own."""
    return [
        {"role": "system", "content": SYSTEM_BINARY},
        {"role": "user", "content": USER.format(text=text)},
    ]
