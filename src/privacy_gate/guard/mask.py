# SPDX-License-Identifier: Apache-2.0
"""Masking: replace each maskable span with its surrogate from the session vault.

After masking, the text is scanned again. Any finding left that is not exactly a surrogate this vault issued
means masking did not hold, and the caller blocks: masking is never allowed to send what it meant to hide.
"""

from __future__ import annotations

from collections.abc import Iterable

from .detect import Finding, detect
from .rules import Ruleset
from .vault import Vault


def surrogate_kind(finding: Finding, ruleset: Ruleset) -> tuple[str, str]:
    category = ruleset.categories[finding.category]
    # A value found inside an encoded run masks the whole run, which is not shaped like the value.
    kind = "tag" if finding.encoded else category.surrogate
    return kind, category.label


def mask_text(text: str, findings: Iterable[Finding], vault: Vault, ruleset: Ruleset) -> str:
    """`text` with each finding's span replaced by its surrogate. Findings must be disjoint."""
    out = text
    for finding in sorted(findings, key=lambda f: f.start, reverse=True):
        kind, label = surrogate_kind(finding, ruleset)
        value = text[finding.start:finding.end]
        out = out[:finding.start] + vault.surrogate(finding.category, value, kind, label) + out[finding.end:]
    return out


def residual(text: str, vault: Vault, ruleset: Ruleset, layers: Iterable[str]) -> list[Finding]:
    """Findings in masked text other than the surrogates the vault issued."""
    left = []
    for finding in detect(text, ruleset, layers):
        if not vault.issued(text[finding.start:finding.end]):
            left.append(finding)
    return left
