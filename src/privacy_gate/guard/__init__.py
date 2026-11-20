# SPDX-License-Identifier: Apache-2.0
"""The guard: finds secrets, identifiers and contact details in text bound for an LLM provider, and decides
whether to block, mask or allow it. Standard library only.

The rules live in `rules/privacy-v2.yaml`, compiled to `privacy-v2.json` beside this module (P0). The core (P1)
is `Guard`: walk the body, normalise and decode (L0), match the rules (L1), apply the policy, mask with session
surrogates, and unmask the reply as it streams. It is a net, not a proof, and not a medical device: see
SECURITY.md.
"""

from .audit import AuditWriter, Decision
from .detect import Detector, Finding, detect
from .guard import Guard, Protected, safe_path
from .policy import Policy, PolicyError, check_local_url
from .rules import Category, Rule, RulesError, Ruleset
from .unmask import StreamUnmasker, ToolArgsBuffer, unmask, unmask_json_text
from .vault import SessionStore, Vault
from .walker import ScanLimitError, ScanLimits, scan_units

__all__ = [
    "AuditWriter",
    "Category",
    "Decision",
    "Detector",
    "Finding",
    "Guard",
    "Policy",
    "PolicyError",
    "Protected",
    "Rule",
    "RulesError",
    "Ruleset",
    "ScanLimitError",
    "ScanLimits",
    "SessionStore",
    "StreamUnmasker",
    "ToolArgsBuffer",
    "Vault",
    "check_local_url",
    "detect",
    "safe_path",
    "scan_units",
    "unmask",
    "unmask_json_text",
]
