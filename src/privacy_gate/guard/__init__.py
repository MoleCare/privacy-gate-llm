# SPDX-License-Identifier: Apache-2.0
"""The guard: finds secrets, identifiers and contact details in text bound for an LLM provider, and decides
whether to block, mask or allow it. Standard library only.

P0 ships the rules (`rules/privacy-v2.yaml`, compiled to `privacy-v2.json` beside this module) and the L1
matcher that the conformance suite runs. A net, not a proof: see SECURITY.md.
"""

from .rules import Category, Rule, RulesError, Ruleset

__all__ = ["Category", "Rule", "RulesError", "Ruleset"]
