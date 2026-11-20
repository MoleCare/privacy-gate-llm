# SPDX-License-Identifier: Apache-2.0
"""The rules file, loaded and checked, and one rule matched against one text (layer L1).

`rules/privacy-v2.yaml` is the source a person edits. `scripts/build_rules.py` compiles it into
`privacy-v2.json` next to this module, which is what the engine loads: the core stays standard library only,
and Python has no YAML parser in the standard library. The generator also writes `rules/conformance.v2.json`,
every fixture with the spans it must produce, which every engine (this one, the TypeScript one, llm-harness)
runs byte for byte.

Matching semantics, which the conformance suite pins down for every engine:

- Regexes are RE2-portable: no lookaround, no backreferences, no named groups, no possessive quantifiers, no
  `\\A`/`\\Z`, an inline `(?i)` only at the start. They are compiled with ASCII semantics (`\\b`, `\\w`, `\\d`,
  `\\s`), which is what JavaScript and RE2 do; Unicode lookalikes are folded by L0 before L1 runs.
- A pattern may name a capture `group`; the span is that group, not the whole match. That is how a home path
  masks only the user name and a secret assignment reports only the value.
- A span is dropped when its rule's `validator` rejects it, when an `except` regex matches the *whole* span
  text (exceptions are judged per span, never per text: a real key beside AWS's placeholder still fires), or
  when the span, with spaces and hyphens removed, is one of `except_values`.
- What is left from all of a rule's patterns is merged where spans overlap, giving sorted, disjoint spans.
- Offsets are in Unicode code points.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .validators import VALIDATORS

SCHEMA = "privacy-gate-rules/2"
ACTIONS_BY_CATEGORY = ("block", "mask")
SURROGATE_KINDS = ("email", "phone", "private_ip", "home_path", "tag", "none")
LAYERS = ("L0", "L1", "L2", "L3")
COMBINATION_MODES = ("block", "shadow")

DEFAULT_RULES_PATH = Path(__file__).with_name("privacy-v2.json")


class RulesError(ValueError):
    """The rules file is malformed. A guard that cannot load its rules refuses everything (fail closed)."""


# Constructs that RE2 or JavaScript do not share with Python, so a rule using one would match differently
# (or not compile) in another engine.
_NOT_PORTABLE = (
    (re.compile(r"\(\?<?[=!]"), "lookaround"),
    (re.compile(r"\(\?P?<[A-Za-z_]"), "named group"),
    (re.compile(r"\(\?P="), "named backreference"),
    (re.compile(r"\(\?>"), "atomic group"),
    (re.compile(r"(?<!\\)\\[1-9]"), "backreference"),
    (re.compile(r"(?<!\\)\\[AZz]"), "\\A or \\Z anchor"),
    (re.compile(r"(?<!\\)[*+?}]\+"), "possessive quantifier"),
    (re.compile(r".\(\?[a-zA-Z]+\)"), "inline flag after the start"),
)


def check_portable(regex: str) -> None:
    for probe, what in _NOT_PORTABLE:
        if probe.search(regex):
            raise RulesError(f"pattern is not RE2-portable ({what}): {regex!r}")


def _compile(regex: str) -> re.Pattern[str]:
    check_portable(regex)
    try:
        return re.compile(regex, re.ASCII)
    except re.error as error:
        raise RulesError(f"pattern does not compile ({error}): {regex!r}") from None


def _strip_separators(text: str) -> str:
    return text.replace(" ", "").replace("-", "")


def _require(mapping: dict[str, Any], allowed: set[str], required: set[str], where: str) -> None:
    if not isinstance(mapping, dict):
        raise RulesError(f"{where}: expected a mapping")
    unknown = set(mapping) - allowed
    if unknown:
        raise RulesError(f"{where}: unknown keys {sorted(unknown)}")
    missing = required - set(mapping)
    if missing:
        raise RulesError(f"{where}: missing keys {sorted(missing)}")


@dataclass(frozen=True)
class Pattern:
    regex: str
    group: int
    compiled: re.Pattern[str] = field(repr=False, compare=False)


@dataclass(frozen=True)
class Category:
    name: str
    title: str
    action: str           # the shipped default: block or mask
    overridable: bool     # False: no policy, header or marker may loosen it
    surrogate: str        # how a masked value is written; "none" for block categories
    layers: tuple[str, ...]
    label: str = ""       # the word in a tag surrogate: <IBAN_7qk2fa>


@dataclass(frozen=True)
class Rule:
    id: str
    title: str
    category: str
    patterns: tuple[Pattern, ...]
    validator: str | None = None
    excepts: tuple[re.Pattern[str], ...] = field(default=(), repr=False, compare=False)
    except_values: frozenset[str] = frozenset()
    aliases: tuple[str, ...] = ()

    def spans(self, text: str) -> list[tuple[int, int]]:
        """Every span of `text` this rule holds, sorted and disjoint."""
        found: list[tuple[int, int]] = []
        check = VALIDATORS[self.validator] if self.validator else None
        for pattern in self.patterns:
            for match in pattern.compiled.finditer(text):
                start, end = match.span(pattern.group)
                if start < 0 or start == end:
                    continue
                value = text[start:end]
                if check is not None and not check(value):
                    continue
                if any(e.fullmatch(value) for e in self.excepts):
                    continue
                if self.except_values and _strip_separators(value) in self.except_values:
                    continue
                found.append((start, end))
        found.sort(key=lambda s: (s[0], -s[1]))
        merged: list[tuple[int, int]] = []
        for start, end in found:
            if merged and start < merged[-1][1]:
                if end > merged[-1][1]:
                    merged[-1] = (merged[-1][0], end)
                continue
            merged.append((start, end))
        return merged


_CATEGORY_KEYS = {"title", "action", "overridable", "surrogate", "layers", "label", "rationale"}
_RULE_KEYS = {"id", "title", "category", "patterns", "validator", "except", "except_values", "aliases",
              "rationale", "fixtures"}
_TOP_KEYS = {"schema", "id", "version", "updated", "note", "source_sha256", "seeded_from", "categories",
             "profiles", "combination", "rules"}
_ID = re.compile(r"PG-[A-Z0-9]+-\d{3}")
# Ids older engines and downstream readers already use (llm-harness PRIV001-005). Reported beside the
# rule id, never instead of it, so headers, audit lines and dashboards keyed on them keep working.
_ALIAS = re.compile(r"[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*")
_SEED_KEYS = {"repo", "pr", "commit", "ruleset_v1_sha256", "note"}


@dataclass(frozen=True)
class Ruleset:
    id: str
    version: str
    sha256: str
    categories: dict[str, Category]
    rules: tuple[Rule, ...]
    profiles: dict[str, dict[str, Any]]
    combination: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path | None = None) -> Ruleset:
        raw = Path(path or DEFAULT_RULES_PATH).read_bytes()
        try:
            data = json.loads(raw)
        except ValueError as error:
            raise RulesError(f"rules file is not JSON: {error}") from None
        return cls.from_mapping(data, sha256=hashlib.sha256(raw).hexdigest())

    @classmethod
    def from_mapping(cls, data: dict[str, Any], sha256: str | None = None) -> Ruleset:
        _require(data, _TOP_KEYS, {"schema", "id", "version", "categories", "rules"}, "ruleset")
        if "seeded_from" in data:
            _require(data["seeded_from"], _SEED_KEYS, {"repo", "commit", "ruleset_v1_sha256"}, "seeded_from")
        if data["schema"] != SCHEMA:
            raise RulesError(f"ruleset: schema must be {SCHEMA!r}, got {data['schema']!r}")
        categories = {name: _category(name, spec) for name, spec in data["categories"].items()}
        rules: list[Rule] = []
        seen: set[str] = set()
        for spec in data["rules"]:
            rule = _rule(spec, categories)
            if rule.id in seen:
                raise RulesError(f"rule {rule.id}: id used twice")
            seen.add(rule.id)
            rules.append(rule)
        profiles = data.get("profiles", {})
        for name, profile in profiles.items():
            _require(profile, {"layers", "combination", "note"}, {"layers", "combination"}, f"profile {name}")
            if not set(profile["layers"]) <= set(LAYERS) or profile["combination"] not in COMBINATION_MODES:
                raise RulesError(f"profile {name}: unknown layer or combination mode")
        combination = data.get("combination", {})
        if combination:
            _require(combination, {"rule_id", "min_distinct", "rationale"}, {"rule_id", "min_distinct"},
                     "combination")
        if sha256 is None:
            canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            sha256 = hashlib.sha256(canonical.encode()).hexdigest()
        return cls(id=data["id"], version=str(data["version"]), sha256=sha256, categories=categories,
                   rules=tuple(rules), profiles=profiles, combination=combination)

    def rule(self, rule_id: str) -> Rule:
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        raise KeyError(rule_id)

    def with_overlay(self, overlay: Ruleset) -> Ruleset:
        """Add an organisation's own rules (same schema). An overlay may add rules to existing categories
        only; it cannot add, remove or loosen a category, and it cannot reuse a shipped rule id."""
        ids = {r.id for r in self.rules}
        for rule in overlay.rules:
            if rule.id in ids:
                raise RulesError(f"overlay rule {rule.id}: id already used by the shipped rules")
            if rule.category not in self.categories:
                raise RulesError(f"overlay rule {rule.id}: unknown category {rule.category!r}")
        combined = hashlib.sha256(f"{self.sha256}\n{overlay.sha256}".encode()).hexdigest()
        return Ruleset(id=self.id, version=self.version, sha256=combined, categories=self.categories,
                       rules=self.rules + overlay.rules, profiles=self.profiles, combination=self.combination)


def _category(name: str, spec: dict[str, Any]) -> Category:
    where = f"category {name}"
    _require(spec, _CATEGORY_KEYS, {"title", "action", "overridable", "surrogate", "layers"}, where)
    if spec["action"] not in ACTIONS_BY_CATEGORY:
        raise RulesError(f"{where}: action must be one of {ACTIONS_BY_CATEGORY}")
    if spec["surrogate"] not in SURROGATE_KINDS:
        raise RulesError(f"{where}: surrogate must be one of {SURROGATE_KINDS}")
    if not isinstance(spec["overridable"], bool):
        raise RulesError(f"{where}: overridable must be true or false")
    # A block category that could be loosened would be a mask category with extra steps; the design keeps
    # the two apart so that "never overridable" is a property of the file, not of a policy.
    if spec["action"] == "block" and spec["overridable"]:
        raise RulesError(f"{where}: a block category cannot be overridable")
    if spec["action"] == "mask" and spec["surrogate"] == "none":
        raise RulesError(f"{where}: a mask category needs a surrogate")
    if not spec["layers"] or not set(spec["layers"]) <= set(LAYERS):
        raise RulesError(f"{where}: layers must be a non-empty subset of {LAYERS}")
    return Category(name=name, title=spec["title"], action=spec["action"], overridable=spec["overridable"],
                    surrogate=spec["surrogate"], layers=tuple(spec["layers"]),
                    label=str(spec.get("label") or name.upper()))


def _rule(spec: dict[str, Any], categories: dict[str, Category]) -> Rule:
    rule_id = spec.get("id", "?") if isinstance(spec, dict) else "?"
    where = f"rule {rule_id}"
    _require(spec, _RULE_KEYS, {"id", "title", "category", "patterns"}, where)
    if not _ID.fullmatch(rule_id):
        raise RulesError(f"{where}: id must look like PG-SECRET-001")
    if spec["category"] not in categories:
        raise RulesError(f"{where}: unknown category {spec['category']!r}")
    validator = spec.get("validator")
    if validator is not None and validator not in VALIDATORS:
        raise RulesError(f"{where}: unknown validator {validator!r}")
    patterns: list[Pattern] = []
    for item in spec["patterns"] or []:
        if isinstance(item, str):
            regex, group = item, 0
        else:
            _require(item, {"regex", "group"}, {"regex"}, where)
            regex, group = item["regex"], int(item.get("group", 0))
        compiled = _compile(regex)
        if not 0 <= group <= compiled.groups:
            raise RulesError(f"{where}: group {group} does not exist in {regex!r}")
        patterns.append(Pattern(regex=regex, group=group, compiled=compiled))
    if not patterns:
        raise RulesError(f"{where}: no patterns")
    aliases = spec.get("aliases") or []
    if not isinstance(aliases, list) or not all(isinstance(a, str) and _ALIAS.fullmatch(a) for a in aliases):
        raise RulesError(f"{where}: aliases must be a list of ids like PRIV003")
    if _ID.fullmatch(rule_id) and rule_id in aliases:
        raise RulesError(f"{where}: a rule cannot alias itself")
    excepts = tuple(_compile(e) for e in spec.get("except", []) or [])
    except_values = frozenset(_strip_separators(str(v)) for v in spec.get("except_values", []) or [])
    return Rule(id=rule_id, title=spec["title"], category=spec["category"], patterns=tuple(patterns),
                validator=validator, excepts=excepts, except_values=except_values, aliases=tuple(aliases))
