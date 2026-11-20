# SPDX-License-Identifier: Apache-2.0
"""The guard: one call that takes a request body and says what may leave (design, section 5.4).

    guard = Guard.load()                      # lite profile, the shipped rules, enforce mode
    result = guard.protect(body, caller=key_hash, session="s1", provider="anthropic")
    if result.decision.sends:
        send(result.body)                     # masked where needed
        reply = guard.unmasker(caller=key_hash, session="s1")   # StreamUnmasker for the reply

The order of a decision:
1. Walk the body (`walker`): every leaf as ``key: value``, numbers too, binary blocks held, caps fail closed.
2. Strip ``[[send-as-is:...]]`` markers from string values and note the categories they name.
3. Detect (`detect`): L1 on the text, L0 views, and L2/L3 detectors when the profile has them.
4. Decide each finding: the policy's action for the category and provider; an override or an allowlisted
   value may turn *mask* into *allow*, never *block* into anything.
5. Four or more distinct maskable categories in one value fire PG-COMBO-001: a block in the standard and full
   profiles, a shadow report in lite.
6. Any block -> refuse (or ``local`` when the policy says so); else mask and re-scan the masked text, and
   block if anything but an issued surrogate is left; else allow.

Fail closed: if the rules, the policy or a named overlay cannot be loaded, every call returns a block with the
reason. If anything raises while deciding, the call returns a block (architecture review R9). The decision
never carries a value, and an exception's message is not copied into it, since it might quote one.

``on_block: local`` returns the original body for the caller to send to its local model; the caller must
strip provider credentials (Authorization, x-api-key, anthropic-beta) first (review R18). That is the
proxy's job from P3.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .audit import Decision
from .detect import Detector, Finding, detect
from .mask import mask_text, residual
from .normalize import DecodeLimitError
from .policy import Policy, PolicyError, parse_overrides, strip_markers
from .rules import RulesError, Ruleset
from .unmask import StreamUnmasker
from .vault import SessionStore, Vault
from .walker import (
    BINARY_ALIAS,
    BINARY_RULE,
    DEFAULT_LIMITS,
    SCAN_ALIAS,
    SCAN_RULE,
    BinaryUnit,
    ScanLimitError,
    ScanLimits,
    TextUnit,
    path_text,
    scan_units,
)

Hasher = Callable[[str, str], str]
_PLAIN_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,63}")


@dataclass
class Protected:
    decision: Decision
    #: What to send: the masked body (mask), the body as given (allow, local), or None (block).
    body: Any


class Guard:
    def __init__(
        self,
        ruleset: Ruleset | None,
        policy: Policy | None,
        store: SessionStore,
        detectors: dict[str, Detector],
        limits: ScanLimits,
        failure: str | None,
    ) -> None:
        self.ruleset = ruleset
        self.policy = policy
        self.store = store
        self.detectors = detectors
        self.limits = limits
        self.failure = failure

    @classmethod
    def load(
        cls,
        policy: str | Path | dict[str, Any] | Policy | None = None,
        profile: str | None = None,
        *,
        ruleset: Ruleset | None = None,
        rules_path: str | Path | None = None,
        store: SessionStore | None = None,
        detectors: dict[str, Detector] | None = None,
        limits: ScanLimits = DEFAULT_LIMITS,
    ) -> Guard:
        """Never raises: a guard that cannot load refuses every request and says why."""
        detectors = dict(detectors or {})
        store = store or SessionStore()
        try:
            rules = ruleset or Ruleset.load(rules_path)
            if isinstance(policy, Policy):
                loaded = policy
            elif isinstance(policy, dict):
                loaded = Policy.from_mapping(policy, rules)
            elif policy is not None:
                loaded = Policy.load(policy, rules)
            else:
                loaded = Policy.from_mapping({"extends": f"builtin:{profile or 'lite'}"}, rules)
            if profile is not None and profile != loaded.profile:
                if profile not in rules.profiles:
                    raise PolicyError(f"unknown profile {profile!r}")
                loaded = replace(loaded, profile=profile)
            for path, pinned in loaded.overlays:
                rules = rules.with_overlay(_load_overlay(path, pinned))
            if loaded.profile not in rules.profiles:
                raise PolicyError(f"the rules have no profile {loaded.profile!r}")
            for layer in rules.profiles[loaded.profile]["layers"]:
                if layer in ("L2", "L3") and layer not in detectors:
                    raise PolicyError(
                        f"profile {loaded.profile} needs the {layer} detector, which is not installed"
                    )
        except (RulesError, PolicyError, OSError, ValueError, KeyError, TypeError) as error:
            return cls(None, None, store, detectors, limits, failure=f"{type(error).__name__}: {error}")
        return cls(rules, loaded, store, detectors, limits, failure=None)

    def vault(self, caller: str = "", session: str = "default") -> Vault:
        return self.store.vault(caller, session)

    def unmasker(
        self, caller: str = "", session: str = "default", escape_json: bool = False
    ) -> StreamUnmasker:
        return StreamUnmasker(self.vault(caller, session), escape_json=escape_json)

    def protect(
        self,
        body: Any,
        *,
        caller: str = "",
        session: str = "default",
        provider: str = "",
        wire: str = "",
        overrides: str | None = None,
        hasher: Hasher | None = None,
    ) -> Protected:
        started = time.perf_counter()
        decision = Decision(action="block", provider=provider, wire=wire)
        try:
            if self.failure is not None or self.ruleset is None or self.policy is None:
                decision.reason = f"guard not loaded, refusing everything ({self.failure})"
                decision.fail_closed = True
                return Protected(decision, None)
            decision.mode = self.policy.mode
            decision.profile = self.policy.profile
            decision.ruleset_sha256 = self.ruleset.sha256
            return self._protect(body, decision, caller, session, provider, overrides, hasher)
        except (ScanLimitError, DecodeLimitError) as error:
            _fail(
                decision, SCAN_RULE, SCAN_ALIAS, f"the body could not be read in full ({error}); nothing sent"
            )
            return Protected(decision, None)
        except Exception as error:  # noqa: BLE001 - R9: whatever went wrong, nothing unread may leave
            _fail(decision, SCAN_RULE, SCAN_ALIAS, f"engine error ({type(error).__name__}); nothing sent")
            return Protected(decision, None)
        finally:
            decision.guard_ms = (time.perf_counter() - started) * 1000

    def _protect(
        self,
        body: Any,
        decision: Decision,
        caller: str,
        session: str,
        provider: str,
        overrides: str | None,
        hasher: Hasher | None,
    ) -> Protected:
        rules, policy = self.ruleset, self.policy
        assert rules is not None and policy is not None
        layers = tuple(rules.profiles[policy.profile]["layers"])
        combination_mode = rules.profiles[policy.profile]["combination"]
        requested = parse_overrides(overrides)
        timings: dict[str, float] = {}

        units = list(scan_units(body, self.limits))
        rewrites: dict[tuple[str | int, ...], str] = {}
        per_unit: list[tuple[TextUnit, str, list[tuple[Finding, str]]]] = []
        block_hits: list[tuple[str, str, tuple[str, ...], tuple[str | int, ...], bool]] = []

        for unit in units:
            if isinstance(unit, BinaryUnit):
                block_hits.append(("image_document", BINARY_RULE, (BINARY_ALIAS,), unit.path, False))
                _note(decision, BINARY_RULE, (BINARY_ALIAS,), "image_document", unit.path, False)
                continue
            text = unit.text
            if unit.is_string:
                stripped, named = strip_markers(text[unit.value_offset :])
                if named:
                    requested |= named
                    text = text[: unit.value_offset] + stripped
                    rewrites[unit.path] = stripped
            per_unit.append(
                (unit, text, [(f, "") for f in detect(text, rules, layers, self.detectors, timings)])
            )

        allowed_overrides = {
            c for c in requested if c in rules.categories and rules.categories[c].overridable
        }
        decision.refused_override = sorted(requested - allowed_overrides)
        overridden: set[str] = set()
        to_mask: list[tuple[TextUnit, str, list[Finding]]] = []
        acted: list[tuple[Finding, str]] = []

        for unit, text, items in per_unit:
            masks: list[Finding] = []
            maskable_categories: set[str] = set()
            for finding, _ in items:
                value = text[finding.start : finding.end]
                action = self._action(finding.category, value, provider, allowed_overrides)
                if action == "allow":
                    if (
                        policy.action(finding.category, rules, provider) == "mask"
                        and finding.category in allowed_overrides
                    ):
                        overridden.add(finding.category)
                    continue
                in_key = finding.start < unit.value_offset
                if action == "mask" and (in_key or not unit.is_string):
                    # A key cannot be masked in place, and a number would change type: hold instead.
                    action = "block"
                acted.append((finding, value))
                _note(decision, finding.rule_id, finding.aliases, finding.category, unit.path, in_key)
                if action == "block":
                    block_hits.append((finding.category, finding.rule_id, finding.aliases, unit.path, in_key))
                else:
                    masks.append(finding)
                    maskable_categories.add(finding.category)
            if len(maskable_categories) >= int(rules.combination.get("min_distinct", 0) or 10**9):
                combo = str(rules.combination["rule_id"])
                if combination_mode == "block":
                    block_hits.append(("combination", combo, (), unit.path, False))
                    _add_rule(decision, combo, ())
                else:
                    decision.shadow_rules.append(combo)
            if masks:
                to_mask.append((unit, text, masks))

        decision.override = sorted(overridden)
        if hasher is not None:
            decision.value_hashes = sorted({hasher(f.category, value) for f, value in acted})
        decision.detectors = timings

        would = "block" if block_hits else "mask" if to_mask else "allow"
        decision.would_action = would if policy.mode == "shadow" else None
        if policy.mode == "shadow":
            # Shadow mode reports; it does not loosen the never-overridable categories unless the policy says
            # so explicitly (review R10). Binary blocks are image_document, a block category, so they hold too.
            hard = [
                h
                for h in block_hits
                if h[0] in rules.categories
                and rules.categories[h[0]].action == "block"
                and not policy.shadow_block_categories
            ]
            if not hard:
                decision.shadow_rules = sorted(set(decision.shadow_rules) | set(decision.rules))
                decision.action = "allow"
                decision.reason = f"shadow mode: would {would}"
                return Protected(decision, _rewritten(body, rewrites))
            block_hits = hard

        if block_hits:
            decision.action = "local" if policy.on_block == "local" else "block"
            names = sorted({h[0] for h in block_hits})
            decision.reason = f"held: {', '.join(names)}"
            return Protected(decision, _rewritten(body, rewrites) if decision.action == "local" else None)

        if not to_mask:
            decision.action = "allow"
            return Protected(decision, _rewritten(body, rewrites))

        vault = self.store.vault(caller, session)
        for unit, text, masks in to_mask:
            offset = unit.value_offset
            value = text[offset:]
            shifted = [replace(f, start=f.start - offset, end=f.end - offset) for f in masks]
            masked = mask_text(value, shifted, vault, rules)
            left = residual(text[:offset] + masked, vault, rules, layers)
            masked_text = text[:offset] + masked
            if any(
                self._action(f.category, masked_text[f.start : f.end], provider, allowed_overrides) != "allow"
                for f in left
            ):
                _fail(decision, SCAN_RULE, SCAN_ALIAS, "masking left a finding in place; nothing sent")
                return Protected(decision, None)
            rewrites[unit.path] = masked
        decision.action = "mask"
        decision.reason = f"masked: {', '.join(sorted({f.category for _, _, m in to_mask for f in m}))}"
        return Protected(decision, _rewritten(body, rewrites))

    def _action(self, category: str, value: str, provider: str, overrides: set[str]) -> str:
        """The action for one finding: the policy's, then an override or the allowlist may turn mask into allow."""
        assert self.ruleset is not None and self.policy is not None
        action = self.policy.action(category, self.ruleset, provider)
        if action == "mask" and (category in overrides or self.policy.allowed(value)):
            return "allow"
        return action


def _fail(decision: Decision, rule_id: str, alias: str, reason: str) -> None:
    decision.action = "block"
    decision.fail_closed = True
    decision.reason = reason
    _add_rule(decision, rule_id, (alias,))


def _add_rule(decision: Decision, rule_id: str, aliases: tuple[str, ...]) -> None:
    for name in (rule_id, *aliases):
        if name not in decision.rules:
            decision.rules.append(name)


def _note(
    decision: Decision,
    rule_id: str,
    aliases: tuple[str, ...],
    category: str,
    path: tuple[str | int, ...],
    in_key: bool,
) -> None:
    _add_rule(decision, rule_id, aliases)
    decision.categories[category] = decision.categories.get(category, 0) + 1
    where = safe_path(path, in_key)
    if where not in decision.paths:
        decision.paths.append(where)


def safe_path(path: tuple[str | int, ...], last_key_has_finding: bool = False) -> str:
    """A path fit for a decision: keys that are not plain identifiers, or that held a finding, become ``*``,
    because a key can itself be an address or a name."""
    parts: list[str | int] = []
    for i, part in enumerate(path):
        hidden = last_key_has_finding and i == _last_key(path)
        if isinstance(part, int) or (_PLAIN_KEY.fullmatch(part) and not hidden):
            parts.append(part)
        else:
            parts.append("*")
    return path_text(tuple(parts))


def _last_key(path: tuple[str | int, ...]) -> int:
    for i in range(len(path) - 1, -1, -1):
        if isinstance(path[i], str):
            return i
    return -1


def _rewritten(body: Any, rewrites: dict[tuple[str | int, ...], str]) -> Any:
    if not rewrites:
        return body
    out = copy.deepcopy(body)
    for path, value in rewrites.items():
        if not path:
            return value
        target = out
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
    return out


def _load_overlay(path: str, pinned: str | None) -> Ruleset:
    try:
        raw = Path(path).read_bytes()
    except OSError:
        raise RulesError(f"overlay {Path(path).name} is named by the policy but cannot be read") from None
    digest = hashlib.sha256(raw).hexdigest()
    if pinned is not None and digest != pinned:
        raise RulesError(f"overlay {Path(path).name} does not match its pinned sha256")
    try:
        data = json.loads(raw)
    except ValueError:
        raise RulesError(f"overlay {Path(path).name} is not JSON") from None
    return Ruleset.from_mapping(data, sha256=digest)
