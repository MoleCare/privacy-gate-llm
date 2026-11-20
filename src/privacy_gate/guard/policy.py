# SPDX-License-Identifier: Apache-2.0
"""Policy: what to do with each category, for each provider (design, section 5.3).

The rules file says what each category does by default. A policy may only move a category in the safe
direction, except for one loosening: a *maskable* category may be set to ``allow``. Concretely:

- ``categories`` (top level) may set a maskable category to block, mask or allow. A block category may only
  stay ``block``; anything else is a PolicyError, so "never overridable" cannot be undone by a file.
- ``providers.<name>.categories`` may only tighten (allow -> mask -> block) for that provider.
- ``on_block`` is ``refuse`` (default) or ``local``. ``local.url`` must be loopback, or a host the file names
  in ``local.hosts``; a public IP literal is refused even when named.
- ``allow.values_file`` lists values sent as they are (one per line). They are hashed when loaded and the
  plain values are not kept. An allowed value only ever skips masking, never a block.
- ``mode: shadow`` reports what would happen and changes nothing, except that block categories still block
  unless ``shadow_block_categories: true`` is set as well (architecture review R10).

JSON policy files need nothing; YAML ones need PyYAML until the proxy extra brings it.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .rules import Ruleset

ACTIONS = ("allow", "mask", "block")
_STRENGTH = {"allow": 0, "mask": 1, "block": 2}
PROFILES = ("lite", "standard", "full")
_TOP_KEYS = {
    "version",
    "extends",
    "mode",
    "providers",
    "categories",
    "on_block",
    "local",
    "allow",
    "notice",
    "audit",
    "overlays",
    "shadow_block_categories",
}
_MARKER = re.compile(r"\[\[send-as-is:([a-z_]+(?:,[a-z_]+)*)\]\]")


class PolicyError(ValueError):
    """A policy that cannot be loaded. The guard refuses everything until it is fixed."""


@dataclass(frozen=True)
class Policy:
    profile: str = "lite"
    mode: str = "enforce"
    categories: dict[str, str] = field(default_factory=dict)
    providers: dict[str, dict[str, str]] = field(default_factory=dict)
    on_block: str = "refuse"
    local_url: str | None = None
    shadow_block_categories: bool = False
    notice: bool = False
    audit: dict[str, Any] = field(default_factory=dict)
    #: (path, pinned sha256 or None). A named overlay that is missing or differs refuses everything (R11).
    overlays: tuple[tuple[str, str | None], ...] = ()
    _allow_salt: bytes = field(default=b"", repr=False)
    _allow_hashes: frozenset[bytes] = field(default=frozenset(), repr=False)

    @classmethod
    def from_mapping(cls, data: dict[str, Any], ruleset: Ruleset, base_dir: Path | None = None) -> Policy:
        if not isinstance(data, dict):
            raise PolicyError("policy: expected a mapping")
        unknown = set(data) - _TOP_KEYS
        if unknown:
            raise PolicyError(f"policy: unknown keys {sorted(unknown)}")
        if data.get("version", 1) != 1:
            raise PolicyError("policy: version must be 1")
        extends = data.get("extends", "builtin:lite")
        if not isinstance(extends, str) or not extends.startswith("builtin:") or extends[8:] not in PROFILES:
            raise PolicyError(f"policy: extends must be builtin:{'|'.join(PROFILES)}")
        mode = data.get("mode", "enforce")
        if mode not in ("enforce", "shadow"):
            raise PolicyError("policy: mode must be enforce or shadow")

        categories: dict[str, str] = {}
        for name, action in (data.get("categories") or {}).items():
            _check_category(name, action, ruleset, "categories")
            default = ruleset.categories[name].action
            if default == "block" and action != "block":
                raise PolicyError(f"categories.{name}: a block category is never overridable")
            categories[name] = action

        providers: dict[str, dict[str, str]] = {}
        for provider, spec in (data.get("providers") or {}).items():
            if not isinstance(spec, dict) or set(spec) - {"categories"}:
                raise PolicyError(f"providers.{provider}: only `categories` may be set")
            tightened: dict[str, str] = {}
            for name, action in (spec.get("categories") or {}).items():
                _check_category(name, action, ruleset, f"providers.{provider}.categories")
                base = categories.get(name, ruleset.categories[name].action)
                if _STRENGTH[action] < _STRENGTH[base]:
                    raise PolicyError(f"providers.{provider}.categories.{name}: a provider may only tighten")
                tightened[name] = action
            providers[provider] = tightened

        on_block = data.get("on_block", "refuse")
        if on_block not in ("refuse", "local"):
            raise PolicyError("policy: on_block must be refuse or local")
        local = data.get("local") or {}
        if not isinstance(local, dict) or set(local) - {"url", "hosts"}:
            raise PolicyError("local: only `url` and `hosts` may be set")
        local_url = local.get("url")
        if on_block == "local" and not local_url:
            raise PolicyError("on_block is local but local.url is not set")
        if local_url is not None:
            check_local_url(local_url, tuple(local.get("hosts") or ()))

        shadow_block = data.get("shadow_block_categories", False)
        if not isinstance(shadow_block, bool):
            raise PolicyError("shadow_block_categories must be true or false")

        salt = os.urandom(16)
        hashes: set[bytes] = set()
        allow = data.get("allow") or {}
        if not isinstance(allow, dict) or set(allow) - {"values_file"}:
            raise PolicyError("allow: only `values_file` may be set")
        if allow.get("values_file"):
            values_path = Path(allow["values_file"]).expanduser()
            if not values_path.is_absolute() and base_dir is not None:
                values_path = base_dir / values_path
            try:
                lines = values_path.read_text(encoding="utf-8").splitlines()
            except OSError as error:
                raise PolicyError(f"allow.values_file cannot be read ({error.strerror})") from None
            for line in lines:
                value = line.strip()
                if value and not value.startswith("#"):
                    hashes.add(_hash(salt, value))

        overlays: list[tuple[str, str | None]] = []
        for item in data.get("overlays") or []:
            if isinstance(item, str):
                item = {"path": item}
            if (
                not isinstance(item, dict)
                or set(item) - {"path", "sha256"}
                or not isinstance(item.get("path"), str)
            ):
                raise PolicyError("overlays: each is a path or {path, sha256}")
            overlay_path = Path(item["path"]).expanduser()
            if not overlay_path.is_absolute() and base_dir is not None:
                overlay_path = base_dir / overlay_path
            overlays.append((str(overlay_path), item.get("sha256")))
        return cls(
            profile=extends[8:],
            mode=mode,
            categories=categories,
            providers=providers,
            on_block=on_block,
            local_url=local_url,
            shadow_block_categories=shadow_block,
            notice=bool(data.get("notice", False)),
            audit=dict(data.get("audit") or {}),
            overlays=tuple(overlays),
            _allow_salt=salt,
            _allow_hashes=frozenset(hashes),
        )

    @classmethod
    def load(cls, path: str | Path, ruleset: Ruleset) -> Policy:
        path = Path(path).expanduser()
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as error:
            raise PolicyError(f"policy file cannot be read ({error.strerror})") from None
        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError:
                raise PolicyError("a YAML policy needs PyYAML; write it as JSON or install PyYAML") from None
            try:
                data = yaml.safe_load(raw)
            except yaml.YAMLError:
                raise PolicyError("policy file is not valid YAML") from None
        else:
            try:
                data = json.loads(raw)
            except ValueError:
                raise PolicyError("policy file is not valid JSON") from None
        return cls.from_mapping(data, ruleset, base_dir=path.parent)

    def action(self, category: str, ruleset: Ruleset, provider: str | None = None) -> str:
        """The action for a category and provider before overrides and the allowlist."""
        action = self.categories.get(category, ruleset.categories[category].action)
        tightened = self.providers.get(provider or "", {}).get(category)
        if tightened is not None and _STRENGTH[tightened] > _STRENGTH[action]:
            action = tightened
        return action

    def allowed(self, value: str) -> bool:
        return bool(self._allow_hashes) and _hash(self._allow_salt, value) in self._allow_hashes


def _hash(salt: bytes, value: str) -> bytes:
    return hashlib.sha256(salt + value.encode("utf-8")).digest()


def _check_category(name: str, action: Any, ruleset: Ruleset, where: str) -> None:
    if name not in ruleset.categories:
        raise PolicyError(f"{where}.{name}: unknown category")
    if action not in ACTIONS:
        raise PolicyError(f"{where}.{name}: action must be one of {ACTIONS}")


def check_local_url(url: str, hosts: tuple[str, ...] = ()) -> None:
    """A local model URL: http or https, and a loopback host or one named in `hosts`. A public IP literal is
    refused even when named: "local" that is a public address is not local."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise PolicyError("local.url must be an http or https URL with a host")
    host = parts.hostname
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None:
        if address.is_loopback:
            return
        if not (address.is_private or address.is_link_local):
            raise PolicyError("local.url is a public IP address")
        if host in hosts:
            return
        raise PolicyError("local.url is not loopback and its host is not listed in local.hosts")
    if host == "localhost" or host in hosts:
        return
    raise PolicyError("local.url is not loopback and its host is not listed in local.hosts")


def parse_overrides(header: str | None) -> set[str]:
    """``x-privacy-gate-allow: email,phone`` as a set of category names."""
    if not header:
        return set()
    return {part.strip() for part in header.split(",") if part.strip()}


def strip_markers(text: str) -> tuple[str, set[str]]:
    """Remove ``[[send-as-is:email]]`` markers and return the categories they named."""
    named: set[str] = set()

    def take(match: re.Match[str]) -> str:
        named.update(match.group(1).split(","))
        return ""

    return _MARKER.sub(take, text), named
