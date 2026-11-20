# SPDX-License-Identifier: Apache-2.0
"""The session vault: which surrogate stands for which value, held in memory only (design, section 5.2).

Each session has its own random key. A surrogate is derived from HMAC-SHA256(key, category, value), so the
same value gets the same surrogate for the life of a session (the model sees a consistent placeholder) and a
different one in the next session (nothing links two sessions). The vault keeps both directions so a reply can
be unmasked; only surrogates this vault issued are ever replaced.

Surrogate shapes keep the text readable to the model and are chosen so they cannot be real:

| kind | shape |
|---|---|
| email | ``m-<tag>@masked.invalid`` (RFC 2606 reserves .invalid) |
| phone | ``+44 7700 900ddd`` (Ofcom's drama range); after collisions, ``<PHONE_tag>`` |
| private_ip | IPv4 in the RFC 5737 documentation ranges; IPv6 in ``2001:db8::/32`` (RFC 3849) |
| home_path | the user segment becomes ``u-<tag>`` |
| tag | ``<LABEL_tag>``, e.g. ``<IBAN_k3x9qa>`` |

A tag is 6 base-32 characters. If a surrogate is already taken by another value, the HMAC is taken again
with a counter until it is free.

Sessions are bound to the caller (architecture review R20): the store names a vault by a keyed hash of the
caller's credential and the session id, so one caller choosing another's session id gets an empty vault, not
the other caller's values. Vaults expire after 12 hours idle.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import threading
import time
from collections.abc import Callable

IDLE_SECONDS = 12 * 3600
_MAX_BUMPS = 64


class Vault:
    def __init__(self, key: bytes | None = None) -> None:
        self._key = key or os.urandom(32)
        self._by_value: dict[tuple[str, str], str] = {}
        self._by_surrogate: dict[str, str] = {}
        #: Bumped on every new surrogate, so unmaskers can cache their matcher.
        self.version = 0
        self.lock = threading.Lock()

    def __repr__(self) -> str:  # never show values
        return f"Vault({len(self._by_surrogate)} surrogates)"

    def _mac(self, category: str, value: str, bump: int) -> bytes:
        message = f"{category}\0{value}\0{bump}".encode()
        return hmac.new(self._key, message, hashlib.sha256).digest()

    def surrogate(self, category: str, value: str, kind: str, label: str) -> str:
        with self.lock:
            known = self._by_value.get((category, value))
            if known is not None:
                return known
            for bump in range(_MAX_BUMPS):
                candidate = make_surrogate(self._mac(category, value, bump), kind, label, value, fallback=False)
                if candidate is not None and candidate not in self._by_surrogate:
                    break
            else:
                bump = _MAX_BUMPS
                while True:  # tags: 30 bits each, so this ends almost at once
                    candidate = make_surrogate(self._mac(category, value, bump), kind, label, value, fallback=True)
                    if candidate is not None and candidate not in self._by_surrogate:
                        break
                    bump += 1
            self._by_value[(category, value)] = candidate
            self._by_surrogate[candidate] = value
            self.version += 1
            return candidate

    def value(self, surrogate: str) -> str | None:
        return self._by_surrogate.get(surrogate)

    def surrogates(self) -> list[str]:
        with self.lock:
            return list(self._by_surrogate)

    def issued(self, text: str) -> bool:
        return text in self._by_surrogate


def tag_of(mac: bytes) -> str:
    return base64.b32encode(mac[:5]).decode().lower()[:6]


def make_surrogate(mac: bytes, kind: str, label: str, value: str, fallback: bool) -> str | None:
    tag = tag_of(mac)
    number = int.from_bytes(mac[5:13], "big")
    if fallback and kind in ("phone", "private_ip"):
        return f"<{label}_{tag}>"
    if kind == "email":
        return f"m-{tag}@masked.invalid"
    if kind == "phone":
        return f"+44 7700 900{number % 1000:03d}"
    if kind == "private_ip":
        if ":" in value:
            return f"2001:db8::{(number >> 16) % 0xFFFF + 1:x}:{number % 0xFFFF + 1:x}"
        network = ("192.0.2", "198.51.100", "203.0.113")[number % 3]
        return f"{network}.{(number >> 8) % 254 + 1}"
    if kind == "home_path":
        return f"u-{tag}"
    return f"<{label}_{tag}>"


class SessionStore:
    """Vaults by (caller, session), each expiring after `idle_seconds` without use. The clock is injectable."""

    def __init__(self, idle_seconds: float = IDLE_SECONDS, clock: Callable[[], float] = time.monotonic) -> None:
        self._key = os.urandom(32)
        self._idle = idle_seconds
        self._clock = clock
        self._vaults: dict[bytes, tuple[Vault, float]] = {}
        self._lock = threading.Lock()

    def _name(self, caller: str, session: str) -> bytes:
        return hmac.new(self._key, f"{caller}\0{session}".encode(), hashlib.sha256).digest()

    def vault(self, caller: str = "", session: str = "default") -> Vault:
        now = self._clock()
        name = self._name(caller, session)
        with self._lock:
            for key, (_, used) in list(self._vaults.items()):
                if now - used > self._idle:
                    del self._vaults[key]
            entry = self._vaults.get(name)
            vault = entry[0] if entry else Vault()
            self._vaults[name] = (vault, now)
            return vault

    def __len__(self) -> int:
        return len(self._vaults)
