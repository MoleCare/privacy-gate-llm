# SPDX-License-Identifier: Apache-2.0
"""The decision object, and an audit writer that consumes it (design, section 8; architecture review R4).

`Decision` is what the engine returns for every request: the action, which rules fired (with their old ids),
how many findings per category, where in the body (as a path), the ruleset hash, detector timings and whether
it failed closed. It never holds a value. A caller that embeds the engine (llm-harness) maps it to its own
headers and audit; the engine writes no file of its own.

`AuditWriter` is one such consumer, for the standalone guard. It appends one JSON line per decision to a file
created with mode 0600, rotates at 32 MB keeping 4 old files, and can hash values for correlation under a
local key (a 0600 key file): HMAC-SHA256 truncated to 8 hex characters, enough to see that the same value
came back and too short to search for one.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MAX_BYTES = 32 * 1024 * 1024
KEEP = 4


@dataclass
class Decision:
    action: str  # allow | mask | block | local
    reason: str = ""
    would_action: str | None = None  # in shadow mode, what enforce would have done
    mode: str = "enforce"
    provider: str = ""
    wire: str = ""
    profile: str = ""
    rules: list[str] = field(default_factory=list)  # rule ids and their aliases
    categories: dict[str, int] = field(default_factory=dict)
    paths: list[str] = field(default_factory=list)
    value_hashes: list[str] = field(default_factory=list)
    override: list[str] = field(default_factory=list)
    refused_override: list[str] = field(default_factory=list)
    shadow_rules: list[str] = field(
        default_factory=list
    )  # fired but not enforced (shadow, combination in lite)
    ruleset_sha256: str = ""
    detectors: dict[str, float] = field(default_factory=dict)
    guard_ms: float = 0.0
    fail_closed: bool = False
    decision_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    ts: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    @property
    def sends(self) -> bool:
        """True when the request may go to the provider (allowed or masked)."""
        return self.action in ("allow", "mask")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def audit_record(self) -> dict[str, Any]:
        """The audit line of design section 8."""
        return {
            "ts": self.ts,
            "decision_id": self.decision_id,
            "provider": self.provider,
            "wire": self.wire,
            "mode": self.mode,
            "action": self.action,
            "would_action": self.would_action,
            "rules": self.rules,
            "categories": self.categories,
            "paths": self.paths,
            "value_hashes": self.value_hashes,
            "override": self.override,
            "refused_override": self.refused_override,
            "shadow_rules": self.shadow_rules,
            "ruleset": self.ruleset_sha256[:8],
            "detectors": {k: round(v, 3) for k, v in self.detectors.items()},
            "guard_ms": round(self.guard_ms, 3),
            "fail_closed": self.fail_closed,
            "reason": self.reason,
        }


def _open_private(path: Path, flags: int) -> int:
    return os.open(path, flags | getattr(os, "O_NOFOLLOW", 0), 0o600)


class AuditWriter:
    def __init__(
        self,
        path: str | Path,
        key_path: str | Path | None = None,
        max_bytes: int = MAX_BYTES,
        keep: int = KEEP,
    ) -> None:
        self.path = Path(path).expanduser()
        self.key_path = Path(key_path).expanduser() if key_path else self.path.with_name("audit.key")
        self.max_bytes = max_bytes
        self.keep = keep
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._key = self._load_key()

    def _load_key(self) -> bytes:
        try:
            fd = _open_private(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            if self.key_path.stat().st_mode & 0o077:
                raise PermissionError(
                    f"{self.key_path} is readable by others; it must be mode 0600"
                ) from None
            return bytes.fromhex(self.key_path.read_text().strip())
        key = os.urandom(32)
        with os.fdopen(fd, "w") as handle:
            handle.write(key.hex() + "\n")
        return key

    def hash_value(self, category: str, value: str) -> str:
        """Keyed, truncated hash of a value: for the `hasher` argument of `Guard.protect`."""
        return hmac.new(self._key, f"{category}\0{value}".encode(), hashlib.sha256).hexdigest()[:8]

    def write(self, decision: Decision) -> None:
        line = (json.dumps(decision.audit_record(), ensure_ascii=True, separators=(",", ":")) + "\n").encode()
        with self._lock:
            try:
                size = self.path.stat().st_size
            except FileNotFoundError:
                size = 0
            if size and size + len(line) > self.max_bytes:
                self._rotate()
            fd = _open_private(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)

    def _rotate(self) -> None:
        oldest = self.path.with_name(f"{self.path.name}.{self.keep}")
        oldest.unlink(missing_ok=True)
        for n in range(self.keep - 1, 0, -1):
            older = self.path.with_name(f"{self.path.name}.{n}")
            if older.exists():
                older.rename(self.path.with_name(f"{self.path.name}.{n + 1}"))
        self.path.rename(self.path.with_name(f"{self.path.name}.1"))
