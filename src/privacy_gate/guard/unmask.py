# SPDX-License-Identifier: Apache-2.0
"""Unmasking the reply: put the real values back, locally, for the person (design, section 3.3).

Only surrogates this session's vault issued are replaced, and only exactly: ``<EMAIL_k3x9qa>`` written
differently by the model stays as written. Matching is longest first, so ``192.0.2.12`` is not read as
``192.0.2.1`` followed by ``2``.

Streaming: a surrogate can be split across chunks. `StreamUnmasker` holds back the longest end of what it has
seen that is the start of some surrogate, and releases it when the next chunk shows whether it was one. At the
end of a block, `flush` releases everything. Whatever the split, the stream's output equals `unmask` of the
whole text (a property test checks every split point).

Tool calls: arguments arrive as pieces of JSON. `ToolArgsBuffer` collects them and unmasks the JSON text once
it is complete, writing each value JSON-escaped, so a value with a quote or a backslash keeps the JSON valid.

SSE framing (events, ``data:`` lines) is the provider adapters' job, from P3; this module works on text.
"""

from __future__ import annotations

import json
import re

from .vault import Vault


class _Matcher:
    def __init__(self, vault: Vault) -> None:
        self.vault = vault
        self.version = -1
        self.pattern: re.Pattern[str] | None = None
        self.prefixes: frozenset[str] = frozenset()
        self.longest = 0

    def refresh(self) -> None:
        if self.version == self.vault.version:
            return
        surrogates = sorted(self.vault.surrogates(), key=len, reverse=True)
        self.version = self.vault.version
        self.pattern = re.compile("|".join(map(re.escape, surrogates))) if surrogates else None
        self.prefixes = frozenset(s[:i] for s in surrogates for i in range(1, len(s)))
        self.longest = len(surrogates[0]) if surrogates else 0

    def replace(self, text: str, escape_json: bool = False) -> str:
        self.refresh()
        if self.pattern is None:
            return text

        def value(match: re.Match[str]) -> str:
            real = self.vault.value(match.group())
            if real is None:
                return match.group()
            return json.dumps(real, ensure_ascii=False)[1:-1] if escape_json else real

        return self.pattern.sub(value, text)


def unmask(text: str, vault: Vault) -> str:
    return _Matcher(vault).replace(text)


def unmask_json_text(text: str, vault: Vault) -> str:
    """Unmask JSON text (tool arguments), escaping each value for a JSON string."""
    return _Matcher(vault).replace(text, escape_json=True)


class StreamUnmasker:
    def __init__(self, vault: Vault, escape_json: bool = False) -> None:
        self._matcher = _Matcher(vault)
        self._escape = escape_json
        self._buffer = ""

    def feed(self, chunk: str) -> str:
        m = self._matcher
        m.refresh()
        buffer = self._buffer + chunk
        if m.pattern is None:
            self._buffer = ""
            return buffer
        hold = 0
        for size in range(min(m.longest - 1, len(buffer)), 0, -1):
            if buffer[-size:] in m.prefixes:
                hold = size
                break
        cut = len(buffer) - hold
        out: list[str] = []
        pos = 0
        for match in m.pattern.finditer(buffer):
            if match.end() > cut:
                # A whole surrogate that runs into the held part waits with it, so the longer one can win.
                cut = min(cut, match.start())
                break
            out.append(buffer[pos : match.start()])
            out.append(m.replace(match.group(), self._escape))
            pos = match.end()
        out.append(buffer[pos:cut])
        self._buffer = buffer[cut:]
        return "".join(out)

    def flush(self) -> str:
        rest, self._buffer = self._buffer, ""
        return self._matcher.replace(rest, self._escape)

    @property
    def held(self) -> int:
        return len(self._buffer)


class ToolArgsBuffer:
    """Collects a tool call's argument pieces and unmasks them once complete."""

    def __init__(self, vault: Vault) -> None:
        self._vault = vault
        self._parts: list[str] = []

    def feed(self, piece: str) -> None:
        self._parts.append(piece)

    def finish(self) -> str:
        text, self._parts = "".join(self._parts), []
        return unmask_json_text(text, self._vault)
