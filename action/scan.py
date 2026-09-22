#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The GitHub Action's scanner: the lines a change adds, one at a time, through the gate.

Reads the unified diff between PG_BASE and HEAD, keeps the added lines of text files that are at least
PG_MIN_LENGTH characters, scores them in batches, writes one `::error file=..,line=..` annotation per hold,
a Markdown summary, the `holds` and `scanned` outputs, and exits 1 when anything was held and
PG_FAIL_ON_HOLD is true. When the encoder cannot be reached it fails the job: a scanner that cannot scan must
not pass.

The lines themselves are never printed in full: the annotation shows the score and the first 60 characters,
because the log of a public repository is public too.
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".gz", ".tar", ".whl", ".pyc",
                 ".woff", ".woff2", ".ttf", ".otf", ".mp3", ".mp4", ".wav", ".onnx", ".safetensors", ".bin", ".lock",
                 ".svg", ".min.js", ".min.css", ".map"}
SKIP_NAMES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "uv.lock", "Cargo.lock", "go.sum"}
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def added_lines(diff: str) -> list[tuple[str, int, str]]:
    """(file, line number in the new file, text) for every added line in a unified diff."""
    out, path, line = [], None, 0
    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            target = raw[4:].strip()
            path = None if target == "/dev/null" else target[2:] if target.startswith("b/") else target
            continue
        if raw.startswith("--- "):
            continue
        m = HUNK.match(raw)
        if m:
            line = int(m.group(1))
            continue
        if path is None:
            continue
        if raw.startswith("+"):
            out.append((path, line, raw[1:]))
            line += 1
        elif raw.startswith("-") or raw.startswith("\\"):
            continue
        else:
            line += 1
    return out


def is_text_file(path: str) -> bool:
    p = Path(path)
    name = p.name
    if name in SKIP_NAMES:
        return False
    return not any(name.endswith(s) for s in SKIP_SUFFIXES)


def wanted(path: str, patterns: list[str]) -> bool:
    return is_text_file(path) and (not patterns or any(fnmatch.fnmatch(path, pat) for pat in patterns))


def git_diff(base: str) -> str:
    if base:
        rev = f"{base}...HEAD"
    else:
        rev = "HEAD~1...HEAD"
    return subprocess.run(["git", "diff", "--unified=0", "--no-color", "--diff-filter=AM", rev],
                          capture_output=True, text=True, errors="replace", check=True).stdout


def main() -> int:
    from privacy_gate import backends
    from privacy_gate.gate import Gate

    backend = os.environ.get("PG_BACKEND", "local")
    url = os.environ.get("PG_URL") or None
    base = os.environ.get("PG_BASE", "")
    patterns = os.environ.get("PG_PATHS", "").split()
    min_length = int(os.environ.get("PG_MIN_LENGTH", "30") or 30)
    fail_on_hold = os.environ.get("PG_FAIL_ON_HOLD", "true").lower() != "false"
    threshold = os.environ.get("PG_THRESHOLD", "")

    try:
        diff = git_diff(base)
    except subprocess.CalledProcessError as error:
        print(f"::error::privacy-gate: git diff failed: {error.stderr[:200]}")
        return 1
    candidates = [(f, n, t.strip()) for f, n, t in added_lines(diff) if wanted(f, patterns) and len(t.strip()) >= min_length]
    seen, lines = set(), []
    for item in candidates:
        if item[2] not in seen:
            seen.add(item[2])
            lines.append(item)
    outputs = Path(os.environ.get("GITHUB_OUTPUT", "/dev/null"))
    summary = Path(os.environ.get("GITHUB_STEP_SUMMARY", "/dev/null"))
    if not lines:
        print("privacy-gate: no added lines to score")
        with outputs.open("a") as f:
            f.write("holds=0\nscanned=0\n")
        return 0

    gate = Gate.load(embedder=backends.make_embedder(backend, url))
    if threshold:
        gate.threshold = float(threshold)
    holds = []
    try:
        for start in range(0, len(lines), 32):
            batch = lines[start:start + 32]
            for (path, number, text), decision in zip(batch, gate.decide_many([t for _, _, t in batch])):
                if decision.hold:
                    holds.append((path, number, text, decision))
    except Exception as error:  # noqa: BLE001 - a scanner that cannot scan must not pass
        print(f"::error::privacy-gate: cannot score ({type(error).__name__}: {str(error)[:160]}); failing closed")
        return 1

    by_file: dict[str, list] = defaultdict(list)
    for path, number, text, decision in holds:
        by_file[path].append((number, text, decision))
        preview = text[:60] + ("…" if len(text) > 60 else "")
        print(f"::error file={path},line={number},title=privacy-gate: must this stay?::score {decision.score:.2f} "
              f"(threshold {decision.threshold:.2f}): {preview}")
    with outputs.open("a") as f:
        f.write(f"holds={len(holds)}\nscanned={len(lines)}\n")
    with summary.open("a") as f:
        f.write(f"## privacy-gate\n\n{len(lines)} added lines scored, **{len(holds)} held**.\n\n")
        for path, items in by_file.items():
            f.write(f"- `{path}`: " + ", ".join(f"line {n} (score {d.score:.2f})" for n, _, d in items) + "\n")
        if holds:
            f.write("\nA hold means the line reads like health data, a credential or personal data written as prose. "
                    "If it is an invented fixture, say so in the file and in the pull request; see the project's SECURITY.md.\n")
    print(f"privacy-gate: {len(lines)} lines scored, {len(holds)} held")
    return 1 if holds and fail_on_hold else 0


if __name__ == "__main__":
    sys.exit(main())
