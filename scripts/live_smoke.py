#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests against what users actually install and open: the PyPI package, the GHCR image, the demo.

The live monitor (.github/workflows/monitor.yml) runs one subcommand per step, so each check passes or fails
on its own line in the run summary:

    python scripts/live_smoke.py cli  --bin /tmp/fresh/bin/privacy-gate --url http://127.0.0.1:11434
    /tmp/fresh/bin/python -I scripts/live_smoke.py api --url http://127.0.0.1:11434
    python scripts/live_smoke.py sidecar --url http://127.0.0.1:8231
    python scripts/live_smoke.py page --url https://example.org/ --expect "Must this stay" \
        --head-url https://example.org/head-v1.json --head-file model/head-v1.json

Standard library only, so it runs inside a fresh venv that holds nothing but the released package; `api`
imports `privacy_gate` from that venv, never from this checkout. Exit 0 when the check passes, 1 when it fails,
with one line per sample saying why.

The samples are invented and say so: no real person, patient or credential. Their scores sit several points
from the threshold on bge-m3 through Ollama and through sentence-transformers, so a failure means the product
broke, not that a borderline example drifted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

# (text, must the gate hold it?)
SAMPLES: tuple[tuple[str, bool], ...] = (
    ("Explain the difference between a list and a tuple in Python.", False),
    ("Please refactor this function to use a list comprehension.", False),
    ("Add a created_at column to the orders table.", False),
    (
        "Synthetic test patient Jane Example was diagnosed with type 2 diabetes and starts metformin next week.",
        True,
    ),
    (
        "Fictional patient John Placeholder, born 01.01.1900, is HIV positive and on antiretroviral therapy.",
        True,
    ),
    (
        "Placeholder test: my colleague Erika Mustermann told me about her depression diagnosis "
        "and her new antidepressant.",
        True,
    ),
)

HEAD_DIMENSIONS = 1024
USER_AGENT = "privacy-gate-live-monitor"


class CheckFailed(Exception):
    """A check ran and the thing it checked is broken."""


@dataclass(frozen=True)
class Verdict:
    text: str
    expected_hold: bool
    hold: bool
    score: float

    @property
    def ok(self) -> bool:
        return self.hold == self.expected_hold


def _label(hold: bool) -> str:
    return "hold" if hold else "send"


def judge(
    decisions: Sequence[dict[str, Any]], samples: Sequence[tuple[str, bool]] = SAMPLES
) -> list[Verdict]:
    """Pair each decision (a dict with `hold` and `score`) with its sample; raise if the shape is wrong."""
    if len(decisions) != len(samples):
        raise CheckFailed(f"{len(decisions)} decisions for {len(samples)} samples")
    verdicts = []
    for (text, expected), decision in zip(samples, decisions):
        if not isinstance(decision, dict) or not isinstance(decision.get("hold"), bool):
            raise CheckFailed(f"malformed decision {decision!r}")
        verdicts.append(Verdict(text, expected, decision["hold"], float(decision.get("score", "nan"))))
    return verdicts


def report(verdicts: Sequence[Verdict], out: Callable[[str], None] = print) -> None:
    """Print one line per sample; raise if any verdict is wrong."""
    for v in verdicts:
        mark = "ok  " if v.ok else "FAIL"
        out(f"{mark} {_label(v.hold)} (want {_label(v.expected_hold)}) score={v.score:+.3f}  {v.text[:70]}")
    wrong = [v for v in verdicts if not v.ok]
    if wrong:
        raise CheckFailed(f"{len(wrong)} of {len(verdicts)} samples got the wrong verdict")


def _texts(samples: Sequence[tuple[str, bool]]) -> list[str]:
    return [text for text, _ in samples]


# --- the checks -----------------------------------------------------------------------------------------------


def check_cli(binary: str, url: str | None, run: Callable[..., Any] = subprocess.run) -> None:
    """`privacy-gate info` loads the bundled head; `privacy-gate check --stdin --json` scores every sample."""
    info = run([binary, "info"], capture_output=True, text=True, timeout=120)
    if info.returncode != 0:
        raise CheckFailed(f"`privacy-gate info` exited {info.returncode}: {info.stderr.strip()[:300]}")
    try:
        meta = json.loads(info.stdout)
    except json.JSONDecodeError as error:
        raise CheckFailed(f"`privacy-gate info` printed no JSON: {error}") from error
    if meta.get("dimensions") != HEAD_DIMENSIONS or not meta.get("head_sha256"):
        raise CheckFailed(f"`privacy-gate info` describes an unexpected head: {meta}")
    print(f"info: head {meta['head_sha256'][:16]}, threshold {meta.get('threshold')}")

    command = [binary, "check", "--stdin", "--json"]
    if url:
        command += ["--url", url]
    result = run(
        command, input="\n".join(_texts(SAMPLES)) + "\n", capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        raise CheckFailed(f"`privacy-gate check` exited {result.returncode}: {result.stderr.strip()[:300]}")
    try:
        decisions = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    except json.JSONDecodeError as error:
        raise CheckFailed(f"`privacy-gate check --json` printed a line that is not JSON: {error}") from error
    report(judge(decisions))


def check_api(url: str | None) -> None:
    """The documented Python entry point: `Gate.load(url=...)` then `decide_many`."""
    from privacy_gate.gate import Gate  # the installed package, imported here so the module stays stdlib-only

    gate = Gate.load(url=url) if url else Gate.load()
    decisions = gate.decide_many(_texts(SAMPLES))
    report(judge([{"hold": d.hold, "score": d.score} for d in decisions]))


def _request(url: str, data: bytes | None = None, timeout: float = 60) -> tuple[int, bytes]:
    headers = {"user-agent": USER_AGENT}
    if data is not None:
        headers["content-type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()
    except (urllib.error.URLError, OSError) as error:
        raise CheckFailed(f"{url}: {error}") from error


def wait_healthy(
    url: str,
    deadline_s: float,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Poll GET /health until it answers 200 with ok=true, or the deadline passes."""
    end = clock() + deadline_s
    last = "no answer"
    while True:
        try:
            status, body = _request(url.rstrip("/") + "/health", timeout=10)
            if status == 200:
                health: dict[str, Any] = json.loads(body)
                if health.get("ok") is True:
                    return health
            last = f"HTTP {status}: {body[:200]!r}"
        except (CheckFailed, json.JSONDecodeError) as error:
            last = str(error)
        if clock() >= end:
            raise CheckFailed(f"/health not ready after {deadline_s:.0f}s ({last})")
        sleep(5)


def check_sidecar(url: str, wait_s: float = 0) -> None:
    """The HTTP sidecar (the Docker image's entry point): /health, then POST /check with every sample."""
    health = wait_healthy(url, wait_s)
    print(f"health: backend {health.get('backend')}, head {str(health.get('head_sha256'))[:16]}")
    payload = json.dumps({"texts": _texts(SAMPLES)}).encode()
    status, body = _request(url.rstrip("/") + "/check", data=payload, timeout=600)
    if status != 200:
        raise CheckFailed(f"POST /check answered HTTP {status}: {body[:300]!r}")
    try:
        answer = json.loads(body)
    except json.JSONDecodeError as error:
        raise CheckFailed(f"POST /check answered something that is not JSON: {error}") from error
    decisions = answer.get("results") if isinstance(answer, dict) else None
    if not isinstance(decisions, list):
        raise CheckFailed(f"POST /check answered an unexpected shape: {body[:300]!r}")
    report(judge(decisions))


def check_page(url: str, expect: Sequence[str], head_url: str | None, head_file: str | None) -> None:
    """A demo page answers 200 with the expected text, and serves a 1024-weight head (the repository's, if given)."""
    status, body = _request(url)
    if status != 200:
        raise CheckFailed(f"{url} answered HTTP {status}")
    page = body.decode("utf-8", "replace")
    missing = [marker for marker in expect if marker not in page]
    if missing:
        raise CheckFailed(f"{url} does not contain {missing}")
    print(f"page: {url} 200, {len(body)} bytes, has {list(expect)}")
    if not head_url:
        return
    status, raw = _request(head_url)
    if status != 200:
        raise CheckFailed(f"{head_url} answered HTTP {status}")
    try:
        head = json.loads(raw)
    except json.JSONDecodeError as error:
        raise CheckFailed(f"{head_url} is not JSON: {error}") from error
    if not isinstance(head, dict) or len(head.get("weights", [])) != HEAD_DIMENSIONS:
        raise CheckFailed(f"{head_url} is not a {HEAD_DIMENSIONS}-weight head")
    served = hashlib.sha256(raw).hexdigest()
    if head_file:
        with open(head_file, "rb") as handle:
            want = hashlib.sha256(handle.read()).hexdigest()
        if served != want:
            raise CheckFailed(f"{head_url} serves head {served[:16]}, the repository has {want[:16]}")
    print(f"head: {head_url} sha256 {served[:16]}")


# --- command line ---------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    cli = sub.add_parser("cli", help="the installed command line")
    cli.add_argument("--bin", default="privacy-gate")
    cli.add_argument("--url", help="the Ollama endpoint (default: the package's)")
    api = sub.add_parser("api", help="the installed Python API")
    api.add_argument("--url", help="the Ollama endpoint (default: the package's)")
    sidecar = sub.add_parser("sidecar", help="a running HTTP sidecar")
    sidecar.add_argument("--url", default="http://127.0.0.1:8231")
    sidecar.add_argument("--wait", type=float, default=0, help="seconds to wait for /health")
    page = sub.add_parser("page", help="a demo page and its head")
    page.add_argument("--url", required=True)
    page.add_argument("--expect", action="append", default=[], help="text the page must contain (repeatable)")
    page.add_argument("--head-url")
    page.add_argument("--head-file", help="the head the page must serve, byte for byte")
    args = parser.parse_args(argv)

    try:
        if args.command == "cli":
            check_cli(args.bin, args.url)
        elif args.command == "api":
            check_api(args.url)
        elif args.command == "sidecar":
            check_sidecar(args.url, args.wait)
        else:
            check_page(args.url, args.expect, args.head_url, args.head_file)
    except CheckFailed as error:
        print(f"::error title=live check failed ({args.command})::{error}")
        return 1
    print(f"{args.command}: pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
