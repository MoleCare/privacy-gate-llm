# SPDX-License-Identifier: Apache-2.0
"""The live monitor's checks, against fakes: a fake `privacy-gate` process, a fake Gate, and a local HTTP server
standing in for the sidecar and the demo pages. No network beyond 127.0.0.1, no model."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import threading
import types
import unittest
from collections.abc import Callable
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("live_smoke", HERE.parent / "scripts" / "live_smoke.py")
assert spec and spec.loader
live_smoke = importlib.util.module_from_spec(spec)
sys.modules["live_smoke"] = live_smoke  # its dataclass looks itself up there
spec.loader.exec_module(live_smoke)

EXPECTED = [hold for _, hold in live_smoke.SAMPLES]
RIGHT = [{"hold": hold, "score": 5.0 if hold else -5.0} for hold in EXPECTED]
HEAD = json.dumps({"weights": [0.0] * 1024, "bias": 0.0, "threshold": 0.12}).encode()


def quiet(fn: Callable[[], Any]) -> str:
    out = io.StringIO()
    with redirect_stdout(out):
        fn()
    return out.getvalue()


class JudgeTests(unittest.TestCase):
    def test_samples_are_balanced_and_obviously_invented(self) -> None:
        self.assertEqual(EXPECTED.count(True), EXPECTED.count(False))
        for text, hold in live_smoke.SAMPLES:
            if hold:
                self.assertRegex(text, r"Synthetic|Fictional|Placeholder")

    def test_right_verdicts_pass(self) -> None:
        lines: list[str] = []
        live_smoke.report(live_smoke.judge(RIGHT), out=lines.append)
        self.assertEqual(len(lines), len(RIGHT))
        self.assertTrue(all(line.startswith("ok") for line in lines))

    def test_one_wrong_verdict_fails_and_is_named(self) -> None:
        decisions = [dict(d) for d in RIGHT]
        decisions[-1]["hold"] = False  # a sensitive sample let through
        lines: list[str] = []
        with self.assertRaisesRegex(live_smoke.CheckFailed, "1 of 6"):
            live_smoke.report(live_smoke.judge(decisions), out=lines.append)
        self.assertTrue(lines[-1].startswith("FAIL send (want hold)"))

    def test_wrong_count_and_malformed_decisions_fail(self) -> None:
        with self.assertRaisesRegex(live_smoke.CheckFailed, "5 decisions for 6"):
            live_smoke.judge(RIGHT[:-1])
        with self.assertRaisesRegex(live_smoke.CheckFailed, "malformed"):
            live_smoke.judge([{"hold": "yes"}, *RIGHT[1:]])


class FakeRun:
    """Stands in for subprocess.run: `info` and `check` answer from the given values."""

    def __init__(self, info: tuple[int, str], check: tuple[int, str]) -> None:
        self.answers = {"info": info, "check": check}
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> types.SimpleNamespace:
        self.calls.append(command)
        code, stdout = self.answers[command[1]]
        return types.SimpleNamespace(
            returncode=code, stdout=stdout, stderr="cannot score: connection refused"
        )


INFO = json.dumps({"head_sha256": "ab" * 32, "threshold": 0.1209, "dimensions": 1024})
CHECK = "\n".join(json.dumps(d) for d in RIGHT) + "\n"


class CliTests(unittest.TestCase):
    def test_pass_and_the_url_reaches_the_command(self) -> None:
        run = FakeRun((0, INFO), (0, CHECK))
        quiet(lambda: live_smoke.check_cli("privacy-gate", "http://127.0.0.1:11434", run=run))
        self.assertEqual(
            run.calls[1], ["privacy-gate", "check", "--stdin", "--json", "--url", "http://127.0.0.1:11434"]
        )

    def test_info_failing_or_describing_another_head_fails(self) -> None:
        with self.assertRaisesRegex(live_smoke.CheckFailed, "info` exited 1"):
            live_smoke.check_cli("pg", None, run=FakeRun((1, ""), (0, CHECK)))
        with self.assertRaisesRegex(live_smoke.CheckFailed, "no JSON"):
            live_smoke.check_cli("pg", None, run=FakeRun((0, "usage: ..."), (0, CHECK)))
        with self.assertRaisesRegex(live_smoke.CheckFailed, "unexpected head"):
            live_smoke.check_cli("pg", None, run=FakeRun((0, json.dumps({"dimensions": 768})), (0, CHECK)))

    def test_failing_closed_is_a_failure_here(self) -> None:
        # Exit 1 is the CLI refusing to score (no encoder): right for a user, a broken install for the monitor.
        with self.assertRaisesRegex(live_smoke.CheckFailed, "check` exited 1"):
            quiet(lambda: live_smoke.check_cli("pg", None, run=FakeRun((0, INFO), (1, ""))))

    def test_output_that_is_not_json_fails(self) -> None:
        with self.assertRaisesRegex(live_smoke.CheckFailed, "not JSON"):
            quiet(lambda: live_smoke.check_cli("pg", None, run=FakeRun((0, INFO), (0, "hold score=1\n"))))


class ApiTests(unittest.TestCase):
    def test_gate_load_and_decide_many(self) -> None:
        decisions = [types.SimpleNamespace(hold=d["hold"], score=d["score"]) for d in RIGHT]
        gate = mock.Mock(decide_many=mock.Mock(return_value=decisions))
        with mock.patch("privacy_gate.gate.Gate.load", return_value=gate) as load:
            quiet(lambda: live_smoke.check_api("http://127.0.0.1:11434"))
        load.assert_called_once_with(url="http://127.0.0.1:11434")
        gate.decide_many.assert_called_once_with([text for text, _ in live_smoke.SAMPLES])


class Site:
    """A local HTTP server answering from a dict of path -> (status, body)."""

    def __init__(self, routes: dict[str, tuple[int, bytes]]) -> None:
        self.routes = routes
        routes_of = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self) -> None:
                status, body = routes_of.routes.get(self.path, (404, b"{}"))
                self.send_response(status)
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                self._answer()

            def do_POST(self) -> None:  # noqa: N802
                self.rfile.read(int(self.headers["content-length"]))
                self._answer()

            def log_message(self, *args: object) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


HEALTH = (200, json.dumps({"ok": True, "backend": "local", "head_sha256": "ab" * 32}).encode())


class SidecarTests(unittest.TestCase):
    def site(self, routes: dict[str, tuple[int, bytes]]) -> Site:
        site = Site(routes)
        self.addCleanup(site.close)
        return site

    def test_pass(self) -> None:
        site = self.site({"/health": HEALTH, "/check": (200, json.dumps({"results": RIGHT}).encode())})
        self.assertIn("pass", quiet(lambda: live_smoke.main(["sidecar", "--url", site.url])))

    def test_fail_closed_503_fails(self) -> None:
        site = self.site({"/health": HEALTH, "/check": (503, b'{"hold": true}')})
        with self.assertRaisesRegex(live_smoke.CheckFailed, "HTTP 503"):
            quiet(lambda: live_smoke.check_sidecar(site.url))

    def test_unexpected_shape_and_non_json_fail(self) -> None:
        site = self.site({"/health": HEALTH, "/check": (200, json.dumps(RIGHT[0]).encode())})
        with self.assertRaisesRegex(live_smoke.CheckFailed, "unexpected shape"):
            quiet(lambda: live_smoke.check_sidecar(site.url))
        site.routes["/check"] = (200, b"<html>")
        with self.assertRaisesRegex(live_smoke.CheckFailed, "not JSON"):
            quiet(lambda: live_smoke.check_sidecar(site.url))

    def test_health_that_never_turns_ok_times_out(self) -> None:
        site = self.site({"/health": (200, b'{"ok": false}')})
        ticks = iter([0.0, 10.0, 70.0])
        with self.assertRaisesRegex(live_smoke.CheckFailed, "not ready after 60s"):
            live_smoke.wait_healthy(site.url, 60, sleep=lambda _: None, clock=lambda: next(ticks))

    def test_nothing_listening_fails_after_the_deadline(self) -> None:
        with self.assertRaisesRegex(live_smoke.CheckFailed, "not ready"):
            live_smoke.wait_healthy("http://127.0.0.1:9", 0)


class PageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.site = Site(
            {"/": (200, b"<h1>Must this stay on this machine?</h1>"), "/head-v1.json": (200, HEAD)}
        )
        self.addCleanup(self.site.close)
        self.head_file = Path(self.enterContext(tempfile.TemporaryDirectory())) / "head-v1.json"
        self.head_file.write_bytes(HEAD)

    def page(self, expect: str = "Must this stay", head: str | None = "/head-v1.json") -> str:
        return quiet(
            lambda: live_smoke.check_page(
                self.site.url + "/",
                [expect],
                self.site.url + head if head else None,
                str(self.head_file),
            )
        )

    def test_pass_with_the_repository_head(self) -> None:
        self.assertIn(hashlib.sha256(HEAD).hexdigest()[:16], self.page())

    def test_page_only(self) -> None:
        self.assertNotIn("head:", self.page(head=None))

    def test_missing_page_marker_or_head_fails(self) -> None:
        with self.assertRaisesRegex(live_smoke.CheckFailed, "does not contain"):
            self.page(expect="something else")
        with self.assertRaisesRegex(live_smoke.CheckFailed, "HTTP 404"):
            self.page(head="/head-v0.json")
        self.site.routes["/"] = (404, b"")
        with self.assertRaisesRegex(live_smoke.CheckFailed, "HTTP 404"):
            self.page()

    def test_a_different_or_broken_head_fails(self) -> None:
        self.head_file.write_bytes(b"{}")
        with self.assertRaisesRegex(live_smoke.CheckFailed, "the repository has"):
            self.page()
        self.site.routes["/head-v1.json"] = (200, b'{"weights": [1, 2]}')
        with self.assertRaisesRegex(live_smoke.CheckFailed, "1024-weight"):
            self.page()
        self.site.routes["/head-v1.json"] = (200, b"<html>")
        with self.assertRaisesRegex(live_smoke.CheckFailed, "not JSON"):
            self.page()

    def test_main_turns_a_failure_into_exit_1_and_a_workflow_error(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            code = live_smoke.main(["page", "--url", self.site.url + "/", "--expect", "absent"])
        self.assertEqual(code, 1)
        self.assertIn("::error title=live check failed (page)::", out.getvalue())


if __name__ == "__main__":
    unittest.main()
