# SPDX-License-Identifier: Apache-2.0
"""The integrations, without their hosts: the LiteLLM guardrail with a fake gate, the Open WebUI filter with a
fake gate, and the Action's diff parsing and file filters. No network, no model, no LiteLLM installed."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeDecision:
    def __init__(self, score: float, threshold: float = 0.12) -> None:
        self.score, self.threshold = score, threshold
        self.hold = score > threshold
        self.margin = score - threshold


class FakeGate:
    """Holds any text containing 'biopsy'; fails when asked to score 'DOWN'."""

    threshold = 0.12

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def decide_many(self, texts: list[str]) -> list[FakeDecision]:
        self.calls.append(list(texts))
        if any("DOWN" in t for t in texts):
            raise ConnectionError("no encoder")
        return [FakeDecision(6.0 if "biopsy" in t else -2.0) for t in texts]


class LiteLLMGuardrailTests(unittest.TestCase):
    def setUp(self) -> None:
        stub = types.ModuleType("litellm.integrations.custom_guardrail")

        class CustomGuardrail:
            def __init__(self, **kwargs):
                pass

        stub.CustomGuardrail = CustomGuardrail
        self._saved = {k: sys.modules.get(k) for k in ("litellm", "litellm.integrations", "litellm.integrations.custom_guardrail")}
        sys.modules["litellm"] = types.ModuleType("litellm")
        sys.modules["litellm.integrations"] = types.ModuleType("litellm.integrations")
        sys.modules["litellm.integrations.custom_guardrail"] = stub
        self.mod = load("pg_guardrail", HERE / "integrations" / "litellm" / "privacy_gate_guardrail.py")

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def run_hook(self, guardrail, data):
        return asyncio.run(guardrail.async_pre_call_hook({}, None, data, "completion"))

    def test_texts_of_reads_messages_parts_and_tool_arguments(self) -> None:
        data = {"messages": [{"role": "system", "content": "be brief"},
                             {"role": "user", "content": [{"type": "text", "text": "hello there"}, {"type": "image_url", "image_url": {}}]},
                             {"role": "assistant", "content": None, "tool_calls": [{"function": {"arguments": '{"q": "biopsy"}'}}]}]}
        self.assertEqual(self.mod.PrivacyGateGuardrail.texts_of(data), ["be brief", "hello there", '{"q": "biopsy"}'])

    def test_route_switches_the_model_and_records_why(self) -> None:
        gate = FakeGate()
        g = self.mod.PrivacyGateGuardrail(action="route", local_model="ollama/local", gate=gate, min_length=5)
        data = {"model": "gpt-4o", "messages": [{"role": "user", "content": "her biopsy is booked for the 20th"}]}
        out = self.run_hook(g, data)
        self.assertEqual(out["model"], "ollama/local")
        self.assertTrue(out["metadata"]["privacy_gate"]["hold"])
        self.assertEqual(out["metadata"]["privacy_gate"]["routed_from"], "gpt-4o")
        clean = self.run_hook(g, {"model": "gpt-4o", "messages": [{"role": "user", "content": "the build is green today"}]})
        self.assertEqual(clean["model"], "gpt-4o")
        self.assertFalse(clean["metadata"]["privacy_gate"]["hold"])

    def test_block_raises(self) -> None:
        g = self.mod.PrivacyGateGuardrail(action="block", gate=FakeGate(), min_length=5)
        with self.assertRaises(PermissionError):
            self.run_hook(g, {"model": "gpt-4o", "messages": [{"role": "user", "content": "her biopsy is booked"}]})

    def test_short_texts_are_not_scored(self) -> None:
        gate = FakeGate()
        g = self.mod.PrivacyGateGuardrail(action="block", gate=gate, min_length=12)
        out = self.run_hook(g, {"model": "m", "messages": [{"role": "user", "content": "biopsy"}]})
        self.assertEqual(gate.calls, [])
        self.assertEqual(out["metadata"]["privacy_gate"]["scored"], 0)

    def test_unreachable_encoder_fails_closed(self) -> None:
        g = self.mod.PrivacyGateGuardrail(action="route", local_model="local", gate=FakeGate(), min_length=1)
        out = self.run_hook(g, {"model": "gpt-4o", "messages": [{"role": "user", "content": "DOWN encoder"}]})
        self.assertEqual(out["model"], "local")
        self.assertIn("failing closed", out["metadata"]["privacy_gate"]["reason"])
        g = self.mod.PrivacyGateGuardrail(action="block", gate=FakeGate(), min_length=1)
        with self.assertRaises(PermissionError):
            self.run_hook(g, {"model": "gpt-4o", "messages": [{"role": "user", "content": "DOWN encoder"}]})

    def test_config_errors(self) -> None:
        with self.assertRaises(ValueError):
            self.mod.PrivacyGateGuardrail(action="route", gate=FakeGate())
        with self.assertRaises(ValueError):
            self.mod.PrivacyGateGuardrail(action="drop", gate=FakeGate())


class OpenWebUIFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import pydantic  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("pydantic is not installed here")
        self.mod = load("pg_filter", HERE / "integrations" / "open-webui" / "privacy_gate_filter.py")

    def test_routes_on_hold_and_leaves_clean_alone(self) -> None:
        f = self.mod.Filter()
        f._gate = FakeGate()
        body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "her biopsy is booked for the 20th"}]}
        out = f.inlet(body)
        self.assertEqual(out["model"], "qwen3-coder:30b")
        self.assertEqual(out["metadata"]["privacy_gate"]["routed_from"], "gpt-4o")
        out = f.inlet({"model": "gpt-4o", "messages": [{"role": "user", "content": "the build is green today"}]})
        self.assertEqual(out["model"], "gpt-4o")

    def test_block_and_fail_closed(self) -> None:
        f = self.mod.Filter()
        f._gate = FakeGate()
        f.valves.action = "block"
        with self.assertRaises(Exception):
            f.inlet({"model": "m", "messages": [{"role": "user", "content": "her biopsy is booked for the 20th"}]})
        with self.assertRaises(Exception):
            f.inlet({"model": "m", "messages": [{"role": "user", "content": "DOWN encoder please"}]})


DIFF = """diff --git a/docs/notes.md b/docs/notes.md
--- a/docs/notes.md
+++ b/docs/notes.md
@@ -3,0 +4,2 @@
+The woman from Tuesday's clinic has a 7mm lesion on her shoulder and a biopsy on the 20th.
+short
@@ -10,2 +12,1 @@
-old line
-old line two
+a replacement line that is long enough to be scored by the gate here
diff --git a/img/a.png b/img/a.png
Binary files differ
diff --git a/new.txt b/new.txt
--- /dev/null
+++ b/new.txt
@@ -0,0 +1 @@
+a brand new file with one added line of ordinary text in it
diff --git a/gone.txt b/gone.txt
--- a/gone.txt
+++ /dev/null
@@ -1 +0,0 @@
-removed
"""


class ActionScanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scan = load("pg_scan", HERE / "action" / "scan.py")

    def test_added_lines_with_numbers(self) -> None:
        lines = self.scan.added_lines(DIFF)
        self.assertEqual([(f, n) for f, n, _ in lines], [("docs/notes.md", 4), ("docs/notes.md", 5), ("docs/notes.md", 12), ("new.txt", 1)])
        self.assertTrue(lines[0][2].startswith("The woman"))

    def test_file_filters(self) -> None:
        self.assertTrue(self.scan.wanted("docs/notes.md", []))
        self.assertFalse(self.scan.wanted("img/a.png", []))
        self.assertFalse(self.scan.wanted("package-lock.json", []))
        self.assertFalse(self.scan.wanted("dist/app.min.js", []))
        self.assertTrue(self.scan.wanted("src/x.py", ["src/*.py"]))
        self.assertFalse(self.scan.wanted("docs/x.md", ["src/*.py"]))


if __name__ == "__main__":
    unittest.main()
