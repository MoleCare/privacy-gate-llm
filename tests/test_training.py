# SPDX-License-Identifier: Apache-2.0
"""The maintainer tools in training/ that had no unit tests of their own: the chat client the calibration
and evaluation runs use, and the `training.head` command that cross-validates and exports a head. No
network and no model: the transport is a fake, the embeddings are four synthetic dimensions."""

from __future__ import annotations

import io
import json
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from unittest import mock

from privacy_gate.head import Model
from training import chat, head


class ChatTest(unittest.TestCase):
    def test_greedy_payload_and_reply(self) -> None:
        body = {"message": {"content": "HEALTH"}, "eval_count": 2}
        with mock.patch.object(chat, "_post", return_value=body) as post:
            reply = chat.chat("m", [{"role": "user", "content": "x"}], url="http://fake.test", predict=4)
        url, path, payload, _timeout = post.call_args.args
        self.assertEqual((url, path), ("http://fake.test", "/api/chat"))
        self.assertEqual(payload["model"], "m")
        self.assertIs(payload["stream"], False)
        self.assertIs(payload["think"], False)
        self.assertEqual(payload["options"], {"temperature": 0, "seed": 0, "num_predict": 4})
        # num_ctx is never sent: it would reload the model under a second cache entry on a shared box.
        self.assertNotIn("num_ctx", payload["options"])
        self.assertNotIn("format", payload)
        self.assertNotIn("logprobs", payload)
        self.assertEqual((reply.content, reply.eval_count), ("HEALTH", 2))
        self.assertIsNone(reply.position_logprobs)

    def test_schema_constrains_the_answer(self) -> None:
        with mock.patch.object(chat, "_post", return_value={}) as post:
            chat.chat("m", [], schema=chat.LABEL_SCHEMA)
        self.assertEqual(post.call_args.args[2]["format"], chat.LABEL_SCHEMA)

    def test_logprobs_keep_every_position(self) -> None:
        body: dict[str, Any] = {
            "message": {"content": "KEEP"},
            "logprobs": [
                {"top_logprobs": [{"token": "<|channel|>", "logprob": -0.1}]},
                {"top_logprobs": [{"token": "KEEP", "logprob": -0.2}, {"token": "SEND", "logprob": -1.5}]},
            ],
        }
        with mock.patch.object(chat, "_post", return_value=body) as post:
            reply = chat.chat("m", [], logprobs=3)
        payload = post.call_args.args[2]
        self.assertIs(payload["logprobs"], True)
        self.assertEqual(payload["top_logprobs"], 3)
        self.assertEqual(reply.position_logprobs, [{"<|channel|>": -0.1}, {"KEEP": -0.2, "SEND": -1.5}])

    def test_missing_fields_read_as_empty(self) -> None:
        with mock.patch.object(chat, "_post", return_value={"logprobs": None}):
            reply = chat.chat("m", [], logprobs=2)
        self.assertEqual((reply.content, reply.eval_count, reply.position_logprobs), ("", 0, []))


class HeadCommandTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        rng = random.Random(0)
        lines = []
        for i in range(40):
            sensitive = i % 2 == 0
            centre = 1.0 if sensitive else -1.0
            vector = [centre + rng.gauss(0, 0.3)] + [rng.gauss(0, 1) for _ in range(3)]
            label = "HEALTH" if sensitive else "CLEAN"
            lines.append(json.dumps({"id": f"r{i}", "label": label, "vector": vector}))
        self.embeddings = self.dir / "emb.jsonl"
        self.embeddings.write_text("\n".join(lines) + "\n")

    def run_main(self, *extra: str) -> tuple[int, str]:
        out = io.StringIO()
        with redirect_stdout(out):
            code = head.main(["--embeddings", str(self.embeddings), "--k", "4", *extra])
        return code, out.getvalue()

    def test_reports_writes_and_exports(self) -> None:
        report, exported = self.dir / "out" / "report.json", self.dir / "model" / "head.json"
        code, text = self.run_main("--out", str(report), "--export", str(exported))
        self.assertEqual(code, 0)
        self.assertIn("AUC", text)
        self.assertIn("Head written to", text)
        result = json.loads(report.read_text())
        self.assertTrue({"auc", "k", "chosen", "points", "scores"} <= result.keys())
        self.assertGreater(result["auc"], 0.9)
        self.assertEqual(len(result["scores"]), 40)
        raw = json.loads(exported.read_text())
        self.assertEqual((raw["dimensions"], raw["trained_on"]), (4, 40))
        self.assertIs(raw["embeddings_l2_normalised"], True)
        self.assertEqual(raw["threshold"], result["chosen"]["threshold"])
        self.assertEqual(len(Model.from_json(exported.read_text()).weights), 4)

    def test_rules_compose_with_the_head(self) -> None:
        rules = self.dir / "rules.json"
        rules.write_text(json.dumps({"r0": True, "r1": True}))
        report = self.dir / "report.json"
        code, _ = self.run_main("--rules", str(rules), "--out", str(report))
        self.assertEqual(code, 0)
        points = json.loads(report.read_text())["points"]
        self.assertTrue(all("composed_catch_rate" in p for p in points))
        # r1 is clean, so a rule that fires on it is friction at every threshold.
        self.assertTrue(all(p["composed_friction_rate"] > 0 for p in points))

    def test_unreachable_catch_is_said_not_hidden(self) -> None:
        code, text = self.run_main("--min-catch", "1.01")
        self.assertEqual(code, 0)
        self.assertIn("No threshold reaches", text)


if __name__ == "__main__":
    unittest.main()
