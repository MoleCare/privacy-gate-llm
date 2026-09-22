# SPDX-License-Identifier: Apache-2.0
"""The public-set evaluation, without the sets: the converters' label maps, the per-slice summary, the
compositions and the fixed-friction catch. No network, no model."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("public_sets", HERE.parent / "scripts" / "public_sets.py")
public_sets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(public_sets)

YAML = '''lang: en
cases:
  - {id: benign-01, category: benign, expect: sota, text: "Hi, help me build a microservice."}
  - {id: health-01, category: health, expect: local, text: "The patient received a diabetes diagnosis."}
  - {id: secret-01, category: secret, expect: local, text: "Use the key sk-test-000 for staging."}
  - {id: implicit-01, category: implicit, expect: local, text: "I said \\"keep it private\\" to her."}
'''


class ConvertTests(unittest.TestCase):
    def test_openshift_label_map(self) -> None:
        path = Path(tempfile.mkdtemp()) / "en.yaml"
        path.write_text(YAML)
        rows = public_sets.convert_openshift(path)
        self.assertEqual([r["label"] for r in rows], ["CLEAN", "HEALTH", "SECRET", "PII"])
        self.assertEqual(rows[0]["id"], "os-en-benign-01")
        self.assertEqual(rows[3]["text"], 'I said "keep it private" to her.')      # the escaped quote is unescaped
        self.assertEqual({r["slice"] for r in rows}, {"benign", "health", "secret", "implicit"})

    def test_piimb_balanced_sample_and_exclusions(self) -> None:
        rows = [
            {"uid": "a", "language": "en", "text": "Please read the full safety guidelines before the ride.", "entities": []},
            {"uid": "b", "language": "en", "text": "Send questions to J@outlook.com by Friday please.", "entities": [{"label": "EMAIL"}]},
            {"uid": "c", "language": "en", "text": "Starting on 10th June 1999 our team will pilot flexible hours.", "entities": [{"label": "DATE"}]},
            {"uid": "d", "language": "it", "text": "Per favore leggi le linee guida prima di partire, grazie.", "entities": []},
            {"uid": "e", "language": "en", "text": "short", "entities": []},
        ]
        path = Path(tempfile.mkdtemp()) / "t.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        out = public_sets.convert_piimb(path, sample=10, seed=1)
        self.assertEqual(sorted((r["id"], r["label"]) for r in out), [("piimb-a", "CLEAN"), ("piimb-b", "PII")])   # c: dates only; d: not en; e: short


class ReportTests(unittest.TestCase):
    def test_summarise_per_slice(self) -> None:
        labels = ["PII", "PII", "CLEAN", "CLEAN", "HEALTH"]
        slices = ["x", "x", "x", "y", "y"]
        holds = [True, False, True, False, True]
        s = public_sets.summarise(labels, slices, holds)
        self.assertEqual((s["catch"], s["friction"]), (round(2 / 3, 3), 0.5))
        self.assertEqual(s["slices"]["x"], {"n": 3, "catch": 0.5, "friction": 1.0})
        self.assertEqual(s["slices"]["y"], {"n": 2, "catch": 1.0, "friction": 0.0})

    def test_report_composes_and_fixes_friction(self) -> None:
        run = {"data": "/x/set.jsonl", "n": 6, "ids": list("abcdef"), "labels": ["PII", "PII", "PII", "CLEAN", "CLEAN", "CLEAN"],
               "slices": ["s"] * 6,
               "systems": {"head": {"scores": [3.0, 2.0, -1.0, 1.0, -2.0, -3.0], "holds": [True, True, False, True, False, False],
                                    "threshold": 0.1, "per_text_ms": 1.0},
                           "presidio_identifying": {"holds": [False, False, True, False, False, False]}}}
        d = Path(tempfile.mkdtemp())
        (d / "run.json").write_text(json.dumps(run))
        (d / "rules.json").write_text(json.dumps({"c": True}))
        out = io.StringIO()
        with redirect_stdout(out):
            code = public_sets.main(["report", str(d / "run.json"), "--rules", str(d / "rules.json"), "--out", str(d / "rep.json")])
        self.assertEqual(code, 0)
        rep = json.loads((d / "rep.json").read_text())
        self.assertEqual(rep["systems"]["head"]["catch"], round(2 / 3, 3))
        self.assertEqual(rep["systems"]["rules+head"]["catch"], 1.0)                     # the rules add the third
        self.assertEqual(rep["systems"]["presidio_identifying+head"]["catch"], 1.0)
        self.assertEqual(rep["systems"]["regex_rules"]["friction"], 0.0)
        self.assertGreater(rep["head_auc"], 0.8)
        self.assertIn("0.05", json.dumps(rep["head_catch_at_friction"]))
        self.assertIn("| rules+head |", out.getvalue())


if __name__ == "__main__":
    unittest.main()
