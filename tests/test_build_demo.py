# SPDX-License-Identifier: Apache-2.0
"""The demo build: one `demo/` folder, the head copied in from `model/head-v1.json` at build time, and the
search metadata the GitHub Pages mirror is indexed by. No network."""

from __future__ import annotations

import importlib.util
import io
import json
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("build_demo", ROOT / "scripts" / "build_demo.py")
assert spec and spec.loader
build_demo = importlib.util.module_from_spec(spec)
sys.modules["build_demo"] = build_demo
spec.loader.exec_module(build_demo)

# What the Space holds today besides .gitattributes, which the hub client never deletes. The upload's
# delete_patterns in space.yml are written against exactly these names.
SPACE_FILES = ["README.md", "explorer/data.json", "explorer/index.html", "head-v1.json", "index.html"]


class BuildTests(unittest.TestCase):
    def out(self) -> Path:
        return Path(self.enterContext(tempfile.TemporaryDirectory())) / "site"

    def test_space_is_exactly_the_published_file_set_with_the_canonical_head(self) -> None:
        out = self.out()
        self.assertEqual(build_demo.build("space", out), SPACE_FILES)
        self.assertEqual((out / "head-v1.json").read_bytes(), (ROOT / "model" / "head-v1.json").read_bytes())

    def test_space_yml_deletes_only_the_published_names(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "space.yml").read_text()
        match = re.search(r"delete_patterns=(\[.*?\])", workflow)
        assert match
        patterns = json.loads(match.group(1))
        self.assertNotIn("*", patterns)
        self.assertEqual(patterns, ["index.html", "head-v*.json", "explorer/*"])

    def test_pages_adds_robots_and_sitemap_for_the_base_url(self) -> None:
        out = self.out()
        files = build_demo.build("pages", out, base_url="https://example.org/pg/")
        self.assertEqual(
            files,
            [
                "explorer/data.json",
                "explorer/index.html",
                "head-v1.json",
                "index.html",
                "robots.txt",
                "sitemap.xml",
            ],
        )
        self.assertIn("Sitemap: https://example.org/pg/sitemap.xml", (out / "robots.txt").read_text())
        sitemap = (out / "sitemap.xml").read_text()
        self.assertIn("<loc>https://example.org/pg/</loc>", sitemap)
        self.assertIn("<loc>https://example.org/pg/explorer/index.html</loc>", sitemap)

    def test_gradio_ships_the_app_and_the_explorer_but_no_head(self) -> None:
        files = build_demo.build("gradio", self.out())
        self.assertEqual(
            files, ["README.md", "app.py", "explorer/data.json", "explorer/index.html", "requirements.txt"]
        )

    def test_refusals(self) -> None:
        out = self.out()
        with self.assertRaisesRegex(build_demo.BuildError, "unknown target"):
            build_demo.build("netlify", out)
        out.mkdir()
        (out / "stale.txt").write_text("x")
        with self.assertRaisesRegex(build_demo.BuildError, "not empty"):
            build_demo.build("space", out)
        with self.assertRaisesRegex(build_demo.BuildError, "end with '/'"):
            build_demo.build("pages", self.out(), base_url="https://example.org/pg")

    def test_a_page_that_stops_loading_the_head_or_a_missing_head_fails_the_build(self) -> None:
        demo = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (demo / "explorer").mkdir()
        (demo / "index.html").write_text('<script>fetch("head-v2.json")</script>')
        with self.assertRaisesRegex(build_demo.BuildError, "does not load head-v1.json"):
            build_demo.build("pages", self.out(), demo=demo)
        (demo / "index.html").write_text('<script>fetch("head-v1.json")</script>')
        with self.assertRaisesRegex(build_demo.BuildError, "missing"):
            build_demo.build("pages", self.out(), demo=demo, head=demo / "absent.json")

    def test_main(self) -> None:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(build_demo.main(["space", "--out", str(self.out())]), 0)
        self.assertIn("head-v1.json sha256 ", out.getvalue())
        with redirect_stderr(err):
            self.assertEqual(build_demo.main(["pages", "--out", str(self.out()), "--base-url", "x"]), 1)
        self.assertIn("build_demo:", err.getvalue())


class SourceTreeTests(unittest.TestCase):
    def test_one_demo_folder_and_no_committed_copy_of_the_head(self) -> None:
        self.assertFalse((ROOT / "space").exists())
        self.assertFalse((ROOT / "space-static").exists())
        self.assertEqual(list((ROOT / "demo").rglob("head-*.json")), [])

    def test_gradio_requirements_are_pinned_and_name_the_published_package(self) -> None:
        lines = [
            line.strip()
            for line in (ROOT / "demo" / "gradio" / "requirements.txt").read_text().splitlines()
            if line.strip() and not line.startswith(("#", "--"))
        ]
        self.assertTrue(lines)
        for line in lines:
            self.assertRegex(line, r"^[A-Za-z0-9_.\-\[\]]+==[0-9][^ ]*$", line)
        self.assertTrue(any(line.startswith("molecare-privacy-gate[local]==") for line in lines))

    def test_the_page_carries_search_metadata(self) -> None:
        html = (ROOT / "demo" / "index.html").read_text()
        for needle in (
            '<link rel="canonical" href="https://yauhenbichel.github.io/privacy-gate-llm/">',
            '<meta name="description"',
            '<meta property="og:title"',
            '<meta property="og:url" content="https://yauhenbichel.github.io/privacy-gate-llm/">',
            '<meta name="twitter:card"',
        ):
            self.assertIn(needle, html)
        match = re.search(r'<script type="application/ld\+json">(.*?)</script>', html, re.S)
        assert match
        data = json.loads(match.group(1))
        self.assertEqual(data["@type"], "SoftwareApplication")
        self.assertEqual(data["url"], build_demo.PAGES_URL)

    def test_space_card_short_description_fits_the_hub_limit(self) -> None:
        for card in ("README.md", "gradio/README.md"):
            text = (ROOT / "demo" / card).read_text()
            match = re.search(r"^short_description: (.+)$", text, re.M)
            assert match
            self.assertLessEqual(len(match.group(1)), 60, card)


if __name__ == "__main__":
    unittest.main()
