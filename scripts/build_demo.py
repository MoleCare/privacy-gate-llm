#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Assemble the demo for one destination from `demo/` and the canonical head, `model/head-v1.json`.

    python scripts/build_demo.py space  --out _site/space    # the static Hugging Face Space
    python scripts/build_demo.py pages  --out _site/pages    # the GitHub Pages mirror (adds robots.txt, sitemap.xml)
    python scripts/build_demo.py gradio --out _site/gradio   # the optional Gradio Space (needs CPU quota)

The head is copied in at build time and is never committed under `demo/`, so the demo cannot drift from the
head the package ships. `space.yml` and `pages.yml` run this; nothing here touches the network.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo"
HEAD = ROOT / "model" / "head-v1.json"
PAGES_URL = "https://yauhenbichel.github.io/privacy-gate-llm/"

TARGETS = ("space", "pages", "gradio")

ROBOTS = """User-agent: *
Allow: /

Sitemap: {base}sitemap.xml
"""

SITEMAP = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>{base}</loc><changefreq>monthly</changefreq><priority>1.0</priority></url>
  <url><loc>{base}explorer/index.html</loc><changefreq>monthly</changefreq><priority>0.6</priority></url>
</urlset>
"""


class BuildError(Exception):
    """The sources are not in the shape the demo needs."""


def _copy(src: Path, dst: Path) -> None:
    if not src.exists():
        raise BuildError(f"missing {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copyfile(src, dst)


def build(
    target: str, out: Path, demo: Path = DEMO, head: Path = HEAD, base_url: str = PAGES_URL
) -> list[str]:
    """Write the files for `target` into `out` (which must not exist or be empty); return them, sorted."""
    if target not in TARGETS:
        raise BuildError(f"unknown target {target!r}; choose from {', '.join(TARGETS)}")
    if out.exists() and any(out.iterdir()):
        raise BuildError(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)

    _copy(demo / "explorer", out / "explorer")
    if target == "gradio":
        # app.py loads the head the installed package ships, so the Gradio Space gets no head file.
        for name in ("app.py", "requirements.txt", "README.md"):
            _copy(demo / "gradio" / name, out / name)
    else:
        index = demo / "index.html"
        if 'fetch("head-v1.json")' not in (index.read_text(encoding="utf-8") if index.exists() else ""):
            raise BuildError(f"{index} does not load head-v1.json; the build and the page disagree")
        _copy(index, out / "index.html")
        _copy(head, out / "head-v1.json")
        if target == "space":
            _copy(demo / "README.md", out / "README.md")  # the Space card: front matter and description
        else:
            if not base_url.endswith("/"):
                raise BuildError(f"base URL must end with '/': {base_url}")
            (out / "robots.txt").write_text(ROBOTS.format(base=base_url), encoding="utf-8")
            (out / "sitemap.xml").write_text(SITEMAP.format(base=base_url), encoding="utf-8")

    return sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("target", choices=TARGETS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--base-url", default=PAGES_URL, help="pages: the public URL, for robots.txt and sitemap.xml"
    )
    args = parser.parse_args(argv)
    try:
        files = build(args.target, args.out, base_url=args.base_url)
    except BuildError as error:
        print(f"build_demo: {error}", file=sys.stderr)
        return 1
    for name in files:
        print(name)
    if args.target != "gradio":
        digest = hashlib.sha256((args.out / "head-v1.json").read_bytes()).hexdigest()
        print(f"head-v1.json sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
