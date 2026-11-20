# SPDX-License-Identifier: Apache-2.0
"""Maintainer tools: retrain and measure the head. Not part of the wheel.

Run from the repository root, against the package installed in editable mode (`pip install -e .`):

    python -m training.embed --data data/gold.jsonl --out runs/gold-bge-m3.jsonl
    python -m training.head --embeddings runs/gold-bge-m3.jsonl --rules data/rules-baseline.json

`python -m` puts the current directory on the import path, which is how `training` is found; the package
itself comes from the environment. See training/README.md.
"""
