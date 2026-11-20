#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The loopback HTTP sidecar, kept here for the old call shape:

    python3 scripts/serve.py --head model/head-v1.json

It is the same code as `privacy-gate serve` (src/privacy_gate/serve.py) and takes the same options. It
imports the installed package: `pip install -e .` in a checkout, or `pip install privacy-gate-llm`.
Prefer `privacy-gate serve` or `python -m privacy_gate serve`.
"""

from __future__ import annotations

import sys

from privacy_gate.cli import main

if __name__ == "__main__":
    sys.exit(main(["serve", *sys.argv[1:]]))
