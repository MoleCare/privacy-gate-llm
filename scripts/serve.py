#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The loopback HTTP sidecar, kept here for the old call shape:

    python3 scripts/serve.py --head model/head-v1.json

It is the same code as `privacy-gate serve` (src/privacy_gate/serve.py) and takes the same options.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from privacy_gate.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["serve", *sys.argv[1:]]))
