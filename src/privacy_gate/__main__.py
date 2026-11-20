# SPDX-License-Identifier: Apache-2.0
"""`python -m privacy_gate ...` is the `privacy-gate` command, for environments without the script on PATH."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
