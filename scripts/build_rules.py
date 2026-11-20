# SPDX-License-Identifier: Apache-2.0
"""Compile rules/privacy-v2.yaml and write the conformance suite.

    uv run --with pyyaml==6.0.2 scripts/build_rules.py           # write the generated files
    uv run --with pyyaml==6.0.2 scripts/build_rules.py --check   # CI: fail if they are stale or a fixture fails

This is a development tool, not part of the package: it is the only place that parses YAML, so the engine
stays standard library only and loads the compiled JSON.

It refuses to write anything unless every check passes:
- the schema loads (strict keys, known categories, actions and validators, RE2-portable regexes);
- every rule has at least one `fail` and one `pass` fixture;
- every `fail` fixture produces exactly the spans its markers say, and every `pass` fixture produces none;
- every pattern of every rule produces a span in at least one of that rule's `fail` fixtures, so no pattern
  ships untested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from privacy_gate.guard.rules import Rule, RulesError, Ruleset  # noqa: E402  (needs the sys.path line above)

SOURCE = ROOT / "rules" / "privacy-v2.yaml"
COMPILED = ROOT / "src" / "privacy_gate" / "guard" / "privacy-v2.json"
CONFORMANCE = ROOT / "rules" / "conformance.v2.json"
OPEN, CLOSE = "⟦", "⟧"
STRIPPED_KEYS = ("rationale", "fixtures")


def fixture_text(item: Any, where: str) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        unknown = set(item) - {"prefix", "repeat", "times", "suffix"}
        if unknown or "repeat" not in item or "times" not in item:
            raise RulesError(f"{where}: a mapping fixture is {{prefix, repeat, times, suffix}}")
        return (
            str(item.get("prefix", ""))
            + str(item["repeat"]) * int(item["times"])
            + str(item.get("suffix", ""))
        )
    raise RulesError(f"{where}: a fixture is a string or a mapping")


def unmark(marked: str, where: str) -> tuple[str, list[list[int]]]:
    """Remove the markers and return the text with the spans they enclosed."""
    out: list[str] = []
    spans: list[list[int]] = []
    start: int | None = None
    for ch in marked:
        if ch == OPEN:
            if start is not None:
                raise RulesError(f"{where}: nested {OPEN}")
            start = len(out)
        elif ch == CLOSE:
            if start is None or start == len(out):
                raise RulesError(f"{where}: {CLOSE} without {OPEN}, or an empty span")
            spans.append([start, len(out)])
            start = None
        else:
            out.append(ch)
    if start is not None:
        raise RulesError(f"{where}: unclosed {OPEN}")
    return "".join(out), spans


def build(source: dict[str, Any]) -> tuple[bytes, bytes]:
    ruleset = Ruleset.from_mapping(source)
    cases: list[dict[str, Any]] = []
    errors: list[str] = []
    for spec, rule in zip(source["rules"], ruleset.rules, strict=True):
        fixtures = spec.get("fixtures") or {}
        if set(fixtures) - {"fail", "pass"}:
            raise RulesError(f"rule {rule.id}: fixtures are `fail` and `pass`")
        fails, passes = fixtures.get("fail") or [], fixtures.get("pass") or []
        if not fails or not passes:
            errors.append(f"{rule.id}: needs at least one fail and one pass fixture")
        fail_texts: list[str] = []
        for kind, items in (("fail", fails), ("pass", passes)):
            for n, item in enumerate(items):
                where = f"{rule.id} {kind}[{n}]"
                text, expected = unmark(fixture_text(item, where), where)
                if kind == "fail" and not expected:
                    errors.append(f"{where}: a fail fixture must mark at least one span")
                if kind == "pass" and expected:
                    errors.append(f"{where}: a pass fixture must not mark spans")
                got = [list(s) for s in rule.spans(text)]
                if got != expected:
                    errors.append(f"{where}: expected {expected}, got {got} in {text[:80]!r}")
                if kind == "fail":
                    fail_texts.append(text)
                cases.append(
                    {
                        "rule": rule.id,
                        "category": rule.category,
                        "kind": kind,
                        "text": text,
                        "spans": expected,
                        "aliases": list(rule.aliases),
                    }
                )
        for pattern in rule.patterns:
            alone = Rule(
                id=rule.id,
                title=rule.title,
                category=rule.category,
                patterns=(pattern,),
                validator=rule.validator,
                excepts=rule.excepts,
                except_values=rule.except_values,
                aliases=rule.aliases,
            )
            if not any(alone.spans(t) for t in fail_texts):
                errors.append(f"{rule.id}: pattern {pattern.regex!r} is not exercised by any fail fixture")
    if errors:
        raise RulesError("fixtures failed:\n  " + "\n  ".join(errors))

    compiled = {k: v for k, v in source.items()}
    compiled["note"] = (
        "Generated from rules/privacy-v2.yaml by scripts/build_rules.py. Do not edit by hand. "
        "Rationale and fixtures live in the source; fixtures are in rules/conformance.v2.json."
    )
    compiled["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    compiled["categories"] = {
        name: {k: v for k, v in spec.items() if k not in STRIPPED_KEYS}
        for name, spec in source["categories"].items()
    }
    compiled["combination"] = {
        k: v for k, v in source.get("combination", {}).items() if k not in STRIPPED_KEYS
    }
    compiled["rules"] = [
        {k: v for k, v in spec.items() if k not in STRIPPED_KEYS} for spec in source["rules"]
    ]
    compiled_bytes = (json.dumps(compiled, indent=2, ensure_ascii=True) + "\n").encode()

    conformance = {
        "schema": "privacy-gate-conformance/2",
        "note": (
            "Generated from rules/privacy-v2.yaml by scripts/build_rules.py. Do not edit by hand. For each "
            "case, the rule alone, run on `text` with no normalisation, must report exactly `spans` "
            "([start, end) in Unicode code points, sorted, disjoint). `kind: pass` cases report none."
        ),
        "ruleset": {
            "id": ruleset.id,
            "version": ruleset.version,
            "sha256": hashlib.sha256(compiled_bytes).hexdigest(),
        },
        "cases": cases,
    }
    conformance_bytes = (json.dumps(conformance, indent=1, ensure_ascii=True) + "\n").encode()
    return compiled_bytes, conformance_bytes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="fail if the generated files are stale; write nothing"
    )
    args = parser.parse_args(argv)
    import yaml  # development dependency only: `uv run --with pyyaml==6.0.2`

    source = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    try:
        compiled, conformance = build(source)
    except RulesError as error:
        print(f"build_rules: {error}", file=sys.stderr)
        return 1
    outputs = ((COMPILED, compiled), (CONFORMANCE, conformance))
    if args.check:
        stale = [str(p.relative_to(ROOT)) for p, data in outputs if not p.exists() or p.read_bytes() != data]
        if stale:
            print(f"build_rules: stale, re-run scripts/build_rules.py: {', '.join(stale)}", file=sys.stderr)
            return 1
        print(
            f"build_rules: up to date, {len(source['rules'])} rules, "
            f"{len(json.loads(conformance)['cases'])} conformance cases"
        )
        return 0
    for path, data in outputs:
        path.write_bytes(data)
        print(f"wrote {path.relative_to(ROOT)} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
