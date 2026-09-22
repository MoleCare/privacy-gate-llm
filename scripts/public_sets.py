#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The head on data sets other people made, beside the baselines everyone knows.

Three steps, so that each can run where its dependencies are:

  convert   other people's files into the gold-set row shape (id, label, slice, text), with a label map
            that is written down here and nowhere else
  score     every text through the systems asked for; writes ids, labels, slices, verdicts and scores,
            never the texts (the sets have their own licences; only derived numbers are committed)
  report    a Markdown table per set and per slice: catch on the sensitive rows, friction on the clean
            rows, and the head's AUC; the regex ruleset joins from `rules_oracle.py`'s output

Sets and their label maps:

  openshift   the "privacy-plus" evaluation corpus of an OpenShift AI router (amedeos.github.io, July 2026):
              332 English and 314 Italian cases, each with `expect: local` (keep on the local model) or
              `sota` (may leave). local -> HEALTH for category health, SECRET for secret, PII otherwise;
              sota -> CLEAN. Its author's routing policy is not our taxonomy: `finance` and `legal` are
              local there and would often be CLEAN here; `pii_weak` and `location` are sota there. The
              per-slice table is what to read.
  piimb       piimb/pii-masking-benchmark (CC-BY-NC-4.0), `data/test_sentences.jsonl`, English sentences
              with span labels. A balanced sample: PII when a sentence carries an identifying span (a name,
              an email, a phone, a date of birth, an address, an account or record number, a credential),
              CLEAN when it carries no span at all. Sentences whose only spans are dates, titles, company
              names or job titles are left out: neither label would be defensible.

Usage:
  python3 scripts/public_sets.py convert --openshift privacy-plus-english.yaml --out /tmp/openshift-en.jsonl
  python3 scripts/public_sets.py convert --piimb test_sentences.jsonl --sample 1000 --out /tmp/piimb.jsonl
  python3 scripts/public_sets.py score --data /tmp/openshift-en.jsonl --systems head,presidio,gliner \
      --backend openai --url http://127.0.0.1:11500 --out runs/public-openshift-en.json
  python3 scripts/rules_oracle.py --data /tmp/openshift-en.jsonl --out /tmp/openshift-en.rules.json
  python3 scripts/public_sets.py report runs/public-openshift-en.json --rules /tmp/openshift-en.rules.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from privacy_gate import backends  # noqa: E402
from privacy_gate.dataset import load  # noqa: E402
from privacy_gate.gate import Gate  # noqa: E402
from privacy_gate.head import auc  # noqa: E402

# ---- convert -------------------------------------------------------------------------------------------------

CASE = re.compile(r'\{id: ([^,]+), category: ([^,]+), expect: ([^,]+), text: "((?:[^"\\]|\\.)*)"\}')

# piimb span labels come from several source sets, so the same thing has several names
IDENTIFYING = {
    "first_name", "last_name", "GIVENNAME", "SURNAME", "name", "person", "email", "EMAIL", "phone_number",
    "TELEPHONENUMBER", "phone", "date_of_birth", "DATEOFBIRTH", "BUILDINGNUM", "STREET", "street_address",
    "address", "CITY", "ZIPCODE", "postcode", "ssn", "SOCIALNUM", "social_security_number", "medical_record_number",
    "account_number", "ACCOUNTNUM", "credit_card_number", "CREDITCARDNUMBER", "iban", "IBAN", "passport_number",
    "PASSPORTNUM", "driver_license_number", "DRIVERLICENSENUM", "TAXNUM", "tax_id", "ip_address", "IPADDRESS",
    "username", "USERNAME", "password", "PASSWORD", "api_key", "secret", "license_plate", "VEHICLEVIN",
    "health_insurance_id", "IDCARDNUM", "national_id", "employee_id", "customer_id",
}


def convert_openshift(path: Path) -> list[dict]:
    text = path.read_text()
    lang = re.search(r"^lang:\s*(\w+)", text, re.M)
    rows = []
    for id_, category, expect, body in CASE.findall(text):
        body = body.replace('\\"', '"').replace("\\\\", "\\")
        if expect == "local":
            label = {"health": "HEALTH", "secret": "SECRET"}.get(category, "PII")
        else:
            label = "CLEAN"
        rows.append({"id": f"os-{lang.group(1) if lang else 'xx'}-{id_}", "label": label, "slice": category, "text": body})
    return rows


def convert_piimb(path: Path, sample: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pii, clean = [], []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("language") != "en" or len(row["text"].strip()) < 20:
                continue
            labels = {e["label"] for e in row.get("entities", [])}
            if not labels:
                clean.append(row)
            elif labels & IDENTIFYING:
                pii.append(row)
    rng.shuffle(pii)
    rng.shuffle(clean)
    rows = []
    for kind, rows_in, label in (("pii", pii[:sample], "PII"), ("clean", clean[:sample], "CLEAN")):
        for row in rows_in:
            rows.append({"id": f"piimb-{row['uid']}", "label": label, "slice": f"piimb-{kind}", "text": row["text"].strip()})
    return rows


# ---- score -----------------------------------------------------------------------------------------------------

def score_head(texts: list[str], backend: str, url: str | None, batch: int) -> list[float]:
    gate = Gate.load(embedder=backends.make_embedder(backend, url))
    scores: list[float] = []
    for start in range(0, len(texts), batch):
        scores += [d.score for d in gate.decide_many(texts[start:start + batch])]
    return scores


PRESIDIO_IDENTIFYING = {"PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "IBAN_CODE", "US_SSN", "UK_NHS",
                        "MEDICAL_LICENSE", "CRYPTO", "IP_ADDRESS", "US_PASSPORT", "US_DRIVER_LICENSE", "US_BANK_NUMBER"}


def score_presidio(texts: list[str], language: str) -> tuple[list[bool], list[bool]]:
    """Two verdicts per text: any entity at all, and an identifying entity only. Presidio's default
    recognisers also flag dates, URLs and locations, which is friction for a gate but is what a user gets."""
    from presidio_analyzer import AnalyzerEngine

    engine = AnalyzerEngine()
    any_hit, ident_hit = [], []
    for text in texts:
        found = engine.analyze(text=text, language=language, score_threshold=0.5)
        any_hit.append(bool(found))
        ident_hit.append(any(r.entity_type in PRESIDIO_IDENTIFYING for r in found))
    return any_hit, ident_hit


GLINER_LABELS = ["person", "email", "phone number", "address", "date of birth", "credit card number", "iban",
                 "medical condition", "diagnosis", "passport number", "api key", "password", "social security number"]


def score_gliner(texts: list[str], model_id: str) -> list[bool]:
    from gliner import GLiNER

    model = GLiNER.from_pretrained(model_id)
    return [bool(model.predict_entities(text, GLINER_LABELS, threshold=0.5)) for text in texts]


def cmd_score(args: argparse.Namespace) -> int:
    rows = load(args.data)
    texts = [r.text for r in rows]
    out = {"data": str(args.data), "n": len(rows), "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "language": args.language,
           "ids": [r.id for r in rows], "labels": [r.label for r in rows], "slices": [r.slice for r in rows], "systems": {}}
    for system in args.systems.split(","):
        started = time.time()
        if system == "head":
            scores = score_head(texts, args.backend, args.url, args.batch)
            threshold = Gate.load(embedder=backends.make_embedder("ollama")).threshold
            out["systems"]["head"] = {"scores": scores, "holds": [s > threshold for s in scores], "threshold": threshold,
                                      "backend": args.backend}
        elif system == "presidio":
            any_hit, ident_hit = score_presidio(texts, args.language)
            out["systems"]["presidio_all"] = {"holds": any_hit}
            out["systems"]["presidio_identifying"] = {"holds": ident_hit}
        elif system == "gliner":
            out["systems"]["gliner_pii"] = {"holds": score_gliner(texts, args.gliner_model), "model": args.gliner_model}
        else:
            raise SystemExit(f"unknown system {system}")
        seconds = time.time() - started
        for name in [k for k in out["systems"] if k.startswith(system.split("_")[0])]:
            out["systems"][name].setdefault("seconds", round(seconds, 1))
            out["systems"][name].setdefault("per_text_ms", round(1000 * seconds / len(texts), 1))
        print(f"{system}: {len(texts)} texts in {seconds:.0f} s", flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("written", args.out)
    return 0


# ---- report ----------------------------------------------------------------------------------------------------

def summarise(labels: list[str], slices: list[str], holds: list[bool]) -> dict:
    sensitive = [lab != "CLEAN" for lab in labels]
    by_slice: dict[str, dict] = {}
    for name in sorted(set(slices)):
        idx = [i for i, s in enumerate(slices) if s == name]
        pos = [i for i in idx if sensitive[i]]
        neg = [i for i in idx if not sensitive[i]]
        by_slice[name] = {"n": len(idx),
                          "catch": round(sum(holds[i] for i in pos) / len(pos), 3) if pos else None,
                          "friction": round(sum(holds[i] for i in neg) / len(neg), 3) if neg else None}
    pos = [i for i, s in enumerate(sensitive) if s]
    neg = [i for i, s in enumerate(sensitive) if not s]
    return {"catch": round(sum(holds[i] for i in pos) / len(pos), 3) if pos else None,
            "friction": round(sum(holds[i] for i in neg) / len(neg), 3) if neg else None,
            "n_sensitive": len(pos), "n_clean": len(neg), "slices": by_slice}


def cmd_report(args: argparse.Namespace) -> int:
    run = json.loads(Path(args.run).read_text())
    labels, slices = run["labels"], run["slices"]
    systems = dict(run["systems"])
    if args.rules:
        matched = json.loads(Path(args.rules).read_text())
        systems["regex_rules"] = {"holds": [bool(matched.get(i, False)) for i in run["ids"]]}
    # The compositions the design intends: the head only ever adds a hold behind the rules, and beside an
    # entity tool it covers what that tool cannot see.
    if "head" in systems and "regex_rules" in systems:
        systems["rules+head"] = {"holds": [a or b for a, b in zip(systems["regex_rules"]["holds"], systems["head"]["holds"])]}
    if "head" in systems and "presidio_identifying" in systems:
        systems["presidio_identifying+head"] = {"holds": [a or b for a, b in zip(systems["presidio_identifying"]["holds"], systems["head"]["holds"])]}
    summaries = {name: summarise(labels, slices, sys_["holds"]) for name, sys_ in systems.items()}
    head_auc = None
    fixed = {}
    if "head" in systems:
        truth = [int(lab != "CLEAN") for lab in labels]
        head_auc = round(auc(systems["head"]["scores"], truth), 4)
        # the catch the head would reach at a friction the reader chooses, whatever the shipped threshold
        scores = systems["head"]["scores"]
        neg = sorted((s for s, t in zip(scores, truth) if not t), reverse=True)
        pos = [s for s, t in zip(scores, truth) if t]
        for friction in (0.05, 0.10):
            k = int(friction * len(neg))
            cut = neg[k] if k < len(neg) else neg[-1]
            fixed[friction] = round(sum(1 for s in pos if s > cut) / max(1, len(pos)), 3)
        print(f"\nhead: catch at 5 % friction {fixed[0.05]}, at 10 % friction {fixed[0.10]} (threshold moved for this set only)")
    names = list(summaries)
    print(f"\n### {Path(run['data']).name}: {run['n']} rows, {summaries[names[0]]['n_sensitive']} sensitive, "
          f"{summaries[names[0]]['n_clean']} clean\n")
    print("| System | Catch | Friction | AUC | ms per text |")
    print("|---|---|---|---|---|")
    for name in names:
        s = summaries[name]
        ms = systems[name].get("per_text_ms", "")
        print(f"| {name} | {s['catch']} | {s['friction']} | {head_auc if name == 'head' else ''} | {ms} |")
    print("\n| Slice | n | " + " | ".join(f"{n} catch / friction" for n in names) + " |")
    print("|---|---|" + "---|" * len(names))
    for slice_name in sorted(set(slices)):
        cells = []
        for name in names:
            s = summaries[name]["slices"][slice_name]
            cells.append(f"{s['catch'] if s['catch'] is not None else '-'} / {s['friction'] if s['friction'] is not None else '-'}")
        print(f"| {slice_name} | {summaries[names[0]]['slices'][slice_name]['n']} | " + " | ".join(cells) + " |")
    if args.out:
        Path(args.out).write_text(json.dumps({"run": args.run, "head_auc": head_auc, "head_catch_at_friction": fixed,
                                              "systems": summaries}, indent=1))
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    if args.openshift:
        rows = convert_openshift(Path(args.openshift))
    elif args.piimb:
        rows = convert_piimb(Path(args.piimb), args.sample, args.seed)
    else:
        raise SystemExit("give --openshift or --piimb")
    with open(args.out, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} rows -> {args.out}; labels {dict(Counter(r['label'] for r in rows))}; "
          f"slices {len(set(r['slice'] for r in rows))}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--openshift")
    c.add_argument("--piimb")
    c.add_argument("--sample", type=int, default=1000, help="piimb: rows per class")
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_convert)
    s = sub.add_parser("score")
    s.add_argument("--data", required=True)
    s.add_argument("--systems", default="head")
    s.add_argument("--backend", default="ollama", choices=backends.BACKENDS)
    s.add_argument("--url")
    s.add_argument("--batch", type=int, default=16)
    s.add_argument("--language", default="en")
    s.add_argument("--gliner-model", default="knowledgator/gliner-pii-small-v1.0")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_score)
    r = sub.add_parser("report")
    r.add_argument("run")
    r.add_argument("--rules", help="rules_oracle.py output for the same data")
    r.add_argument("--out")
    r.set_defaults(func=cmd_report)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
