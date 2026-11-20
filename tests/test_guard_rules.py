# SPDX-License-Identifier: Apache-2.0
"""The rules file and the L1 matcher: conformance, schema checks, validators and a time budget.

Every fixture lives in rules/privacy-v2.yaml and reaches these tests through the generated
rules/conformance.v2.json, so the suite here is the same one the TypeScript engine and llm-harness will run.
No literal credential shapes are written in this file on purpose: CI greps *.py for them.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import time
import unittest
from pathlib import Path

from privacy_gate.guard import RulesError, Ruleset
from privacy_gate.guard.rules import DEFAULT_RULES_PATH, check_portable
from privacy_gate.guard.validators import (
    IBAN_LENGTHS,
    iban_mod97,
    luhn,
    nhs_mod11,
    phone_digits,
    private_ipv4,
    private_ipv6,
    us_ssn_area,
)

ROOT = Path(__file__).resolve().parent.parent
CONFORMANCE = ROOT / "rules" / "conformance.v2.json"
SOURCE = ROOT / "rules" / "privacy-v2.yaml"


def compiled() -> dict:
    return json.loads(DEFAULT_RULES_PATH.read_text())


class Conformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rules = Ruleset.load()
        cls.suite = json.loads(CONFORMANCE.read_text())

    def test_every_case(self) -> None:
        for case in self.suite["cases"]:
            with self.subTest(rule=case["rule"], text=case["text"][:60]):
                rule = self.rules.rule(case["rule"])
                self.assertEqual(rule.category, case["category"])
                self.assertEqual([list(s) for s in rule.spans(case["text"])], case["spans"])

    def test_suite_is_for_the_shipped_rules(self) -> None:
        # The suite names the ruleset by the hash of the compiled file, which is what every engine loads.
        self.assertEqual(self.suite["ruleset"]["sha256"], hashlib.sha256(DEFAULT_RULES_PATH.read_bytes()).hexdigest())
        self.assertEqual(self.suite["ruleset"]["sha256"], self.rules.sha256)

    def test_compiled_rules_match_the_yaml_source(self) -> None:
        # The standard-library check that the generated files were rebuilt after the YAML changed; CI also
        # runs `build_rules.py --check`, which needs PyYAML.
        self.assertEqual(compiled()["source_sha256"], hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                         "rules/privacy-v2.yaml changed: run scripts/build_rules.py")

    def test_every_rule_has_hits_and_near_misses(self) -> None:
        for rule in self.rules.rules:
            kinds = {c["kind"] for c in self.suite["cases"] if c["rule"] == rule.id}
            self.assertEqual(kinds, {"fail", "pass"}, rule.id)

    def test_v1_rules_are_all_carried_over_under_their_old_ids(self) -> None:
        # Architecture review R3: downstream headers, audit lines and dashboards say PRIV00x, so every v1 rule
        # survives as an alias of the v2 rules that replace it.
        carried = {a for c in self.suite["cases"] for a in c["aliases"]}
        self.assertEqual(carried, {"PRIV001", "PRIV002", "PRIV003", "PRIV004", "PRIV005"})
        self.assertGreaterEqual(sum(1 for c in self.suite["cases"] if c["aliases"]), 46)
        self.assertEqual(Ruleset.load().rule("PG-SECRET-003").aliases, ("PRIV004",))

    def test_the_seed_is_recorded(self) -> None:
        seed = compiled()["seeded_from"]
        self.assertEqual(seed["repo"], "llm-harness")
        self.assertEqual(len(seed["commit"]), 40)
        self.assertEqual(len(seed["ruleset_v1_sha256"]), 64)

    def test_block_categories_are_never_overridable(self) -> None:
        for category in self.rules.categories.values():
            if category.action == "block":
                self.assertFalse(category.overridable, category.name)
        self.assertEqual({c.name for c in self.rules.categories.values() if c.action == "block"},
                         {"secret", "card", "patient_or_national_id", "health_prose", "image_document"})


class Schema(unittest.TestCase):
    def load(self, mutate) -> None:
        data = copy.deepcopy(compiled())
        mutate(data)
        Ruleset.from_mapping(data)

    def test_the_shipped_file_loads(self) -> None:
        rules = Ruleset.load()
        self.assertEqual(rules.id, "privacy-v2")
        self.assertEqual(len(rules.sha256), 64)

    def test_a_block_category_cannot_be_made_overridable(self) -> None:
        with self.assertRaisesRegex(RulesError, "cannot be overridable"):
            self.load(lambda d: d["categories"]["secret"].update(overridable=True))

    def test_unknown_keys_fail_to_load(self) -> None:
        with self.assertRaisesRegex(RulesError, "unknown keys"):
            self.load(lambda d: d["rules"][0].update(severity="high"))
        with self.assertRaisesRegex(RulesError, "unknown keys"):
            self.load(lambda d: d.update(extra=1))

    def test_unknown_validator_or_category_fails(self) -> None:
        with self.assertRaisesRegex(RulesError, "unknown validator"):
            self.load(lambda d: d["rules"][0].update(validator="crc32"))
        with self.assertRaisesRegex(RulesError, "unknown category"):
            self.load(lambda d: d["rules"][0].update(category="hobbies"))

    def test_duplicate_ids_and_bad_groups_fail(self) -> None:
        with self.assertRaisesRegex(RulesError, "used twice"):
            self.load(lambda d: d["rules"].append(copy.deepcopy(d["rules"][0])))
        with self.assertRaisesRegex(RulesError, "group 2 does not exist"):
            self.load(lambda d: d["rules"][0].update(patterns=[{"regex": "(a)b", "group": 2}]))

    def test_bad_aliases_and_seed_fail(self) -> None:
        with self.assertRaisesRegex(RulesError, "aliases"):
            self.load(lambda d: d["rules"][0].update(aliases="PRIV003"))
        with self.assertRaisesRegex(RulesError, "aliases"):
            self.load(lambda d: d["rules"][0].update(aliases=["priv 3"]))
        with self.assertRaisesRegex(RulesError, "alias itself"):
            self.load(lambda d: d["rules"][0].update(aliases=[d["rules"][0]["id"]]))
        with self.assertRaisesRegex(RulesError, "seeded_from"):
            self.load(lambda d: d["seeded_from"].pop("commit"))

    def test_wrong_schema_fails(self) -> None:
        with self.assertRaisesRegex(RulesError, "schema"):
            self.load(lambda d: d.update(schema="privacy-gate-rules/1"))

    def test_non_portable_regexes_are_refused(self) -> None:
        for regex in (r"a(?=b)", r"a(?!b)", r"(?<=a)b", r"(?<!a)b", r"(a)\1", r"(?P<n>a)", r"(?<n>a)",
                      r"(?P=n)", r"(?>a)", r"a*+", r"a++", r"\Aabc", r"abc\Z", r"a(?i)b"):
            with self.subTest(regex=regex), self.assertRaises(RulesError):
                check_portable(regex)
        for regex in (r"(?i)abc", r"\*+", r"[A-Za-z0-9+/]{512,}", r"(?:a|b)+", r"\\1"):
            with self.subTest(regex=regex):
                check_portable(regex)

    def test_overlay_adds_rules_but_cannot_reuse_ids_or_invent_categories(self) -> None:
        base = Ruleset.load()
        overlay_data = {"schema": "privacy-gate-rules/2", "id": "clinic", "version": "1",
                        "categories": {"patient_or_national_id": compiled()["categories"]["patient_or_national_id"]},
                        "rules": [{"id": "PG-CLINIC-001", "title": "Clinic record", "category": "patient_or_national_id",
                                   "patterns": [r"\bCLN-\d{6}\b"]}]}
        merged = base.with_overlay(Ruleset.from_mapping(overlay_data))
        self.assertEqual(merged.rule("PG-CLINIC-001").spans("see CLN-000123 today"), [(4, 14)])
        self.assertNotEqual(merged.sha256, base.sha256)
        clash = copy.deepcopy(overlay_data)
        clash["rules"][0]["id"] = "PG-CARD-001"
        with self.assertRaisesRegex(RulesError, "already used"):
            base.with_overlay(Ruleset.from_mapping(clash))
        stray = copy.deepcopy(overlay_data)
        stray["categories"] = {"hobbies": {"title": "x", "action": "mask", "overridable": True,
                                           "surrogate": "tag", "layers": ["L1"]}}
        stray["rules"][0]["category"] = "hobbies"
        with self.assertRaisesRegex(RulesError, "unknown category"):
            base.with_overlay(Ruleset.from_mapping(stray))


class Matching(unittest.TestCase):
    def test_spans_from_several_patterns_merge(self) -> None:
        rule = Ruleset.from_mapping({
            "schema": "privacy-gate-rules/2", "id": "t", "version": "1",
            "categories": {"secret": compiled()["categories"]["secret"]},
            "rules": [{"id": "PG-T-001", "title": "t", "category": "secret", "patterns": ["abcd", "cdef", "x"]}],
        }).rules[0]
        self.assertEqual(rule.spans("abcdef x"), [(0, 6), (7, 8)])

    def test_word_boundaries_are_ascii_as_in_re2_and_javascript(self) -> None:
        # Python's default \b treats "é" as a word character; RE2 and JavaScript do not. Compiled with
        # re.ASCII, an address after an accented letter is found the same way in every engine.
        rule = Ruleset.load().rule("PG-EMAIL-001")
        self.assertEqual(rule.spans("é" + "ana@northgate.internal"), [(1, 23)])

    def test_iban_pattern_lengths_follow_the_registry(self) -> None:
        # PG-IBAN-001 takes exactly the country's length, so its alternatives must list the same countries
        # and lengths as IBAN_LENGTHS: a country missing from the pattern would never be found.
        (pattern,) = Ruleset.load().rule("PG-IBAN-001").patterns
        lengths: dict[str, int] = {}
        for countries, rest in re.findall(r"([A-Z]{2}(?:\|[A-Z]{2})*)\)?\\d\{2\}\(\?: \?\[A-Z0-9\]\)\{(\d+)\}",
                                          pattern.regex):
            for country in countries.split("|"):
                self.assertNotIn(country, lengths, country)
                lengths[country] = int(rest) + 4
        self.assertEqual(lengths, IBAN_LENGTHS)

    def test_unbroken_digit_runs_are_neither_cards_nor_ibans(self) -> None:
        # A valid card or IBAN followed by more digits is found (the conformance fixtures), but a run of 20 to
        # 30 digits with no separator is a reference or a hash, and must not fire however its prefixes check.
        rules = Ruleset.load()
        card, iban = rules.rule("PG-CARD-001"), rules.rule("PG-IBAN-001")
        rng = random.Random(97)
        for _ in range(2000):
            run = str(rng.randint(2, 6)) + "".join(rng.choice("0123456789") for _ in range(rng.randint(19, 29)))
            for text in (run, f"id {run} ok", f"GB{run}"):
                with self.subTest(text=text):
                    self.assertEqual(card.spans(text), [])
                    if not text.startswith("GB"):
                        self.assertEqual(iban.spans(text), [])


class Validators(unittest.TestCase):
    def test_luhn(self) -> None:
        self.assertTrue(luhn("4000 1234 5678 9017"))
        self.assertFalse(luhn("4000 1234 5678 9010"))
        self.assertFalse(luhn("4017"))  # too short to be a card

    def test_iban(self) -> None:
        self.assertTrue(iban_mod97("GB82 WEST 1234 5698 7654 32"))
        self.assertTrue(iban_mod97("DE89370400440532013000"))
        self.assertFalse(iban_mod97("GB82 WEST 1234 5698 7654 33"))
        self.assertFalse(iban_mod97("DE8937040044053201300"))  # one character short for Germany
        self.assertFalse(iban_mod97("ZZ82 WEST 1234 5698 7654 32"))  # no such country

    def test_nhs_number(self) -> None:
        self.assertTrue(nhs_mod11("943 476 5919"))
        self.assertFalse(nhs_mod11("943 476 5918"))
        self.assertFalse(nhs_mod11("123 456 7890"))  # check digit would be 10: never issued

    def test_us_ssn(self) -> None:
        self.assertTrue(us_ssn_area("123-45-6789"))
        for invalid in ("000-12-3456", "666-12-3456", "912-34-5678", "123-00-4567", "123-45-0000"):
            self.assertFalse(us_ssn_area(invalid), invalid)

    def test_private_addresses(self) -> None:
        for address in ("10.1.2.3", "172.16.0.5", "172.31.255.254", "192.168.1.20", "169.254.1.1", "100.64.0.1"):
            self.assertTrue(private_ipv4(address), address)
        for address in ("172.32.0.1", "8.8.8.8", "192.0.2.10", "127.0.0.1", "10.0.0.0", "999.1.1.1", "010.1.1.1"):
            self.assertFalse(private_ipv4(address), address)
        self.assertTrue(private_ipv6("fd12:3456:789a::1"))
        self.assertTrue(private_ipv6("fe80::1"))
        self.assertFalse(private_ipv6("2001:db8::1"))

    def test_phone(self) -> None:
        self.assertTrue(phone_digits("+44 7700 900123"))
        self.assertFalse(phone_digits("000 000 0000"))
        self.assertFalse(phone_digits("0123 456"))
        self.assertFalse(phone_digits("+1 234 567 890 123 456 7"))  # 16 digits: longer than E.164


class Budget(unittest.TestCase):
    """L1 has a budget of 10 ms for 60,000 characters on a laptop (design, section 4). CI runners are slower
    and shared, so the gate here is generous; its job is to catch a quadratic regex, not to measure."""

    def scan(self, text: str) -> float:
        rules = Ruleset.load()
        started = time.perf_counter()
        for rule in rules.rules:
            rule.spans(text)
        return time.perf_counter() - started

    def test_ordinary_text(self) -> None:
        line = "the deploy on 2026-11-20 failed at step 4 of 12; see build 1732096800 and retry, version 1.2.3\n"
        self.assertLess(self.scan(line * (60_000 // len(line))), 1.0)

    def test_inputs_that_make_backtracking_regexes_slow(self) -> None:
        for text in ("a." * 30_000, "1 " * 30_000, "@" * 60_000, "a" * 60_000, "-" * 60_000,
                     "/Users/" * 8_000, "+4" * 30_000, "x@" * 30_000):
            with self.subTest(text=text[:12]):
                self.assertLess(self.scan(text), 2.0)


if __name__ == "__main__":
    unittest.main()
