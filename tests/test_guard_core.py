# SPDX-License-Identifier: Apache-2.0
"""The guard core (P1): walker, L0 views, policy, vault, masking, unmasking, audit and the `Guard` facade.

Every value here is fake and low-entropy: addresses at northgate.internal, the Ofcom drama range, documented
or Luhn-valid test card numbers, and EXAMPLE-NOT-A-KEY. No literal credential shapes on purpose: CI greps *.py.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import stat
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from privacy_gate.guard import (
    AuditWriter,
    Guard,
    Policy,
    PolicyError,
    Ruleset,
    ScanLimitError,
    ScanLimits,
    SessionStore,
    StreamUnmasker,
    ToolArgsBuffer,
    Vault,
    check_local_url,
    detect,
    safe_path,
    scan_units,
    unmask,
    unmask_json_text,
)
from privacy_gate.guard.mask import residual
from privacy_gate.guard.normalize import MAX_VIEWS, DecodeLimitError, views
from privacy_gate.guard.policy import parse_overrides, strip_markers
from privacy_gate.guard.rules import DEFAULT_RULES_PATH as RULES_PATH
from privacy_gate.guard.vault import make_surrogate
from privacy_gate.guard.walker import BinaryUnit, TextUnit, binary_kind, js_string

ROOT = Path(__file__).resolve().parent.parent
BODIES = ROOT / "rules" / "conformance.bodies.v2.json"
EMAIL = "ana.duarte@northgate.internal"
OTHER_EMAIL = "j.doe@northgate.internal"
CARD = "4000 1234 5678 9017"
PHONE = "07700 900123"
SECRET = "EXAMPLE-NOT-A-KEY"
RULES = Ruleset.load()


def nested(depth: int) -> list:
    body: list = []
    inner = body
    for _ in range(depth - 1):
        inner.append([])
        inner = inner[0]
    return body


class Walker(unittest.TestCase):
    def test_leaves_are_paired_with_their_key_in_document_order(self) -> None:
        units = list(scan_units({"a": "x", "b": [1, True, None, {"c": "y"}], "d": 1.5}))
        self.assertEqual([u.text for u in units], ["a: x", "b: 1", "b: true", "c: y", "d: 1.5"])
        self.assertEqual(units[3].path, ("b", 3, "c"))
        self.assertTrue(units[0].is_string)
        self.assertFalse(units[1].is_string)
        self.assertEqual(units[0].value_offset, 3)

    def test_a_bare_string_and_unpaired_mode(self) -> None:
        self.assertEqual(list(scan_units("hi")), [TextUnit((), None, "hi", 0, True)])
        self.assertEqual([u.text for u in scan_units({"k": "v", "n": 3}, pair_keys=False)], ["v"])

    def test_js_string_matches_javascript(self) -> None:
        cases = [(True, "true"), (False, "false"), (40219, "40219"), (1.0, "1"), (1.5, "1.5"), (1e-7, "1e-7"),
                 (1e21, "1e+21"), (float("nan"), "NaN"), (float("inf"), "Infinity"), (float("-inf"), "-Infinity")]
        for value, expected in cases:
            self.assertEqual(js_string(value), expected, value)

    def test_binary_kinds(self) -> None:
        self.assertEqual(binary_kind({"type": "image", "source": {"type": "base64"}}), "image")
        self.assertEqual(binary_kind({"type": "image_url", "image_url": {"url": "u"}}), "image")
        self.assertEqual(binary_kind({"type": "input_image"}), "image")
        self.assertEqual(binary_kind({"type": "document", "source": {"type": "base64"}}), "document")
        self.assertIsNone(binary_kind({"type": "document", "source": {"type": "text"}}))
        self.assertIsNone(binary_kind({"type": "document", "source": {"type": "content"}}))
        self.assertEqual(binary_kind({"type": "input_audio"}), "audio")
        self.assertEqual(binary_kind({"type": "file", "file": {}}), "file")
        self.assertIsNone(binary_kind({"type": "file"}))
        self.assertEqual(binary_kind({"type": "container_upload"}), "file")
        self.assertEqual(binary_kind({"type": "search_result", "source": {"type": "url"}}), "search_result")
        self.assertEqual(binary_kind({"source": {"type": "file"}}), "binary")
        self.assertIsNone(binary_kind({"type": "text", "text": "hi"}))
        unit = next(u for u in scan_units({"c": [{"type": "image"}]}) if isinstance(u, BinaryUnit))
        self.assertEqual(unit.path, ("c", 0))

    def test_caps_and_non_json_values_raise(self) -> None:
        with self.assertRaises(ScanLimitError):
            list(scan_units(nested(70)))
        list(scan_units(nested(60)))
        with self.assertRaises(ScanLimitError):
            list(scan_units(["x"] * 20, ScanLimits(max_nodes=10)))
        with self.assertRaises(ScanLimitError):
            list(scan_units(["x" * 50] * 3, ScanLimits(max_chars=100)))
        for bad in ({1: "x"}, {"a": b"x"}, {"a": {1, 2}}):
            with self.assertRaises(ScanLimitError):
                list(scan_units(bad))

    def test_a_cycle_hits_the_depth_cap(self) -> None:
        loop: list = []
        loop.append(loop)
        with self.assertRaises(ScanLimitError):
            list(scan_units(loop))


class Normalise(unittest.TestCase):
    def found(self, text: str) -> list[tuple[str, str]]:
        return [(f.category, text[f.start:f.end]) for f in detect(text, RULES)]

    def test_zero_width_and_escapes_and_full_width(self) -> None:
        for text in ("ana.duarte\u200b@northgate.internal", "ana.duarte%40northgate.internal",
                     "ana.duarte\\u0040northgate.internal", "ana.duarte＠northgate.internal"):
            self.assertEqual([c for c, _ in self.found(text)], ["email"], text)
            self.assertEqual(self.found(text)[0][1], text, "the span maps back to the whole original")

    def test_dashes_fold(self) -> None:
        self.assertEqual([c for c, _ in self.found("card 4000–1234–5678–9017")], ["card"])

    def test_base64_and_hex_at_depth_one_and_two(self) -> None:
        once = base64.b64encode(f"mail {EMAIL}".encode()).decode()
        twice = base64.b64encode(once.encode()).decode()
        hexed = f"mail {EMAIL}".encode().hex()
        for text in (once, twice, hexed):
            found = detect(f"see {text} ok", RULES)
            self.assertEqual([f.category for f in found], ["email"], text)
            self.assertTrue(found[0].encoded)
            self.assertEqual((found[0].start, found[0].end), (4, 4 + len(text)), "the whole run")
        thrice = base64.b64encode(twice.encode()).decode()
        self.assertEqual(detect(thrice, RULES), [], "depth is capped at 2")

    def test_clean_text_has_no_views_and_too_many_runs_raise(self) -> None:
        self.assertEqual(views("plain words"), [])
        run = base64.b64encode(b"hello there, friend!").decode()
        with self.assertRaises(DecodeLimitError):
            views(" ".join([run] * (MAX_VIEWS + 1)))


class PolicyTests(unittest.TestCase):
    def test_defaults_and_loosening_rules(self) -> None:
        policy = Policy.from_mapping({}, RULES)
        self.assertEqual((policy.profile, policy.mode, policy.on_block), ("lite", "enforce", "refuse"))
        self.assertEqual(policy.action("email", RULES), "mask")
        self.assertEqual(Policy.from_mapping({"categories": {"email": "allow"}}, RULES).action("email", RULES),
                         "allow")
        with self.assertRaises(PolicyError):
            Policy.from_mapping({"categories": {"card": "mask"}}, RULES)

    def test_providers_only_tighten(self) -> None:
        policy = Policy.from_mapping({"providers": {"deepseek": {"categories": {"email": "block"}}}}, RULES)
        self.assertEqual(policy.action("email", RULES, "deepseek"), "block")
        self.assertEqual(policy.action("email", RULES, "anthropic"), "mask")
        with self.assertRaises(PolicyError):
            Policy.from_mapping({"categories": {"email": "block"},
                                 "providers": {"x": {"categories": {"email": "mask"}}}}, RULES)

    def test_bad_policies_are_refused(self) -> None:
        for bad in ([], {"nope": 1}, {"version": 2}, {"extends": "builtin:huge"}, {"mode": "loud"},
                    {"categories": {"nope": "mask"}}, {"categories": {"email": "maybe"}}, {"on_block": "local"},
                    {"on_block": "drop"}, {"local": {"url": "ftp://x"}}, {"shadow_block_categories": "yes"},
                    {"allow": {"values_file": "/does/not/exist"}}, {"overlays": [3]}, {"providers": {"x": []}}):
            with self.assertRaises(PolicyError, msg=bad):
                Policy.from_mapping(bad, RULES)  # type: ignore[arg-type]

    def test_local_url_must_be_local(self) -> None:
        check_local_url("http://127.0.0.1:11500")
        check_local_url("http://localhost:11434/v1")
        check_local_url("http://[::1]:8080")
        check_local_url("http://192.168.1.20:8080", ("192.168.1.20",))
        check_local_url("http://yserver.lan:8080", ("yserver.lan",))
        for bad, hosts in (("http://192.168.1.20", ()), ("http://8.8.8.8", ("8.8.8.8",)),
                           ("http://api.example.org", ()), ("file:///x", ())):
            with self.assertRaises(PolicyError, msg=bad):
                check_local_url(bad, hosts)

    def test_files_json_yaml_allowlist_and_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "allow.txt").write_text(f"# team inbox\n{EMAIL}\n")
            path = Path(tmp) / "policy.json"
            path.write_text(json.dumps({"allow": {"values_file": "allow.txt"}}))
            policy = Policy.load(path, RULES)
            self.assertTrue(policy.allowed(EMAIL))
            self.assertFalse(policy.allowed(OTHER_EMAIL))
            self.assertNotIn(EMAIL, repr(policy))
            (Path(tmp) / "bad.json").write_text("{")
            with self.assertRaises(PolicyError):
                Policy.load(Path(tmp) / "bad.json", RULES)
            with self.assertRaises(PolicyError):
                Policy.load(Path(tmp) / "missing.json", RULES)
        self.assertEqual(parse_overrides(" email, phone ,"), {"email", "phone"})
        self.assertEqual(parse_overrides(None), set())
        self.assertEqual(strip_markers("a [[send-as-is:email,phone]] b"), ("a  b", {"email", "phone"}))


class VaultTests(unittest.TestCase):
    def test_shapes(self) -> None:
        mac = bytes(range(32))
        self.assertRegex(make_surrogate(mac, "email", "", EMAIL, False) or "", r"^m-[a-z2-7]{6}@masked\.invalid$")
        self.assertRegex(make_surrogate(mac, "phone", "", PHONE, False) or "", r"^\+44 7700 900\d{3}$")
        self.assertRegex(make_surrogate(mac, "private_ip", "", "10.42.0.7", False) or "",
                         r"^(192\.0\.2|198\.51\.100|203\.0\.113)\.\d{1,3}$")
        self.assertRegex(make_surrogate(mac, "private_ip", "", "fd00::1", False) or "", r"^2001:db8::")
        self.assertRegex(make_surrogate(mac, "home_path", "", "jdoe", False) or "", r"^u-[a-z2-7]{6}$")
        self.assertRegex(make_surrogate(mac, "tag", "IBAN", "x", False) or "", r"^<IBAN_[a-z2-7]{6}>$")
        self.assertRegex(make_surrogate(mac, "phone", "PHONE", PHONE, True) or "", r"^<PHONE_[a-z2-7]{6}>$")

    def test_same_value_same_surrogate_and_sessions_differ(self) -> None:
        vault = Vault()
        first = vault.surrogate("email", EMAIL, "email", "")
        self.assertEqual(vault.surrogate("email", EMAIL, "email", ""), first)
        self.assertNotEqual(vault.surrogate("email", OTHER_EMAIL, "email", ""), first)
        self.assertEqual(vault.value(first), EMAIL)
        self.assertTrue(vault.issued(first))
        self.assertNotEqual(Vault().surrogate("email", EMAIL, "email", ""), first)
        self.assertNotIn(EMAIL, repr(vault))

    def test_collisions_bump_and_never_reuse_a_surrogate(self) -> None:
        vault = Vault(key=b"k" * 32)
        phones = {vault.surrogate("phone", f"07700 9{n:05d}", "phone", "PHONE") for n in range(1500)}
        self.assertEqual(len(phones), 1500, "1000 drama numbers, then tag surrogates")
        self.assertTrue(any(p.startswith("<PHONE_") for p in phones))

    def test_sessions_are_bound_to_the_caller_and_expire(self) -> None:
        now = [0.0]
        store = SessionStore(idle_seconds=100, clock=lambda: now[0])
        mine = store.vault("caller-a", "s1")
        mine.surrogate("email", EMAIL, "email", "")
        self.assertIs(store.vault("caller-a", "s1"), mine)
        self.assertEqual(store.vault("caller-b", "s1").surrogates(), [], "another caller, same session id")
        now[0] = 50
        self.assertIs(store.vault("caller-a", "s1"), mine)
        now[0] = 500
        self.assertIsNot(store.vault("caller-a", "s1"), mine)
        self.assertEqual(len(store), 1)


class Unmasking(unittest.TestCase):
    def setUp(self) -> None:
        self.vault = Vault(key=b"u" * 32)
        self.email = self.vault.surrogate("email", EMAIL, "email", "")
        self.tag = self.vault.surrogate("iban", 'GB "quoted" \\ value', "tag", "IBAN")
        self.ip = self.vault.surrogate("private_ip", "10.42.0.7", "private_ip", "")

    def test_only_issued_surrogates_exactly(self) -> None:
        text = f"to {self.email}, from {self.ip}; not <IBAN_zzzzzz> or {self.email.upper()}"
        out = unmask(text, self.vault)
        self.assertIn(EMAIL, out)
        self.assertIn("10.42.0.7", out)
        self.assertIn("<IBAN_zzzzzz>", out)
        self.assertIn(self.email.upper(), out)
        self.assertEqual(unmask("nothing here", Vault()), "nothing here")

    def test_longest_first(self) -> None:
        vault = Vault()
        vault._by_surrogate.update({"192.0.2.1": "a", "192.0.2.12": "b"})
        vault.version += 1
        self.assertEqual(unmask("192.0.2.12 and 192.0.2.1", vault), "b and a")

    def test_stream_equals_whole_text_at_every_split(self) -> None:
        text = f"Dear {self.email}, the host {self.ip} and {self.tag}. Bye {self.email}"
        whole = unmask(text, self.vault)
        for cut in range(len(text) + 1):
            stream = StreamUnmasker(self.vault)
            self.assertEqual(stream.feed(text[:cut]) + stream.feed(text[cut:]) + stream.flush(), whole, cut)
        stream = StreamUnmasker(self.vault)
        self.assertEqual("".join(stream.feed(ch) for ch in text) + stream.flush(), whole)

    def test_holds_back_only_a_possible_surrogate(self) -> None:
        stream = StreamUnmasker(self.vault)
        self.assertEqual(stream.feed("plain text"), "plain text")
        self.assertEqual(stream.feed(" " + self.email[:4]), " ")
        self.assertEqual(stream.held, 4)
        self.assertEqual(stream.feed(self.email[4:] + " end"), EMAIL + " end")
        self.assertEqual(stream.flush(), "")

    def test_json_values_stay_valid_json(self) -> None:
        args = json.dumps({"to": self.email, "ref": self.tag})
        out = json.loads(unmask_json_text(args, self.vault))
        self.assertEqual(out, {"to": EMAIL, "ref": 'GB "quoted" \\ value'})
        buffer = ToolArgsBuffer(self.vault)
        for i in range(0, len(args), 5):
            buffer.feed(args[i:i + 5])
        self.assertEqual(json.loads(buffer.finish())["ref"], 'GB "quoted" \\ value')


class Audit(unittest.TestCase):
    def test_files_are_private_hashes_short_and_rotation_keeps_four(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = AuditWriter(Path(tmp) / "audit.jsonl", max_bytes=600, keep=4)
            self.assertEqual(stat.S_IMODE((Path(tmp) / "audit.key").stat().st_mode), 0o600)
            digest = writer.hash_value("email", EMAIL)
            self.assertRegex(digest, r"^[0-9a-f]{8}$")
            self.assertEqual(writer.hash_value("email", EMAIL), digest)
            self.assertEqual(AuditWriter(Path(tmp) / "audit.jsonl").hash_value("email", EMAIL), digest,
                             "the key file is reused")
            guard = Guard.load()
            for _ in range(30):
                writer.write(guard.protect({"q": f"mail {EMAIL}"}, hasher=writer.hash_value).decision)
            files = sorted(p.name for p in Path(tmp).iterdir())
            self.assertEqual(files, ["audit.jsonl", "audit.jsonl.1", "audit.jsonl.2", "audit.jsonl.3",
                                     "audit.jsonl.4", "audit.key"])
            self.assertEqual(stat.S_IMODE((Path(tmp) / "audit.jsonl").stat().st_mode), 0o600)
            record = json.loads((Path(tmp) / "audit.jsonl").read_text().splitlines()[-1])
            self.assertEqual(record["action"], "mask")
            self.assertEqual(record["value_hashes"], [digest])

    def test_a_readable_key_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            key = Path(tmp) / "audit.key"
            key.write_text("00" * 32)
            os.chmod(key, 0o644)
            with self.assertRaises(PermissionError):
                AuditWriter(Path(tmp) / "audit.jsonl")


class BodyConformance(unittest.TestCase):
    """rules/conformance.bodies.v2.json: whole bodies, as the TypeScript engine and llm-harness will run them."""

    def test_every_case(self) -> None:
        suite = json.loads(BODIES.read_text())
        self.assertEqual(suite["schema"], "privacy-gate-body-conformance/2")
        guard = Guard.load(profile=suite["profile"])
        for case in suite["cases"]:
            with self.subTest(case["name"]):
                body = nested(case["deep"]) if "deep" in case else case["body"]
                if "base64" in case:
                    encoded = base64.b64encode(case["base64"].encode()).decode()
                    body = json.loads(json.dumps(body).replace("{base64}", encoded))
                result = guard.protect(body, provider="anthropic", wire=case.get("wire", "anthropic"))
                expect, decision = case["expect"], result.decision
                self.assertEqual(decision.action, expect["action"], decision.reason)
                if expect["rules"]:
                    self.assertLessEqual(set(expect["rules"]), set(decision.rules), decision.rules)
                else:
                    self.assertEqual(decision.rules, [])
                for key in ("paths", "categories", "refused_override", "shadow_rules", "fail_closed"):
                    if key in expect:
                        self.assertEqual(getattr(decision, key), expect[key], key)
                sent = json.dumps(result.body)
                for gone in expect.get("masked_absent", []):
                    self.assertNotIn(gone, sent)
                for kept in expect.get("masked_present", []):
                    self.assertIn(kept, sent)
                if decision.action == "block":
                    self.assertIsNone(result.body)


class Facade(unittest.TestCase):
    def test_mask_then_unmask_round_trip(self) -> None:
        guard = Guard.load()
        body = {"messages": [{"role": "user", "content": f"mail {EMAIL} from 10.42.0.7 on {PHONE}"}]}
        result = guard.protect(body, caller="c", session="s")
        self.assertEqual(result.decision.action, "mask")
        self.assertEqual(body["messages"][0]["content"], f"mail {EMAIL} from 10.42.0.7 on {PHONE}",
                         "the caller's body is not changed")
        sent = result.body["messages"][0]["content"]
        for value in (EMAIL, "10.42.0.7", PHONE):
            self.assertNotIn(value, sent)
        stream = guard.unmasker("c", "s")
        self.assertEqual(stream.feed(sent) + stream.flush(), body["messages"][0]["content"])
        self.assertEqual(guard.unmasker("other", "s").feed(sent), sent, "another caller cannot unmask")

    def test_overrides_header_and_allowlist(self) -> None:
        guard = Guard.load()
        result = guard.protect({"q": f"mail {EMAIL}"}, overrides="email,card")
        self.assertEqual(result.decision.action, "allow")
        self.assertEqual(result.decision.override, ["email"])
        self.assertEqual(result.decision.refused_override, ["card"])
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "allow.txt").write_text(EMAIL + "\n")
            guard = Guard.load({"allow": {"values_file": str(Path(tmp) / "allow.txt")}})
        result = guard.protect({"q": f"mail {EMAIL} and {OTHER_EMAIL}"})
        self.assertEqual(result.decision.action, "mask")
        self.assertIn(EMAIL, result.body["q"])
        self.assertNotIn(OTHER_EMAIL, result.body["q"])

    def test_shadow_mode_and_r10(self) -> None:
        shadow = Guard.load({"mode": "shadow"})
        masked = shadow.protect({"q": f"mail {EMAIL}"})
        self.assertEqual((masked.decision.action, masked.decision.would_action), ("allow", "mask"))
        self.assertIn("PG-EMAIL-001", masked.decision.shadow_rules)
        self.assertEqual(masked.body, {"q": f"mail {EMAIL}"})
        self.assertEqual(shadow.protect({"q": f"card {CARD}"}).decision.action, "block",
                         "shadow mode does not release a block category")
        loose = Guard.load({"mode": "shadow", "shadow_block_categories": True}).protect({"q": f"card {CARD}"})
        self.assertEqual((loose.decision.action, loose.decision.would_action), ("allow", "block"))

    def test_combination(self) -> None:
        text = f"{EMAIL} {PHONE} 10.42.0.7 /home/jdoe/notes"
        lite = Guard.load().protect({"q": text}).decision
        self.assertEqual((lite.action, lite.shadow_rules), ("mask", ["PG-COMBO-001"]))
        standard = Guard.load(profile="standard", detectors={"L2": lambda text: []}).protect({"q": text})
        self.assertEqual(standard.decision.action, "block")
        self.assertIn("PG-COMBO-001", standard.decision.rules)
        three = Guard.load().protect({"q": f"{EMAIL} {PHONE} 10.42.0.7"}).decision
        self.assertEqual(three.shadow_rules, [])

    def test_neighbouring_surrogates_do_not_trip_the_residual_check(self) -> None:
        # A phone surrogate then an address surrogate reads as a Luhn-valid card about one session in ten.
        text = f"{EMAIL} {PHONE} 10.42.0.7 /home/jdoe/notes"
        for n in range(200):
            decision = Guard.load().protect({"q": text}, session=str(n)).decision
            self.assertEqual(decision.action, "mask", decision.reason)

    def test_the_residual_check_still_catches_original_text(self) -> None:
        vault = Vault()
        phone = vault.surrogate("phone", PHONE, "phone", "PHONE")
        left = residual(f"call {phone}, card {CARD} today", vault, RULES, ("L0", "L1"))
        self.assertEqual([f.category for f in left], ["card"])
        self.assertEqual([f.category for f in residual(f"{phone} at 10.42.0.99", vault, RULES, ("L0", "L1"))
                          if f.category != "card"], ["private_ip"], "an address the vault did not issue is kept")
        cleared = 0
        for _ in range(300):
            fresh = Vault()
            text = (f'{fresh.surrogate("phone", PHONE, "phone", "PHONE")} '
                    f'{fresh.surrogate("private_ip", "10.42.0.7", "private_ip", "")}')
            self.assertEqual(residual(text, fresh, RULES, ("L0", "L1")), [], text)
            cleared += any(f.category == "card" for f in detect(text, RULES))
        self.assertGreater(cleared, 0, "some pairs read as a card, and all of them are cleared")

    def test_on_block_local_and_provider_tightening(self) -> None:
        guard = Guard.load({"on_block": "local", "local": {"url": "http://127.0.0.1:11500"},
                            "providers": {"deepseek": {"categories": {"email": "block"}}}})
        result = guard.protect({"q": f"card {CARD}"})
        self.assertEqual((result.decision.action, result.body), ("local", {"q": f"card {CARD}"}))
        self.assertFalse(result.decision.sends)
        self.assertEqual(guard.protect({"q": EMAIL}, provider="deepseek").decision.action, "local")
        self.assertEqual(guard.protect({"q": EMAIL}, provider="anthropic").decision.action, "mask")

    def test_l2_detector_findings_are_used(self) -> None:
        def names(text: str) -> list[tuple[str, int, int]]:
            at = text.find("Ana Duarte")
            return [("person_name", at, at + 10)] if at >= 0 else []

        guard = Guard.load(profile="standard", detectors={"L2": names})
        result = guard.protect({"q": "letter for Ana Duarte today"})
        self.assertEqual(result.decision.action, "mask")
        self.assertRegex(result.body["q"], r"^letter for <PERSON_[a-z2-7]{6}> today$")

    def test_a_maskable_value_in_a_number_or_a_key_is_held(self) -> None:
        def digits(text: str) -> list[tuple[str, int, int]]:
            at = text.find("7700900123")
            return [("phone", at, at + 10)] if at >= 0 else []

        guard = Guard.load(profile="standard", detectors={"L2": digits})
        decision = guard.protect({"meta": {"phone": 7700900123}}).decision
        self.assertEqual((decision.action, decision.paths), ("block", ["meta.phone"]))
        decision = guard.protect({"meta": {"x7700900123": "owner"}}).decision
        self.assertEqual((decision.action, decision.paths), ("block", ["meta.*"]))
        self.assertEqual(guard.protect({"meta": {"phone": "7700900123"}}).decision.action, "mask")

    def test_fail_closed(self) -> None:
        self.assertIn("L2", Guard.load(profile="standard").failure or "")
        for guard in (Guard.load({"overlays": ["/does/not/exist.json"]}), Guard.load({"mode": "loud"}),
                      Guard.load(rules_path="/does/not/exist.json"), Guard.load(profile="huge")):
            decision = guard.protect({"q": "hello"}).decision
            self.assertEqual((decision.action, decision.fail_closed), ("block", True), guard.failure)
        boom = Guard.load(profile="standard", detectors={"L2": lambda text: 1 / 0})
        decision = boom.protect({"q": "hello"}).decision
        self.assertEqual((decision.action, decision.fail_closed), ("block", True))
        self.assertEqual(decision.reason, "engine error (ZeroDivisionError); nothing sent")
        wrong = Guard.load(profile="standard", detectors={"L2": lambda text: [("nope", 0, 1)]})
        self.assertEqual(wrong.protect({"q": "hello"}).decision.action, "block")
        small = Guard.load(limits=ScanLimits(max_chars=50))
        self.assertTrue(small.protect({"q": "x" * 100}).decision.fail_closed)
        self.assertTrue(Guard.load().protect({"q": b"bytes"}).decision.fail_closed)

    def test_overlay_pinning(self) -> None:
        categories = {"secret": json.loads(Path(RULES_PATH).read_text())["categories"]["secret"]}
        overlay = {"schema": "privacy-gate-rules/2", "id": "org", "version": "1", "categories": categories,
                   "rules": [{"id": "PG-ORG-001", "title": "project code", "category": "secret",
                              "patterns": ["\\bPRJ-[0-9]{4}\\b"]}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "org.json"
            raw = json.dumps(overlay).encode()
            path.write_bytes(raw)
            good = hashlib.sha256(raw).hexdigest()
            guard = Guard.load({"overlays": [{"path": str(path), "sha256": good}]})
            self.assertIsNone(guard.failure)
            self.assertEqual(guard.protect({"q": "see PRJ-1234"}).decision.rules, ["PG-ORG-001"])
            self.assertNotEqual(guard.ruleset.sha256 if guard.ruleset else "", RULES.sha256)
            bad = Guard.load({"overlays": [{"path": str(path), "sha256": "0" * 64}]})
            self.assertIn("pinned", bad.failure or "")

    def test_safe_paths(self) -> None:
        self.assertEqual(safe_path(("messages", 0, "content")), "messages.0.content")
        self.assertEqual(safe_path(("metadata", EMAIL)), "metadata.*")
        self.assertEqual(safe_path(("metadata", "owner"), True), "metadata.*")
        self.assertEqual(safe_path(()), "$")

    def test_no_value_reaches_a_decision_an_audit_line_or_the_console(self) -> None:
        planted = [EMAIL, PHONE, "10.42.0.7", CARD.replace(" ", ""), SECRET, "jdoe"]
        bodies = [
            {"q": f"mail {EMAIL} on {PHONE} from 10.42.0.7 at /home/jdoe/x"},
            {"q": f"card {CARD.replace(' ', '')}"},
            {"tools": [{"input": {"password": SECRET}}]},
            {EMAIL: {"note": f"call {PHONE}"}},
            {"q": f"mail {EMAIL} [[send-as-is:email,card]]"},
        ]
        guards = [Guard.load(), Guard.load({"mode": "shadow"}),
                  Guard.load(profile="standard", detectors={"L2": lambda t: 1 / 0})]
        out, err = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(out), redirect_stderr(err):
            writer = AuditWriter(Path(tmp) / "audit.jsonl")
            dumps = []
            for guard in guards:
                for body in bodies:
                    decision = guard.protect(body, hasher=writer.hash_value).decision
                    writer.write(decision)
                    dumps.append(json.dumps(decision.to_dict()))
                    dumps.append(repr(decision))
            dumps.append((Path(tmp) / "audit.jsonl").read_text())
        dumps += [out.getvalue(), err.getvalue()]
        for value in planted:
            for dump in dumps:
                self.assertNotIn(value, dump)


if __name__ == "__main__":
    unittest.main()
