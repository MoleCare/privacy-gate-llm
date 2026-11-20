// SPDX-License-Identifier: Apache-2.0
// The same conformance suite the Python engine runs (tests/test_guard_rules.py): rules/conformance.v2.json,
// copied in by scripts/generate.mjs. Every case must give exactly the spans Python gives.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { RULESET_SHA256, VALIDATORS, findSensitive, ruleset, translate } from "../dist/index.js";

const here = new URL(".", import.meta.url);
const suite = JSON.parse(readFileSync(new URL("conformance.v2.json", here), "utf8"));

test("the suite is for the bundled rules", () => {
  const bytes = readFileSync(new URL("../../src/privacy_gate/guard/privacy-v2.json", here));
  assert.equal(createHash("sha256").update(bytes).digest("hex"), RULESET_SHA256);
  assert.equal(suite.ruleset.sha256, RULESET_SHA256);
  assert.equal(ruleset().sha256, RULESET_SHA256);
});

test("every conformance case", () => {
  const rules = ruleset();
  let n = 0;
  for (const c of suite.cases) {
    const rule = rules.rule(c.rule);
    assert.equal(rule.category, c.category, c.rule);
    assert.deepEqual(rule.aliases, c.aliases, c.rule);
    assert.deepEqual(rule.spans(c.text), c.spans, `${c.rule} ${c.kind}: ${JSON.stringify(c.text.slice(0, 80))}`);
    n += 1;
  }
  assert.ok(n >= 100, `only ${n} cases`);
});

test("every rule has hits and near misses", () => {
  for (const rule of ruleset().rules) {
    const kinds = new Set(suite.cases.filter((c) => c.rule === rule.id).map((c) => c.kind));
    assert.deepEqual([...kinds].sort(), ["fail", "pass"], rule.id);
  }
});

test("every validator the rules name exists", () => {
  for (const rule of ruleset().rules) if (rule.validator) assert.ok(rule.validator in VALIDATORS, rule.id);
});

test("offsets are code points, as in Python", () => {
  const text = "\u{1F600} mail ana.real@company.co.uk now";
  const [f] = findSensitive(text).filter((x) => x.category === "email");
  assert.ok(f);
  assert.equal([...text].slice(f.start, f.end).join(""), "ana.real@company.co.uk");
});

test("case folding is ASCII only, as Python's re.ASCII", () => {
  // U+017F (long s) folds to "s" in JavaScript's /iu but not in Python's ASCII mode.
  const r = new RegExp(translate("(?i)secret"), "u");
  assert.ok(r.test("SeCrEt"));
  assert.ok(!r.test("\u017Fecret"));
  // \s is ASCII whitespace only: a no-break space is not \s in Python's ASCII mode.
  assert.ok(!new RegExp(translate("a\\sb"), "u").test("a\u00A0b"));
});

test("findings carry spans and categories, never the value", () => {
  const found = findSensitive("contact ana.real@company.co.uk, key sk-ant-api03-QQ11ww22EE33rr44TT55yy66");
  const cats = found.map((f) => f.category);
  assert.ok(cats.includes("email"));
  assert.ok(cats.includes("secret"));
  for (const f of found) assert.ok(!("value" in f));
});
