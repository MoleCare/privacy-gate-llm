// SPDX-License-Identifier: Apache-2.0
// Python and JavaScript must give the same spans on random texts, not only on the fixtures. The input is written
// by scripts/differential.py (CI runs it first); without it this test is skipped.
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { test } from "node:test";

import { RULESET_SHA256, ruleset } from "../dist/index.js";

const file = new URL("differential.json", import.meta.url);

test("the JavaScript engine agrees with the Python engine on random texts", { skip: !existsSync(file) }, () => {
  const data = JSON.parse(readFileSync(file, "utf8"));
  assert.equal(data.ruleset, RULESET_SHA256);
  const rules = ruleset();
  const mismatches = [];
  for (const c of data.cases) {
    for (const rule of rules.rules) {
      const got = rule.spans(c.text);
      if (JSON.stringify(got) !== JSON.stringify(c.spans[rule.id])) {
        mismatches.push(`${rule.id} ${JSON.stringify(c.text).slice(0, 120)} js=${JSON.stringify(got)}`);
      }
    }
  }
  assert.deepEqual(mismatches.slice(0, 10), []);
});
