// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { HEAD, HEAD_SHA256, THRESHOLD, decide, hold, score } from "../dist/index.js";

const here = new URL(".", import.meta.url);
const fixture = JSON.parse(readFileSync(new URL("head-scores.json", here), "utf8"));

const unit = (v) => {
  const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  return v.map((x) => x / n);
};
const vector = (k) =>
  k === "mean" ? unit(HEAD.mean) : unit(Array.from({ length: 1024 }, (_, i) => Math.sin(k * (i + 1))));

test("the bundled head is model/head-v1.json, byte for byte", () => {
  const bytes = readFileSync(new URL("../../model/head-v1.json", here));
  assert.equal(createHash("sha256").update(bytes).digest("hex"), HEAD_SHA256);
  assert.equal(HEAD.weights.length, 1024);
  assert.equal(THRESHOLD, HEAD.threshold);
});

test("scores agree with the Python package", () => {
  for (const c of fixture.cases) {
    const v = vector(c.k);
    assert.ok(Math.abs(score(v) - c.score) < 1e-9, `k=${c.k}: ${score(v)} vs ${c.score}`);
    assert.equal(hold(v), c.hold, `k=${c.k}`);
    assert.equal(hold(Float32Array.from(v)), c.hold, `k=${c.k} as Float32Array`);
  }
});

test("decide carries score, threshold and margin", () => {
  const d = decide(vector(7));
  assert.equal(d.threshold, THRESHOLD);
  assert.equal(d.margin, d.score - THRESHOLD);
  assert.equal(d.hold, d.score > THRESHOLD);
});

test("an un-normalised vector is refused, not scored", () => {
  const v = vector(7).map((x) => x * 3);
  assert.throws(() => score(v), /not L2-normalised/);
  assert.throws(() => score(new Array(1024).fill(Number.NaN)), /not L2-normalised/);
});

test("a vector from another model (wrong length) is refused", () => {
  assert.throws(() => score(unit(new Array(768).fill(1))), /768 dimensions/);
});
