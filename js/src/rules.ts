// SPDX-License-Identifier: Apache-2.0
// Port of the L1 matcher in src/privacy_gate/guard/rules.py and the overlap resolution in guard/detect.py.
//
// The rules are RE2-portable by construction (rules.py refuses lookaround, backreferences, named groups,
// possessive quantifiers, \A/\Z and late inline flags). What still differs between Python's `re.ASCII` and a
// JavaScript RegExp is handled when a pattern is translated, so the conformance suite passes byte for byte:
//
// - `(?i)` at the start: JavaScript has no inline flag, and its `i` flag (with `u`) folds non-ASCII letters into
//   ASCII ones (U+017F to s, U+212A to k). Python's ASCII mode does not, so each letter becomes `[aA]` instead.
// - `\s`: ASCII in Python ([ \t\n\r\f\v]), Unicode in JavaScript. Written out as that ASCII set.
// - `.`: Python excludes only "\n"; JavaScript also excludes "\r", U+2028 and U+2029. Written as [^\n].
// - Offsets: the `u` flag makes quantifiers count code points; spans are reported in code points, as in Python.
// - `fullmatch` for `except` patterns is ^(?:...)$.
import { RULES, RULESET_SHA256 } from "./generated/rules.js";
import type { PatternSpec, RulesFile } from "./types.js";
import { VALIDATORS } from "./validators.js";

export { RULES, RULESET_SHA256 };

export type Span = [start: number, end: number];

export interface Finding {
  ruleId: string;
  category: string;
  /** "block" or "mask", the category's shipped default. */
  action: "block" | "mask";
  /** Old ids the rule replaces (llm-harness PRIV001-005), reported beside the rule id. */
  aliases: string[];
  /** [start, end) in Unicode code points. */
  start: number;
  end: number;
}

const ASCII_SPACE = "\\t\\n\\v\\f\\r ";

/** Translate one Python (`re.ASCII`) pattern from the rules file to the source of an equivalent `u` RegExp. */
export function translate(pattern: string): string {
  let src = pattern;
  let ignoreCase = false;
  if (src.startsWith("(?i)")) {
    ignoreCase = true;
    src = src.slice(4);
  }
  const fold = (ch: string): string => {
    if (!ignoreCase || !/[A-Za-z]/.test(ch)) return ch;
    return ch.toLowerCase() + ch.toUpperCase();
  };
  let out = "";
  let i = 0;
  while (i < src.length) {
    const ch = src[i]!;
    if (ch === "\\") {
      const next = src[i + 1];
      if (next === undefined) throw new Error(`trailing backslash in ${pattern}`);
      if (next === "s") out += `[${ASCII_SPACE}]`;
      else if (next === "S") out += `[^${ASCII_SPACE}]`;
      else out += ch + next;
      i += 2;
    } else if (ch === "[") {
      const [cls, end] = translateClass(src, i, ignoreCase, pattern);
      out += cls;
      i = end;
    } else if (ch === ".") {
      out += "[^\\n]";
      i += 1;
    } else if (ch === "^" || ch === "$") {
      throw new Error(`anchor ${ch} is not supported by this engine: ${pattern}`);
    } else if (ch === "(" && src[i + 1] === "?" && src[i + 2] !== ":") {
      throw new Error(`group construct is not RE2-portable: ${pattern}`);
    } else if (/[A-Za-z]/.test(ch) && ignoreCase) {
      out += `[${fold(ch)}]`;
      i += 1;
    } else {
      out += ch;
      i += 1;
    }
  }
  return out;
}

function translateClass(src: string, start: number, ignoreCase: boolean, pattern: string): [string, number] {
  let i = start + 1;
  let out = "[";
  if (src[i] === "^") {
    out += "^";
    i += 1;
  }
  const items: string[] = [];
  let first = true;
  while (i < src.length && (src[i] !== "]" || first)) {
    first = false;
    let atom: string;
    let literal: string | null; // the character itself, when the atom is a single character
    if (src[i] === "\\") {
      const next = src[i + 1]!;
      i += 2;
      if (next === "s") {
        items.push(ASCII_SPACE);
        continue;
      }
      if (next === "S" || next === "W" || next === "D") {
        throw new Error(`negated shorthand \\${next} inside a class is not supported: ${pattern}`);
      }
      atom = "\\" + next;
      literal = /[dwbtnrfv0-9]/.test(next) ? null : next;
    } else {
      atom = src[i]!;
      literal = atom;
      i += 1;
      if ("{}()|/".includes(atom) || atom === "[") atom = "\\" + atom;
    }
    if (src[i] === "-" && src[i + 1] !== "]" && i + 1 < src.length) {
      // a range lo-hi
      i += 1;
      let hiLiteral: string;
      let hiAtom: string;
      if (src[i] === "\\") {
        hiLiteral = src[i + 1]!;
        hiAtom = "\\" + hiLiteral;
        i += 2;
      } else {
        hiLiteral = src[i]!;
        hiAtom = hiLiteral;
        i += 1;
      }
      if (literal === null) throw new Error(`range from a class escape: ${pattern}`);
      items.push(`${atom}-${hiAtom}`);
      if (ignoreCase) {
        const lower = /^[a-z]$/.test(literal) && /^[a-z]$/.test(hiLiteral);
        const upper = /^[A-Z]$/.test(literal) && /^[A-Z]$/.test(hiLiteral);
        if (lower) items.push(`${literal.toUpperCase()}-${hiLiteral.toUpperCase()}`);
        else if (upper) items.push(`${literal.toLowerCase()}-${hiLiteral.toLowerCase()}`);
        else if (/[A-Za-z]/.test(String.fromCharCode(...rangeCodes(literal, hiLiteral)))) {
          throw new Error(`a case-insensitive range mixing letters and other characters: ${pattern}`);
        }
      }
      continue;
    }
    if (atom === "-") atom = "\\-";
    items.push(atom);
    if (ignoreCase && literal !== null && /^[A-Za-z]$/.test(literal)) {
      items.push(literal === literal.toLowerCase() ? literal.toUpperCase() : literal.toLowerCase());
    }
  }
  if (src[i] !== "]") throw new Error(`unterminated class: ${pattern}`);
  return [out + items.join("") + "]", i + 1];
}

function rangeCodes(lo: string, hi: string): number[] {
  const codes: number[] = [];
  for (let c = lo.charCodeAt(0); c <= hi.charCodeAt(0) && codes.length < 256; c++) codes.push(c);
  return codes;
}

const stripSeparators = (text: string): string => text.replace(/[ -]/g, "");

/** UTF-16 index to code point index. */
function codePoints(text: string, index: number): number {
  let count = 0;
  for (let i = 0; i < index; i++) {
    const c = text.charCodeAt(i);
    if (c >= 0xd800 && c <= 0xdbff && i + 1 < index) {
      const d = text.charCodeAt(i + 1);
      if (d >= 0xdc00 && d <= 0xdfff) i += 1;
    }
    count += 1;
  }
  return count;
}

export class Rule {
  readonly id: string;
  readonly title: string;
  readonly category: string;
  readonly aliases: string[];
  readonly validator: string | null;
  private readonly patterns: { regex: RegExp; group: number }[];
  private readonly excepts: RegExp[];
  private readonly exceptValues: Set<string>;

  constructor(spec: RulesFile["rules"][number]) {
    this.id = spec.id;
    this.title = spec.title;
    this.category = spec.category;
    this.aliases = [...(spec.aliases ?? [])];
    this.validator = spec.validator ?? null;
    if (this.validator !== null && !(this.validator in VALIDATORS)) {
      // A rule naming a validator this engine lacks must not load: matching more than Python would.
      throw new Error(`rule ${spec.id}: unknown validator ${this.validator}`);
    }
    this.patterns = spec.patterns.map((p: PatternSpec) => {
      const regex = typeof p === "string" ? p : p.regex;
      const group = typeof p === "string" ? 0 : (p.group ?? 0);
      return { regex: new RegExp(translate(regex), "gdu"), group };
    });
    this.excepts = (spec.except ?? []).map((e) => new RegExp(`^(?:${translate(e)})$`, "u"));
    this.exceptValues = new Set((spec.except_values ?? []).map((v) => stripSeparators(String(v))));
  }

  /** Every span of `text` this rule holds, in code points, sorted and disjoint. */
  spans(text: string): Span[] {
    const found: Span[] = [];
    const check = this.validator ? VALIDATORS[this.validator]! : null;
    for (const { regex, group } of this.patterns) {
      regex.lastIndex = 0;
      for (let m = regex.exec(text); m !== null; m = regex.exec(text)) {
        if (m[0] === "") {
          regex.lastIndex += text.codePointAt(regex.lastIndex)! > 0xffff ? 2 : 1;
        }
        const where = m.indices?.[group];
        if (!where || where[0] === where[1]) continue;
        const value = text.slice(where[0], where[1]);
        if (check && !check(value)) continue;
        if (this.excepts.some((e) => e.test(value))) continue;
        if (this.exceptValues.size && this.exceptValues.has(stripSeparators(value))) continue;
        found.push([codePoints(text, where[0]), codePoints(text, where[1])]);
      }
    }
    found.sort((a, b) => a[0] - b[0] || b[1] - a[1]);
    const merged: Span[] = [];
    for (const [start, end] of found) {
      const last = merged[merged.length - 1];
      if (last && start < last[1]) {
        if (end > last[1]) last[1] = end;
        continue;
      }
      merged.push([start, end]);
    }
    return merged;
  }
}

export class Ruleset {
  readonly id: string;
  readonly version: string;
  readonly sha256: string;
  readonly categories: RulesFile["categories"];
  readonly rules: Rule[];

  constructor(data: RulesFile = RULES, sha256: string = RULESET_SHA256) {
    if (data.schema !== "privacy-gate-rules/2") {
      throw new Error(`ruleset: schema must be privacy-gate-rules/2, got ${data.schema}`);
    }
    this.id = data.id;
    this.version = String(data.version);
    this.sha256 = sha256;
    this.categories = data.categories;
    this.rules = data.rules.map((spec) => {
      if (!(spec.category in data.categories)) throw new Error(`rule ${spec.id}: unknown category ${spec.category}`);
      return new Rule(spec);
    });
  }

  rule(id: string): Rule {
    const found = this.rules.find((r) => r.id === id);
    if (!found) throw new Error(`no rule ${id}`);
    return found;
  }

  /**
   * Every L1 finding in `text`, with overlaps resolved as in the Python guard: every block finding is kept; a
   * mask finding overlapping a block is dropped; of overlapping mask findings the longer, then the earlier, wins.
   * Returns block findings first (by position), then mask findings by position.
   */
  find(text: string): Finding[] {
    const all: Finding[] = [];
    for (const rule of this.rules) {
      const action = this.categories[rule.category]!.action;
      for (const [start, end] of rule.spans(text)) {
        all.push({ ruleId: rule.id, category: rule.category, action, aliases: rule.aliases, start, end });
      }
    }
    const overlaps = (a: Finding, b: Finding): boolean => a.start < b.end && b.start < a.end;
    const blocks = all
      .filter((f) => f.action === "block")
      .sort((a, b) => a.start - b.start || a.end - b.end || cmp(a.ruleId, b.ruleId));
    const masks = all
      .filter((f) => f.action !== "block" && !blocks.some((b) => overlaps(b, f)))
      .sort((a, b) => b.end - b.start - (a.end - a.start) || a.start - b.start || cmp(a.ruleId, b.ruleId));
    const kept: Finding[] = [];
    for (const f of masks) if (!kept.some((k) => overlaps(k, f))) kept.push(f);
    kept.sort((a, b) => a.start - b.start);
    return [...blocks, ...kept];
  }
}

const cmp = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);

let shipped: Ruleset | undefined;

/** The bundled ruleset (privacy-v2), compiled once. */
export function ruleset(): Ruleset {
  shipped ??= new Ruleset();
  return shipped;
}

/** L1 findings for `text` with the bundled rules. Never contains the matched values, only spans. */
export function findSensitive(text: string): Finding[] {
  return ruleset().find(text);
}
