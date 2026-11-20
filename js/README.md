# privacy-gate-llm (JavaScript / TypeScript)

**Must this prompt stay on this machine?** The JavaScript half of
[privacy-gate-llm](https://github.com/YauhenBichel/privacy-gate-llm): an open-source PII, PHI and secret
gate for LLM prompts, for anyone who routes between a local model and a cloud one (Claude, OpenAI, DeepSeek).

```bash
npm install privacy-gate-llm
```

No runtime dependencies. ESM with TypeScript types. Runs in Node 18+, Deno, Bun, and browsers or edge runtimes
through a bundler. It ships two things:

1. **The gate's head** (`model/head-v1.json`: 1024 weights, a bias, a threshold). Give it a `bge-m3` embedding
   and it says *hold* or *send*. It catches health data, personal data and credentials written as ordinary
   English, which regex rules cannot see.
2. **The guard's pattern rules** (`privacy-v2`, layer L1): API keys, private keys, JWTs, passwords in
   assignments and URLs, payment cards (Luhn), IBANs (mod 97), NHS numbers (mod 11), US SSNs, e-mail, phone
   numbers, private IPs, home paths, patient ids and dates of birth. It reports spans, never values.

Both are the same files the Python package (`pip install privacy-gate-llm`) uses, byte for byte, with the same
SHA-256 that `privacy-gate info` prints.

## Score an embedding

This package does not call a model. Get a `bge-m3` embedding (L2-normalised) from wherever you already run one:
Ollama `/api/embed`, any OpenAI-compatible `/v1/embeddings`, or transformers.js in the browser.

```ts
import { decide, hold } from "privacy-gate-llm";

const res = await fetch("http://127.0.0.1:11434/api/embed", {
  method: "POST",
  body: JSON.stringify({ model: "bge-m3", input: "the woman from Tuesday's clinic has a 7mm lesion" }),
});
const { embeddings } = await res.json();

hold(embeddings[0]);   // true: keep it local
decide(embeddings[0]); // { hold: true, score: ..., threshold: 0.1209, margin: ... }
```

`score()` refuses rather than guesses. A vector that is not 1024-dimensional (not `bge-m3`), or not
L2-normalised, throws an error. A wrong vector would not fail: it would give a confident, wrong score.
sentence-transformers needs `normalize_embeddings=True`. Ollama already normalises.

Log the score and the margin, never the text. A margin near zero means the gate is unsure.

## Find patterns

```ts
import { findSensitive, ruleset } from "privacy-gate-llm";

findSensitive("mail ana.real@company.co.uk, key sk-ant-api03-QQ11ww22EE33rr44TT55yy66");
// [ { ruleId: "PG-SECRET-001", category: "secret", action: "block", aliases: ["PRIV003"], start: 33, end: 70 },
//   { ruleId: "PG-EMAIL-001",  category: "email",  action: "mask",  aliases: ["PRIV005"], start: 5,  end: 27 } ]

ruleset().rule("PG-CARD-001").spans("card 4539 1488 0343 6467"); // [[5, 24]]
```

Offsets are Unicode code points, as in Python: use `[...text].slice(start, end)`, not `text.slice`. Overlaps are
resolved as in the Python guard. Block findings are all kept and come first. A mask finding that overlaps a block
is dropped. Of two overlapping mask findings, the longer one wins.

This is layer L1 only. The Python `Guard` also decodes the text first (L0: zero-width characters, `%xx`, base64,
NFKC lookalikes), walks whole request bodies, applies a policy and masks with surrogates. Those parts are not in
this package yet. Run the Python guard (or its HTTP sidecar) when you need them.

### How the rules stay identical

The rules are written once, in `rules/privacy-v2.yaml`, and are RE2-portable by construction. The Python loader
refuses lookaround, backreferences, named groups, possessive quantifiers, `\A`/`\Z` and late inline flags. The few
remaining differences between Python's `re.ASCII` and JavaScript's RegExp are translated at load time:

- `(?i)` becomes per-letter `[aA]`. JavaScript's `/iu` would also fold `ſ` (U+017F) into `s` and `K` (U+212A)
  into `k`, and Python's ASCII mode does not.
- `\s` becomes ASCII whitespace.
- `.` becomes `[^\n]`.
- `fullmatch` becomes `^(?:…)$`.

CI runs the repository's conformance suite (`rules/conformance.v2.json`, the cases the Python tests run). It also
runs a differential test: 3,000 random texts, scored by both engines, which must report the same spans.

## Exports

| Export | What |
|---|---|
| `score(v)`, `hold(v)`, `decide(v)` | the head on one embedding (`number[]` or `Float32Array`) |
| `THRESHOLD`, `HEAD`, `HEAD_SHA256` | the operating point, the head file, its hash |
| `findSensitive(text)` | L1 findings with the bundled rules |
| `ruleset()`, `Ruleset`, `Rule` | the compiled rules; `new Ruleset(data, sha256)` loads another file of the same schema |
| `RULES`, `RULESET_SHA256`, `VALIDATORS` | the rules file, its hash, the checksum validators |
| `privacy-gate-llm/head-v1.json`, `privacy-gate-llm/privacy-v2.json` | the raw files |

## Not a medical device

This is a privacy tool. It decides whether text should leave a machine and says nothing about health.

## Licence

Apache-2.0, see `LICENSE` and `NOTICE`. Built from `js/` in
[YauhenBichel/privacy-gate-llm](https://github.com/YauhenBichel/privacy-gate-llm). Releases are published from
CI with npm provenance.

To regenerate `test/head-scores.json` after a head change, run the following from the repository root:

```python
import json, math
from privacy_gate.head import Model
raw = open("model/head-v1.json").read(); m = Model.from_json(raw)
for k in (1, 7, 31, 101):
    v = [math.sin(k * (i + 1)) for i in range(1024)]; n = math.sqrt(sum(x * x for x in v))
    print(k, m.score([x / n for x in v]))
```
