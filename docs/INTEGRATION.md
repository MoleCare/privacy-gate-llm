# Using the gate in another project

The gate is deliberately trivial to consume. Whatever the language, it is:

1. embed the text (one call to an embedding endpoint), then
2. standardise, dot with 1024 weights, add the bias, compare to a threshold.

Step 2 is about ten lines in any language. Step 1 is the only real dependency.

> **Read this first.** The gate may only ever *add* a hold, never clear one. A
> `send` verdict is not an assurance that text is safe — it is the absence of a
> second opinion. Keep whatever deterministic checks you already have, and put
> this behind them. See `SECURITY.md`.

---

## Three ways to call it

### 1. Python, in process

```python
from privacy_gate.gate import Gate

gate = Gate.load("model/head-v0.json", url="http://127.0.0.1:11434")
d = gate.decide("her biopsy is booked for the 20th")
# Decision(hold=True, score=7.23, threshold=-0.4545)

gate.decide_many([...])   # one embedding call for the whole batch
```

### 2. The HTTP sidecar, for everything else

```bash
python3 scripts/serve.py --head model/head-v0.json --port 8231
```

```bash
curl -s localhost:8231/check -d '{"text":"her biopsy is booked for the 20th"}'
{"hold": true, "score": 7.2272, "threshold": -0.4545, "margin": 7.6817}

curl -s localhost:8231/check -d '{"texts":["reformat this YAML","the root password is Autumn-Ledger-77"]}'
{"results":[{"hold":false,"score":-7.5608,...},{"hold":true,"score":10.5691,...}]}
```

It binds to `127.0.0.1`, has no authentication, and **fails closed**: if the
embedding endpoint is unreachable it returns `503` with `"hold": true`. Treat any
non-200 as hold.

### 3. Port the arithmetic

If you would rather not run Python at all, `model/head-v0.json` is public data:
`weights`, `bias`, `mean`, `stdev`, `threshold`. In TypeScript:

```ts
import head from "./head-v0.json";

export function score(embedding: number[]): number {
  // The head was fitted on L2-normalised vectors. An un-normalised one does not
  // fail, it scores confidently wrong, so check rather than trust.
  const norm = Math.hypot(...embedding);
  if (Math.abs(norm - 1) > 0.01) {
    throw new Error(`embedding is not L2-normalised (norm ${norm.toFixed(4)})`);
  }
  let total = head.bias;
  for (let i = 0; i < embedding.length; i++) {
    total += head.weights[i] * (embedding[i] - head.mean[i]) / head.stdev[i];
  }
  return total;
}

export const hold = (embedding: number[]) => score(embedding) > head.threshold;
```

Two ways to get this silently wrong, both of which produce a confident score
rather than an error:

- **Not `bge-m3`.** The head is 1024 dimensions fitted to that model's space and
  means nothing in another. Pin it.
- **Not L2-normalised.** Ollama's `/api/embed` normalises; `sentence-transformers`
  does not unless you pass `normalize_embeddings=True`. `Gate` refuses a vector
  whose norm is not 1, and a hand-rolled port should too — as above.

---

## llm-harness

This is what the gate was built for. `rules/v1.yaml` runs first and keeps
precedence; the gate is a second opinion on everything the rules let through.

```
prompt ──▶ privacy rules ──hold──▶ local, locked
              │ pass
              ▼
          the gate ──hold──▶ local, locked  (reason: "semantic")
              │ pass
              ▼
      the caller's tag decides
```

In `src/router.ts`, after the privacy step and **before** the explicit-tag step,
so the gate cannot be overridden by `[[cloud]]` any more than a rule can:

```ts
const verdict = await gate.check(text);          // POST /check
if (verdict.hold) {
  trace.push({ step: "gate", matched: true, score: verdict.score });
  return { backend: "local", locked: true, why: "the gate held this back" };
}
trace.push({ step: "gate", matched: false, score: verdict.score });
```

Two things worth doing at the same time:

- **Log the score, not the text.** The score plus the example id is enough to
  tune the threshold later. The text is the thing you are trying not to copy.
- **Log near-misses.** Anything within about a point of the threshold is a
  candidate for the gold set, and that is how the eval set stops being 147
  examples written in one day.

The three `negative-fixture` cases in `docs/JOURNAL.md` are worth fixing in
`v1.yaml` while you are in there: they are the whole of the composed system's
avoidable friction, and the gate already scores them correctly.

---

## molecare-mcp

These two fit together well, but **only one of the two obvious ways actually
protects anything.** The difference matters, so it is worth being precise.

### The one that works: check tool *results* on the way out

An MCP tool result does not stay on your machine. It is returned to the client,
which puts it into the model's context — and if that client is Claude Desktop or
any hosted model, the result has just been sent to a third party.

So the risky direction is **outbound**: a MoleCare API tool (`moles.ts`) that
returns real user data hands that data to whatever model is driving. That is
precisely the leak this project exists to stop, and the regex ruleset never sees
it because it is a tool result, not a prompt.

`src/runtime.ts` has a single choke point. `registerTools` installs one
`CallToolRequestSchema` handler through which every tool call and every result
passes, so the check goes there and nowhere else:

```ts
// src/runtime.ts, inside the CallToolRequestSchema handler,
// after dispatch and before returning.
const result = await dispatch(name, args, context);

if (process.env.PRIVACY_GATE_URL) {
  const text = JSON.stringify(result);
  const verdict = await checkGate(text);           // fail closed on error
  if (verdict.hold) {
    logger.warn(`gate held the result of ${name}`, { score: verdict.score });
    return errorResult(
      "PRIVACY_HOLD",
      "This result contains data that must not leave the machine. " +
      "Query the API directly rather than through an assistant.",
      name
    );
  }
}
return result;
```

Notes that matter more than the snippet:

- **Off unless configured.** Gated on `PRIVACY_GATE_URL`, so the published npm
  package behaves exactly as it does today for everyone who has not opted in.
  molecare-mcp works with no credentials by design; do not break that.
- **Do not gate the knowledge tools.** `knowledge.ts` returns ABCDE education and
  SNOMED/ICD lookups — public reference text, no person in it. Gating it buys
  nothing and adds an embedding call per lookup. Apply the check to the tools
  that touch the API (`moles.ts`) and, if you want, `mlflow.ts`.
- **Latency is real.** One embedding call per tool result. Batch where a tool
  returns a list, and skip results under a few dozen characters.

### The one that does not: a `privacy_check` MCP tool

It is tempting to expose the gate *as* a tool, so an assistant can ask "is this
safe to send?".

For **inbound** text this is theatre. By the time the model can call a tool with
the text, the text is already in the model's context and has already left the
machine. The check has to happen before the text enters the prompt — which means
it belongs in the harness, the hook, or the middleware above, never in a tool the
model calls.

There is one honest use for it, and it is worth having. An agent that is about to
publish something it *composed* — a PR comment, a commit message, a Slack reply,
an issue body — can check its own draft first. The draft came from the context, so
asking costs no new exposure, and the gate stops it being written somewhere
public. That is the same job as `comms-harness`, and if you add the tool, describe
it that way so no one reaches for it as an input filter:

```
name: privacy_check
description:
  Check text you are about to publish (a comment, commit message or message)
  for health data, credentials or personal data before it is posted. This does
  NOT protect text you have already been given — by then it is already in
  context. Returns hold/send and a score.
```

### The combination worth building

`molecare-mcp` already knows which of its tools touch real data, and
`privacy-gate-llm` knows what real data looks like in prose. Together they give
you something neither has alone: **an MCP server that can be pointed at a live
MoleCare account and still be safe to drive from a hosted assistant**, because
anything that comes back carrying patient narrative is refused rather than
returned.

That is a genuinely useful property for a public MCP server, and it is the
argument for doing this at all.

---

## molecare-server (Spring Boot, Java 21)

The in-app helper and the clinical PDF path both build text and send it to a
model. Same shape, as a pre-flight:

```java
@Component
public class PrivacyGate {
    private final RestTemplate rest;
    private final String url;   // ${privacy.gate.url:} — empty disables it

    /** Fails closed: any error means hold. */
    public boolean mustStayLocal(String text) {
        if (url == null || url.isBlank()) return false;
        try {
            var body = rest.postForObject(url + "/check", Map.of("text", text), Map.class);
            return Boolean.TRUE.equals(body.get("hold"));
        } catch (RestClientException e) {
            log.warn("privacy gate unreachable, holding", e);
            return true;
        }
    }
}
```

Backward compatibility applies as it does everywhere in that repo: default the
property to empty so an existing deployment is unchanged until someone sets it.

---

## comms-harness and skin-care-harness

Both are deterministic guards with the same blind spot as `llm-harness`: they
match patterns. The gate slots in behind each of them in exactly the same way —
rules first, gate second, gate may only add.

For `comms-harness` this is the strongest fit after llm-harness, because an
outbound chat message is a person writing prose about other people, which is the
case regex handles worst and this handles best.

---

## Pre-commit and CI

The gate reads text, and a diff is text:

```bash
git diff --cached | python3 -c "
import json, sys, urllib.request
text = sys.stdin.read()
if text.strip():
    req = urllib.request.Request('http://127.0.0.1:8231/check',
        data=json.dumps({'text': text}).encode(),
        headers={'content-type': 'application/json'})
    if json.load(urllib.request.urlopen(req))['hold']:
        sys.exit('privacy gate: this diff looks like it contains personal data')
"
```

Be careful with the threshold here. A commit hook that fires on ordinary work
gets `--no-verify`'d within a day, and then it protects nothing. Start at the
95.8%-catch operating point rather than the 100% one, and move it only if
something gets through.

---

## Choosing a threshold

The head ships with the catch ≥ 98% point. `docs/JOURNAL.md` run 6 has the full
curve; these are the useful corners:

| threshold | catch | friction | use when |
|---|---|---|---|
| −1.385 | 100% | 16.4% | nothing may leak and interruptions are acceptable |
| −0.455 | 98.6% | 12.7% | the default in `model/head-v0.json` |
| −0.232 | 97.2% | 9.1% | high-volume paths where friction compounds |

Override it per caller rather than editing the file:

```bash
python3 scripts/serve.py --threshold -1.385
```

Those numbers come from 127 examples written by one person. Treat them as a
starting point and re-measure on your own traffic — logging scores from day one
is what makes that possible later.

---

## Operational notes

**Fail closed, everywhere.** If the embedding endpoint is down, hold. The sidecar
already does this; make sure your caller treats a non-200 the same way.

**Do not send the gate's input anywhere else.** It receives exactly the text you
were unsure about. Keep the endpoint on loopback, keep it out of request logs,
and log the score rather than the text.

**Measure the latency before you commit to it.** One `bge-m3` forward pass on a
quiet machine is tens of milliseconds; on a shared or loaded box it is not. A gate
that adds a second to every call will be turned off whatever it catches, which is
the same failure as too much friction.

**`bge-m3` only.** The head is fitted to that embedding space. Another model of
the same width would produce confident nonsense rather than an error, so pin it.
