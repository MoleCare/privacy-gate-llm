# Journal

Every run, in order, with its numbers. Including the ones that did not work —
especially those, because the negative results here are what stopped the project
spending a day training the wrong thing.

Two numbers recur:

- **catch rate** — of the sensitive examples, the share the gate holds back.
  A miss is an incident.
- **friction rate** — of the clean examples, the share the gate holds back.
  A guard that fires on ordinary engineering gets switched off, and then it
  protects nothing.

Runs 0–9 use the first 127 rows of `data/gold.jsonl`, 55 clean and 72
sensitive. Run 10 grows the set to 206, 95 clean and 111 sensitive, by
appending rows, so the first 127 are unchanged. Greedy decoding, `temperature 0`, `seed 0`, thinking off.

**Every figure here was re-measured after run 9**, which changed one example and
the ruleset it is scored against. Numbers describe the data and rules as they
stand, not as they were on the day each run was first written. Run 0 is the
exception and is deliberately left at its original values, because it is the
measurement that motivated the project.

---

## Run 0 — what the regex rules already do (2026-09-12)

The point of reference. `scripts/rules_oracle.py` puts every gold example through
`llm-harness explain` and records whether a privacy rule fires.

| | catch rate | friction rate |
|---|---|---|
| `rules/v1.yaml` | **13.9%** (10/72) | **5.5%** (3/55) |

The ruleset behaves exactly as designed, and the design has a shape:

- The 10 it catches are precisely the 10 `regex-visible` examples — the ones
  carrying a marker: `sk-ant-…`, `AKIA…`, `patient_id: 40219`, a JWT, a base64
  image. It catches every one of them.
- It catches **none** of the 62 written as ordinary English.
- The 3 it holds back wrongly are exactly the three predicted in
  `docs/TAXONOMY.md` D3: AWS's own documentation placeholder, a Jest assertion on
  `test@example.com`, and an OpenAPI example row.

So the gap is not a weakness in the patterns. It is what patterns are.
**62 sensitive examples out of 72 go straight through**, and that is the whole
opportunity.

*Run 9 later cut the 5.5% friction to 1.8% by adding exceptions upstream, with
no loss of catch. The 13.9% did not move, and could not: it is what a pattern
can see.*

---

## Run 1 — the 0.8B, asked for a category (2026-09-12)

`factory-qwen3.5-0.8b:q4_K_M`, the model factory's step-14 base. Four-way label
(`CLEAN` / `HEALTH` / `SECRET` / `PII`), decoding constrained to those four by an
Ollama `format` enum, so it cannot answer anything else.

| | catch rate | friction rate | leaks |
|---|---|---|---|
| model alone | 95.8% | **67.3%** | 3 |
| rules then model | 95.8% | 69.1% | 3 |

At first glance a 95.8% catch rate and 95.2% residual catch. **It is worthless,
and the distribution says why:**

| predicted | count |
|---|---|
| `HEALTH` | 106 |
| `CLEAN` | 21 |
| `SECRET` | 0 |
| `PII` | 0 |

It answered `HEALTH` for five sixths of everything and never once said `SECRET`
or `PII`, on a set containing 24 secrets and 19 pieces of personal data. A model
that always says `HEALTH` scores 100% catch and 100% friction. This one is barely
off that: it held back 37 of 55 clean examples, including "reformat this YAML and
sort the keys alphabetically" and "add a retry with exponential backoff around
the S3 upload".

**Lesson: a high catch rate proves nothing on its own.** Only the pair means
anything, and the prediction histogram is the first thing to look at.

Median 0.11s per call.

---

## Run 2 — the 0.8B, asked only the gate question (2026-09-12)

Same model, same set. The category is dropped and the model is asked for `KEEP`
or `SEND`, on the theory that four names was too much to hold and the gate only
ever needs one bit.

| | catch rate | friction rate | leaks |
|---|---|---|---|
| model alone | 100.0% | **100.0%** | 0 |

It answered `KEEP` for all 127. A constant function. Simplifying the question
made it collapse harder, not less.

Median 0.23s per call.

---

## Run 3 — the same 0.8B, scored rather than asked (2026-09-12)

Reading a word back throws away almost everything the model produced. Argmax is
just a threshold at zero, and zero is an arbitrary place to cut. So instead of
the word, take the margin at the first generated position:

```
margin = logP(KEEP) - logP(SEND)
```

Ollama reports `top_logprobs` **before** the format grammar is applied — tokens
the grammar forbids still appear in the list — so these are the model's raw
preferences, not a restatement of what it was allowed to say.

This separates two very different failures. A model that always answers `KEEP`
may still rank a clinical note above a Gradle question, and if it does, a
threshold recovers a working gate for nothing.

It does not.

> ### AUC 0.4609
>
> over 127 examples. 0.5 is a coin flip. Below 0.5 means the ranking is very
> slightly *inverted*.

The sweep confirms it: at every one of the 127 operating points, catch and
friction move together down the diagonal. There is no cut that buys catch more
cheaply than it buys friction.

The measurement itself is sound — all 127 examples produced both a `KEEP` and a
`SEND` logprob (0 absent) and 127 distinct margins.

**Conclusion, and it is a firm one: `Qwen3.5-0.8B` cannot do this task by
prompting, at any threshold.** Not a prompt-wording problem, not a
decision-boundary problem. There is no signal to move.

This cost about forty minutes, and it is the reason no LoRA has been trained
yet. Training a base whose ranking is at chance is a much larger undertaking
than nudging a boundary, and the next run has to establish whether the task is
expressible at all before anyone spends that time.

---

## Run 4 — is it the size, or the task? (2026-09-12, BLOCKED)

**This is the run that decides what happens next, and it has no number yet.**

If a large model scores well, the taxonomy and the prompt are sound and the
problem is capacity: the answer is then a larger small model, or distillation
from the large one. If a large model *also* fails, the fault is in the boundary,
the prompt, or the gold labels — and no amount of training fixes that. Nothing
should be trained until this is known.

Two attempts, neither of which produced a usable number.

**`gpt-oss:20b`** returned AUC of exactly 0.5000 with a single distinct margin.
That is a broken measurement, not a result: neither `KEEP` nor `SEND` appeared in
the top 20 tokens at generated position 0 for any of the 127 examples, so every
margin hit the same floor. `gpt-oss` uses the harmony format and puts control
tokens where the answer is expected. Fixed — `answer_position()` now scans for
the position actually deciding between the two words, and raises rather than
returning a floor. Not re-run.

**`gemma4:31b`** hangs. With `logprobs` it never returns; without them it managed
one example in ten minutes, and a bare "Say ok" did not answer inside 95 seconds
despite the model being resident with 21 GB in VRAM.

That is not a bug in this repository. The inference host was saturated: an
unrelated GPU job pinning fourteen cores, another client holding chat completions
open for six to thirteen minutes at a time, and Ollama started with `-np 1`, so
everything serialises behind them. Under that, requests returned `500` after two
minutes.

The runs were stopped rather than left to add to the queue, and nothing was
unloaded or re-pinned. `scripts/run-when-free.sh` exists because of this: on a
shared box, waiting is faster than queueing.

**To resume**, once the host is quiet:

```bash
PYTHONPATH=src python3 -m privacy_gate.calibrate \
    --model qwen3-coder:30b --rules data/rules-baseline.json --out runs/cal-ceiling.json
```

It resumes from its `.partial.jsonl`, so an interrupted run costs nothing. Avoid
`gemma4:31b` for this: its 256k vocabulary is the suspected reason `logprobs`
hangs.

---

## Run 5 — the no-model baseline, and an audit of the gold set (2026-09-12)

Run while the inference host was busy, because it needs nothing but Python. Hashed word
unigrams, 512 dimensions, L2-normalised, into the same logistic head, 5-fold
cross-validated. Hashing rather than a fitted vocabulary, because a vocabulary
built over the whole set before cross-validation leaks the evaluation fold.

### The audit first

The gold set was written by one person in one sitting. If the clean examples were
short imperatives about code and the sensitive ones long narratives about people,
any classifier could score well by learning that habit rather than the boundary.

| class | n | chars (median) | chars (mean) | tokens (median) | tokens (mean) |
|---|---|---|---|---|---|
| sensitive | 72 | 86 | 125.4 | 16 | 22.8 |
| clean | 55 | 83 | 79.4 | 14 | 14.2 |

**Medians are 86 against 83 characters and 16 against 14 tokens, so there is no
length artefact worth worrying about.** The gap in the means is larger and is
fully explained: the 8 `buried-*` examples are long by construction and all of
them are sensitive. Median is the honest statistic here.

### The baseline

> ### AUC 0.7429

| target | friction |
|---|---|
| catch ≥ 98% | 78.2% |
| catch ≥ 95% | 63.6% |
| catch ≥ 90% | 58.2% |

Real signal, and useless as a gate. Words like *biopsy*, *passphrase* and
*lesion* genuinely carry information, which is why it beats chance, but no
operating point is usable.

Where it fails is the interesting part:

| slice | gate accuracy at catch ≥ 98% |
|---|---|
| semantic-health / secret / pii | 100% |
| negative-fixture | **0%** |
| negative-dev | **7.7%** |
| negative-schema | **11.1%** |
| negative-secrets-talk | 25.0% |

It catches every semantic positive and fails almost every hard negative. That is
exactly the split the boundary predicts: a bag of words sees *password* in
"read the key from `ANTHROPIC_API_KEY`" and in "the password is my dog's name
followed by 1998" and cannot tell *talking about* from *containing*. That
distinction is the whole task, and it is the one thing lexical features cannot
represent.

### Two things this settles

**The bar is 0.7429, not 0.5.** Any embedding or fine-tuned model has to beat a
baseline that costs nothing to run. Quoting a neural result against the regex
ruleset alone would have overstated it.

**`Qwen3.5-0.8B` is worse than a bag of words.** AUC 0.4609 against 0.7429. That
is a sharper statement of run 3 than run 3 could make on its own.

**And the gold set holds up.** Had this scored 0.98, the set would have been too
easy or contaminated by style. Scoring 0.74, with the hard negatives failing and
the semantic positives passing, is what a well-made set looks like: the easy
signal is there, and the difficulty is concentrated exactly where the taxonomy
says the difficulty is.

---

## Run 6 — embeddings and a logistic head. This one works. (2026-09-12)

`bge-m3` (1.16 GB, 1024 dimensions) embeds each example; a logistic head over
those vectors decides. Five-fold cross-validated, so every score below comes from
a head that never saw that example. L2 = 1.0.

> ### AUC 0.9927

The model on its own, which is the honest view of what the head contributes:

| threshold | catch | friction | leaks |
|---|---|---|---|
| −0.823 | **100.0%** | **14.5%** | **0** |
| −0.519 | 98.6% | 12.7% | 1 |
| −0.065 | 97.2% | 3.6% | 2 |

**Every one of the 72 sensitive examples is caught at 14.5% friction, with no
leaks at all.** For comparison, on the same set:

| | AUC | catch | friction |
|---|---|---|---|
| regex ruleset | — | 13.9% | 1.8% |
| hashed unigrams | 0.7429 | 98% | 78.2% |
| `Qwen3.5-0.8B` prompted | 0.4609 | — | — |
| **bge-m3 + logistic head** | **0.9927** | **100%** | **14.5%** |

### The remaining friction is mostly not the model's

At the 98.6% operating point the composed system holds back 10 clean examples.
**Three of those were the regex rules, not the head** — `cln-050`, `cln-051` and
`cln-052`, the documentation-placeholder cases from `docs/TAXONOMY.md` D3. The
head scored all three comfortably clean and cannot release them, because it may
only add holds. Run 9 fixed two of the three upstream; `cln-052` remains and
cannot be fixed by a pattern.

The head's own seven: a null-check in a `Patient` entity, a photo retention
policy question, GitHub PAT rotation, an environment-variable placeholder, a CI
billing question, a support macro about a bleeding mole, and a question about how
many users are in the EU. All sit within one point of the threshold, and all are
recognisably "about the topic without containing anything", which is the
boundary's hardest edge.

The single leak at that threshold is `sec-006`, *"our internal API accepts the
shared secret letmein-2026 in the X-Auth header"*. At the 100%-catch threshold it
is caught.

### By slice

| slice | n | gate accuracy |
|---|---|---|
| semantic-health | 20 | 100% |
| semantic-pii | 16 | 100% |
| semantic-secret | 18 | 94.4% |
| buried-health / secret / pii | 8 | **100%** |
| regex-visible | 10 | 100% |
| negative-schema | 18 | 88.9% |
| negative-dev | 13 | 84.6% |
| negative-policy | 6 | 83.3% |
| negative-secrets-talk | 12 | 83.3% |
| negative-fixture | 6 | 50% (all three failures are the rules) |

The `buried-*` result is the one worth pausing on: 8 for 8 on long prompts where
a single clinical sentence or a password sits in the middle of a support queue, a
bug report, a test fixture or a log excerpt. That is the case regex has no answer
to and the one that actually happens.

---

## Run 7 — twenty examples written after training (2026-09-12)

Cross-validation can still flatter a set written in one sitting. So the head was
refitted on all 127 and then shown 20 sentences invented afterwards, in
`data/fresh.jsonl`. Nothing here was held out; it is simply new.

**18 of 20 correct, and 11 of 11 sensitive caught. No leaks.**

Margins on the sensitive ones are wide, from +2.44 to +10.53 against a −0.455
threshold, so these are not near misses.

Both errors are friction, and both are marginal:

| score | text | comment |
|---|---|---|
| just above | why is the Kotlin coroutine leaking on screen rotation? | a plain bug question. Probably *leaking*. |
| just above | the incident report says a photo was attached to a public issue by mistake | arguable — it describes a privacy incident but contains nothing. Counted as an error. |

Friction of 2 in 9 is higher than run 6's 16.4%, but on nine examples that is
noise, not a trend.

---

## Run 8 — latency, which was a blocking question (2026-09-12)

A gate that runs inline on every request and adds a second to each one gets
switched off, and a guard that is off protects nothing. That is the same failure
as too much friction, reached from the other side, so it belongs in the journal
next to the accuracy numbers rather than in a footnote.

Measured end to end — embed, standardise, dot, compare — with
`scripts/latency.py`, on a **shared box under real load** (1-minute load average
4.86, an unrelated video encode running). A quiet-box figure would be a nicer
number and a less useful one.

| text size | n | median chars | p50 | p95 | max |
|---|---|---|---|---|---|
| short | 270 | 66 | **34 ms** | 44 ms | 302 ms |
| typical | 325 | 92 | **36 ms** | 44 ms | 60 ms |
| long | 40 | 482 | **43 ms** | 46 ms | 58 ms |

Batched, which is the shape most callers should use: 16 texts in one call,
282 ms total, **18 ms each**.

Three things worth drawing out.

**p95 is under 50 ms at every size.** On a loaded box. This is not close to the
threshold where anyone would notice, let alone disable it.

**Cost is nearly flat in length.** 482 characters costs 43 ms against 34 ms for
66. A transformer encoder pays mostly for the forward pass, not the tokens, at
these sizes — so gating a whole support queue costs about what gating one
sentence does. That is the opposite of the chat-model path, where the buried
examples were the expensive ones.

**Cold start is 2.41 s**, and it is a model swap, not a per-request cost. It only
appears when something else has evicted `bge-m3`. On a machine where the gate is
the point, pin it.

The single 302 ms outlier is the shared box, not the model: the same text at the
same size has a 44 ms p95. Any caller should have a timeout and, per
`SECURITY.md`, treat a timeout as **hold**.

---

## Run 9 — fixing the rules instead of the model (2026-09-12)

Runs 0 and 6 both noted the same thing: most of the composed system's avoidable
friction was not the model. Three clean examples were held back by the *regex
ruleset*, and the head scored all three comfortably clean and could not release
them, because it may only add holds.

So the fix belonged upstream, in `llm-harness`.

### It was not "a `v1.yaml` change", as this journal previously claimed

That claim was wrong and worth correcting. The engine had no exception
mechanism at all: `firstMatch` fired on any pattern match, so there was nowhere
to express "this form is not a hit". It needed a small engine feature.

### Exceptions apply per match, never per prompt

This is the whole safety of the feature. If one allowed form could quiet a rule,
then

> "mail real.person@gmail.com, the fixture is test@example.com"

would scan clean — the fixture would excuse the real address beside it, and the
exception would have turned a working guard into a bypass. So every match is
enumerated and judged alone, and the rule still fires when any one survives.
Exceptions are also anchored, so `example\.com` cannot excuse
`victim@example.com.attacker.net`.

Two exceptions were added, both standards-based and narrow:

| rule | excuses | why it is safe |
|---|---|---|
| PRIV003 | `AKIAIOSFODNN7EXAMPLE` | AWS's own published documentation key. Not a credential and cannot become one. |
| PRIV005 | addresses at RFC 2606 / RFC 6761 reserved names | `example.com`, `.test`, `.invalid`, `.localhost` are never delegated, so no message reaches a person and no address identifies one. |

### It found a second bug, and then a third

**The generated ruleset dropped the field.** `scripts/build-rules.ts` picks
fields explicitly when building the JSON the engine actually reads, so `except`
was silently discarded and the exceptions did nothing. The YAML looked right and
the behaviour was unchanged.

**The conformance test did not test the router.** It decided whether a rule
fires with its own `rule.compiled.some(p => p.test(text))`, which skipped both
the folding in `scan.ts` and, once they existed, the exceptions. A fixture could
pass while production disagreed with it. There is now one exported function,
`matchIndex`, used by the router and by the test, and a test asserting the two
agree. A guard whose tests do not exercise the guard is worse than no tests,
because it reads as evidence.

### And it found a mislabelled example here

With the exceptions live, `rgx-009` stopped firing:

> "the account for ana.duarte@example.org cannot log in" — labelled `PII`

`example.org` is reserved, so it cannot be anyone's mailbox — and
`docs/TAXONOMY.md` rule 1 says documentation samples are `CLEAN`. **The label
was wrong by this project's own boundary**, and only looked right because the
rules over-fired. The same flaw sat in llm-harness's own fixture, which used a
reserved domain to stand for a real address.

Fixed by changing the example's text to a non-reserved domain, which is what it
always meant, rather than by relabelling it to make a number look better.

### Result

| | catch | friction |
|---|---|---|
| rules before | 13.9% | 5.5% |
| **rules after** | **13.9%** | **1.8%** |

**Two thirds of the rules' over-firing removed, with no loss of catch.** The one
remaining case is `cln-052`, `"patient_id": 12345` in an OpenAPI example, and it
cannot be fixed by a pattern: nothing in the text distinguishes it from a real
record other than the word "example" elsewhere in the sentence. That one is the
model's to catch, and it does — it scores it clean, and would release it if the
composition allowed releasing.

`llm-harness` went from 133 to 148 tests.


---

## Run 10 — short prompts, and head v1 (2026-09-13)

Switched on in llm-harness, head v0 held short harmless prompts. Sent through
`/route/explain`, 21 short prompts showed it:

| | sensitive caught | harmless held |
|---|---|---|
| v0 | 6/6 | **8/15** |

`ok` scored +3.73, `thanks!` +5.11, `Reply with exactly: ok` +4.23. Nothing
leaked, but a gate that holds `continue` and `yes` gets switched off. The gold
set's shortest row was 24 characters, so nothing that short had ever been
scored. A threshold could not fix it: clearing those prompts needs about +5.5,
and real sensitive sentences in the fresh set score +2.4.

### The probe is held out

Those 21 prompts are `data/short-probe.jsonl`, and no head is ever fitted on
them. Training on `ok` and then testing on `ok` would make any fix look perfect
and prove nothing. The training rows added below are different prompts of the
same shapes, and the script that added them refuses any row that duplicates the
probe.

### The first attempt taught the set that length predicts the label

The first pass added 40 short benign rows and 12 short sensitive ones. Length on
its own then scored **AUC 0.7014**, against 0.5813 on the original 127, and 41 of
the 55 rows under 40 characters were clean. A head fitted on that could learn
that short means safe, which would release exactly the short sensitive prompts
this run exists to protect, and inflate its cross-validated score while doing it.

27 more short sensitive rows fixed that:

| | length-only AUC | token-count AUC | under 40 chars, sensitive / clean |
|---|---|---|---|
| 127 rows | 0.5813 | 0.6261 | 1 row |
| 179 rows | 0.7014 | 0.7209 | 14 / 41 |
| **206 rows** | **0.6256** | **0.6389** | **41 / 41** |

### Baselines on 206 rows

- **Rules:** catch 10/111 (9.0%), friction 1/95 (1.1%). They fire on none of the
  79 new short rows, sensitive or not.
- **Hashed unigrams:** AUC **0.7818**, inside CI's 0.60–0.90 band.

### Head v1

Fitted on yserver, five-fold cross-validated.

> ### AUC 0.9927

That is v0's cross-validated AUC to four places, on a different set. It was
recomputed from the saved held-out scores (0.992698 both ways) and the weights
differ from v0's, so the coincidence is real rather than a stale file.

| threshold | catch | friction | |
|---|---|---|---|
| −1.455 | 100% | 12.6% | nothing leaks |
| **+0.121** | **99.1%** | **5.3%** | **shipped** |
| +0.881 | 90.1% | 2.1% | 11 leaks |

The shipped threshold moved from −0.519 to +0.121 because the score distribution
moved; the rule that picks it did not change. Friction is flat at 5.3% for every
catch level from 95% to 99%, so there is nothing to gain below the shipped point.

- **The one leak** is `sec-006`, *"our internal API accepts the shared secret
  letmein-2026 in the X-Auth header"*, at −1.38. v0 missed it too.
- **The five harmless rows still held:** the photo retention policy question, PAT
  rotation, `AWS_SECRET_KEY` missing from the vault, the support macro about a
  bleeding mole, and one new short row, *"give me a summary of the PR"*.
- **The new slices:** short sensitive 13/13 in each of health, secrets and
  personal data; short benign 39/40.

### The held-out sets decide it

| | v0 | v1 |
|---|---|---|
| short probe, harmless held | 8/15 | **0/15** |
| short probe, sensitive caught | 6/6 | 6/6 |
| fresh set, harmless held | 2/9 | **1/9** |
| fresh set, sensitive caught | 11/11 | 11/11 |

Every probe prompt v0 got wrong moved a long way: `ok` from +3.73 to −9.62,
`thanks!` from +5.13 to −5.21, `summarise the diff` from +0.58 to −8.37. None got
worse. On the probe the lowest sensitive score is +5.04, almost five points above
the threshold, and the highest harmless one is −4.20. On the fresh set the
thinnest sensitive margin is *"her scan showed nothing but the consultant wants to
review…"* at +1.13, one point above the threshold.

Run from llm-harness source with the head vendored, before any deploy: `ok`
−9.62 and `thanks!` −5.23 pass; a clinical sentence at +6.73 and
`my password is hunter2` at +10.35 are held.

Latency was not re-measured. The encoder and the arithmetic are the same, so run 8
still holds. `model/head-v0.json` stays in the repository, so runs 6–9 remain
reproducible.

---

## Run 11 — the package, and whether three encoders agree (2026-09-22)

Not a model run. The question was whether the gate could be *installed* by
someone who is not me, and whether the head, fitted on Ollama's `/api/embed`
output, gives the same verdicts when the embedding comes from somewhere else.

### What changed

`pip install privacy-gate`: the package, a `privacy-gate` command (`check`,
`serve`, `info`) and the head shipped inside the wheel, so `Gate.load()` needs
no path. Three backends behind the one head: Ollama (as before), any
OpenAI-compatible `/v1/embeddings`, and sentence-transformers in process
(`pip install 'privacy-gate[local]'`). The core stays standard library only; CI
now installs the wheel in an empty environment and fails if anything came with
it, or if the bundled head's SHA-256 differs from `model/head-v1.json`
(`47f18031…`). The sidecar moved into the package; its `/health` names the
backend and the hash. Exit codes for CI: 3 on a hold with `--fail-on-hold`, 1
when the gate could not score, which is a hold.

The Ollama backend takes `options`, and `num_gpu: 0` keeps the encoder on the
CPU. On a box with one GPU slot, an embedding request that loads `bge-m3` on the
GPU evicts whatever chat model somebody is using; this is the same rule the
gateway applies, and the reason the run below could happen on a busy machine.

### The three encoders, on the gold set

`scripts/backends_agree.py`, on the shared box, 206 rows, batches of 16.
Reference: Ollama's `bge-m3` (a GGUF), CPU. Against: the gateway's
`/v1/embeddings` (its own embedding server) and sentence-transformers with
`BAAI/bge-m3` from Hugging Face, CPU. Raw output:
`runs/backends-agree-gold-2026-09-22.json`.

| Backend | Same verdict as Ollama | Score delta, median | Score delta, largest | Cosine, smallest | Per text |
|---|---|---|---|---|---|
| Ollama `/api/embed`, CPU (reference) | — | — | — | — | 23.5 ms |
| gateway `/v1/embeddings` | **206 of 206** | 0.007 | 0.048 | 0.99998 | 18.1 ms |
| sentence-transformers, CPU | **206 of 206** | 0.007 | 0.048 | 0.99998 | 61.3 ms |

Two things worth knowing. First, the gateway and sentence-transformers agree
with each other to five decimals (largest score difference 0.00002): the
gateway's embedding server serves the Hugging Face weights, so those two are
one encoder, and the comparison that matters is GGUF against the original
weights. Second, the largest score delta between them, 0.048, sits against a
smallest margin of 4.3 on this set (the head was fitted on all 206 rows, so
in-sample margins are wide; the cross-validated numbers of run 10 are the
honest ones). A delta of 0.05 cannot flip a verdict here. It could flip one on
a text that sits within 0.05 of the threshold, which is exactly the text the
`margin` field exists to flag.

So the answer to condition 3 below, "a decision on where it runs", has become:
anywhere with any of the three, and the verdicts will match. The remaining
conditions stand.

### Not done

The container image is written (`Dockerfile`, the encoder baked in) but was
not built here; the release workflow builds it. PyPI trusted publishing needs a
one-time setup on PyPI by the owner before the first tag.

## Run 12 — the head on data other people made (2026-09-22)

Condition 1 below asks for an evaluation set not written by me. Three public
ones exist that ask a question close enough to ours, and none needs a GPU.
`scripts/public_sets.py` converts each into the gold-set row shape, scores the
head and the baselines everyone knows, and reports per slice. Only ids, labels,
slices, verdicts and scores are committed (`runs/public-*-2026-09-22.json`);
the texts stay with their owners and their licences.

| Set | What it is | Rows | Label map |
|---|---|---|---|
| OpenShift router corpus, English | The "privacy-plus" evaluation set of an OpenShift AI router (amedeos.github.io, July 2026): synthetic prompts, each `expect: local` or `sota`, in 15 categories including 60 `implicit` cases with no marker to match | 332: 191 local, 141 sota | local → HEALTH, SECRET or PII by category; sota → CLEAN |
| The same corpus, Italian | A strict 1:1 translation by its author, same ids | 314: 191 local, 123 sota | as above |
| piimb/pii-masking-benchmark (CC-BY-NC-4.0) | English sentences from personal documents with span labels; a balanced sample, seed 1 | 2,000: 1,000 with an identifying span, 1,000 with no span | span → PII; no span → CLEAN; sentences whose only spans are dates, titles or company names left out |

SPeDaC, the closest academic benchmark, is restricted: its files need a signed
agreement with the authors (Zenodo 8091725). Not done; the owner's call.

Systems: the head at its shipped threshold (0.1209), through the gateway's
embedding server on the box (the Hugging Face weights; run 11 showed they agree
with Ollama's); Presidio with its default recognisers, counted two ways (any
entity; an identifying entity only); `knowledgator/gliner-pii-small-v1.0` with
thirteen PII labels; the regex ruleset through `rules_oracle.py`; and the two
compositions the design intends.

### Results

| Set | System | Catch | Friction | AUC |
|---|---|---|---|---|
| OpenShift EN | **head** | **0.702** | 0.135 | 0.867 |
| | Presidio, identifying | 0.257 | 0.043 | |
| | GLiNER-PII small | 0.377 | 0.149 | |
| | regex rules | 0.016 | 0.000 | |
| | rules + head | 0.707 | 0.135 | |
| | Presidio identifying + head | 0.801 | 0.156 | |
| OpenShift IT | **head** | **0.832** | 0.228 | 0.870 |
| | GLiNER-PII small | 0.571 | 0.350 | |
| | regex rules | 0.016 | 0.000 | |
| piimb | **head** | **0.929** | **0.454** | 0.866 |
| | Presidio, identifying | 0.594 | 0.006 | |
| | GLiNER-PII small | 0.830 | 0.066 | |
| | regex rules | 0.265 | 0.000 | |
| | Presidio identifying + head | 0.976 | 0.456 | |

The head's catch when the threshold is moved to a fixed friction, on each set
alone: at 5 % friction 0.398 (OpenShift EN), 0.366 (IT), 0.468 (piimb); at 10 %:
0.623, 0.670, 0.596. Per-slice tables are in the `.report.json` files and in
the script's output.

### What it says

1. **The prose gap is real, and the head is the only system here that sees
   it.** On the 60 English `implicit` cases ("I've made an important personal
   decision that I want to keep private") the head catches 0.933; Presidio
   0.000, GLiNER 0.083, the rules 0.000. On `secret` the head catches 0.929
   where Presidio finds 0.071. This is the claim the project makes, on a set
   its author designed for exactly that gap, before this project existed.
2. **The entity tools see what the head misses.** A bare name ("Write a
   thank-you letter for Mario Rossi"): Presidio 1.000, the head 0.500. The two
   are complementary, and the composition Presidio-identifying + head reaches
   0.801 catch on the English set at 0.156 friction, against 0.702 for either
   alone at best.
3. **The threshold from the gold set does not transfer.** Friction on the
   OpenShift set is 0.135, on piimb 0.454: the head holds formal text from
   personal documents ("Please review the attached policy and send any
   questions by Friday") that the gold set never showed it. At a friction of
   5 % the head's catch on these sets is 0.37 to 0.47, nowhere near the 99 %
   on gold. The gold set's number is an in-domain number.
4. **Some disagreement is taxonomy, not error.** The corpus's author routes
   `finance` ("check the balance of the corporate bank account") and `legal`
   local; under our taxonomy those are CLEAN, and the head says so (catch
   0.375 and 0.200). `pii_weak` and `location` are sota there and the head
   holds them (friction 0.75, 0.29). The per-slice table separates the two
   kinds of miss; the headline numbers do not.
5. **Italian works without any Italian training**: 0.832 catch, 0.228
   friction, the same AUC as English. `bge-m3` is multilingual and the head
   does not care. GLiNER-PII small, English-trained, drops to 0.571 / 0.350.

### What changes because of it

The README's numbers gain a second table, this one, and the claim becomes:
the head catches what the rules and the entity tools cannot see, on someone
else's data; it also holds more than it should on text unlike its training
set, and the shipped threshold is a gold-set threshold. The hard negatives the
contributing guide asks for now have a description: formal personal
correspondence, notices, policies, letters. piimb's 1,000 clean sentences
are 454 of them, and a head fitted with them in the negative class is the
next run. Condition 1 below is met in part: an independent set exists and was
scored; the threshold has not been reset on it, on purpose, because that is a
policy decision with a friction cost the owner has to accept.

## What is decided, and what is not

**Decided.** The rules miss 62 of 72 sensitive examples, so the gap is real. The
gold set is not separable by length or by vocabulary alone. Prompting a small
chat model does not work: `Qwen3.5-0.8B` scores 0.4609, at chance, measured three
ways, and worse than hashed unigrams at 0.7429.

**Embeddings plus a logistic head do work.** Head v1: AUC 0.9927 cross-validated
over 206 examples, 110 of 111 sensitive ones caught at 5.3% friction, none of 15
held-out short harmless prompts held, and 18 of 20 on
sentences written after training with all 11 sensitive ones caught. The
architecture question is answered, and the answer is not the one the project
started out assuming.

Run 4 — the prompted chat ceiling — is now a curiosity rather than a blocker. It
would tell us what a large model could do with no training, which is interesting
but no longer on the critical path.

### What would have to be true before this ships

The result is strong and the evidence behind it is thin in one specific way:
**147 examples, all written by one person, in one day.** That is enough to choose
an architecture. It is not enough to set a threshold that decides what leaves a
machine holding real patient data.

1. **A larger, independently written evaluation set.** Ideally not by the same
   author. Real prompts from actual sessions would be better still, if they can
   be reviewed and redacted safely.
2. ~~**Latency, measured.**~~ **Done, run 8.** p95 under 50 ms at every text
   size on a loaded box, 18 ms each when batched, and nearly flat in length.
   Not a concern. Pin `bge-m3` where the gate matters, so the 2.4 s cold start
   never lands on a user.
3. **A decision on where it runs.** A remote Ollama is a network hop and a
   shared queue. The head is 3k floats and could run anywhere; the embedding
   model is the constraint.
4. ~~**The three D3 rules cases.**~~ **Two of three done, run 9.** Fixed
   upstream in `llm-harness` with per-match rule exceptions; friction there fell
   from 5.5% to 1.8% with no loss of catch. `cln-052` remains and cannot be
   fixed by a pattern.
5. **`docs/TAXONOMY.md` D1 and D2 confirmed by the owner.** The model has learned
   whatever those defaults said, so changing them later means relabelling and
   refitting.
