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

All runs use `data/gold.jsonl`: 127 hand-written examples, 55 clean and 72
sensitive. Greedy decoding, `temperature 0`, `seed 0`, thinking off.

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

> ### AUC 0.7391

| target | friction |
|---|---|
| catch ≥ 98% | 85.5% |
| catch ≥ 95% | 65.5% |
| catch ≥ 90% | 63.6% |

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

**The bar is 0.7391, not 0.5.** Any embedding or fine-tuned model has to beat a
baseline that costs nothing to run. Quoting a neural result against the regex
ruleset alone would have overstated it.

**`Qwen3.5-0.8B` is worse than a bag of words.** AUC 0.4609 against 0.7391. That
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

> ### AUC 0.9924

The model on its own, which is the honest view of what the head contributes:

| threshold | catch | friction | leaks |
|---|---|---|---|
| −1.385 | **100.0%** | **16.4%** | **0** |
| −0.455 | 98.6% | 12.7% | 1 |
| −0.232 | 97.2% | 9.1% | 2 |

**Every one of the 72 sensitive examples is caught at 16.4% friction, with no
leaks at all.** For comparison, on the same set:

| | AUC | catch | friction |
|---|---|---|---|
| regex ruleset | — | 13.9% | 5.5% |
| hashed unigrams | 0.7391 | 98% | 85.5% |
| `Qwen3.5-0.8B` prompted | 0.4609 | — | — |
| **bge-m3 + logistic head** | **0.9924** | **100%** | **16.4%** |

### The remaining friction is mostly not the model's

At the 98.6% operating point the composed system holds back 10 clean examples.
**Three of those are the regex rules, not the head** — `cln-050`, `cln-051` and
`cln-052`, the documentation-placeholder cases from `docs/TAXONOMY.md` D3. The
head scored all three comfortably clean (−4.29, −3.40, −1.73) and cannot release
them, because it may only add holds. Fixing those means changing `v1.yaml`.

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

## What is decided, and what is not

**Decided.** The rules miss 62 of 72 sensitive examples, so the gap is real. The
gold set is not separable by length or by vocabulary alone. Prompting a small
chat model does not work: `Qwen3.5-0.8B` scores 0.4609, at chance, measured three
ways, and worse than hashed unigrams at 0.7391.

**Embeddings plus a logistic head do work.** AUC 0.9924 cross-validated, every
sensitive example caught at 16.4% friction with zero leaks, and 18 of 20 on
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
2. **Latency, measured.** One `bge-m3` forward pass per request, on a box that is
   sometimes heavily loaded. If the gate adds a second to every call it will
   be turned off, whatever it catches.
3. **A decision on where it runs.** A remote Ollama is a network hop and a
   shared queue. The head is 3k floats and could run anywhere; the embedding
   model is the constraint.
4. **The three D3 rules cases.** They are the whole of the composed system's
   avoidable friction, and the head already scores them correctly. That is a
   `v1.yaml` change, not a model change.
5. **`docs/TAXONOMY.md` D1 and D2 confirmed by the owner.** The model has learned
   whatever those defaults said, so changing them later means relabelling and
   refitting.
