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

That is not a bug in this repository. yserver was saturated:

| | |
|---|---|
| load average | 18.5 on 32 cores |
| `sd-cli` | 1470% CPU, generating images for `tiktok-posts-work` |
| another client | `/v1/chat/completions` calls of 6, 7 and 13 minutes |
| Ollama | started with `-np 1`, so everything serialises behind them |

Under that, requests returned `500` after two minutes. The runs were stopped
rather than left to add to the queue, and nothing was unloaded or re-pinned.

**To resume:** check the box is quiet (`uptime`, and no `sd-cli` in `ps`), then

```bash
ssh yserver 'cd ~/privacy-gate-llm && PYTHONPATH=src python3 -m privacy_gate.calibrate \
    --model gemma4:31b --rules data/rules-baseline.json --out runs/cal-31b.json'
```

It resumes from `runs/cal-31b.json.partial.jsonl`, so an interrupted run costs
nothing. Try `--model qwen3-coder:30b` if gemma4's 256k vocabulary turns out to
be what makes `logprobs` hang; `qwen3.6:35b` is the other candidate.

---

## What is decided, and what is not

**Decided.** The rules miss 62 of 72 sensitive examples, so the gap is real and
worth closing. `Qwen3.5-0.8B` does not close it by prompting: AUC 0.4609, at
chance, measured three ways.

**Not decided, and blocking everything downstream.** Whether a capable model can
do this task at all on this gold set. Until Run 4 has a number, there is no way
to tell a capacity problem from a specification problem, and training either way
would be guessing.

**If Run 4 comes back strong** (AUC above roughly 0.9), the path is: generate a
training set of a few thousand examples by construction — the generator is told
the category, so the label is known a priori and never guessed, the same trick
java-dsa-harness uses to avoid hand-written expectations — then either LoRA a
larger small base (Qwen3.5 2B or 4B) or, probably better for a classifier, put a
logistic head on `bge-m3` embeddings, which is already on the box. An embedding
plus a head has no generation step at all, which is the right shape for something
that must run inline on every request.

**If Run 4 comes back weak**, the gold set or the boundary is wrong, and the fix
is in `docs/TAXONOMY.md`, not in a training run.
