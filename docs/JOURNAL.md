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

## Run 4 — is it the size, or the task? (2026-09-12, in progress)

If a large model scores well, the taxonomy and prompt are sound and the problem
is capacity: the answer is then a bigger small model, or distillation from the
large one. If a large model *also* fails, the fault is mine — in the boundary,
the prompt, or the gold labels — and no amount of training fixes that.

A first attempt at `gpt-oss:20b` returned AUC exactly 0.5000 with a single
distinct margin. That is not a result, it is a broken measurement: neither `KEEP`
nor `SEND` appeared anywhere in the top 20 tokens at the first generated position
for any of the 127 examples, so every margin was the same floor value. `gpt-oss`
uses the harmony format and puts control tokens where the answer is expected.

The extraction needs to scan generated positions for the first one carrying the
answer rather than assuming position 0. Until it does, there is no ceiling
number, and Run 4 is open.
