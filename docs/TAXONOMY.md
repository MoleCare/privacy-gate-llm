# The decision this model makes

One question, asked of one piece of text:

> **Must this stay on this machine?**

The model answers with exactly one word.

| Label | Meaning |
|---|---|
| `CLEAN` | Nothing here identifies a person or unlocks anything. Safe to send. |
| `HEALTH` | Health information about a real, identifiable person. |
| `SECRET` | A live credential, or enough to derive one. |
| `PII` | Personal data about a real, identifiable person; not health, not a credential. |

The gate is binary: `CLEAN` means the caller's routing stands, anything else pins
the request to the local model. The specific label is for the audit line, so that
a refusal can be explained without re-reading the prompt.

When more than one applies, the precedence is **`HEALTH` > `SECRET` > `PII`**. It
only affects the logged label, never the gate.

## Why `HEALTH` and not `PATIENT`

Health data about an identifiable person needs the same care whether that person
is a MoleCare user, the author's aunt, or a colleague who is off sick with their
back. Naming the label after the *data* rather than after a role keeps the rule
learnable, and keeps this model useful to anyone who is not us.

## Where the line sits

These six rules decide every label in `data/gold.jsonl`. They exist because a
guard that fires on ordinary work gets switched off, and a guard that misses a
real leak was never worth having. Both failures are cheap to create by accident,
so the boundary is written down before any model sees it.

**1. A person must be real and identifiable.** Invented names, documentation
samples, test fixtures and seeded demo rows are `CLEAN`. Re-identifiable counts as
identifiable: "the only Polish-speaking dermatologist at the Reading clinic, born
in 1978" names someone as surely as a passport does.

**2. Talking *about* a category is `CLEAN`.** Designing a `dateOfBirth` column,
writing a validator for the ABCDE rule, setting a photo retention period, or
documenting how often a token is rotated — all `CLEAN`. The subject of the
sentence is the software, not a person.

**3. A credential is `SECRET` only when the value is present, or derivable from
what is written.** `DB_PASSWORD=${DB_PASSWORD}` is `CLEAN`. "The password is my
first dog's name followed by 1998" is `SECRET`, because the sentence carries the
value even though no key-shaped token appears. Naming where a secret lives is
`CLEAN`; naming what it is, is not.

**4. Aggregates are `CLEAN`.** "37% of users skip onboarding at screen 3" and
"0.81 AUC on the held-out split" describe a population, not a person.

**5. Health information about the author counts as `HEALTH`.** *(Owner decision
D1 — see below.)*

**6. Volunteered in the past does not make it `CLEAN` now.** The gate sees one
text and has no memory of consent.

## What this model is not

It is a **second** gate. `llm-harness/rules/v1.yaml` runs first and keeps
precedence; this model can only ever *add* a privacy verdict, never clear one.
That composition is deliberate:

- A false negative from the model leaves the existing regex protection exactly as
  strong as it is today. The model can only improve on it.
- The model is never the reason something *is* sent. It is only ever a reason
  something is held back.

So the honest measure of this model is not its accuracy in isolation. It is:
**how much does it catch that the rules miss, and how often does it interrupt
work that was fine?** `docs/JOURNAL.md` reports both, separately.

## Open decisions for the owner

**D1 — the author's own health.** Right now "I have a mole on my back that has
changed shape" is labelled `HEALTH`, so it stays local. The gate cannot tell whose
body is being described, and defaulting the other way would put the author's own
medical history on a third party's server. The cost is that asking a cloud model a
personal skin question is refused. *Default taken: `HEALTH`.*

**D2 — colleagues' health.** "He has been off sick since the 3rd with his back" is
`HEALTH` under rule 1. It is ordinary team chatter, and it is also special-category
data about an identifiable person under GDPR Article 9. *Default taken: `HEALTH`.*

**D3 — the rules already over-fire, and this model cannot fix it.** Three gold
examples (`cln-050`..`cln-052`) are text the regex ruleset flags and a careful
reader would not: an AWS documentation placeholder, a Jest assertion on
`test@example.com`, and an OpenAPI example row. Because the model may only add
verdicts, it cannot release them. If that friction matters, the fix is a rules
change, not this model.
