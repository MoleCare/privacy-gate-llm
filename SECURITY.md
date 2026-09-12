# Security

## Reporting

Please report anything sensitive privately, through GitHub's "Report a
vulnerability" on this repository, rather than in a public issue.

## What this project is, in security terms

It is a **defence in depth** layer, and it is the weaker of the two layers.

`llm-harness`'s regex ruleset is the hard gate: deterministic, testable, and
first in the chain. This model runs second and may only ever **add** a hold. It
can never clear one. A model that is wrong therefore degrades to the protection
that already exists.

**Do not treat a `SEND` from this model as an assurance that text is safe.** It is
a statistical classifier with a measured miss rate, recorded in
`docs/JOURNAL.md`. It is a net, not a proof.

## Data in this repository

Every example in `data/` is invented. There is no real patient data, no real
personal data, and no live credential.

The fixtures deliberately *look* like credentials, because that is what a scanner
has to be tested against:

- `AKIAIOSFODNN7EXAMPLE` is AWS's own published documentation placeholder.
- The `sk-ant-…` string is a fabricated shape, not a key.
- Phone numbers come from Ofcom's `07700 900xxx` drama range, which is never
  allocated to a real subscriber.
- The `BEGIN RSA PRIVATE KEY` fixture contains base64 of an English sentence
  saying it is not a key.

If you add an example, invent it. Never paste something real, not even expired,
and never anything from a user, a ticket, or a log.

## Running the evaluation

`scripts/rules_oracle.py` shells out to `llm-harness explain`, which decides
locally and sends nothing to any backend. That is why it is safe to run over a
file full of fixtures.

The evaluation itself sends every example to an Ollama endpoint. Point it only at
a model you host. Do not point it at a hosted API: the whole set would then leave
the machine, which is the exact thing this project exists to prevent.
