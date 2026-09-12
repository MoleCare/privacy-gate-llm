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

## Secret scanning will flag the evaluation data, and that is expected

Two fixtures trip GitHub's scanner, and both are meant to:

| | |
|---|---|
| `rgx-007` | a JWT — the canonical jwt.io example header and payload, with a fabricated signature |
| `rgx-008` | an RSA private key block whose body is base64 of an English sentence saying it is not a key |

Neither can be defanged without breaking what it tests. The PRIV003 pattern this
set is scored against needs two base64 segments to recognise a JWT, so a JWT
fixture has to look like a JWT. And a private-key fixture whose body plainly
reads "not a real key" stops reading as a secret to the *model* as well, which
would make its `SECRET` label wrong rather than the fixture safe — the same trap
that produced the mislabelled `rgx-009` (see `docs/JOURNAL.md` run 9).

`.github/secret_scanning.yml` therefore excludes `data/gold.jsonl` and
`data/fresh.jsonl`, and only those. `src/`, `scripts/` and `docs/` are still
scanned, because a real credential committed there would be a real problem.

**If an alert fires on anything outside `data/`, treat it as real until proven
otherwise.** That is the entire reason for keeping the exclusion this narrow.

## Running the evaluation

`scripts/rules_oracle.py` shells out to `llm-harness explain`, which decides
locally and sends nothing to any backend. That is why it is safe to run over a
file full of fixtures.

The evaluation itself sends every example to an Ollama endpoint. Point it only at
a model you host. Do not point it at a hosted API: the whole set would then leave
the machine, which is the exact thing this project exists to prevent.
