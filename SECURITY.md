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

## The privacy guard: threat model

The guard (in development for 2.0; design in
`docs/design/DESIGN-2026-11-20-privacy-guard.md`, decision record in
`docs/architecture/0002-guard-lives-with-the-detector.md`) checks what a coding
assistant is about to send to a model provider, and blocks or masks what it finds.
It is a net, not a proof. It is not a medical device, and it does not make a
provider fit to receive patient data.

**It protects against** a person or an agent *accidentally* sending secrets,
identifiers, contact details, clinical text or images to a provider, including
data riding in tool results, history, tool definitions, metadata, headers or
encoded forms (base64, hex, URL encoding, escapes).

**It does not protect against:**

| | Limit | What the guard does about it |
|---|---|---|
| L1 | Re-identification by combination: a role, a town, dates and a rare condition identify someone with no name present | health and identifier categories are blocked outright; four or more distinct maskable categories in one text are blocked (PG-COMBO-001); the rest is a documented residual |
| L2 | Paraphrased or transformed secrets: reversed, spelled out, split far apart, rot13, translated | common encodings and adjacent splits are decoded; the rest is beyond any pattern |
| L3 | Personal data in the model's reply | not the guard's job; replies may be reported, never altered |
| L4 | Prompt injection asking for the value behind a placeholder | the provider never had the value; an injection that makes an agent *send* data in a new form is L2 |
| L5 | Bypass: a client that ignores the base URL, a second tool, a deliberate user | custody mode and `doctor` narrow it; a malicious insider is out of scope |
| L6 | Detector misses: a new key format, an unusual name, a language the entity model handles poorly | layered detectors; every reported miss becomes a fixture |
| L7 | Files already uploaded and referenced by id | upload endpoints are not forwarded by default; references are audited |
| L8 | The guard sees everything, so it is a target | loopback only, no content in logs, pinned dependencies and actions, signed releases |

**Fail closed.** If the rules or the policy cannot be loaded, or an input is too
large or too deep to scan in full, or the engine raises, the request is refused.
Nothing the guard did not finish reading is sent.

**The guard's own records** hold no raw values: categories, rule ids, counts and
values hashed under a local key, truncated to 8 hex characters.

**Rules fixtures are scanned like everything else.** `rules/privacy-v2.yaml` and
the generated `rules/conformance.v2.json` hold hundreds of near-credentials, but
every value is low-entropy and obviously invented (AWS's documentation key,
`EXAMPLE-NOT-A-KEY`, `NOTAREAL` markers, published test card numbers, RFC 2606
domains, documentation IP ranges). They are deliberately *not* excluded from
secret scanning: an alert on them means a fixture is too realistic, and the
fix is a lower-entropy fixture, not an exclusion.

## Running the evaluation

`scripts/rules_oracle.py` shells out to `llm-harness explain`, which decides
locally and sends nothing to any backend. That is why it is safe to run over a
file full of fixtures.

The evaluation itself sends every example to an Ollama endpoint. Point it only at
a model you host. Do not point it at a hosted API: the whole set would then leave
the machine, which is the exact thing this project exists to prevent.
