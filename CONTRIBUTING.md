# Contributing to privacy-gate-llm

Thanks for being here. This project is small, has no dependencies, and every
number in it is reproducible on a laptop — which makes it an unusually easy place
to make a real contribution.

## The one rule that is not negotiable

**Never add a real secret, a real person, or real patient data.** Not expired,
not redacted, not "just for a test".

Everything in `data/` is invented, and that is what allows this repository to be
public at all. If you add an example, invent it:

| Use | Not |
|---|---|
| `07700 900123` (Ofcom's drama range, never allocated) | a number that might reach someone |
| `AKIAIOSFODNN7EXAMPLE` (AWS's published placeholder) | a key from anywhere, however dead |
| a name you made up | a name from a ticket, a log, or a commit |
| a symptom you invented | anything a real person told you |

See `SECURITY.md` for why each existing fixture is safe.

## The second rule: numbers must describe the published data

Every figure in `docs/JOURNAL.md` is a ratio over `data/gold.jsonl`. Change the
data, the prompt, the features or the head, and those figures are wrong.

So if you touch any of those, **re-run the affected measurements and update the
journal in the same pull request**. The commands are in the README and none of
them take long. A PR that moves a number without moving the journal will be asked
to do so.

## The third rule: only cross-validated numbers get quoted

The head has 1024 parameters and there are 127 examples. A linear model with more
parameters than examples can separate almost any labelling, *including a random
one* — there is a test that proves it
(`test_in_sample_fit_separates_random_labels`). In-sample accuracy here is not a
weak result, it is a meaningless one.

`head.cross_validate` is the only function whose output belongs in the journal.

## Getting set up

There is nothing to install:

```bash
git clone https://github.com/MoleCare/privacy-gate-llm.git
cd privacy-gate-llm
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Python 3.11 or newer. The tests need no network and no model.

To run the parts that need embeddings you need an [Ollama](https://ollama.com)
endpoint with `bge-m3`:

```bash
ollama pull bge-m3
PYTHONPATH=src python3 -m privacy_gate.embed --data data/gold.jsonl --out runs/gold-bge-m3.jsonl
PYTHONPATH=src python3 -m privacy_gate.head --embeddings runs/gold-bge-m3.jsonl \
    --rules data/rules-baseline.json
```

## What is most useful

**More examples, especially hard negatives.** The gold set's weakest property is
that one person wrote all of it in a day. Text that *looks* sensitive and is not
is worth more than another obvious positive — ordinary engineering talk that
mentions passwords, patients or people without containing any.

**Examples where the gate is wrong.** If you find one, it is a contribution even
without a fix. Include the score (`Gate.decide` returns it) so it can be placed
against the threshold.

**A disagreement with `docs/TAXONOMY.md`.** The boundary is six rules and three
open decisions, and it was written by one person too. If a rule produces a label
you would not defend to a regulator, say so in an issue.

**Another embedding model.** The head is 60 lines and the pipeline is model
agnostic. A smaller or faster encoder that holds the AUC would be a direct
improvement.

## Style

- Standard library only in the core. A dependency there would need to justify
  itself against "it runs anywhere Python does". The one exception is the
  optional `local` extra (`sentence-transformers`), which is never imported
  unless that backend is asked for; CI installs the wheel in an empty
  environment and fails if anything else came with it.
- Comments explain *why*, not *what*. Several in this codebase exist because of a
  specific bug — keep that habit; it is why the journal is short.
- British spelling, to match the rest of MoleCare.
- One change per pull request.

## Medical scope

This is a privacy tool, not a clinical one. It decides whether text should leave
a machine. It must never be extended to say anything about what a lesion *is*.

MoleCare is not a medical device and does not diagnose.

## Licence

By contributing you agree your work is licensed under Apache-2.0, as in `LICENSE`.
