# privacy-gate-llm

**[Try it in your browser](https://huggingface.co/spaces/YauhenBichel/privacy-gate-llm-demo)** (nothing you type
leaves the page; mirror on [GitHub Pages](https://yauhenbichel.github.io/privacy-gate-llm/)) ·
`pip install privacy-gate-llm` · `docker run ghcr.io/yauhenbichel/privacy-gate`

> Formerly published as `molecare-privacy-gate` (1.0.0 and 1.0.1). Same code, same `privacy-gate` command and
> `privacy_gate` import; new releases go to `privacy-gate-llm` only. To move: `pip uninstall molecare-privacy-gate`
> then `pip install privacy-gate-llm`.

**privacy-gate is an open-source PII and PHI detection gate for LLM prompts.** Before a prompt leaves your machine
for a cloud model (Claude, OpenAI, DeepSeek), it answers one question: must this text stay local? It catches health
data, personal data and credentials written as ordinary English, which regex rules cannot see, and it is built for
developers who run a **local LLM router** and want a **prompt privacy** check or **LLM guardrail** in front of the
cloud: data loss prevention for prompts, as a second gate that may only ever add a hold. A small model, `bge-m3` plus
1,024 weights, that runs on a CPU, with a Python package, a CLI, an HTTP sidecar and a Docker image.

[![PyPI](https://img.shields.io/pypi/v/privacy-gate-llm.svg)](https://pypi.org/project/privacy-gate-llm/)
[![GHCR](https://img.shields.io/badge/ghcr.io-privacy--gate-2496ED?logo=docker&logoColor=white)](https://github.com/YauhenBichel/privacy-gate-llm/pkgs/container/privacy-gate)
[![PR checks](https://github.com/YauhenBichel/privacy-gate-llm/actions/workflows/pr-checks.yml/badge.svg?branch=main)](https://github.com/YauhenBichel/privacy-gate-llm/actions/workflows/pr-checks.yml)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-try%20it-yellow)](https://huggingface.co/spaces/YauhenBichel/privacy-gate-llm-demo)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-demo%20mirror-222?logo=github)](https://yauhenbichel.github.io/privacy-gate-llm/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.11-brightgreen.svg)](https://www.python.org/)
[![Dependencies](https://img.shields.io/badge/dependencies-none-lightgrey.svg)](#install)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97-model-yellow.svg)](https://huggingface.co/YauhenBichel/privacy-gate-llm)
[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97-dataset-yellow.svg)](https://huggingface.co/datasets/YauhenBichel/privacy-gate-gold)
[![Contributors](https://img.shields.io/github/contributors/YauhenBichel/privacy-gate-llm)](https://github.com/YauhenBichel/privacy-gate-llm#contributors)

A very small model that answers one question about a piece of text:

> **Must this stay on this machine?**

**[Try the demo](https://huggingface.co/spaces/YauhenBichel/privacy-gate-llm-demo)** ·
**[Model on Hugging Face](https://huggingface.co/YauhenBichel/privacy-gate-llm)** ·
**[Using it in your own project](docs/INTEGRATION.md)**

> **Not a medical device.** This is a privacy tool. It decides whether text should
> leave a machine and says nothing about what a lesion is. MoleCare does not
> diagnose.

It exists to sit in front of [`llm-harness`](https://github.com/YauhenBichel/llm-harness)
as a **second** gate, behind the regex ruleset in `rules/privacy-v2.yaml` (compiled to
`src/privacy_gate/guard/privacy-v2.json`, which is what the guard loads), and to catch what patterns cannot see.

## What kind of model this is

**A text classifier on top of a bidirectional encoder.** Not a generative model:
it never writes a word. `BAAI/bge-m3` turns a piece of text into 1024 numbers,
and a linear head with 1024 weights turns those into one score, compared against
a threshold of 0.1209.

The encoder belongs to the family the diagrams label **MLM** — pretrained by
masking words and predicting them, like BERT, which is what makes it good at
understanding text rather than continuing it. Only the head is trained here; the
encoder is used as it ships. That is why the whole decision costs tens of
milliseconds on a CPU, and why swapping the embedding model breaks it — the head
has exactly as many weights as bge-m3 has dimensions.

## Why a model at all, when there are already rules

The rules are good at anything shaped like a marker: `sk-ant-…`, `AKIA…`,
`patient_id: 40219`, a base64 image. They are blind to the same information
written as ordinary English.

This sentence passes every privacy rule in the ruleset today:

> *"The woman from Tuesday's clinic, 34, has a 7mm asymmetric lesion on her left
> shoulder and a biopsy booked for the 20th."*

Measured on the 206-example gold set in `data/gold.jsonl`:

| | catch rate | friction rate |
|---|---|---|
| regex ruleset alone | **9.0%** (10 of 111) | 1.1% |

**101 of the 111 sensitive examples go straight through.** That is the gap this
model is for.

## What works

Not a fine-tuned chat model. **An embedding model and a logistic head.**

| | AUC | catch | friction |
|---|---|---|---|
| regex ruleset alone | — | 9.0% | 1.1% |
| hashed word unigrams | 0.7818 | 98% | 83.2% |
| `Qwen3.5-0.8B`, prompted | 0.4609 | *at chance* | |
| **`bge-m3` + logistic head, v1** | **0.9927** | **99.1%** | **5.3%** |

Five-fold cross-validated: every score comes from a head that never saw that
example. **110 of 111 sensitive examples caught, at 5.3% friction.** On 20 sentences
written *after* training (`data/fresh.jsonl`) it catches all 11 sensitive ones
and holds 1 of 9 harmless ones. On 21 short prompts held out of training
(`data/short-probe.jsonl`) it catches all 6 sensitive ones and holds none of the
15 harmless ones; head v0 held 8.

Prompting a small chat model asks it to reason its way to a word, and a 0.8B
cannot. An embedding does one forward pass and a linear head decides. That is
also the right shape for something that runs inline on every request: no
generation step, and the output is a calibrated score, so catch against friction
is a dial rather than whatever the model felt like saying.

The whole gate is `bge-m3` plus **1024 weights and a bias** — `model/head-v1.json`.
It costs **34–43 ms** per call at p50 and under 50 ms at p95, measured on a
loaded shared box, and 18 ms per text when batched.

## Install

```bash
pip install privacy-gate-llm        # standard library only; the embedding comes from an endpoint you run
ollama pull bge-m3                  # the default backend: Ollama on 127.0.0.1:11434

privacy-gate check "the woman from Tuesday's clinic has a 7mm lesion on her shoulder"
# hold  score=6.7200 threshold=0.1209 margin=6.5991
```

To install the latest `main` instead: `pip install "privacy-gate-llm @ git+https://github.com/YauhenBichel/privacy-gate-llm"`.

Three places the embedding can come from, one head:

| Backend | Command | Needs |
|---|---|---|
| `ollama` (default) | `privacy-gate check --backend ollama --url http://127.0.0.1:11434 "..."` | Ollama with `bge-m3` |
| `openai` | `privacy-gate check --backend openai --url http://127.0.0.1:11500 "..."` | any OpenAI-compatible `/v1/embeddings`: a gateway, vLLM, LM Studio, llama.cpp; a key via `PRIVACY_GATE_API_KEY` if it wants one |
| `local` | `pip install 'privacy-gate-llm[local]'` then `privacy-gate check --backend local "..."` | sentence-transformers; downloads `BAAI/bge-m3` (2.2 GB) once; CPU is enough |

The head was fitted on Ollama's output. Whether the other two backends give the same verdicts is measured, not
assumed: `training/backends_agree.py`, result in `docs/JOURNAL.md`. Every backend must return unit vectors; the
gate refuses anything else rather than score it.

```python
from privacy_gate.gate import Gate

gate = Gate.load()                      # the head shipped with this version, Ollama on 127.0.0.1:11434
gate.decide("the woman from Tuesday's clinic has a 7mm lesion on her shoulder")
# Decision(hold=True, score=6.72, threshold=0.1209)

from privacy_gate.backends import make_embedder
gate = Gate.load(embedder=make_embedder("openai", url="http://127.0.0.1:11500"))
gate = Gate.load(embedder=make_embedder("local"))
```

`privacy-gate serve` is the loopback HTTP sidecar for other languages (below); `privacy-gate check
--fail-on-hold` exits 3 on a hold, for CI; `privacy-gate info` prints the head's SHA-256, so what you run is
what was measured. A container with the in-process encoder: `docker run --rm -p 127.0.0.1:8231:8231
ghcr.io/yauhenbichel/privacy-gate` (built by the release workflow from `Dockerfile`).

Inside tools you already run: a **LiteLLM** guardrail that sends a held request to a local model instead of
the cloud one, an **Open WebUI** filter, and a **GitHub Action** (`uses: YauhenBichel/privacy-gate-llm@v1`) that
fails a pull request whose added lines read like health data, credentials or personal data written as prose.
See [`integrations/`](integrations/).

**It is not ready to ship.** 247 examples written by one person are
enough to choose an architecture, not enough to set a threshold that decides what
leaves a machine holding real patient data. See the end of `docs/JOURNAL.md` for
what would have to be true first.

## On data other people made

The 99 % above is an in-domain number. Run 12 in `docs/JOURNAL.md` scores the
shipped head, unchanged, on three public sets beside the baselines everyone
knows (`training/public_sets.py`; only ids and scores are committed, the texts
stay with their owners):

| Set | System | Catch | Friction | AUC |
|---|---|---|---|---|
| OpenShift router corpus, EN, 332 prompts | **head** | **0.70** | 0.14 | 0.87 |
| | Presidio (identifying entities) | 0.26 | 0.04 | |
| | GLiNER-PII small | 0.38 | 0.15 | |
| | regex rules | 0.02 | 0.00 | |
| | Presidio + head | 0.80 | 0.16 | |
| OpenShift router corpus, IT, 314 prompts | **head** | **0.83** | 0.23 | 0.87 |
| piimb PII benchmark, 2,000 sentences | **head** | 0.93 | **0.45** | 0.87 |
| | Presidio (identifying entities) | 0.59 | 0.01 | |
| | GLiNER-PII small | 0.83 | 0.07 | |

What it shows, in three lines. On the 60 prompts that corpus's author wrote
with **no marker to match**, the head catches 0.93 where Presidio catches
0.00: the prose gap is real and this is the only system here that sees it.
The entity tools catch the bare names the head misses, so the two compose.
And the shipped threshold is a gold-set threshold: on formal text from
personal documents (piimb's clean half) the head holds 45 %, and at 5 %
friction its catch on these sets is 0.37 to 0.47. Those are the hard
negatives the gold set lacks; see `CONTRIBUTING.md`.

## How it composes

The model may only ever **add** a hold, never clear one.

```
text ──▶ regex ruleset ──▶ hold?  ──yes──▶ local, locked
              │
              no
              ▼
         this model ──▶ hold?  ──yes──▶ local, locked
              │
              no
              ▼
      the caller's own routing stands
```

Two consequences, and they are the point:

- A **false negative** from the model leaves today's protection exactly as strong
  as it already is. The model cannot make things worse.
- The model is never the reason something *is* sent, only ever a reason something
  is held back.

So it is not judged on accuracy in isolation. It is judged on **residual catch**
(what it finds that the rules miss) against **friction** (how often it interrupts
work that was fine). A guard that fires on ordinary engineering gets switched off,
and then it protects nothing.

## Where things are

| Path | What |
|---|---|
| `docs/TAXONOMY.md` | The decision boundary. The spec every label obeys, and the owner decisions still open. |
| `docs/JOURNAL.md` | Every run, with its numbers, in order. Including the ones that did not work. |
| `data/gold.jsonl` | 206 hand-written examples: 95 clean, 111 sensitive. The test set. **Never trained on.** |
| `data/rules-baseline.json` | What the regex ruleset catches, per example. Generated by `training/rules_oracle.py`. |
| `docs/INTEGRATION.md` | How to use it from llm-harness, molecare-mcp, Spring Boot, CI, or over HTTP. |
| `data/fresh.jsonl` | 20 examples written *after* training, as a generalisation check. |
| `data/short-probe.jsonl` | 21 short prompts held out of training: the ones that showed v0 holding "ok" and "thanks!". |
| `model/head-v1.json` | The gate: 1024 weights and a bias. 68 KB. |
| `model/head-v0.json` | The previous head, kept so runs 6–9 stay reproducible. |
| `src/privacy_gate/` | What the wheel ships: the head, the embedding backends, the CLI, the sidecar (`privacy-gate serve`) and the guard. Standard library only. |
| `training/` | Maintainer tools, not in the wheel: embedding the gold set, fitting and cross-validating the head, the LLM baselines and every measurement. Run from a checkout. |
| `scripts/` | `build_rules.py` (compiles `rules/privacy-v2.yaml`) and `serve.py` (the old call shape of `privacy-gate serve`). |
| `integrations/` | The gate inside LiteLLM (a guardrail that routes or blocks), Open WebUI (a filter) and any language (the sidecar). |
| `action.yml`, `action/` | The GitHub Action: fails a pull request whose added lines read like health data, credentials or personal data in prose. |
| `training/public_sets.py`, `runs/public-*` | The head on data other people made, beside Presidio, GLiNER-PII and the rules (run 12). |

## Running it from a checkout

Everything here is standard library only. Retraining and measuring live in
`training/`, outside the wheel; run them from the repository root against the
package installed in editable mode. The one external dependency is an
[Ollama](https://ollama.com) endpoint serving `bge-m3`, which can be your own
machine (or, for the gate alone, any of the three backends above):

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e .                        # the package, editable; still no dependencies
ollama pull bge-m3

# Embed the gold set, then cross-validate the head over it.
python -m training.embed \
    --data data/gold.jsonl --out runs/gold-bge-m3.jsonl
python -m training.head \
    --embeddings runs/gold-bge-m3.jsonl --rules data/rules-baseline.json
```

Point it elsewhere with `--url http://host:11434` if the models live on another
box. If that box is shared, `training/run-when-free.sh` waits for it to go idle
rather than queueing behind whatever else is running. `training/README.md` lists
every tool and the old command each one replaces.

The no-model baseline needs nothing at all, and is worth running first so you
know what a neural result has to beat:

```bash
python -m training.lexical_baseline
```

Regenerating `data/rules-baseline.json` needs the `llm-harness` CLI, which is a
separate project; the generated file is committed so you do not need it.
`llm-harness explain` decides locally and sends nothing anywhere, which is why it
is safe to run over a file of fixtures.

## Data

Every example in `data/gold.jsonl` is **invented**. No real person, no real
credential, no real patient. Phone numbers come from Ofcom's drama range
(`07700 900xxx`), the AWS key is Amazon's own documentation placeholder, and the
private key fixture is base64 for a sentence saying it is not a key.

That is a deliberate property, not a convenience: it is what allows this
repository to be public at all. The same files are on the Hub as
[`YauhenBichel/privacy-gate-gold`](https://huggingface.co/datasets/YauhenBichel/privacy-gate-gold), with the
taxonomy and the rules baseline beside them. `SECURITY.md` explains why each fixture is safe,
and `CONTRIBUTING.md` explains what to do if you add one.

## Status

**Nothing consumes this in production yet.** `docs/JOURNAL.md` records what has
actually been measured, including the two architectures that did not work, and
ends with the five things that would have to be true before it should guard
anything real. `docs/TAXONOMY.md` lists the decisions still open.

Contributions that would help most are in `CONTRIBUTING.md`; the short version is
**more hard negatives** — text that looks sensitive and is not.

## Where it is used and mentioned

Used, behind a switch, by two of our own tools:

- [`llm-harness`](https://github.com/YauhenBichel/llm-harness): step 1b of the
  router, after the regex ruleset and before anything is sent to a cloud model.
- [`molecare-mcp`](https://github.com/MoleCare/molecare-mcp): checks tool results
  before they leave the machine when `PRIVACY_GATE_URL` is set; off in every
  published install.

Mentioned elsewhere:

- The National Research Council of Canada's
  [AI Safety Radar](https://nrc-cnrc.github.io/ai-safety-radar-securite-ia/)
  covered the v1.0.0 release in its
  [9 October 2026 digest](https://github.com/nrc-cnrc/ai-safety-radar-securite-ia/blob/main/digests/2026-10-09/safety-community.md).
- On PyPI as [`privacy-gate-llm`](https://pypi.org/project/privacy-gate-llm/), and
  since 8 October 2026 under its former name
  [`molecare-privacy-gate`](https://pypi.org/project/molecare-privacy-gate/).

If you use it somewhere, open an issue and it goes on this list.

## Contributors

Thank you to everyone who has helped privacy-gate-llm. The most useful
contribution is a **hard negative**: text that looks sensitive and is not.

<!-- readme: contributors,bots/- -start -->
<p align="center">
  <a href="https://github.com/YauhenBichel" title="Yauhen Bichel" aria-label="Yauhen Bichel"><img src=".github/faces/YauhenBichel.svg" width="87" height="99" alt="Yauhen Bichel" /></a>
</p>
<!-- readme: contributors,bots/- -end -->

The list is filled by [Contributors](./.github/workflows/contributors.yml) from
GitHub commits, bots omitted — never hand-maintained, because a stale list is
worse than none. [Contributor graph](https://github.com/YauhenBichel/privacy-gate-llm/graphs/contributors) ·
[good first issue](https://github.com/YauhenBichel/privacy-gate-llm/labels/good%20first%20issue)

## Licence

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). The base encoder,
[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3), is MIT and is neither included
nor redistributed here — only the head is ours.
