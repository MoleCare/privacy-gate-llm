# training/

Maintainer tools: retrain the head and measure it. None of this is in the wheel or the sdist.
`pip install privacy-gate-llm` ships only `src/privacy_gate`, which is the runtime.

## Running them

From the repository root, with the package installed in editable mode:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e .                 # still no dependencies
python -m training.lexical_baseline
```

`python -m` puts the current directory on the import path, and that is how `training` is found. The package
itself comes from the environment. Nothing in this folder edits `sys.path`. On a host with no install step (the
inference box that `run-when-free.sh` drives), `PYTHONPATH=src` gives the same result without installing.

## What is here

| Module | What | Was |
|---|---|---|
| `embed` | Embed a labelled set through Ollama's `/api/embed`. | `privacy_gate.embed` |
| `head` | Fit the logistic head, cross-validate it, sweep the threshold, `--export` a head file. `Model` (scoring) stays in the package as `privacy_gate.head.Model`. | `privacy_gate.head` |
| `calibrate` | Score a chat model as a continuous decision, then choose the cut. | `privacy_gate.calibrate` |
| `evaluate` | Score a model on a labelled set, and in composition with the rules. | `privacy_gate.evaluate` |
| `chat` | The Ollama chat client `calibrate` and `evaluate` use (`chat`, `Reply`, `resident`, `pin`, `LABEL_SCHEMA`). It shares the retrying transport in `privacy_gate.ollama`. | part of `privacy_gate.ollama` |
| `prompt` | The prompt the untrained model is judged on. | `privacy_gate.prompt` |
| `dataset`, `labels` | Loading and checking the labelled sets; the four labels. | `privacy_gate.dataset`, `privacy_gate.labels` |
| `features` | Hashed-unigram features for the no-model baseline. | `privacy_gate.features` |
| `lexical_baseline` | The no-model baseline and the gold set's length audit. CI runs it. | `scripts/lexical_baseline.py` |
| `score_heldout` | Score a held-out set with a shipped head. | `scripts/score_heldout.py` |
| `backends_agree` | Whether the three embedding backends give the same verdicts. | `scripts/backends_agree.py` |
| `latency` | How long the gate takes, cold and warm, by text length. | `scripts/latency.py` |
| `public_sets` | The head on public data sets, beside Presidio, GLiNER-PII and the rules. | `scripts/public_sets.py` |
| `rules_oracle` | What the regex ruleset already catches (needs the `llm-harness` CLI). | `scripts/rules_oracle.py` |
| `run-when-free.sh` | Wait for a shared box to go idle, then run the embedding and calibration jobs on it. | `scripts/run-when-free.sh` |

## Old commands

`docs/JOURNAL.md` records each command the way it was run on its date. To repeat one now:

| Journal | Now |
|---|---|
| `PYTHONPATH=src python3 -m privacy_gate.<tool> ...` | `python -m training.<tool> ...` |
| `PYTHONPATH=src python3 scripts/<tool>.py ...` | `python -m training.<tool> ...` |

The code, the data and the head file are unchanged, so the numbers reproduce. Only the paths moved.

The rules compiler is not a training tool, so it stays at `scripts/build_rules.py`.
