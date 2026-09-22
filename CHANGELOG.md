# Changelog

All notable changes. The head file has its own history in `docs/JOURNAL.md`; a new head is a new minor version.

## 1.0.0 — unreleased

The first packaged release. The gate itself, `model/head-v1.json`, is the head of 13 September 2026 (run 9),
unchanged: `bge-m3` plus 1,024 weights and a bias, AUC 0.9927 five-fold on the 206-example gold set.

- `pip install privacy-gate`: the package, a `privacy-gate` command (`check`, `serve`, `info`), and the head
  shipped inside the wheel so `Gate.load()` needs no path and no checkout.
- Three embedding backends behind one head: Ollama's `/api/embed` (as before, the default), any
  OpenAI-compatible `/v1/embeddings`, and sentence-transformers in process (`pip install 'privacy-gate[local]'`).
  `scripts/backends_agree.py` measures whether they agree on the gold set; the result is in the journal.
- The HTTP sidecar moved into the package (`privacy-gate serve`); `scripts/serve.py` stays as a wrapper.
  `/health` names the backend and the head's SHA-256. Unreachable model: 503 with `"hold": true`, as before.
- Exit codes for CI: `privacy-gate check --fail-on-hold` exits 3 on a hold, 1 when it could not score.
- A container image with the in-process encoder, and a release workflow (PyPI by trusted publishing, the
  GitHub release, ghcr.io).
- Standard library only stays true for the core; the optional extra is the only dependency anywhere.

## Before 1.0.0

Runs 1 to 9 of `docs/JOURNAL.md`, 12 and 13 September 2026: the rules baseline, the lexical baseline, the
prompted small model at chance, head v0, the short-prompt gap, head v1.
