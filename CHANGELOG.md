# Changelog

All notable changes. The head file has its own history in `docs/JOURNAL.md`; a new head is a new minor version.

## Unreleased

Renamed on PyPI: the distribution is now **`privacy-gate-llm`**, the same name as the repository, without the
MoleCare brand. 1.0.0 and 1.0.1 stay on PyPI as `molecare-privacy-gate`; later releases are published as
`privacy-gate-llm` only.

- `pip install privacy-gate-llm` (and `privacy-gate-llm[local]`). The `privacy-gate` command, the `privacy_gate`
  import, the container image `ghcr.io/yauhenbichel/privacy-gate` and the head are unchanged.
- Moving from the old name: `pip uninstall molecare-privacy-gate && pip install privacy-gate-llm`. Both install
  the same `privacy_gate` package, so do not keep both in one environment.
- Project URLs: `Homepage` is now the GitHub Pages site, https://yauhenbichel.github.io/privacy-gate-llm/ (also
  the `url` in `CITATION.cff`); the Hugging Face Space stays as `Demo`.
- Docs: README, `docs/INTEGRATION.md`, `docs/TAXONOMY.md` and the `Gate` docstring name the rules the guard
  loads, `rules/privacy-v2.yaml` (compiled to `src/privacy_gate/guard/privacy-v2.json`), instead of `rules/v1.yaml`.
- The live monitor installs `privacy-gate-llm`, and falls back to the former name until the first release under
  the new one is on PyPI.

Package split: the wheel carries the runtime only.

- **Breaking for anyone importing the training code from the wheel.** `privacy_gate.calibrate`, `.evaluate`,
  `.embed`, `.dataset`, `.labels`, `.prompt` and `.features` moved to `training/` in the repository, outside the
  wheel and the sdist, together with the fitting half of `privacy_gate.head` (`fit`, `cross_validate`, `auc`,
  `folds`, `sweep`, the CLI) and the chat client in `privacy_gate.ollama` (`chat`, `Reply`, `resident`, `pin`,
  `LABEL_SCHEMA`). Run them from a checkout as `python -m training.<tool>`. `training/README.md` maps the old
  commands to the new ones.
- Unchanged: `Gate`, `Decision`, `bundled_head_path`, `privacy_gate.head.Model`, `privacy_gate.backends`,
  `privacy_gate.ollama.OllamaError` and `DEFAULT_URL`, `privacy_gate.guard`, the bundled head, and the
  `privacy-gate` command. New: `python -m privacy_gate` is the same command.
- The measurement scripts moved from `scripts/` to `training/`. No script edits `sys.path` any more: they import
  the installed package (`pip install -e .` in a checkout). `scripts/build_rules.py` and `scripts/serve.py` stay.
- Development: `pip install -e . --group dev`. The pre-commit rules check now runs in that environment
  (`language: system`). Strict mypy covers `training/` and `scripts/` too, and coverage measures `training/`.
  New tests for the chat client and the `training.head` command raise the coverage floor from 82 to 84.

Guard, phase P0 (design `docs/design/DESIGN-2026-11-20-privacy-guard.md`):

- `rules/privacy-v2.yaml`, schema `privacy-gate-rules/2`: 12 categories, 17 rules with validators, the lite,
  standard and full profiles, and fixtures for every rule. Seeded from llm-harness ruleset v1 at its pull
  request 32; the old ids PRIV001-005 are kept as `aliases`.
- `scripts/build_rules.py` compiles it to `src/privacy_gate/guard/privacy-v2.json` (loaded by the engine,
  standard library only) and writes `rules/conformance.v2.json` (123 cases) for every engine.
- `privacy_gate.guard`: the rules loader and the L1 matcher, with an RE2-portability check.
- `privacy-gate info` also reports the ruleset id, version, SHA-256 and rule count.
- ADR 0002 and a threat-model section in `SECURITY.md`.

Guard, phase P1 (the core, standard library only, no proxy):

- `Guard.load(...).protect(body, caller=..., session=..., provider=...)` returns a decision (allow, mask, block
  or local) and the body to send. It walks the body as llm-harness pull request 32 does (`key: value` leaves,
  numbers too, image and document blocks held, depth, node and size caps), normalises and decodes (L0: zero-width,
  `%xx` and `\uXXXX` escapes, NFKC, dashes, base64 and hex to depth 2), matches the rules (L1), applies the policy
  (block categories never overridable; overrides and the allowlist only loosen mask to allow; providers only
  tighten), masks with HMAC surrogates from a per-session vault bound to the caller, and re-scans what it masked.
- Fails closed: a policy, rules file or overlay that cannot be loaded (or does not match its pinned SHA-256)
  refuses everything; any error while deciding is a block (`PG-SCAN-001`, alias `PRIV-SCAN`).
- `StreamUnmasker` puts real values back in a streamed reply, holding back a possible surrogate across chunks;
  `ToolArgsBuffer` unmasks tool-call arguments as JSON.
- `Decision` never carries a value; `AuditWriter` writes it as one 0600 JSON line, with 8-hex keyed value hashes.
- `rules/conformance.bodies.v2.json`: whole-body cases for every engine (21 cases).

## 1.0.1 — 2026-11-20

The repository moved from `MoleCare/privacy-gate-llm` to
[`YauhenBichel/privacy-gate-llm`](https://github.com/YauhenBichel/privacy-gate-llm). Old links redirect.
The code and the gate (`model/head-v1.json`) are unchanged from 1.0.0; this release only updates the project URLs
PyPI shows.

- PyPI: still `molecare-privacy-gate`; the `privacy-gate` command and the `privacy_gate` import are unchanged.
- Container image: now `ghcr.io/yauhenbichel/privacy-gate` (was `ghcr.io/molecare/privacy-gate`).
- GitHub Action: `uses: YauhenBichel/privacy-gate-llm@v1`. The release workflow now moves the `v1` tag to each
  1.x release, so `@v1` resolves (before, only `v1.0.0` existed).
- Homebrew: `brew install yauhenbichel/tap/privacy-gate`.

## 1.0.0 — 2026-10-08

Published on PyPI as `molecare-privacy-gate`: PyPI does not allow `privacy-gate`, which it treats as too
similar to an existing, unrelated project. The `privacy-gate` command and the `privacy_gate` import are unchanged.

The first packaged release. The gate itself, `model/head-v1.json`, is the head of 13 September 2026 (run 9),
unchanged: `bge-m3` plus 1,024 weights and a bias, AUC 0.9927 five-fold on the 206-example gold set.

- `pip install molecare-privacy-gate`: the package, a `privacy-gate` command (`check`, `serve`, `info`), and the head
  shipped inside the wheel so `Gate.load()` needs no path and no checkout.
- Three embedding backends behind one head: Ollama's `/api/embed` (as before, the default), any
  OpenAI-compatible `/v1/embeddings`, and sentence-transformers in process (`pip install 'molecare-privacy-gate[local]'`).
  `scripts/backends_agree.py` measures whether they agree on the gold set; the result is in the journal.
- The HTTP sidecar moved into the package (`privacy-gate serve`); `scripts/serve.py` stays as a wrapper.
  `/health` names the backend and the head's SHA-256. Unreachable model: 503 with `"hold": true`, as before.
- Exit codes for CI: `privacy-gate check --fail-on-hold` exits 3 on a hold, 1 when it could not score.
- A container image with the in-process encoder, and a release workflow (PyPI by trusted publishing, the
  GitHub release, ghcr.io).
- Standard library only stays true for the core; the optional extra is the only dependency anywhere.
- The demo Space scores text (Gradio, the encoder on the Space's CPU); the gold-set explorer stays as a tab.
- Run 12: the head on three public sets beside Presidio, GLiNER-PII and the rules (`scripts/public_sets.py`).
  The README says the 99 % is an in-domain number and shows the table.
- Integrations: a LiteLLM guardrail (route to a local model, or block), an Open WebUI filter, and the GitHub
  Action `MoleCare/privacy-gate-llm@v1` that annotates and fails a pull request on held lines.
- The gold set on the Hub: `YauhenBichel/privacy-gate-gold`, with the taxonomy and the rules baseline.

## Before 1.0.0

Runs 1 to 9 of `docs/JOURNAL.md`, 12 and 13 September 2026: the rules baseline, the lexical baseline, the
prompted small model at chance, head v0, the short-prompt gap, head v1.
