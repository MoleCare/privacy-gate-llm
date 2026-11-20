# 0002. The guard lives with the detector

- Status: accepted (design merged in #35; architecture review comment on #35, 20 November 2026)
- Date: 2026-11-20
- Numbering: 0002 as the design names it. There is no 0001 in this repository yet; the number is kept so that
  the design, the review and this record all say the same thing.

## Context

The privacy guard sits between a coding assistant and a model provider (Anthropic, OpenAI, DeepSeek) and keeps
secrets, identifiers, contact details and clinical text from leaving the machine by accident. It is mostly
detection, plus a thin layer that knows each provider's wire format.

This repository already holds detection: the prose head, its evaluation, a PyPI distribution, a security policy
and a release pipeline. llm-harness, the owner's private router, holds a second copy of the pattern rules
(`rules/v1.yaml`, PRIV001-005) and the walker that turns a request body into the text those rules read. Two
copies of rules drift; the architecture review found that the last five leaks fixed in llm-harness (its pull
request #32) were in the walker, not in the regexes.

## Decision

1. The guard grows inside this repository (design D1), under the MoleCare brand (D2).
2. The package owns, as public, versioned code and data:
   - the rules (`rules/privacy-v2.yaml`), compiled to `src/privacy_gate/guard/privacy-v2.json`, and the
     conformance suite (`rules/conformance.v2.json`) every engine must pass;
   - the validators, L0 decoding, policy, masking, unmasking and audit primitives;
   - **the body walker**: key and value read together as `key: value`, numbers and booleans read, image,
     document, audio and file blocks classified as unreadable, and depth and size caps that refuse rather than
     pass. Its semantics are those of llm-harness #32 (review R1).
3. The core stays standard library only. YAML is parsed by a development script, never at run time; heavy
   parts (the proxy, the entity model) are extras.
4. Rule ids carried over from llm-harness keep their old id as an alias (`aliases: [PRIV004]`), reported
   beside the new id (review R3).
5. A consumer that embeds the engine decides what a block means for it. For llm-harness, any exception or
   unexpected result from the engine is a block (review R9); the package cannot enforce that for its callers,
   so its own facade does the same and documents it.

llm-harness keeps its ordering (privacy decided first), its policy file, its lanes and credentials, its
headers, audit sink and dashboards, and its private overlays. Until the cut-over phase (P10), its `v1.yaml`
stays the live source for yserver and a fix lands there first.

## Consequences

- One rules file and one conformance suite, read by the Python engine here, the TypeScript engine (P8) and
  llm-harness after P10. A disagreement about a rule is settled by reading the YAML.
- `pip install privacy-gate-llm` keeps zero dependencies.
- Generated files are checked in and CI fails when they are stale; the generator needs PyYAML.
- A rule that fires less is a major version, so consumers take such bumps by reviewed pull request only.
- The repository's scope widens from "a small model" to "a small model and a guard"; the README and
  `SECURITY.md` say what each part does and does not do.

## Alternatives

- A new repository for the guard, depending on this one: a second release pipeline and security policy for one
  maintainer, and rules split from the detectors that use them. Rejected unless the owner wants the guard
  outside MoleCare (D2).
- Moving only the regexes and leaving the walker in llm-harness: the two walkers would drift exactly where the
  last leaks were. Rejected by the review (R1).
