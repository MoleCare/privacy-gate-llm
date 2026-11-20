// SPDX-License-Identifier: Apache-2.0
/**
 * privacy-gate-llm for JavaScript and TypeScript: the gate's head (score an embedding you already have) and
 * the guard's pattern rules (layer L1), with no runtime dependencies.
 *
 * The embedding itself must come from `bge-m3` (Ollama `/api/embed`, any OpenAI-compatible `/v1/embeddings`,
 * or transformers.js), L2-normalised. This package does not call a model.
 */
export { HEAD, HEAD_SHA256, THRESHOLD, decide, hold, score } from "./head.js";
export type { Decision } from "./head.js";
export { RULES, RULESET_SHA256, Rule, Ruleset, findSensitive, ruleset, translate } from "./rules.js";
export type { Finding, Span } from "./rules.js";
export { VALIDATORS } from "./validators.js";
export type { CategorySpec, HeadFile, PatternSpec, RuleSpec, RulesFile } from "./types.js";
