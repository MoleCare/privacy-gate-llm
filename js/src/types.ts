// SPDX-License-Identifier: Apache-2.0

/** `model/head-v1.json`: a logistic head over 1024-dimensional, L2-normalised `bge-m3` embeddings. */
export interface HeadFile {
  weights: number[];
  bias: number;
  mean: number[];
  stdev: number[];
  threshold: number;
  dimensions: number;
  embedding_model: string;
  embedding_model_hf: string;
  embeddings_l2_normalised: boolean;
  [key: string]: unknown;
}

export interface CategorySpec {
  title: string;
  action: "block" | "mask";
  overridable: boolean;
  surrogate: string;
  layers: string[];
  label?: string;
  [key: string]: unknown;
}

export type PatternSpec = string | { regex: string; group?: number };

export interface RuleSpec {
  id: string;
  title: string;
  category: string;
  patterns: PatternSpec[];
  validator?: string | null;
  except?: string[] | null;
  except_values?: (string | number)[] | null;
  aliases?: string[] | null;
  [key: string]: unknown;
}

/** `src/privacy_gate/guard/privacy-v2.json`, schema `privacy-gate-rules/2`. */
export interface RulesFile {
  schema: string;
  id: string;
  version: string;
  categories: Record<string, CategorySpec>;
  rules: RuleSpec[];
  [key: string]: unknown;
}
