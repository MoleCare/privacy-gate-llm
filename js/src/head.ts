// SPDX-License-Identifier: Apache-2.0
// The same arithmetic as privacy_gate.head.Model.score and Gate._decide_vector in the Python package.
import { HEAD, HEAD_SHA256 } from "./generated/head.js";

export { HEAD, HEAD_SHA256 };

/** score > THRESHOLD means hold: the text must stay on this machine. */
export const THRESHOLD: number = HEAD.threshold;

export interface Decision {
  hold: boolean;
  /** Log-odds that the text must be held back. */
  score: number;
  threshold: number;
  /** score - threshold. Near zero means the gate is unsure: worth logging, never the text itself. */
  margin: number;
}

/**
 * Score one `bge-m3` embedding (1024 numbers, L2-normalised) against the bundled head.
 *
 * Throws on a vector of the wrong length or one whose norm is not 1 (within 0.01). The head was fitted on unit
 * vectors; an un-normalised one does not fail, it scores confidently wrong, so it is refused, as in Python.
 */
export function score(embedding: ArrayLike<number>): number {
  const n = HEAD.weights.length;
  if (embedding.length !== n) {
    throw new Error(`embedding has ${embedding.length} dimensions, head expects ${n} (bge-m3)`);
  }
  let squares = 0;
  for (let i = 0; i < n; i++) squares += embedding[i]! * embedding[i]!;
  const norm = Math.sqrt(squares);
  if (!(Math.abs(norm - 1) <= 0.01)) {
    throw new Error(
      `embedding is not L2-normalised (norm ${norm.toFixed(4)}). The head was fitted on unit vectors. ` +
        "With sentence-transformers pass normalize_embeddings=True; Ollama already does this.",
    );
  }
  let total = HEAD.bias;
  for (let i = 0; i < n; i++) {
    total += (HEAD.weights[i]! * (embedding[i]! - HEAD.mean[i]!)) / HEAD.stdev[i]!;
  }
  return total;
}

/** True when the text this embedding came from must stay local. */
export function hold(embedding: ArrayLike<number>): boolean {
  return score(embedding) > THRESHOLD;
}

/** Score, verdict and margin in one call, like the Python `Decision`. */
export function decide(embedding: ArrayLike<number>): Decision {
  const s = score(embedding);
  return { hold: s > THRESHOLD, score: s, threshold: THRESHOLD, margin: s - THRESHOLD };
}
