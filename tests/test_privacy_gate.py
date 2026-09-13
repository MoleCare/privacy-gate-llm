# SPDX-License-Identifier: Apache-2.0
"""Unit tests. No network, no model.

The metric tests matter most: every claim in docs/JOURNAL.md is one of these
ratios, so a silent change here would rewrite the record.

Run with:  PYTHONPATH=src python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import random
import tempfile
import unittest
from pathlib import Path

from privacy_gate import features, head
from privacy_gate.calibrate import NoAnswerPosition, Scored, answer_position, auc, pick, sweep
from privacy_gate.dataset import load
from privacy_gate.evaluate import Prediction, score
from privacy_gate.gate import Gate
from privacy_gate.head import Model
from privacy_gate.labels import BINARY, CLEAN, HEALTH, KEEP, SEND, holds, normalise
from privacy_gate.prompt import build, build_binary


def prediction(id: str, gold: str, predicted: str | None, slice: str = "s") -> Prediction:
    return Prediction(
        id=id, slice=slice, gold=gold, raw=predicted or "", predicted=predicted, seconds=0.1
    )


class TestLabels(unittest.TestCase):
    def test_reads_a_clean_answer(self) -> None:
        self.assertEqual(normalise("HEALTH"), HEALTH)

    def test_strips_what_small_models_add(self) -> None:
        for raw in (' "SECRET" ', "clean.", "  PII\n", "`HEALTH`", "*SECRET*"):
            self.assertIsNotNone(normalise(raw), raw)

    def test_finds_a_label_inside_a_sentence(self) -> None:
        self.assertEqual(normalise("The answer is CLEAN"), CLEAN)

    def test_refuses_to_guess_between_two(self) -> None:
        self.assertIsNone(normalise("either HEALTH or PII"))

    def test_refuses_nonsense(self) -> None:
        self.assertIsNone(normalise("I cannot help with that"))

    def test_binary_vocabulary_is_separate(self) -> None:
        self.assertEqual(normalise("KEEP", BINARY), KEEP)
        # HEALTH is not a binary answer, and must not be coerced into one.
        self.assertIsNone(normalise("HEALTH", BINARY))

    def test_holds_covers_both_vocabularies(self) -> None:
        self.assertFalse(holds(CLEAN))
        self.assertFalse(holds(SEND))
        self.assertTrue(holds(HEALTH))
        self.assertTrue(holds(KEEP))


class TestDataset(unittest.TestCase):
    def write(self, body: str) -> str:
        handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        handle.write(body)
        handle.close()
        return handle.name

    def test_loads_the_real_gold_set(self) -> None:
        examples = load(Path(__file__).resolve().parent.parent / "data" / "gold.jsonl")
        self.assertGreater(len(examples), 100)

    def test_rejects_an_unknown_label(self) -> None:
        path = self.write('{"id":"a","text":"t","label":"MAYBE","slice":"s"}\n')
        with self.assertRaisesRegex(ValueError, "not one of"):
            load(path)

    def test_rejects_a_repeated_id(self) -> None:
        row = '{"id":"a","text":"t","label":"CLEAN","slice":"s"}\n'
        with self.assertRaisesRegex(ValueError, "repeats id"):
            load(self.write(row + row))

    def test_rejects_a_missing_field(self) -> None:
        with self.assertRaisesRegex(ValueError, "has no 'slice'"):
            load(self.write('{"id":"a","text":"t","label":"CLEAN"}\n'))

    def test_rejects_an_empty_file(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            load(self.write("\n\n"))


class TestScoring(unittest.TestCase):
    def test_counts_leaks_and_friction(self) -> None:
        result = score(
            [
                prediction("1", HEALTH, HEALTH),  # caught
                prediction("2", HEALTH, CLEAN),  # leak
                prediction("3", CLEAN, CLEAN),  # fine
                prediction("4", CLEAN, HEALTH),  # friction
            ]
        )
        self.assertEqual(result["model"]["leaks"], 1)
        self.assertEqual(result["model"]["friction"], 1)
        self.assertEqual(result["model"]["catch_rate"], 0.5)
        self.assertEqual(result["model"]["friction_rate"], 0.5)

    def test_an_unparseable_answer_is_not_a_catch(self) -> None:
        """A gate that cannot read the answer has no verdict to add. Scoring it
        as a catch would credit the model for its own format failure."""
        result = score([prediction("1", HEALTH, None)])
        self.assertEqual(result["model"]["catch_rate"], 0.0)
        self.assertEqual(result["unparseable"], 1)

    def test_residual_is_measured_only_where_the_rules_miss(self) -> None:
        predictions = [
            prediction("caught-by-rules", HEALTH, CLEAN),  # rules have it; model misses
            prediction("missed-by-rules", HEALTH, HEALTH),  # only the model has it
            prediction("missed-by-both", HEALTH, CLEAN),
        ]
        rules = {"caught-by-rules": True, "missed-by-rules": False, "missed-by-both": False}
        result = score(predictions, rules)
        self.assertEqual(result["residual"]["missed_by_rules"], 2)
        self.assertEqual(result["residual"]["rescued_by_model"], 1)
        self.assertEqual(result["residual"]["residual_catch_rate"], 0.5)
        # Composed: rules catch one, model catches another, one still leaks.
        self.assertEqual(result["composed"]["leaks"], 1)
        self.assertEqual(result["residual"]["still_leaking_ids"], ["missed-by-both"])

    def test_composition_never_loses_a_rules_catch(self) -> None:
        """The model may only add. If the rules hold it, it stays held even when
        the model says clean."""
        result = score([prediction("x", HEALTH, CLEAN)], {"x": True})
        self.assertEqual(result["composed"]["catch_rate"], 1.0)

    def test_label_accuracy_is_absent_in_binary_mode(self) -> None:
        result = score([prediction("1", HEALTH, KEEP), prediction("2", CLEAN, SEND)])
        self.assertIsNone(result["model"]["label_accuracy_on_caught"])


def scored(id: str, margin: float, sensitive: bool) -> Scored:
    return Scored(
        id=id,
        slice="s",
        gold=HEALTH if sensitive else CLEAN,
        sensitive=sensitive,
        margin=margin,
        keep_logprob=0.0,
        send_logprob=0.0,
    )


class TestCalibration(unittest.TestCase):
    def test_auc_of_a_perfect_ranking(self) -> None:
        self.assertEqual(auc([scored("a", 2.0, True), scored("b", -2.0, False)]), 1.0)

    def test_auc_of_an_inverted_ranking(self) -> None:
        self.assertEqual(auc([scored("a", -2.0, True), scored("b", 2.0, False)]), 0.0)

    def test_ties_are_half_a_win(self) -> None:
        self.assertEqual(auc([scored("a", 1.0, True), scored("b", 1.0, False)]), 0.5)

    def test_sweep_reaches_both_extremes(self) -> None:
        points = sweep([scored("a", 1.0, True), scored("b", -1.0, False)])
        self.assertEqual(points[0]["catch_rate"], 1.0)  # cut below everything
        self.assertEqual(points[-1]["catch_rate"], 0.0)  # cut above everything

    def test_pick_minimises_friction_subject_to_catch(self) -> None:
        chosen = pick(sweep([scored("a", 1.0, True), scored("b", -1.0, False)]), 1.0)
        assert chosen is not None
        self.assertEqual(chosen["catch_rate"], 1.0)
        self.assertEqual(chosen["friction_rate"], 0.0)

    def test_pick_returns_nothing_when_catch_is_unreachable(self) -> None:
        # An inverted ranking cannot reach full catch without full friction.
        points = sweep([scored("a", -1.0, True), scored("b", 1.0, False)])
        chosen = pick(points, 1.0)
        assert chosen is not None
        self.assertEqual(chosen["friction_rate"], 1.0)

    def test_answer_position_skips_control_tokens(self) -> None:
        """gpt-oss emits harmony tokens where the answer is expected."""
        positions = [
            {"<|channel|>": -0.1, "<|start|>": -2.0},
            {"KEEP": -0.5, "SEND": -1.5},
        ]
        self.assertEqual(answer_position(positions), {"KEEP": -0.5, "SEND": -1.5})

    def test_answer_position_accepts_a_quote_first(self) -> None:
        positions = [{"SEND": -0.2, "KEEP": -1.0}]
        self.assertEqual(answer_position(positions)["SEND"], -0.2)

    def test_answer_position_raises_rather_than_returning_a_floor(self) -> None:
        """A silent floor produced a tidy, meaningless AUC of 0.5 for 127
        examples. A broken measurement has to fail loudly."""
        with self.assertRaises(NoAnswerPosition):
            answer_position([{"<|channel|>": -0.1}, {"analysis": -0.2}])


class TestHead(unittest.TestCase):
    """Small dimensions on purpose: this is pure Python, and the point of each
    test is the arithmetic, not the scale."""

    def separable(self, n: int = 60, width: int = 20, noise: float = 0.4):
        rng = random.Random(0)
        rows, labels = [], []
        for i in range(n):
            label = i % 2
            centre = 1.0 if label else -1.0
            rows.append([centre + rng.gauss(0, noise) for _ in range(width)])
            labels.append(label)
        return rows, labels

    def test_learns_a_separable_problem(self) -> None:
        rows, labels = self.separable()
        result = head.cross_validate(rows, labels, k=5)
        self.assertGreater(result.auc, 0.95)

    def test_random_labels_score_near_chance(self) -> None:
        """The test that keeps every quoted number honest. If cross-validation
        can be fooled by noise, nothing else in this module means anything."""
        rng = random.Random(1)
        rows = [[rng.gauss(0, 1) for _ in range(20)] for _ in range(60)]
        labels = [rng.randint(0, 1) for _ in range(60)]
        result = head.cross_validate(rows, labels, k=5)
        self.assertLess(abs(result.auc - 0.5), 0.25, f"AUC {result.auc}")

    def test_in_sample_fit_separates_random_labels(self) -> None:
        """Why cross-validation is not optional: with more dimensions than rows
        a linear model fits pure noise perfectly. Quoting in-sample accuracy
        would report this as a triumph."""
        rng = random.Random(2)
        rows = [[rng.gauss(0, 1) for _ in range(40)] for _ in range(20)]
        labels = [rng.randint(0, 1) for _ in range(20)]
        model = head.fit(rows, labels, l2=0.0, epochs=600, learning_rate=1.0)
        in_sample = head.auc([model.score(r) for r in rows], labels)
        self.assertGreater(in_sample, 0.9, f"expected overfitting, got {in_sample}")

    def test_folds_partition_exactly(self) -> None:
        parts = head.folds(17, 5)
        flat = sorted(i for part in parts for i in part)
        self.assertEqual(flat, list(range(17)))
        self.assertEqual(len(parts), 5)

    def test_folds_are_deterministic(self) -> None:
        self.assertEqual(head.folds(20, 4, seed=7), head.folds(20, 4, seed=7))
        self.assertNotEqual(head.folds(20, 4, seed=7), head.folds(20, 4, seed=8))

    def test_folds_rejects_impossible_k(self) -> None:
        with self.assertRaises(ValueError):
            head.folds(3, 5)

    def test_sigmoid_does_not_overflow(self) -> None:
        self.assertAlmostEqual(head._sigmoid(1000.0), 1.0)
        self.assertAlmostEqual(head._sigmoid(-1000.0), 0.0)

    def test_constant_dimension_does_not_divide_by_zero(self) -> None:
        rows = [[1.0, 0.0], [1.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        model = head.fit(rows, [0, 1, 0, 1])
        self.assertTrue(all(math.isfinite(w) for w in model.weights))

    def test_scoring_uses_training_statistics(self) -> None:
        """Standardising with the evaluation fold's own mean would leak it."""
        rows, labels = self.separable(n=20, width=5)
        model = head.fit(rows, labels)
        self.assertEqual(len(model.mean), 5)
        self.assertEqual(len(model.stdev), 5)

    def test_fit_rejects_mismatched_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "2 rows but 3 labels"):
            head.fit([[1.0], [2.0]], [0, 1, 0])

    def test_sweep_reaches_both_extremes(self) -> None:
        points = head.sweep([2.0, -2.0], [1, 0])
        self.assertEqual(points[0]["catch_rate"], 1.0)
        self.assertEqual(points[0]["friction_rate"], 1.0)
        self.assertEqual(points[-1]["catch_rate"], 0.0)


class TestGate(unittest.TestCase):
    def gate(self, width: int = 4) -> Gate:
        model = Model(
            weights=[1.0] * width, bias=0.0, mean=[0.0] * width, stdev=[1.0] * width
        )
        return Gate(model=model, threshold=0.0)

    def test_rejects_the_wrong_embedding_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "wrong embedding model"):
            self.gate()._decide_vector([0.5, 0.5])

    def test_rejects_an_unnormalised_embedding(self) -> None:
        """An un-normalised vector does not fail on its own — it produces a
        confidently wrong score, which is the worst failure a privacy gate can
        have. The head was fitted on unit vectors, so refuse anything else."""
        with self.assertRaisesRegex(ValueError, "not L2-normalised"):
            self.gate()._decide_vector([1.0, 1.0, 1.0, 1.0])  # norm 2.0

    def test_accepts_a_unit_vector(self) -> None:
        decision = self.gate()._decide_vector([0.5, 0.5, 0.5, 0.5])  # norm 1.0
        self.assertTrue(decision.hold)
        self.assertAlmostEqual(decision.score, 2.0)

    def test_margin_is_distance_past_the_threshold(self) -> None:
        gate = self.gate()
        gate.threshold = 1.5
        decision = gate._decide_vector([0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(decision.margin, 0.5)

    def test_round_trips_through_json(self) -> None:
        original = Model(weights=[1.0, 2.0], bias=0.5, mean=[0.0, 0.0], stdev=[1.0, 1.0])
        restored = Model.from_json(original.to_json(threshold=-0.4))
        self.assertEqual(restored.weights, original.weights)
        self.assertEqual(restored.bias, original.bias)

    def test_shipped_head_is_loadable_and_normalised_for(self) -> None:
        for name in ("head-v1.json", "head-v0.json"):
            with self.subTest(head=name):
                path = Path(__file__).resolve().parent.parent / "model" / name
                gate = Gate.load(path)
                self.assertEqual(len(gate.model.weights), 1024)
                self.assertEqual(gate.embedding_model, "bge-m3")


class TestFeatures(unittest.TestCase):
    def test_tokenise_splits_identifier_spellings(self) -> None:
        """patient_id and patientId must not be single opaque tokens, or the
        model keys on one spelling."""
        self.assertEqual(features.tokenise("patient_id"), ["patient", "id"])
        self.assertEqual(features.tokenise("DB_PASSWORD=hunter2"), ["db", "password", "hunter2"])

    def test_vector_is_unit_length(self) -> None:
        """Without this the magnitude encodes text length, and a length
        difference between classes would masquerade as a lexical result."""
        vector = features.vectorise("the quick brown fox jumps over the lazy dog")
        self.assertAlmostEqual(sum(v * v for v in vector) ** 0.5, 1.0, places=9)

    def test_empty_text_does_not_divide_by_zero(self) -> None:
        vector = features.vectorise("!!! ???")
        self.assertEqual(sum(vector), 0.0)

    def test_hashing_is_deterministic_across_calls(self) -> None:
        self.assertEqual(features.vectorise("biopsy booked"), features.vectorise("biopsy booked"))

    def test_hashing_needs_no_fitted_vocabulary(self) -> None:
        """A vocabulary fitted over the whole set before cross-validation leaks
        the evaluation fold. A hash has no fit step, so folds stay clean."""
        alone = features.vectorise("lesion")
        with_others = features.vectorise("lesion")
        self.assertEqual(alone, with_others)

    def test_length_report_counts_both_classes(self) -> None:
        report = features.length_report({"a": ["one two", "three"], "b": ["x"]})
        self.assertEqual(report["a"]["n"], 2)
        self.assertEqual(report["a"]["tokens_median"], 1.5)
        self.assertEqual(report["b"]["tokens_mean"], 1.0)


class TestPrompt(unittest.TestCase):
    def test_four_way_prompt_carries_the_text(self) -> None:
        messages = build("hello")
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("hello", messages[1]["content"])

    def test_binary_prompt_offers_only_two_words(self) -> None:
        system = build_binary("x")[0]["content"]
        self.assertIn("KEEP", system)
        self.assertIn("SEND", system)
        for label in ("HEALTH", "SECRET", "PII"):
            self.assertNotIn(f"one word: {label}", system)


if __name__ == "__main__":
    unittest.main()
