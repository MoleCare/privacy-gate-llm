# SPDX-License-Identifier: Apache-2.0
"""Unit tests. No network, no model.

The metric tests matter most: every claim in docs/JOURNAL.md is one of these
ratios, so a silent change here would rewrite the record.

Run with:  PYTHONPATH=src python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from privacy_gate.dataset import load
from privacy_gate.evaluate import Prediction, score
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
