#!/usr/bin/env python3
"""Model-free tests for the near-tie judge (issue #3343)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import near_tie_check as nt

# A fake vocabulary: id -> text piece. Decoding concatenates the pieces.
VOCAB = {1: "The", 2: " cat", 3: " sat", 4: "\n", 5: "there", 6: " dog"}


def decode(ids: list[int]) -> str:
    return "".join(VOCAB[i] for i in ids)


REF = [1, 2, 3, 4]


def oracle(gaps=None, ids=None) -> dict:
    out: dict = {"ref_token_ids": REF}
    if gaps is not None:
        out["ref_top2_gaps"] = gaps
    if ids is not None:
        out["ref_top2_ids"] = ids
    return out


class IdsPathTests(unittest.TestCase):
    def test_exact_and_shorter_stream_are_exact(self) -> None:
        self.assertIsNone(nt.first_divergence_ids(REF, REF))
        self.assertIsNone(nt.first_divergence_ids([1, 2], REF))

    def test_first_divergence_index(self) -> None:
        self.assertEqual(nt.first_divergence_ids([1, 2, 6, 4], REF), 2)


class TextPathTests(unittest.TestCase):
    def test_matching_text_is_exact(self) -> None:
        self.assertEqual(nt.first_divergence_text("The cat sat\n", REF, decode), (None, 3))

    def test_text_that_stops_inside_a_token_is_exact(self) -> None:
        self.assertEqual(nt.first_divergence_text("The cat", REF, decode), (None, 2))

    def test_divergence_on_a_word_boundary(self) -> None:
        index, _ = nt.first_divergence_text("The cat dog", REF, decode)
        self.assertEqual(index, 2)

    def test_divergence_inside_a_word_is_ambiguous(self) -> None:
        # Oracle "The" + " cat"; the stream "Thecats" could be one longer token.
        with self.assertRaises(nt.AmbiguousDivergence):
            nt.first_divergence_text("Thecats", [1, 2], decode)

    def test_empty_text_compares_nothing(self) -> None:
        self.assertEqual(nt.first_divergence_text("", REF, decode), (None, 0))

    def test_strip_mlxcel_cli(self) -> None:
        out = "banner\nPROMPT cat sat\n[Generated 3 tokens]\n"
        self.assertEqual(nt.strip_mlxcel_cli(out, "PROMPT"), " cat sat")
        with self.assertRaises(ValueError):
            nt.strip_mlxcel_cli("banner", "PROMPT")


class JudgeTests(unittest.TestCase):
    def test_exact(self) -> None:
        self.assertEqual(nt.judge(oracle(), None, 0.01)[0], nt.EXIT_EXACT)

    def test_near_tie_and_mismatch(self) -> None:
        o = oracle([0.5, 0.5, 0.004, 0.5], [9, 9, 6, 9])
        self.assertEqual(nt.judge(o, 2, 0.01)[0], nt.EXIT_NEAR_TIE)
        self.assertEqual(nt.judge(o, 1, 0.01)[0], nt.EXIT_MISMATCH)

    def test_missing_top2_is_a_usage_error(self) -> None:
        self.assertEqual(nt.judge(oracle(), 1, 0.01)[0], nt.EXIT_USAGE)

    def test_malformed_oracle_is_a_usage_error_not_an_exception(self) -> None:
        for bad in (
            oracle([0.1, 0.2]),
            oracle([0.1] * 4, [1, 2]),
            oracle("nope"),
            oracle([0.1, 0.1, "x", 0.1]),
        ):
            status, message = nt.judge(bad, 2, 0.01)
            self.assertEqual(status, nt.EXIT_USAGE, message)
        self.assertEqual(nt.judge(oracle([0.1] * 4), 9, 0.01)[0], nt.EXIT_USAGE)


if __name__ == "__main__":
    unittest.main()
