#!/usr/bin/env python3
# Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
"""The Python verdict port against the fixture webui/scripts/activity-performance-verdict.spec.mjs uses."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import activity_performance_verdict as verdict  # noqa: E402

FIXTURE = json.loads(verdict.FIXTURE_PATH.read_text(encoding="utf-8"))
POLICY = verdict.load_policy()


def evaluate(case: dict) -> dict:
    by_mode = verdict.pairs_from_samples(case["samples"])
    if case["check"] == "pairs":
        return {}
    if case["check"] == "look":
        return {mode: verdict.look_verdict(pairs, POLICY) for mode, pairs in by_mode.items()}
    return {mode: verdict.mode_summary(mode, pairs, POLICY) for mode, pairs in by_mode.items()}


class FixtureTests(unittest.TestCase):
    def test_fixture_was_generated_under_the_policy_in_force(self) -> None:
        self.assertEqual(FIXTURE["policy"], POLICY)

    def test_every_fixture_case_reproduces_exactly(self) -> None:
        for case in FIXTURE["cases"]:
            with self.subTest(case["name"]):
                expected = case["expected"]
                if "error" in expected:
                    with self.assertRaisesRegex(AssertionError, expected["error"]):
                        evaluate(case)
                    continue
                result = evaluate(case)
                self.assertEqual(sorted(result), sorted(expected))
                for mode, want in expected.items():
                    self.assertEqual({key: result[mode][key] for key in want}, want)

    def test_no_recorded_noise_failure_is_a_slowdown(self) -> None:
        recorded = [case for case in FIXTURE["cases"] if case["name"].startswith("recorded ")]
        self.assertGreaterEqual(len(recorded), 10)
        for case in recorded:
            for want in case["expected"].values():
                self.assertNotEqual(want["status"], "investigate")

    def test_fixture_covers_every_final_status(self) -> None:
        statuses = {want.get("status") for case in FIXTURE["cases"] for want in case["expected"].values() if isinstance(want, dict)}
        self.assertTrue({"investigate", "within-target", "inconclusive", "continue"} <= statuses)


class RuleTests(unittest.TestCase):
    def test_sign_test_rank_matches_the_exact_binomial_tail(self) -> None:
        self.assertEqual([verdict.sign_test_rank(n, 0.01) for n in (6, 7, 12, 40)], [0, 1, 2, 13])
        self.assertEqual(verdict.median_interval([5, 1, 4, 2, 3, 6, 7, 8, 9, 10, 11, 12], POLICY), (6.5, [2, 11]))
        self.assertEqual(verdict.median_interval([1, 2, 3], POLICY), (2, None))

    def test_policy_validation_rejects_unsafe_or_incomplete_policies(self) -> None:
        for change in ({"tail_alpha": 0}, {"tail_alpha": 0.5}, {"min_pairs": 6}, {"pair_step": 1}, {"max_pairs": 52}, {"max_pairs": 41}, {"precision_ceiling_percent": 1}, {"budget_percent": float("nan")}, {"extra": 1}, {"min_pairs": True}):
            with self.subTest(change), self.assertRaisesRegex(AssertionError, "policy"):
                verdict.validate_policy({**POLICY, **change})
        missing = dict(POLICY)
        del missing["max_pairs"]
        with self.assertRaisesRegex(AssertionError, "policy"):
            verdict.validate_policy(missing)

    def test_summary_consistency_follows_the_policy(self) -> None:
        good = {"mode": "one-visible", "status": "within-target", "paired_runs": 12, "median_decode_degradation_percent": 0.5, "interval_percent": [-1.0, 3.0]}
        verdict.check_summary_consistency(good, POLICY)
        cases = [
            ({"status": "investigate"}, "contradicts"),
            ({"interval_percent": [-1.0, 4.5]}, "below the 40-pair cap"),
            ({"interval_percent": [2.5, 4.5], "median_decode_degradation_percent": 3.0}, "contradicts"),
            ({"paired_runs": 13}, "not a look"),
            ({"paired_runs": 42}, "not a look"),
            ({"median_decode_degradation_percent": 5.0}, "inside its interval"),
            ({"interval_percent": [3.0, 1.0]}, "ordered"),
            ({"status": "continue"}, "final verdict"),
        ]
        for change, message in cases:
            with self.subTest(change), self.assertRaisesRegex(AssertionError, message):
                verdict.check_summary_consistency({**good, **change}, POLICY)
        verdict.check_summary_consistency({**good, "status": "inconclusive", "paired_runs": 40, "interval_percent": [-1.0, 4.5]}, POLICY)
        verdict.check_summary_consistency({**good, "status": "investigate", "interval_percent": [2.5, 4.5], "median_decode_degradation_percent": 3.0}, POLICY)

    def test_failure_message_names_status_kind_and_rule(self) -> None:
        message = verdict.failure_message([
            {"mode": "one-visible", "status": "within-target", "rule": "fine"},
            {"mode": "two-visible", "status": "investigate", "rule": "slow rule"},
            {"mode": "hidden", "status": "inconclusive", "rule": "noisy rule"},
        ])
        self.assertEqual(message, "two-visible: investigate (slowdown): slow rule; hidden: inconclusive (measurement noise): noisy rule")

    def test_expected_counts_follow_paired_runs(self) -> None:
        self.assertEqual(verdict.expected_counts({"one-visible": 12, "two-visible": 14}), ({"warmup": 1, "off": 26, "one-visible": 12, "two-visible": 14}, {"one-visible": 13, "two-visible": 30}))


if __name__ == "__main__":
    unittest.main()
