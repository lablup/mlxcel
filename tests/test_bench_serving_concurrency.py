# Copyright 2025-2026 Lablup Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the batch-counter deltas of the serving benchmark (issue #2156).

Covers scripts/bench_serving_metrics.py and the --metrics wiring of
scripts/bench_serving_concurrency.py on canned /metrics text. No server or GPU
is needed.

Run with:
    python3 -m pytest tests/test_bench_serving_concurrency.py
"""

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bench_serving_concurrency as bsc  # noqa: E402
import bench_serving_metrics as bsm  # noqa: E402


def metrics_text(decode_steps, decode_tokens, mixed, chunks, pred_s, pred_tok, fused=0):
    """A /metrics body shaped like src/server/routes/metrics.rs renders it."""
    return f"""# HELP mlxcel_batch_decode_tokens_total Cumulative decode tokens generated
# TYPE mlxcel_batch_decode_tokens_total counter
mlxcel_batch_decode_tokens_total {decode_tokens}
# HELP mlxcel_batch_decode_steps_total Total decode steps executed
# TYPE mlxcel_batch_decode_steps_total counter
mlxcel_batch_decode_steps_total {decode_steps}
mlxcel_batch_decode_lookahead_steps_total 999
mlxcel_batch_prefill_chunks_total {chunks}
mlxcel_batch_mixed_steps_total {mixed}
mlxcel_paged_decode_launches_total{{path="fused_v2"}} {fused}
# HELP llamacpp:tokens_predicted_total Number of generation tokens processed
# TYPE llamacpp:tokens_predicted_total counter
llamacpp:tokens_predicted_total {pred_tok}
# HELP llamacpp:tokens_predicted_seconds_total Total time spent generating tokens
# TYPE llamacpp:tokens_predicted_seconds_total counter
llamacpp:tokens_predicted_seconds_total {pred_s}
llamacpp:n_decode_total not-a-number
"""


class ParseMetricsTest(unittest.TestCase):
    def test_picks_only_named_samples_and_skips_comments(self):
        got = bsm.parse_metrics(metrics_text(10, 40, 0, 4, 1.5, 44), bsm.BATCH_METRICS)
        self.assertEqual(
            got,
            {
                "mlxcel_batch_decode_steps_total": 10.0,
                "mlxcel_batch_decode_tokens_total": 40.0,
                "mlxcel_batch_mixed_steps_total": 0.0,
                "mlxcel_batch_prefill_chunks_total": 4.0,
                "llamacpp:tokens_predicted_seconds_total": 1.5,
                "llamacpp:tokens_predicted_total": 44.0,
            },
        )

    def test_unparsable_value_is_skipped(self):
        got = bsm.parse_metrics(
            metrics_text(1, 1, 0, 0, 0, 0), ("llamacpp:n_decode_total",)
        )
        self.assertEqual(got, {})

    def test_labelled_sample_matches_by_full_name(self):
        name = 'mlxcel_paged_decode_launches_total{path="fused_v2"}'
        got = bsm.parse_metrics(metrics_text(1, 1, 0, 0, 0, 0, fused=32), (name,))
        self.assertEqual(got, {name: 32.0})


class BatchDeltaTest(unittest.TestCase):
    def delta(self, before, after):
        return bsm.batch_delta(
            bsm.parse_metrics(before, bsm.BATCH_METRICS),
            bsm.parse_metrics(after, bsm.BATCH_METRICS),
        )

    def test_four_requests_side_by_side(self):
        # 4 requests x 128 tokens: 127 decode steps of 4 rows after the
        # prefill-sampled first tokens, 4 prefill chunks, each request
        # generating for 3.81 s. A 30 ms step seen by every request.
        before = metrics_text(100, 100, 0, 1, 2.0, 100)
        after = metrics_text(227, 608, 0, 5, 2.0 + 4 * 3.81, 612)
        d = self.delta(before, after)
        self.assertEqual(d.decode_steps, 127)
        self.assertEqual(d.decode_tokens, 508)
        self.assertEqual(d.prefill_chunks, 4)
        self.assertEqual(d.mixed_steps, 0)
        self.assertAlmostEqual(d.predicted_s, 15.24)
        self.assertEqual(d.predicted_tokens, 512)
        self.assertAlmostEqual(d.occupancy, 4.0)
        # The raw ratio counts every step once per request.
        self.assertAlmostEqual(d.request_seconds_per_step_ms, 15240 / 127)
        self.assertAlmostEqual(d.step_ms, 15240 / 508)
        self.assertAlmostEqual(d.step_ms * d.occupancy, d.request_seconds_per_step_ms)

    def test_single_request_step_ms_is_the_raw_ratio(self):
        d = self.delta(
            metrics_text(0, 0, 0, 0, 0, 0), metrics_text(127, 127, 0, 1, 3.429, 128)
        )
        self.assertAlmostEqual(d.occupancy, 1.0)
        self.assertAlmostEqual(d.step_ms, 27.0)
        self.assertAlmostEqual(d.request_seconds_per_step_ms, 27.0)

    def test_guard_steps_lower_the_occupancy(self):
        # Zero-row guard steps add a step and no rows (decode_tick.rs).
        d = self.delta(
            metrics_text(0, 0, 0, 0, 0, 0), metrics_text(10, 30, 2, 0, 0.9, 30)
        )
        self.assertAlmostEqual(d.occupancy, 3.0)
        self.assertEqual(d.mixed_steps, 2)
        self.assertAlmostEqual(d.step_ms, 30.0)

    def test_no_decode_steps_yields_none_not_a_division_error(self):
        d = self.delta(metrics_text(5, 5, 0, 0, 1, 5), metrics_text(5, 5, 0, 3, 1, 5))
        self.assertIsNone(d.occupancy)
        self.assertIsNone(d.step_ms)
        self.assertIsNone(d.request_seconds_per_step_ms)
        self.assertIn("decode_step_ms=n/a", bsm.format_batch_delta(d)[1])

    def test_missing_endpoint_is_reported(self):
        self.assertIsNone(bsm.batch_delta({}, {}))
        self.assertEqual(
            bsm.format_batch_delta(None),
            ["    batch steps: /metrics unavailable, step cost unknown"],
        )

    def test_counter_absent_before_counts_from_zero(self):
        after = bsm.parse_metrics(metrics_text(8, 16, 0, 2, 0.4, 18), bsm.BATCH_METRICS)
        d = bsm.batch_delta({}, after)
        self.assertEqual(d.decode_steps, 8)
        self.assertAlmostEqual(d.step_ms, 25.0)

    def test_printed_lines(self):
        d = self.delta(
            metrics_text(0, 0, 0, 0, 0, 0), metrics_text(127, 508, 0, 4, 15.24, 512)
        )
        lines = bsm.format_batch_delta(d)
        self.assertEqual(
            lines[0],
            "    batch counters: decode_steps=+127  decode_tokens=+508  "
            "mixed_steps=+0  prefill_chunks=+4  tokens_predicted_seconds=+15.240",
        )
        self.assertEqual(
            lines[1],
            "    batch steps: occupancy=4.00  decode_step_ms=30.00  "
            "(predicted_seconds/steps=120.00 ms)",
        )


class HarnessWiringTest(unittest.TestCase):
    def test_scrape_set_carries_path_and_batch_counters(self):
        names = bsc._PATH_METRICS + bsc.BATCH_METRICS
        self.assertIn("mlxcel_batch_decode_steps_total", names)
        self.assertIn('mlxcel_paged_decode_launches_total{path="cascade"}', names)

    def test_metric_names_match_the_server(self):
        # A rename on the server side would silently zero every delta.
        source = (ROOT / "src" / "server" / "routes" / "metrics.rs").read_text()
        for name in bsm.BATCH_METRICS:
            bare = name.split(":", 1)[-1]
            self.assertIn(bare, source, name)
        self.assertIn('"# HELP llamacpp:{name} {help}"', source)


if __name__ == "__main__":
    unittest.main()
