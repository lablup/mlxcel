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

"""Tests for the decode-window conversion in scripts/bench_mlxlm.py (#2929).

mlx-lm reports `generation_tps = N / window`; mlxcel reports
`(N - 1) / window` because the first token comes out of prefill. The
baseline CSV must use mlxcel's convention or the "vs mlx-lm" columns are
off by N/(N-1).
"""

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bench_mlxlm  # noqa: E402


class DecodeWindowTests(unittest.TestCase):
    def test_rate_counts_only_tokens_after_the_first(self):
        # 128 tokens at mlx-lm's 128 tok/s: a 1000 ms window holding 127.
        decode_ms, rate = bench_mlxlm.decode_window(128, 128.0)
        self.assertAlmostEqual(decode_ms, 1000.0)
        self.assertAlmostEqual(rate, 127.0)

    def test_accepts_csv_strings(self):
        decode_ms, rate = bench_mlxlm.decode_window("11", "20.0")
        self.assertAlmostEqual(decode_ms, 550.0)
        self.assertAlmostEqual(rate, 10 / 0.55)

    def test_single_token_has_an_empty_window(self):
        decode_ms, rate = bench_mlxlm.decode_window(1, 50.0)
        self.assertAlmostEqual(decode_ms, 20.0)
        self.assertEqual(rate, 0.0)

    def test_missing_or_zero_inputs_give_no_value(self):
        for gen, tps in [(None, 10.0), ("", 10.0), (10, None), (10, 0.0), (0, 10.0)]:
            with self.subTest(gen=gen, tps=tps):
                self.assertEqual(bench_mlxlm.decode_window(gen, tps), (None, None))


if __name__ == "__main__":
    unittest.main()
