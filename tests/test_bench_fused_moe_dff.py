# Copyright 2026 Lablup Inc.
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

"""The benchmark process must not report success after exhausting retries."""
import contextlib
import csv
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("bench_dff", Path(__file__).resolve().parents[1] / "scripts/bench_fused_moe_dff.py")
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)

class MeasurementExitTests(unittest.TestCase):
    def test_failed_or_jit_only_measurements_fail_without_discarding_records(self):
        for outcome in ("ok", "failed", "jit"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as tmp:
                calls = 0
                def run(*args):
                    nonlocal calls
                    calls += 1
                    text = "Decode: 100 ms (10 tok/s)\nGenerated tokens: 2"
                    if calls <= 3 or outcome == "ok":
                        return 0, text, 0.1
                    return (1, "error", 0.1) if outcome == "failed" else (0, text + bench.JIT_NOTICE, 0.1)
                argv = ["bench", "--harness", "primary", "--bin-dir", tmp, "--model", tmp, "--name", "fixture", "--rounds", "1", "--out", tmp]
                with mock.patch.object(sys, "argv", argv), mock.patch.object(bench, "run_once", side_effect=run), mock.patch.object(bench, "wait_quiet", return_value=0), contextlib.redirect_stdout(io.StringIO()):
                    result = bench.main()
                self.assertEqual(result, 0 if outcome == "ok" else 3)
                with open(Path(tmp) / "runs.csv") as f:
                    rows = list(csv.DictReader(f))
                measured = [r for r in rows if r["round"] == "1"]
                self.assertEqual(len(measured), 3 if outcome == "ok" else 9)
                self.assertEqual({r["status"] for r in measured}, {"ok" if outcome == "ok" else "failed" if outcome == "failed" else "discarded_jit"})

if __name__ == "__main__":
    unittest.main()
