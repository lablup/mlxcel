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

"""Tests for `scripts/oxide_fused_norm_rounds.py summarize` (issue #2834).

A cell with no rows for an arm used to crash the peak-memory table
(`min()` of an empty list) before `--out` was written. It must read as
unresolved and the summary file must still be written.

Run with:
    python3 -m unittest tests/test_oxide_fused_norm_rounds.py
"""

import contextlib
import csv
import importlib.util
import io
import pathlib
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "oxide_fused_norm_rounds", REPO / "scripts" / "oxide_fused_norm_rounds.py"
)
rounds = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rounds)

ARMS = ("graph", "fused", "oxide", "fused-null")


def make_rows(drop=None):
    """Two cells, three rounds, four arms; `drop` is a (cell index, arm) to omit."""
    rows = []
    cells = [("cli", "1", "128"), ("server", "4", "2048")]
    for ci, (path, batch, plen) in enumerate(cells):
        for rnd in range(3):
            for arm in ARMS:
                if drop == (ci, arm):
                    continue
                rows.append({
                    "arm": arm, "round": str(rnd), "path": path, "batch": batch,
                    "prompt_target_len": plen, "ttft_ms": "10.0", "decode_tok_s": "100.0",
                    "peak_memory_bytes": str(2**30), "tokens_sha256_16": "abcd",
                })
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class Args:
    def __init__(self, csv_path, out):
        self.csv = str(csv_path)
        self.out = str(out)


def summarize(rows):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        write_csv(tmp / "in.csv", rows)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rounds.cmd_summarize(Args(tmp / "in.csv", tmp / "out.csv"))
        with open(tmp / "out.csv", newline="") as fh:
            return buf.getvalue(), list(csv.DictReader(fh))


class SummarizeTest(unittest.TestCase):
    def test_complete_input_is_resolved(self):
        text, out = summarize(make_rows())
        self.assertEqual(len(out), 4)
        self.assertTrue(all(r["verdict"] == "no regression" for r in out), text)

    def test_arm_missing_from_one_cell_is_unresolved_and_out_is_written(self):
        for arm in ("oxide", "fused", "fused-null", "graph"):
            with self.subTest(arm=arm):
                text, out = summarize(make_rows(drop=(0, arm)))
                by_cell = {}
                for r in out:
                    by_cell.setdefault(r["path"], []).append(r["verdict"])
                if arm != "graph":
                    self.assertTrue(all(v.startswith("unresolved") for v in by_cell["cli"]), text)
                self.assertTrue(all(v == "no regression" for v in by_cell["server"]), text)
                self.assertIn("unresolved (no rows)", text)


if __name__ == "__main__":
    unittest.main()
