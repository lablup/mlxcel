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

"""Tests for scripts/engine_bench_rounds.py (issue #2167).

Run with:
    python3 -m unittest tests/test_engine_bench_rounds.py
"""

import argparse
import importlib.util
import json
import pathlib
import stat
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "engine_bench_rounds.py"

_spec = importlib.util.spec_from_file_location("engine_bench_rounds", SCRIPT)
rounds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rounds)


def _fake_bench(directory: pathlib.Path, body: str) -> str:
    """Write an executable stand-in for mlxcel-bench-engine."""
    path = directory / "fake-bench"
    path.write_text("#!/usr/bin/env python3\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def _run_script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=60
    )


class ArmNameTests(unittest.TestCase):
    def test_parse_arm_splits_name_and_flags(self):
        name, flags = rounds.parse_arm("a=--path cli --prefill-chunk 512")
        self.assertEqual((name, flags), ("a", ["--path", "cli", "--prefill-chunk", "512"]))

    def test_parse_arm_rejects_missing_name_or_separator(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            rounds.parse_arm("no-separator")
        with self.assertRaises(argparse.ArgumentTypeError):
            rounds.parse_arm("=--path cli")

    def test_plain_names_are_accepted(self):
        rounds.check_arm_names(["dense", "paged", "null-arm", "nullish"])

    def test_duplicate_names_are_rejected(self):
        with self.assertRaises(SystemExit) as ctx:
            rounds.check_arm_names(["a", "a"])
        self.assertIn("unique", str(ctx.exception))

    def test_name_ending_in_null_collides_with_a_generated_null_arm(self):
        with self.assertRaises(SystemExit) as ctx:
            rounds.check_arm_names(["dense", "dense-null"])
        self.assertIn("dense-null", str(ctx.exception))
        self.assertIn("reserved", str(ctx.exception))

    def test_cli_rejects_a_reserved_arm_name_before_running_anything(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "out.jsonl"
            proc = _run_script(
                "--bin", "/nonexistent/bench", "--model", "m", "--out", str(out),
                "--arm", "a=--path cli", "--arm", "b-null=--path server",
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("reserved", proc.stderr)
            self.assertFalse(out.exists(), "no output file may be created on a rejected run")


class TimeoutTests(unittest.TestCase):
    def _args(self, binary, timeout):
        return argparse.Namespace(
            bin=binary, model="m", max_tokens=4, prompt_tokens=[16], extra=[], run_timeout=timeout
        )

    def test_hung_benchmark_process_is_killed_and_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = _fake_bench(pathlib.Path(tmp), "import time\ntime.sleep(30)\n")
            with self.assertRaises(SystemExit) as ctx:
                rounds.run_arm(self._args(binary, 0.5), "arm", [], 0, None)
        self.assertIn("timed out", str(ctx.exception))
        self.assertIn("--run-timeout", str(ctx.exception))

    def test_records_are_collected_within_the_timeout(self):
        record = {
            "path": "cli", "prompt_target_len": 16, "ttft_ms": 1.0, "decode_tok_s": 2.0,
        }
        body = f"print('[engine-bench] ' + {json.dumps(json.dumps(record))})\n"
        with tempfile.TemporaryDirectory() as tmp:
            binary = _fake_bench(pathlib.Path(tmp), body)
            records = rounds.run_arm(self._args(binary, 30), "arm", [], 3, None)
        self.assertEqual(len(records), 1)
        self.assertEqual((records[0]["arm"], records[0]["round"]), ("arm", 3))
        self.assertEqual(records[0]["decode_tok_s"], 2.0)

    def test_zero_timeout_disables_the_limit(self):
        record = {
            "path": "cli", "prompt_target_len": 16, "ttft_ms": 1.0, "decode_tok_s": 2.0,
        }
        body = f"print('[engine-bench] ' + {json.dumps(json.dumps(record))})\n"
        with tempfile.TemporaryDirectory() as tmp:
            binary = _fake_bench(pathlib.Path(tmp), body)
            records = rounds.run_arm(self._args(binary, 0), "arm", [], 0, None)
        self.assertEqual(len(records), 1)

    def test_failing_process_and_missing_records_are_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            failing = _fake_bench(pathlib.Path(tmp), "import sys\nsys.exit(3)\n")
            with self.assertRaises(SystemExit) as ctx:
                rounds.run_arm(self._args(failing, 30), "arm", [], 0, None)
            self.assertIn("exit 3", str(ctx.exception))
            silent = _fake_bench(pathlib.Path(tmp), "print('nothing to see')\n")
            with self.assertRaises(SystemExit) as ctx:
                rounds.run_arm(self._args(silent, 30), "arm", [], 0, None)
            self.assertIn("no [engine-bench] records", str(ctx.exception))

    def test_cli_rejects_a_negative_timeout(self):
        proc = _run_script(
            "--model", "m", "--out", "/dev/null", "--arm", "a=--path cli", "--run-timeout", "-1"
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--run-timeout", proc.stderr)


class EndToEndTests(unittest.TestCase):
    def test_rotating_schedule_runs_every_arm_and_a_null_repeat(self):
        body = (
            "import json, sys\n"
            "label = sys.argv[sys.argv.index('--label') + 1]\n"
            "rec = {'path': 'cli', 'prompt_target_len': 16, 'ttft_ms': 10.0, 'decode_tok_s': 100.0}\n"
            "print('[engine-bench] ' + json.dumps(rec))\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            binary = _fake_bench(pathlib.Path(tmp), body)
            out = pathlib.Path(tmp) / "out.jsonl"
            proc = _run_script(
                "--bin", binary, "--model", "m", "--out", str(out), "--rounds", "2",
                "--prompt-tokens", "16", "--arm", "a=--path cli", "--arm", "b=--path server",
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            lines = [json.loads(line) for line in out.read_text().splitlines()]
        header, records = lines[0], lines[1:]
        self.assertEqual(header["kind"], "header")
        # Two rounds, arms a and b plus a-null each round.
        self.assertEqual(len(records), 6)
        self.assertEqual({r["arm"] for r in records}, {"a", "b", "a-null"})
        self.assertIn("a-null", proc.stdout)


if __name__ == "__main__":
    unittest.main()
