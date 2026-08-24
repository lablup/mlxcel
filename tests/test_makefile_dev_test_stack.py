# Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

import pathlib
import re
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"


class MakefileDevTestStackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = MAKEFILE.read_text()

    def test_dev_test_targets_export_stack_override(self) -> None:
        expected_commands = {
            "test": "RUST_MIN_STACK=$(DEV_TEST_RUST_MIN_STACK) $(CARGO) test -- --test-threads=1",
            "test-verbose": "RUST_MIN_STACK=$(DEV_TEST_RUST_MIN_STACK) $(CARGO) test -- --nocapture --test-threads=1",
            "test-lib": "RUST_MIN_STACK=$(DEV_TEST_RUST_MIN_STACK) $(CARGO) test --lib -- --test-threads=1",
        }

        for target, command in expected_commands.items():
            with self.subTest(target=target):
                block = self._target_block(target)
                self.assertIn(command, block)

    def test_dev_test_stack_default_is_16_mib(self) -> None:
        self.assertRegex(self.text, r"(?m)^DEV_TEST_RUST_MIN_STACK \?= 16777216$")

    def test_dev_test_dry_runs_expand_stack_override(self) -> None:
        expected_commands = {
            "test": "RUST_MIN_STACK=16777216 cargo test -- --test-threads=1",
            "test-verbose": "RUST_MIN_STACK=16777216 cargo test -- --nocapture --test-threads=1",
            "test-lib": "RUST_MIN_STACK=16777216 cargo test --lib -- --test-threads=1",
        }

        for target, command in expected_commands.items():
            with self.subTest(target=target):
                self.assertIn(command, self._make_dry_run(target))

    def test_verify_test_command_stays_ci_faithful(self) -> None:
        block = self._target_block("verify-test")
        self.assertIn(
            "$(CARGO) test --workspace --profile test-fast --features metal,accelerate --no-fail-fast -- --test-threads=1",
            block,
        )
        self.assertNotIn("RUST_MIN_STACK", block)
        self.assertIn(
            "cargo test --workspace --profile test-fast --features metal,accelerate --no-fail-fast -- --test-threads=1",
            self._make_dry_run("verify-test"),
        )

    def _target_block(self, target: str) -> str:
        pattern = re.compile(
            rf"(?ms)^\.PHONY: {re.escape(target)}\n{re.escape(target)}:.*?(?=^\.(?:PHONY|SUFFIXES):|\Z)"
        )
        match = pattern.search(self.text)
        self.assertIsNotNone(match, f"missing target block for {target}")
        return match.group(0)

    def _make_dry_run(self, target: str) -> str:
        result = subprocess.run(
            ["make", "-n", target],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout


if __name__ == "__main__":
    unittest.main()
