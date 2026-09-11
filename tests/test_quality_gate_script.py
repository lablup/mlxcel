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

import pathlib
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_quality_gate.sh"


class QualityGateScriptTests(unittest.TestCase):
    def test_dry_run_lists_root_and_core_baseline(self) -> None:
        result = subprocess.run(
            [str(SCRIPT), "--dry-run"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        output = result.stdout
        self.assertIn("cargo fmt --all", output)
        self.assertIn("cargo test --lib --quiet", output)
        self.assertIn("cargo clippy --all-targets -- -D warnings", output)
        self.assertIn(
            "cargo test --manifest-path src/lib/mlxcel-core/Cargo.toml -- --test-threads=1",
            output,
        )

    def test_dry_run_with_serial_helpers_lists_ignored_tests(self) -> None:
        result = subprocess.run(
            [str(SCRIPT), "--dry-run", "--include-serial-helpers"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        output = result.stdout
        self.assertIn(
            "cargo test gemma3n_helpers_tests -- --ignored --test-threads=1",
            output,
        )
        self.assertIn(
            "cargo test llama4_helpers_tests -- --ignored --test-threads=1",
            output,
        )
        self.assertIn(
            "cargo test models::gated_delta::tests -- --ignored --test-threads=1",
            output,
        )

    def test_dry_run_with_smoke_includes_cpu_only_commands(self) -> None:
        result = subprocess.run(
            [
                str(SCRIPT),
                "--dry-run",
                "--include-smoke",
                "--text-model",
                "models/custom-text",
                "--vlm-model",
                "models/custom-vlm",
                "--vlm-image",
                "tests/fixtures/test_image.png",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        output = result.stdout
        self.assertIn("MLXCEL_BUILD_METAL=OFF", output)
        self.assertIn("models/custom-text", output)
        self.assertIn("models/custom-vlm", output)
        self.assertIn("tests/fixtures/test_image.png", output)

    def test_dry_run_full_mode_includes_serial_helpers_and_smoke(self) -> None:
        result = subprocess.run(
            [str(SCRIPT), "--dry-run", "--full"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        output = result.stdout
        self.assertIn(
            "cargo test gemma3n_helpers_tests -- --ignored --test-threads=1",
            output,
        )
        self.assertIn(
            "cargo test models::gated_delta::tests -- --ignored --test-threads=1",
            output,
        )
        self.assertIn("MLXCEL_BUILD_METAL=OFF", output)

    def test_unknown_argument_fails(self) -> None:
        result = subprocess.run(
            [str(SCRIPT), "--unknown"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unknown argument", result.stderr)


if __name__ == "__main__":
    unittest.main()
