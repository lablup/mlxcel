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

"""Keep tracked Python entry points consistent with their Git executable modes."""

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ScriptExecutableModeTests(unittest.TestCase):
    def test_python_script_modes_match_shebangs(self) -> None:
        entries = subprocess.check_output(
            ["git", "ls-files", "--stage", "-z", "--", "scripts/"], cwd=ROOT
        ).split(b"\0")
        checked = 0
        for entry in filter(None, entries):
            metadata, raw_path = entry.split(b"\t", 1)
            path = raw_path.decode("utf-8")
            if not path.endswith(".py"):
                continue
            mode, _, stage = metadata.split()
            with self.subTest(path=path):
                self.assertEqual(stage, b"0", "Resolve index conflicts first")
                with (ROOT / path).open("rb") as script:
                    has_shebang = script.readline().startswith(b"#!")
                self.assertEqual(
                    mode, b"100755" if has_shebang else b"100644",
                    "Python entry points need a shebang and executable Git mode; "
                    "shebang-less helpers are invoked with python3",
                )
            checked += 1
        self.assertGreater(checked, 0, "The script inventory must not be empty")


if __name__ == "__main__":
    unittest.main()
