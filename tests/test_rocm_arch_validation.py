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

"""Tests for the HIP architecture list validation in the ROCm overlay CMakeLists.txt (issue #2639).

The test extracts the real validation loop from the overlay and runs it with
`cmake -P`, so it follows the file and needs no SDK, GPU or compiler. It is
skipped when cmake is not installed.

Run with:
    python3 -m unittest tests/test_rocm_arch_validation.py
"""

import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CMAKELISTS = (
    ROOT / "src" / "lib" / "mlxcelverse" / "overlays" / "rocm" / "mlx" / "backend" / "rocm" / "CMakeLists.txt"
)

_BLOCK = re.compile(r"^foreach\(arch \$\{CMAKE_HIP_ARCHITECTURES\}\).*?^endforeach\(\)\n", re.S | re.M)


@unittest.skipUnless(shutil.which("cmake"), "cmake not installed")
class RocmArchitectureValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        match = _BLOCK.search(CMAKELISTS.read_text(encoding="utf-8"))
        assert match, "validation loop not found in the overlay CMakeLists.txt"
        cls.tmp = tempfile.TemporaryDirectory()
        cls.script = pathlib.Path(cls.tmp.name) / "validate.cmake"
        cls.script.write_text(match.group(0), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_list(self, archs):
        return subprocess.run(
            ["cmake", f"-DCMAKE_HIP_ARCHITECTURES={archs}", "-P", str(self.script)],
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_accepts_target_ids(self):
        for archs in (
            "gfx1151",
            "gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201",
            "gfx90a:xnack+",
            "gfx908:sramecc-:xnack+",
            "gfx11-generic",
            "gfx9-4-generic",
        ):
            with self.subTest(archs=archs):
                result = self.run_list(archs)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_non_target_entries(self):
        for archs in (
            "native",
            "gfx1151;--offload-arch=evil",
            'gfx1151"; system("x")',
            "GFX1151",
            "sm_80",
            "gfx1151 gfx1100",
        ):
            with self.subTest(archs=archs):
                result = self.run_list(archs)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Not a HIP target ID", result.stderr)


if __name__ == "__main__":
    unittest.main()
