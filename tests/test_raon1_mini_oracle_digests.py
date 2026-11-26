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

"""Tests for the Raon-1 mini oracle's checkpoint-code digest pin (issue #3457).

The oracle runs the checkpoint's own `modeling_raon1_mini.py`, so a changed
upstream file would silently change the reference. No model and no torch:
the module imports torch only inside the functions that run the model.

Run with:
    python3 -m unittest tests/test_raon1_mini_oracle_digests.py
"""

import contextlib
import hashlib
import io
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import raon1_mini_oracle_trace as oracle  # noqa: E402


class CodeDigestPinTests(unittest.TestCase):
    def test_torch_is_not_imported_at_module_scope(self):
        self.assertNotIn("torch", vars(oracle))

    def test_matching_digests_pass(self):
        oracle.check_code_digests(dict(oracle.PINNED_CODE_DIGESTS))
        self.assertEqual(oracle.code_pin_label(dict(oracle.PINNED_CODE_DIGESTS)), "6691a076")

    def test_a_changed_digest_is_refused_by_name(self):
        digests = dict(oracle.PINNED_CODE_DIGESTS)
        digests["modeling_raon1_mini.py"] = "0" * 64
        with self.assertRaises(oracle.OracleError) as raised:
            oracle.check_code_digests(digests)
        message = str(raised.exception)
        self.assertIn("modeling_raon1_mini.py", message)
        self.assertIn("0" * 64, message)
        self.assertIn(oracle.PINNED_CODE_DIGESTS["modeling_raon1_mini.py"], message)
        self.assertNotIn("configuration_raon1_mini.py", message)

    def test_allow_unpinned_downgrades_to_a_warning_and_marks_the_header(self):
        digests = dict(oracle.PINNED_CODE_DIGESTS)
        digests["configuration_raon1_mini.py"] = "f" * 64
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            oracle.check_code_digests(digests, allow_unpinned=True)
        self.assertIn("warning: configuration_raon1_mini.py", stderr.getvalue())
        self.assertEqual(oracle.code_pin_label(digests), "unpinned")

    def test_import_refuses_changed_code_before_running_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = pathlib.Path(tmp)
            marker = model_dir / "ran"
            for name in oracle.CODE_FILES:
                (model_dir / name).write_text(f"open({str(marker)!r}, 'w').close()\n")
            with self.assertRaises(oracle.OracleError) as raised:
                oracle.import_checkpoint_code(model_dir)
            self.assertIn("configuration_raon1_mini.py", str(raised.exception))
            self.assertFalse(marker.exists(), "checkpoint code ran before the digest check")

    def test_pinned_digests_are_sha256_hex(self):
        for name, digest in oracle.PINNED_CODE_DIGESTS.items():
            self.assertIn(name, oracle.CODE_FILES)
            self.assertEqual(len(digest), len(hashlib.sha256().hexdigest()))
            int(digest, 16)


if __name__ == "__main__":
    unittest.main()
