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

"""Tests for scripts/ci/check_env_var_docs.py (issue #1140).

A rule that is only ever run against a tree that satisfies it cannot be told
apart from no rule, so most cases here build a synthetic tree, break it the way
a real change would, and assert the checker rejects it. One case runs the
checker on this repository, which must pass.

Run with:
    python3 -m unittest tests/test_check_env_var_docs.py
"""

import contextlib
import importlib.util
import io
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ci" / "check_env_var_docs.py"

_spec = importlib.util.spec_from_file_location("check_env_var_docs", SCRIPT)
checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checker)

DOC = """# Environment variables

| Variable | Values | Default | Notes |
|----------|--------|---------|-------|
| `MLXCEL_DEVICE` | `gpu`, `cpu` | `gpu` | Execution device. |
| `MLXCEL_PROFILE_*` | presence | off | A wildcard documents no single name. |

Set `MLXCEL_DIFFUSION_DEBUG_CANVAS=1` to pin the canvas; MLXCEL_PROSE_ONLY is
mentioned in prose only.
"""

LIB_RS = """\
pub fn device() -> Option<String> {
    std::env::var("MLXCEL_DEVICE").ok()
}

pub fn canvas() -> bool {
    std::env::var_os(r"MLXCEL_DIFFUSION_DEBUG_CANVAS").is_some()
}

pub fn prose() -> bool {
    std::env::var("MLXCEL_PROSE_ONLY").is_ok()
}
"""

TESTS_RS = """\
#[test]
fn reads_a_fixture() {
    let _ = std::env::var("MLXCEL_TEST_FIXTURE");
}
"""

FIXTURE_ALLOWLIST = {"MLXCEL_TEST_FIXTURE": "test-only: fixture path"}


class CheckEnvVarDocsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        (self.root / "docs").mkdir()
        (self.root / "src" / "nested").mkdir(parents=True)
        self.write("docs/environment-variables.md", DOC)
        self.write("src/lib.rs", LIB_RS)
        self.write("src/nested/thing_tests.rs", TESTS_RS)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write(self, rel: str, text: str) -> None:
        (self.root / rel).write_text(text)

    def run_check(self, allowlist=None):
        return checker.check(
            self.root, FIXTURE_ALLOWLIST if allowlist is None else allowlist
        )

    def test_repository_passes(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = checker.main([])
        self.assertEqual(status, 0, out.getvalue())
        self.assertIn("env-var-docs: OK", out.getvalue())

    def test_synthetic_tree_passes(self) -> None:
        undocumented, stale, _ = self.run_check()
        self.assertEqual(undocumented, [])
        self.assertEqual(stale, [])

    def test_new_call_site_without_a_row_fails(self) -> None:
        self.write(
            "src/nested/new_thing.rs",
            'pub fn f() -> bool {\n'
            '    std::env::var("MLXCEL_NEW_THING").is_ok()\n'
            '}\n',
        )
        undocumented, _, _ = self.run_check()
        self.assertEqual(len(undocumented), 1)
        self.assertIn("MLXCEL_NEW_THING", undocumented[0])
        self.assertIn("src/nested/new_thing.rs:2", undocumented[0])

    def test_deleted_row_for_a_name_still_read_fails(self) -> None:
        doc = DOC.replace(
            "| `MLXCEL_DEVICE` | `gpu`, `cpu` | `gpu` | Execution device. |\n", ""
        )
        self.assertNotEqual(doc, DOC)
        self.write("docs/environment-variables.md", doc)
        undocumented, _, _ = self.run_check()
        self.assertEqual(len(undocumented), 1)
        self.assertIn("MLXCEL_DEVICE: in src/lib.rs:2", undocumented[0])

    def test_test_file_names_are_scanned_not_excluded(self) -> None:
        undocumented, _, _ = self.run_check(allowlist={})
        self.assertEqual(
            undocumented, ["MLXCEL_TEST_FIXTURE: in src/nested/thing_tests.rs:3"]
        )

    def test_wildcard_does_not_document_a_name(self) -> None:
        self.write(
            "src/profile.rs", 'const P: &str = "MLXCEL_PROFILE_FORWARD";\n'
        )
        undocumented, _, _ = self.run_check()
        self.assertEqual(len(undocumented), 1)
        self.assertIn("MLXCEL_PROFILE_FORWARD", undocumented[0])

    def test_longer_name_does_not_document_its_prefix(self) -> None:
        self.write("src/dev.rs", 'const D: &str = "MLXCEL_DEV";\n')
        undocumented, _, _ = self.run_check()
        self.assertEqual(len(undocumented), 1)
        self.assertIn("MLXCEL_DEV:", undocumented[0])

    def test_locations_are_capped(self) -> None:
        body = 'const N: &str = "MLXCEL_MANY";\n' * 5
        self.write("src/many.rs", body)
        undocumented, _, _ = self.run_check()
        self.assertEqual(len(undocumented), 1)
        self.assertIn("src/many.rs:3 (+2 more)", undocumented[0])

    def test_allowlist_entry_no_longer_in_src_fails(self) -> None:
        allowlist = dict(FIXTURE_ALLOWLIST, MLXCEL_GONE="test-only: removed")
        _, stale, _ = self.run_check(allowlist)
        self.assertEqual(len(stale), 1)
        self.assertIn("MLXCEL_GONE: allowlisted but no longer in src/", stale[0])

    def test_allowlisted_and_documented_fails(self) -> None:
        allowlist = dict(FIXTURE_ALLOWLIST, MLXCEL_DEVICE="internal")
        _, stale, _ = self.run_check(allowlist)
        self.assertEqual(len(stale), 1)
        self.assertIn("MLXCEL_DEVICE: allowlisted and also on", stale[0])

    def test_empty_scan_fails(self) -> None:
        for rel in ("src/lib.rs", "src/nested/thing_tests.rs"):
            (self.root / rel).unlink()
        _, stale, _ = self.run_check(allowlist={})
        self.assertEqual(len(stale), 1)
        self.assertIn("would pass over nothing", stale[0])

    def test_main_reports_both_resolutions(self) -> None:
        self.write("src/new.rs", 'const N: &str = "MLXCEL_NEW_THING";\n')
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = checker.main(["--root", str(self.root)])
        self.assertEqual(status, 1)
        text = out.getvalue()
        self.assertIn("env-var-docs: FAIL", text)
        self.assertIn("MLXCEL_NEW_THING: in src/new.rs:1", text)
        self.assertIn("document it on docs/environment-variables.md", text)
        self.assertIn("add it\n     to ALLOWLIST", text)

    def test_main_fails_without_the_doc_page(self) -> None:
        (self.root / "docs" / "environment-variables.md").unlink()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = checker.main(["--root", str(self.root)])
        self.assertEqual(status, 1)
        self.assertIn("does not exist", out.getvalue())

    def test_duplicate_allowlist_entry_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            checker.build_allowlist(
                (("a", ("MLXCEL_X",)), ("b", ("MLXCEL_X",)))
            )

    def test_every_allowlist_entry_has_a_reason(self) -> None:
        for name, reason in checker.ALLOWLIST.items():
            self.assertTrue(name.startswith("MLXCEL_"), name)
            self.assertTrue(reason.strip(), name)


if __name__ == "__main__":
    unittest.main()
