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

"""Tests for scripts/ci/gen_reports_index.py (issue #1708).

The synthetic cases build a reports directory with one released line (0.8) and
the line under development (0.9), generate the indexes, then make the change a
real report PR would make without regenerating, and assert ``--check`` rejects
it. The repository cases check that every released line's committed index is
current and links every report in its folder, and that the current line has no
committed index.

Run with:
    python3 -m unittest tests/test_gen_reports_index.py
"""

import contextlib
import importlib.util
import io
import pathlib
import re
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ci" / "gen_reports_index.py"

_spec = importlib.util.spec_from_file_location("gen_reports_index", SCRIPT)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


RELEASED = "0.8"
CURRENT = (0, 9)


def _run(reports_dir, *argv, current=CURRENT):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        return gen.main(list(argv), reports_dir=reports_dir, current=current)


def _print(reports_dir, *argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        code = gen.main(["--print", *argv], reports_dir=reports_dir, current=CURRENT)
    return code, out.getvalue()


class SyntheticReportsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name) / "TECHNICAL_REPORTS"
        self.dir = self.root / RELEASED
        self.dir.mkdir(parents=True)
        (self.root / "0.9").mkdir()
        (self.root / "0.9" / "3500-current-20261128.en.md").write_text(
            "# PR #3500: Current work\n", encoding="utf-8"
        )
        self.write("1058-paged-key-20260807.en.md", "# Technical Report: PR #1058 - Key the cache\n")
        self.write("1058-paged-key-20260807.ko.md", "# 기술 보고서: PR #1058\n")
        self.write(
            "pr-2522-compaction.en.md",
            "# PR #2522: Automatic compaction\n\n**Date**: 2026-10-21\n",
        )
        self.write("pr-2522-compaction.ko.md", "# PR #2522: 자동 압축\n")

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name, text):
        (self.dir / name).write_text(text, encoding="utf-8")

    def index(self, *argv):
        return _run(self.root, *argv)

    def rows(self):
        text = (self.dir / gen.INDEX_NAME).read_text(encoding="utf-8")
        return [line for line in text.splitlines() if line.startswith("| [")]

    def test_generated_index_passes_check(self):
        self.assertEqual(self.index(), 0)
        self.assertEqual(self.index("--check"), 0)

    def test_rows_are_newest_first_with_cleaned_titles_and_dates(self):
        self.index()
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        self.assertIn("[#2522]", rows[0])
        self.assertIn("| Automatic compaction | 2026-10-21 |", rows[0])
        self.assertIn("| Key the cache | 2026-08-07 |", rows[1])
        self.assertIn("[ko](1058-paged-key-20260807.ko.md)", rows[1])

    def test_missing_index_fails_check(self):
        self.assertEqual(self.index("--check"), 1)

    def test_new_report_without_regeneration_fails_check(self):
        self.index()
        self.write("2700-new-thing-20261030.en.md", "# PR #2700: New thing\n")
        self.write("2700-new-thing-20261030.ko.md", "# PR #2700: 새 것\n")
        self.assertEqual(self.index("--check"), 1)

    def test_retitled_report_without_regeneration_fails_check(self):
        self.index()
        self.write("pr-2522-compaction.en.md", "# PR #2522: Renamed\n")
        self.assertEqual(self.index("--check"), 1)

    def test_hand_edited_index_fails_check(self):
        self.index()
        index = self.dir / gen.INDEX_NAME
        index.write_text(index.read_text(encoding="utf-8") + "extra\n", encoding="utf-8")
        self.assertEqual(self.index("--check"), 1)

    def test_unpaired_report_is_listed_with_a_dash(self):
        self.write("2701-english-only-20261030.en.md", "# PR #2701: English only\n")
        self.index()
        row = next(r for r in self.rows() if "[#2701]" in r)
        self.assertTrue(row.endswith("| [en](2701-english-only-20261030.en.md) | - |"))

    def test_current_line_gets_no_index(self):
        self.index()
        self.assertFalse((self.root / "0.9" / gen.INDEX_NAME).exists())

    def test_committed_index_for_current_line_fails_check(self):
        self.index()
        (self.root / "0.9" / gen.INDEX_NAME).write_text("# index\n", encoding="utf-8")
        self.assertEqual(self.index("--check"), 1)

    def test_print_defaults_to_current_line_and_writes_nothing(self):
        code, text = _print(self.root)
        self.assertEqual(code, 0)
        self.assertIn("# Technical reports: 0.9", text)
        self.assertIn("[en](3500-current-20261128.en.md)", text)
        self.assertFalse((self.root / "0.9" / gen.INDEX_NAME).exists())
        self.assertFalse((self.dir / gen.INDEX_NAME).exists())

    def test_print_rejects_unknown_line(self):
        self.assertEqual(_print(self.root, "0.42")[0], 1)

    def test_version_bump_requires_the_old_line_index(self):
        self.index()
        self.assertEqual(_run(self.root, "--check", current=(0, 10)), 1)
        _run(self.root, current=(0, 10))
        self.assertTrue((self.root / "0.9" / gen.INDEX_NAME).exists())
        self.assertEqual(_run(self.root, "--check", current=(0, 10)), 0)

    def test_pipe_in_title_is_escaped(self):
        self.write("2702-pipe-20261030.en.md", "# PR #2702: a | b\n")
        self.index()
        row = next(r for r in self.rows() if "[#2702]" in r)
        self.assertIn("a \\| b", row)


class TitleCleanupTest(unittest.TestCase):
    def test_known_heading_shapes(self):
        cases = {
            "Technical Report: PR #1058 - fix(paged): key": "fix(paged): key",
            "Technical Report: PR #2682 — quantized vision": "quantized vision",
            "Technical Report: PR (issue #1982), paged block reclaim": "paged block reclaim",
            "Technical Report: Issue #1373 - p-less Sampling": "p-less Sampling",
            "Technical Report: Port Mage-VL 4B (#1365)": "Port Mage-VL 4B",
            "Model library WebUI — PR #1891": "Model library WebUI",
            "PR #2516: repair breakages left by PR #2505": "repair breakages left by PR #2505",
        }
        for heading, expected in cases.items():
            with self.subTest(heading=heading):
                self.assertEqual(gen.clean_title(heading), expected)


class RepositoryIndexTest(unittest.TestCase):
    def test_committed_indexes_are_current(self):
        self.assertEqual(
            _run(gen.REPORTS_DIR, "--check", current=gen.current_line()),
            0,
            "a TECHNICAL_REPORTS/<X.Y>/README.md index is stale; run `make reports-index` and commit it",
        )

    def test_every_released_report_is_linked(self):
        current = gen.current_line()
        released = [f for line, f in gen.line_folders(gen.REPORTS_DIR) if line < current]
        self.assertTrue(released)
        for folder in released:
            with self.subTest(line=folder.name):
                text = (folder / gen.INDEX_NAME).read_text(encoding="utf-8")
                linked = set(re.findall(r"\]\(([^)/]+\.(?:en|ko)\.md)\)", text))
                on_disk = {p.name for p in folder.iterdir() if re.search(r"\.(en|ko)\.md$", p.name)}
                self.assertEqual(linked, on_disk)


if __name__ == "__main__":
    unittest.main()
