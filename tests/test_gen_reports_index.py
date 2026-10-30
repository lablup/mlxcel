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

The synthetic cases build a reports directory, generate the index, then make
the change a real report PR would make without regenerating, and assert
``--check`` rejects it. One case checks this repository's committed index,
which must be current and must link every report file on disk.

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


def _run(reports_dir, *argv):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        return gen.main(list(argv), reports_dir=reports_dir)


class SyntheticReportsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name) / "TECHNICAL_REPORTS"
        self.dir.mkdir()
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

    def rows(self):
        text = (self.dir / gen.INDEX_NAME).read_text(encoding="utf-8")
        return [line for line in text.splitlines() if line.startswith("| [")]

    def test_generated_index_passes_check(self):
        self.assertEqual(_run(self.dir), 0)
        self.assertEqual(_run(self.dir, "--check"), 0)

    def test_rows_are_newest_first_with_cleaned_titles_and_dates(self):
        _run(self.dir)
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        self.assertIn("[#2522]", rows[0])
        self.assertIn("| Automatic compaction | 2026-10-21 |", rows[0])
        self.assertIn("| Key the cache | 2026-08-07 |", rows[1])
        self.assertIn("[ko](1058-paged-key-20260807.ko.md)", rows[1])

    def test_missing_index_fails_check(self):
        self.assertEqual(_run(self.dir, "--check"), 1)

    def test_new_report_without_regeneration_fails_check(self):
        _run(self.dir)
        self.write("2700-new-thing-20261030.en.md", "# PR #2700: New thing\n")
        self.write("2700-new-thing-20261030.ko.md", "# PR #2700: 새 것\n")
        self.assertEqual(_run(self.dir, "--check"), 1)

    def test_retitled_report_without_regeneration_fails_check(self):
        _run(self.dir)
        self.write("pr-2522-compaction.en.md", "# PR #2522: Renamed\n")
        self.assertEqual(_run(self.dir, "--check"), 1)

    def test_hand_edited_index_fails_check(self):
        _run(self.dir)
        index = self.dir / gen.INDEX_NAME
        index.write_text(index.read_text(encoding="utf-8") + "extra\n", encoding="utf-8")
        self.assertEqual(_run(self.dir, "--check"), 1)

    def test_unpaired_report_is_listed_with_a_dash(self):
        self.write("2701-english-only-20261030.en.md", "# PR #2701: English only\n")
        _run(self.dir)
        row = next(r for r in self.rows() if "[#2701]" in r)
        self.assertTrue(row.endswith("| [en](2701-english-only-20261030.en.md) | - |"))

    def test_pipe_in_title_is_escaped(self):
        self.write("2702-pipe-20261030.en.md", "# PR #2702: a | b\n")
        _run(self.dir)
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
    def test_committed_index_is_current(self):
        self.assertEqual(
            _run(gen.REPORTS_DIR, "--check"),
            0,
            "TECHNICAL_REPORTS/README.md is stale; run `make reports-index` and commit it",
        )

    def test_every_report_file_is_linked(self):
        text = (gen.REPORTS_DIR / gen.INDEX_NAME).read_text(encoding="utf-8")
        linked = set(re.findall(r"\]\(([^)/]+\.(?:en|ko)\.md)\)", text))
        on_disk = {
            p.name for p in gen.REPORTS_DIR.iterdir() if re.search(r"\.(en|ko)\.md$", p.name)
        }
        self.assertEqual(linked, on_disk)


if __name__ == "__main__":
    unittest.main()
