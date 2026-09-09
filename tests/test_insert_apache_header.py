import tempfile
import unittest
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.insert_apache_header import (  # noqa: E402
    PROJECT_COPYRIGHT_LINE,
    PROJECT_HEADER,
    has_existing_header,
    insert_header,
    insert_header_in_file,
    iter_target_files,
    should_process_path,
)


class InsertApacheHeaderTests(unittest.TestCase):
    def test_should_process_only_owned_code_roots_and_suffixes(self):
        self.assertTrue(should_process_path(Path("src/lib.rs")))
        self.assertTrue(should_process_path(Path("examples/demo.cpp")))
        self.assertTrue(should_process_path(Path("tests/sample.h")))
        self.assertTrue(should_process_path(Path("benches/audio_fft.rs")))
        self.assertFalse(should_process_path(Path("references/upstream.rs")))
        self.assertFalse(should_process_path(Path("src/lib/mlxcel-core/target/tmp.rs")))
        self.assertFalse(should_process_path(Path("scripts/insert_apache_header.py")))

    def test_has_existing_header_detects_project_and_external_provenance(self):
        self.assertTrue(has_existing_header(PROJECT_HEADER + "fn main() {}\n"))
        self.assertTrue(
            has_existing_header(
                "// Copyright 2025 upstream authors\n"
                "// SPDX-License-Identifier: Apache-2.0\n"
                "fn main() {}\n"
            )
        )
        self.assertFalse(has_existing_header("fn main() {}\n"))

    def test_insert_header_prepends_at_top_of_file(self):
        content = "//! crate docs\n\nfn main() {}\n"
        updated = insert_header(content)

        self.assertTrue(updated.startswith(PROJECT_COPYRIGHT_LINE))
        self.assertIn("//! crate docs", updated)
        self.assertLess(updated.index(PROJECT_COPYRIGHT_LINE), updated.index("//! crate docs"))

    def test_insert_header_in_file_skips_existing_external_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bridge.cpp"
            original = (
                "// Copyright 2025 external authors\n"
                "// Direct bridge\n\n"
                "int main() { return 0; }\n"
            )
            path.write_text(original, encoding="utf-8")

            inserted = insert_header_in_file(path)

            self.assertFalse(inserted)
            self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_iter_target_files_skips_target_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "src").mkdir()
            (root / "src" / "owned.rs").write_text("fn main() {}\n", encoding="utf-8")
            (root / "src" / "lib").mkdir()
            (root / "src" / "lib" / "mlxcel-core").mkdir()
            (root / "src" / "lib" / "mlxcel-core" / "target").mkdir()
            (root / "src" / "lib" / "mlxcel-core" / "target" / "generated.rs").write_text(
                "fn generated() {}\n", encoding="utf-8"
            )

            paths = sorted(path.relative_to(root).as_posix() for path in iter_target_files(root))

            self.assertEqual(paths, ["src/owned.rs"])


if __name__ == "__main__":
    unittest.main()
