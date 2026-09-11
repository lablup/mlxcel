import tempfile
import unittest
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.insert_apache_header import (  # noqa: E402
    PROJECT_COPYRIGHT_LINE,
    PROJECT_HEADER,
    TARGET_FILES,
    files_missing_header,
    has_existing_header,
    insert_header,
    insert_header_in_file,
    iter_target_files,
    main,
    should_process_path,
)


class InsertApacheHeaderTests(unittest.TestCase):
    def test_should_process_only_owned_code_roots_and_suffixes(self):
        self.assertTrue(should_process_path(Path("src/lib.rs")))
        self.assertTrue(should_process_path(Path("examples/demo.cpp")))
        self.assertTrue(should_process_path(Path("tests/sample.h")))
        self.assertTrue(should_process_path(Path("benches/audio_fft.rs")))
        # Top-level shipped files reached via TARGET_FILES, not a root walk:
        # `Path("build.rs").parts[0]` is the file itself and matches no root.
        self.assertTrue(should_process_path(Path("build.rs")))
        self.assertFalse(should_process_path(Path("references/upstream.rs")))
        # `spike/` is outside the workspace members and deliberately uncovered.
        self.assertFalse(should_process_path(Path("spike/rust-emitter/src/main.rs")))
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

    def test_target_files_are_repo_relative_posix_paths(self):
        # `should_process_path` compares against `path.as_posix()`, so a
        # backslash or a leading "./" here would silently never match.
        for name in TARGET_FILES:
            self.assertEqual(name, Path(name).as_posix())

    def test_iter_target_files_yields_top_level_target_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "build.rs").write_text("fn main() {}\n", encoding="utf-8")
            (root / "src").mkdir()
            (root / "src" / "owned.rs").write_text("fn main() {}\n", encoding="utf-8")
            # A top-level Rust file that is NOT in TARGET_FILES stays out.
            (root / "stray.rs").write_text("fn main() {}\n", encoding="utf-8")

            paths = sorted(path.relative_to(root).as_posix() for path in iter_target_files(root))

            self.assertEqual(paths, ["build.rs", "src/owned.rs"])

    def test_files_missing_header_agrees_with_the_writer(self):
        # The check mode must report exactly the files the default insert mode
        # would write to. If these two ever diverge, CI would demand an edit
        # that running the script does not produce.
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "src").mkdir()
            bare = root / "src" / "bare.rs"
            bare.write_text("fn main() {}\n", encoding="utf-8")
            covered = root / "src" / "covered.rs"
            covered.write_text(PROJECT_HEADER + "fn main() {}\n", encoding="utf-8")
            external = root / "src" / "external.rs"
            external.write_text(
                "// Copyright 2025 upstream authors\nfn main() {}\n", encoding="utf-8"
            )

            missing = files_missing_header(root)
            self.assertEqual([p.relative_to(root).as_posix() for p in missing], ["src/bare.rs"])

            # The writer touches exactly that one file, and no other.
            written = {
                path.relative_to(root).as_posix()
                for path in iter_target_files(root)
                if insert_header_in_file(path)
            }
            self.assertEqual(written, {"src/bare.rs"})

    def test_files_missing_header_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "src").mkdir()
            bare = root / "src" / "bare.rs"
            original = "fn main() {}\n"
            bare.write_text(original, encoding="utf-8")

            files_missing_header(root)

            self.assertEqual(bare.read_text(encoding="utf-8"), original)

    def test_check_mode_passes_on_the_real_repository(self):
        # The gate `.github/workflows/ci.yml` runs. Kept as a unit test too so
        # a missing header fails locally before it reaches CI.
        self.assertEqual(main(["--check"]), 0)

    def test_check_mode_reports_and_fails_on_a_missing_header(self):
        # Negative coverage: without this, a --check that always returned 0
        # would pass every other test here.
        import scripts.insert_apache_header as mod

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "src").mkdir()
            (root / "src" / "bare.rs").write_text("fn main() {}\n", encoding="utf-8")

            original_root = mod.REPO_ROOT
            mod.REPO_ROOT = root
            try:
                self.assertEqual(mod.main(["--check"]), 1)
            finally:
                mod.REPO_ROOT = original_root


if __name__ == "__main__":
    unittest.main()
