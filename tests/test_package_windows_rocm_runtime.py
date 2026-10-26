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

"""Tests for scripts/package_windows_rocm_runtime.py (issue #2606).

Each case builds a synthetic ROCm SDK in a temporary directory, so the suite
needs no SDK, GPU or Windows.

Run with:
    python3 -m unittest tests/test_package_windows_rocm_runtime.py
"""

import contextlib
import importlib.util
import io
import pathlib
import tempfile
import unittest
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "package_windows_rocm_runtime.py"

_spec = importlib.util.spec_from_file_location("package_windows_rocm_runtime", SCRIPT)
pkg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pkg)


def touch(path: pathlib.Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_sdk(root: pathlib.Path, arches=("gfx1151", "gfx1201")) -> pathlib.Path:
    touch(root / ".info" / "version", "10.1.0\n")
    for dll in pkg.RUNTIME_DLLS:
        touch(root / "bin" / dll)
    touch(root / "bin" / "MIOpen.dll")  # not part of the closure
    for arch in arches:
        for family in pkg.KPACK_FAMILIES:
            touch(root / ".kpack" / f"{family}_{arch}.kpack")
        touch(root / ".kpack" / f"solver_lib_{arch}.kpack")
        touch(root / "bin" / "rocblas" / "library" / arch / f"Kernels.so-000-{arch}.hsaco")
        touch(
            root / "bin" / "hipblaslt" / "library" / f"TensileLibrary_BB_{arch}.co"
        )
    touch(root / "bin" / "rocblas" / "library" / "TensileManifest.txt")
    touch(root / "include" / "hip" / "hip_runtime.h")
    touch(root / "include" / "hip" / "amd_detail" / "amd_hip_runtime.h")
    touch(root / "include" / "rocblas" / "rocblas.h")  # not shipped
    for component in pkg.LICENSE_COMPONENTS:
        touch(root / "share" / "doc" / component / "LICENSE.md")
        touch(root / "share" / "doc" / component / "README.md")
    return root


def dests(plan) -> set:
    return {dest for _, dest in plan.files}


class BuildPlanTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._tmp.name)
        self.sdk = make_sdk(self.tmp / "rocm")

    def tearDown(self):
        self._tmp.cleanup()

    def test_contents_keep_the_sdk_layout(self):
        got = dests(pkg.build_plan(self.sdk, ["gfx1151"], []))
        for dll in pkg.RUNTIME_DLLS:
            self.assertIn(f"bin/{dll}", got)
        self.assertNotIn("bin/MIOpen.dll", got)
        self.assertIn(".kpack/blas_lib_gfx1151.kpack", got)
        self.assertIn(".kpack/fft_lib_gfx1151.kpack", got)
        self.assertNotIn(".kpack/solver_lib_gfx1151.kpack", got)
        self.assertIn("include/hip/hip_runtime.h", got)
        self.assertIn("include/hip/amd_detail/amd_hip_runtime.h", got)
        self.assertNotIn("include/rocblas/rocblas.h", got)
        self.assertIn("licenses/rocblas/LICENSE.md", got)
        self.assertNotIn("licenses/rocblas/README.md", got)

    def test_kernel_libraries_are_filtered_by_arch(self):
        got = dests(pkg.build_plan(self.sdk, ["gfx1151"], []))
        self.assertIn("bin/rocblas/library/gfx1151/Kernels.so-000-gfx1151.hsaco", got)
        self.assertIn("bin/hipblaslt/library/TensileLibrary_BB_gfx1151.co", got)
        self.assertIn("bin/rocblas/library/TensileManifest.txt", got)
        self.assertFalse(any("gfx1201" in d for d in got), got)

    def test_several_arches(self):
        got = dests(pkg.build_plan(self.sdk, ["gfx1151", "GFX1201"], []))
        self.assertIn(".kpack/blas_lib_gfx1201.kpack", got)
        self.assertIn("bin/rocblas/library/gfx1201/Kernels.so-000-gfx1201.hsaco", got)

    def test_missing_dll_fails_and_names_it(self):
        (self.sdk / "bin" / "rocblas.dll").unlink()
        with self.assertRaises(pkg.PackagingError) as ctx:
            pkg.build_plan(self.sdk, ["gfx1151"], [])
        self.assertIn("rocblas.dll", str(ctx.exception))

    def test_missing_kpack_fails(self):
        (self.sdk / ".kpack" / "blas_lib_gfx1151.kpack").unlink()
        with self.assertRaises(pkg.PackagingError) as ctx:
            pkg.build_plan(self.sdk, ["gfx1151"], [])
        self.assertIn("blas_lib_gfx1151.kpack", str(ctx.exception))

    def test_missing_license_fails(self):
        (self.sdk / "share" / "doc" / "amd_comgr" / "LICENSE.md").unlink()
        with self.assertRaises(pkg.PackagingError) as ctx:
            pkg.build_plan(self.sdk, ["gfx1151"], [])
        self.assertIn("amd_comgr", str(ctx.exception))

    def test_missing_jit_header_fails(self):
        (self.sdk / "include" / "hip" / "hip_runtime.h").unlink()
        with self.assertRaises(pkg.PackagingError) as ctx:
            pkg.build_plan(self.sdk, ["gfx1151"], [])
        self.assertIn("hip_runtime.h", str(ctx.exception))

    def test_arch_the_sdk_lacks_fails(self):
        for family in pkg.KPACK_FAMILIES:
            touch(self.sdk / ".kpack" / f"{family}_gfx942.kpack")
        with self.assertRaises(pkg.PackagingError) as ctx:
            pkg.build_plan(self.sdk, ["gfx942"], [])
        self.assertIn("gfx942", str(ctx.exception))

    def test_arch_is_required_and_validated(self):
        with self.assertRaises(pkg.PackagingError):
            pkg.build_plan(self.sdk, [], [])
        with self.assertRaises(pkg.PackagingError):
            pkg.build_plan(self.sdk, ["../gfx1151"], [])

    def test_add_binary_goes_into_bin(self):
        exe = self.tmp / "mlxcel.exe"
        touch(exe)
        got = dests(pkg.build_plan(self.sdk, ["gfx1151"], [exe]))
        self.assertIn("bin/mlxcel.exe", got)
        with self.assertRaises(pkg.PackagingError):
            pkg.build_plan(self.sdk, ["gfx1151"], [self.tmp / "absent.exe"])

    def test_add_binary_may_not_replace_a_runtime_dll(self):
        dll = self.tmp / "other" / "AMDHIP64_7.dll"
        touch(dll)
        with self.assertRaises(pkg.PackagingError):
            pkg.build_plan(self.sdk, ["gfx1151"], [dll])

    def test_keep_for_arches(self):
        self.assertTrue(pkg.keep_for_arches("TensileManifest.txt", {"gfx1151"}))
        self.assertTrue(pkg.keep_for_arches("x_gfx1151.co", {"gfx1151"}))
        self.assertFalse(pkg.keep_for_arches("x_gfx1201.co", {"gfx1151"}))


class MainTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._tmp.name)
        self.sdk = make_sdk(self.tmp / "rocm")

    def tearDown(self):
        self._tmp.cleanup()

    def run_main(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = pkg.main(["--rocm-path", str(self.sdk), *args])
        return code, out.getvalue(), err.getvalue()

    def test_zip_is_named_by_version_and_wrapped_in_one_folder(self):
        code, out, _ = self.run_main("--arch", "gfx1151", "--out", str(self.tmp / "dist"))
        self.assertEqual(code, 0)
        zip_path = self.tmp / "dist" / "mlxcel-rocm-runtime-10.1.0-win-x64.zip"
        self.assertTrue(zip_path.is_file(), out)
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
        top = "mlxcel-rocm-runtime-10.1.0-win-x64/"
        self.assertTrue(all(n.startswith(top) for n in names))
        self.assertIn(top + "bin/amdhip64_7.dll", names)
        self.assertFalse((zip_path.parent / (zip_path.name + ".partial")).exists())

    def test_stage_dir(self):
        exe = self.tmp / "mlxcel.exe"
        touch(exe)
        stage = self.tmp / "app"
        code, _, _ = self.run_main(
            "--arch", "gfx1151", "--stage-dir", str(stage), "--add-binary", str(exe)
        )
        self.assertEqual(code, 0)
        self.assertTrue((stage / "bin" / "mlxcel.exe").is_file())
        self.assertTrue((stage / "bin" / "amdhip64_7.dll").is_file())
        self.assertTrue((stage / ".kpack" / "blas_lib_gfx1151.kpack").is_file())
        # A second run refuses to mix into a populated directory.
        code, _, err = self.run_main("--arch", "gfx1151", "--stage-dir", str(stage))
        self.assertEqual(code, 1)
        self.assertIn("not empty", err)

    def test_failure_writes_nothing(self):
        (self.sdk / "bin" / "hiprtc0716.dll").unlink()
        code, _, err = self.run_main("--arch", "gfx1151", "--out", str(self.tmp / "dist"))
        self.assertEqual(code, 1)
        self.assertIn("hiprtc0716.dll", err)
        self.assertFalse((self.tmp / "dist").exists())

    def test_rocm_version_may_not_hold_a_path(self):
        dist = self.tmp / "dist"
        for version in ("../../evil", "10.1.0/x", "..", "a:b"):
            code, _, err = self.run_main(
                "--arch", "gfx1151", "--out", str(dist), "--rocm-version", version
            )
            self.assertEqual(code, 1, version)
            self.assertIn("--rocm-version", err)
        self.assertFalse(dist.exists())
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir()), ["rocm"])

    def test_rocm_version_override_names_the_zip(self):
        code, _, _ = self.run_main(
            "--arch", "gfx1151", "--out", str(self.tmp / "dist"),
            "--rocm-version", "10.1.0-rc1",
        )
        self.assertEqual(code, 0)
        self.assertTrue(
            (self.tmp / "dist" / "mlxcel-rocm-runtime-10.1.0-rc1-win-x64.zip").is_file()
        )

    def test_stage_dir_that_is_a_file_fails_cleanly(self):
        target = self.tmp / "app"
        touch(target)
        code, _, err = self.run_main("--arch", "gfx1151", "--stage-dir", str(target))
        self.assertEqual(code, 1)
        self.assertIn("not a directory", err)


if __name__ == "__main__":
    unittest.main()
