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

"""Tests for scripts/package_windows_cuda_runtime.py (issue #2640).

Each case builds a synthetic CUDA toolkit and build tree in a temporary
directory, so the suite needs no toolkit, GPU or Windows.

Run with:
    python3 -m unittest tests/test_package_windows_cuda_runtime.py
"""

import contextlib
import importlib.util
import io
import pathlib
import tempfile
import unittest
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "package_windows_cuda_runtime.py"

_spec = importlib.util.spec_from_file_location("package_windows_cuda_runtime", SCRIPT)
pkg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pkg)

# The DLL names of CUDA 13.4 and cuDNN 9.27 for each pattern.
CUDA_NAMES = {
    "cublas64_*.dll": "cublas64_13.dll",
    "cublasLt64_*.dll": "cublasLt64_13.dll",
    "cufft64_*.dll": "cufft64_12.dll",
    "cusolver64_*.dll": "cusolver64_12.dll",
    "cusparse64_*.dll": "cusparse64_12.dll",
    "nvJitLink_*.dll": "nvJitLink_130_0.dll",
    "nvrtc64_*.dll": "nvrtc64_130_0.dll",
    "nvrtc-builtins64_*.dll": "nvrtc-builtins64_134.dll",
}

LICENSE_TEXT = {
    "cuda": "End User License Agreement\n2. CUDA Toolkit Supplement to ...\n",
    "cudnn": "LICENSE AGREEMENT FOR NVIDIA SOFTWARE DEVELOPMENT KITS\ncuDNN ...\n",
    "cccl": "libcu++ is under the Apache License v2.0 with LLVM Exceptions\n",
    "cutlass": "Redistribution and use in source and binary forms, with or without\n",
}


def touch(path: pathlib.Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_toolkit(root: pathlib.Path, arch: str = "x64") -> pathlib.Path:
    """A toolkit with cuDNN unpacked into it, installer style (EULA.txt)."""
    for name in CUDA_NAMES.values():
        touch(root / "bin" / arch / name)
    for pattern in pkg.CUDNN_DLLS:
        touch(root / "bin" / arch / pattern.replace("*", "9"))
    touch(root / "bin" / "nvcc.exe")
    for rel in pkg.JIT_HEADERS:
        touch(root / "include" / rel)
    touch(root / "include" / "cuda_runtime_api.h", "#define CUDART_VERSION  13040\n")
    touch(root / "EULA.txt", LICENSE_TEXT["cuda"])
    touch(root / "LICENSE", LICENSE_TEXT["cudnn"])
    return root


def make_deps(root: pathlib.Path, openblas: str = "libopenblas.dll") -> pathlib.Path:
    deps = root / "target" / "release" / "build" / "mlxcel-core-abc" / "out" / "build" / "_deps"
    touch(deps / "cccl-src" / "include" / "cuda" / "std" / "tuple")
    touch(deps / "cccl-src" / "include" / "nv" / "target")
    touch(deps / "cccl-src" / "include" / "thrust" / "version.h")
    touch(deps / "cccl-src" / "LICENSE", LICENSE_TEXT["cccl"])
    touch(deps / "cutlass-src" / "include" / "cute" / "tensor.hpp")
    touch(deps / "cutlass-src" / "include" / "cutlass" / "cutlass.h")
    touch(deps / "cutlass-src" / "LICENSE.txt", LICENSE_TEXT["cutlass"])
    touch(deps / "openblas-src" / "bin" / openblas)
    # The build-script half of the crate has no _deps.
    (root / "target" / "release" / "build" / "mlxcel-core-def").mkdir(parents=True)
    return deps


class PackageTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._tmp.name)
        self.cuda = make_toolkit(self.tmp / "cuda")
        self.deps = make_deps(self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def plan(self, arch="x64", add=(), licenses=None, cudnn=None):
        return pkg.build_plan(
            self.cuda,
            cudnn or self.cuda,
            self.deps,
            arch,
            [pathlib.Path(p) for p in add],
            licenses or {},
        )

    def dests(self, plan):
        return {dest for _, dest in plan.files}

    def assert_fails_naming(self, expected, **kwargs):
        with self.assertRaises(pkg.PackagingError) as cm:
            self.plan(**kwargs)
        self.assertIn(str(expected), str(cm.exception))

    def test_contents(self):
        dests = self.dests(self.plan())
        for name in CUDA_NAMES.values():
            self.assertIn(f"bin/{name}", dests)
        self.assertIn("bin/cudnn64_9.dll", dests)
        self.assertIn("bin/cudnn_engines_runtime_compiled64_9.dll", dests)
        self.assertIn("bin/libopenblas.dll", dests)
        self.assertIn("include/cuda_fp16.h", dests)
        self.assertIn("include/cooperative_groups/details/sync.h", dests)
        self.assertIn("include/cccl/cuda/std/tuple", dests)
        self.assertIn("include/cccl/nv/target", dests)
        self.assertIn("include/cute/tensor.hpp", dests)
        self.assertIn("include/cutlass/cutlass.h", dests)
        self.assertIn("licenses/cuda/EULA.txt", dests)
        self.assertIn("licenses/cudnn/LICENSE", dests)
        self.assertIn("licenses/cccl/LICENSE", dests)
        self.assertIn("licenses/cutlass/LICENSE.txt", dests)
        # Only the CCCL trees MLX installs, and no driver or unrelated DLL.
        self.assertFalse(any(d.startswith("include/cccl/thrust") for d in dests))
        self.assertFalse(any("nvcuda" in d or "cudart" in d for d in dests))

    def test_arm64_layout(self):
        self.cuda = make_toolkit(self.tmp / "cuda-arm", arch="arm64")
        make_deps(self.tmp / "arm", openblas="openblas.dll")
        self.deps = self.tmp / "arm" / "target" / "release" / "build" / "mlxcel-core-abc" / "out" / "build" / "_deps"
        dests = self.dests(self.plan(arch="arm64"))
        self.assertIn("bin/cublasLt64_13.dll", dests)
        self.assertIn("bin/openblas.dll", dests)
        self.assert_fails_naming(self.cuda / "bin" / "x64" / "cublas64_*.dll", arch="x64")

    def test_unknown_arch_fails(self):
        self.assert_fails_naming("--arch must be one of", arch="amd64")

    def test_missing_dll_fails_and_names_it(self):
        (self.cuda / "bin" / "x64" / "nvrtc-builtins64_134.dll").unlink()
        self.assert_fails_naming(self.cuda / "bin" / "x64" / "nvrtc-builtins64_*.dll")

    def test_renamed_dll_does_not_count(self):
        dll = self.cuda / "bin" / "x64" / "cudnn_graph64_9.dll"
        dll.rename(dll.with_name("cudnn_graph64_9.dll.off"))
        self.assert_fails_naming(self.cuda / "bin" / "x64" / "cudnn_graph64_*.dll")

    def test_alt_variant_is_not_a_second_match(self):
        touch(self.cuda / "bin" / "x64" / "nvrtc64_130_0.alt.dll")
        dests = self.dests(self.plan())
        self.assertIn("bin/nvrtc64_130_0.dll", dests)
        self.assertNotIn("bin/nvrtc64_130_0.alt.dll", dests)

    def test_two_versions_of_a_dll_fail(self):
        touch(self.cuda / "bin" / "x64" / "nvrtc-builtins64_135.dll")
        self.assert_fails_naming("more than one nvrtc-builtins64_*.dll")

    def test_every_missing_file_is_named(self):
        (self.cuda / "bin" / "x64" / "cufft64_12.dll").unlink()
        (self.cuda / "include" / "math_constants.h").unlink()
        with self.assertRaises(pkg.PackagingError) as cm:
            self.plan()
        self.assertIn("cufft64_*.dll", str(cm.exception))
        self.assertIn("math_constants.h", str(cm.exception))

    def test_missing_header_fails_and_names_it(self):
        (self.cuda / "include" / "cuda_bf16.hpp").unlink()
        self.assert_fails_naming(self.cuda / "include" / "cuda_bf16.hpp")

    def test_missing_header_tree_fails_and_names_it(self):
        for p in (self.deps / "cutlass-src" / "include" / "cute").rglob("*"):
            p.unlink()
        self.assert_fails_naming(self.deps / "cutlass-src" / "include" / "cute")

    def test_missing_openblas_fails(self):
        (self.deps / "openblas-src" / "bin" / "libopenblas.dll").unlink()
        self.assert_fails_naming(self.deps / "openblas-src" / "bin" / "libopenblas.dll")

    def test_missing_license_fails_and_names_it(self):
        (self.deps / "cutlass-src" / "LICENSE.txt").unlink()
        self.assert_fails_naming(self.deps / "cutlass-src" / "LICENSE.txt")

    def test_missing_cudnn_license_fails(self):
        (self.cuda / "LICENSE").unlink()
        self.assert_fails_naming("cudnn license")

    def test_overwritten_license_is_not_taken_for_another(self):
        # Archives unpacked into one directory: LICENSE is cuDNN's, EULA.txt
        # absent. The CUDA EULA must not be satisfied by the cuDNN license.
        (self.cuda / "EULA.txt").unlink()
        self.assert_fails_naming("--cuda-license")

    def test_license_flags(self):
        (self.cuda / "EULA.txt").unlink()
        (self.deps / "cccl-src" / "LICENSE").unlink()
        cuda_lic = self.tmp / "lic" / "cuda" / "LICENSE"
        cccl_lic = self.tmp / "lic" / "cccl" / "LICENSE"
        touch(cuda_lic, LICENSE_TEXT["cuda"])
        touch(cccl_lic, LICENSE_TEXT["cccl"])
        plan = self.plan(licenses={"cuda": cuda_lic, "cccl": cccl_lic})
        sources = {dest: src for src, dest in plan.files}
        self.assertEqual(sources["licenses/cuda/LICENSE"], cuda_lic)
        self.assertEqual(sources["licenses/cccl/LICENSE"], cccl_lic)

    def test_license_flag_with_the_wrong_file_fails(self):
        wrong = self.tmp / "wrong" / "LICENSE"
        touch(wrong, LICENSE_TEXT["cudnn"])
        self.assert_fails_naming(wrong, licenses={"cuda": wrong})

    def test_separate_cudnn_root(self):
        cudnn = self.tmp / "cudnn"
        for dll in (self.cuda / "bin" / "x64").glob("cudnn*.dll"):
            touch(cudnn / "bin" / "x64" / dll.name)
            dll.unlink()
        (self.cuda / "LICENSE").rename(cudnn / "LICENSE")
        plan = self.plan(cudnn=cudnn)
        sources = {dest: src for src, dest in plan.files}
        self.assertEqual(sources["bin/cudnn64_9.dll"], cudnn / "bin" / "x64" / "cudnn64_9.dll")
        self.assertEqual(sources["licenses/cudnn/LICENSE"], cudnn / "LICENSE")

    def test_add_binary_goes_into_bin(self):
        exe = self.tmp / "out" / "mlxcel.exe"
        touch(exe)
        self.assertIn("bin/mlxcel.exe", self.dests(self.plan(add=[exe])))

    def test_add_binary_may_not_replace_a_runtime_dll(self):
        dll = self.tmp / "other" / "cublasLt64_13.dll"
        touch(dll)
        self.assert_fails_naming("two files map to bin/cublaslt64_13.dll", add=[dll])

    def test_cuda_version(self):
        self.assertEqual(pkg.cuda_version(self.cuda), "13.4")

    def test_find_deps_dir(self):
        self.assertEqual(pkg.find_deps_dir(self.tmp / "target", "release"), self.deps)
        with self.assertRaises(pkg.PackagingError):
            pkg.find_deps_dir(self.tmp / "target", "debug")
        make_deps(self.tmp / "second")
        twin = self.tmp / "target" / "release" / "build" / "mlxcel-core-xyz" / "out" / "build" / "_deps"
        (twin / "cutlass-src" / "include").mkdir(parents=True)
        with self.assertRaises(pkg.PackagingError) as cm:
            pkg.find_deps_dir(self.tmp / "target", "release")
        self.assertIn("--deps-dir", str(cm.exception))

    def run_main(self, *args):
        err = io.StringIO()
        out = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            code = pkg.main(
                ["--cuda-path", str(self.cuda), "--deps-dir", str(self.deps), "--arch", "x64", *args]
            )
        return code, out.getvalue(), err.getvalue()

    def test_zip_is_named_by_version_and_wrapped_in_one_folder(self):
        code, out, _ = self.run_main("--out", str(self.tmp / "dist"))
        self.assertEqual(code, 0)
        zpath = self.tmp / "dist" / "mlxcel-cuda-runtime-13.4-win-x64.zip"
        self.assertTrue(zpath.is_file(), out)
        with zipfile.ZipFile(zpath) as zf:
            names = zf.namelist()
        self.assertTrue(all(n.startswith("mlxcel-cuda-runtime-13.4-win-x64/") for n in names))
        self.assertIn("mlxcel-cuda-runtime-13.4-win-x64/bin/cudnn64_9.dll", names)

    def test_stage_dir(self):
        dest = self.tmp / "stage"
        code, _, _ = self.run_main("--stage-dir", str(dest))
        self.assertEqual(code, 0)
        self.assertTrue((dest / "bin" / "nvrtc64_130_0.dll").is_file())
        self.assertTrue((dest / "include" / "cuda_runtime_api.h").is_file())
        code, _, err = self.run_main("--stage-dir", str(dest))
        self.assertEqual(code, 1)
        self.assertIn("is not empty", err)

    def test_failure_exits_non_zero_names_the_file_and_writes_nothing(self):
        (self.cuda / "bin" / "x64" / "cusparse64_12.dll").unlink()
        code, _, err = self.run_main("--out", str(self.tmp / "dist"))
        self.assertEqual(code, 1)
        self.assertIn("cusparse64_*.dll", err)
        self.assertFalse((self.tmp / "dist").exists())

    def test_cuda_version_may_not_hold_a_path(self):
        code, _, err = self.run_main("--cuda-version", "..\\x", "--out", str(self.tmp / "d"))
        self.assertEqual(code, 1)
        self.assertIn("--cuda-version", err)


if __name__ == "__main__":
    unittest.main()
