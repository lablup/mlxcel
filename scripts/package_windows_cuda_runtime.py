#!/usr/bin/env python3
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
"""Build the Windows CUDA runtime bundle for mlxcel (issue #2640).

A Windows ``--features cuda`` build loads its CUDA, cuDNN and NVRTC DLLs from
one runtime directory: ``MLXCEL_CUDA_RUNTIME_DIR``, else the executable's
directory when it holds the CUDA runtime DLL the binary delay-loads, else the
toolkit recorded at build time (see docs/installation.md). This script copies
what that directory needs out of a CUDA toolkit, cuDNN and the build tree, so
mlxcel runs on a machine with only the NVIDIA display driver::

    mlxcel-cuda-runtime-<version>-win-<x64|arm64>/
      bin/                 the DLLs below and the build's OpenBLAS DLL
                           (the runtime directory)
      include/             headers the NVRTC JIT compiles against: the CUDA
                           headers it reaches, cccl/ (the CCCL the build
                           fetched), cute/ and cutlass/
      licenses/<component>/

The display driver's ``nvcuda.dll`` is not bundled. ``--add-binary`` copies
mlxcel.exe and mlxcel-server.exe into ``bin/`` to make a complete
application folder. A file listed below that the inputs lack is an error, so a
DLL renamed in a later CUDA or cuDNN release fails here and not on a user's
machine.

Usage::

    python scripts/package_windows_cuda_runtime.py --cuda-path C:\\cuda --out dist
    python scripts/package_windows_cuda_runtime.py --stage-dir dist\\mlxcel \\
        --add-binary target\\release\\mlxcel.exe ...

Standard library only.
"""

from __future__ import annotations

import argparse
import collections
import fnmatch
import os
import platform
import re
import shutil
import sys
import zipfile
from pathlib import Path

# The DLL closure of `mlxcel generate` and `mlxcel-server`, as file patterns
# that must each match exactly one DLL in the toolkit's bin\<arch>. The
# binaries delay-load cuBLASLt, cuFFT, cuSOLVER, NVRTC and cuDNN (nvcc links the
# static CUDA runtime); cuSOLVER imports cuBLAS and cuSPARSE; and these DLLs
# are opened by plain name at run time: nvrtc-builtins (by NVRTC), NVRTC (by
# cuBLASLt and cuDNN), cuBLAS and cuBLASLt (by cuDNN), nvJitLink (by cuSPARSE).
# Derived statically from the import tables and embedded DLL names of CUDA
# 13.4.3 / cuDNN 9.27 for windows-x86_64 and the delay-load hook's preload
# list; to be confirmed against the loaded-module list of a run on an NVIDIA
# host (lablup/mlxcel#2640).
CUDA_DLLS = (
    "cublas64_*.dll",
    "cublasLt64_*.dll",
    "cufft64_*.dll",
    "cusolver64_*.dll",
    "cusparse64_*.dll",
    "nvJitLink_*.dll",
    "nvrtc64_*.dll",
    "nvrtc-builtins64_*.dll",
)

# cuDNN and the sub-libraries cudnn64_9 and cudnn_graph open by plain name,
# in cuDNN's bin\<arch>.
CUDNN_DLLS = (
    "cudnn64_*.dll",
    "cudnn_adv64_*.dll",
    "cudnn_cnn64_*.dll",
    "cudnn_engines_precompiled64_*.dll",
    "cudnn_engines_runtime_compiled64_*.dll",
    "cudnn_engines_tensor_ir64_*.dll",
    "cudnn_ext64_*.dll",
    "cudnn_graph64_*.dll",
    "cudnn_heuristic64_*.dll",
    "cudnn_ops64_*.dll",
)

# Toolkit headers the NVRTC JIT reaches from the kernels MLX compiles at run
# time (cooperative_groups.h, cuda_bf16.h, cuda_fp16.h, cuda_fp8.h,
# math_constants.h, and cuda_runtime.h and mma.h through CUTLASS/CuTe, with
# their includes). Checked by compiling each header MLX embeds for the JIT
# with NVRTC 13.4 (sm_120a, sm_90a for qmm_sm90) against only a staged
# bundle's include/, which needs no GPU. The CUDA EULA's Attachment A lists
# cuComplex.h, the floating-point type headers, crt/host_defines.h,
# cuda_runtime_api.h, device_types.h, vector_functions.h and vector_types.h
# as redistributable; the others below are not on that list (see
# docs/installation.md).
JIT_HEADERS = (
    "builtin_types.h",
    "cooperative_groups.h",
    "cooperative_groups/details/driver_abi.h",
    "cooperative_groups/details/helpers.h",
    "cooperative_groups/details/info.h",
    "cooperative_groups/details/invoke.h",
    "cooperative_groups/details/memory.h",
    "cooperative_groups/details/partitioning.h",
    "cooperative_groups/details/sync.h",
    "crt/host_defines.h",
    "crt/mma.h",
    "crt/mma.hpp",
    "cuComplex.h",
    "cuda_bf16.h",
    "cuda_bf16.hpp",
    "cuda_device_runtime_api.h",
    "cuda_fp16.h",
    "cuda_fp16.hpp",
    "cuda_fp8.h",
    "cuda_fp8.hpp",
    "cuda_runtime.h",
    "cuda_runtime_api.h",
    "device_types.h",
    "driver_types.h",
    "math_constants.h",
    "mma.h",
    "surface_types.h",
    "texture_types.h",
    "vector_functions.h",
    "vector_functions.hpp",
    "vector_types.h",
)

# Header trees from the build's FetchContent dependencies (build/_deps):
# (source below _deps, destination below include/). MLX installs CCCL's cuda/
# and nv/ under include/cccl, which jit_module.cpp searches.
DEPS_TREES = (
    ("cccl-src/include/cuda", "cccl/cuda"),
    ("cccl-src/include/nv", "cccl/nv"),
    ("cutlass-src/include/cute", "cute"),
    ("cutlass-src/include/cutlass", "cutlass"),
)

# License files: (component, text every candidate must contain). The CUDA
# EULA covers the toolkit DLLs and headers, the cuDNN license cuDNN, and the
# open-source licenses CCCL (Apache 2.0 with LLVM Exceptions for libcu++) and
# CUTLASS (BSD-3-Clause). The markers tell the files apart when the
# redistributable archives were unpacked into one directory, where each
# archive's LICENSE overwrote the previous one.
LICENSE_MARKERS = {
    "cuda": "CUDA Toolkit Supplement",
    "cudnn": "cuDNN",
    "cccl": "LLVM Exceptions",
    "cutlass": "Redistribution and use in source and binary forms",
}

ARCHES = ("x64", "arm64")
VERSION_PATTERN = re.compile(r"[0-9A-Za-z][0-9A-Za-z.+_-]*")


class PackagingError(Exception):
    """The inputs lack a file the bundle needs, or the arguments are invalid."""


class Plan:
    """Files to copy: (source, path relative to the bundle root)."""

    def __init__(self) -> None:
        self.files: list[tuple[Path, str]] = []

    def add(self, source: Path, dest: str) -> None:
        self.files.append((source, dest))

    def total_bytes(self) -> int:
        return sum(src.stat().st_size for src, _ in self.files)


def match_one(directory: Path, pattern: str) -> Path | str:
    """The one DLL in ``directory`` matching ``pattern`` (case-insensitive,
    ``.alt.dll`` variants ignored), or the expected path as a string when
    there is none. Several matches are an error naming them."""
    names = sorted(p for p in directory.iterdir() if p.is_file()) if directory.is_dir() else []
    found = [
        p
        for p in names
        if fnmatch.fnmatchcase(p.name.lower(), pattern.lower()) and ".alt." not in p.name.lower()
    ]
    if not found:
        return str(directory / pattern)
    if len(found) > 1:
        raise PackagingError(
            f"{directory} holds more than one {pattern}: "
            + ", ".join(p.name for p in found)
        )
    return found[0]


def cuda_version(cuda_path: Path) -> str:
    """``<major>.<minor>`` from CUDART_VERSION in include/cuda_runtime_api.h."""
    header = cuda_path / "include" / "cuda_runtime_api.h"
    if not header.is_file():
        raise PackagingError(f"{header} does not exist; pass --cuda-version.")
    m = re.search(r"#define\s+CUDART_VERSION\s+(\d+)", header.read_text(errors="replace"))
    if not m:
        raise PackagingError(f"{header} defines no CUDART_VERSION; pass --cuda-version.")
    value = int(m.group(1))
    return f"{value // 1000}.{value % 1000 // 10}"


def checked_version(version: str) -> str:
    if not VERSION_PATTERN.fullmatch(version):
        raise PackagingError(
            f"--cuda-version {version!r} must be letters, digits and . + _ - only."
        )
    return version


def find_license(candidates: list[Path], marker: str) -> Path | None:
    """The first existing candidate whose text contains ``marker``."""
    for path in candidates:
        if path.is_file() and marker in path.read_text(encoding="utf-8", errors="replace"):
            return path
    return None


def find_deps_dir(target_dir: Path, profile: str) -> Path:
    """The build's ``_deps`` directory: the one mlxcel-core build output under
    ``<target_dir>/<profile>/build`` that fetched CUTLASS."""
    found = sorted(
        p
        for p in (target_dir / profile / "build").glob("mlxcel-core-*/out/build/_deps")
        if (p / "cutlass-src" / "include").is_dir()
    )
    if not found:
        raise PackagingError(
            f"no CUDA build under {target_dir / profile / 'build'} (looked for "
            "mlxcel-core-*/out/build/_deps/cutlass-src); build with --features cuda "
            "first, or pass --deps-dir."
        )
    if len(found) > 1:
        raise PackagingError(
            "more than one CUDA build output; pass --deps-dir with one of:\n  "
            + "\n  ".join(str(p) for p in found)
        )
    return found[0]


def build_plan(
    cuda_path: Path,
    cudnn_path: Path,
    deps_dir: Path,
    arch: str,
    add_binaries: list[Path],
    licenses: dict[str, Path | None],
) -> Plan:
    """Every file of the bundle, or PackagingError naming every gap."""
    if arch not in ARCHES:
        raise PackagingError(f"--arch must be one of {', '.join(ARCHES)}, not {arch!r}")
    plan = Plan()
    missing: list[str] = []

    for root, patterns in ((cuda_path, CUDA_DLLS), (cudnn_path, CUDNN_DLLS)):
        for pattern in patterns:
            found = match_one(root / "bin" / arch, pattern)
            if isinstance(found, Path):
                plan.add(found, f"bin/{found.name}")
            else:
                missing.append(found)

    openblas = [
        deps_dir / "openblas-src" / "bin" / name
        for name in ("libopenblas.dll", "openblas.dll")
        if (deps_dir / "openblas-src" / "bin" / name).is_file()
    ]
    if openblas:
        plan.add(openblas[0], f"bin/{openblas[0].name}")
    else:
        missing.append(str(deps_dir / "openblas-src" / "bin" / "libopenblas.dll"))

    for rel in JIT_HEADERS:
        path = cuda_path / "include" / rel
        if path.is_file():
            plan.add(path, f"include/{rel}")
        else:
            missing.append(str(path))

    for src, dest in DEPS_TREES:
        root = deps_dir / src
        files = sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else []
        if not files:
            missing.append(str(root))
        for path in files:
            plan.add(path, f"include/{dest}/{path.relative_to(root).as_posix()}")

    cutlass_root = deps_dir / "cutlass-src"
    defaults = {
        "cuda": [cuda_path / "EULA.txt", cuda_path / "LICENSE"],
        "cudnn": [cudnn_path / "LICENSE", cudnn_path / "LICENSE.txt"],
        "cccl": [deps_dir / "cccl-src" / "LICENSE", cuda_path / "include" / "cccl" / "LICENSE"],
        "cutlass": [cutlass_root / "LICENSE.txt"],
    }
    for component, marker in LICENSE_MARKERS.items():
        given = licenses.get(component)
        candidates = [given] if given else defaults[component]
        found = find_license(candidates, marker)
        if found is None:
            missing.append(
                f"{' or '.join(str(c) for c in candidates)} "
                f"({component} license, a file containing {marker!r}; "
                f"pass --{component}-license)"
            )
        else:
            plan.add(found, f"licenses/{component}/{found.name}")
    if (cutlass_root / "EULA.txt").is_file():
        plan.add(cutlass_root / "EULA.txt", "licenses/cutlass/EULA.txt")

    if missing:
        raise PackagingError(
            "the inputs lack files the bundle needs:\n  " + "\n  ".join(missing)
        )

    for binary in add_binaries:
        if not binary.is_file():
            raise PackagingError(f"--add-binary {binary} does not exist.")
        plan.add(binary, f"bin/{binary.name}")

    counts = collections.Counter(dest.lower() for _, dest in plan.files)
    dupes = sorted(d for d, n in counts.items() if n > 1)
    if dupes:
        raise PackagingError(f"two files map to {', '.join(dupes)}")
    return plan


def long_path(path: Path) -> Path:
    """``path`` in the form Windows opens past MAX_PATH (260 characters);
    CUTLASS header paths are deep."""
    if os.name != "nt":
        return path
    text = str(path.resolve())
    if text.startswith("\\\\?\\"):
        return path
    if text.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + text[2:])
    return Path("\\\\?\\" + text)


def stage(plan: Plan, dest: Path) -> None:
    if dest.exists() and not dest.is_dir():
        raise PackagingError(f"{dest} is not a directory.")
    if dest.exists() and any(dest.iterdir()):
        raise PackagingError(f"{dest} is not empty.")
    for source, rel in plan.files:
        target = long_path(dest) / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def write_zip(plan: Plan, zip_path: Path, top: str) -> None:
    if zip_path.exists():
        raise PackagingError(f"{zip_path} already exists.")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = zip_path.with_name(zip_path.name + ".partial")
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for source, rel in plan.files:
                zf.write(source, f"{top}/{rel}")
        os.replace(tmp, zip_path)
    finally:
        if tmp.exists():
            tmp.unlink()


def default_arch() -> str:
    machine = platform.machine().lower()
    return "arm64" if machine in ("arm64", "aarch64") else "x64"


def parse_args(argv: list[str]) -> argparse.Namespace:
    env = os.environ
    parser = argparse.ArgumentParser(
        description="Build the mlxcel Windows CUDA runtime bundle from a CUDA toolkit."
    )
    parser.add_argument(
        "--cuda-path",
        default=env.get("CUDA_HOME") or env.get("CUDA_PATH") or None,
        help="CUDA toolkit root (default: CUDA_HOME, else CUDA_PATH)",
    )
    parser.add_argument(
        "--cudnn-path",
        default=env.get("CUDNN_PATH") or None,
        help="cuDNN root when it is not inside the toolkit (default: CUDNN_PATH)",
    )
    parser.add_argument(
        "--arch",
        default=default_arch(),
        help="bin\\ and lib\\ subdirectory, x64 or arm64 (default: this machine)",
    )
    parser.add_argument(
        "--target-dir",
        default=env.get("CARGO_TARGET_DIR") or "target",
        help="cargo target directory of the CUDA build (default: CARGO_TARGET_DIR or target)",
    )
    parser.add_argument("--profile", default="release", help="cargo profile (default: release)")
    parser.add_argument(
        "--deps-dir",
        help="the build's out/build/_deps directory (default: found under --target-dir)",
    )
    out = parser.add_mutually_exclusive_group()
    out.add_argument(
        "--out",
        default=".",
        help="directory to write mlxcel-cuda-runtime-<version>-win-<arch>.zip into "
        "(default: the current directory)",
    )
    out.add_argument(
        "--stage-dir",
        help="copy the bundle tree into this empty directory instead of zipping it",
    )
    parser.add_argument(
        "--add-binary",
        action="append",
        default=[],
        help="file to copy into bin/, e.g. mlxcel.exe and mlxcel-server.exe (repeatable)",
    )
    parser.add_argument("--cuda-version", help="version for the bundle name")
    for component in LICENSE_MARKERS:
        parser.add_argument(
            f"--{component}-license",
            help=f"the {component} license file, when it is not found in its root",
        )
    args = parser.parse_args(argv)
    if not args.cuda_path:
        parser.error("pass --cuda-path or set CUDA_PATH")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    cuda_path = Path(args.cuda_path)
    cudnn_path = Path(args.cudnn_path) if args.cudnn_path else cuda_path
    try:
        if not (cuda_path / "bin").is_dir():
            raise PackagingError(f"{cuda_path} is not a CUDA toolkit root (no bin/).")
        version = (
            checked_version(args.cuda_version) if args.cuda_version else cuda_version(cuda_path)
        )
        deps_dir = (
            Path(args.deps_dir)
            if args.deps_dir
            else find_deps_dir(Path(args.target_dir), args.profile)
        )
        licenses = {
            c: Path(v) if (v := getattr(args, f"{c}_license")) else None
            for c in LICENSE_MARKERS
        }
        plan = build_plan(
            cuda_path,
            cudnn_path,
            deps_dir,
            args.arch,
            [Path(p) for p in args.add_binary],
            licenses,
        )
        top = f"mlxcel-cuda-runtime-{version}-win-{args.arch}"
        if args.stage_dir:
            dest = Path(args.stage_dir)
            stage(plan, dest)
        else:
            dest = Path(args.out) / f"{top}.zip"
            write_zip(plan, dest, top)
    except PackagingError as e:
        print(f"package_windows_cuda_runtime: {e}", file=sys.stderr)
        return 1
    mib = plan.total_bytes() / (1 << 20)
    print(f"{dest}: {len(plan.files)} files, {mib:.1f} MiB uncompressed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
