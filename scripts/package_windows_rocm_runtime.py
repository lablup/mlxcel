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
"""Build the Windows ROCm runtime bundle for mlxcel (issue #2606).

A Windows ``--features rocm`` build loads its ROCm DLLs from one runtime
directory: ``MLXCEL_ROCM_RUNTIME_DIR``, else the executable's directory when
it holds the HIP runtime, else the SDK ``bin`` recorded at build time (see
docs/installation.md). This script copies what that directory needs out of a
ROCm SDK, so mlxcel runs on a machine without the SDK.

The bundle keeps the SDK's layout, because the math libraries find their data
relative to their own DLL::

    mlxcel-rocm-runtime-<version>-win-x64/
      bin/                 the DLLs below (the runtime directory)
      bin/rocblas/library/ rocBLAS kernels for the requested --arch targets
      bin/hipblaslt/library/
      .kpack/              rocBLAS, hipBLASLt and rocFFT device code, which
                           they open as <their dir>/../.kpack/<lib>_<arch>.kpack
      include/hip/         headers the hiprtc JIT compiles against
      licenses/<component>/

``--add-binary`` copies mlxcel.exe, mlxcel-server.exe and libopenblas.dll into
``bin/`` to make a complete application folder. A file listed below that the
SDK lacks is an error, so a renamed DLL in a later ROCm release fails here and
not on a user's machine.

Usage::

    python scripts/package_windows_rocm_runtime.py --rocm-path C:\\rocm\\10.1.0 \\
        --arch gfx1151 --out dist
    python scripts/package_windows_rocm_runtime.py --arch gfx1151 \\
        --stage-dir dist\\mlxcel --add-binary target\\release\\mlxcel.exe ...

Standard library only.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

# The DLL closure of `mlxcel generate` and `mlxcel-server`: the five DLLs the
# binaries delay-load (amdhip64_7, rocblas, libhipblaslt, hipfft, hiprtc0716),
# their imports (rocm_kpack, origami, rocfft), and the two DLLs loaded by name
# at run time that the delay-load hook preloads (amd_comgr, hiprtc-builtins).
# Derived from the loaded-module list of both binaries and the import tables
# of these DLLs on ROCm 10.1.0; hipfft and rocfft are imported but load only
# when an FFT runs.
RUNTIME_DLLS = (
    "amdhip64_7.dll",
    "amd_comgr.dll",
    "rocm_kpack.dll",
    "hiprtc0716.dll",
    "hiprtc-builtins0716.dll",
    "rocblas.dll",
    "libhipblaslt.dll",
    "origami.dll",
    "hipfft.dll",
    "rocfft.dll",
)

# Kernel-data directories under bin/ that rocBLAS and hipBLASLt open beside
# their DLLs.
KERNEL_LIBRARIES = ("rocblas/library", "hipblaslt/library")

# Device-code archives under .kpack/, one per library family and architecture.
KPACK_FAMILIES = ("blas_lib", "fft_lib")

# Headers under include/ that the hiprtc JIT needs (hip/hip_runtime.h,
# hip/hip_fp16.h and hip/hip_bfloat16.h and what they include).
INCLUDE_DIRS = ("hip",)

# share/doc/<component> directories whose license files ship with the bundle.
# The HIP runtime, hiprtc and rocm_kpack have no license file in the SDK.
LICENSE_COMPONENTS = (
    "amd_comgr",
    "hipblaslt",
    "hipfft",
    "origami",
    "rocblas",
    "rocfft",
    "rocm-core",
)

GFX_TOKEN = re.compile(r"gfx[0-9a-z]+", re.IGNORECASE)
ARCH_PATTERN = re.compile(r"^gfx[0-9a-f]+$")


class PackagingError(Exception):
    """The SDK lacks a file the bundle needs, or the arguments are invalid."""


class Plan:
    """Files to copy: (source, path relative to the bundle root)."""

    def __init__(self) -> None:
        self.files: list[tuple[Path, str]] = []

    def add(self, source: Path, dest: str) -> None:
        self.files.append((source, dest))

    def total_bytes(self) -> int:
        return sum(src.stat().st_size for src, _ in self.files)


def rocm_version(rocm_path: Path) -> str:
    """The SDK version from ``.info/version`` (TheRock tarballs and wheels)."""
    marker = rocm_path / ".info" / "version"
    if not marker.is_file():
        raise PackagingError(
            f"{marker} does not exist; pass --rocm-version to name the bundle."
        )
    version = marker.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9A-Za-z.+_-]+", version):
        raise PackagingError(f"{marker} holds an unexpected version {version!r}.")
    return version


def keep_for_arches(name: str, arches: set[str]) -> bool:
    """Whether a kernel-library entry belongs in the bundle.

    Entries that name no GPU architecture are arch-independent and kept; the
    rest are kept when they name one of ``arches``.
    """
    tokens = {t.lower() for t in GFX_TOKEN.findall(name)}
    return not tokens or bool(tokens & arches)


def plan_kernel_library(
    plan: Plan, rocm_path: Path, rel: str, arches: set[str]
) -> None:
    root = rocm_path / "bin" / rel
    if not root.is_dir():
        raise PackagingError(f"{root} does not exist.")
    found: set[str] = set()
    for entry in sorted(root.iterdir()):
        if not keep_for_arches(entry.name, arches):
            continue
        found |= {t.lower() for t in GFX_TOKEN.findall(entry.name)} & arches
        if entry.is_dir():
            for path in sorted(p for p in entry.rglob("*") if p.is_file()):
                found |= {
                    t.lower() for t in GFX_TOKEN.findall(path.name)
                } & arches
                plan.add(path, f"bin/{rel}/{path.relative_to(root).as_posix()}")
        else:
            plan.add(entry, f"bin/{rel}/{entry.name}")
    missing = sorted(arches - found)
    if missing:
        raise PackagingError(
            f"{root} has no kernels for {', '.join(missing)}; the SDK was not "
            "built for that architecture."
        )


def build_plan(
    rocm_path: Path, arches: list[str], add_binaries: list[Path]
) -> Plan:
    """Every file of the bundle, or PackagingError naming the first gap."""
    arch_set = {a.lower() for a in arches}
    bad = sorted(a for a in arch_set if not ARCH_PATTERN.match(a))
    if bad:
        raise PackagingError(f"not a GPU architecture: {', '.join(bad)}")
    if not arch_set:
        raise PackagingError("pass at least one --arch, e.g. --arch gfx1151")
    plan = Plan()
    missing: list[str] = []

    for dll in RUNTIME_DLLS:
        path = rocm_path / "bin" / dll
        if path.is_file():
            plan.add(path, f"bin/{dll}")
        else:
            missing.append(str(path))

    for family in KPACK_FAMILIES:
        for arch in sorted(arch_set):
            path = rocm_path / ".kpack" / f"{family}_{arch}.kpack"
            if path.is_file():
                plan.add(path, f".kpack/{path.name}")
            else:
                missing.append(str(path))

    for rel in INCLUDE_DIRS:
        root = rocm_path / "include" / rel
        if not root.is_dir():
            missing.append(str(root))
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            plan.add(path, f"include/{rel}/{path.relative_to(root).as_posix()}")

    for component in LICENSE_COMPONENTS:
        doc = rocm_path / "share" / "doc" / component
        licenses = (
            sorted(p for p in doc.iterdir() if p.name.upper().startswith("LICENSE"))
            if doc.is_dir()
            else []
        )
        if not licenses:
            missing.append(str(doc / "LICENSE*"))
        for path in licenses:
            plan.add(path, f"licenses/{component}/{path.name}")

    jit_header = rocm_path / "include" / "hip" / "hip_runtime.h"
    if not jit_header.is_file() and str(rocm_path / "include" / "hip") not in missing:
        missing.append(str(jit_header))

    if missing:
        raise PackagingError(
            "the ROCm SDK lacks files the bundle needs:\n  " + "\n  ".join(missing)
        )

    for rel in KERNEL_LIBRARIES:
        plan_kernel_library(plan, rocm_path, rel, arch_set)

    for binary in add_binaries:
        if not binary.is_file():
            raise PackagingError(f"--add-binary {binary} does not exist.")
        plan.add(binary, f"bin/{binary.name}")

    dests = [dest.lower() for _, dest in plan.files]
    dupes = sorted({d for d in dests if dests.count(d) > 1})
    if dupes:
        raise PackagingError(f"two files map to {', '.join(dupes)}")
    return plan


def long_path(path: Path) -> Path:
    """``path`` in the form Windows opens past MAX_PATH (260 characters).

    Kernel file names reach 128 characters below ``bin/``, so staging into a
    deep directory exceeds the limit without the ``\\\\?\\`` prefix.
    """
    if os.name != "nt":
        return path
    text = str(path.resolve())
    if text.startswith("\\\\?\\"):
        return path
    if text.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + text[2:])
    return Path("\\\\?\\" + text)


def stage(plan: Plan, dest: Path) -> None:
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
        with zipfile.ZipFile(
            tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as zf:
            for source, rel in plan.files:
                zf.write(source, f"{top}/{rel}")
        os.replace(tmp, zip_path)
    finally:
        if tmp.exists():
            tmp.unlink()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the mlxcel Windows ROCm runtime bundle from a ROCm SDK."
    )
    parser.add_argument(
        "--rocm-path",
        default=os.environ.get("ROCM_PATH") or None,
        help="ROCm SDK root (default: ROCM_PATH)",
    )
    parser.add_argument(
        "--arch",
        action="append",
        default=[],
        help="GPU architecture to keep kernels for, e.g. gfx1151 (repeatable)",
    )
    out = parser.add_mutually_exclusive_group()
    out.add_argument(
        "--out",
        default=".",
        help="directory to write mlxcel-rocm-runtime-<version>-win-x64.zip into "
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
        help="file to copy into bin/, e.g. mlxcel.exe, mlxcel-server.exe and "
        "libopenblas.dll for a complete application (repeatable)",
    )
    parser.add_argument(
        "--rocm-version", help="version for the bundle name (default: .info/version)"
    )
    args = parser.parse_args(argv)
    if not args.rocm_path:
        parser.error("pass --rocm-path or set ROCM_PATH")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rocm_path = Path(args.rocm_path)
    try:
        if not (rocm_path / "bin").is_dir():
            raise PackagingError(f"{rocm_path} is not a ROCm SDK root (no bin/).")
        version = args.rocm_version or rocm_version(rocm_path)
        plan = build_plan(rocm_path, args.arch, [Path(p) for p in args.add_binary])
        top = f"mlxcel-rocm-runtime-{version}-win-x64"
        if args.stage_dir:
            dest = Path(args.stage_dir)
            stage(plan, dest)
        else:
            dest = Path(args.out) / f"{top}.zip"
            write_zip(plan, dest, top)
    except PackagingError as e:
        print(f"package_windows_rocm_runtime: {e}", file=sys.stderr)
        return 1
    mib = plan.total_bytes() / (1 << 20)
    print(f"{dest}: {len(plan.files)} files, {mib:.1f} MiB uncompressed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
