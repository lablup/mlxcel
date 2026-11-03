#!/usr/bin/env python3
"""Regression guards for the cuda-oxide smoke tooling (issue #2831).

Each guard feeds a malformed input through the real code and requires a clean
non-zero result: a message, no traceback, and no hang. The cases correspond to
the review findings: a zero-length `.oxart` bundle, absolute and escaping
artifact paths, an implausible kparam alignment exponent, an out-of-range
parameter ordinal, a zero block dimension, a shortened run whose own totals are
consistent, and a duplicate case.

Only the Python standard library is used. The harness is run from the package
root by `scripts/check-lint.sh`.
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
SPIKE = SCRIPTS.parent
PYTHON = sys.executable
FAILURES: list[str] = []


def record(label: str, problems: list[str], output: str = "") -> None:
    if problems:
        FAILURES.append(label)
        print(f"FAIL guard={label} reason={'; '.join(problems)}")
        for line in output.strip().splitlines()[-5:]:
            print(f"  {line}")
    else:
        print(f"PASS guard={label}")


def run_tool(arguments: list[str], timeout: float = 20.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PYTHON, *arguments],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=SPIKE,
    )


def expect_clean_failure(
    label: str, arguments: list[str], needles: tuple[str, ...], timeout: float = 20.0
) -> None:
    try:
        result = run_tool(arguments, timeout)
    except subprocess.TimeoutExpired:
        record(label, ["timed out (hang)"])
        return
    output = result.stdout + result.stderr
    problems = []
    if result.returncode == 0:
        problems.append("exit=0")
    if "Traceback" in output:
        problems.append("traceback")
    for needle in needles:
        if needle not in output:
            problems.append(f"missing {needle!r}")
    record(label, problems, output)


def expect_clean_snippet(
    label: str, snippet: str, needles: tuple[str, ...], timeout: float = 10.0
) -> None:
    """Run a snippet whose tested call must raise cleanly; the wrapper exits 0."""
    try:
        result = run_tool(["-c", snippet], timeout)
    except subprocess.TimeoutExpired:
        record(label, ["timed out (hang)"])
        return
    output = result.stdout + result.stderr
    problems = []
    if result.returncode != 0:
        problems.append(f"exit={result.returncode}")
    if "Traceback" in output:
        problems.append("traceback")
    if "CLEAN" not in output:
        problems.append("missing 'CLEAN'")
    for needle in needles:
        if needle not in output:
            problems.append(f"missing {needle!r}")
    record(label, problems, output)


def expect_value_error(label: str, snippet: str, needles: tuple[str, ...]) -> None:
    """Run a snippet in-process; it must raise a clean ValueError with a message."""
    try:
        exec(compile(snippet, label, "exec"), {"__name__": "__guard__"})
    except ValueError as error:
        message = str(error)
        problems = [f"missing {needle!r}" for needle in needles if needle not in message]
        record(label, problems, message)
        return
    except BaseException as error:  # noqa: BLE001 - the guard reports the type
        record(label, [f"raised {type(error).__name__}: {error}"])
        return
    record(label, ["no exception raised"])


def artifact_manifest(overrides: dict | None = None) -> dict:
    manifest = json.loads((SPIKE / "artifacts" / "vecadd.manifest.json").read_text())
    if overrides:
        for path, value in overrides.items():
            target = manifest
            keys = path.split(".")
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
    return manifest


def write_partial_artifact_root(root: Path, manifest: dict) -> Path:
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    for name in ("vecadd.sm_121.ptx", "vecadd.sm_121.cubin"):
        shutil.copy2(SPIKE / "artifacts" / name, artifacts / name)
    manifest_path = artifacts / "vecadd.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def guard_zero_length_bundle(root: Path) -> None:
    blob = (
        b"OXIDEART"
        + struct.pack("<HHI", 1, 32, 0)
        + struct.pack("<HHHH", 0, 0, 0, 0)
        + struct.pack("<Q", 0)
    )
    blob_path = root / "zero-length.oxart"
    blob_path.write_bytes(blob)
    snippet = f"""
import sys
sys.path.insert(0, {str(SCRIPTS)!r})
import oxide_format
data = open({str(blob_path)!r}, 'rb').read()
try:
    oxide_format.parse_artifact_section(data)
except ValueError as error:
    print('CLEAN', error)
    sys.exit(0)
print('NO-ERROR')
sys.exit(3)
"""
    expect_clean_snippet(
        "zero-length-bundle",
        snippet,
        ("CLEAN", "smaller than the"),
    )


def guard_huge_log_align(root: Path) -> None:
    fake = root / "fake-cuobjdump"
    fake.write_text(
        """#!/bin/sh
case "$1" in
  -elf)
    cat <<'EOF'
EIATTR_KPARAM_INFO
\tFormat:\tEIFMT_SVAL
\tValue:\tIndex : 0x0\tOrdinal : 0x0\tOffset  : 0x0\tSize    : 0x8
\t\tPointee's logAlignment : 0x41\tSpace : 0x5
EOF
    ;;
  -sass) printf '\\t.target\\tsm_121\\n' ;;
  -symbols) printf 'STT_FUNC STB_GLOBAL STO_ENTRY vecadd\\n' ;;
esac
"""
    )
    fake.chmod(0o755)
    snippet = f"""
import sys
from pathlib import Path
sys.path.insert(0, {str(SCRIPTS)!r})
import oxide_format
try:
    oxide_format.parse_cubin({str(fake)!r}, Path('unused.cubin'))
except ValueError as error:
    print('CLEAN', error)
    sys.exit(0)
print('NO-ERROR')
sys.exit(3)
"""
    expect_clean_snippet(
        "huge-log-align",
        snippet,
        ("CLEAN", "logAlignment", "above the supported bound"),
    )


def guard_ordinal_out_of_range() -> None:
    snippet = f"""
import sys
sys.path.insert(0, {str(SCRIPTS)!r})
import oxide_artifact
schema = {{"arguments": [{{"ordinal": i, "name": f"a{{i}}"}} for i in range(7)]}}
ptx_info = {{"parameters": [{{"ptx_type": ".u64", "pointer": False, "pointee_align_bytes": 0}}] * 6}}
cubin_info = {{"kparams": [
    {{"ordinal": i, "size_bytes": 8, "byte_offset": i * 8,
      "space": "value", "pointee_align_bytes": 1}}
    for i in range(6)
]}}
oxide_artifact.build_arguments(schema, ptx_info, cubin_info)
"""
    expect_value_error("ordinal-out-of-range", snippet, ("ordinal 6", "6 parameters"))


def guard_artifact_paths(root: Path) -> None:
    for label, relative, needle in (
        ("absolute-artifact-path", "/dev/zero", "absolute"),
        ("escaping-artifact-path", "../../../etc/passwd", "escapes the artifact root"),
    ):
        manifest = artifact_manifest({"artifacts.ptx.path": relative})
        manifest_path = write_partial_artifact_root(root / label, manifest)
        expect_clean_failure(
            label,
            [
                "scripts/oxide_artifact.py",
                "validate",
                "--manifest",
                str(manifest_path),
                "--schema",
                "abi/vecadd.schema.json",
            ],
            (needle,),
        )


def guard_zero_block(root: Path) -> None:
    manifest = artifact_manifest({"launch.block_dim": [0, 1, 1]})
    manifest_path = root / "zero-block.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log = root / "zero-block.log"
    log.write_text(
        "PASS case=small n=1 grid=1 block=1 elements=1\n"
        "RESULT standalone=1 mlx=0 cases_passed=1 cases_total=1 "
        "elements_checked=1 comparison=exact-bitwise\n"
    )
    expect_clean_failure(
        "zero-block-dim",
        [
            "scripts/oxide_artifact.py",
            "check-launch",
            "--manifest",
            str(manifest_path),
            "--log",
            str(log),
        ],
        ("block dimension must be positive",),
    )


def write_launch_log(path: Path, cases: list[tuple[str, int]], block: int = 256) -> None:
    lines = []
    elements = 0
    for name, n in cases:
        grid = (n + block - 1) // block
        lines.append(f"PASS case={name} n={n} grid={grid} block={block} elements={n}")
        elements += n
    lines.append(
        "RESULT standalone=1 mlx=0 "
        f"cases_passed={len(cases)} cases_total={len(cases)} "
        f"elements_checked={elements} comparison=exact-bitwise"
    )
    path.write_text("\n".join(lines) + "\n")


def guard_shortened_run(root: Path) -> None:
    manifest_path = root / "shortened.manifest.json"
    manifest_path.write_text(json.dumps(artifact_manifest(), indent=2))
    log = root / "shortened.log"
    # 13 internally consistent cases against the committed case_count of 14.
    write_launch_log(log, [(f"case{i}", i) for i in range(1, 14)])
    expect_clean_failure(
        "shortened-run",
        [
            "scripts/oxide_artifact.py",
            "check-launch",
            "--manifest",
            str(manifest_path),
            "--log",
            str(log),
        ],
        ("manifest case_count=14", "13 PASS lines"),
    )


def guard_duplicate_case(root: Path) -> None:
    manifest = artifact_manifest(
        {"launch.case_count": 3, "launch.elements_total": 3}
    )
    manifest_path = root / "duplicate.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log = root / "duplicate.log"
    write_launch_log(log, [("a", 1), ("a", 1), ("b", 1)])
    expect_clean_failure(
        "duplicate-case",
        [
            "scripts/oxide_artifact.py",
            "check-launch",
            "--manifest",
            str(manifest_path),
            "--log",
            str(log),
        ],
        ("duplicate case (a, n=1)",),
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cuda-oxide-guards-") as temporary:
        root = Path(temporary)
        guard_zero_length_bundle(root)
        guard_huge_log_align(root)
        guard_ordinal_out_of_range()
        guard_artifact_paths(root)
        guard_zero_block(root)
        guard_shortened_run(root)
        guard_duplicate_case(root)

    if FAILURES:
        print(f"GUARD RESULT failed={len(FAILURES)}")
        return 1
    print("GUARD RESULT failed=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
