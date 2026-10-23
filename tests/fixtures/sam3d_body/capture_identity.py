# Copyright 2025-2026 Lablup Inc.
# SPDX-License-Identifier: Apache-2.0
"""Authenticate Rust capture evidence against an independently pinned plan."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import struct
import stat
import tarfile
from typing import Any

TEST = "sam3d_body_real_model_capture"
ARTIFACTS = {"result.safetensors", "mesh.obj", "meta.json", "instances.json"}
INSTANCE_SIDECAR = "instances.json"
HELPERS = {"run_rust_capture.py", "capture_identity.py", "provenance.json", "compare_reference.py", "reference-manifest.json"}
RESERVE = 3 * 2**30
POLICY = {"interval_seconds": 0.1, "reserve_bytes": RESERVE, "low_samples": 2,
          "swap_loss_bytes": 256 * 2**20, "swap_pressure_available_bytes": 8 * 2**30, "child_swap_bytes": 64 * 2**20, "term_grace_seconds": 5, "framebuffer_reserve_bytes": 4 * 2**30}
SOURCE_REQUIRED = {"Cargo.toml", "Cargo.lock", "build.rs", "src/lib.rs",
                   "src/vision/dense/sam3d_body/model.rs",
                   "src/lib/mlxcel-core/src/lib.rs", "src/lib/mlxcel-core/cpp/mlx_cxx_objects.cpp",
                   "tests/sam3d_body_real_model.rs", "tests/fixtures/sam3d_body/provenance.json"}
SOURCE_SUFFIXES = {".rs", ".cpp", ".h", ".hpp", ".c", ".cu", ".metal", ".mm", ".m", ".cmake", ".hip", ".cuh", ".inc"}


def require(condition: bool, message: str) -> None:
    """Reject evidence that does not satisfy the pinned capture contract."""
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    """Hash actual file contents using bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> Any:
    """Read JSON without accepting duplicate keys or non-finite numbers."""
    def bad_number(value: str) -> None:
        raise ValueError(f"Non-finite JSON number: {value}")
    return json.loads(path.read_text(), object_pairs_hook=_pairs, parse_constant=bad_number)


def _hash(value: Any) -> str:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "Invalid SHA256")
    return value


def _path(value: Any, *, exists: bool = True) -> Path:
    require(isinstance(value, str), "Path must be text")
    path = Path(value)
    require(path.is_absolute() and path.resolve(strict=exists) == path, f"Noncanonical or symlink path: {value}")
    return path


def _relative(name: str) -> str:
    p = PurePosixPath(name)
    require(bool(p.parts) and not p.is_absolute() and ".." not in p.parts and str(p) == name and "\\" not in name,
            f"Unsafe relative file name: {name}")
    return name


def _bound_file(row: dict[str, Any]) -> Path:
    require(isinstance(row, dict) and set(row) == {"path", "sha256"}, "Invalid file binding")
    path = _path(row["path"])
    require(path.is_file() and sha(path) == _hash(row["sha256"]), f"Pinned file changed: {path}")
    return path



def _bound_executable(row: dict[str, Any]) -> Path:
    """Hash and inspect one regular ELF64 little-endian x86_64 file descriptor."""
    require(isinstance(row, dict) and set(row) == {"path", "sha256"}, "Invalid executable binding")
    path = _path(row["path"])
    expected = _hash(row["sha256"])
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_mode & 0o111 != 0, "Pinned executable must be a regular executable file")
        header = stream.read(64)
        require(len(header) == 64 and header[:7] == b"\x7fELF\x02\x01\x01", "Pinned executable must be ELF64 little-endian version 1")
        kind, machine, version = struct.unpack_from('<HHI', header, 16)
        require(kind in (2, 3) and machine == 62 and version == 1 and struct.unpack_from('<H', header, 52)[0] == 64,
                "Pinned executable must be an x86_64 ELF executable or PIE")
        digest = hashlib.sha256(header)
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(stream.fileno())
        def identity(value: os.stat_result) -> tuple[int, ...]:
            return (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        require(identity(before) == identity(after) == identity(path.stat(follow_symlinks=False)), "Executable changed during authentication")
        require(digest.hexdigest() == expected, "Pinned executable bytes changed")
    return path


def _source(plan: dict[str, Any]) -> dict[str, str]:
    source = plan["source"]
    require(set(source) == {"revision", "root", "archive", "manifest"}, "Invalid source binding")
    require(isinstance(source["revision"], str) and re.fullmatch(r"[0-9a-f]{40}", source["revision"]) is not None,
            "Invalid Rust source revision")
    root = _path(source["root"])
    manifest_path = _bound_file(source["manifest"])
    manifest = read_json(manifest_path)
    require(isinstance(manifest, dict) and SOURCE_REQUIRED <= set(manifest) and "SOURCE_REVISION" not in manifest, "Incomplete compiled-source manifest")
    required = set(SOURCE_REQUIRED)
    for directory in (root / "src", root / "build"):
        if directory.exists():
            required.update(str(p.relative_to(root)) for p in directory.rglob("*")
                            if p.is_file() and (p.suffix in SOURCE_SUFFIXES or p.name in {"Cargo.toml", "CMakeLists.txt"}))
    require(required <= set(manifest), "Compiled-source manifest omits current implementation files")
    for name, digest in manifest.items():
        path = root / _relative(name)
        require(path.resolve(strict=True) == path and path.is_file(), f"Invalid source path: {name}")
        require(sha(path) == _hash(digest), f"Source differs from compiled manifest: {name}")
    archive = _bound_file(source["archive"])
    seen: set[str] = set()
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            name = _relative(member.name)
            require(name not in seen and member.isfile(), f"Duplicate or nonregular archive member: {name}")
            seen.add(name)
            require(name in manifest or name == "SOURCE_REVISION", f"Unmanifested archive member: {name}")
            stream = tar.extractfile(member)
            require(stream is not None, f"Unreadable archive member: {name}")
            with stream:
                if name == "SOURCE_REVISION":
                    require(member.size == 41 and stream.read() == (source["revision"] + "\n").encode(), "Archive revision mismatch")
                else:
                    digest = hashlib.sha256()
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                    require(digest.hexdigest() == manifest[name], f"Archived source differs: {name}")
    require(seen == set(manifest) | {"SOURCE_REVISION"}, "Archive does not cover exact compiled manifest")
    return manifest


def authenticate_plan(plan_path: Path | str, expected_sha: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Return (plan, canonical provenance, compiled-source manifest), authenticated."""
    plan_path = _path(str(plan_path))
    require(sha(plan_path) == _hash(expected_sha), "Unapproved capture plan")
    plan = read_json(plan_path)
    require(set(plan) == {"schema_version", "case", "run_dir", "executable", "source", "checkpoint_dir", "image", "helpers"}, "Capture plan key set mismatch")
    require(type(plan["schema_version"]) is int and plan["schema_version"] == 1, "Unsupported capture plan")
    require(plan["case"] in ("boxed", "default"), "Invalid capture case")
    _path(plan["run_dir"], exists=False)
    _bound_executable(plan["executable"])
    require(set(plan["helpers"]) == HELPERS, "Capture helper set mismatch")
    for name, row in plan["helpers"].items():
        require(_bound_file(row).name == name, f"Helper filename mismatch: {name}")
    require(sha(Path(__file__).resolve()) == plan["helpers"]["capture_identity.py"]["sha256"], "Running validator is not pinned")
    canonical = read_json(Path(plan["helpers"]["provenance.json"]["path"]))
    manifest = _source(plan)
    require(manifest["tests/fixtures/sam3d_body/provenance.json"] == plan["helpers"]["provenance.json"]["sha256"], "Compiled canonical provenance differs")
    bundle = _path(plan["checkpoint_dir"])
    for name, row in canonical["checkpoint"]["files"].items():
        path = _path(str(bundle / _relative(name)))
        require(path.is_file() and path.stat().st_size == row["bytes"] and sha(path) == _hash(row["sha256"]), f"Checkpoint authentication failed: {name}")
    require({p.name for p in bundle.glob('*.safetensors')} == {n for n in canonical['checkpoint']['files'] if n.endswith('.safetensors')}, "Unpinned checkpoint weight files")
    image = _path(plan["image"])
    require(image.is_file() and image.name == canonical["input"]["canonical_image"] and sha(image) == _hash(canonical["input"]["canonical_sha256"]), "Canonical image differs")
    return plan, canonical, manifest


def options_for_case(plan: dict[str, Any]) -> dict[str, Any]:
    """Map canonical omitted Python options to the Rust public options object."""
    pin = read_json(Path(plan["helpers"]["provenance.json"]["path"]))
    return {k: v for k, v in pin["cases"][plan["case"]].items() if v is not None}


def command_environment(plan: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    """Return the exact one-test invocation and controlled runtime environment."""
    command = [plan["executable"]["path"], "--exact", TEST, "--ignored", "--nocapture", "--test-threads=1"]
    environment = {"MLX_ENABLE_TF32": "0", "MLXCEL_DEVICE": "gpu",
                   "BODY_MODEL_DIR": plan["checkpoint_dir"], "BODY_IMAGE": plan["image"],
                   "BODY_CAPTURE_DIR": str(Path(plan["run_dir"]) / "output"),
                   "BODY_OPTIONS_JSON": json.dumps(options_for_case(plan), sort_keys=True, separators=(",", ":"))}
    return command, environment


def validate_log(text: str) -> None:
    """Require an actual resolved GPU and exactly one successful named test."""
    require(re.findall(r"Runtime device:\s*([^\r\n]+)", text) == ["GPU"], "Capture must report exactly one actual GPU runtime")
    require(len(re.findall(r"^running 1 test$", text, re.MULTILINE)) == 1, "Capture did not run exactly one test")
    summaries = re.findall(r"^test result: (.+)$", text, re.MULTILINE)
    require(len(summaries) == 1 and re.fullmatch(r"ok\. 1 passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out; finished in [0-9.]+s", summaries[0]) is not None, "Capture test summary differs")
    require(re.search(r"^test " + TEST + r" \.\.\.", text, re.MULTILINE) is not None, "Capture test identity missing")


def _tensor_schema(path: Path) -> dict[str, Any]:
    expected = {"vertices": [18439, 3], "keypoints_3d": [70, 3], "joint_coords": [127, 3], "camera": [3]}
    with path.open("rb") as stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, "Truncated tensor header")
        size = struct.unpack("<Q", prefix)[0]
        require(2 <= size <= 1024 * 1024, "Invalid tensor header size")
        header = json.loads(stream.read(size), object_pairs_hook=_pairs)
        header.pop("__metadata__", None)
        require(set(header) == set(expected), "Body must expose exactly four tensors")
        intervals = []
        for name, dims in expected.items():
            row = header[name]
            require(set(row) == {"shape", "dtype", "data_offsets"} and row["dtype"] == "F32" and row["shape"] == dims and all(type(v) is int for v in row["shape"]), f"Invalid output tensor {name}")
            lo, hi = row["data_offsets"]
            require(type(lo) is int and type(hi) is int and 0 <= lo < hi and hi - lo == math.prod(dims) * 4, f"Invalid offsets {name}")
            intervals.append((lo, hi))
        cursor = 0
        for lo, hi in sorted(intervals):
            require(lo == cursor, "Tensor payload gap/overlap")
            data = stream.read(hi - lo)
            require(len(data) == hi - lo and all(math.isfinite(v[0]) for v in struct.iter_unpack('<f', data)), "Truncated/nonfinite tensor payload")
            cursor = hi
        require(stream.read(1) == b"", "Trailing tensor bytes")
    return {k: {"shape": v, "dtype": "F32"} for k, v in expected.items()}


def _outputs(output: Path, receipt: dict[str, Any], pin: dict[str, Any], case: str) -> tuple[dict[str, str], dict[str, Any]]:
    require(output.resolve(strict=True) == output and output.is_dir(), "Invalid output directory")
    require({p.name for p in output.iterdir()} == ARTIFACTS | {"run.json"}, "Capture artifact set differs")
    hashes = {}
    for name in ARTIFACTS | {"run.json"}:
        path = _path(str(output / name))
        require(path.is_file() and path.stat().st_size > 0, f"Missing artifact {name}")
        hashes[name] = sha(path)
    require(receipt["artifacts"] == {n: {"sha256": hashes[n], "bytes": (output / n).stat().st_size} for n in ARTIFACTS}, "Run artifact hashes/sizes differ")
    require((output / "instances.json").read_bytes() == b"[]", "Body instances must be canonical empty array")
    schema = _tensor_schema(output / "result.safetensors")
    vertices = faces = 0
    with (output / "mesh.obj").open() as stream:
        for line in stream:
            fields = line.split()
            require(len(fields) == 4 and fields[0] in ('v', 'f'), "Unexpected OBJ row")
            if fields[0] == 'v':
                require(all(math.isfinite(float(v)) for v in fields[1:]), "Nonfinite OBJ vertex")
                vertices += 1
            else:
                require(all(re.fullmatch(r'[1-9][0-9]*', v) and int(v) <= 18439 for v in fields[1:]), "Invalid OBJ face")
                faces += 1
    require((vertices, faces) == (18439, 36874), "Body OBJ topology size differs")
    meta = read_json(output / "meta.json")
    require(meta["family"] == "sam3d_body" and meta["task"] == "body_mesh", "Output identity differs")
    h, w, _ = pin["input"]["shape"]
    require(meta["input_size"] == {"width": w, "height": h}, "Input dimensions differ")
    bbox = pin["cases"][case]["bbox"] or [0, 0, w, h]
    require(meta["bbox"] == bbox and meta["image_size"] == pin["released_image_size"], "Capture box/crop differs")
    camera = [v for row in pin["input"]["default_cam_int"] for v in row]
    actual_camera = meta["cam_int"]
    require(isinstance(actual_camera, list) and len(actual_camera) == 9 and
            all(type(v) in (int, float) and math.isfinite(v) for v in actual_camera), "Invalid camera metadata")
    # serde emits shortest decimal F32 strings; compare the represented F32 bits,
    # not Python's F64 parse of those strings against expanded F32 decimals.
    require(struct.pack('<9f', *actual_camera) == struct.pack('<9f', *camera), "Capture camera intrinsics differ")
    require(all(meta.get(k) == v for k, v in receipt["metadata"].items()), "Receipt metadata differs from product")
    return hashes, schema


def _telemetry(start: dict[str, Any], sample: dict[str, Any], low: int) -> int:
    fields = ("MemAvailable", "SwapFree", "MemTotal", "framebuffer_total", "framebuffer_used", "framebuffer_free")
    require(all(type(sample.get(k)) is int and sample[k] >= 0 for k in fields), "Invalid memory telemetry")
    require(sample['framebuffer_total'] >= 8 * 2**30 and sample['framebuffer_free'] + sample['framebuffer_used'] == sample['framebuffer_total'], "Invalid framebuffer accounting")
    require(all(type(sample[k]) is int and sample[k] >= 0 for k in ("VmRSS", "VmSwap", "VmHWM") if k in sample), "Invalid child telemetry")
    low = low + 1 if sample["MemAvailable"] < RESERVE else 0
    require(low < POLICY["low_samples"] and sample["framebuffer_free"] >= POLICY["framebuffer_reserve_bytes"], "Capture violated memory reserve")
    require(start["SwapFree"] - sample["SwapFree"] <= POLICY["swap_loss_bytes"] or sample["MemAvailable"] >= POLICY["swap_pressure_available_bytes"], "Capture violated conditional swap reserve")
    require(sample.get("VmSwap", 0) <= POLICY["child_swap_bytes"], "Capture child swapped")
    return low


def validate_capture(plan_path: Path | str, expected_sha: str) -> dict[str, Any]:
    """Return normalized identity derived from actual files, exits and telemetry."""
    plan, pin, manifest = authenticate_plan(plan_path, expected_sha)
    run = _path(plan["run_dir"])
    require(not (run / "failure.json").exists(), "Capture retained a failure record")
    frozen = _path(str(run / "frozen"))
    require({p.name for p in frozen.iterdir()} == HELPERS | {"plan.json", "source-manifest.json"}, "Frozen file set differs")
    require(sha(_path(str(frozen / "plan.json"))) == expected_sha, "Frozen plan changed")
    require(sha(_path(str(frozen / "source-manifest.json"))) == plan['source']['manifest']['sha256'], "Frozen source manifest changed")
    for name in HELPERS:
        require(sha(_path(str(frozen / name))) == plan['helpers'][name]['sha256'], f"Frozen helper changed: {name}")
    for name in ('child.exit', 'wrapper.exit'):
        require(_path(str(run / name)).read_text() == '0\n', f"Unsuccessful capture: {name}")
    invocation = read_json(_path(str(run / 'invocation.json')))
    command, environment = command_environment(plan)
    require(invocation['command'] == command and invocation['environment'] == environment and invocation['cwd'] == plan['source']['root'], "Capture invocation differs")
    require(invocation['plan_sha256'] == expected_sha and invocation['policy'] == POLICY, "Capture plan/policy differs")
    require(invocation['frozen_sha256'] == {n: plan['helpers'][n]['sha256'] for n in HELPERS}, "Invocation helpers differ")
    log = _path(str(run / 'child.log'))
    validate_log(log.read_text())
    resources = read_json(_path(str(run / 'resources.json')))
    require(type(resources['child_exit']) is int and type(resources['wrapper_exit']) is int and resources['child_exit'] == resources['wrapper_exit'] == 0 and resources['stop_reason'] is None and resources['stopped_for_memory'] is False, "Capture stopped or failed")
    for key, filename in [('invocation_sha256','invocation.json'),('child_log_sha256','child.log'),('memory_sha256','memory.jsonl')]:
        require(resources[key] == sha(_path(str(run / filename))), f"Capture telemetry binding differs: {key}")
    require(resources['monitor_sha256'] == plan['helpers']['run_rust_capture.py']['sha256'], "Monitor differs")
    duration = resources['elapsed_seconds']
    require(type(duration) in (int,float) and math.isfinite(duration) and duration > 0, "Invalid telemetry duration")
    start = invocation['start_memory']
    _telemetry(start, start, 0)
    require(start['MemAvailable'] > RESERVE, "Capture started below host reserve")
    low = 0; previous = -1.; count = 0
    with _path(str(run / 'memory.jsonl')).open() as stream:
        for line in stream:
            sample = json.loads(line, object_pairs_hook=_pairs)
            elapsed = sample['elapsed_seconds']
            require(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0 <= elapsed and previous <= elapsed <= duration, "Invalid telemetry chronology")
            previous = elapsed; count += 1
            low = _telemetry(start, sample, low)
    require(count > 0, "No actual memory telemetry")
    receipt = read_json(_path(str(run / 'output' / 'run.json')))
    require(receipt['schema'] == 1 and receipt['family'] == 'sam3d_body' and receipt['execution_api'] == 'DensePredictor.predict', "Capture API identity differs")
    require(receipt['runtime_device'] == 'GPU' and receipt['tf32'] == '0', "Run runtime differs from log")
    require(receipt['options'] == options_for_case(plan), "Capture options differ")
    require(receipt['model'] == plan['checkpoint_dir'] and receipt['image'] == plan['image'] and receipt['image_sha256'] == pin['input']['canonical_sha256'], "Capture input/checkpoint differs")
    require(receipt['executable'] == command[0] and receipt['executable_sha256'] == plan['executable']['sha256'], "Capture executable differs")
    hashes, schema = _outputs(run / 'output', receipt, pin, plan['case'])
    return {'producer':'mlxcel','family':'sam3d_body','runtime_device':'GPU','tf32':'0','case':plan['case'],
            'options':options_for_case(plan),'checkpoint':pin['checkpoint'],'input':pin['input'],
            'output_dir':str(run/'output'),'outputs':schema,'artifact_sha256':hashes,
            'output_sha256':hashes['result.safetensors'],'run_json_sha256':hashes['run.json'],
            'binary_sha256':plan['executable']['sha256'],'source_revision':plan['source']['revision'],
            'source_archive_sha256':plan['source']['archive']['sha256'],'source_manifest_sha256':plan['source']['manifest']['sha256'],
            'source_files':manifest,'upstream_source_revision':pin['source_revision'],'plan_sha256':expected_sha,
            'invocation_sha256':sha(run/'invocation.json'),'log_sha256':sha(log),'resources_sha256':sha(run/'resources.json'),
            'memory_sha256':sha(run/'memory.jsonl'),'helper_sha256':invocation['frozen_sha256'],
            'child_exit':0,'wrapper_exit':0,'executable':command[0],'run_dir':str(run)}
