"""Authenticate Rust capture evidence against an independently pinned plan."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import tarfile
from typing import Any

TEST = "sam3d_objects_full_default_capture"
ARTIFACTS = {"full.safetensors", "result.safetensors", "mesh.obj", "gaussian.ply", "meta.json"}
HELPERS = {"run_rust_capture.py", "capture_identity.py", "provenance.py", "provenance.json"}
RESERVE = 3 * 2**30
POLICY = {"interval_seconds": 0.1, "reserve_bytes": RESERVE, "low_samples": 2,
          "swap_loss_bytes": 256 * 2**20, "child_swap_bytes": 64 * 2**20, "term_grace_seconds": 5}
SOURCE_REQUIRED = {"Cargo.toml", "Cargo.lock", "build.rs", "src/lib.rs",
                   "src/vision/dense/sam3d_objects/pipeline.rs",
                   "src/lib/mlxcel-core/src/lib.rs", "src/lib/mlxcel-core/cpp/mlx_cxx_objects.cpp",
                   "tests/sam3d_objects_real_model.rs", "tests/fixtures/sam3d_objects/provenance.json"}
SOURCE_SUFFIXES = {".rs", ".cpp", ".h", ".hpp", ".c", ".cu", ".metal", ".mm", ".m", ".cmake"}


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


def authenticate_plan(plan_path: Path, expected_plan_sha: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Authenticate the plan, executable, source archive, inputs and checkpoint."""
    plan_path = _path(str(plan_path))
    require(sha(plan_path) == _hash(expected_plan_sha), "Unapproved capture plan")
    plan = read_json(plan_path)
    require(set(plan) == {"schema_version", "case", "run_dir", "executable", "source", "checkpoint_dir", "input_dir", "helpers"}, "Capture plan key set mismatch")
    require(type(plan["schema_version"]) is int and plan["schema_version"] == 1, "Unsupported capture plan")
    require(plan["case"] in ("default", "rgba-dropout"), "Invalid capture case")
    _path(plan["run_dir"], exists=False)
    _bound_file(plan["executable"])
    require(set(plan["helpers"]) == HELPERS, "Capture helper set mismatch")
    for name, row in plan["helpers"].items():
        path = _bound_file(row)
        require(path.name == name, f"Helper filename mismatch: {name}")
    require(sha(Path(__file__).resolve()) == plan["helpers"]["capture_identity.py"]["sha256"], "Running validator is not pinned")
    canonical = read_json(Path(plan["helpers"]["provenance.json"]["path"]))
    manifest = _source(plan)
    require(manifest["tests/fixtures/sam3d_objects/provenance.json"] == plan["helpers"]["provenance.json"]["sha256"], "Compiled canonical provenance differs")
    bundle = _path(plan["checkpoint_dir"])
    inputs = _path(plan["input_dir"])
    names = ["object.png", "mask.png"] if plan["case"] == "default" else ["object-rgba.png"]
    for root, entries in [(bundle, canonical["checkpoint"]["files"]), (inputs, {n: canonical["inputs"][n] for n in names})]:
        for name, entry in entries.items():
            path = _path(str(root / _relative(name)))
            require(path.is_file() and path.stat().st_size == entry["bytes"] and sha(path) == _hash(entry["sha256"]), f"Input/checkpoint authentication failed: {name}")
    return plan, canonical, manifest


def command_environment(plan: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    """Return the sole supported integration command and controlled environment."""
    run = Path(plan["run_dir"])
    command = [plan["executable"]["path"], "--exact", TEST, "--ignored", "--nocapture", "--test-threads=1"]
    environment = {"MLX_ENABLE_TF32": "0", "MLXCEL_DEVICE": "gpu",
                   "MLXCEL_SAM3D_OBJECTS_MODEL": plan["checkpoint_dir"],
                   "MLXCEL_SAM3D_OBJECTS_INPUT_DIR": plan["input_dir"],
                   "MLXCEL_SAM3D_OBJECTS_OUTPUT_DIR": str(run / "output"),
                   "MLXCEL_SAM3D_OBJECTS_SOURCE_MANIFEST": str(run / "frozen" / "source-manifest.json"),
                   "MLXCEL_SAM3D_OBJECTS_CASE": plan["case"]}
    return command, environment


def validate_log(text: str) -> None:
    """Require one actual GPU report and exactly one successful integration test."""
    devices = re.findall(r"Runtime device:\s*([^\r\n]+)", text)
    require(devices == ["GPU"], "Capture must report exactly one actual GPU runtime")
    require(len(re.findall(r"^running 1 test$", text, re.MULTILINE)) == 1, "Capture did not run exactly one test")
    summaries = re.findall(r"^test result: (.+)$", text, re.MULTILINE)
    require(len(summaries) == 1 and re.fullmatch(r"ok\. 1 passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out; finished in [0-9.]+s", summaries[0]) is not None, "Capture test summary is not one successful test")
    require(re.search(r"^test " + TEST + r" \.\.\.", text, re.MULTILINE) is not None, "Capture test identity missing")


def _tensor_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as stream:
        raw = stream.read(8)
        require(len(raw) == 8, "Truncated safetensors header")
        size = struct.unpack("<Q", raw)[0]
        require(2 <= size <= 1024 * 1024, "Invalid tensor header size")
        header = json.loads(stream.read(size), object_pairs_hook=_pairs)
    header.pop("__metadata__", None)
    widths = {"F32": 4, "I32": 4, "U32": 4, "BOOL": 1}
    intervals = []
    for name, row in header.items():
        require(set(row) == {"shape", "dtype", "data_offsets"} and row["dtype"] in widths, f"Invalid captured tensor {name}")
        require(isinstance(row["shape"], list) and all(type(v) is int and v > 0 for v in row["shape"]), f"Empty/invalid tensor shape {name}")
        lo, hi = row["data_offsets"]
        require(type(lo) is int and type(hi) is int and 0 <= lo < hi and hi - lo == math.prod(row["shape"]) * widths[row["dtype"]], f"Invalid tensor offsets {name}")
        intervals.append((lo, hi))
    cursor = 0
    for lo, hi in sorted(intervals):
        require(lo == cursor, "Captured tensor gap/overlap")
        cursor = hi
    require(8 + size + cursor == path.stat().st_size, "Captured tensor payload length mismatch")
    return header, 8 + size


def _outputs(output: Path, receipt: dict[str, Any], conditioned: bool) -> dict[str, str]:
    require(output.resolve(strict=True) == output and output.is_dir(), "Invalid output directory")
    require({p.name for p in output.iterdir()} == ARTIFACTS | {"run.json"}, "Capture output artifact set mismatch")
    require(set(receipt["artifact_sha256"]) == ARTIFACTS, "Receipt artifact set mismatch")
    hashes = {}
    for name in ARTIFACTS | {"run.json"}:
        path = _path(str(output / name))
        require(path.is_file() and path.stat().st_size > 0, f"Missing/empty artifact: {name}")
        hashes[name] = sha(path)
    require(receipt["artifact_sha256"] == {n: hashes[n] for n in ARTIFACTS}, "Artifact hashes differ from actual files")
    full, _ = _tensor_header(output / "full.safetensors")
    compact, start = _tensor_header(output / "result.safetensors")
    core = {"coords", "latents", "pose.rotation", "pose.rotation_matrix", "pose.scale", "pose.translation"}
    geometry = {"mesh.vertices", "mesh.faces", "mesh.vertex_colors", "gaussian.positions", "gaussian.sh_dc", "gaussian.scales", "gaussian.rotations", "gaussian.opacities"}
    require(set(full) == core | geometry and set(compact) == core | {"pointmap_conditioned"}, "Capture tensor keys mismatch")
    require(receipt["outputs"] == {n: {"shape": v["shape"], "dtype": v["dtype"]} for n, v in full.items()}, "Run tensor schema differs from artifact")
    for name in core:
        require({k: compact[name][k] for k in ("shape", "dtype")} == {k: full[name][k] for k in ("shape", "dtype")}, f"Compact/full tensor schema mismatch: {name}")
    state = compact["pointmap_conditioned"]
    require(state["shape"] == [1] and state["dtype"] == "BOOL", "Invalid conditioning tensor")
    with (output / "result.safetensors").open("rb") as stream:
        stream.seek(start + state["data_offsets"][0])
        require(stream.read(1) == bytes([conditioned]), "Conditioning tensor disagrees with capture case")
    meta = read_json(output / "meta.json")
    for key in ("seed", "ss_steps", "slat_steps", "formats"):
        require(meta[key] == receipt["options"][key], f"Metadata options differ: {key}")
    require(meta["family"] == "sam3d_objects" and meta["task"] == "mesh" and meta["pointmap_conditioned"] is conditioned, "Product metadata conditioning/identity differs")
    return hashes


def validate_capture(plan_path: Path | str, expected_plan_sha: str) -> dict[str, Any]:
    """Derive a normalized receipt from authenticated files, never claimed hashes."""
    plan_path = Path(plan_path)
    plan, pin, manifest = authenticate_plan(plan_path, expected_plan_sha)
    run = _path(plan["run_dir"])
    require(not (run / "failure.json").exists(), "Capture retained a failure record")
    frozen = run / "frozen"
    require(frozen.resolve(strict=True) == frozen and {p.name for p in frozen.iterdir()} == HELPERS | {"plan.json", "source-manifest.json"}, "Frozen capture file set mismatch")
    require(sha(_path(str(frozen / "plan.json"))) == expected_plan_sha, "Frozen plan changed")
    require(sha(_path(str(frozen / "source-manifest.json"))) == plan["source"]["manifest"]["sha256"], "Frozen source manifest changed")
    for name in HELPERS:
        require(sha(_path(str(frozen / name))) == plan["helpers"][name]["sha256"], f"Frozen helper changed: {name}")
    for name in ("child.exit", "wrapper.exit"):
        require(_path(str(run / name)).read_text() == "0\n", f"Unsuccessful capture: {name}")
    invocation = read_json(_path(str(run / "invocation.json")))
    command, environment = command_environment(plan)
    require(invocation["command"] == command and invocation["environment"] == environment and invocation["cwd"] == plan["source"]["root"], "Actual invocation differs from pinned command")
    require(invocation["plan_sha256"] == expected_plan_sha and invocation["policy"] == POLICY, "Capture invocation plan/policy differs")
    require(invocation["frozen_sha256"] == {n: plan["helpers"][n]["sha256"] for n in HELPERS}, "Capture invocation helpers differ")
    log = _path(str(run / "child.log"))
    validate_log(log.read_text())
    resources = read_json(_path(str(run / "resources.json")))
    require(type(resources["child_exit"]) is int and type(resources["wrapper_exit"]) is int and resources["child_exit"] == 0 and resources["wrapper_exit"] == 0 and resources["stop_reason"] is None and resources["stopped_for_memory"] is False, "Capture did not finish normally")
    for key, name in [("invocation_sha256", "invocation.json"), ("child_log_sha256", "child.log"), ("memory_sha256", "memory.jsonl")]:
        require(resources[key] == sha(_path(str(run / name))), f"Capture resource correlation differs: {key}")
    require(resources["monitor_sha256"] == plan["helpers"]["run_rust_capture.py"]["sha256"], "Capture monitor differs")
    require(type(resources["elapsed_seconds"]) in (int, float) and math.isfinite(resources["elapsed_seconds"]) and resources["elapsed_seconds"] > 0, "Invalid capture duration")
    start = invocation["start_memory"]
    require(all(type(start.get(k)) is int and start[k] >= 0 for k in ("MemAvailable", "SwapFree", "MemTotal")), "Invalid initial memory telemetry")
    require(start["MemAvailable"] > RESERVE, "Capture started below host reserve")
    low = 0
    lines = (run / "memory.jsonl").read_text().splitlines()
    require(bool(lines), "Capture has no actual memory telemetry")
    previous = -1.0
    for line in lines:
        sample = json.loads(line, object_pairs_hook=_pairs)
        require(all(type(sample.get(k)) is int and sample[k] >= 0 for k in ("MemAvailable", "SwapFree", "MemTotal")), "Invalid memory telemetry")
        require(all(type(sample[k]) is int and sample[k] >= 0 for k in ("VmRSS", "VmSwap", "VmHWM") if k in sample), "Invalid child memory telemetry")
        elapsed = sample["elapsed_seconds"]
        require(type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 <= elapsed and previous <= elapsed <= resources["elapsed_seconds"], "Invalid telemetry chronology")
        previous = elapsed
        low = low + 1 if sample["MemAvailable"] < RESERVE else 0
        require(low < 2 and start["SwapFree"] - sample["SwapFree"] <= POLICY["swap_loss_bytes"] and sample.get("VmSwap", 0) <= POLICY["child_swap_bytes"], "Successful capture violated containment policy")
    receipt = read_json(_path(str(run / "output" / "run.json")))
    options = dict(pin["default_options"])
    options["estimate_depth"] = plan["case"] == "default"
    names = ["object.png", "mask.png"] if plan["case"] == "default" else ["object-rgba.png"]
    require(receipt["case"] == plan["case"] and json.dumps(receipt["options"], sort_keys=True) == json.dumps(options, sort_keys=True), "Run options differ from canonical case")
    require(receipt["runtime_device"] == "GPU" and receipt["tf32"] == "0", "Run runtime identity differs from GPU log")
    require(receipt["executable"] == command[0] and receipt["executable_sha256"] == plan["executable"]["sha256"], "Run executable differs")
    require(receipt["source_manifest_sha256"] == plan["source"]["manifest"]["sha256"] and receipt["source_files"] == manifest, "Run compiled source differs")
    require(receipt["upstream_source_revision"] == pin["source_revision"] and receipt["checkpoint"] == pin["checkpoint"], "Run upstream/checkpoint identity differs")
    require(receipt["inputs"] == {n: pin["inputs"][n]["sha256"] for n in names}, "Run input identity differs")
    require(receipt["pointmap_conditioned"] is options["estimate_depth"], "Run conditioning differs")
    hashes = _outputs(run / "output", receipt, options["estimate_depth"])
    return {"producer": "mlxcel", "runtime_device": "GPU", "tf32": "0", "case": plan["case"],
            "options": options, "input_files": {n: pin["inputs"][n] for n in names}, "checkpoint": pin["checkpoint"],
            "pointmap_conditioned": receipt["pointmap_conditioned"], "output_sha256": hashes["full.safetensors"],
            "artifact_sha256": {n: hashes[n] for n in ARTIFACTS}, "run_json_sha256": hashes["run.json"],
            "binary_sha256": plan["executable"]["sha256"], "source_archive_sha256": plan["source"]["archive"]["sha256"],
            "source_revision": plan["source"]["revision"], "source_manifest_sha256": plan["source"]["manifest"]["sha256"],
            "source_files": manifest, "upstream_source_revision": pin["source_revision"],
            "plan_sha256": expected_plan_sha, "invocation_sha256": sha(run / "invocation.json"),
            "log_sha256": sha(log), "resources_sha256": sha(run / "resources.json"),
            "memory_sha256": sha(run / "memory.jsonl"), "helper_sha256": invocation["frozen_sha256"],
            "child_exit": 0, "wrapper_exit": 0, "executable": command[0], "run_dir": str(run)}
