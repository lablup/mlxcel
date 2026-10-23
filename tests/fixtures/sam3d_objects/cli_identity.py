"""Authenticate an actual default CLI capture and compare every exported value.

CLI exports omit three mesh color channels and transform Gaussian scale/opacity;
those raw representations remain covered by the separate integration capture.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import Any

HELPERS = {"run_cli_capture.py", "cli_identity.py", "capture_identity.py", "run_rust_capture.py",
           "compare_reference.py", "provenance.py", "provenance.json", "comparison-contract.json"}
ARTIFACTS = {"result.safetensors", "mesh.obj", "gaussian.ply", "meta.json"}
INSTANCE_SIDECAR = "instances.json"
OUTPUT_FILES = ARTIFACTS | {INSTANCE_SIDECAR}
FIELDS = ("x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
          "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")
POSE = {"pose.rotation": (4,), "pose.rotation_matrix": (3, 3), "pose.scale": (3,), "pose.translation": (3,)}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_module(path: Path, name: str, expected_sha: str) -> ModuleType:
    """Execute helper bytes only after the caller authenticates their hashes."""
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None, "Cannot load authenticated helper")
    module = importlib.util.module_from_spec(spec)
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == expected_sha, "Helper changed before import")
    exec(compile(payload, str(path), "exec"), module.__dict__)
    return module


def authenticate_helpers(plan_path: Path, expected_sha: str) -> tuple[dict[str, Any], dict[str, ModuleType]]:
    """Authenticate every local dependency before importing executable helpers."""
    require(re.fullmatch(r"[0-9a-f]{64}", expected_sha) is not None and sha(plan_path) == expected_sha, "Unapproved CLI plan")
    # This bootstrap identifies files only; the pinned strict parser runs next.
    plan = json.loads(plan_path.read_text())
    require(set(plan["helpers"]) == HELPERS, "CLI helper set mismatch")
    for name, row in plan["helpers"].items():
        path = Path(row["path"])
        require(set(row) == {"path", "sha256"} and path.is_absolute() and path.resolve(strict=True) == path and path.name == name, "Invalid helper binding")
        require(sha(path) == row["sha256"], f"Helper changed: {name}")
    require(sha(Path(__file__).resolve()) == plan["helpers"]["cli_identity.py"]["sha256"], "Running CLI validator differs")
    modules = {}
    for name in ("capture_identity", "run_rust_capture", "provenance"):
        modules[name] = load_module(Path(__file__).resolve().parent / (name + ".py"), "cli_pinned_" + name, plan["helpers"][name + ".py"]["sha256"])
    old = sys.modules.get("provenance")
    try:
        sys.modules["provenance"] = modules["provenance"]
        modules["compare_reference"] = load_module(Path(__file__).resolve().parent / "compare_reference.py", "cli_pinned_compare_reference", plan["helpers"]["compare_reference.py"]["sha256"])
    finally:
        if old is None:
            sys.modules.pop("provenance", None)
        else:
            sys.modules["provenance"] = old
    strict = modules["capture_identity"]
    plan = strict.read_json(plan_path)
    for row in plan["helpers"].values():
        strict._bound_file(row)
    return plan, modules


def authenticate_plan(plan_path: Path, expected_sha: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Rehash source, binary, canonical inputs/checkpoint and authentic reference."""
    plan, modules = authenticate_helpers(plan_path, expected_sha)
    c = modules["capture_identity"]
    require(set(plan) == {"schema_version", "case", "run_dir", "executable", "source", "checkpoint_dir", "input_dir", "helpers", "reference"}, "CLI plan schema mismatch")
    require(type(plan["schema_version"]) is int and plan["schema_version"] == 1 and plan["case"] == "default", "Only default CLI capture is supported")
    c._path(str(plan_path)); c._path(plan["run_dir"], exists=False)
    c._bound_file(plan["executable"])
    manifest = c._source(plan)
    canonical = c.read_json(Path(plan["helpers"]["provenance.json"]["path"]))
    require(manifest["tests/fixtures/sam3d_objects/provenance.json"] == plan["helpers"]["provenance.json"]["sha256"], "Compiled canonical identity differs")
    expected = {"seed": 42, "ss_steps": 25, "slat_steps": 25, "formats": ["gaussian", "mesh"], "estimate_depth": True}
    require(json.dumps(canonical["default_options"], sort_keys=True) == json.dumps(expected, sort_keys=True), "Canonical defaults differ")
    for root, entries in [(c._path(plan["checkpoint_dir"]), canonical["checkpoint"]["files"]),
                          (c._path(plan["input_dir"]), {n: canonical["inputs"][n] for n in ("object.png", "mask.png")})]:
        for name, row in entries.items():
            path = c._path(str(root / c._relative(name)))
            require(path.is_file() and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], f"Canonical asset changed: {name}")
    reference = c._bound_file(plan["reference"])
    require(reference.name == "reference.json", "Reference receipt filename differs")
    ref = c.read_json(reference)
    for key in ("source_revision", "source_sha256", "checkpoint"):
        require(ref[key] == canonical[key], f"Reference {key} mismatch")
    for key in ("version", "native_sha256", "binding_sha256"):
        require(ref["runtime"][key] == canonical["runtime"][key], f"Reference runtime {key} mismatch")
    require(ref["case"] == "default" and ref["device"] == "Device(gpu, 0)" and ref["tf32"] == "0" and ref["pointmap_conditioned"] is True, "Reference execution differs")
    require(json.dumps(ref["options"], sort_keys=True) == json.dumps(expected, sort_keys=True), "Reference options differ")
    require(ref["input_files"] == {n: canonical["inputs"][n] for n in ("object.png", "mask.png")}, "Reference inputs differ")
    require(set(ref["artifact_sha256"]) == {"reference.safetensors", "mesh.obj", "gaussian.ply", "pointmap.safetensors"}, "Reference artifact set differs")
    require(ref["artifact_sha256"]["reference.safetensors"] == ref["output_sha256"], "Reference tensor identity differs")
    for name, digest in ref["artifact_sha256"].items():
        require(sha(c._path(str(reference.parent / name))) == c._hash(digest), f"Reference artifact changed: {name}")
    contract = c.read_json(Path(plan["helpers"]["comparison-contract.json"]["path"]))
    require(contract["schema_version"] == 1 and contract["latents"]["relative_rms_strict_upper_bound"] == .01 and contract["mesh"]["vertex_count_relative_error_maximum"] == .01 and contract["mesh"]["normalized_chamfer_strict_upper_bound"] == .001, "Unsupported comparison contract")
    return plan, {"canonical": canonical, "manifest": manifest, "reference": ref, "contract": contract}, modules


def command_environment(plan: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    """Construct the literal no-options CLI command, never a test or surrogate."""
    inputs, output = Path(plan["input_dir"]), Path(plan["run_dir"]) / "output"
    return ([plan["executable"]["path"], "predict", "--model", plan["checkpoint_dir"],
             "--image", str(inputs / "object.png"), "--mask", str(inputs / "mask.png"), "--output", str(output)],
            {"MLX_ENABLE_TF32": "0", "MLXCEL_DEVICE": "gpu"})


def validate_log(text: str, output: Path) -> None:
    require(re.findall(r"^Runtime device: ([^\r\n]+)$", text, re.MULTILINE) == ["GPU"], "CLI must report exactly one actual GPU runtime")
    require(re.findall(r"Runtime device:", text) == ["Runtime device:"], "Duplicate runtime report")
    saved = re.findall(r"^saved (.+)$", text, re.MULTILINE)
    require(len(saved) == len(OUTPUT_FILES) and set(saved) == {str(output / n) for n in OUTPUT_FILES}, "CLI saved artifact reports differ")


def empty_instances_sha(path: Path) -> str:
    """Authenticate the dense writer's canonical empty instance sidecar."""
    require(path.resolve(strict=True) == path, "Noncanonical or symlink instances sidecar")
    require(path.is_file() and path.stat().st_size == 2 and path.read_bytes() == b"[]",
            "Instances sidecar must be the canonical empty array")
    return sha(path)


def read_obj(path: Path) -> tuple[Any, Any]:
    """Read every XYZ/RGB vertex and triangle without sampling or filtering."""
    import numpy as np
    vertices, faces = [], []
    in_faces = False
    with path.open() as stream:
        for line in stream:
            row = line.split()
            require(bool(row), "Empty OBJ row")
            if row[0] == "v":
                require(not in_faces and len(row) == 7, "Invalid OBJ vertex schema/order")
                vertices.append([float(v) for v in row[1:]])
            else:
                require(row[0] == "f" and len(row) == 4 and all(re.fullmatch(r"[1-9][0-9]*", v) for v in row[1:]), "Invalid OBJ triangle")
                in_faces = True
                faces.append([int(v) - 1 for v in row[1:]])
    require(bool(vertices) and bool(faces), "Empty OBJ geometry")
    v, f = np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)
    require(bool(np.isfinite(v).all()) and bool((np.abs(v) <= np.finfo(np.float32).max).all()) and bool(((f >= 0) & (f < len(v))).all()), "Nonfinite OBJ or invalid index")
    return v, f


def read_ply(path: Path) -> Any:
    """Validate the exact binary export and map every stored Gaussian field."""
    import numpy as np
    with path.open("rb") as stream:
        require(stream.readline(128) == b"ply\n" and stream.readline(128) == b"format binary_little_endian 1.0\n", "Invalid Gaussian PLY format")
        count = re.fullmatch(rb"element vertex ([1-9][0-9]*)\n", stream.readline(128))
        require(count is not None, "Invalid Gaussian vertex count")
        count = int(count[1])
        for field in FIELDS:
            require(stream.readline(128) == f"property float {field}\n".encode(), "Gaussian property schema differs")
        require(stream.readline(128) == b"end_header\n", "Unexpected Gaussian header field")
        start = stream.tell()
    require(path.stat().st_size == start + count * 17 * 4, "Gaussian payload size mismatch")
    values = np.memmap(path, dtype="<f4", mode="r", offset=start, shape=(count, 17))
    require(bool(np.isfinite(values).all()) and bool((values[:, 3:6] == 0).all()), "Nonfinite Gaussian fields or nonzero exported normals")
    return values


def validate_compact(values: dict[str, Any]) -> None:
    import numpy as np
    require(set(values) == set(POSE) | {"coords", "latents", "pointmap_conditioned"}, "Compact key set differs")
    coords = values["coords"]
    require(coords.dtype == np.dtype("int32") and coords.ndim == 2 and coords.shape[1] == 4 and len(coords) > 0, "Invalid compact coordinates")
    require(bool((coords[:, 0] == 0).all()) and bool(((coords[:, 1:] >= 0) & (coords[:, 1:] < 64)).all()), "Compact coordinate range differs")
    for name, shape in {**POSE, "latents": (len(coords), 8)}.items():
        require(values[name].dtype == np.dtype("float32") and values[name].shape == shape and bool(np.isfinite(values[name]).all()), f"Invalid compact {name}")
    state = values["pointmap_conditioned"]
    require(state.dtype == np.dtype("bool") and state.shape == (1,) and bool(state[0]), "Default CLI must be pointmap-conditioned")
    require(bool((values["pose.scale"] > 0).all()), "Invalid pose scale")


def _resources(run: Path, plan: dict[str, Any], expected_sha: str, c: ModuleType) -> dict[str, Any]:
    require(not (run / "failure.json").exists(), "CLI retained failure evidence")
    frozen = c._path(str(run / "frozen"))
    require({p.name for p in frozen.iterdir()} == HELPERS | {"plan.json", "source-manifest.json", "reference.json"}, "Frozen CLI file set differs")
    for name, digest in {**{n: plan["helpers"][n]["sha256"] for n in HELPERS}, "plan.json": expected_sha,
                         "source-manifest.json": plan["source"]["manifest"]["sha256"], "reference.json": plan["reference"]["sha256"]}.items():
        require(sha(c._path(str(frozen / name))) == digest, f"Frozen CLI file changed: {name}")
    for name in ("child.exit", "wrapper.exit"):
        require(c._path(str(run / name)).read_text() == "0\n", f"Unsuccessful CLI {name}")
    invocation = c.read_json(c._path(str(run / "invocation.json")))
    command, environment = command_environment(plan)
    require(invocation["command"] == command and invocation["environment"] == environment and invocation["cwd"] == plan["source"]["root"], "CLI invocation differs")
    require(invocation["plan_sha256"] == expected_sha and invocation["policy"] == c.POLICY, "CLI plan/policy differs")
    require(invocation["frozen_sha256"] == {n: plan["helpers"][n]["sha256"] for n in HELPERS}, "Invocation helper set differs")
    executed = c.read_json(c._path(str(run / "executed.json")))
    require(type(executed["pid"]) is int and executed["pid"] > 0 and c._path(str(run / "child.pid")).read_text() == str(executed["pid"]) + "\n", "Child PID differs")
    require(executed["executable"] == command[0] and executed["executable_sha256"] == plan["executable"]["sha256"] and executed["environment"] == environment, "Actual child executable/environment differs")
    validate_log(c._path(str(run / "child.log")).read_text(), run / "output")
    resource = c.read_json(c._path(str(run / "resources.json")))
    require(type(resource["child_exit"]) is int and type(resource["wrapper_exit"]) is int and resource["child_exit"] == resource["wrapper_exit"] == 0 and resource["stopped_for_memory"] is False and resource["stop_reason"] is None, "CLI did not finish normally")
    for key, name in [("invocation_sha256", "invocation.json"), ("executed_sha256", "executed.json"), ("child_log_sha256", "child.log"), ("memory_sha256", "memory.jsonl")]:
        require(resource[key] == sha(c._path(str(run / name))), f"CLI resource correlation differs: {key}")
    require(resource["monitor_sha256"] == plan["helpers"]["run_rust_capture.py"]["sha256"], "CLI monitor differs")
    duration, start = resource["elapsed_seconds"], invocation["start_memory"]
    require(type(duration) in (int, float) and math.isfinite(duration) and duration > 0, "Invalid CLI duration")
    require(all(type(start.get(k)) is int and start[k] >= 0 for k in ("MemAvailable", "SwapFree", "MemTotal")) and start["MemAvailable"] > c.RESERVE, "Invalid initial memory")
    lines = c._path(str(run / "memory.jsonl")).read_text().splitlines()
    require(bool(lines), "Missing CLI memory telemetry")
    previous, low = 0.0, 0
    for line in lines:
        row = json.loads(line, object_pairs_hook=c._pairs)
        require(all(type(row.get(k)) is int and row[k] >= 0 for k in ("MemAvailable", "SwapFree", "MemTotal")), "Invalid host telemetry")
        require(all(type(row[k]) is int and row[k] >= 0 for k in ("VmRSS", "VmSwap", "VmHWM") if k in row), "Invalid child telemetry")
        elapsed = row["elapsed_seconds"]
        require(type(elapsed) in (int, float) and math.isfinite(elapsed) and previous <= elapsed <= duration, "Invalid telemetry chronology")
        previous = elapsed
        low = low + 1 if row["MemAvailable"] < c.RESERVE else 0
        require(low < 2 and start["SwapFree"] - row["SwapFree"] <= c.POLICY["swap_loss_bytes"] and row.get("VmSwap", 0) <= c.POLICY["child_swap_bytes"], "CLI violated memory containment")
    return {"child_pid": executed["pid"], **{n.replace(".", "_") + "_sha256": sha(run / n) for n in ("invocation.json", "executed.json", "child.log", "resources.json", "memory.jsonl")}}


def _difference(reference: Any, actual: Any) -> dict[str, Any]:
    import numpy as np
    r, a = np.asarray(reference, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    result = {"reference_shape": list(r.shape), "actual_shape": list(a.shape), "row_aligned": r.shape == a.shape}
    if r.shape == a.shape:
        result.update(max_absolute_error=float(np.max(np.abs(a-r))), rms_error=float(np.sqrt(np.mean((a-r)**2))))
    return result


def validate_capture(plan_path: Path | str, expected_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return authenticated CLI identity and complete-array numerical diagnostics."""
    import numpy as np
    plan, bindings, modules = authenticate_plan(Path(plan_path), expected_sha)
    c, comparator = modules["capture_identity"], modules["compare_reference"]
    run = c._path(plan["run_dir"])
    process = _resources(run, plan, expected_sha, c)
    output = c._path(str(run / "output"))
    require({p.name for p in output.iterdir()} == OUTPUT_FILES, "CLI artifact set differs")
    hashes = {n: sha(c._path(str(output / n))) for n in ARTIFACTS}
    hashes[INSTANCE_SIDECAR] = empty_instances_sha(output / INSTANCE_SIDECAR)
    c._tensor_header(output / "result.safetensors")
    compact = comparator.load_file(output / "result.safetensors")
    validate_compact(compact)
    meta = c.read_json(output / "meta.json")
    options = bindings["canonical"]["default_options"]
    for key in ("seed", "ss_steps", "slat_steps", "formats"):
        require(json.dumps(meta[key]) == json.dumps(options[key]), f"CLI default metadata differs: {key}")
    height, width = bindings["canonical"]["input_shape"]
    require(meta["input_size"] == {"width": width, "height": height} and meta["family"] == "sam3d_objects" and meta["task"] == "mesh" and meta["pointmap_conditioned"] is True, "CLI metadata identity differs")
    require(type(meta["timing_ms"]) in (int, float) and math.isfinite(meta["timing_ms"]) and meta["timing_ms"] >= 0, "Invalid CLI timing")
    require(set(meta["maps"]) == set(compact), "CLI metadata map set differs")
    for name, value in compact.items():
        bounds = meta["maps"][name]
        require(isinstance(bounds, dict) and set(bounds) == {"min", "max"}, "Invalid CLI metadata range")
        for key, expected in (("min", np.min(value)), ("max", np.max(value))):
            require(type(bounds[key]) in (int, float) and math.isfinite(bounds[key]) and np.float32(bounds[key]) == np.float32(expected), f"CLI metadata range differs: {name}")
    vertices, faces = read_obj(output / "mesh.obj")
    gaussian = read_ply(output / "gaussian.ply")
    refroot = Path(plan["reference"]["path"]).parent
    reference = comparator.load_file(refroot / "reference.safetensors")
    modules["provenance"].validate_result(reference, options["formats"], True)
    for name, value in reference.items():
        require(value.dtype.kind in ("iu" if name in ("coords", "mesh.faces") else "f"), f"Reference semantic dtype differs: {name}")
    refvertices, reffaces = read_obj(refroot / "mesh.obj")
    refgaussian = read_ply(refroot / "gaussian.ply")
    keys = ("coords", "latents", "mesh.vertices")
    actual = {"coords": compact["coords"], "latents": compact["latents"], "mesh.vertices": vertices[:, :3]}
    metrics, failures = comparator.compare({n: reference[n] for n in keys}, actual)
    report = {"kind": "default-no-opts-cli", "metrics": metrics, "failures": failures,
              "pose": {n: _difference(reference[n], compact[n]) for n in POSE},
              "mesh_export": {"vertices_rgb": _difference(refvertices, vertices), "faces": _difference(reffaces, faces),
                              "faces_exact": bool(np.array_equal(reffaces, faces))},
              "gaussian_export": {"reference_count": len(refgaussian), "actual_count": len(gaussian),
                                  "fields": {n: _difference(refgaussian[:, i], gaussian[:, i]) for i, n in enumerate(FIELDS)}},
              "coverage": "All exported rows; OBJ contains only RGB and rounded XYZ; PLY contains log-scales and clipped logit-opacity. Six-channel colors/raw Gaussian representation require integration full capture.",
              "diagnostic_only": ["pose", "mesh_export", "gaussian_export"], "contract_sha256": plan["helpers"]["comparison-contract.json"]["sha256"]}
    identity = {"producer": "mlxcel-cli", "case": "default", "runtime_device": "GPU", "tf32": "0", "options": options,
                "pointmap_conditioned": True, "plan_sha256": expected_sha, "executable": plan["executable"], "source": plan["source"],
                "source_files": bindings["manifest"], "checkpoint": bindings["canonical"]["checkpoint"],
                "input_files": {n: bindings["canonical"]["inputs"][n] for n in ("object.png", "mask.png")},
                "reference_receipt_sha256": plan["reference"]["sha256"], "reference_artifact_sha256": bindings["reference"]["artifact_sha256"],
                "artifact_sha256": hashes, "helper_sha256": {n: plan["helpers"][n]["sha256"] for n in HELPERS},
                "child_exit": 0, "wrapper_exit": 0, "run_dir": str(run), **process}
    report["identity"] = identity
    return identity, report
