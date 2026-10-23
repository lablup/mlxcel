"""Fail-closed identity and semantic checks for authentic MoGe3 references."""

import hashlib
import json
import pathlib

import numpy as np

MANIFEST = pathlib.Path(__file__).with_name("provenance.json")


def sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical():
    data = json.loads(MANIFEST.read_text())
    require(set(data) == {"source_revision", "source_sha256", "runtime", "image_sha256", "image_shape", "checkpoints"}, "Malformed canonical provenance")
    require(bool(data["source_sha256"]) and set(data["checkpoints"]) == {"vitl", "vitg"}, "Empty provenance")
    return data


def verify_source_runtime(source, mx):
    expected = canonical()
    for name, digest in expected["source_sha256"].items():
        require(sha(source / name) == digest, f"Unpinned source: {name}")
    runtime = expected["runtime"]
    require(mx.__version__ == runtime["version"], "MLX version mismatch")
    require(sha(mx.__file__) == runtime["binding_sha256"], "MLX binding hash mismatch")
    paths = set()
    for line in pathlib.Path("/proc/self/maps").read_text().splitlines():
        fields = line.split()
        if len(fields) >= 6 and pathlib.Path(fields[-1]).name == "libmlx.so":
            paths.add(pathlib.Path(fields[-1]).resolve())
    require(len(paths) == 1, f"Expected exactly one mapped libmlx.so: {paths}")
    native = next(iter(paths))
    require(sha(native) == runtime["native_sha256"], "Mapped MLX native library mismatch")
    return {"version": mx.__version__, "binding_path": mx.__file__, "binding_sha256": sha(mx.__file__), "native_path": str(native), "native_sha256": sha(native)}


def verify_inputs(model, image):
    expected = canonical()
    require(sha(image) == expected["image_sha256"], "Fixed image hash mismatch")
    matches = []
    for size, checkpoint in expected["checkpoints"].items():
        if all(sha(model / name) == value["sha256"] for name, value in checkpoint["files"].items()):
            matches.append(size)
    require(len(matches) == 1, "Checkpoint/config/preprocessor hashes do not match a published pin")
    return matches[0]


def options(raw):
    result = {"resolution_level": 9, "num_tokens": None, "refine_steps": 3,
              "return_per_step": False, "force_projection": True, "apply_mask": True, "fov_x": None}
    require(isinstance(raw, dict) and not (raw.keys() - result.keys()), "Unknown inference options")
    result.update(raw)
    require(type(result["refine_steps"]) is int and 0 <= result["refine_steps"] <= 16, "Invalid refinement count")
    require(type(result["resolution_level"]) is int and 0 <= result["resolution_level"] <= 9, "Invalid resolution")
    require(result["num_tokens"] is None or (type(result["num_tokens"]) is int and 1200 <= result["num_tokens"] <= 3600), "Invalid token count")
    for key in ("return_per_step", "force_projection", "apply_mask"):
        require(type(result[key]) is bool, f"{key} must be boolean")
    fov = result["fov_x"]
    require(fov is None or (type(fov) in (int, float) and np.isfinite(fov) and 0 < fov < 180), "Invalid fov")
    return result


def validate_outputs(values, opts, image_shape, reference=False):
    height, width, channels = image_shape
    require(channels == 3 and height > 0 and width > 0, "Invalid input shape")
    shapes = {"points": (height, width, 3), "depth": (height, width), "normal": (height, width, 3), "mask": (height, width), "intrinsics": (3, 3)}
    if opts["return_per_step"]:
        for step in range(opts["refine_steps"] + 1):
            for family in ("points", "depth", "intrinsics"):
                shapes[f"{family}_step{step}"] = shapes[family]
    require(values.keys() == shapes.keys(), f"Unexpected schema: {sorted(values)} != {sorted(shapes)}")
    for key, shape in shapes.items():
        value = np.asarray(values[key])
        require(value.shape == shape, f"{key}: shape {value.shape} != {shape}")
        if key == "mask":
            require(np.isin(value, [0, 1]).all(), "Mask must be binary")
            if reference:
                require(value.dtype == np.bool_, "Upstream mask must be bool")
        else:
            require(value.dtype == np.float32, f"{key}: expected F32, got {value.dtype}")


def verify_receipt(receipt, tensor_path, reference):
    expected = canonical()
    require(receipt.get("output_sha256") == sha(tensor_path), "Output artifact hash mismatch")
    require(receipt.get("image_sha256") == expected["image_sha256"], "Receipt image mismatch")
    hashes = {entry["files"]["model.safetensors"]["sha256"] for entry in expected["checkpoints"].values()}
    require(receipt.get("checkpoint_sha256") in hashes, "Receipt checkpoint is unpinned")
    require(receipt.get("tf32") == "0", "Receipt must record disabled TF32")
    device = receipt.get("device", "")
    require(device in ("cpu", "gpu", "Device(cpu, 0)", "Device(gpu, 0)"), "Invalid receipt device")
    if reference:
        require(receipt.get("source_revision") == expected["source_revision"], "Reference source revision mismatch")
        require(receipt.get("source_sha256") == expected["source_sha256"], "Reference source hashes mismatch")
        runtime = receipt.get("runtime_receipt", {})
        require(runtime.get("version") == expected["runtime"]["version"], "Reference runtime version mismatch")
        for key in ("native_sha256", "binding_sha256"):
            require(runtime.get(key) == expected["runtime"][key], f"Reference {key} mismatch")
    else:
        require(receipt.get("implementation") == "rust" and bool(receipt.get("revision")), "Actual receipt must identify Rust build revision")
    require(receipt.get("image_shape") == expected["image_shape"], "Receipt image shape mismatch")
    require(isinstance(receipt.get("options"), dict), "Missing inference options")
    return options(receipt["options"])
