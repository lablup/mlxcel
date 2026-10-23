"""Run the actual CLI, record its invocation, and verify serialized geometry."""

import argparse
import json
import os
import pathlib
import subprocess

import mlx.core as mx
import numpy as np
from PIL import Image

from compare_reference import compare
from provenance import canonical, require, sha, verify_inputs, verify_receipt


def resolved_device(log):
    values = [line.removeprefix("Runtime device: ").strip()
              for line in log.splitlines() if line.startswith("Runtime device: ")]
    require(len(values) == 1, "Missing or ambiguous CLI runtime-device attestation")
    devices = {"CPU": "cpu", "GPU": "gpu", "NVIDIA GPU (CUDA)": "gpu", "Apple GPU (Metal)": "gpu"}
    require(values[0] in devices, "Unknown CLI runtime-device attestation")
    return devices[values[0]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--image", type=pathlib.Path, required=True)
    parser.add_argument("--reference", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    mx.set_default_device(mx.cpu)
    verify_inputs(args.model, args.image)
    verified = args.reference.with_suffix(".verified.json")
    receipt = json.loads((verified if verified.exists() else args.reference.with_suffix(".json")).read_text())
    opts = verify_receipt(receipt, args.reference, reference=True)
    require(receipt["device"] in (args.device, f"Device({args.device}, 0)"), "Reference backend mismatch")
    require(not args.output.exists(), "Choose a new output directory to preserve prior artifacts")
    args.output.mkdir(parents=True)
    env = dict(os.environ, MLX_ENABLE_TF32="0", MLXCEL_DEVICE=args.device)
    command = [str(args.binary.resolve()), "predict", "-m", str(args.model.resolve()),
               "--image", str(args.image.resolve()), "-o", str(args.output.resolve())]
    # An all-default case deliberately exercises the actual no --opts CLI path.
    defaults = {"resolution_level": 9, "num_tokens": None, "refine_steps": 3,
                "return_per_step": False, "force_projection": True, "apply_mask": True, "fov_x": None}
    if opts != defaults:
        command += ["--opts", json.dumps({key: value for key, value in opts.items() if value is not None})]
    with (args.output / "process.log").open("w") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    (args.output / "process.exit").write_text(str(result.returncode) + "\n")
    require(result.returncode == 0, f"CLI failed with exit {result.returncode}; inspect process.log")
    actual_device = resolved_device((args.output / "process.log").read_text())
    require(actual_device == args.device, "CLI resolved device differs from requested/reference backend")
    tensor_path = args.output / "result.safetensors"
    actual_receipt = {
        "implementation": "rust", "revision": args.revision,
        "binary_sha256": sha(args.binary), "command": command,
        "output_sha256": sha(tensor_path), "checkpoint_sha256": sha(args.model / "model.safetensors"),
        "image_sha256": sha(args.image), "image_shape": canonical()["image_shape"],
        "options": opts, "device": actual_device, "tf32": "0",
    }
    tensor_path.with_suffix(".json").write_text(json.dumps(actual_receipt, indent=2) + "\n")
    metrics, failures = compare(tensor_path, args.reference, actual_device)
    meta = json.loads((args.output / "meta.json").read_text())
    require(meta.get("family") == "moge3" and meta.get("task") == "geometry", "CLI metadata routing mismatch")
    require(meta.get("input_size") == {"height": 480, "width": 640}, "CLI input dimensions mismatch")
    intrinsics = meta.get("intrinsics", {})
    require(intrinsics.get("shape") == [3, 3], "Missing camera intrinsics metadata")
    saved = np.asarray(mx.load(str(tensor_path))["intrinsics"])
    listed = np.asarray(intrinsics.get("values", []), dtype=np.float32)
    require(listed.shape == (9,) and np.array_equal(listed.reshape(3, 3), saved), "Metadata intrinsics differ from tensor")
    for name in ("depth", "normal", "mask"):
        with Image.open(args.output / f"{name}.png") as preview:
            require(preview.size == (640, 480), f"{name} preview dimensions mismatch")
    arch = subprocess.run([str(args.binary.resolve()), "arch"], env=env, capture_output=True, text=True)
    require(arch.returncode == 0 and "moge3" in arch.stdout, "Architecture catalog omits moge3")
    report = {"metrics": metrics, "failures": failures, "previews": ["depth.png", "normal.png", "mask.png"], "intrinsics_metadata": True, "arch": True}
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    require(not failures, f"CLI parity failed: {failures}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
