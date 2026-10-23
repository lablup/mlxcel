#!/usr/bin/env python3
# Copyright 2025-2026 Lablup Inc.
# SPDX-License-Identifier: Apache-2.0
"""Verify authentic RF-DETR fixtures without loading either model."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parent
PINS = {
    "base": ("eddb85a12c5996074fac2f2e769ca5e947bd80d3", 487, 498, 560, 300, 5),
    "seg": ("a7a2375b4ca7b2ef0427af91b55d8176ca647b35", 544, 559, 384, 100, 4),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    inventory = read(ROOT / "SHA256.json")
    actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file() and p.name != "SHA256.json"}
    require(actual == set(inventory), "exact fixture inventory")
    for name, entry in inventory.items():
        path = ROOT / name
        require(path.resolve().is_relative_to(ROOT), "contained fixture path")
        require(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], "fixture hash: " + name)
    image = ROOT.parent / "sam3/cats.png"
    require(sha(image) == "2a7ba590d525fe31f5627c73b2e83a5b63b3cba3840fb10ae9d726187aca6769", "source image")
    generation = ROOT / "generation"
    source = read(generation / "upstream-manifest.json")
    require(sha(generation / "upstream-manifest.json") == "d9853592f09e0d4faca0b976a75dfa40959a75328d9890ed49f23dc0da948c71", "source manifest")
    for name, expected in source["files"].items():
        require(sha(generation / "upstream" / name) == expected, "unchanged upstream source")
    for suffix in ["guard-status", "wrapper-status"]:
        require((ROOT / "execution" / ("epic-2382-rfdetr-reference-v2." + suffix)).read_text().strip() == "0", "successful " + suffix)
    for kind, (revision, checkpoint_count, loaded_count, resolution, queries, count) in PINS.items():
        directory = ROOT / kind
        expected_receipt = {
            "base": "cd322b62ae9902a1ef05f2f62c02ff0935e6ff92514d1255efab205bbde53cd9",
            "seg": "6cd1e05437fcbf975ccee2917f44094751c4442840e94250e9cd0b4e3897c49c",
        }[kind]
        require(sha(directory / "receipt.json") == expected_receipt, "reviewed immutable receipt")
        receipt = read(directory / "receipt.json")
        require(receipt["kind"] == kind and receipt["checkpoint"]["revision"] == revision, "checkpoint identity")
        require(receipt["source_manifest_sha256"] == sha(generation / "upstream-manifest.json"), "source binding")
        require(receipt["native_sha256"] == "c1d8203b257364271e1ea6298705dd3bb0c82a7f06634f5578a209883601a9fc", "corrected native runtime")
        require(receipt["binding_sha256"] == "f8ec0f363bc6eab1cda644a2579ae43aa663939f30ace531d5fb8cc33c7090cc", "binding")
        require(receipt["runtime_identity_sha256"] == "a34c99a7509385a0bc7f45081f7d0b215fae6ac65f738005ae6d469ef086a793", "runtime receipt")
        require(receipt["actual_device"] == "Device(gpu, 0)" and receipt["tf32"] == "0", "device and precision")
        require(receipt["image_sha256"] == sha(image) and receipt["image_size"] == [480, 640], "input binding")
        require(receipt["checkpoint_tensor_count"] == checkpoint_count and receipt["parameter_count"] == loaded_count, "separate raw/loaded counts")
        require(len(receipt["loaded_parameters"]) == loaded_count and all(v["dtype"] == "mlx.core.float32" for v in receipt["loaded_parameters"].values()), "actual F32 parameters")
        for field, name in [("generator_sha256", "epic-2382-rfdetr-reference-v2.py"), ("wrapper_sha256", "epic-2382-rfdetr-reference-v2.sh"), ("launch_sha256", "epic-2382-rfdetr-reference-v2-launch.sha256")]:
            require(receipt[field] == sha(generation / name), "executed generator binding")
        require(set(receipt["files"]) == {p.name for p in directory.iterdir() if p.name != "receipt.json"}, "receipt file inventory")
        for name, entry in receipt["files"].items():
            require(sha(directory / name) == entry["sha256"] and (directory / name).stat().st_size == entry["bytes"], "output receipt")
        inputs = load_file(directory / "input.safetensors")
        require(set(inputs) == {"pixel_values"}, "input keys")
        x = inputs["pixel_values"]
        require(x.shape == (1, resolution, resolution, 3) and x.dtype == np.float32 and np.isfinite(x).all(), "input shape/dtype/finite")
        require(hashlib.sha256(x.tobytes()).hexdigest() == receipt["processed_sha256"], "actual processor bytes")
        raw = load_file(directory / "raw.safetensors")
        shapes = {"pred_logits": (1, queries, 91), "pred_boxes": (1, queries, 4)}
        if kind == "seg":
            shapes["pred_masks"] = (1, queries, 96, 96)
        require(set(raw) == set(shapes) == set(receipt["outputs"]), "complete raw keys")
        for name, shape in shapes.items():
            require(raw[name].shape == shape and raw[name].dtype == np.float32 and np.isfinite(raw[name]).all(), "raw shape/dtype/finite")
            require(receipt["outputs"][name] == {"shape": list(shape), "dtype": "float32", "finite": True}, "raw metadata")
        detections = read(directory / "detections.json")
        require(detections["threshold"] == 0.5 and detections["nms_iou"] == 0.5 and detections["num_select"] == 300 and detections["source_size"] == [480, 640], "postprocess contract")
        records = detections["detections"]
        require(len(records) == count, "actual nonempty detection count")
        previous = 1.0
        for record in records:
            require(0.5 < record["score"] <= previous and 0 <= record["query_index"] < queries, "score order/query mapping")
            previous = record["score"]
            box = record["box"]
            require(len(box) == 4 and np.isfinite(box).all() and 0 <= box[0] <= box[2] <= 640 and 0 <= box[1] <= box[3] <= 480, "source boxes")
            if kind == "seg":
                mask = np.array(Image.open(directory / record["mask"]))
                require(mask.shape == (480, 640) and mask.dtype == np.uint8 and set(np.unique(mask)) == {0, 255}, "binary source-size mask")
                require(int(np.count_nonzero(mask)) == record["foreground_pixels"] and sha(directory / record["mask"]) == record["mask_sha256"], "mask evidence")
            else:
                require("mask" not in record, "detection-only masks absent")
        status = ROOT / "execution" / f"epic-2382-rfdetr-{kind}-reference-v2.child-status"
        log = (ROOT / "execution" / f"epic-2382-rfdetr-{kind}-reference-v2.log").read_text()
        require(status.read_text().strip() == "0" and "bound HIP device 0: gfx1151" in log and f"PASS {kind} {count} detections" in log, "actual successful child")
    print(f"PASS: {len(inventory)} authentic RF-DETR fixture files")


if __name__ == "__main__":
    main()
