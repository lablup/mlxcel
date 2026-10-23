#!/usr/bin/env python3
# Copyright 2025-2026 Lablup Inc.
# SPDX-License-Identifier: Apache-2.0
"""Verify the complete PPDoc fixture inventory and genuine tensor contracts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    inventory = json.loads((root / "SHA256.json").read_text())
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    require(actual == set(inventory) | {"SHA256.json"}, "fixture inventory differs")
    for name, entry in inventory.items():
        path = root / name
        require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),
                f"unsafe fixture path: {name}")
        require(path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"],
                f"fixture changed: {name}")
    require(digest(root / "input.safetensors") ==
            "0023b22357d4fb93a6c6a721c6386df75e44d85eba9020b804c5a2e0fe9b7faf",
            "original processor input changed")
    inputs = load_file(root / "input.safetensors")
    require(set(inputs) == {"decoded_rgb_u8", "pixel_values_nhwc"}, "input tensor keys")
    require(inputs["decoded_rgb_u8"].shape == (1684, 1191, 3) and
            inputs["decoded_rgb_u8"].dtype == np.uint8, "decoded input schema")
    pixels = inputs["pixel_values_nhwc"]
    require(pixels.shape == (1, 800, 800, 3) and pixels.dtype == np.float32 and
            np.isfinite(pixels).all(), "processed input schema")
    expected = {"logits": (1, 300, 25), "pred_boxes": (1, 300, 4),
                "order_logits": (1, 300, 300)}
    for kind in ("paddle", "hashnuke"):
        path = root / f"{kind}_raw.safetensors"
        require(digest(path) ==
                "170221e4b16da6abb1d5e2676c3906da61e7a5a61af362bd0b43869b0b24d7d1",
                f"original {kind} raw output changed")
        tensors = load_file(path)
        require(set(tensors) == set(expected), f"{kind} output keys")
        for name, shape in expected.items():
            value = tensors[name]
            require(value.shape == shape and value.dtype == np.float32 and
                    np.isfinite(value).all(), f"{kind} {name} schema/finite")
        provenance = json.loads((root / f"{kind}_provenance.json").read_text())
        require(provenance["output_sha256"] == digest(path), "output provenance")
        loaded = provenance["loaded_parameters"]
        require(len(loaded) == 857 and all(x["dtype"] == "mlx.core.float32"
                                         for x in loaded.values()), "loaded dtype")
        for suffix in ("child-status", "wrapper-status"):
            require((root / "execution" / f"epic-2382-ppdoc-raw-v2-{kind}.{suffix}")
                    .read_text().strip() == "0", f"{kind} {suffix}")
    post = json.loads((root / "expected_detections.json").read_text())
    require(post["results"]["paddle"] == post["results"]["hashnuke"],
            "checkpoint postprocess differs")
    detections = post["results"]["paddle"]["rust_expected_detections"]
    require([d["reading_order"] for d in detections] == list(range(1, 17)),
            "dense model reading order")
    require(post["results"]["paddle"]["order_differs"], "two-column discriminator")
    rt = json.loads((root / "rt_detr_heron_expected.json").read_text())
    require(rt["child_exit"] == 0 and rt["wrapper_exit"] == 0, "RT baseline exits")
    require(digest(root / "execution/rt-heron-baseline.json") == rt["raw_json_sha256"],
            "RT baseline identity")
    require(len(rt["detections"]) == 28 and all(
        np.isfinite(d["bbox"] + [d["score"]]).all() and "reading_order" not in d
        for d in rt["detections"]), "RT baseline finite/schema")
    print(f"PASS: exact {len(inventory)}-file inventory, authentic inputs/raw outputs, "
          "loaded F32 evidence, two-column order, and RT baseline")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent)
    verify(parser.parse_args().directory)
