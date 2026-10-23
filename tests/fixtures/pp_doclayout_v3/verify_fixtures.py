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


def verify_final_cli(root):
    """Validate the historical real CLI acceptance independently of inventory."""
    base = root / "execution/final-cli-falcon"
    expected = json.loads((root / "falcon_ocr_expected.json").read_text())
    require(expected["capture_receipt"] == "execution/final-cli-falcon/receipt.json",
            "CLI receipt pointer")
    receipt = json.loads((base / "receipt.json").read_text())
    require(receipt["source_manifest_sha256"] ==
            "b3314f9abf302a4ebc6f89fd7a3ef9fd61b46b942fe0ba657bb7607c79373f66"
            == digest(base / "source.sha256"), "historical source manifest")
    binding = json.loads((base / "source-binding.json").read_text())
    require(binding["source_commit"] == "2de95857d8f7e381cc78ee20b1f070b91f2e5578"
            and binding["source_manifest_sha256"] == receipt["source_manifest_sha256"],
            "historical source commit")
    require(receipt["cli_binary_sha256"] ==
            "069b3eaecfb8f238cdb64cbcebf99f38caa193e9c3a203b5c138c118532fa614"
            and receipt["integration_binary_sha256"] ==
            "bbca4537642faaea71ff49e5bf4c469e33a953f6d5ad34f29f12f56946c88415",
            "accepted binary identities")
    require(receipt["checkpoint_layouts"] == ["hashnuke", "paddle"]
            and receipt["falcon_checkpoint_revision"] == expected["falcon_checkpoint_revision"]
            and receipt["mlx_enable_tf32"] == "0"
            and receipt["device"] == "ROCm gfx1151 via serialized GPU guard"
            and receipt["selector"] == "real_cli_preserves_two_column_model_order_into_falcon_ocr",
            "accepted runtime/options")
    for kind in ("child", "wrapper"):
        require((base / f"epic-2382-ppdoc-falcon-fix-v1-full.{kind}.status")
                .read_text().strip() == "0", f"CLI {kind} exit")
    for name in ("detect-hashnuke", "detect-paddle", "falcon-ocr"):
        require((base / f"{name}.status").read_text().strip() == "exit_code=0",
                f"CLI {name} exit")
        require("[mlx-rocm] bound HIP device 0: gfx1151" in
                (base / f"{name}.stderr").read_text(), f"CLI {name} device")
    require((base / "detect-hashnuke.stdout").read_bytes() ==
            (base / "detect-paddle.stdout").read_bytes(), "both layout CLI equality")
    require(digest(base / "detect-hashnuke.stdout") ==
            "877e6a5a7562d89b031414968bc83b6589062110ba9fbae750550cad7241ec69",
            "accepted detector output")
    detections = json.loads((base / "detect-hashnuke.stdout").read_text())["detections"]
    require(len(detections) == expected["detector_expected_count"] == 16
            and [d["reading_order"] for d in detections] == list(range(1, 17)),
            "CLI complete model order")
    require(digest(base / "falcon-ocr.stdout") == expected["capture_transcript_sha256"] ==
            "d202733dbaa3157741f1953a1178f685d411073372d4e8f9318bcf30adf8472d",
            "accepted transcript identity")
    require(receipt["required_anchor_order"] == expected["anchors"] ==
            ["Introduction", "Language model pre-training", "There are two existing strategies"],
            "independent anchor specification")
    text = " ".join((base / "falcon-ocr.stdout").read_text().lower().split())
    positions = [text.find(" ".join(a.lower().split())) for a in expected["anchors"]]
    require(all(p >= 0 for p in positions) and positions == sorted(set(positions)),
            "actual cross-column anchor order")
    log = (base / "epic-2382-ppdoc-falcon-fix-v1-full.child.log").read_text()
    require("1 passed; 0 failed; 0 ignored" in log
            and expected["capture_transcript_sha256"] in log, "actual acceptance child log")


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
    verify_final_cli(root)
    print(f"PASS: exact {len(inventory)}-file inventory, authentic inputs/raw outputs, "
          "loaded F32 evidence, two-column order, and RT baseline")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent)
    verify(parser.parse_args().directory)
