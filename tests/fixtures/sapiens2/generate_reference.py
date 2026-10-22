#!/usr/bin/env python3
"""Generate Sapiens2 parity outputs with the pinned upstream implementation."""

import argparse
import json
import pathlib

import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_vlm.models.sapiens2 import Model, ModelConfig
from mlx_vlm.models.sapiens2.generate import Sapiens2Predictor


def canonical_config(raw: dict) -> dict:
    """Adapt the published early segmentation metadata to the flat schema."""
    if "backbone_config" not in raw:
        return raw
    backbone = raw["backbone_config"]
    head = raw["head_config"]
    task = raw["task"]
    architectures = {
        "seg": "Sapiens2ForSemanticSegmentation",
        "pose": "Sapiens2ForPoseEstimation",
        "normal": "Sapiens2ForNormalEstimation",
        "pointmap": "Sapiens2ForPointmapEstimation",
        "matting": "Sapiens2ForImageMatting",
    }
    return {
        "architectures": [architectures[task]],
        "task": task,
        "hidden_size": backbone["embed_dims"],
        "num_hidden_layers": backbone["num_layers"],
        "num_attention_heads": backbone["num_heads"],
        "intermediate_size": backbone["feedforward_channels"],
        "image_size": backbone["image_size"],
        "patch_size": backbone["patch_size"],
        "num_register_tokens": backbone["n_storage_tokens"],
        "rope_theta": backbone.get("rope_base", 100.0),
        "num_first_full_attention_layers": backbone.get("mhsa_early", 8),
        "num_last_full_attention_layers": backbone.get("mhsa_late", 8),
        "normalize_backbone_outputs": backbone.get("final_norm", True),
        "num_labels": head.get("num_classes", head.get("num_keypoints")),
        "head_config": {
            "upsample_out_channels": head.get(
                "deconv_out_channels", head.get("upsample_channels", [])
            ),
            "upsample_kernel_sizes": head.get("deconv_kernel_sizes", []),
            "use_pixel_shuffle": head.get("upsample_channels") is not None,
            "conv_out_channels": head.get("conv_out_channels") or [],
            "conv_kernel_sizes": head.get("conv_kernel_sizes") or [],
            "scale_conv_out_channels": head.get("scale_conv_out_channels") or [],
            "scale_conv_kernel_sizes": head.get("scale_conv_kernel_sizes") or [],
            "scale_final_input_size": (head.get("scale_final_layer") or [None])[0],
            "scale_final_hidden_sizes": (head.get("scale_final_layer") or [])[1:-1],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("image", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--dtype", choices=("bf16", "f16"), default="bf16")
    parser.add_argument("--flip", action="store_true")
    args = parser.parse_args()

    mx.set_default_device(mx.cpu)
    raw = json.loads((args.checkpoint / "config.json").read_text())
    model = Model(ModelConfig.from_dict(canonical_config(raw)))
    weights = model.sanitize(mx.load(str(args.checkpoint / "model.safetensors")))
    if args.dtype == "f16":
        weights = {
            key: value.astype(mx.float16)
            if mx.issubdtype(value.dtype, mx.floating)
            else value
            for key, value in weights.items()
        }
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    image = Image.open(args.image).convert("RGB")
    result = Sapiens2Predictor(model).infer(image, flip_test=args.flip)
    arrays = {
        key: np.array(
            value.astype(mx.float32)
            if mx.issubdtype(value.dtype, mx.floating)
            else value
        )
        for key, value in result.items()
    }
    if args.output.suffix == ".png":
        if set(arrays) != {"segmentation"}:
            raise ValueError("PNG output requires a segmentation checkpoint")
        labels = arrays["segmentation"]
        if labels.ndim != 2 or labels.min() < 0 or labels.max() > 255:
            raise ValueError("segmentation labels must be a 2D uint8-compatible map")
        Image.fromarray(labels.astype(np.uint8), mode="L").save(args.output)
    elif args.output.suffix == ".npz":
        np.savez(args.output, **arrays)
    elif args.output.suffix == ".json":
        output = {key: value.tolist() for key, value in arrays.items()}
        args.output.write_text(json.dumps(output, separators=(",", ":")))
    else:
        raise ValueError("output suffix must be .json, .npz, or .png")


if __name__ == "__main__":
    main()
