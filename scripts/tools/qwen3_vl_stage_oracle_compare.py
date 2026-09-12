#!/usr/bin/env python3
"""Compare Qwen3-VL/Cohere Compass vision stage dumps against transformers.

This helper is intentionally out of the Rust request path. It consumes dumps produced by the
ignored Rust test `vision::encoders::qwen3_vl::tests::dump_north_micro_vision_stage_tensors`
and compares them with the current transformers Cohere Compass vision implementation using
`AutoProcessor` pixel values for the same image.

The `copy-f32` subcommand creates a local, untracked checkpoint copy that converts BF16
safetensor tensors to float32 while preserving uint32 quantized tensors. That mirrors the #1735
oracle setup closely enough to separate forward-pass differences from BF16 storage effects.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import shutil
from typing import Any

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoProcessor
from transformers.models.cohere_compass.modeling_cohere_compass import CohereCompassVisionModel
from transformers.vision_utils import (
    get_vision_attention_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)

TRANSFORMERS_GIT_FOR_ISSUE_1738 = "df04b012229d50d2b6dfba32c61c3057c3a40ea1"


def _load_manifest(rust_dump: pathlib.Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads((rust_dump / "manifest.json").read_text())
    return manifest, {stage["stage"]: stage for stage in manifest["stages"]}


def _read_rust_stage(rust_dump: pathlib.Path, stages: dict[str, Any], stage: str) -> np.ndarray:
    meta = stages[stage]
    arr = np.fromfile(rust_dump / meta["file"], dtype="<f4")
    return arr.reshape(tuple(meta["shape"]))


def _diff_record(stage: str, got: np.ndarray, ref: np.ndarray, **extra: Any) -> dict[str, Any]:
    got = np.asarray(got, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    record: dict[str, Any] = {
        "stage": stage,
        "shape_rust": list(got.shape),
        "shape_oracle": list(ref.shape),
        **extra,
    }
    if got.shape != ref.shape:
        record["shape_match"] = False
        return record
    diff = got - ref
    record.update(
        {
            "shape_match": True,
            "max_abs": float(np.max(np.abs(diff))),
            "mean_abs": float(np.mean(np.abs(diff))),
            "rmse": float(math.sqrt(float(np.mean(diff * diff)))),
        }
    )
    return record


def _dtype_from_arg(name: str) -> torch.dtype:
    if name == "f32":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def _load_vision(model_dir: pathlib.Path, model_dtype: torch.dtype) -> CohereCompassVisionModel:
    cfg = AutoConfig.from_pretrained(model_dir)
    vision = CohereCompassVisionModel(cfg.vision_config).to(model_dtype)
    state: dict[str, torch.Tensor] = {}
    with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if not key.startswith("vision_tower."):
                continue
            dst = key[len("vision_tower.") :]
            tensor = handle.get_tensor(key).to(model_dtype)
            if dst == "patch_embed.proj.weight":
                # MLX stores Conv3d kernels as [out, T, H, W, C]; transformers expects [out, C, T, H, W].
                tensor = tensor.permute(0, 4, 1, 2, 3).contiguous()
            state[dst] = tensor
    missing, unexpected = vision.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"vision state mismatch: missing={missing}, unexpected={unexpected}")
    vision.eval()
    return vision


def compare(args: argparse.Namespace) -> None:
    model_dir = args.model.resolve()
    rust_dump = args.rust_dump.resolve()
    manifest, stages = _load_manifest(rust_dump)
    cfg = AutoConfig.from_pretrained(model_dir)
    model_dtype = _dtype_from_arg(args.model_dtype)
    input_dtype = _dtype_from_arg(args.input_dtype)
    vision = _load_vision(model_dir, model_dtype)
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    image_path = pathlib.Path(manifest["image"])
    if not image_path.is_absolute():
        image_path = pathlib.Path.cwd() / image_path
    image = Image.open(image_path)
    inputs = processor(images=[image], text=[args.processor_text], return_tensors="pt")
    grid = inputs["image_grid_thw"].to(torch.long)
    manifest_grid = torch.tensor(manifest["grid_thw"], dtype=torch.long)
    if not torch.equal(grid, manifest_grid):
        raise RuntimeError(f"processor grid {grid.tolist()} != Rust manifest grid {manifest_grid.tolist()}")
    pixel_values = inputs["pixel_values"].to(input_dtype)

    records: list[dict[str, Any]] = []
    rust_input = _read_rust_stage(rust_dump, stages, "input_patches")
    temporal = cfg.vision_config.temporal_patch_size
    channels = cfg.vision_config.in_channels
    patch = cfg.vision_config.patch_size
    rust_ct = rust_input.reshape(pixel_values.shape[0], temporal, channels, patch, patch).transpose(0, 2, 1, 3, 4).reshape(pixel_values.shape[0], -1)
    records.append(
        _diff_record(
            "input_patches_processor_ct_order",
            rust_ct,
            pixel_values.float().cpu().numpy(),
            rust_original_shape=list(rust_input.shape),
        )
    )
    records.append(
        _diff_record(
            "input_patches_raw_tc_order",
            rust_input.reshape(pixel_values.shape[0], -1),
            pixel_values.float().cpu().numpy(),
            rust_original_shape=list(rust_input.shape),
        )
    )

    def stats(stage: str, ref: torch.Tensor) -> dict[str, Any]:
        return _diff_record(stage, _read_rust_stage(rust_dump, stages, stage), ref.detach().float().cpu().numpy())

    with torch.no_grad():
        interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
            grid,
            num_grid_per_side=vision.num_grid_per_side,
            mode=vision.interpolation_mode,
            align_corners=vision.interpolation_align_corners,
            spatial_merge_size=vision.config.spatial_merge_size,
            kwargs={},
        )
        position_ids = get_vision_position_ids(grid, vision.spatial_merge_size, kwargs={})
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid, vision.config, kwargs={})

        hidden = vision.patch_embed(pixel_values)
        records.append(stats("patch_embed", hidden))
        pos = (vision.pos_embed(interp_indices).float() * interp_weights[:, :, None]).sum(1)
        records.append(stats("position_embedding", pos))
        hidden = hidden + pos.to(hidden.dtype)
        records.append(stats("after_position_embedding", hidden))

        freqs = position_ids[..., None].float() * vision.rotary_pos_emb.inv_freq.float()
        rust_freqs = torch.cat([freqs[:, 0], freqs[:, 1]], dim=-1)
        records.append(stats("rotary_position_embedding", rust_freqs))
        position_embeddings = vision.rotary_pos_emb(hidden, position_ids)

        hidden = hidden.reshape(hidden.shape[0], -1)
        for layer, block in enumerate(vision.blocks):
            attn = block.attn(
                block.norm1(hidden),
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                max_seqlen=max_seqlen,
            )
            records.append(stats(f"block_{layer:02}_attention", attn))
            hidden = hidden + attn
            records.append(stats(f"block_{layer:02}_after_attention", hidden))
            mlp = block.mlp(block.norm2(hidden))
            records.append(stats(f"block_{layer:02}_mlp", mlp))
            hidden = hidden + mlp
            records.append(stats(f"block_{layer:02}_output", hidden))
            if layer in vision.deepstack_visual_indexes:
                index = vision.deepstack_visual_indexes.index(layer)
                deepstack = vision.deepstack_merger_list[index](hidden)
                records.append(stats(f"deepstack_{index}_after_block_{layer:02}", deepstack))
        records.append(stats("pre_merger", hidden))
        records.append(stats("post_merger", vision.merger(hidden)))

    result = {
        "transformers_version": __import__("transformers").__version__,
        "transformers_git_expected_for_issue_1738": TRANSFORMERS_GIT_FOR_ISSUE_1738,
        "torch_version": torch.__version__,
        "model": str(model_dir),
        "rust_dump": str(rust_dump),
        "image": str(image_path),
        "processor_text": args.processor_text,
        "model_dtype": args.model_dtype,
        "input_dtype": args.input_dtype,
        "grid_thw": grid.tolist(),
        "interpolation_mode": vision.interpolation_mode,
        "interpolation_align_corners": vision.interpolation_align_corners,
        "records": records,
    }
    print(json.dumps(result, indent=2))


def copy_f32(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    dest = args.dest.resolve()
    if dest.exists():
        raise FileExistsError(dest)
    dest.mkdir(parents=True)
    try:
        for path in source.iterdir():
            if path.name == "model.safetensors" or path.is_dir():
                continue
            try:
                os.link(path, dest / path.name)
            except OSError:
                shutil.copy2(path, dest / path.name)
        state: dict[str, torch.Tensor] = {}
        with safe_open(source / "model.safetensors", framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.float()
                state[key] = tensor
        save_file(state, dest / "model.safetensors")
    except Exception:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    print(dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    copy_parser = subparsers.add_parser("copy-f32", help="create a local checkpoint copy with BF16 tensors converted to float32")
    copy_parser.add_argument("--source", type=pathlib.Path, required=True)
    copy_parser.add_argument("--dest", type=pathlib.Path, required=True)
    copy_parser.set_defaults(func=copy_f32)

    compare_parser = subparsers.add_parser("compare", help="compare a Rust stage dump against the transformers vision oracle")
    compare_parser.add_argument("--model", type=pathlib.Path, required=True)
    compare_parser.add_argument("--rust-dump", type=pathlib.Path, required=True)
    compare_parser.add_argument("--model-dtype", choices=["f32", "bf16"], default="f32")
    compare_parser.add_argument("--input-dtype", choices=["f32", "bf16"], default="f32")
    compare_parser.add_argument("--processor-text", default="<|image|>Describe this image.")
    compare_parser.set_defaults(func=compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
