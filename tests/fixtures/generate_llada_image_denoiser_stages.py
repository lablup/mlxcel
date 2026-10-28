#!/usr/bin/env python3
"""Write upstream's per-stage LLaDA-Image denoiser tensors for one oracle case.

Reads ``<case>/tensors.safetensors`` written by
``generate_llada_image_oracle.py`` (text or VQ mode), runs upstream mlx-vlm 0.7.8
(git 73d366ec) ``LLaDAImageTransformer`` once on that case's first-step
inputs (the initial latents, the first sigma, ``caption_0`` and, for a VQ
case, the SigVQ ``semantic`` features) and writes
``<case>/denoiser_stages.safetensors``: the input and output of every stage
(x/cap embedders, RoPE tables, t embedder, each noise/context refiner and main
block, the final layer) and upstream's own whole-pass output.
``transformer_stage_tests.rs`` feeds each mlxcel stage upstream's input to
that stage and compares only its output. The file is about 2 GB for Turbo
text 1024 and is not committed.

    MLX_ENABLE_TF32=0 python -I generate_llada_image_denoiser_stages.py \\
        --model models/mlx/llada-image-turbo --case <oracle>/turbo_text_1024
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mlx.core as mx
from mlx import nn

from mlx_vlm.models.llada_image.scheduler import LLaDAImageScheduler
from mlx_vlm.models.llada_image.transformer import (
    image_positions,
    pad_tokens,
    padded_positions,
    sequence_positions,
)
from mlx_vlm.models.llada_image.weights import load_transformer

REC: dict[str, mx.array] = {}


def rec(name: str, value: mx.array) -> mx.array:
    mx.eval(value)
    REC[name] = value
    return value


def run_block(block, x, cos, sin, adaln, tag):
    rec(f"{tag}.in", x)
    out = block(x, cos, sin, adaln if block.modulation else None)
    return rec(f"{tag}.out", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--case", required=True, help="an oracle text or VQ case directory")
    args = ap.parse_args()
    model, case = Path(args.model), Path(args.case)
    full = mx.load(str(case / "tensors.safetensors"))
    config = json.loads((model / "scheduler" / "scheduler_config.json").read_text())
    scheduler = LLaDAImageScheduler(config)
    tr = load_transformer(model)
    dtype = tr.x_embedder.weight.dtype
    t = rec("in.t", scheduler.sigmas(scheduler.default_steps)[0:1].astype(dtype))
    x = rec("in.x", full["initial_latents"].astype(dtype))
    caption = rec("in.caption", full["caption_0"].astype(dtype))
    semantic = full.get("semantic")
    if semantic is not None:
        semantic = rec("in.semantic", semantic.astype(dtype))
    reference = rec("ref.out", tr(x, t, caption, semantic=semantic))

    # Upstream's text-mode `__call__`, one stage at a time.
    batch, channels, height, width = x.shape
    length = height * width
    patches = x.transpose(0, 2, 3, 1).reshape(batch, length, channels)
    image_tokens = pad_tokens(rec("x_emb", tr.x_embedder(patches)), tr.x_pad_token)
    normed = rec("cap_norm", tr.cap_embedder[0](caption))
    caption_tokens = rec(
        "cap_tokens", pad_tokens(rec("cap_emb", tr.cap_embedder[1](normed)), tr.cap_pad_token)
    )
    cap_pos, image_pos = padded_positions(caption.shape[1], height, width)
    if semantic is not None:
        features = rec("sem_emb", tr.semantic_embedder[1](rec("sem_norm", tr.semantic_embedder[0](semantic))))
        features = pad_tokens(features, tr.cap_pad_token)
        semantic_pos = sequence_positions(semantic.shape[1], caption_tokens.shape[1] + 1)
        caption_tokens = rec("cap_tokens_sem", mx.concatenate([caption_tokens, features], axis=1))
        cap_pos = mx.concatenate([cap_pos, semantic_pos], axis=1)
        image_pos = image_positions(height, width, caption_tokens.shape[1] + 1)
    cap_cos, cap_sin = tr.rope.compute_freqs(cap_pos)
    image_cos, image_sin = tr.rope.compute_freqs(image_pos)
    for name, value in [
        ("cap_cos", cap_cos),
        ("cap_sin", cap_sin),
        ("image_cos", image_cos),
        ("image_sin", image_sin),
    ]:
        rec(name, value)
    time = rec("t.out", tr.t_embedder(t * tr.config.t_scale))
    for i, block in enumerate(tr.noise_refiner):
        image_tokens = run_block(block, image_tokens, image_cos, image_sin, time, f"noise_refiner.{i}")
    for i, block in enumerate(tr.context_refiner):
        caption_tokens = run_block(block, caption_tokens, cap_cos, cap_sin, None, f"context_refiner.{i}")
    tokens = mx.concatenate([image_tokens, caption_tokens], axis=1)
    cos = mx.concatenate([image_cos, cap_cos], axis=1)
    sin = mx.concatenate([image_sin, cap_sin], axis=1)
    for i, block in enumerate(tr.layers):
        tokens = run_block(block, tokens, cos, sin, time, f"layers.{i}")
    image = rec("final.in", tokens[:, :length])
    output = rec("final.out", tr.final_layer(image, time))
    output = output.reshape(batch, height, width, channels).transpose(0, 3, 1, 2)
    if not mx.array_equal(output, reference).item():
        raise SystemExit("the staged pass does not reproduce upstream's call")
    prediction = -reference.astype(mx.float32)
    diff = prediction - full["prediction_0"]
    rel = math.sqrt(mx.mean(diff * diff).item() / mx.mean(full["prediction_0"] ** 2).item())
    print(f"staged pass == upstream call; vs oracle prediction_0 rel-RMS {rel:.3e}")
    out = case / "denoiser_stages.safetensors"
    mx.save_safetensors(str(out), REC)
    print(f"wrote {out} ({len(REC)} tensors)")


if __name__ == "__main__":
    main()
