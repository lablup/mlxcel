#!/usr/bin/env python3
"""Generate the Ideogram 4 parity fixtures from the upstream Python (mlx-vlm).

Writes into ``--out`` one safetensors file and one PNG per case (the token
ids, the text part of the LLM features, the initial noise, the conditional
and unconditional transformer outputs on the first step's inputs, the final
latents and the decoded image), and a compact summary,
``ideogram4_parity.safetensors``, that is committed as
``tests/fixtures/imagegen/ideogram4_parity.safetensors``. Per case the summary
keeps the token ids, the initial-noise FNV-1a checksum (over the raw f32
bytes) and head, the final latents averaged over 4x4 image-token patches, and
the 8x8 average-pooled image. ``src/imagegen/ideogram4/parity_tests.rs`` reads
the summary, and the full files when ``MLXCEL_IDEOGRAM4_ORACLE_DIR`` points
at ``--out``.

Case ``turbo_512``: the issue's red-square JSON caption, 512x512,
``sampler_preset`` ``V4_TURBO_12`` (12 steps, the preset's guidance
schedule), seed 0.

The denoising loop is upstream's ``Ideogram4ImagePipeline.generate_array``
driven step by step so the intermediate tensors can be recorded; every
tensor comes from upstream's own modules.

Provenance of the committed file: mlx 0.32.4 (CUDA backend, NVIDIA GB10),
mlx-vlm 0.7.8 at git 73d366ecf9233b3fc6549531a7f2386726d09ff6, run with
``MLX_ENABLE_TF32=0``.

Usage::

    MLX_ENABLE_TF32=0 python -I tests/fixtures/generate_ideogram4_fixtures.py \\
        --out /tmp/ideogram4 --model models/mlx/ideogram-4-fp8

``--summary-only`` rebuilds the summary from the case files already in
``--out`` without loading a model.

``--tiny PATH`` instead writes ``ideogram4_tiny_refs.safetensors``
(committed as ``tests/fixtures/imagegen/ideogram4_tiny_refs.safetensors``):
a two-layer ``Ideogram4Transformer`` with random bf16 weights (``w.*``), its
inputs, and upstream's conditional and unconditional outputs, used by
``src/imagegen/ideogram4/transformer_tests.rs`` without the checkpoint.
"""

import argparse
import gc
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

PROMPT = (
    '{"high_level_description":"A red square icon.","compositional_deconstruction":'
    '{"background":"A white background.","elements":[{"type":"obj","desc":"A centered red square."}]}}'
)
SEED = 0
CASES = {"turbo_512": dict(width=512, height=512, preset="V4_TURBO_12")}
POOL = 8
TOKEN_POOL = 4


def fnv1a32_bytes(data: bytes) -> int:
    h = 0x811C9DC5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def run_case(p, out: Path, case: str, *, width: int, height: int, preset: str) -> dict:
    from mlx_vlm.models.ideogram4.pipeline import LATENT_DIM
    from mlx_vlm.models.ideogram4.scheduler import (
        get_preset,
        get_schedule_for_resolution,
        make_step_intervals,
    )

    t0 = time.time()
    prepared = p.prepare_prompt(PROMPT, width=width, height=height)
    inputs = p._build_inputs(prepared.text, height=height, width=width)
    n_text = inputs["num_text_tokens"]
    llm = p._encode_text(inputs["text_token_ids"], num_image_tokens=inputs["num_image_tokens"])
    mx.eval(llm)
    rec = {
        "token_ids": inputs["text_token_ids"].reshape(-1).astype(mx.int32),
        "llm_features_text": llm[:, :n_text],
    }
    mx.eval(rec)
    p.text_encoder = None
    gc.collect()
    mx.clear_cache()
    p._ensure_transformers_and_vae()

    sp = get_preset(preset)
    steps = sp.num_steps
    schedule = get_schedule_for_resolution((height, width), known_mean=sp.mu, std=sp.std)
    intervals = make_step_intervals(steps)
    guidance_schedule = sp.guidance_schedule

    mx.random.seed(SEED)
    z = mx.random.normal((1, inputs["num_image_tokens"], LATENT_DIM), dtype=mx.float32)
    mx.eval(z)
    rec["noise"] = z
    text_z_padding = mx.zeros((1, n_text, LATENT_DIM), dtype=mx.float32)
    neg_pos = inputs["position_ids"][:, n_text:]
    neg_seg = inputs["segment_ids"][:, n_text:]
    neg_ind = inputs["indicator"][:, n_text:]
    for i in range(steps - 1, -1, -1):
        t_val = schedule(intervals[i + 1])
        s_val = schedule(intervals[i])
        t = mx.full((1,), t_val, dtype=mx.float32)
        pos_z = mx.concatenate([text_z_padding, z], axis=1)
        pos_out = p.conditional_transformer(
            llm_features=llm,
            x=pos_z,
            t=t,
            position_ids=inputs["position_ids"],
            segment_ids=inputs["segment_ids"],
            indicator=inputs["indicator"],
        )
        pos_v = pos_out[:, n_text:]
        neg_v = p.unconditional_transformer(
            llm_features=None,
            x=z,
            t=t,
            position_ids=neg_pos,
            segment_ids=neg_seg,
            indicator=neg_ind,
        )
        if i == steps - 1:
            mx.eval(pos_v, neg_v)
            rec["step0_pos"] = pos_v
            rec["step0_neg"] = neg_v
        gw = guidance_schedule[i]
        velocity = gw * pos_v + (1.0 - gw) * neg_v
        z = z + velocity * (s_val - t_val)
        mx.eval(z)
    rec["final_latents"] = z
    rec["image"] = p._decode(z, grid_h=inputs["grid_h"], grid_w=inputs["grid_w"])
    mx.eval(rec)
    mx.save_safetensors(str(out / f"{case}.safetensors"), rec)
    Image.fromarray(np.array(rec["image"])).save(out / f"{case}.png")
    print(f"{case}: {width}x{height}, {preset}, {n_text} text tokens, {time.time() - t0:.1f}s", flush=True)
    return rec


def summarize(summary: dict, case: str, rec: dict, width: int, height: int) -> None:
    noise_bytes = np.array(rec["noise"]).tobytes()
    summary[f"{case}.token_ids"] = rec["token_ids"]
    summary[f"{case}.noise_fnv"] = mx.array(np.array([fnv1a32_bytes(noise_bytes)], dtype=np.uint32))
    summary[f"{case}.noise_head"] = rec["noise"].reshape(-1)[:64].astype(mx.float32)
    gh, gw = height // 16, width // 16
    final = np.array(rec["final_latents"].astype(mx.float32)).reshape(gh, gw, -1)
    pooled = final.reshape(gh // TOKEN_POOL, TOKEN_POOL, gw // TOKEN_POOL, TOKEN_POOL, -1).mean(axis=(1, 3))
    summary[f"{case}.latents_pool4"] = mx.array(pooled.astype(np.float32))
    image = np.array(rec["image"]).astype(np.float64)
    h, w, c = image.shape
    pooled_image = image.reshape(h // POOL, POOL, w // POOL, POOL, c).mean(axis=(1, 3))
    summary[f"{case}.image_pool8"] = mx.array(pooled_image.round().astype(np.uint8))


def tiny(path: Path) -> None:
    from mlx.utils import tree_flatten, tree_unflatten

    from mlx_vlm.models.ideogram4.config import Ideogram4TransformerConfig
    from mlx_vlm.models.ideogram4.pipeline import IMAGE_POSITION_OFFSET
    from mlx_vlm.models.ideogram4.transformer import Ideogram4Transformer

    config = Ideogram4TransformerConfig(
        emb_dim=32, num_layers=2, num_heads=2, intermediate_size=48, adanln_dim=16,
        in_channels=8, llm_features_dim=24, rope_theta=10000, mrope_section=(4, 2, 2), norm_eps=1e-5,
    )
    model = Ideogram4Transformer(config)
    params = {}
    for i, (name, value) in enumerate(tree_flatten(model.parameters())):
        if name.startswith("rotary_emb."):
            continue  # inv_freq and axis_selector are buffers, not weights
        noise = 0.2 * mx.random.normal(value.shape, key=mx.random.key(2427 + i))
        # Norm weights scatter around 1, everything else around 0.
        params[name] = (noise + (1.0 if "norm" in name else 0.0)).astype(mx.bfloat16)
    model.update(tree_unflatten(list(params.items())))
    n_text, gh, gw = 3, 2, 3
    text = mx.arange(n_text, dtype=mx.int32)
    rows = mx.repeat(mx.arange(gh, dtype=mx.int32), gw)
    cols = mx.tile(mx.arange(gw, dtype=mx.int32), gh)
    image = mx.stack([mx.zeros_like(rows), rows, cols], axis=1) + IMAGE_POSITION_OFFSET
    position_ids = mx.concatenate([mx.stack([text, text, text], axis=1), image], axis=0)[None]
    indicator = mx.array([[3] * n_text + [2] * (gh * gw)], dtype=mx.int32)
    segment_ids = mx.ones(indicator.shape, dtype=mx.int32)
    k1, k2 = mx.random.split(mx.random.key(7))
    x = mx.random.normal((1, n_text + gh * gw, 8), key=k1)
    llm = mx.random.normal((1, n_text + gh * gw, 24), key=k2)
    t = mx.array([0.73], dtype=mx.float32)
    cond = model(llm_features=llm, x=x, t=t, position_ids=position_ids, segment_ids=segment_ids, indicator=indicator)
    uncond = model(
        llm_features=None, x=x[:, n_text:], t=t, position_ids=position_ids[:, n_text:],
        segment_ids=segment_ids[:, n_text:], indicator=indicator[:, n_text:],
    )
    cos, sin = model.rotary_emb(position_ids)
    out = {f"w.{k}": v for k, v in params.items()}
    out.update(
        x=x, llm=llm, t=t, position_ids=position_ids, indicator=indicator, cond=cond, uncond=uncond,
        cos=cos, sin=sin, selector=model.rotary_emb.axis_selector,
    )
    mx.eval(out)
    mx.save_safetensors(str(path), out)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/mlx/ideogram-4-fp8"))
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--tiny", type=Path, help="write the tiny transformer reference instead")
    args = parser.parse_args()
    if args.tiny is not None:
        tiny(args.tiny)
        return
    if args.out is None:
        parser.error("--out is required")
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    if args.summary_only:
        for case, spec in CASES.items():
            rec = mx.load(str(args.out / f"{case}.safetensors"))
            summarize(summary, case, rec, spec["width"], spec["height"])
    else:
        from mlx_vlm.models.ideogram4.pipeline import Ideogram4ImagePipeline

        p = Ideogram4ImagePipeline(model_path=args.model)
        for case, spec in CASES.items():
            rec = run_case(p, args.out, case, **spec)
            summarize(summary, case, rec, spec["width"], spec["height"])
    mx.eval(summary)
    mx.save_safetensors(str(args.out / "ideogram4_parity.safetensors"), summary)


if __name__ == "__main__":
    main()
