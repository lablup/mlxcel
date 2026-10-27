"""Generate the FLUX.2 Klein parity fixtures from upstream mlx-vlm.

Run with the mlx-vlm oracle (mlx-vlm 0.7.8, git 73d366ec, mlx 0.32.4) on the
checkpoint `black-forest-labs/FLUX.2-klein-4B` (Diffusers tree, bf16):

    MLX_ENABLE_TF32=0 python -I generate_flux2_fixtures.py \
        --model models/mlx/flux.2-klein-4b-bf16 --out tests/fixtures/flux2

Writes, per case, a safetensors file with the initial noise and final latents
(both bf16, exactly as upstream holds them) and the upstream PNG:

- `generate`: "A cinematic portrait of a glass bonsai tree on a moonlit desk",
  512x512, 4 steps, seed 9909 (tiled VAE decode: upstream's `auto` mode tiles
  at this size).
- `transformer_step0_f32`: the transformer alone, in f32, on the first step of
  `generate` (its noise, the bf16 prompt embeddings upstream computed, timestep
  1000), with every parameter upcast and `ModelConfig.precision` set to f32 so
  no upstream cast rounds to bf16. Only the first 128 output tokens are kept.
- `edit`: "Add black sunglasses" on `edit_reference.png` (an upstream 512x512
  generation of a portrait, also written here), seed 123, 4 steps, untiled
  (upstream's edit default).

Upstream's `_match_conv_layout` leaves the VAE encoder's `conv_in` weight
(128, 3, 3, 3) in PyTorch data order because both layouts share that shape, so
upstream's own encoder is wrong (see docs/image-generation.md). `--fix-conv-in`
(on by default) transposes that one tensor after loading, which is what the
mlxcel port computes; `--no-fix-conv-in` reproduces upstream as published.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

import mlx_vlm.models.flux2.pipeline as pipeline_mod
from mlx_vlm.models.flux2.pipeline import Flux2Image, Flux2ImageEdit, _to_pil
from mlx_vlm.models.flux2.constants import ModelConfig
from mlx_vlm.models.flux2.weights import load_vae as upstream_load_vae
from mlx.utils import tree_map

PROMPT = "A cinematic portrait of a glass bonsai tree on a moonlit desk"
REFERENCE_PROMPT = "A studio photo portrait of a smiling man with short brown hair, plain grey background"
EDIT_PROMPT = "Add black sunglasses"


class Capture:
    def __init__(self) -> None:
        self.noise = None
        self.final = None
        self.step0 = None


def install_hooks(capture: Capture, fix_conv_in: bool) -> None:
    original_prepare = pipeline_mod.prepare_packed_latents

    def prepare(**kwargs):
        out = original_prepare(**kwargs)
        capture.noise = out[0]
        return out

    pipeline_mod.prepare_packed_latents = prepare

    def load_vae(model_path, *, include_encoder=False):
        vae = upstream_load_vae(model_path, include_encoder=include_encoder)
        if include_encoder and fix_conv_in:
            # The data is (out, in, kh, kw) under an (out, kh, kw, in) shape.
            w = vae.encoder.conv_in.weight
            vae.encoder.conv_in.weight = w.transpose(0, 2, 3, 1)
        original_decode = vae.decode_packed_latents

        def decode_packed_latents(packed, tiling_config=None):
            capture.final = packed
            return original_decode(packed, tiling_config=tiling_config)

        vae.decode_packed_latents = decode_packed_latents
        return vae

    pipeline_mod.load_vae = load_vae


def capture_first_step(gen, capture: Capture) -> None:
    original = gen._predict

    def predict(**kwargs):
        if capture.step0 is None:
            capture.step0 = kwargs
        return original(**kwargs)

    gen._predict = predict


def transformer_f32_case(gen, out: Path, capture: Capture) -> None:
    from safetensors.numpy import save_file

    step0 = capture.step0
    embeds_bf16 = np.ascontiguousarray(np.array(step0["prompt_embeds"].astype(mx.float32)))
    ModelConfig.precision = mx.float32
    try:
        tr = gen.transformer
        tr.update(tree_map(lambda p: p.astype(mx.float32), tr.parameters()))
        pred = gen._predict(
            latents=step0["latents"].astype(mx.float32),
            latent_ids=step0["latent_ids"],
            prompt_embeds=step0["prompt_embeds"].astype(mx.float32),
            text_ids=step0["text_ids"],
            timestep=mx.array(1000.0, dtype=mx.float32),
        )
        pred = np.ascontiguousarray(np.array(pred[:, :128, :].astype(mx.float32)))
    finally:
        ModelConfig.precision = mx.bfloat16
    save_file(
        {
            "embeds_bf16_bits": (embeds_bf16.view(np.uint32) >> 16).astype(np.uint16),
            "pred_first128_f32": pred,
        },
        str(out / "transformer_step0_f32.safetensors"),
    )


def save(out: Path, name: str, capture: Capture, image) -> None:
    from safetensors.numpy import save_file

    def bf16_bytes(a: mx.array) -> np.ndarray:
        # The exact bf16 bits as uint16, in logical (row-major) order. Read the
        # values through f32 (bf16 is exact in f32) rather than `mx.array.view`,
        # which reinterprets the physical buffer of a strided array such as the
        # transposed `pack_latents` output.
        f32 = np.ascontiguousarray(np.array(a.astype(mx.float32), dtype=np.float32))
        return (f32.view(np.uint32) >> 16).astype(np.uint16)

    save_file(
        {
            "noise_bf16_bits": bf16_bytes(capture.noise),
            "final_latents_bf16_bits": bf16_bytes(capture.final),
        },
        str(out / f"{name}.safetensors"),
    )
    image.save(out / f"{name}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fix-conv-in", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--only", choices=["generate", "edit", "all"], default="all")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    capture = Capture()
    install_hooks(capture, args.fix_conv_in)

    if args.only in ("generate", "all"):
        gen = Flux2Image(variant="flux2-klein-4b", model_path=args.model)
        capture_first_step(gen, capture)
        array = gen.generate_array(PROMPT, seed=9909, steps=4, width=512, height=512, guidance=1.0)
        save(out, "generate_512_seed9909", capture, _to_pil(array))
        ref = gen.generate_array(REFERENCE_PROMPT, seed=7, steps=4, width=512, height=512, guidance=1.0)
        _to_pil(ref).save(out / "edit_reference.png")
        transformer_f32_case(gen, out, capture)
        del gen
        mx.clear_cache()

    if args.only in ("edit", "all"):
        editor = Flux2ImageEdit(
            variant="flux2-klein-4b",
            model_path=args.model,
            runtime_config=pipeline_mod.Flux2RuntimeConfig(bucketed_seq_len=False, tiled_vae="off"),
        )
        array = editor.edit_array(
            EDIT_PROMPT, [str(out / "edit_reference.png")], seed=123, steps=4, guidance=1.0
        )
        suffix = "" if args.fix_conv_in else "_upstream_conv_in"
        save(out, f"edit_512_seed123{suffix}", capture, _to_pil(array))


if __name__ == "__main__":
    main()
