"""Generate the Bonsai parity fixtures from upstream mlx-vlm.

Run with the mlx-vlm oracle (mlx-vlm 0.7.8, git 73d366ec, mlx 0.32.4) on the
checkpoint `prism-ml/bonsai-image-ternary-4B-mlx-2bit`:

    MLX_ENABLE_TF32=0 python -I generate_bonsai_fixtures.py \
        --model models/mlx/bonsai-image-ternary-4b-mlx-2bit --out tests/fixtures/bonsai

This runs upstream's own Bonsai pipeline (`BonsaiImage.generate_array`: the
`klein_fast` transformer, `load_text_encoder_4bit`, unbucketed prompts) with
two substitutions, both needed to run it off Apple Silicon or to compare it
with mlxcel:

- `klein_fast` fuses the q/k RMSNorm and RoPE into `mx.fast.metal_kernel`
  kernels, which exist only on Metal. They are replaced by the same
  arithmetic in MLX ops: f32 RMSNorm times the bf16 gain, f32 RoPE over
  interleaved pairs, cast to bf16, laid out `[B, H, S, D]`. Every other
  operation, including the `mx.quantized_matmul` packed linears, is
  upstream's.
- Upstream decodes with `black-forest-labs/FLUX.2-small-decoder` downloaded
  at load time (`bonsai/weights.py::load_vae`), not the checkpoint's `vae/`.
  mlxcel decodes with the checkpoint's `vae/`, so by default the fixture does
  too (`flux2.weights.load_vae(model)`). `--small-decoder` reproduces upstream
  as shipped (it needs the small decoder in the Hugging Face cache).

Writes `generate_512_seed9909.safetensors` (initial noise and final latents as
bf16 bits) and `generate_512_seed9909.png` for "A tiny glass bonsai tree on a
moonlit desk", 512x512, 4 steps, seed 9909, guidance 1.0. With `--step0 DIR`
it also writes the step-0 transformer inputs and output there (prompt
embeddings are 7.5 MiB, so they are not committed).

`--flux2-transformer` swaps `klein_fast` for upstream's FLUX.2
`Flux2Transformer` loaded with Bonsai's packed weights (`nn.quantize` on the
modules that have `.scales`, 2-bit group 128 inferred from the shapes, as
`flux2/weights.py::_apply_weights` does). It also writes
`transformer_step0_f32.safetensors`: that transformer in f32 on step 0's
noise, the first 32 prompt-embedding rows and timestep 1000, keeping the
first 128 output tokens. Every float parameter is upcast, `ModelConfig.precision`
is f32, and each packed linear runs as a dense f32 linear over
`mx.dequantize` of its weights: mlx 0.32.4's CUDA backend fails 2-bit
`quantized_matmul` in f32 for more than one row (`cuGraphAddKernelNode ...
invalid argument`), and the dequantized weights are the values the packed
kernel multiplies by.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_map, tree_unflatten
from PIL import Image

import mlx_vlm.models.bonsai.klein_fast.blocks as blocks
import mlx_vlm.models.bonsai.pipeline as pipeline_mod
from mlx_vlm.models.bonsai.pipeline import BonsaiImage
from mlx_vlm.models.flux2.config import get_variant as flux2_variant
from mlx_vlm.models.flux2.constants import ModelConfig
from mlx_vlm.models.flux2.transformer import Flux2Transformer
from mlx_vlm.models.flux2.weights import _apply_weights, _load_safetensors
from mlx_vlm.models.flux2.weights import load_vae as flux2_load_vae

PROMPT = "A tiny glass bonsai tree on a moonlit desk"


def _norm_rope(x, gain, cos, sin, eps):
    """`[B, S, H, D]` bf16 -> RMSNorm (f32) * gain -> RoPE (f32) -> bf16 BHSD."""
    xf = x.astype(mx.float32)
    inv = mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
    xf = xf * inv * gain.astype(mx.float32)
    b, s, h, d = xf.shape
    pairs = xf.reshape(b, s, h, d // 2, 2)
    x0, x1 = pairs[..., 0], pairs[..., 1]
    c = cos.astype(mx.float32).reshape(1, s, 1, d // 2)
    sn = sin.astype(mx.float32).reshape(1, s, 1, d // 2)
    out = mx.stack([x0 * c - x1 * sn, x1 * c + x0 * sn], axis=-1).reshape(b, s, h, d)
    return out.astype(mx.bfloat16).transpose(0, 2, 1, 3)


def _split_qkv(qkv, spec):
    b, s, _ = qkv.shape
    h, d, dim = spec.num_heads, spec.head_dim, spec.dim
    q = qkv[..., :dim].reshape(b, s, h, d)
    k = qkv[..., dim : 2 * dim].reshape(b, s, h, d)
    v = qkv[..., 2 * dim : 3 * dim].reshape(b, s, h, d)
    return q, k, v


def install_unfused_kernels() -> None:
    def single(spec, seq_len):
        def kernel(*, inputs, **_):
            fused, norm_q, norm_k, cos, sin = inputs
            q, k, v = _split_qkv(fused, spec)
            eps = spec.rms_norm_eps
            return [
                _norm_rope(q, norm_q, cos, sin, eps),
                _norm_rope(k, norm_k, cos, sin, eps),
                v.astype(mx.bfloat16).transpose(0, 2, 1, 3),
            ]

        return kernel

    def double(spec, img_seq, txt_seq):
        def kernel(*, inputs, **_):
            img_qkv, txt_qkv, nq, nk, naq, nak, cos, sin = inputs
            iq, ik, iv = _split_qkv(img_qkv, spec)
            tq, tk, tv = _split_qkv(txt_qkv, spec)
            eps = spec.rms_norm_eps
            ct, st = cos[:txt_seq], sin[:txt_seq]
            ci, si = cos[txt_seq:], sin[txt_seq:]

            def cat(t, i):
                return mx.concatenate([t, i], axis=2)

            return [
                cat(_norm_rope(tq, naq, ct, st, eps), _norm_rope(iq, nq, ci, si, eps)),
                cat(_norm_rope(tk, nak, ct, st, eps), _norm_rope(ik, nk, ci, si, eps)),
                cat(
                    tv.astype(mx.bfloat16).transpose(0, 2, 1, 3),
                    iv.astype(mx.bfloat16).transpose(0, 2, 1, 3),
                ),
            ]

        return kernel

    blocks._get_fused_single_norm_rope_kernel = single
    blocks._get_fused_double_norm_rope_kernel = double


class Capture:
    def __init__(self) -> None:
        self.noise = None
        self.final = None
        self.step0 = None
        self.pred0 = None


def install_hooks(capture: Capture, model: Path, small_decoder: bool) -> None:
    original_prepare = pipeline_mod.prepare_packed_latents

    def prepare(**kwargs):
        out = original_prepare(**kwargs)
        capture.noise = out[0]
        return out

    pipeline_mod.prepare_packed_latents = prepare
    original_load_vae = pipeline_mod.load_vae

    def load_vae():
        vae = original_load_vae() if small_decoder else flux2_load_vae(model)
        original_decode = vae.decode_packed_latents

        def decode_packed_latents(packed, tiling_config=None):
            capture.final = packed
            return original_decode(packed, tiling_config=tiling_config)

        vae.decode_packed_latents = decode_packed_latents
        return vae

    pipeline_mod.load_vae = load_vae


def flux2_transformer(model: Path) -> Flux2Transformer:
    raw, _ = _load_safetensors(model / "transformer-packed-mflux")
    weights = {}
    for key, value in raw.items():
        mapped = key.replace(
            "time_guidance_embed.timestep_embedder.", "time_guidance_embed."
        ).replace(".to_out.0.", ".to_out.")
        if mx.issubdtype(value.dtype, mx.floating):
            value = value.astype(ModelConfig.precision)
        weights[mapped] = value
    transformer = Flux2Transformer(**flux2_variant("flux2-klein-4b").transformer_overrides)
    return _apply_weights(transformer, weights, {})


def transformer_f32_case(model: Path, out: Path, capture: Capture) -> None:
    from safetensors.numpy import save_file

    s = capture.step0
    embeds = s["prompt_embeds"][:, :32]
    ModelConfig.precision = mx.float32
    try:
        tr = flux2_transformer(model)
        dense = []
        for path, module in tr.named_modules():
            if isinstance(module, nn.QuantizedLinear):
                w = mx.dequantize(
                    module.weight,
                    module.scales.astype(mx.float32),
                    module.biases.astype(mx.float32),
                    group_size=module.group_size,
                    bits=module.bits,
                )
                lin = nn.Linear(w.shape[1], w.shape[0], bias="bias" in module)
                lin.weight = w
                if "bias" in module:
                    lin.bias = module.bias
                dense.append((path, lin))
        tr.update_modules(tree_unflatten(dense))
        tr.update(
            tree_map(
                lambda p: p.astype(mx.float32) if mx.issubdtype(p.dtype, mx.floating) else p,
                tr.parameters(),
            )
        )
        pred = tr(
            hidden_states=s["latents"].astype(mx.float32),
            encoder_hidden_states=embeds.astype(mx.float32),
            timestep=mx.array(1000.0, dtype=mx.float32),
            img_ids=s["latent_ids"],
            txt_ids=s["text_ids"][:, :32],
            guidance=None,
        )
        pred = np.ascontiguousarray(np.array(pred[:, :128, :].astype(mx.float32)))
    finally:
        ModelConfig.precision = mx.bfloat16
    save_file(
        {"embeds_bf16_bits": bf16_bits(embeds), "pred_first128_f32": pred},
        str(out / "transformer_step0_f32.safetensors"),
    )


def bf16_bits(a: mx.array) -> np.ndarray:
    # Through f32 (exact for bf16) rather than `view`, which reads the physical
    # buffer of a strided array.
    f32 = np.ascontiguousarray(np.array(a.astype(mx.float32), dtype=np.float32))
    return (f32.view(np.uint32) >> 16).astype(np.uint16)


def main() -> None:
    from safetensors.numpy import save_file

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--small-decoder", action="store_true")
    parser.add_argument("--step0", type=Path, default=None)
    parser.add_argument("--name", default="generate_512_seed9909")
    parser.add_argument("--flux2-transformer", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    install_unfused_kernels()
    capture = Capture()
    install_hooks(capture, args.model, args.small_decoder)
    if args.flux2_transformer:
        pipeline_mod.load_transformer = lambda model_path, precision: flux2_transformer(
            Path(model_path)
        )
    gen = BonsaiImage(model_path=args.model)
    original_predict = gen._predict

    def predict(**kwargs):
        out = original_predict(**kwargs)
        if capture.step0 is None:
            capture.step0 = kwargs
            capture.pred0 = out
        return out

    gen._predict = predict
    image = gen.generate_array(PROMPT, seed=9909, steps=4, width=512, height=512, guidance=1.0)
    save_file(
        {
            "noise_bf16_bits": bf16_bits(capture.noise),
            "final_latents_bf16_bits": bf16_bits(capture.final),
        },
        str(args.out / f"{args.name}.safetensors"),
    )
    Image.fromarray(np.array(image)).save(args.out / f"{args.name}.png")
    if args.step0 is not None:
        args.step0.mkdir(parents=True, exist_ok=True)
        s = capture.step0
        save_file(
            {
                "latents_bf16_bits": bf16_bits(s["latents"]),
                "embeds_bf16_bits": bf16_bits(s["prompt_embeds"]),
                "text_ids_f32": np.array(s["text_ids"].astype(mx.float32)),
                "timestep_f32": np.array(mx.array(s["timestep"]).astype(mx.float32)).reshape(1),
                "pred_bf16_bits": bf16_bits(capture.pred0),
            },
            str(args.step0 / "step0.safetensors"),
        )
    if args.flux2_transformer:
        transformer_f32_case(args.model, args.out, capture)
    print(f"wrote {args.out / args.name}.*")


if __name__ == "__main__":
    main()
