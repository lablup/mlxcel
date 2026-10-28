#!/usr/bin/env python3
"""Write the LLaDA-Image parity oracle from the upstream Python (mlx-vlm).

Runs mlx-vlm 0.7.8 (git 73d366ec) ``LLaDAImagePipeline`` on mlx 0.32.4 and
writes one directory per case into ``--out``: ``tensors.safetensors`` (token
ids, the text backbone's first outputs, the prompt features, the VQ ids and
SigVQ features, the edit source latents, the initial latents, every
stochastic re-noise draw, the first two predictions, the final latents and the
image), ``image.png`` and ``meta.json``. ``src/imagegen/llada_image/
parity_tests.rs`` reads it through ``MLXCEL_LLADA_IMAGE_ORACLE_DIR``. Nothing
is committed: the cases are full resolution.

Two upstream defects need a flag each for the oracle to be meaningful:

- ``--unsorted-gather``: on mlx 0.32.4 CUDA, ``gather_mm(...,
  sorted_indices=True)`` returns wrong rows (rel-RMS about 0.8 against a
  per-row matmul, for random weights too), and the text encoder's MoE uses
  that form. Without the flag the text features are corrupted from the first
  MoE layer and VQ generation fails its own codebook check.
- ``--fix-conv-in``: the VAE ships in torch layout and upstream matches conv
  layouts per tensor against the target shape; ``encoder.conv_in`` is
  ``[128, 3, 3, 3]`` in both layouts, so it is never transposed and the edit
  source is encoded with a scrambled first convolution.

Cases (``MLX_ENABLE_TF32=0``, seed 42, prompt "A red panda serving tea in a
moonlit bamboo forest"; edits use the checkpoint's ``assets/edit_demo.jpg``
with "make it blue")::

    --case turbo_text_1024 --model <turbo>
    --case turbo_vq_512 --model <turbo> --mode vq --size 512x512
    --case turbo_edit --model <turbo> --mode edit --image <jpg> --prompt "make it blue"
    --case base_text_1024 --model <base>
    --case base_vq_512 --model <base> --mode vq --size 512x512
    --case base_edit --model <base> --mode edit --image <jpg> --prompt "make it blue"

each with ``--unsorted-gather --fix-conv-in``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

import mlx_vlm.models.llada_image.pipeline as pl_mod
from mlx_vlm.models.llada_image.pipeline import LLaDAImagePipeline
from mlx_vlm.models.llada_image.scheduler import LLaDAImageScheduler
from mlx_vlm.models.llada_image.text_encoder import LLaDAImageTextEncoder

REC: dict[str, mx.array] = {}
META: dict = {}


def rec(name, value):
    mx.eval(value)
    REC[name] = value


def install_hooks(pipe: LLaDAImagePipeline):
    orig_tok = pipe.tokenize

    def tokenize(prompt):
        ids = orig_tok(prompt)
        META.setdefault("tokenized", []).append({"prompt": prompt, "ids": ids})
        return ids

    pipe.tokenize = tokenize

    orig_encode = pipe._encode_prompt
    count = {"n": 0}

    def encode(prompt):
        out = orig_encode(prompt)
        rec(f"caption_{count['n']}", out)
        count["n"] += 1
        return out

    pipe._encode_prompt = encode

    orig_call = LLaDAImageTextEncoder.__call__
    calls = {"n": 0}

    def te_call(self, *args, **kwargs):
        out = orig_call(self, *args, **kwargs)
        n = calls["n"]
        calls["n"] += 1
        if n < 3:
            rec(f"te_hidden_{n}", out)
            if kwargs.get("position_ids") is not None:
                rec(f"te_pos_{n}", kwargs["position_ids"])
            if args:
                rec(f"te_ids_{n}", args[0])
        return out

    LLaDAImageTextEncoder.__call__ = te_call

    orig_vq = pl_mod.generate_vq_tokens

    def gen_vq(text_encoder, lm_head, ids, uncond, count_, **kw):
        META["vq_ids"] = list(ids)
        META["vq_uncond"] = list(uncond)
        META["vq_count"] = count_
        out = orig_vq(text_encoder, lm_head, ids, uncond, count_, **kw)
        rec("vq_tokens", out)
        return out

    pl_mod.generate_vq_tokens = gen_vq

    orig_sem = pipe._generate_semantic_features

    def sem(*a, **k):
        out = orig_sem(*a, **k)
        rec("semantic", out)
        return out

    pipe._generate_semantic_features = sem

    orig_src = pipe._encode_source_image

    def src(image, height, width):
        pixels = np.array(
            image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS),
            dtype=np.uint8,
        )
        REC["source_pixels_u8"] = mx.array(pixels)
        source, semantic = orig_src(image, height, width)
        rec("source_latents", source)
        rec("source_semantic", semantic)
        return source, semantic

    pipe._encode_source_image = src

    orig_step = LLaDAImageScheduler.step
    steps = {"n": 0}

    def step(self, prediction, sample, sigma, sigma_next, *, noise=None):
        i = steps["n"]
        steps["n"] += 1
        if i == 0:
            rec("initial_latents", sample)
        if self.stochastic_sampling and noise is None:
            noise = mx.random.normal(sample.shape, dtype=mx.float32)
            rec(f"step_noise_{i}", noise)
        if i < 2:
            rec(f"prediction_{i}", prediction)
        out = orig_step(self, prediction, sample, sigma, sigma_next, noise=noise)
        rec(f"latents_{i}", out)
        REC["final_latents"] = out
        return out

    LLaDAImageScheduler.step = step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="A red panda serving tea in a moonlit bamboo forest")
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--mode", default="text", choices=["text", "vq", "edit"])
    ap.add_argument("--image", default=None)
    ap.add_argument("--edit-size", default=None, help="explicit WxH for edit")
    ap.add_argument(
        "--unsorted-gather",
        action="store_true",
        help="force gather_mm(sorted_indices=False): mlx 0.32.4 CUDA returns "
        "wrong rows for the sorted form TextMoE uses (rel-RMS ~0.8 vs per-row matmul)",
    )
    ap.add_argument(
        "--fix-conv-in",
        action="store_true",
        help="transpose the torch-layout VAE encoder.conv_in [128,3,3,3], which "
        "upstream's per-tensor _match_conv_layout leaves untransposed because the "
        "torch and MLX shapes coincide",
    )
    args = ap.parse_args()
    if args.fix_conv_in:
        orig_load_vae = pl_mod.load_vae

        def load_vae(path, *, include_encoder=False):
            vae = orig_load_vae(path, include_encoder=include_encoder)
            if vae.encoder is not None:
                w = vae.encoder.conv_in.weight
                # Torch (O, I, kh, kw) read as MLX (O, kh, kw, I): permute once.
                vae.encoder.conv_in.weight = w.transpose(0, 2, 3, 1)
            return vae

        pl_mod.load_vae = load_vae
        META["fix_conv_in"] = True
    if args.unsorted_gather:
        orig_gmm = mx.gather_mm

        def gather_mm(*a, **kw):
            kw["sorted_indices"] = False
            return orig_gmm(*a, **kw)

        mx.gather_mm = gather_mm
        META["unsorted_gather"] = True
    out = Path(args.out) / args.case
    out.mkdir(parents=True, exist_ok=True)
    pipe = LLaDAImagePipeline(args.model)
    install_hooks(pipe)
    t0 = time.time()
    if args.mode == "edit":
        kw = {}
        if args.edit_size:
            w, h = (int(v) for v in args.edit_size.split("x"))
            kw = {"width": w, "height": h}
        image = pipe.edit_array(
            args.prompt, args.image, seed=args.seed, steps=args.steps,
            guidance=args.guidance, **kw,
        )
    else:
        w, h = (int(v) for v in args.size.split("x"))
        image = pipe.generate_array(
            args.prompt, seed=args.seed, steps=args.steps, width=w, height=h,
            guidance=args.guidance, generation_mode=args.mode,
        )
    elapsed = time.time() - t0
    arr = np.array(image)
    Image.fromarray(arr).save(out / "image.png")
    REC["image"] = mx.array(arr)
    mx.save_safetensors(str(out / "tensors.safetensors"), REC)
    META.update(
        {
            "case": args.case,
            "model": args.model,
            "prompt": args.prompt,
            "seed": args.seed,
            "mode": args.mode,
            "size": [int(arr.shape[1]), int(arr.shape[0])],
            "steps": args.steps,
            "guidance": args.guidance,
            "elapsed_s": elapsed,
            "mlx": mx.__version__,
            "device": str(mx.default_device()),
            "peak_gb": mx.get_peak_memory() / 1e9,
        }
    )
    (out / "meta.json").write_text(json.dumps(META, indent=1))
    print(f"{args.case}: done in {elapsed:.1f}s -> {out}")


if __name__ == "__main__":
    main()
