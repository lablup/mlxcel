#!/usr/bin/env python3
"""Generate the ERNIE-Image parity fixtures from the upstream Python (mlx-vlm).

Writes into ``--out`` one safetensors file and one PNG per case (the token ids,
text conditioning, initial noise, the first transformer prediction on the
first step's inputs, the final latents and the decoded image), and a compact
summary, ``ernie_image_parity.safetensors``, that is committed as
``tests/fixtures/imagegen/ernie_image_parity.safetensors``. Per case the
summary keeps the initial-noise FNV-1a checksum (over the raw bf16 bytes) and
head, the 8x8 average-pooled final latents and the 8x8 average-pooled image.
``src/imagegen/ernie_image/parity_tests.rs`` reads the summary, and the full
files when ``MLXCEL_ERNIE_IMAGE_ORACLE_DIR`` points at ``--out``.

Cases (prompt "A red panda serving tea in a moonlit bamboo forest", seed 42,
prompt enhancer off):

* ``turbo_1024``: ERNIE-Image-Turbo, 1024x1024, 8 steps, guidance 1.
* ``turbo_edit_1024``: ERNIE-Image-Turbo SDEdit of the ``turbo_1024`` image
  at 1024x1024, 8 steps, guidance 3, image_strength 0.7.
* ``base_1024``: ERNIE-Image, 1024x1024, 50 steps, guidance 4, empty negative
  prompt (classifier-free guidance on).

All cases run at 1024x1024, the issue's size. At batch 1 mlxcel reproduces
these bits exactly; the batch-2 (CFG) cases differ in the last bit of one
small-M GEMM between the two MLX builds, see
``src/imagegen/ernie_image/parity_tests.rs``.

Provenance of the committed file: mlx 0.32.4 (CUDA backend, NVIDIA GB10),
mlx-vlm 0.7.8 at git 73d366ecf9233b3fc6549531a7f2386726d09ff6, run with
``MLX_ENABLE_TF32=0``.

Usage::

    MLX_ENABLE_TF32=0 python -I tests/fixtures/generate_ernie_image_fixtures.py \\
        --out /tmp/ernie --models models/mlx

``--summary-only`` rebuilds the summary from the case files already in
``--out`` without loading a model.
"""

import argparse
import gc
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

PROMPT = "A red panda serving tea in a moonlit bamboo forest"
SEED = 42


def fnv1a32_bytes(data: bytes) -> int:
    h = 0x811C9DC5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


POOL = 8
CASES = ("turbo_1024", "turbo_edit_1024", "base_1024")


def pool(a: np.ndarray, axes: tuple[int, int]) -> np.ndarray:
    """Average-pool two axes by POOL."""
    a = np.moveaxis(a, axes, (-2, -1))
    *lead, h, w = a.shape
    a = a.reshape(*lead, h // POOL, POOL, w // POOL, POOL).mean(axis=(-3, -1))
    return np.moveaxis(a, (-2, -1), axes)


def pipeline(models: Path, name: str):
    from mlx_vlm.models.ernie_image.pipeline import ErnieImagePipeline, ErnieImageRuntimeConfig

    return ErnieImagePipeline(
        variant=name,
        model_path=models / name,
        runtime_config=ErnieImageRuntimeConfig(evict_text_encoder=False, use_prompt_enhancer=False),
    )


def run_case(p, out: Path, case: str, *, size, steps, guidance, edit_src=None, strength=0.6) -> dict:
    from mlx_vlm.models.ernie_image.pipeline import _load_edit_image
    from mlx_vlm.models.ernie_image.scheduler import ErnieImageFlowMatchScheduler
    from mlx_vlm.models.flux2.latent import patchify_latents

    t0 = time.time()
    width, height = size
    prompts = ["", PROMPT] if guidance > 1.0 else [PROMPT]
    rec = {f"ids_{i}": p.tokenizer.encode(s).astype(mx.int32) for i, s in enumerate(prompts)}
    rec.update({f"text_{i}": p._encode_prompt(s) for i, s in enumerate(prompts)})
    text, lengths = p._encode_prompts(prompts)
    p._ensure_components(require_encoder=edit_src is not None)
    sched = ErnieImageFlowMatchScheduler(num_inference_steps=steps)
    start = 0
    if edit_src is None:
        noise = mx.random.normal((1, 128, height // 16, width // 16), key=mx.random.key(SEED), dtype=mx.bfloat16)
        latents = noise
    else:
        pixels, width, height = _load_edit_image(edit_src, width=width, height=height)
        src = patchify_latents(p.vae.encode(pixels))
        mean = p.vae.bn.running_mean.reshape(1, -1, 1, 1).astype(src.dtype)
        std = mx.sqrt(p.vae.bn.running_var.reshape(1, -1, 1, 1) + 1e-5).astype(src.dtype)
        src = (src - mean) / std
        rec["source_latents"] = src
        noise = mx.random.normal(src.shape, key=mx.random.key(SEED), dtype=src.dtype)
        start = steps - max(1, min(steps, round(steps * strength)))
        sigma = sched.sigmas[start].astype(src.dtype)
        latents = (1.0 - sigma) * src + sigma * noise
    mx.eval(latents)
    rec["noise"] = noise
    rec["init_latents"] = latents
    rec["text_padded"] = text
    rec["text_lengths"] = lengths
    rec["start_index"] = mx.array([start], dtype=mx.int32)
    batch_in = mx.concatenate([latents, latents], axis=0) if guidance > 1.0 else latents
    t = mx.full((batch_in.shape[0],), float(sched.timesteps[start].item()), dtype=latents.dtype)
    rec["step0_prediction"] = p.transformer(batch_in, timestep=t, text_hidden_states=text, text_lengths=lengths)
    final = p._denoise(
        latents=latents,
        scheduler=sched,
        start_index=start,
        text_hidden_states=text,
        text_lengths=lengths,
        guidance=guidance,
    )
    rec["final_latents"] = final
    rec["image"] = p._decode(final)
    mx.eval(rec)
    mx.save_safetensors(str(out / f"{case}.safetensors"), rec)
    Image.fromarray(np.array(rec["image"])).save(out / f"{case}.png")
    print(f"{case}: {width}x{height}, {steps} steps, guidance {guidance}, start {start}, {time.time() - t0:.1f}s", flush=True)
    return rec


def summarize(summary: dict, case: str, rec: dict) -> None:
    noise_bytes = np.array(rec["noise"].view(mx.uint16)).tobytes()
    summary[f"{case}.start_fnv"] = mx.array(np.array([fnv1a32_bytes(noise_bytes)], dtype=np.uint32))
    summary[f"{case}.start_head"] = rec["noise"].reshape(-1)[:64].astype(mx.float32)
    final = np.array(rec["final_latents"].astype(mx.float32))
    summary[f"{case}.latents_pool8"] = mx.array(pool(final, (2, 3)).astype(np.float32))
    image = np.array(rec["image"]).astype(np.float64)
    summary[f"{case}.image_pool8"] = mx.array(pool(image, (0, 1)).round().astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=Path("models/mlx"))
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    if args.summary_only:
        for case in CASES:
            summarize(summary, case, mx.load(str(args.out / f"{case}.safetensors")))
        mx.save_safetensors(str(args.out / "ernie_image_parity.safetensors"), summary)
        return

    p = pipeline(args.models, "ernie-image-turbo")
    rec = run_case(p, args.out, "turbo_1024", size=(1024, 1024), steps=8, guidance=1.0)
    summarize(summary, "turbo_1024", rec)
    src = args.out / "turbo_1024.png"
    rec = run_case(p, args.out, "turbo_edit_1024", size=(1024, 1024), steps=8, guidance=3.0, edit_src=src, strength=0.7)
    summarize(summary, "turbo_edit_1024", rec)
    del p, rec
    gc.collect()
    mx.clear_cache()

    p = pipeline(args.models, "ernie-image")
    rec = run_case(p, args.out, "base_1024", size=(1024, 1024), steps=50, guidance=4.0)
    summarize(summary, "base_1024", rec)
    mx.eval(summary)
    mx.save_safetensors(str(args.out / "ernie_image_parity.safetensors"), summary)


if __name__ == "__main__":
    main()
