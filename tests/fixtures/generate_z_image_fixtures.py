#!/usr/bin/env python3
"""Generate the Z-Image parity fixtures from the upstream Python (mlx-vlm).

Two outputs, both compared against by ``src/imagegen/z_image/*_tests.rs``:

* ``tiny``: ``tests/fixtures/imagegen/z_image_tiny_refs.safetensors``, the
  upstream ``ZImageTransformer`` outputs for a tiny random-weight config.
  Weights are not stored: they come from the closed-form ``det`` / ``fill``
  generator of ``generate_imagegen_fixtures.py``, which the Rust test mirrors.
* ``real``: ``tests/fixtures/imagegen/z_image_parity.safetensors``, a compact
  summary of seed-fixed runs of the real checkpoints (``turbo``: Z-Image-Turbo
  512x512, 8 steps, seed 42; ``base``: Z-Image 512x512, 50 steps, guidance 4,
  negative prompt "blurry, low quality", cfg_truncation 1.0; ``edit``:
  Z-Image-Turbo img2img of a Turbo-generated fox at strength 0.6, 8 steps,
  seed 42). Per case it keeps the initial-noise FNV-1a checksum and head, the
  4x4 average-pooled final latents, and the 4x4 average-pooled image. The
  full-resolution tensors and PNGs stay in ``--out`` and are read by the
  ignored real-checkpoint test when ``MLXCEL_Z_IMAGE_ORACLE_DIR`` points there.

Provenance of the committed files: mlx 0.32.4 (CUDA backend, NVIDIA GB10),
mlx-vlm 0.7.8 at git 73d366ecf9233b3fc6549531a7f2386726d09ff6, run with
``MLX_ENABLE_TF32=0``.

Usage::

    python -I tests/fixtures/generate_z_image_fixtures.py tiny --out /tmp/zi
    python -I tests/fixtures/generate_z_image_fixtures.py real --out /tmp/zi \\
        --turbo models/mlx/z-image-turbo-bf16 --base models/mlx/z-image-bf16
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

PROMPT = "A red panda serving tea in a moonlit bamboo forest"
FOX_PROMPT = "a red fox sitting in a snowy forest, photograph"
EDIT_PROMPT = "the same fox at night under a full moon"


def _shared():
    path = Path(__file__).with_name("generate_imagegen_fixtures.py")
    spec = importlib.util.spec_from_file_location("imagegen_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_tiny(out: Path) -> None:
    from mlx_vlm.models.z_image.config import ZImageTransformerConfig
    from mlx_vlm.models.z_image.transformer import ZImageTransformer

    shared = _shared()
    config = ZImageTransformerConfig(
        hidden_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=48,
        in_channels=4,
        text_embed_dim=12,
        num_hidden_layers=2,
        n_refiner_layers=1,
        n_context_refiner_layers=1,
        patch_size=2,
        f_patch_size=1,
        adaln_embed_dim=24,
        rope_sections=(4, 6, 6),
        rope_theta=256.0,
        norm_eps=1e-5,
        timestep_scale=1000.0,
    )
    model = ZImageTransformer(config)
    shared.fill(model, "zt")
    refs = {}
    # 15 image tokens and 5 caption tokens: both streams get padded to 32.
    x = shared.det("latents", (1, 4, 1, 6, 10))
    cap = shared.det("cap", (1, 5, 12))
    t = mx.array(np.array([0.3], dtype=np.float32))
    refs["out"] = model(x, t, cap)
    # Batch 2 with a shared caption; 4 image tokens.
    x2 = shared.det("latents_b2", (2, 4, 1, 4, 4))
    refs["out_b2"] = model(x2, mx.array(np.array([0.75], dtype=np.float32)), cap)
    mx.eval(list(refs.values()))
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "z_image_tiny_refs.safetensors"), refs)


def fnv1a32_bytes(data: bytes) -> int:
    h = 0x811C9DC5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def pool4(a: np.ndarray, axes: tuple[int, int]) -> np.ndarray:
    """Average-pool two axes by 4."""
    a = np.moveaxis(a, axes, (-2, -1))
    *lead, h, w = a.shape
    a = a.reshape(*lead, h // 4, 4, w // 4, 4).mean(axis=(-3, -1))
    return np.moveaxis(a, (-2, -1), axes)


def run_case(case: str, model_dir: Path, out: Path) -> dict:
    from PIL import Image

    from mlx_vlm.models.z_image.pipeline import ZImagePipeline

    pipe = ZImagePipeline(model_dir)
    rec = {}
    original = pipe._denoise

    def denoise(latents, **kw):
        rec["start_latents"] = latents
        sched = kw["scheduler"]
        t = (1.0 - sched.sigmas[kw["start_index"]]).reshape(1)
        rec["first_pred"] = pipe.transformer(latents, t, kw["cap_feats"])
        rec["cap0"] = kw["cap_feats"]
        mx.eval(rec["first_pred"])
        rec["final_latents"] = original(latents, **kw)
        return rec["final_latents"]

    pipe._denoise = denoise
    start = time.time()
    if case == "turbo":
        img = pipe.generate_array(PROMPT, seed=42, steps=8, width=512, height=512, guidance=0.0)
    elif case == "base":
        img = pipe.generate_array(
            PROMPT, seed=42, steps=50, width=512, height=512, guidance=4.0,
            negative_prompt="blurry, low quality", cfg_truncation=1.0,
        )
    elif case == "fox":
        img = pipe.generate_array(FOX_PROMPT, seed=7, steps=8, width=512, height=512)
    elif case == "edit":
        img = pipe.edit_array(
            EDIT_PROMPT, [str(out / "fox.png")], seed=42, steps=8, guidance=0.0, strength=0.6
        )
    else:
        raise SystemExit(f"unknown case {case}")
    mx.eval(img)
    elapsed = time.time() - start
    Image.fromarray(np.array(img)).save(out / f"{case}.png")
    tensors = {k: v.astype(mx.float32) for k, v in rec.items()}
    mx.save_safetensors(str(out / f"{case}.safetensors"), tensors)
    return {"case": case, "elapsed_s": elapsed, "peak_gb": mx.get_peak_memory() / 1e9}


def write_real(out: Path, turbo: Path, base: Path, reuse: bool) -> None:
    from PIL import Image

    if not reuse:
        meta = [run_case(c, turbo, out) for c in ("turbo", "fox", "edit")]
        meta.append(run_case("base", base, out))
        (out / "z_image_parity.json").write_text(json.dumps(meta, indent=2))
    summary = {}
    for case in ("turbo", "base", "edit"):
        t = mx.load(str(out / f"{case}.safetensors"))
        start = np.array(t["start_latents"]).astype(np.float32)
        final = np.array(t["final_latents"]).astype(np.float32)
        image = np.asarray(Image.open(out / f"{case}.png").convert("RGB")).astype(np.float32)
        summary[f"{case}.start_fnv"] = mx.array(np.array([fnv1a32_bytes(start.tobytes())], dtype=np.uint32))
        summary[f"{case}.start_head"] = mx.array(start.reshape(-1)[:64])
        summary[f"{case}.latents_pool4"] = mx.array(pool4(final[:, :, 0], (2, 3)).astype(np.float32))
        summary[f"{case}.image_pool4"] = mx.array(pool4(image, (0, 1)).round().astype(np.uint8))
    mx.save_safetensors(str(out / "z_image_parity.safetensors"), summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("what", choices=["tiny", "real"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--turbo", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument(
        "--reuse", action="store_true", help="summarize existing case outputs in --out"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.what == "tiny":
        write_tiny(args.out)
    else:
        write_real(args.out, args.turbo, args.base, args.reuse)
    print(f"wrote fixtures to {args.out}")


if __name__ == "__main__":
    main()
