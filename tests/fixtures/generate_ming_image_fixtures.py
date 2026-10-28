#!/usr/bin/env python3
"""Generate the Ming-Image-0.1-Design parity fixtures from the upstream Python (mlx-vlm).

Two outputs, both compared against by ``src/imagegen/ming_image/*_tests.rs``:

* ``tiny``: ``tests/fixtures/imagegen/ming_image_tiny_refs.safetensors``, the
  upstream ``MingImageTransformer`` on a tiny config (weights and inputs from
  the closed-form ``det`` / ``fill`` generator of
  ``generate_imagegen_fixtures.py``, which the Rust test mirrors, cast to
  bf16) for a caption and an image stream that both need padding to the
  32-token multiple, with and without ``cap_feats_2``.

* ``real``: ``tests/fixtures/imagegen/ming_image_parity.safetensors``, a compact
  summary of seed-fixed runs of a real checkpoint, one per case: ``gen``
  (bf16 checkpoint, the poster prompt), ``rgba`` (bf16 checkpoint, the RGBA
  prompt) and ``q4`` (the MLX 4-bit checkpoint, the poster prompt), all
  1024x1024, 12 steps, seed 0. Per case it keeps the prompt token ids, the
  initial-noise FNV-1a checksum and head, the 8x8 average-pooled final latents
  and the 32x32 average-pooled image. The full-resolution tensors (conditioning,
  the conditioner's selected hidden states, first-step inputs and velocity,
  final latents, decoded pixels) and the PNGs stay in ``--out`` and are read by
  the ignored real-checkpoint tests when ``MLXCEL_MING_IMAGE_ORACLE_DIR`` points
  there.

Provenance of the committed files: mlx 0.32.4 (CUDA backend, NVIDIA GB10),
mlx-vlm 0.7.8 at git 73d366ecf9233b3fc6549531a7f2386726d09ff6, run with
``MLX_ENABLE_TF32=0``, unmodified upstream modules except the VAE's large
convolutions, which run in row bands with identical arithmetic (see
``generate_qwen_image_fixtures.tile_vae_convs``).

Usage::

    python -I tests/fixtures/generate_ming_image_fixtures.py tiny --out /tmp/ming
    python -I tests/fixtures/generate_ming_image_fixtures.py real --out /tmp/ming \\
        --model models/mlx/ming-image-0.1-design --cases gen,rgba
    python -I tests/fixtures/generate_ming_image_fixtures.py real --out /tmp/ming \\
        --model models/mlx/ming-image-0.1-design-mlx-4bit --cases q4
"""

import argparse
import gc
import importlib.util
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

PROMPTS = {
    "gen": "a minimalist poster, bold word HELLO, blue background",
    "rgba": "RGBA, 4-channel, transparent background, a red apple icon",
    "q4": "a minimalist poster, bold word HELLO, blue background",
}
SEED = 0
STEPS = 12
SIZE = 1024
# Average-pool sizes of the committed summary: latents 8x8, images 32x32.
POOL = 8
IMAGE_POOL = 32


def _load(name: str):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pool_latents(latents: mx.array, k: int) -> mx.array:
    """``[1, C, 1, h, w]`` to ``[C, h/k, w/k]`` average pools in f32."""
    x = np.array(latents.astype(mx.float32))[0, :, 0]
    c, h, w = x.shape
    return mx.array(x.reshape(c, h // k, k, w // k, k).mean(axis=(2, 4)))


class Recorder:
    """Wraps the pipeline's transformer and scheduler to keep intermediates."""

    def __init__(self, pipeline):
        self.calls = []
        self.latents = []
        transformer = pipeline.transformer
        recorder = self

        def wrapped(x, t, cap_feats, cap_feats_2=None):
            out = transformer(x, t, cap_feats, cap_feats_2)
            if not recorder.calls:
                mx.eval(out)
                recorder.calls.append({"x": x, "t": t, "out": out})
            return out

        pipeline.transformer = wrapped
        import mlx_vlm.models.ming_image.pipeline as module

        base = module.FlowMatchEulerDiscreteScheduler

        class Scheduler(base):
            def step(self, *, noise, step_index, latents):
                out = super().step(noise=noise, step_index=step_index, latents=latents)
                recorder.latents.append(out)
                return out

        module.FlowMatchEulerDiscreteScheduler = Scheduler
        self._restore = lambda: (
            setattr(module, "FlowMatchEulerDiscreteScheduler", base),
            setattr(pipeline, "transformer", transformer),
        )

    def close(self):
        self._restore()


def encode_with_states(encoder, input_ids: mx.array) -> dict:
    """Upstream ``MingImageTextEncoder.encode`` keeping its intermediates."""
    ids = input_ids.reshape(-1).tolist()
    start = ids.index(encoder.image_patch_token)
    count = encoder.query_token_count
    block = {
        encoder.image_patch_token,
        encoder.image_start_token,
        encoder.image_end_token,
    }
    text_positions = [i for i, token in enumerate(ids) if token not in block]
    embeds = encoder.mllm.word_embeddings(input_ids)
    embeds = mx.concatenate(
        [
            embeds[:, :start],
            encoder.query_tokens[None].astype(embeds.dtype),
            embeds[:, start + count :],
        ],
        axis=1,
    )
    image_mask = mx.array([token == encoder.image_patch_token for token in ids])[None]
    hidden_states = encoder.mllm(embeds, image_mask)
    query_hidden = hidden_states[-1][:, start : start + count]
    cap_feats = encoder.proj_out(encoder.connector(encoder.proj_in(query_hidden)))
    stacked = mx.concatenate(
        [hidden_states[layer] for layer in encoder.selected_layers], axis=-1
    )
    text = stacked[:, mx.array(text_positions)]
    cap_feats_2 = encoder.directvlm_proj(encoder.directvlm_norm(text))
    out = {
        "cap_feats": cap_feats,
        "cap_feats_2": cap_feats_2,
        "query_hidden": query_hidden,
        "text_states": text,
    }
    mx.eval(list(out.values()))
    return out


def noise_summary(prefix: str, noise: mx.array, summary: dict) -> None:
    raw = np.array(noise.astype(mx.float32)).astype(np.float32)
    summary[f"{prefix}.noise_fnv"] = mx.array(
        np.array([_load("generate_qwen_image_fixtures.py").fnv1a32_bytes(raw.tobytes())], dtype=np.uint32)
    )
    summary[f"{prefix}.noise_head"] = mx.array(raw.reshape(-1)[:16])


def run_case(pipeline, case: str, out: Path, summary: dict) -> None:
    from PIL import Image

    prompt = PROMPTS[case]
    input_ids = pipeline._tokenize(prompt)
    summary[f"{case}.input_ids"] = input_ids.reshape(-1).astype(mx.int32)
    states = encode_with_states(pipeline.text_encoder, input_ids)
    lat = SIZE // 8
    noise = mx.random.normal(
        (1, pipeline.config.vae.z_dim, 1, lat, lat),
        key=mx.random.key(SEED),
        dtype=mx.float32,
    )
    mx.eval(noise)
    noise_summary(case, noise, summary)
    rec = Recorder(pipeline)
    start = time.time()
    image = pipeline.generate_array(
        prompt, seed=SEED, steps=STEPS, width=SIZE, height=SIZE
    )
    mx.eval(image)
    elapsed = time.time() - start
    rec.close()
    pixels = np.array(image)
    final = rec.latents[-1]
    full = {
        **{f"cond.{k}": v for k, v in states.items()},
        "noise": noise,
        "step0_x": rec.calls[0]["x"],
        "step0_t": rec.calls[0]["t"],
        "step0_out": rec.calls[0]["out"],
        "final_latents": final,
        "pixels": mx.array(pixels),
    }
    mx.save_safetensors(str(out / f"{case}_full.safetensors"), full)
    Image.fromarray(pixels, "RGBA").save(out / f"{case}.png")
    summary[f"{case}.latents_pool"] = pool_latents(final, POOL)
    summary[f"{case}.image_pool"] = _load("generate_qwen_image_fixtures.py").pool_image(
        pixels, IMAGE_POOL
    )
    print(f"{case}: {elapsed:.1f}s peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)


def write_real(out: Path, model: Path, cases: list[str]) -> None:
    from mlx_vlm.models.ming_image.pipeline import MingImagePipeline

    qi = _load("generate_qwen_image_fixtures.py")
    out.mkdir(parents=True, exist_ok=True)
    qi.tile_vae_convs()
    summary_path = out / "ming_image_parity.safetensors"
    summary = dict(mx.load(str(summary_path))) if summary_path.exists() else {}
    for case in cases:
        # Keep the conditioner: every case encodes before the pipeline runs it.
        pipeline = MingImagePipeline(model, evict_text_encoder=False)
        mx.eval(pipeline.text_encoder.parameters())
        pipeline._ensure_components()
        mx.eval(pipeline.transformer.parameters(), pipeline.vae.parameters())
        qi.drop_page_cache(model)
        qi.release_before_decode(pipeline)
        run_case(pipeline, case, out, summary)
        del pipeline
        gc.collect()
        mx.clear_cache()
    mx.eval(list(summary.values()))
    mx.save_safetensors(str(summary_path), summary)


def write_layers(out: Path, model: Path, cases: list[str]) -> None:
    """Every backbone hidden state of each case's prompt (``<case>_layers``),
    for a per-layer conditioner diagnosis; only the text encoder loads."""
    from mlx_vlm.models.ming_image.pipeline import MingImagePipeline

    out.mkdir(parents=True, exist_ok=True)
    pipeline = MingImagePipeline(model, evict_text_encoder=False)
    encoder = pipeline.text_encoder
    for case in cases:
        input_ids = pipeline._tokenize(PROMPTS[case])
        ids = input_ids.reshape(-1).tolist()
        start = ids.index(encoder.image_patch_token)
        count = encoder.query_token_count
        embeds = encoder.mllm.word_embeddings(input_ids)
        embeds = mx.concatenate(
            [
                embeds[:, :start],
                encoder.query_tokens[None].astype(embeds.dtype),
                embeds[:, start + count :],
            ],
            axis=1,
        )
        image_mask = mx.array([t == encoder.image_patch_token for t in ids])[None]
        states = encoder.mllm(embeds, image_mask)
        layers = {f"state.{i}": s for i, s in enumerate(states)}
        mx.eval(list(layers.values()))
        mx.save_safetensors(str(out / f"{case}_layers.safetensors"), layers)
        print(f"{case}: {len(states)} states", flush=True)


def write_perturbed(out: Path, model: Path, cases: list[str], eps: float) -> None:
    """Upstream against itself: every backbone layer's output is multiplied by
    ``1 + eps * N(0, 1)`` in f32 and rounded back to bf16 (a rounding-level
    perturbation of the size a different kernel order makes), then the case
    runs end to end. Writes ``<case>_perturbed.png`` and prints the
    conditioner drift and the PNG PSNR against the unperturbed run in
    ``out`` (``<case>_full.safetensors``)."""
    from mlx_vlm.models.ming_image import text_encoder as te
    from mlx_vlm.models.ming_image.pipeline import MingImagePipeline
    from PIL import Image

    qi = _load("generate_qwen_image_fixtures.py")
    original = te.MingDecoderLayer.__call__
    counter = {"i": 0}

    def perturbed(self, x, mask, image_mask):
        y = original(self, x, mask, image_mask)
        counter["i"] += 1
        noise = mx.random.normal(y.shape, key=mx.random.key(1000 + counter["i"]))
        z = (y.astype(mx.float32) * (1.0 + eps * noise)).astype(y.dtype)
        return z

    te.MingDecoderLayer.__call__ = perturbed
    qi.tile_vae_convs()
    for case in cases:
        counter["i"] = 0
        pipeline = MingImagePipeline(model, evict_text_encoder=False)
        mx.eval(pipeline.text_encoder.parameters())
        pipeline._ensure_components()
        mx.eval(pipeline.transformer.parameters(), pipeline.vae.parameters())
        qi.drop_page_cache(model)
        qi.release_before_decode(pipeline)
        ref = mx.load(str(out / f"{case}_full.safetensors"))
        states = encode_with_states(pipeline.text_encoder, pipeline._tokenize(PROMPTS[case]))
        for name in ("query_hidden", "text_states", "cap_feats", "cap_feats_2"):
            a = np.array(states[name].astype(mx.float32))
            b = np.array(ref[f"cond.{name}"].astype(mx.float32))
            rel = np.sqrt(((a - b) ** 2).sum() / (b**2).sum())
            print(f"{case} eps={eps}: conditioner {name} rel-RMS {rel:.4e}", flush=True)
        counter["i"] = 0  # the image run draws the same perturbation
        image = pipeline.generate_array(
            PROMPTS[case], seed=SEED, steps=STEPS, width=SIZE, height=SIZE
        )
        pixels = np.array(image)
        theirs = np.array(ref["pixels"])
        mse = ((pixels.astype(np.float64) - theirs.astype(np.float64)) ** 2).mean()
        psnr = 10 * np.log10(255.0**2 / max(mse, 1e-12))
        print(f"{case} eps={eps}: PNG PSNR vs unperturbed upstream {psnr:.2f} dB", flush=True)
        Image.fromarray(pixels, "RGBA").save(out / f"{case}_perturbed_{eps:g}.png")
        del pipeline
        gc.collect()
        mx.clear_cache()


# ------------------------------------------------------------------ tiny mode
TINY = dict(
    dim=64,
    n_heads=2,
    n_kv_heads=2,
    n_layers=2,
    n_refiner_layers=1,
    intermediate_size=96,
    cap_feat_dim=24,
    in_channels=4,
    patch_size=2,
    f_patch_size=1,
    axes_dims=(8, 12, 12),
    rope_theta=256.0,
    norm_eps=1e-5,
    t_scale=1000.0,
    adaln_embed_dim=32,
)


def write_tiny(out: Path) -> None:
    from mlx_vlm.models.ming_image.config import MingImageDiTConfig
    from mlx_vlm.models.ming_image.transformer import MingImageTransformer

    shared = _load("generate_imagegen_fixtures.py")
    model = MingImageTransformer(MingImageDiTConfig(**TINY))
    # Weights from the closed-form generator (mirrored by the Rust test), in bf16.
    shared.fill(model, "tf")
    model.set_dtype(mx.bfloat16)
    # A 6x10 latent: 3x5 = 15 image tokens (padded to 32); 7 + 5 caption tokens.
    x = shared.det("x", (1, 4, 1, 6, 10)).astype(mx.bfloat16)
    t = mx.array([0.375], dtype=mx.bfloat16)
    cap = shared.det("cap", (1, 7, 24)).astype(mx.bfloat16)
    cap2 = shared.det("cap2", (1, 5, 64)).astype(mx.bfloat16)
    refs = {
        "out": model(x, t, cap, cap2),
        "out_no_cap2": model(x, t, cap, None),
    }
    mx.eval(list(refs.values()))
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "ming_image_tiny_refs.safetensors"), refs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("tiny", "real", "layers", "perturb"))
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--cases", default="gen,rgba")
    args = parser.parse_args()
    if args.mode == "tiny":
        write_tiny(args.out)
    elif args.mode == "perturb":
        write_perturbed(args.out, args.model, args.cases.split(","), args.eps)
    elif args.mode == "layers":
        write_layers(args.out, args.model, args.cases.split(","))
    else:
        write_real(args.out, args.model, args.cases.split(","))


if __name__ == "__main__":
    main()
