#!/usr/bin/env python3
"""Generate the Mage-Flow parity fixtures from the upstream Python (mlx-vlm).

Two outputs, both compared against by ``src/imagegen/mage_flow/*_tests.rs``:

* ``tiny``: ``tests/fixtures/imagegen/mage_flow_tiny_refs.safetensors``:

  - ``rope.*``: upstream ``image_rope_frequencies`` for one 32x48 grid (every
    16th row of cos/sin, plus the per-token frame/height/width positions the
    tables are built from) and for a three-frame 4x6 edit layout (every row).
  - ``tf.*``: upstream ``MageFlowTransformer`` outputs for a tiny config whose
    weights come from the closed-form ``det`` / ``fill`` generator of
    ``generate_imagegen_fixtures.py`` (the Rust test mirrors it), in f32 and
    bf16, for a generate layout and a two-frame edit layout.
  - ``vae.*`` (when ``--vae`` names a checkpoint): the real Mage-VAE decoding a
    fixed ``[1, 4, 6, 128]`` bf16 latent and encoding a fixed ``[1, 32, 48, 3]``
    bf16 image (posterior mean and clipped log-variance), plus bf16 same-input
    stage references (see ``vae_stages``).

* ``real``: ``tests/fixtures/imagegen/mage_flow_parity.safetensors``, a compact
  summary of seed-fixed runs of the real checkpoints (``turbo``: Mage-Flow-Turbo
  512x512, 4 steps, seed 42; ``edit``: Mage-Flow-Edit-Turbo 512x512, 4 steps,
  seed 42, two references). Per case it keeps the prompt token ids, the
  initial-noise FNV-1a checksum and head, the 4x4 average-pooled final latents,
  the 4x4 average-pooled image, a 64-channel mean-pool of the conditioner
  hidden states and of the same conditioner widened to f32
  (``precision_references``), and for ``edit`` the M-RoPE position ids. The full-resolution tensors and PNGs stay in ``--out`` and are
  read by the ignored real-checkpoint tests when ``MLXCEL_MAGE_FLOW_ORACLE_DIR``
  points there.

Provenance of the committed files: mlx 0.32.4 (CUDA backend, NVIDIA GB10),
mlx-vlm 0.7.8 at git 73d366ecf9233b3fc6549531a7f2386726d09ff6, run with
``MLX_ENABLE_TF32=0``, with the transformer's and VAE's ``nn.LayerNorm``
computed from separate ops (see ``deterministic_layer_norm``).

Usage::

    python -I tests/fixtures/generate_mage_flow_fixtures.py tiny --out /tmp/mf \\
        --vae models/mlx/mage-flow-turbo
    python -I tests/fixtures/generate_mage_flow_fixtures.py real --out /tmp/mf \\
        --turbo models/mlx/mage-flow-turbo --edit models/mlx/mage-flow-edit-turbo
"""

import argparse
import gc
import importlib.util
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

PROMPT = "Editorial photograph of a cobalt teapot on a red table"
EDIT_PROMPT = "Blend the object from image 2 naturally into image 1"
EDIT_REFS = ("assets/multiref_000000_0.jpg", "assets/multiref_000000_1.png")


def _shared():
    path = Path(__file__).with_name("generate_imagegen_fixtures.py")
    spec = importlib.util.spec_from_file_location("imagegen_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def deterministic_layer_norm(module) -> None:
    """Compute the ``nn.LayerNorm`` layers of ``module`` from separate ops.

    MLX's fused CUDA ``layer_norm`` kernel reuses one shared-memory buffer for
    its two block reductions without a barrier between them, so on 384-wide
    bf16 rows (the Mage-VAE encoder's ``norm_out``) a few rows per call come
    out wrong and two identical edit runs differ (reference latents by ~1e-1,
    PNGs at ~30 dB). The Rust port computes the same f32 statistics from
    separate ops (``mage_flow::layers::LayerNorm``), and so do these
    references, for the Mage-Flow transformer and VAE only (the Qwen3-VL
    conditioner is left as upstream ships it).
    """
    import mlx.nn as nn

    class LayerNorm(nn.LayerNorm):
        def __call__(self, x):
            xf = x.astype(mx.float32)
            centered = xf - xf.mean(axis=-1, keepdims=True)
            var = (centered * centered).mean(axis=-1, keepdims=True)
            y = centered * mx.rsqrt(var + self.eps)
            if "weight" in self:
                y = y * self.weight.astype(mx.float32)
            if "bias" in self:
                y = y + self.bias.astype(mx.float32)
            return y.astype(x.dtype)

    for _, layer in module.named_modules():
        if type(layer) is nn.LayerNorm:
            layer.__class__ = LayerNorm


def fnv1a32_bytes(data: bytes) -> int:
    h = 0x811C9DC5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def rope_positions(shapes):
    rows = []
    for index, (frames, height, width) in enumerate(shapes):
        for f in range(frames):
            for y in range(height):
                for x in range(width):
                    rows.append(
                        (index + f, y - (height - height // 2), x - (width - width // 2))
                    )
    return np.array(rows, dtype=np.float32)


def write_tiny(out: Path, vae_root: Path | None) -> None:
    from mlx_vlm.models.mage_flow.transformer import (
        MageFlowTransformer,
        image_rope_frequencies,
    )

    shared = _shared()
    refs = {}

    grid = [(1, 32, 48)]
    cos, sin = image_rope_frequencies(grid)
    refs["rope.grid_positions"] = mx.array(rope_positions(grid))
    refs["rope.grid_cos_rows16"] = cos[::16]
    refs["rope.grid_sin_rows16"] = sin[::16]
    edit = [(1, 4, 6)] * 3
    cos, sin = image_rope_frequencies(edit)
    refs["rope.edit_positions"] = mx.array(rope_positions(edit))
    refs["rope.edit_cos"] = cos
    refs["rope.edit_sin"] = sin

    model = MageFlowTransformer(
        in_channels=8,
        out_channels=8,
        context_in_dim=24,
        hidden_size=64,
        num_heads=2,
        depth=2,
        axes_dim=(8, 12, 12),
    )
    shared.fill(model, "mft")
    deterministic_layer_norm(model)
    img = shared.det("mft.img", (1, 12, 8))
    txt = shared.det("mft.txt", (1, 5, 24))
    t = mx.array(np.array([0.7], dtype=np.float32))
    refs["tf.f32_generate"] = model(img=img, txt=txt, timesteps=t, img_shapes=[(1, 3, 4)])
    edit_img = shared.det("mft.edit_img", (1, 12, 8))
    refs["tf.f32_edit"] = model(
        img=edit_img, txt=txt, timesteps=t, img_shapes=[(1, 2, 3), (1, 2, 3)]
    )
    model.set_dtype(mx.bfloat16)
    bf = lambda a: a.astype(mx.bfloat16)  # noqa: E731
    refs["tf.bf16_generate"] = model(
        img=bf(img), txt=bf(txt), timesteps=bf(t), img_shapes=[(1, 3, 4)]
    ).astype(mx.float32)

    if vae_root is not None:
        from mlx_vlm.models.mage_flow.weights import load_vae

        vae = load_vae(vae_root, include_encoder=True)
        deterministic_layer_norm(vae)
        latent = shared.det("mage_vae.latent", (1, 4, 6, 128)).astype(mx.bfloat16)
        refs["vae.decode"] = vae.decode(latent).astype(mx.float32)
        image = mx.clip(shared.det("mage_vae.image", (1, 32, 48, 3), 0.9), -1, 1)
        mean, logvar = vae.dconv_encoder(image.astype(mx.bfloat16))
        refs["vae.encode_mean"] = mean.astype(mx.float32)
        refs["vae.encode_logvar"] = logvar.astype(mx.float32)
        refs.update(vae_stages(vae, latent, image.astype(mx.bfloat16)))
    mx.eval(list(refs.values()))
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "mage_flow_tiny_refs.safetensors"), refs)


def vae_stages(vae, latent, image) -> dict:
    """Same-input stage references (bf16): the CoD condition, DiCo blocks 7
    and 20 of the decoder's denoiser and block 20 of the encoder, each with
    its input. A deep bf16 stack amplifies single-ulp reduction-order ties
    (and this oracle's bf16-accumulated ``mx.mean``), so the Rust tests gate
    each stage on upstream's own input and only bound the end-to-end drift."""
    out = {}
    dec = vae.decoder_model
    condition = dec.y_embedder.decoder(latent)
    out["vae.cod"] = condition
    batch, grid_h, grid_w, _ = condition.shape
    noise = mx.zeros((batch, grid_h * 16, grid_w * 16, 3), dtype=condition.dtype)
    temb = dec.t_embedder(mx.zeros((batch,), dtype=condition.dtype), condition.dtype)
    spatial = dec.s_embedder(noise, condition)
    for index, block in enumerate(dec.blocks):
        if index in (7, 20):
            out[f"vae.dec_block{index}_in"] = spatial
        spatial = block(spatial, temb)
        if index in (7, 20):
            out[f"vae.dec_block{index}_out"] = spatial
    enc = vae.dconv_encoder
    temb = enc.t_embedder(mx.zeros((batch,), dtype=image.dtype), image.dtype)
    condition = enc.patch_cond_embed(image)
    for block in enc.head_blocks:
        condition = block(condition)
    condition = enc.proj_down(condition)
    zeros = mx.zeros((*condition.shape[:3], enc.latent_channels), dtype=image.dtype)
    x = enc.fuse_proj(mx.concatenate([condition, enc.z_proj(zeros)], axis=-1))
    for index, block in enumerate(enc.blocks):
        if index == 20:
            out["vae.enc_block20_in"] = x
        x = block(x, temb)
    out["vae.enc_block20_out"] = x
    return {k: v.astype(mx.bfloat16) for k, v in out.items()}


def pool4(a: np.ndarray, axes: tuple[int, int]) -> np.ndarray:
    """Average-pool two axes by 4."""
    a = np.moveaxis(a, axes, (-2, -1))
    *lead, h, w = a.shape
    a = a.reshape(*lead, h // 4, 4, w // 4, 4).mean(axis=(-3, -1))
    return np.moveaxis(a, (-2, -1), axes)


def run_case(case: str, model_dir: Path, out: Path) -> dict:
    from PIL import Image

    from mlx_vlm.models.mage_flow.config import variant_from_local_path
    from mlx_vlm.models.mage_flow.pipeline import MageFlowPipeline
    from mlx_vlm.models.mage_flow.text_encoder import (
        EDIT_TEMPLATE,
        GENERATION_TEMPLATE,
        IMAGE_PLACEHOLDER,
        resize_long_edge,
    )

    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    pipe = MageFlowPipeline.from_pretrained(
        variant_from_local_path(model_dir), model_path=model_dir
    )
    rec = {}
    encoder = pipe.text_encoder
    if case == "turbo":
        formatted = GENERATION_TEMPLATE.format(PROMPT)
        ids = encoder.tokenizer(formatted, max_length=2048 + 34, truncation=True)
        rec["input_ids"] = mx.array(np.array(ids["input_ids"], dtype=np.int32))
    else:
        refs = [Image.open(model_dir / p).convert("RGB") for p in EDIT_REFS]
        small = [resize_long_edge(image, 384) for image in refs]
        prefix = "".join(f"Image {i}: {IMAGE_PLACEHOLDER}" for i in range(1, 3))
        inputs = encoder.processor(
            text=[EDIT_TEMPLATE.format(prefix + EDIT_PROMPT)],
            images=small,
            padding=True,
            return_tensors="np",
        )
        rec["input_ids"] = mx.array(np.asarray(inputs["input_ids"][0], dtype=np.int32))
        rec["pixel_values"] = mx.array(np.asarray(inputs["pixel_values"], dtype=np.float32))
        rec["image_grid_thw"] = mx.array(np.asarray(inputs["image_grid_thw"], dtype=np.int32))

    original_gen = pipe._encode_generation_pair
    original_edit = pipe._encode_edit_pair

    def gen_pair(prompt, negative):
        pos, neg = original_gen(prompt, negative)
        rec["cond"] = pos
        return pos, neg

    def edit_pair(prompt, negative, images, **kw):
        pos, neg = original_edit(prompt, negative, images, **kw)
        rec["cond"] = pos
        return pos, neg

    pipe._encode_generation_pair = gen_pair
    pipe._encode_edit_pair = edit_pair
    original_components = pipe._ensure_components

    def ensure(*, require_encoder):
        original_components(require_encoder=require_encoder)
        deterministic_layer_norm(pipe.transformer)
        deterministic_layer_norm(pipe.vae)
        vae = pipe.vae
        if getattr(vae, "_recording", False):
            return
        decode, encode = vae.decode, vae.encode

        def rec_decode(latent):
            rec["final_latents"] = latent
            return decode(latent)

        def rec_encode(image, **kw):
            rec["reference_pixels"] = image
            result = encode(image, **kw)
            rec["reference_latents"] = result
            return result

        vae.decode, vae.encode, vae._recording = rec_decode, rec_encode, True

    pipe._ensure_components = ensure
    start = time.time()
    if case == "turbo":
        img = pipe.generate_array(PROMPT, seed=42, steps=4, width=512, height=512, guidance=1.0)
    elif case == "edit":
        img = pipe.edit_array(
            EDIT_PROMPT,
            [str(model_dir / p) for p in EDIT_REFS],
            seed=42,
            steps=4,
            width=512,
            height=512,
            guidance=1.0,
        )
    else:
        raise SystemExit(f"unknown case {case}")
    mx.eval(img)
    elapsed = time.time() - start
    peak_gb = mx.get_peak_memory() / 1e9
    precision_references(pipe, case, model_dir, rec)
    rec["start_latents"] = mx.random.normal(
        (1, 32 * 32, 128), key=mx.random.key(42), dtype=mx.bfloat16
    )
    Image.fromarray(np.array(img)).save(out / f"{case}.png")
    tensors = {
        k: (v.astype(mx.float32) if mx.issubdtype(v.dtype, mx.floating) else v)
        for k, v in rec.items()
    }
    mx.save_safetensors(str(out / f"{case}.safetensors"), tensors)
    return {"case": case, "elapsed_s": elapsed, "peak_gb": peak_gb}


def precision_references(pipe, case: str, model_dir: Path, rec: dict) -> None:
    """Conditioner references beyond the run itself.

    ``position_ids``: the M-RoPE positions upstream's ``get_input_embeddings``
    hands the language model (edit only; text-only prompts use plain ranges).
    ``cond_f32``: the conditioner rerun with every bf16 parameter widened to
    f32. A bf16 port cannot match upstream's own bf16 hidden states to better
    than the kernels' accumulation order allows (a single Qwen3-VL layer
    already moves ~5e-3), so the Rust test asks that the port sit no further
    from this f32 reference than upstream's bf16 run does.
    """
    from mlx.utils import tree_map
    from PIL import Image

    from mlx_vlm.models.mage_flow.text_encoder import _to_mx

    encoder = pipe._ensure_text_encoder()
    if case == "edit":
        features = encoder.model.get_input_embeddings(
            _to_mx(rec["input_ids"]).reshape(1, -1),
            _to_mx(rec["pixel_values"]),
            image_grid_thw=rec["image_grid_thw"],
        )
        rec["position_ids"] = features.position_ids.astype(mx.int32)
    model = encoder.model
    model.update(
        tree_map(
            lambda x: x.astype(mx.float32) if x.dtype == mx.bfloat16 else x,
            model.parameters(),
        )
    )
    if case == "turbo":
        rec["cond_f32"] = encoder.encode(PROMPT)
    else:
        refs = [Image.open(model_dir / p).convert("RGB") for p in EDIT_REFS]
        rec["cond_f32"] = encoder.encode_edit(EDIT_PROMPT, refs, vl_cond_long_edge=384)
    mx.eval(rec)


def pool64(cond: np.ndarray) -> np.ndarray:
    b, length, width = cond.shape
    return cond.reshape(b, length, width // 64, 64).mean(axis=-1).astype(np.float32)


def write_real(out: Path, turbo: Path, edit: Path, reuse: bool) -> None:
    from PIL import Image

    if not reuse:
        meta = [run_case("turbo", turbo, out), run_case("edit", edit, out)]
        (out / "mage_flow_parity.json").write_text(json.dumps(meta, indent=2))
    summary = {}
    for case in ("turbo", "edit"):
        t = mx.load(str(out / f"{case}.safetensors"))
        start = np.array(t["start_latents"]).astype(np.float32)
        final = np.array(t["final_latents"]).astype(np.float32)
        cond = np.array(t["cond"]).astype(np.float32)
        image = np.asarray(Image.open(out / f"{case}.png").convert("RGB")).astype(np.float32)
        summary[f"{case}.input_ids"] = t["input_ids"]
        summary[f"{case}.start_fnv"] = mx.array(
            np.array([fnv1a32_bytes(start.tobytes())], dtype=np.uint32)
        )
        summary[f"{case}.start_head"] = mx.array(start.reshape(-1)[:64])
        summary[f"{case}.latents_pool4"] = mx.array(pool4(final[0], (0, 1)).astype(np.float32))
        summary[f"{case}.image_pool4"] = mx.array(pool4(image, (0, 1)).round().astype(np.uint8))
        summary[f"{case}.cond_pool64"] = mx.array(pool64(cond))
        summary[f"{case}.cond_f32_pool64"] = mx.array(
            pool64(np.array(t["cond_f32"]).astype(np.float32))
        )
        if "position_ids" in t:
            summary[f"{case}.position_ids"] = t["position_ids"]
    mx.save_safetensors(str(out / "mage_flow_parity.safetensors"), summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("what", choices=["tiny", "real"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vae", type=Path, help="checkpoint whose vae/ the tiny VAE refs use")
    parser.add_argument("--turbo", type=Path)
    parser.add_argument("--edit", type=Path)
    parser.add_argument(
        "--reuse", action="store_true", help="summarize existing case outputs in --out"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.what == "tiny":
        write_tiny(args.out, args.vae)
    else:
        write_real(args.out, args.turbo, args.edit, args.reuse)
    print(f"wrote fixtures to {args.out}")


if __name__ == "__main__":
    main()
