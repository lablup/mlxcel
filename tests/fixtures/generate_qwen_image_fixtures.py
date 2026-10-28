#!/usr/bin/env python3
"""Generate the Qwen-Image-2.1 parity fixtures from the upstream Python (mlx-vlm).

Two outputs, both compared against by ``src/imagegen/qwen_image/*_tests.rs``:

* ``tiny``: ``tests/fixtures/imagegen/qwen_image_tiny_refs.safetensors``:

  - ``rope.*``: upstream ``QwenImageRope.for_layout`` cos/sin for a generate
    layout (5 text tokens, one 4x6 grid) and an edit layout (text, a 4x4
    reference block between text runs, a trailing 2x6 target).
  - ``tf.*``: upstream ``QwenImageTransformer`` outputs for a tiny config whose
    weights come from the closed-form ``det`` / ``fill`` generator of
    ``generate_imagegen_fixtures.py`` (the Rust test mirrors it), in bf16, for
    a generate call and for the three calls of a two-step reference-KV-cache
    edit (no cache, extract, cached).
  - ``vae.*``: upstream ``QwenImageVAE`` (tiny residual config and tiny
    non-residual config, ``fill`` weights, f32) decoding a fixed latent and
    encoding a fixed image.
  - ``vae_real.*`` (when ``--model`` names the checkpoint): the real VAE
    decoding a fixed ``[1, 64, 1, 4, 6]`` latent and encoding a fixed
    ``[1, 4, 1, 64, 96]`` bf16 image.

* ``real``: ``tests/fixtures/imagegen/qwen_image_parity.safetensors``, a compact
  summary of seed-fixed runs of the real checkpoint: ``gen`` (1024x1024, 4
  steps, seed 7, the issue's prompt) and ``edit`` (one RGBA 1024x1024
  reference, 4 steps, seed 7, with and without the reference KV cache). Per
  case it keeps the prompt token ids, ``drop_idx``, the initial-noise FNV-1a
  checksum and head, the 8x8 average-pooled final latents and the 32x32 average-pooled image. The
  full-resolution tensors (conditioning, first-step velocity, final latents,
  decoded pixels, reference latents) and PNGs stay in ``--out`` and are read by
  the ignored real-checkpoint tests when ``MLXCEL_QWEN_IMAGE_ORACLE_DIR``
  points there.

Provenance of the committed files: mlx 0.32.4 (CUDA backend, NVIDIA GB10),
mlx-vlm 0.7.8 at git 73d366ecf9233b3fc6549531a7f2386726d09ff6, run with
``MLX_ENABLE_TF32=0``, unmodified upstream modules.

Usage::

    python -I tests/fixtures/generate_qwen_image_fixtures.py tiny --out /tmp/qi \\
        --model models/mlx/qwen-image-2.1
    python -I tests/fixtures/generate_qwen_image_fixtures.py real --out /tmp/qi \\
        --model models/mlx/qwen-image-2.1 --cases gen,edit
"""

import argparse
import gc
import importlib.util
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

PROMPT = "A red ceramic teapot on a wooden table in soft window light"
EDIT_PROMPT = "Change the teapot to cobalt blue"
SEED = 7
STEPS = 4
SIZE = 1024
# Average-pool sizes of the committed summary: latents 8x8, images 32x32.
POOL = 8
IMAGE_POOL = 32


def _shared():
    path = Path(__file__).with_name("generate_imagegen_fixtures.py")
    spec = importlib.util.spec_from_file_location("imagegen_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fnv1a32_bytes(data: bytes) -> int:
    h = 0x811C9DC5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def pool_tokens(latents: mx.array, h: int, w: int, k: int) -> mx.array:
    """``[1, h*w, C]`` to ``[C, h/k, w/k]`` average pools in f32."""
    x = np.array(latents.astype(mx.float32)).reshape(h, w, -1).transpose(2, 0, 1)
    c = x.shape[0]
    return mx.array(x.reshape(c, h // k, k, w // k, k).mean(axis=(2, 4)))


def pool_image(pixels: np.ndarray, k: int) -> mx.array:
    """``[H, W, C]`` uint8 to ``[C, H/k, W/k]`` average pools in f32."""
    x = pixels.astype(np.float32).transpose(2, 0, 1)
    c, h, w = x.shape
    return mx.array(x.reshape(c, h // k, k, w // k, k).mean(axis=(2, 4)))


class Recorder:
    """Wraps the pipeline's transformer and scheduler to keep intermediates."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.calls = []
        self.latents = []
        transformer = pipeline.transformer
        recorder = self

        class Wrapped:
            def __init__(self, inner):
                self.inner = inner
                self.causal_condition = inner.causal_condition
                self.transformer_blocks = inner.transformer_blocks

            def __call__(self, **kwargs):
                out = self.inner(**kwargs)
                if len(recorder.calls) < 1:
                    mx.eval(out)
                    recorder.calls.append(
                        {
                            "hidden_states": kwargs["hidden_states"],
                            "timestep": kwargs["timestep"],
                            "out": out,
                        }
                    )
                return out

        pipeline.transformer = Wrapped(transformer)
        import mlx_vlm.models.qwen_image.pipeline as module

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


def initial_noise(h_lat: int, w_lat: int) -> mx.array:
    mx.random.seed(SEED)
    latents = mx.random.normal((1, 1, 64, h_lat, w_lat)).astype(mx.bfloat16)
    return latents.reshape(1, 64, h_lat * w_lat).transpose(0, 2, 1)


def noise_summary(prefix: str, noise: mx.array, summary: dict) -> None:
    raw = np.array(noise.astype(mx.float32)).astype(np.float32)
    bf16_bits = (raw.view(np.uint32) >> 16).astype(np.uint16)
    summary[f"{prefix}.noise_fnv"] = mx.array(
        np.array([fnv1a32_bytes(bf16_bits.tobytes())], dtype=np.uint32)
    )
    summary[f"{prefix}.noise_head"] = mx.array(raw.reshape(-1)[:16])


def run_generate(pipeline, out: Path, summary: dict) -> None:
    from PIL import Image

    enc = pipeline.text_encoder
    formatted = __import__(
        "mlx_vlm.models.qwen_image.text_encoder", fromlist=["x"]
    ).PROMPT_TEMPLATE_T2I.format(PROMPT)
    ids = enc.tokenizer(formatted, return_tensors="np")["input_ids"][0]
    summary["gen.input_ids"] = mx.array(ids.astype(np.int32))
    summary["gen.drop_idx"] = mx.array(np.array([enc.drop_idx], dtype=np.int32))
    emb = enc.encode(PROMPT).astype(mx.bfloat16)
    mx.eval(emb)
    h_lat = w_lat = SIZE // 16
    noise_summary("gen", initial_noise(h_lat, w_lat), summary)
    rec = Recorder(pipeline)
    start = time.time()
    image = pipeline.generate_array(
        PROMPT, seed=SEED, steps=STEPS, width=SIZE, height=SIZE
    )
    mx.eval(image)
    elapsed = time.time() - start
    rec.close()
    pixels = np.array(image)
    final = rec.latents[-1]
    full = {
        "cond": emb,
        "step0_input": rec.calls[0]["hidden_states"],
        "step0_timestep": rec.calls[0]["timestep"],
        "step0_out": rec.calls[0]["out"],
        "final_latents": final,
        "pixels": mx.array(pixels),
    }
    mx.save_safetensors(str(out / "gen_full.safetensors"), full)
    Image.fromarray(pixels).save(out / "gen.png")
    summary["gen.latents_pool"] = pool_tokens(final, h_lat, w_lat, POOL)
    summary["gen.image_pool"] = pool_image(pixels, IMAGE_POOL)
    print(f"gen: {elapsed:.1f}s peak {mx.get_peak_memory() / 1e9:.1f} GB")


def make_reference(out: Path, source_png: Path) -> Path:
    """An RGBA 1024x1024 reference from ``source_png`` with a radial alpha."""
    from PIL import Image

    rgb = np.array(Image.open(source_png).convert("RGB"))
    h, w, _ = rgb.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
    alpha = np.clip((1.15 - r) * 400, 0, 255).astype(np.uint8)
    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
    path = out / "edit_ref_rgba.png"
    Image.fromarray(rgba, "RGBA").save(path)
    return path


def run_edit(pipeline, out: Path, summary: dict, reference: Path) -> None:
    from PIL import Image

    enc = pipeline.text_encoder
    with Image.open(reference) as source:
        ref = source.convert("RGBA")
    emb, pad_mask = enc.encode_edit(EDIT_PROMPT, [ref])
    mx.eval(emb, pad_mask)
    summary["edit.pad_count"] = mx.array(
        np.array([int(pad_mask.astype(mx.int32).sum().item())], dtype=np.int32)
    )
    summary["edit.cond_len"] = mx.array(np.array([emb.shape[1]], dtype=np.int32))
    pixels_in = mx.array(np.asarray(ref).astype(np.float32) / 127.5 - 1.0)
    pixels_in = pixels_in.transpose(2, 0, 1)[None, :, None].astype(mx.bfloat16)
    mean, _ = pipeline.vae.encode(pixels_in)
    normalized = (mean - pipeline.latents_mean.astype(mean.dtype)) / (
        pipeline.latents_std.astype(mean.dtype)
    )
    mx.eval(normalized)
    h_lat = w_lat = SIZE // 16
    noise_summary("edit", initial_noise(h_lat, w_lat), summary)
    full = {
        "cond": emb.astype(mx.bfloat16),
        "pad_mask": pad_mask.astype(mx.uint8),
        "ref_latents": normalized.reshape(1, 64, -1).transpose(0, 2, 1),
    }
    for kv in (True, False):
        tag = "kv" if kv else "nokv"
        rec = Recorder(pipeline)
        start = time.time()
        image = pipeline.edit_array(
            EDIT_PROMPT, [reference], seed=SEED, steps=STEPS, use_kv_cache=kv
        )
        mx.eval(image)
        elapsed = time.time() - start
        rec.close()
        pixels = np.array(image)
        final = rec.latents[-1]
        full[f"{tag}_final_latents"] = final
        full[f"{tag}_pixels"] = mx.array(pixels)
        if kv:
            full["step0_input"] = rec.calls[0]["hidden_states"]
            full["step0_timestep"] = rec.calls[0]["timestep"]
            full["step0_out"] = rec.calls[0]["out"]
        Image.fromarray(pixels, "RGBA").save(out / f"edit_{tag}.png")
        summary[f"edit.{tag}_latents_pool"] = pool_tokens(final, h_lat, w_lat, POOL)
        summary[f"edit.{tag}_image_pool"] = pool_image(pixels, IMAGE_POOL)
        print(f"edit {tag}: {elapsed:.1f}s peak {mx.get_peak_memory() / 1e9:.1f} GB")
    mx.save_safetensors(str(out / "edit_full.safetensors"), full)


def drop_page_cache(root: Path) -> None:
    """Release the page cache of the checkpoint files once they are loaded.

    On GB10 the CUDA allocator draws on free memory only, and page cache of
    the 31 GB of safetensors just read is not reclaimed for it, so the 1024x1024
    VAE decode fails with ``cudaMallocAsync`` out of memory. Dropping clean
    cache pages does not touch the loaded arrays.
    """
    import os

    for path in root.rglob("*.safetensors"):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


# Largest ``rows * width * max(c_in, c_out)`` one stride-1 VAE convolution
# runs over before it is split into row bands; mirrored by
# ``qwen_image::vae::CONV_BAND_ELEMENTS``.
CONV_BAND_ELEMENTS = 1 << 27


def tile_vae_convs() -> None:
    """Run the VAE's large stride-1 convolutions in row bands.

    MLX's CUDA convolution (mlx 0.32.4 on GB10) fails with ``cudaMallocAsync``
    out of memory on a 3x3 conv from 288 to 288 channels at 1024x1024 even with
    40 GB free (288 to 192 succeeds at 4 GB peak), so the 1024x1024 decode
    cannot run as upstream writes it. Each band takes its rows of the padded
    input plus a one-row halo and convolves without padding, so every output
    pixel sums the same products; the Rust port applies the same rule.
    """
    import mlx.nn as nn

    import mlx_vlm.models.qwen_image.vae as vae_module

    def banded(conv: nn.Conv2d, x: mx.array) -> mx.array:
        n, h, w, c_in = x.shape
        c_out, kh, kw, _ = conv.weight.shape
        stride = conv.stride if isinstance(conv.stride, tuple) else (conv.stride,) * 2
        padding = conv.padding if isinstance(conv.padding, tuple) else (conv.padding,) * 2
        if stride != (1, 1) or h * w * max(c_in, c_out) <= CONV_BAND_ELEMENTS:
            # The plain nn.Conv2d call (QwenImageCausalConv overrides __call__).
            return nn.Conv2d.__call__(conv, x)
        rows = max(1, CONV_BAND_ELEMENTS // (w * max(c_in, c_out)))
        ph, pw = padding
        xp = mx.pad(x, [(0, 0), (ph, ph), (pw, pw), (0, 0)])
        out_h = h + 2 * ph - kh + 1
        parts = []
        for r0 in range(0, out_h, rows):
            r1 = min(out_h, r0 + rows)
            y = mx.conv2d(xp[:, r0 : r1 + kh - 1], conv.weight)
            if "bias" in conv:
                y = y + conv.bias
            parts.append(y)
        return mx.concatenate(parts, axis=1)

    def _conv2d(conv, x):
        module = getattr(conv, "__self__", conv)
        x = x.transpose(0, 2, 3, 1)
        x = banded(module, x)
        return x.transpose(0, 3, 1, 2)

    vae_module._conv2d = _conv2d


def release_before_decode(pipeline) -> None:
    """Clear the MLX buffer cache before each VAE decode (no numeric effect)."""
    vae = pipeline.vae
    decode = vae.decode

    def wrapped(z):
        mx.clear_cache()
        return decode(z)

    vae.decode = wrapped


def write_real(out: Path, model: Path, cases: list[str]) -> None:
    from mlx_vlm.models.qwen_image.pipeline import QwenImagePipeline

    out.mkdir(parents=True, exist_ok=True)
    pipeline = QwenImagePipeline.from_pretrained(model_path=model, download=False)
    mx.eval(
        pipeline.transformer.parameters(),
        pipeline.vae.parameters(),
        pipeline.text_encoder.model.parameters(),
    )
    drop_page_cache(model)
    release_before_decode(pipeline)
    tile_vae_convs()
    summary_path = out / "qwen_image_parity.safetensors"
    summary = dict(mx.load(str(summary_path))) if summary_path.exists() else {}
    if "gen" in cases:
        run_generate(pipeline, out, summary)
    if "edit" in cases:
        reference = make_reference(out, out / "gen.png")
        run_edit(pipeline, out, summary, reference)
    mx.eval(list(summary.values()))
    mx.save_safetensors(str(summary_path), summary)
    del pipeline
    gc.collect()


# ------------------------------------------------------------------ tiny mode
TINY_TF = dict(
    in_channels=8,
    out_channels=8,
    num_layers=2,
    num_attention_heads=2,
    attention_head_dim=16,
    context_in_dim=12,
    mlp_ratio=3,
    axes_dims_rope=(4, 6, 6),
    eps=1e-6,
    causal_condition=True,
)
TINY_VAE = dict(
    base_dim=4,
    decoder_base_dim=6,
    z_dim=4,
    dim_mult=(1, 2, 2),
    num_res_blocks=1,
    temperal_downsample=(False, True),
    in_channels=4,
    out_channels=4,
)
# The edit layout of the tiny transformer: 10 prompt tokens whose positions
# 3..7 are the four vision slots of one 4x4 reference, then a 2x6 target.
EDIT_SLOTS = (3, 7)


def _tf_checkpoint_name(key: str) -> str:
    """Upstream module key to the Diffusers checkpoint name (inverse remap)."""
    return key.replace("time_text_embed.", "time_text_embed.timestep_embedder.").replace(
        "modulation.0.", "modulation.1."
    )


def _vae_fill(module, prefix: str) -> None:
    """Closed-form VAE weights: gamma near 1, small biases, fan-in scaled kernels."""
    from mlx.utils import tree_flatten, tree_unflatten

    det = _shared().det
    new = []
    for key, value in tree_flatten(module.parameters()):
        name = f"{prefix}.{key}"
        if key.endswith(".gamma"):
            arr = det(name, value.shape, 0.2, 1.0)
        elif key.endswith(".bias"):
            arr = det(name, value.shape, 0.1)
        else:
            fan = int(np.prod(value.shape[1:])) if value.ndim > 1 else 1
            arr = det(name, value.shape, 1.2 / np.sqrt(fan))
        # bf16-representable, so the fixture stores the weights as bf16.
        new.append((key, arr.astype(mx.bfloat16).astype(mx.float32)))
    module.update(tree_unflatten(new))
    mx.eval(module.parameters())


def _vae_torch_weights(module) -> dict:
    """The module's weights as a Diffusers checkpoint stores them (in bf16)."""
    from mlx.utils import tree_flatten

    out = {}
    for key, value in tree_flatten(module.parameters()):
        if key.endswith(".gamma"):
            value = value.reshape(-1, 1, 1, 1)
        elif value.ndim == 4:
            value = value.transpose(0, 3, 1, 2)
        out[key] = value.astype(mx.bfloat16)
    return out


def _write_tiny_transformer(refs: dict) -> None:
    from mlx.utils import tree_flatten, tree_map

    from mlx_vlm.models.qwen_image.kv_cache import QwenImageKVCache
    from mlx_vlm.models.qwen_image.transformer import QwenImageTransformer

    shared = _shared()
    model = QwenImageTransformer(**TINY_TF)
    shared.fill(model, "qit")
    model.update(tree_map(lambda p: p.astype(mx.bfloat16), model.parameters()))
    mx.eval(model.parameters())
    for key, value in tree_flatten(model.parameters()):
        refs[f"tf.w.{_tf_checkpoint_name(key)}"] = value
    det = shared.det
    bf = lambda a: a.astype(mx.bfloat16)  # noqa: E731
    t0, t1 = bf(mx.array([0.875])), bf(mx.array([0.5]))
    # Generate: 5 prompt tokens and a 4x6 target.
    hidden = bf(det("qit.gen_img", (1, 24, 8)))
    text = bf(det("qit.gen_txt", (1, 5, 12)))
    refs["tf.gen.out"] = model(
        hidden_states=hidden, encoder_hidden_states=text, timestep=t0, img_shape=(1, 4, 6)
    )
    # Edit: a 4x4 reference inside the prompt, a 2x6 target.
    reference = bf(det("qit.ref_img", (1, 16, 8)))
    target = bf(det("qit.edit_img", (1, 12, 8)))
    text = bf(det("qit.edit_txt", (1, 10, 12)))
    mask = mx.array([EDIT_SLOTS[0] <= i < EDIT_SLOTS[1] for i in range(10)])
    joint = mx.concatenate([reference, target], axis=1)
    edit = dict(
        encoder_hidden_states=text,
        img_shape=(1, 2, 6),
        reference_image_shapes=[(1, 4, 4)],
        image_pad_mask=mask,
    )
    refs["tf.edit.nokv0"] = model(hidden_states=joint, timestep=t0, **edit)
    refs["tf.edit.nokv1"] = model(hidden_states=joint, timestep=t1, **edit)
    cache = QwenImageKVCache(len(model.transformer_blocks))
    refs["tf.edit.kv0"] = model(
        hidden_states=joint, timestep=t0, kv_cache=cache, kv_cache_mode="extract", **edit
    )
    refs["tf.edit.kv1"] = model(
        hidden_states=target, timestep=t1, kv_cache=cache, kv_cache_mode="cached", **edit
    )
    # RoPE of the edit layout: text[0:3], the reference at 3, text, the target at 22.
    cos, sin = model.pos_embed.for_layout(34, [(3, 4, 4), (22, 2, 6)])
    refs["rope.edit.cos"], refs["rope.edit.sin"] = cos, sin
    # A layout with trailing text after the last block.
    cos, sin = model.pos_embed.for_layout(20, [(2, 3, 5)])
    refs["rope.tail.cos"], refs["rope.tail.sin"] = cos, sin


def _write_tiny_vae(refs: dict) -> None:
    from mlx_vlm.models.qwen_image.vae import QwenImageVAE

    det = _shared().det
    latent = det("qiv.latent", (1, 4, 1, 4, 6))
    image = det("qiv.image", (1, 4, 1, 16, 24)).astype(mx.bfloat16)
    for residual in (True, False):
        tag = "res" if residual else "plain"
        model = QwenImageVAE(**TINY_VAE, is_residual=residual)
        _vae_fill(model, f"qiv.{tag}")
        for key, value in _vae_torch_weights(model).items():
            refs[f"vae_{tag}.w.{key}"] = value
        refs[f"vae_{tag}.decode"] = model.decode(latent)
        refs[f"vae_{tag}.encode_mean"] = model.encode(image)[0]
    refs["vae.latent"] = latent
    refs["vae.image"] = image


def _write_real_vae(refs: dict, model_path: Path) -> None:
    from mlx_vlm.models.qwen_image.weights import load_vae

    det = _shared().det
    vae = load_vae(model_path)
    latent = det("qiv.real_latent", (1, 64, 1, 4, 6))
    image = det("qiv.real_image", (1, 4, 1, 64, 96)).astype(mx.bfloat16)
    refs["vae_real.decode"] = vae.decode(latent).astype(mx.float16)
    refs["vae_real.encode_mean"] = vae.encode(image)[0]


def write_tiny(out: Path, model: Path | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    refs: dict = {}
    _write_tiny_transformer(refs)
    _write_tiny_vae(refs)
    if model is not None:
        _write_real_vae(refs, model)
    mx.eval(list(refs.values()))
    mx.save_safetensors(str(out / "qwen_image_tiny_refs.safetensors"), refs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("tiny", "real"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--cases", default="gen,edit")
    args = parser.parse_args()
    if args.mode == "real":
        write_real(args.out, args.model, args.cases.split(","))
    else:
        write_tiny(args.out, args.model)


if __name__ == "__main__":
    main()
