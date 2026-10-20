#!/usr/bin/env python3
"""Generate the image-generation parity fixtures from the upstream Python.

Everything under ``tests/fixtures/imagegen/`` and the scheduler tables in
``src/imagegen/scheduler_fixtures.rs`` are outputs of the upstream reference
implementation (Blaizzy/mlx-vlm), run unmodified except where noted. The
fixtures are committed, so this script documents how they were made rather than
being run by the tests.

What it writes to ``--out``:

* ``scheduler_fixtures.rs``      sigma/timestep tables of the qwen_image and
                                 mage_flow ``FlowMatchEulerDiscreteScheduler``
* ``vae_tiny.safetensors``       random-weight tiny FLUX.2 VAE (Diffusers key
                                 names, PyTorch conv layout) + upstream outputs
* ``vae_ernie_refs.safetensors`` upstream f32 outputs of the real
                                 baidu/ERNIE-Image-Turbo VAE (needs ``--ernie``)
* ``dit_blocks_refs.safetensors``upstream outputs of the Z-Image DiT blocks for
                                 weights/inputs derived from a closed-form
                                 generator (also implemented in
                                 ``src/imagegen/dit_blocks_tests.rs``)
* ``qwen3_encoder_refs.safetensors`` upstream ``Qwen3TextEncoder`` hidden states
                                 and stacked prompt embeds for a tiny model with
                                 closed-form weights (``det`` again, mirrored in
                                 ``src/imagegen/qwen3_text_encoder_tests.rs``)

Provenance of the committed files: Blaizzy/mlx-vlm ``main`` fetched 2026-10-20
(latest commit 9527f2da, 2026-10-19), ``mlx`` 0.32.4.

One deliberate deviation from stock upstream: the real-checkpoint VAE reference
transposes ``encoder.conv_in.weight`` explicitly. Upstream's
``_match_conv_layout`` returns a weight unchanged when its shape already equals
the target, and that RGB conv is ``(128, 3, 3, 3)`` in both the PyTorch and the
MLX layout, so upstream keeps the PyTorch data order and its encoder is wrong
(encode -> decode reconstructs with rel-RMS 0.96 instead of 0.096).

Setup and usage::

    git clone https://github.com/Blaizzy/mlx-vlm /tmp/mlx-vlm
    python3 -m venv /tmp/venv && /tmp/venv/bin/pip install mlx numpy
    huggingface-cli download baidu/ERNIE-Image-Turbo --include "vae/*" \\
        "model_index.json" --local-dir models/ERNIE-Image-Turbo
    /tmp/venv/bin/python -I tests/fixtures/generate_imagegen_fixtures.py \\
        --mlx-vlm /tmp/mlx-vlm --ernie models/ERNIE-Image-Turbo --out /tmp/imagegen

Copy the outputs to ``tests/fixtures/imagegen/`` (and the scheduler table to
``src/imagegen/scheduler_fixtures.rs`` under its license header). The real-VAE
file keeps only the ``f32`` references; the script also writes
``vae_ernie_refs_all.safetensors`` with the bf16 ones, which measure upstream's
own precision.
"""

import argparse
import functools
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten


def build_shim(mlx_vlm: Path) -> Path:
    """A minimal ``mlx_vlm`` package so the upstream files import unmodified."""
    root = Path(tempfile.mkdtemp(prefix="imagegen-shim-"))
    models = root / "mlx_vlm" / "models"
    for pkg in (root / "mlx_vlm", models, models / "flux2", models / "z_image"):
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "__init__.py").touch()
    (models / "flux2" / "constants.py").write_text(
        "import mlx.core as mx\nclass ModelConfig:\n    precision = mx.bfloat16\n"
    )
    (models / "flux2" / "tiling.py").write_text(
        "class TilingConfig:\n    vae_decode_tiles_per_dim = None\n"
        "def decode_image_tiled(*a, **k):\n    raise NotImplementedError\n"
    )
    src = mlx_vlm / "mlx_vlm" / "models"
    shutil.copytree(src / "flux2" / "vae", models / "flux2" / "vae", dirs_exist_ok=True)
    shutil.copytree(src / "flux2" / "qwen", models / "flux2" / "qwen", dirs_exist_ok=True)
    for name in ("transformer.py", "config.py"):
        shutil.copy(src / "z_image" / name, models / "z_image" / name)
    sys.path.insert(0, str(root))
    return root


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ schedulers
def write_scheduler_tables(mlx_vlm: Path, out: Path) -> None:
    models = mlx_vlm / "mlx_vlm" / "models"
    qi = load_module(models / "qwen_image" / "scheduler.py", "qi_sched")
    mf = load_module(models / "mage_flow" / "scheduler.py", "mf_sched")

    def lit(arr):
        a = np.array(arr).astype(np.float32)
        return "&[" + ", ".join("{:.9e}".format(float(x)) for x in a) + "]"

    lines = [
        "// Literals carry all nine significant digits of an f32 on purpose.",
        "#![allow(clippy::excessive_precision)]",
        "",
        f"// Generated once from the upstream Python (Blaizzy/mlx-vlm, mlx {mx.__version__}):",
        "//   mlx_vlm/models/qwen_image/scheduler.py  FlowMatchEulerDiscreteScheduler",
        "//   mlx_vlm/models/mage_flow/scheduler.py   FlowMatchEulerDiscreteScheduler",
        "// Each entry: (steps, image_seq_len, shift_terminal, sigmas, timesteps).",
        "pub(super) type DynamicCase = (u32, u32, Option<f32>, &'static [f32], &'static [f32]);",
        "pub(super) const DYNAMIC_CASES: &[DynamicCase] = &[",
    ]
    for steps in (1, 4, 12, 30):
        for seq in (1024, 4096):
            for term in (0.02, None):
                s = qi.FlowMatchEulerDiscreteScheduler(
                    image_seq_len=seq, num_inference_steps=steps, shift_terminal=term
                )
                mx.eval(s.sigmas, s.timesteps)
                t = f"Some({term})" if term is not None else "None"
                lines.append(f"    ({steps}, {seq}, {t}, {lit(s.sigmas)}, {lit(s.timesteps)}),")
    lines += [
        "];",
        "",
        "// (steps, shift, sigmas, timesteps) for the static-shift scheduler.",
        "pub(super) type StaticCase = (u32, f32, &'static [f32], &'static [f32]);",
        "pub(super) const STATIC_CASES: &[StaticCase] = &[",
    ]
    for steps in (1, 4, 8, 30):
        for shift in (6.0, 3.0):
            s = mf.FlowMatchEulerDiscreteScheduler(num_inference_steps=steps, shift=shift)
            mx.eval(s.sigmas, s.timesteps)
            lines.append(f"    ({steps}, {shift!r}, {lit(s.sigmas)}, {lit(s.timesteps)}),")
    lines.append("];")
    (out / "scheduler_fixtures.rs").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------------ VAE
def to_torch_names(items):
    """MLX parameter names/layout -> Diffusers names/PyTorch layout."""
    out = {}
    for key, value in items:
        key = key.replace(".to_out.", ".to_out.0.")
        if value.ndim == 4:
            value = value.transpose(0, 3, 1, 2)
        out[key] = value
    return out


def write_vae_fixtures(ernie: Path | None, out: Path) -> None:
    import mlx_vlm.models.flux2.vae.vae as vae_mod
    from mlx_vlm.models.flux2.constants import ModelConfig
    from mlx_vlm.models.flux2.vae.decoder.decoder import Flux2Decoder
    from mlx_vlm.models.flux2.vae.encoder.encoder import Flux2Encoder
    from mlx_vlm.models.flux2.vae.vae import Flux2VAE

    # Tiny model: 4 latent channels, block channels 4/4/8/8, 2 norm groups.
    ModelConfig.precision = mx.float32
    saved = (vae_mod.Flux2Decoder, vae_mod.Flux2Encoder, Flux2VAE.latent_channels)
    vae_mod.Flux2Decoder = functools.partial(Flux2Decoder, in_channels=4, norm_num_groups=2)
    vae_mod.Flux2Encoder = functools.partial(Flux2Encoder, out_channels=4, norm_num_groups=2)
    Flux2VAE.latent_channels = 4
    tiny = Flux2VAE(
        decoder_block_out_channels=(4, 4, 8, 8),
        include_encoder=True,
        encoder_block_out_channels=(4, 4, 8, 8),
    )
    rs = np.random.RandomState(1234)
    new = []
    for key, value in tree_flatten(tiny.parameters()):
        if key.endswith("running_var"):
            arr = 0.5 + rs.rand(*value.shape)
        elif key.endswith("running_mean"):
            arr = 0.1 * rs.randn(*value.shape)
        elif "norm" in key and key.endswith(".weight"):
            arr = 1.0 + 0.1 * rs.randn(*value.shape)
        elif key.endswith(".bias"):
            arr = 0.1 * rs.randn(*value.shape)
        else:
            fan = int(np.prod(value.shape[1:])) if value.ndim > 1 else 1
            arr = rs.randn(*value.shape) / np.sqrt(max(fan, 1))
        new.append((key, mx.array(arr.astype(np.float32))))
    tiny.update(tree_unflatten(new))
    mx.eval(tiny.parameters())

    latents = mx.array(rs.randn(1, 4, 4, 4).astype(np.float32))
    packed = mx.array(rs.randn(1, 16, 2, 2).astype(np.float32))
    image = mx.array((0.5 * rs.randn(1, 3, 32, 32)).astype(np.float32))
    decode = tiny.decode(latents)
    decode_packed = tiny.decode_packed_latents(packed)
    encode = tiny.encode(image)
    mx.eval(decode, decode_packed, encode)
    tensors = to_torch_names(tree_flatten(tiny.parameters()))
    # Keys a loader must ignore: the batch-norm counter and an unrelated tensor.
    tensors["bn.num_batches_tracked"] = mx.array(np.array([7], dtype=np.int64))
    tensors["unused.weight"] = mx.array(np.ones((2, 2), dtype=np.float32))
    for name, arr in [
        ("latents", latents),
        ("decode", decode),
        ("packed", packed),
        ("decode_packed", decode_packed),
        ("image", image),
        ("encode", encode),
    ]:
        tensors["ref." + name] = arr
    mx.save_safetensors(str(out / "vae_tiny.safetensors"), tensors)
    vae_mod.Flux2Decoder, vae_mod.Flux2Encoder, Flux2VAE.latent_channels = saved

    if ernie is None:
        return
    raw = mx.load(str(ernie / "vae" / "diffusion_pytorch_model.safetensors"))

    def build(dtype):
        ModelConfig.precision = dtype
        vae = Flux2VAE(
            decoder_block_out_channels=(128, 256, 512, 512),
            include_encoder=True,
            encoder_block_out_channels=(128, 256, 512, 512),
        )
        shapes = {k: tuple(x.shape) for k, x in tree_flatten(vae.parameters())}
        prefixes = ("decoder.", "post_quant_conv.", "bn.", "encoder.", "quant_conv.")
        weights = {}
        for key, value in raw.items():
            if key.endswith(".num_batches_tracked") or not key.startswith(prefixes):
                continue
            mapped = key.replace(".to_out.0.", ".to_out.")
            t = value.astype(dtype) if mx.issubdtype(value.dtype, mx.floating) else value
            if t.ndim == 4 and tuple(t.shape) != shapes.get(mapped):
                t = t.transpose(0, 2, 3, 1)
            elif t.ndim == 4 and mapped == "encoder.conv_in.weight":
                t = t.transpose(0, 2, 3, 1)  # see the module docstring
            weights[mapped] = t
        vae.update(tree_unflatten(list(weights.items())), strict=True)
        mx.eval(vae.parameters())
        return vae

    rs = np.random.RandomState(99)
    latents = mx.array(rs.randn(1, 32, 8, 8).astype(np.float32))
    packed = mx.array(rs.randn(1, 128, 4, 4).astype(np.float32))
    image = mx.array(np.clip(0.5 * rs.randn(1, 3, 64, 64), -1, 1).astype(np.float32))
    refs = {"ref.latents": latents, "ref.packed": packed, "ref.image": image}
    for tag, dtype in [("f32", mx.float32), ("bf16", mx.bfloat16)]:
        vae = build(dtype)
        d = vae.decode(latents.astype(dtype))
        dp = vae.decode_packed_latents(packed.astype(dtype))
        e = vae.encode(image.astype(dtype))
        mx.eval(d, dp, e)
        refs[f"ref.{tag}.decode"] = d.astype(mx.float32)
        refs[f"ref.{tag}.decode_packed"] = dp.astype(mx.float32)
        refs[f"ref.{tag}.encode"] = e.astype(mx.float32)
    mx.save_safetensors(str(out / "vae_ernie_refs_all.safetensors"), refs)
    mx.save_safetensors(
        str(out / "vae_ernie_refs.safetensors"),
        {k: v for k, v in refs.items() if "bf16" not in k},
    )


# ------------------------------------------------------------------- DiT blocks
def fnv1a32(text: str) -> int:
    h = 0x811C9DC5
    for b in text.encode():
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def det(name, shape, scale=1.0, offset=0.0):
    """Closed-form tensor keyed by ``name``; mirrored by ``det`` in dit_blocks_tests.rs."""
    n = int(np.prod(shape))
    phase = (fnv1a32(name) % 6283) / 1000.0
    i = np.arange(n, dtype=np.float64)
    v = (np.sin(0.37 * i + phase) + 0.5 * np.cos(1.13 * i + 2.0 * phase)) / 1.5
    return mx.array((offset + scale * v).astype(np.float32).reshape(shape))


def fill(module, prefix):
    new = []
    for key, value in tree_flatten(module.parameters()):
        name = f"{prefix}.{key}"
        owner = key.split(".")[-2] if "." in key else ""
        if key.endswith(("norm_q.weight", "norm_k.weight")) or (
            "norm" in owner and key.endswith(".weight")
        ):
            arr = det(name, value.shape, 0.2, 1.0)
        elif key.endswith(".bias"):
            arr = det(name, value.shape, 0.1)
        else:
            fan = value.shape[-1] if value.ndim > 1 else 1
            arr = det(name, value.shape, 0.8 / np.sqrt(fan))
        new.append((key, arr))
    module.update(tree_unflatten(new))
    mx.eval(module.parameters())


def write_dit_refs(out: Path) -> None:
    from mlx_vlm.models.z_image.transformer import (
        FeedForward,
        FinalLayer,
        MRoPE,
        TimestepEmbedder,
        ZImageTransformerBlock,
        apply_rotary,
    )

    dim, heads, head_dim, mlp, adaln, out_dim = 32, 2, 16, 48, 24, 12
    refs = {}
    te = TimestepEmbedder(adaln, frequency_size=16)
    fill(te, "t_embedder")
    refs["t_embed"] = te(mx.array(np.array([0.25, 0.75], dtype=np.float32)) * 1000.0)

    ff = FeedForward(dim, mlp)
    fill(ff, "feed_forward")
    x = det("x", (1, 6, dim))
    refs["feed_forward"] = ff(x)

    pos = mx.array(
        np.array(
            [[[1, 0, 0], [2, 0, 0], [3, 0, 0], [7, 0, 0], [7, 0, 1], [7, 1, 0]]],
            dtype=np.float32,
        )
    )
    cos, sin = MRoPE((4, 6, 6), 256.0).compute_freqs(pos)
    refs["rope_cos"], refs["rope_sin"] = cos, sin
    refs["apply_rotary"] = apply_rotary(det("q", (1, 6, heads, head_dim)), cos, sin)

    block = ZImageTransformerBlock(dim, heads, head_dim, mlp, modulation=True, adaln_dim=adaln)
    fill(block, "layers.0")
    ada = det("adaln", (1, adaln))
    refs["block_mod"] = block(x, cos, sin, adaln_input=ada)
    mask = mx.array(np.triu(np.full((6, 6), -1e4, dtype=np.float32), k=1))[None, None]
    refs["block_mod_masked"] = block(x, cos, sin, adaln_input=ada, mask=mask)

    plain = ZImageTransformerBlock(dim, heads, head_dim, mlp, modulation=False)
    fill(plain, "context_refiner.0")
    refs["block_plain"] = plain(x, cos, sin)

    final = FinalLayer(dim, out_dim, adaln_dim=adaln)
    fill(final, "final_layer")
    refs["final_layer"] = final(x, ada)

    mx.eval(list(refs.values()))
    mx.save_safetensors(str(out / "dit_blocks_refs.safetensors"), refs)


def write_qwen3_encoder_refs(out: Path) -> None:
    from mlx_vlm.models.flux2.qwen.text_encoder import Qwen3TextEncoder

    hidden, layers, heads, kv_heads, head_dim, inter, vocab = 32, 3, 2, 1, 16, 48, 64
    encoder = Qwen3TextEncoder(
        vocab_size=vocab,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        intermediate_size=inter,
        max_position_embeddings=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        head_dim=head_dim,
    )
    new = []
    for key, value in tree_flatten(encoder.parameters()):
        if key.startswith("rotary_emb."):
            continue  # a computed table that upstream registers as a parameter
        name = "model." + key  # the Rust side names weights as the model loader does
        owner = key.split(".")[-2] if "." in key else ""
        if "norm" in owner or key == "norm.weight":
            arr = det(name, value.shape, 0.2, 1.0)
        elif key == "embed_tokens.weight":
            arr = det(name, value.shape, 0.5)
        else:
            fan = value.shape[-1] if value.ndim > 1 else 1
            arr = det(name, value.shape, 0.8 / np.sqrt(fan))
        new.append((key, arr))
    encoder.update(tree_unflatten(new))
    mx.eval(encoder.parameters())

    ids = mx.array(np.array([[3, 10, 7, 22, 5, 9], [4, 8, 15, 2, 0, 0]], dtype=np.int32))
    mask = mx.array(np.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], dtype=np.int32))
    _, states = encoder(ids, mask, output_hidden_states=True)
    embeds = encoder.get_prompt_embeds(ids, mask, hidden_state_layers=(1, 3))
    refs = {
        "input_ids": ids,
        "attention_mask": mask,
        "hidden_states": mx.stack(states, axis=0),
        "prompt_embeds_1_3": embeds,
    }
    mx.eval(list(refs.values()))
    mx.save_safetensors(str(out / "qwen3_encoder_refs.safetensors"), refs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mlx-vlm", type=Path, required=True, help="checkout of Blaizzy/mlx-vlm")
    parser.add_argument("--ernie", type=Path, help="models/ERNIE-Image-Turbo (vae/ downloaded)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    build_shim(args.mlx_vlm)
    write_scheduler_tables(args.mlx_vlm, args.out)
    write_vae_fixtures(args.ernie, args.out)
    write_dit_refs(args.out)
    write_qwen3_encoder_refs(args.out)
    print(f"wrote fixtures to {args.out}")


if __name__ == "__main__":
    main()
