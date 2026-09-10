#!/usr/bin/env python3
"""Regenerate the Kimi K3 MoonViT3D reference dump (`reference.json`).

Out-of-band tooling: nothing in mlxcel's request path runs Python. This script
is an independent numpy transcription of the tower, merger, projector and
image preprocessing as specified in issue #1342 and in the checkpoint's own
Python sources (`kimi_k3_vision_processing.py`, `media_utils.py`), computed in
float32 from the bf16 vision shards of a local Kimi K3 checkpoint. The Rust
`kimi_k3_tower_real_weights` harness (`src/vision/encoders/moonvit3d_tests.rs`)
compares its projected features against this file, which is what makes the
dump an oracle for the port rather than a copy of its output.

It needs only `numpy` and `pillow`: the safetensors shards are parsed here
(header + raw bf16 bytes), so neither `torch`, `transformers`, `mlx` nor
`safetensors` is imported.

Usage (from the repository root):

    python3 tests/fixtures/kimi_k3_vision/generate_reference.py \
        --checkpoint models/kimi-k3-8l-mxfp4 \
        --image tests/fixtures/test_image.png

The image defaults to a 224x224 RGB fixture already in the repository, which
the navit rule keeps at its size (s = 1, no padding), so the dump does not
depend on Pillow's resampling kernel: only the tower math is under test.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import struct

import numpy as np
from PIL import Image

FIXTURE_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = FIXTURE_DIR.parents[2]

# ---------------------------------------------------------------------------
# Safetensors (bf16) reader
# ---------------------------------------------------------------------------


def read_safetensors_bf16(path: pathlib.Path, keep) -> dict[str, np.ndarray]:
    """Load the tensors of one shard whose name satisfies `keep`, as float32."""
    out: dict[str, np.ndarray] = {}
    with path.open("rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
        base = 8 + header_len
        for name, meta in header.items():
            if name == "__metadata__" or not keep(name):
                continue
            if meta["dtype"] != "BF16":
                raise SystemExit(f"{name}: expected BF16, got {meta['dtype']}")
            start, end = meta["data_offsets"]
            f.seek(base + start)
            raw = np.frombuffer(f.read(end - start), dtype=np.uint16)
            as_f32 = (raw.astype(np.uint32) << 16).view(np.float32)
            out[name] = as_f32.reshape(meta["shape"])
    return out


def load_vision_weights(checkpoint: pathlib.Path) -> dict[str, np.ndarray]:
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    keep = lambda name: name.startswith("vision_tower.") or name.startswith("mm_projector.")
    shards = sorted({shard for name, shard in index["weight_map"].items() if keep(name)})
    weights: dict[str, np.ndarray] = {}
    for shard in shards:
        weights.update(read_safetensors_bf16(checkpoint / shard, keep))
    return weights


# ---------------------------------------------------------------------------
# Preprocessing (media_utils transcription)
# ---------------------------------------------------------------------------


def navit_resize_image(width, height, patch_size, merge_kernel_size, in_patch_limit, side_limit):
    s1 = math.sqrt(in_patch_limit / (max(1.0, width // patch_size) * max(1.0, height // patch_size)))
    s2 = side_limit * patch_size / width
    s3 = side_limit * patch_size / height
    scale = min(1.0, s1, s2, s3)
    new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
    new_w = min(new_w, side_limit * patch_size)
    new_h = min(new_h, side_limit * patch_size)
    factor = merge_kernel_size * patch_size
    pad_h = (factor - new_h % factor) % factor
    pad_w = (factor - new_w % factor) % factor
    token_h = (new_h + pad_h) // factor
    token_w = (new_w + pad_w) // factor
    return dict(num_tokens=token_h * token_w, new_width=new_w, new_height=new_h, pad_width=pad_w, pad_height=pad_h)


def chessboard(height, width, square, white_top_left, white, gray):
    bg = np.ones((height, width, 3), dtype=np.uint8) * white
    for y in range(0, height, square):
        for x in range(0, width, square):
            if (y // square + x // square) % 2 == (1 if white_top_left else 0):
                bg[y : y + square, x : x + square] = gray
    return bg


def fill_transparent(image: Image.Image, bg_cfg: dict | None) -> Image.Image:
    if bg_cfg is None or image.mode == "RGB":
        return image.convert("RGB")
    if "A" not in image.getbands() and "transparency" not in image.info:
        return image.convert("RGB")
    img = np.array(image.convert("RGBA"))
    h, w = img.shape[:2]
    pattern = bg_cfg["pattern"]
    if pattern == "chessboard":
        bg = chessboard(
            h,
            w,
            bg_cfg["chessboard_square_size"],
            bg_cfg["chessboard_square_on_top_left"],
            bg_cfg["chessboard_white_value"],
            bg_cfg["chessboard_gray_value"],
        )
    else:
        value = {"white": 255, "black": 0, "gray": 128}[pattern]
        bg = np.full((h, w, 3), value, dtype=np.uint8)
    alpha = img[:, :, 3].astype(np.float32) / 255.0
    alpha3 = np.stack([alpha] * 3, axis=2)
    result = alpha3 * img[:, :, :3] + (1 - alpha3) * bg
    return Image.fromarray(result.astype(np.uint8))


def preprocess(image_path: pathlib.Path, cfg: dict):
    image = Image.open(image_path)
    w, h = image.size
    bg_cfg = cfg.get("transparent_bg_config")
    stage = cfg.get("transparent_bg_fill_stage", "before_resize")
    if stage == "before_resize":
        image = fill_transparent(image, bg_cfg)
    plan = navit_resize_image(
        w, h, cfg["patch_size"], cfg["merge_kernel_size"], cfg["in_patch_limit"], cfg["patch_limit_on_one_side"]
    )
    image = image.resize((plan["new_width"], plan["new_height"]), resample=Image.Resampling.BICUBIC)
    if stage == "after_resize":
        image = fill_transparent(image, bg_cfg)
    arr = np.asarray(image.convert("RGB"))
    arr = np.pad(arr, ((0, plan["pad_height"]), (0, plan["pad_width"]), (0, 0)), mode="constant", constant_values=0)
    arr = np.expand_dims(arr, 0)  # (t, H, W, C)
    x = (arr / 255.0).astype(np.float32)
    x -= np.array(cfg["image_mean"])
    x *= 1.0 / np.array(cfg["image_std"])
    p = cfg["patch_size"]
    t, hh, ww, c = x.shape
    patches = x.reshape(t, hh // p, p, ww // p, p, c).transpose(0, 1, 3, 5, 2, 4).reshape(-1, c, p, p)
    grid_thw = (t, hh // p, ww // p)
    return patches.astype(np.float32), grid_thw, (w, h), plan


# ---------------------------------------------------------------------------
# Tower (issue #1342 specification)
# ---------------------------------------------------------------------------


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    var = np.mean(np.square(x, dtype=np.float32), axis=-1, keepdims=True, dtype=np.float32)
    return (x * (1.0 / np.sqrt(var + eps))).astype(np.float32) * weight


def bilinear_axis(in_size: int, out_size: int) -> np.ndarray:
    m = np.zeros((out_size, in_size), dtype=np.float32)
    for i in range(out_size):
        src = min(max((i + 0.5) * in_size / out_size - 0.5, 0.0), in_size - 1)
        i0 = int(math.floor(src))
        i1 = min(i0 + 1, in_size - 1)
        frac = src - i0
        m[i, i0] += 1.0 - frac
        m[i, i1] += frac
    return m


def pos2d(pos_emb: np.ndarray, h: int, w: int) -> np.ndarray:
    in_h, in_w, dim = pos_emb.shape
    if (in_h, in_w) == (h, w):
        return pos_emb.astype(np.float32)
    rows = np.einsum("oi,ijd->ojd", bilinear_axis(in_h, h), pos_emb.astype(np.float32))
    return np.einsum("oj,ijd->iod", bilinear_axis(in_w, w), rows).astype(np.float32)


def time_table(t: int, dim: int) -> np.ndarray:
    half = dim // 2
    omega = 1.0 / (10000.0 ** (np.arange(half, dtype=np.float32) / half))
    k = np.arange(t, dtype=np.float32)[:, None]
    return np.concatenate([np.sin(k * omega), np.cos(k * omega)], axis=1).astype(np.float32)


def rope2d_cos_sin(t: int, h: int, w: int, head_dim: int):
    freqs = 1.0 / (10000.0 ** (np.arange(0, head_dim, 4, dtype=np.float32)[: head_dim // 4] / head_dim))
    ys, xs = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    angles = np.stack([xs[..., None] * freqs, ys[..., None] * freqs], axis=-1).reshape(h * w, head_dim // 2)
    angles = np.tile(angles, (t, 1))
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    # x: [L, heads, head_dim]; interleaved pairs (even, odd).
    e = x[..., 0::2]
    o = x[..., 1::2]
    c = cos[:, None, :]
    s = sin[:, None, :]
    out = np.empty_like(x)
    out[..., 0::2] = e * c - o * s
    out[..., 1::2] = e * s + o * c
    return out


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    return (0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))).astype(np.float32)


_erf = np.frompyfunc(math.erf, 1, 1)


def gelu_erf(x: np.ndarray) -> np.ndarray:
    return (0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)).astype(np.float32))).astype(np.float32)


def attention(q, k, v, scale):
    # q, k, v: [L, heads, head_dim] -> [L, heads * head_dim]
    qh = q.transpose(1, 0, 2)
    kh = k.transpose(1, 0, 2)
    vh = v.transpose(1, 0, 2)
    scores = np.einsum("hld,hmd->hlm", qh, kh) * np.float32(scale)
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    out = np.einsum("hlm,hmd->hld", probs.astype(np.float32), vh)
    return out.transpose(1, 0, 2).reshape(q.shape[0], -1)


def stats(x: np.ndarray) -> dict:
    return {
        "mean": float(x.mean()),
        "mean_abs": float(np.abs(x).mean()),
        "std": float(x.std()),
    }


def run_tower(weights: dict[str, np.ndarray], vcfg: dict, patches: np.ndarray, grid_thw):
    t, h, w = grid_thw
    hidden = vcfg["vt_hidden_size"]
    heads = vcfg["vt_num_attention_heads"]
    head_dim = vcfg["qkv_hidden_size"] // heads
    eps = 2.0**-7
    stages: dict[str, dict] = {}

    # Patch embedding: conv2d with kernel == stride == patch is a matmul over
    # the flattened [3, 14, 14] patch.
    proj = weights["vision_tower.patch_embed.proj.weight"]  # [1024, 3, 14, 14]
    x = patches.reshape(patches.shape[0], -1) @ proj.reshape(hidden, -1).T
    pos = pos2d(weights["vision_tower.patch_embed.pos_emb.weight"], h, w)
    if t > 1:
        pos = pos[None] + time_table(t, hidden)[:, None, None, :]
    x = (x + pos.reshape(t * h * w, hidden)).astype(np.float32)
    stages["patch_embed"] = stats(x)

    cos, sin = rope2d_cos_sin(t, h, w, head_dim)
    scale = head_dim**-0.5
    for i in range(vcfg["vt_num_hidden_layers"]):
        p = f"vision_tower.encoder.blocks.{i}"
        n = rms_norm(x, weights[f"{p}.norm0.weight"], eps)
        qkv = (n @ weights[f"{p}.wqkv.weight"].T).reshape(-1, 3, heads, head_dim)
        q = apply_rope(qkv[:, 0], cos, sin)
        k = apply_rope(qkv[:, 1], cos, sin)
        v = qkv[:, 2]
        a = attention(q, k, v, scale)
        x = x + a @ weights[f"{p}.wo.weight"].T
        n = rms_norm(x, weights[f"{p}.norm1.weight"], eps)
        x = x + gelu_tanh(n @ weights[f"{p}.mlp.fc0.weight"].T) @ weights[f"{p}.mlp.fc1.weight"].T
        x = x.astype(np.float32)
        if i in (0, vcfg["vt_num_hidden_layers"] // 2, vcfg["vt_num_hidden_layers"] - 1):
            stages[f"block_{i}"] = stats(x)
    x = rms_norm(x, weights["vision_tower.encoder.final_layernorm.weight"], eps)
    stages["final_norm"] = stats(x)

    # sd2_tpool merge.
    kh, kw = vcfg["merge_kernel_size"]
    merged = x.reshape(t, h // kh, kh, w // kw, kw, hidden).transpose(0, 1, 3, 2, 4, 5).mean(axis=0)
    merged = merged.reshape((h // kh) * (w // kw), kh * kw, hidden)
    stages["merged"] = stats(merged)

    # patchmergerv2 projector.
    flat = merged.reshape(merged.shape[0], -1)
    y = flat @ weights["mm_projector.proj.0.weight"].T
    y = gelu_erf(y)
    y = y @ weights["mm_projector.proj.2.weight"].T
    y = rms_norm(y, weights["mm_projector.post_norm.weight"], vcfg["projector_ln_eps"])
    return stages, y.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="models/kimi-k3-8l-mxfp4", help="checkpoint directory (relative to the repo root)")
    parser.add_argument("--image", default="tests/fixtures/test_image.png", help="image path (relative to the repo root)")
    parser.add_argument("--output", default=str(FIXTURE_DIR / "reference.json"))
    parser.add_argument("--sample-col-stride", type=int, default=64)
    args = parser.parse_args()

    checkpoint = REPO_ROOT / args.checkpoint
    image_path = REPO_ROOT / args.image
    config = json.loads((checkpoint / "config.json").read_text())
    vcfg = config["vision_config"]
    media_cfg = json.loads((checkpoint / "preprocessor_config.json").read_text())["media_proc_cfg"]

    patches, grid_thw, original_size, plan = preprocess(image_path, media_cfg)
    weights = load_vision_weights(checkpoint)
    if len(weights) != 168:
        raise SystemExit(f"expected 168 vision tensors, found {len(weights)}")
    stages, projected = run_tower(weights, vcfg, patches, grid_thw)

    sample = projected[:, :: args.sample_col_stride]
    dump = {
        "generator": "tests/fixtures/kimi_k3_vision/generate_reference.py",
        "checkpoint": args.checkpoint,
        "image": args.image,
        "original_size": list(original_size),
        "grid_thw": list(grid_thw),
        "num_tokens": plan["num_tokens"],
        "pixel_values": {
            "shape": list(patches.shape),
            "mean_abs": float(np.abs(patches).mean()),
            "head": [float(v) for v in patches.reshape(-1)[:16]],
        },
        "stages": stages,
        "projected": {
            "shape": list(projected.shape),
            **stats(projected),
            "row_mean_abs": [float(v) for v in np.abs(projected).mean(axis=1)],
            "sample_col_stride": args.sample_col_stride,
            "sample": [[float(v) for v in row] for row in sample],
        },
    }
    pathlib.Path(args.output).write_text(json.dumps(dump, indent=1) + "\n")
    print(f"wrote {args.output}: grid {grid_thw}, {plan['num_tokens']} tokens, projected {projected.shape}")
    for name, st in stages.items():
        print(f"  {name}: mean {st['mean']:+.5f} mean_abs {st['mean_abs']:.5f} std {st['std']:.5f}")
    print(f"  projected: mean_abs {dump['projected']['mean_abs']:.5f} std {dump['projected']['std']:.5f}")


if __name__ == "__main__":
    main()
