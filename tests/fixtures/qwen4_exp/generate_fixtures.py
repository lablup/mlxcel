#!/usr/bin/env python3
"""Regenerate the Qwen4-Exp (Qwen3.8-Flash-Next) mlx-vlm parity fixtures.

Out-of-band tooling: nothing in mlxcel's request path runs Python. The
fixtures pin mlxcel's `src/models/qwen4_exp` port against mlx-vlm's
`mlx_vlm/models/qwen4_exp` running the same tiny weights.

No weight is stored. Every weight and every input tensor is generated from
its name by a counter hash that `src/models/qwen4_exp_parity_tests.rs`
reimplements bit for bit (`hashed_tensor` below), so each JSON file records
only the config, the tensor names with their shapes and scales, and the
upstream outputs. All tensors are float32, so the comparison measures the
port rather than bf16 rounding.

Usage (mlx-vlm checked out at the revision recorded in each fixture):

    uv venv /tmp/q4venv && uv pip install --python /tmp/q4venv/bin/python mlx numpy
    uv pip install --python /tmp/q4venv/bin/python --no-deps -e <mlx-vlm checkout>
    /tmp/q4venv/bin/python -I tests/fixtures/qwen4_exp/generate_fixtures.py
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.models.qwen4_exp.config import TextConfig
from mlx_vlm.models.qwen4_exp.language import (
    LanguageModel,
    Qwen4ExpGatedDeltaNet,
    Qwen4ExpGatedResidual,
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLELayer,
    Qwen4ExpQSAIndexer,
    Qwen4ExpRMSNorm,
    QSAKVCache,
)
from mlx_vlm.models.qwen3_5.language import Qwen3_5RotaryEmbedding

FIXTURE_DIR = Path(__file__).resolve().parent

MASK64 = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
M1 = 0xBF58476D1CE4E5B9
M2 = 0x94D049BB133111EB


# ---------------------------------------------------------------------------
# Deterministic tensors
# ---------------------------------------------------------------------------


def fnv1a64(name: str) -> int:
    value = 0xCBF29CE484222325
    for byte in name.encode("utf-8"):
        value ^= byte
        value = (value * 0x100000001B3) & MASK64
    return value


def hashed_tensor(name: str, shape, scale: float, offset: float = 0.0) -> np.ndarray:
    """`offset + scale * (2u - 1)` with `u` a 24-bit uniform drawn per element.

    Element `i` hashes `fnv1a64(name) + (i + 1) * GAMMA` through splitmix64
    and keeps the top 24 bits, so `k * 2**-23 - 1` is exact in float32 and
    the only roundings are the final multiply and add, which are IEEE float32
    on both sides.
    """
    count = int(np.prod(shape)) if len(shape) else 1
    with np.errstate(over="ignore"):
        index = np.arange(1, count + 1, dtype=np.uint64)
        z = np.uint64(fnv1a64(name)) + index * np.uint64(GAMMA)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(M1)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(M2)
        z = z ^ (z >> np.uint64(31))
    k = (z >> np.uint64(40)).astype(np.float32)
    unit = k * np.float32(2.0**-23) - np.float32(1.0)
    values = unit * np.float32(scale) + np.float32(offset)
    return values.astype(np.float32).reshape(shape)


def tensor_scale(name: str, shape) -> tuple[float, float]:
    """Scale and offset for one parameter, chosen so activations stay O(1)."""
    leaf = name.rsplit(".", 1)[-1]
    if leaf == "A_log":
        return 0.5, 0.0
    if leaf == "dt_bias":
        return 0.5, 1.0
    if name.endswith("linear_attn.norm.weight"):
        return 0.1, 1.0
    if "norm" in name and leaf == "weight":
        # Zero-centered Qwen4 norms: the checkpoint stores `w`, the module
        # multiplies by `1 + w`.
        return 0.2, 0.0
    if "conv1d" in name:
        return 0.4, 0.0
    if "ngram_embedding" in name or name.endswith("embed_tokens.weight"):
        return 1.0, 0.0
    fan_in = shape[-1]
    return 1.0 / math.sqrt(fan_in), 0.0


def build_weights(module, prefix: str = "", skip=()) -> list[dict]:
    """Overwrite every float parameter of `module` with a hashed tensor."""
    from mlx.utils import tree_flatten, tree_unflatten

    specs = []
    updates = []
    for name, value in tree_flatten(module.parameters()):
        if any(token in name for token in skip):
            continue
        if value.dtype != mx.float32:
            continue
        full = prefix + name
        scale, offset = tensor_scale(full, value.shape)
        updates.append((name, mx.array(hashed_tensor(full, value.shape, scale, offset))))
        specs.append(
            {"name": full, "shape": list(value.shape), "scale": scale, "offset": offset}
        )
    module.update(tree_unflatten(updates))
    mx.eval(module.parameters())
    return specs


def hashed_input(name: str, shape, scale: float = 1.0) -> mx.array:
    return mx.array(hashed_tensor(name, shape, scale))


def as_list(x: mx.array) -> list:
    return np.array(x.astype(mx.float32)).reshape(-1).tolist()


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

EOS = 1

TINY_CONFIG = {
    "model_type": "qwen4_exp_text",
    "hidden_size": 32,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "linear_num_value_heads": 4,
    "linear_num_key_heads": 2,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 4,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "shared_expert_intermediate_size": 24,
    "moe_intermediate_size": 16,
    "rms_norm_eps": 1e-6,
    "vocab_size": 64,
    "max_position_embeddings": 4096,
    "hc_count": 4,
    "hc_lowrank": 16,
    "full_attention_interval": 4,
    "ple_layer_ids": [2],
    "ple_embed_dim": 32,
    "ple_conv_kernel_size": 4,
    "ngram_size": 3,
    "heads_per_ngram": 2,
    "ngram_vocab_size_base": 50,
    "make_ngram_vocab_size_divisible_by": 8,
    "split_ngram_parts": 3,
    "seed": 1234,
    "indexer_n_heads": 2,
    "indexer_kv_heads": 1,
    "indexer_head_dim": 16,
    "indexer_budget": 16,
    "indexer_compress_ratio": 4,
    "output_gate_type": "sigmoid",
    "norm_topk_prob": True,
    "eos_token_id": EOS,
    "tie_word_embeddings": False,
    "rope_parameters": {
        "mrope_interleaved": True,
        "mrope_section": [2, 1, 1],
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
        "rope_type": "default",
    },
}

# The released checkpoint's text config, reduced to what the n-gram hash reads.
RELEASED_NGRAM_CONFIG = {
    "vocab_size": 248320,
    "ngram_size": 3,
    "heads_per_ngram": 8,
    "ngram_vocab_size_base": 20_000_000,
    "make_ngram_vocab_size_divisible_by": 128,
    "seed": 1234,
    "eos_token_id": 248044,
}


def text_config(overrides=None) -> TextConfig:
    raw = dict(TINY_CONFIG)
    raw["rope_parameters"] = dict(raw["rope_parameters"])
    raw.update(overrides or {})
    return TextConfig(**raw)


def rotary_for(config: TextConfig) -> Qwen3_5RotaryEmbedding:
    rope = config.rope_parameters
    return Qwen3_5RotaryEmbedding(
        int(config.head_dim * rope["partial_rotary_factor"]),
        max_position_embeddings=config.max_position_embeddings,
        base=rope["rope_theta"],
        mrope_section=rope["mrope_section"],
    )


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def norms_case() -> dict:
    config = text_config()
    plain = Qwen4ExpRMSNorm(32, eps=config.rms_norm_eps)
    grouped = Qwen4ExpRMSNorm(4 * 8, group_size=8, eps=config.rms_norm_eps)
    specs = build_weights(plain, "plain.") + build_weights(grouped, "grouped.")
    x = hashed_input("input.x", (2, 3, 32), 2.0)
    return {
        "tensors": specs,
        "input": {"name": "input.x", "shape": [2, 3, 32], "scale": 2.0},
        "plain": as_list(plain(x)),
        "grouped": as_list(grouped(x)),
    }


def gated_delta_case() -> dict:
    config = text_config()
    gdn = Qwen4ExpGatedDeltaNet(config)
    gdn.eval()
    specs = build_weights(gdn, "gdn.")
    x = hashed_input("input.gdn", (1, 7, config.hidden_size))
    cache = ArraysCache(size=2)
    first = gdn(x[:, :4], cache=cache)
    second = gdn(x[:, 4:], cache=cache)
    # Rows whose q/k sum of squares sits near the 1e-6 epsilon: this is where
    # adding it after the sum (Qwen4) and after the mean differ.
    small = hashed_input("input.gdn_small", (1, 3, config.hidden_size), 1e-5)
    small_out = gdn(small, cache=ArraysCache(size=2))
    # Qwen4 normalizes q/k op by op in the activation dtype; Qwen3.5 goes
    # through the fused RMSNorm kernel. In float32 the two agree to the last
    # ulp. In float16 the squares of small entries lose their low bits, so
    # the two disagree on almost every element. (bfloat16 also differs, by
    # one ulp, but MLX releases already disagree with each other there on how
    # a bfloat16 sum rounds, so it cannot pin the port.)
    # This case runs on the GPU: the CPU flushes float16 subnormals to zero
    # and turns these rows into infinities, and it has no matmul to lose
    # precision on.
    shape = (1, 6, config.linear_num_key_heads, config.linear_key_head_dim)
    with mx.stream(mx.gpu):
        q = hashed_input("input.norm_q", shape, 1e-4).astype(mx.float16)
        k = hashed_input("input.norm_k", shape, 1e-4).astype(mx.float16)
        nq, nk = gdn._normalize_qk(q, k)
        mx.eval(nq, nk)
    normalization = {"shape": list(shape), "scale": 1e-4, "q": as_list(nq), "k": as_list(nk)}
    return {
        "tensors": specs,
        "input": {"name": "input.gdn", "shape": [1, 7, config.hidden_size], "scale": 1.0},
        "small_input": {
            "name": "input.gdn_small",
            "shape": [1, 3, config.hidden_size],
            "scale": 1e-5,
        },
        "split": 4,
        "output": as_list(mx.concatenate([first, second], axis=1)),
        "small_output": as_list(small_out),
        "normalization": normalization,
    }


def hyper_case() -> dict:
    config = text_config()
    residual = Qwen4ExpGatedResidual(config)
    mixer = Qwen4ExpGatedResidual(config, use_combine=False)
    specs = build_weights(residual, "residual.") + build_weights(mixer, "mixer.")
    width = config.hc_count * config.hidden_size
    h = hashed_input("input.hyper", (1, 3, width))
    branch = hashed_input("input.branch", (1, 3, config.hidden_size))
    mixed, hyper_input, inject = residual(h)
    injection = branch[..., None, :] * inject[..., None]
    out = hyper_input + injection.reshape(*hyper_input.shape)
    return {
        "tensors": specs,
        "input": {"name": "input.hyper", "shape": [1, 3, width], "scale": 1.0},
        "branch": {"name": "input.branch", "shape": [1, 3, config.hidden_size], "scale": 1.0},
        "mixed": as_list(mixed),
        "inject": as_list(inject),
        "output": as_list(out),
        "mixer": as_list(mixer(h)),
    }


PLE_TOKENS = [5, 9, 9, 17, EOS, 3, 44, 3, 3, 60, EOS, EOS, 7, 21]


def ple_case() -> dict:
    config = text_config()
    layer = Qwen4ExpPLELayer(config, layer_idx=1, ple_layer_index=0)
    specs = build_weights(layer, "ple.", skip=("layer_multipliers", "ngram_heads"))
    tokens = mx.array([PLE_TOKENS], dtype=mx.int32)
    width = config.hc_count * config.hidden_size
    h = hashed_input("input.ple", (1, len(PLE_TOKENS), width))

    ids_embedding = layer.ple_embedding
    ids = _ngram_ids(ids_embedding, tokens)

    one_shot = layer(h, tokens, ArraysCache(size=4), None)
    cache = ArraysCache(size=4)
    split = 6
    first = layer(h[:, :split], tokens[:, :split], cache, None)
    second = layer(h[:, split:], tokens[:, split:], cache, None)
    chunked = mx.concatenate([first, second], axis=1)
    assert np.allclose(np.array(one_shot), np.array(chunked), atol=1e-5)

    released = TextConfig(
        **{
            **TINY_CONFIG,
            "rope_parameters": dict(TINY_CONFIG["rope_parameters"]),
            "vocab_size": RELEASED_NGRAM_CONFIG["vocab_size"],
            "ngram_size": RELEASED_NGRAM_CONFIG["ngram_size"],
            "heads_per_ngram": RELEASED_NGRAM_CONFIG["heads_per_ngram"],
            "ngram_vocab_size_base": RELEASED_NGRAM_CONFIG["ngram_vocab_size_base"],
            "make_ngram_vocab_size_divisible_by": RELEASED_NGRAM_CONFIG[
                "make_ngram_vocab_size_divisible_by"
            ],
            "seed": RELEASED_NGRAM_CONFIG["seed"],
            "eos_token_id": RELEASED_NGRAM_CONFIG["eos_token_id"],
            "ple_embed_dim": 16 * 8,
            "split_ngram_parts": 128,
        }
    )
    released_embedding = _ids_only_embedding(released)
    released_tokens = [151, 248044, 9707, 11, 1879, 0, 248320 - 1, 248044, 248044, 77, 13]
    released_ids = _ngram_ids(
        released_embedding, mx.array([released_tokens], dtype=mx.int32)
    )
    return {
        "tensors": specs,
        "tokens": PLE_TOKENS,
        "input": {"name": "input.ple", "shape": [1, len(PLE_TOKENS), width], "scale": 1.0},
        "split": split,
        "ngram_ids": np.array(ids).reshape(-1).tolist(),
        "layer_multipliers": np.array(ids_embedding.layer_multipliers).tolist(),
        "vocab_sizes": np.array(ids_embedding.ngram_heads_vocab_sizes).tolist(),
        "output": as_list(one_shot),
        "released": {
            "config": RELEASED_NGRAM_CONFIG,
            "tokens": released_tokens,
            "ngram_ids": np.array(released_ids).reshape(-1).tolist(),
            "layer_multipliers": np.array(released_embedding.layer_multipliers).tolist(),
            "vocab_sizes": np.array(released_embedding.ngram_heads_vocab_sizes).tolist(),
            "offsets": np.array(released_embedding.ngram_heads_offsets).tolist(),
        },
    }


def _ids_only_embedding(config: TextConfig) -> Qwen4ExpNGramEmbedding:
    """The released hash state. MLX arrays are lazy, so the 320M-row table
    is never allocated as long as nothing gathers from it."""
    return Qwen4ExpNGramEmbedding(config, 16, layer_idx=1, ple_layer_index=0)


def _ngram_ids(embedding: Qwen4ExpNGramEmbedding, tokens: mx.array) -> mx.array:
    """The ids `Qwen4ExpNGramEmbedding.__call__` gathers, without the gather."""
    captured = {}

    def capture(ids):
        captured["ids"] = ids
        return mx.zeros((*ids.shape, 1))

    # An instance attribute shadows the child module for this one call.
    embedding.__dict__["ngram_embedding"] = capture
    try:
        embedding(tokens, None)
    finally:
        del embedding.__dict__["ngram_embedding"]
    return captured["ids"]


QSA_CONFIG = {"indexer_budget": 2048, "indexer_compress_ratio": 4}


def qsa_case() -> dict:
    config = text_config(QSA_CONFIG)
    indexer = Qwen4ExpQSAIndexer(config, rotary_for(config))
    specs = build_weights(indexer, "indexer.")
    hidden = hashed_input("input.qsa", (1, 4100, config.hidden_size))

    def run(prefill: int, decode_steps: int):
        cache = QSAKVCache()
        prefill_selection = indexer.select(hidden[:, :prefill], cache, None)
        cache.offset = prefill
        decode = []
        for step in range(decode_steps):
            pos = prefill + step
            selection = indexer.select(hidden[:, pos : pos + 1], cache, None)
            cache.offset = pos + 1
            decode.append(selection)
        return prefill_selection, decode

    def block_set(selection, row) -> list[int]:
        blocks = np.array(selection.selected_blocks[0, row]).tolist()
        return sorted(b for b in blocks if b >= 0)

    sparse_prefill, sparse_decode = run(4095, 1)
    margin = _selection_margin(indexer, hidden, 4096)
    dense_prefill, dense_decode = run(2047, 5)
    return {
        "config_overrides": QSA_CONFIG,
        "tensors": specs,
        "input": {"name": "input.qsa", "shape": [1, 4100, config.hidden_size], "scale": 1.0},
        "sparse": {
            "prefill": 4095,
            "prefill_rows": {
                str(row): block_set(sparse_prefill, row) for row in (2051, 3000, 4094)
            },
            "decode_blocks": block_set(sparse_decode[0], 0),
            "cutoff_margin": margin,
        },
        "dense_fallback": {
            "prefill": 2047,
            # One entry per decode step: whether the indexer returned a
            # selection. The 512-block budget is first exceeded at 2052 keys.
            "selected": [selection is not None for selection in dense_decode],
            "last_blocks": block_set(dense_decode[-1], 0),
        },
    }


def _selection_margin(indexer, hidden, key_len: int) -> float:
    """Score gap between the last selected and first rejected block for the
    decode query at `key_len`, so an exact set comparison is not a near-tie."""
    qk = indexer.index_qk_proj(hidden[:, :key_len])
    batch, length, _ = qk.shape
    qk = qk.reshape(batch, length, indexer.n_heads + 1, indexer.head_dim)
    query = indexer.q_layernorm(qk[:, -1:, : indexer.n_heads]).transpose(0, 2, 1, 3)
    positions = mx.arange(key_len)[None]
    query = indexer._apply_rope(query, positions[:, -1:])
    raw = qk[:, :, indexer.n_heads :].squeeze(2)
    blocks = key_len // indexer.compress_ratio
    pooled = raw[:, : blocks * indexer.compress_ratio].reshape(
        batch, blocks, indexer.compress_ratio, indexer.head_dim
    )
    pooled = indexer.k_layernorm(mx.mean(pooled, axis=2))[:, None]
    pooled = indexer._apply_rope(pooled, positions[:, :: indexer.compress_ratio][:, :blocks])
    scores = query @ pooled.transpose(0, 1, 3, 2)
    scores = mx.sum(mx.maximum(scores, 0), axis=1) / math.sqrt(indexer.head_dim)
    ordered = np.sort(np.array(scores).reshape(-1))[::-1]
    return float(ordered[indexer.block_topk - 1] - ordered[indexer.block_topk])


MODEL_PROMPT = [3, 7, 11, 2, 40, 41, 9, EOS, 12, 13, 5, 5, 60, 22, 31, 8, 9, 10, 11, 4,
                19, 23, 29, 31, 37, 2, 3, 5, 7, 11, 13, EOS, 17, 19, 23, 6, 50, 51, 52, 53]
DECODE_STEPS = 8


def model_case() -> dict:
    config = text_config()
    vision = SimpleNamespace(spatial_merge_size=2)
    wrapper = SimpleNamespace(
        vision_config=vision,
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
    )
    model = LanguageModel(config, wrapper)
    model.eval()
    specs = build_weights(model, "", skip=("layer_multipliers", "ngram_heads"))

    cache = model.make_cache()
    prompt = mx.array([MODEL_PROMPT], dtype=mx.int32)
    out = model(prompt, cache=cache)
    prefill_logits = out.logits
    token = int(mx.argmax(prefill_logits[0, -1]).item())
    tokens, decode_logits = [], []
    for _ in range(DECODE_STEPS):
        tokens.append(token)
        out = model(mx.array([[token]], dtype=mx.int32), cache=cache)
        logits = out.logits[0, -1]
        decode_logits.append(as_list(logits))
        token = int(mx.argmax(logits).item())
    return {
        "config": {k: v for k, v in TINY_CONFIG.items()},
        "tensors": specs,
        "prompt": MODEL_PROMPT,
        "prefill_logits": as_list(prefill_logits[0]),
        "decode_tokens": tokens,
        "decode_logits": decode_logits,
    }


TINY_VISION_CONFIG = {
    "model_type": "qwen4_exp",
    "depth": 2,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_heads": 2,
    "out_hidden_size": TINY_CONFIG["hidden_size"],
    "patch_size": 4,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "in_channels": 3,
    "num_position_embeddings": 16,
    "hidden_act": "gelu_pytorch_tanh",
    "deepstack_visual_indexes": [],
}
IMAGE_TOKEN, VIDEO_TOKEN, VISION_START, VISION_END = 60, 61, 62, 63
IMAGE_GRID = (1, 4, 4)
# Four merged image tokens between the vision markers, mid-prompt, so the
# MRoPE positions after the image are shifted by the grid.
VLM_PROMPT = [3, 7, 11, VISION_START, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN,
              VISION_END, 12, 13, 5, 9, 21, 33, 40, 41, 2, 8, 19, 23, 29, 31, 37,
              2, 3, 5, 7]


def vlm_case() -> dict:
    from mlx_vlm.models.qwen4_exp import Model, ModelConfig

    config = ModelConfig.from_dict(
        {
            "model_type": "qwen4_exp",
            "text_config": dict(TINY_CONFIG, rope_parameters=dict(TINY_CONFIG["rope_parameters"])),
            "vision_config": dict(TINY_VISION_CONFIG),
            "image_token_id": IMAGE_TOKEN,
            "video_token_id": VIDEO_TOKEN,
            "vision_start_token_id": VISION_START,
            "vision_end_token_id": VISION_END,
            "vocab_size": TINY_CONFIG["vocab_size"],
        }
    )
    model = Model(config)
    model.eval()
    specs = build_weights(model, "", skip=("layer_multipliers", "ngram_heads"))
    t, h, w = IMAGE_GRID
    patch = TINY_VISION_CONFIG["patch_size"]
    pixel_shape = (t * h * w, 3 * TINY_VISION_CONFIG["temporal_patch_size"] * patch * patch)
    pixels = hashed_input("input.pixels", pixel_shape)

    cache = model.language_model.make_cache()
    prompt = mx.array([VLM_PROMPT], dtype=mx.int32)
    out = model(prompt, pixels, cache=cache, image_grid_thw=mx.array([IMAGE_GRID]))
    prefill_logits = out.logits[0]
    token = int(mx.argmax(out.logits[0, -1]).item())
    tokens, decode_logits = [], []
    for _ in range(4):
        tokens.append(token)
        out = model.language_model(mx.array([[token]], dtype=mx.int32), cache=cache)
        decode_logits.append(as_list(out.logits[0, -1]))
        token = int(mx.argmax(out.logits[0, -1]).item())
    return {
        "vision_config": TINY_VISION_CONFIG,
        "token_ids": {
            "image": IMAGE_TOKEN,
            "video": VIDEO_TOKEN,
            "vision_start": VISION_START,
        },
        "tensors": specs,
        "pixels": {"name": "input.pixels", "shape": list(pixel_shape), "scale": 1.0},
        "image_grid": list(IMAGE_GRID),
        "prompt": VLM_PROMPT,
        "prefill_logits": as_list(prefill_logits),
        "decode_tokens": tokens,
        "decode_logits": decode_logits,
    }


def provenance() -> dict:
    import mlx_vlm

    dist = importlib.metadata.distribution("mlx-vlm")
    source = Path(mlx_vlm.__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "device": "cpu",
        "mlx_vlm": dist.version,
        "mlx_vlm_commit": commit,
        "mlx": importlib.metadata.version("mlx"),
    }


def main() -> None:
    # Float32 matmuls on recent Apple GPUs are not float32-accurate (measured
    # 7e-4 against float64 on an M5 Pro, 1e-7 on the CPU), so the reference
    # runs on the CPU.
    mx.set_default_device(mx.cpu)
    header = provenance()
    cases = {
        "norms": norms_case,
        "gated_delta": gated_delta_case,
        "hyper_connection": hyper_case,
        "ple": ple_case,
        "qsa": qsa_case,
        "model": model_case,
        "vlm": vlm_case,
    }
    for name, build in cases.items():
        payload = {"provenance": header, "tiny_config": TINY_CONFIG, **build()}
        path = FIXTURE_DIR / f"{name}.json"
        path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
        print(f"wrote {path.name} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
