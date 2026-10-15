#!/usr/bin/env python3
"""Independent Gemma3 VLM embeddings/mask execution check for issue #869.

The production VLM path hands `prefill_embeddings.main` post-Gemma-scale text
rows, unscaled projected image rows, and one authoritative additive f32 mask.
This fixture compares that graph with an independent Hugging Face Gemma3 eager
model using the same tiny deterministic checkpoint.  The HF side receives the
corresponding pre-scale rows because Gemma3 applies `sqrt(hidden_size)` inside
its model forward.

The canonical case compares last-token logits, every layer's K/V cache, and the
greedy token.  Four negative fixtures must then diverge: accidental causal
masking, a multiplicative 0/1 mask, double Gemma scaling, and a missing image
pre-divide.  The tiny CPU graph keeps the check suitable for a short, bounded
local-task gate; pinned-checkpoint vision and server gates remain separate.

Run from the repository root with the shared OpenXLA spike environment:

    spike/openxla/.venv/bin/python spike/openxla/gemma3_vlm_mask_check.py

Exit 0 means the independent canonical oracle matched and every negative
fixture was detected.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from iree.compiler.tools import compile_file
from iree.runtime import load_vm_flatbuffer_file
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
CARGO = os.environ.get("CARGO", "cargo")
PREFILL_LP = 256
REAL_LEN = 12
HIDDEN = 16
INTERMEDIATE = 32
N_LAYERS = 4
N_Q = 4
N_KV = 2
HEAD_DIM = 4
VOCAB = 64
EPS = 1e-6
ROPE_THETA = 10_000.0
ROPE_LOCAL_BASE = 100.0
SLIDING_WINDOW = 4
SLIDING_PATTERN = 3
MASKED = np.float32(np.finfo(np.float32).min)
LOGIT_ATOL = 2e-2
KV_MAX_ABS = 2.5e-1
KV_RMS = 1e-1
KV_MIN_COSINE = 9.9e-1
NEGATIVE_MIN_DIFF = 1e-4
IMAGE_POSITIONS = (3, 4)


def dimensions() -> dict[str, object]:
    return {
        "hidden_size": HIDDEN,
        "num_attention_heads": N_Q,
        "num_key_value_heads": N_KV,
        "head_dim": HEAD_DIM,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": N_LAYERS,
        "vocab_size": VOCAB,
        "rms_norm_eps": EPS,
        "rope_theta": ROPE_THETA,
        "max_position_embeddings": 512,
        "attention_bias": False,
    }


def hf_config() -> Gemma3TextConfig:
    return Gemma3TextConfig(
        **dimensions(),
        tie_word_embeddings=True,
        query_pre_attn_scalar=HEAD_DIM,
        rope_local_base_freq=ROPE_LOCAL_BASE,
        sliding_window=SLIDING_WINDOW,
        sliding_window_pattern=SLIDING_PATTERN,
        attn_logit_softcapping=None,
        final_logit_softcapping=None,
        hidden_activation="gelu_pytorch_tanh",
    )


def emitter_config() -> dict[str, object]:
    config = dimensions()
    config.pop("max_position_embeddings")
    config.update(
        model_type="gemma3_text",
        tie_word_embeddings=True,
        query_pre_attn_scalar=HEAD_DIM,
        rope_local_base_freq=ROPE_LOCAL_BASE,
        sliding_window=SLIDING_WINDOW,
        sliding_window_pattern=SLIDING_PATTERN,
        attn_logit_softcapping=None,
        final_logit_softcapping=None,
        hidden_activation="gelu_pytorch_tanh",
    )
    return config


def argument_names() -> list[str]:
    names = ["model.embed_tokens.weight", "model.norm.weight"]
    for index in range(N_LAYERS):
        prefix = f"model.layers.{index}."
        names.extend(
            [
                prefix + "mlp.down_proj.weight",
                prefix + "mlp.gate_proj.weight",
                prefix + "input_layernorm.weight",
                prefix + "post_attention_layernorm.weight",
                prefix + "mlp.up_proj.weight",
                prefix + "self_attn.k_proj.weight",
                prefix + "self_attn.o_proj.weight",
                prefix + "self_attn.q_proj.weight",
                prefix + "self_attn.v_proj.weight",
                prefix + "self_attn.q_norm.weight",
                prefix + "self_attn.k_norm.weight",
                prefix + "pre_feedforward_layernorm.weight",
                prefix + "post_feedforward_layernorm.weight",
            ]
        )
    return names


def build_checkpoint() -> tuple[Gemma3ForCausalLM, list[np.ndarray]]:
    torch.manual_seed(869)
    model = Gemma3ForCausalLM(hf_config()).eval().float()
    model.config._attn_implementation = "eager"
    with torch.no_grad():
        for _, parameter in model.named_parameters():
            if parameter.dim() == 1:
                parameter.copy_(torch.randn_like(parameter) * 0.1)
    state = model.state_dict()
    names = argument_names()
    missing = [name for name in names if name not in state]
    if missing:
        raise RuntimeError(f"HF checkpoint is missing emitter weights: {missing[:4]}")
    weights = [
        np.ascontiguousarray(state[name].detach().numpy(), dtype=np.float32)
        for name in names
    ]
    return model, weights


def emit_and_compile() -> object:
    work = Path(tempfile.mkdtemp(prefix="gemma3_vlm_mask_"))
    config_path = work / "config.json"
    config_path.write_text(json.dumps(emitter_config()), encoding="utf-8")

    print("[emit] Gemma3 embeddings prefill StableHLO", flush=True)
    subprocess.run(
        [
            CARGO,
            "test",
            "-p",
            "mlxcel-xla",
            "--lib",
            "emitter::tests::dump_prefill_embeddings_parity_graphs",
            "--",
            "--ignored",
            "--nocapture",
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "MLXCEL_DUMP_CONFIG": str(config_path),
            "MLXCEL_DUMP_DIR": str(work),
        },
        check=True,
    )

    source = work / "prefill_embeddings_logits.mlir"
    output = work / "prefill_embeddings_logits.vmfb"
    print("[compile] Gemma3 embeddings prefill (llvm-cpu)", flush=True)
    compile_file(
        str(source),
        output_file=str(output),
        input_type="stablehlo",
        target_backends=["llvm-cpu"],
    )
    return load_vm_flatbuffer_file(str(output), driver="local-task")


def canonical_inputs(
    embedding_table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(869)
    tokens = rng.integers(1, VOCAB, size=REAL_LEN, dtype=np.int32)
    pre_scale = np.ascontiguousarray(embedding_table[tokens], dtype=np.float32)
    projected = np.ascontiguousarray(
        rng.normal(0.0, 0.2, (len(IMAGE_POSITIONS), HIDDEN)),
        dtype=np.float32,
    )
    normalizer = np.float32(np.sqrt(HIDDEN))
    for row, position in enumerate(IMAGE_POSITIONS):
        pre_scale[position] = projected[row] / normalizer

    post_scale = np.zeros((PREFILL_LP, HIDDEN), dtype=np.float32)
    post_scale[:REAL_LEN] = pre_scale * normalizer

    mask = np.full((PREFILL_LP, PREFILL_LP), MASKED, dtype=np.float32)
    mask[:REAL_LEN, :REAL_LEN] = 0.0
    return pre_scale, np.ascontiguousarray(post_scale), np.ascontiguousarray(mask)


def to_host(value: object) -> np.ndarray:
    host = value.to_host() if hasattr(value, "to_host") else value
    return np.asarray(host, dtype=np.float32)


def run_iree(
    module: object,
    weights: list[np.ndarray],
    embeddings: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.arange(PREFILL_LP, dtype=np.int32)
    real_len = np.asarray(REAL_LEN, dtype=np.int32)
    outputs = module.main(*weights, embeddings, positions, real_len, mask)
    return tuple(to_host(value) for value in outputs)


def legacy_cache(cache: object) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    if isinstance(cache, (tuple, list)):
        return tuple(cache)
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise RuntimeError(f"unsupported HF cache type: {type(cache).__name__}")
    return tuple((layer.keys, layer.values) for layer in layers)


def run_hf(
    model: Gemma3ForCausalLM,
    pre_scale: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    real_mask = np.ascontiguousarray(mask[:REAL_LEN, :REAL_LEN])
    with torch.no_grad():
        outputs = model(
            inputs_embeds=torch.from_numpy(pre_scale[None, ...]),
            attention_mask=torch.from_numpy(real_mask[None, None, ...]),
            position_ids=torch.arange(REAL_LEN, dtype=torch.long)[None, :],
            use_cache=True,
            return_dict=True,
        )
    logits = outputs.logits[0, REAL_LEN - 1].detach().numpy().astype(np.float32)
    cache = legacy_cache(outputs.past_key_values)
    keys = [
        layer[0][0].detach().numpy().transpose(1, 0, 2).astype(np.float32)
        for layer in cache
    ]
    values = [
        layer[1][0].detach().numpy().transpose(1, 0, 2).astype(np.float32)
        for layer in cache
    ]
    return logits, keys, values


def compare_canonical(
    iree: tuple[np.ndarray, np.ndarray, np.ndarray],
    hf: tuple[np.ndarray, list[np.ndarray], list[np.ndarray]],
) -> bool:
    logits_close = np.allclose(iree[0], hf[0], rtol=0.0, atol=LOGIT_ATOL)
    logits_diff = float(np.max(np.abs(iree[0] - hf[0])))
    print(
        f"[canonical/logits] shape={iree[0].shape}/{hf[0].shape} "
        f"max|diff|={logits_diff:.3e} -> {'PASS' if logits_close else 'FAIL'}",
        flush=True,
    )
    ok = bool(logits_close)
    for name, actual, expected_layers in (
        ("kcache", iree[1], hf[1]),
        ("vcache", iree[2], hf[2]),
    ):
        for layer, expected in enumerate(expected_layers):
            cache_len = expected.shape[0]
            actual_slice = actual[layer, REAL_LEN - cache_len : REAL_LEN]
            same_shape = actual_slice.shape == expected.shape
            max_diff = (
                float(np.max(np.abs(actual_slice - expected)))
                if same_shape
                else float("inf")
            )
            rms_diff = (
                float(np.sqrt(np.mean(np.square(actual_slice - expected))))
                if same_shape
                else float("inf")
            )
            denominator = (
                float(np.linalg.norm(actual_slice) * np.linalg.norm(expected))
                if same_shape
                else 0.0
            )
            cosine = (
                float(np.vdot(actual_slice, expected) / denominator)
                if denominator > 0.0
                else 1.0
            )
            close = (
                same_shape
                and max_diff <= KV_MAX_ABS
                and rms_diff <= KV_RMS
                and cosine >= KV_MIN_COSINE
            )
            ok = ok and close
            best = ""
            if not close and cache_len < REAL_LEN:
                candidates = [
                    (
                        start,
                        float(
                            np.max(
                                np.abs(
                                    actual[layer, start : start + cache_len] - expected
                                )
                            )
                        ),
                    )
                    for start in range(REAL_LEN - cache_len + 1)
                ]
                best_start, best_diff = min(candidates, key=lambda item: item[1])
                best = f" best_slice={best_start}:{best_start + cache_len}/{best_diff:.3e}"
            print(
                f"[canonical/{name}/layer{layer}] "
                f"shape={actual_slice.shape}/{expected.shape} "
                f"max|diff|={max_diff:.3e} rms={rms_diff:.3e} "
                f"cos={cosine:.6f}{best} "
                f"-> {'PASS' if close else 'FAIL'}",
                flush=True,
            )
    iree_token = int(np.argmax(iree[0]))
    hf_token = int(np.argmax(hf[0]))
    token_ok = iree_token == hf_token
    print(
        f"[canonical/token] iree={iree_token} hf={hf_token} "
        f"-> {'PASS' if token_ok else 'FAIL'}",
        flush=True,
    )
    return ok and token_ok


def negative_detected(
    label: str,
    canonical_logits: np.ndarray,
    negative_logits: np.ndarray,
) -> bool:
    difference = float(np.max(np.abs(canonical_logits - negative_logits)))
    detected = difference > NEGATIVE_MIN_DIFF
    print(
        f"[negative/{label}] max|logit diff|={difference:.3e} "
        f"-> {'DETECTED' if detected else 'MISSED'}",
        flush=True,
    )
    return detected


def main() -> int:
    model, weights = build_checkpoint()
    module = emit_and_compile()
    pre_scale, embeddings, mask = canonical_inputs(weights[0])

    print("[run] canonical independent HF and IREE paths", flush=True)
    canonical = run_iree(module, weights, embeddings, mask)
    reference = run_hf(model, pre_scale, mask)
    checks = [compare_canonical(canonical, reference)]

    causal = np.full_like(mask, MASKED)
    valid = np.arange(REAL_LEN)
    causal[valid[:, None], valid[None, :]] = np.where(
        valid[None, :] <= valid[:, None], 0.0, MASKED
    )
    checks.append(
        negative_detected(
            "causal-mask",
            canonical[0],
            run_iree(module, weights, embeddings, causal)[0],
        )
    )

    multiplicative = np.zeros_like(mask)
    multiplicative[:REAL_LEN, :REAL_LEN] = 1.0
    checks.append(
        negative_detected(
            "multiplicative-mask",
            canonical[0],
            run_iree(module, weights, embeddings, multiplicative)[0],
        )
    )

    checks.append(
        negative_detected(
            "double-scale",
            canonical[0],
            run_iree(module, weights, embeddings * np.float32(np.sqrt(HIDDEN)), mask)[0],
        )
    )

    missing_image_prescale = embeddings.copy()
    missing_image_prescale[list(IMAGE_POSITIONS)] *= np.float32(np.sqrt(HIDDEN))
    checks.append(
        negative_detected(
            "missing-image-prescale",
            canonical[0],
            run_iree(module, weights, missing_image_prescale, mask)[0],
        )
    )

    ok = all(checks)
    print(
        f"RESULT: {'PASS' if ok else 'FAIL'} "
        f"(logit_atol={LOGIT_ATOL:g}, kv_max={KV_MAX_ABS:g}, "
        f"kv_rms={KV_RMS:g}, kv_cos={KV_MIN_COSINE:g}, local-task)",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
