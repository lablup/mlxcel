#!/usr/bin/env python3
"""Real-IREE parity check for token and embeddings StableHLO prefill entries.

The fixture builds deterministic tiny Llama-shaped tied and untied checkpoints,
emits both prefill modules from each exact config, compiles them for IREE's
llvm-cpu/local-task target, and invokes them with one shared weight set. The
embeddings input is gathered from the checkpoint's token embedding table; the
explicit attention bias uses the emitter's finite 0/-1e30 causal convention.

Every output is compared, not only the selected logits: both K and V tensors for
every layer and every padded bucket row must agree within ATOL. Scenarios cover a
one-token prompt, nonzero padding tokens, an untied LM head, and a prompt one token
below the 256-token bucket capacity.

Run from the repository root with the OpenXLA spike environment:

    spike/openxla/.venv/bin/python spike/openxla/prefill_embeddings_check.py

Exit 0 means both modules compiled and every parity scenario passed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from iree.compiler.tools import compile_file
from iree.runtime import load_vm_flatbuffer_file

REPO_ROOT = Path(__file__).resolve().parents[2]
CARGO = os.environ.get("CARGO", "cargo")
PREFILL_LP = 256
HIDDEN = 8
INTERMEDIATE = 16
N_LAYERS = 2
N_Q = 2
N_KV = 1
HEAD_DIM = 4
VOCAB = 32
MASKED = np.float32(-1e30)
ATOL = 1e-5


def emitter_config(*, tied: bool) -> dict[str, object]:
    return {
        "model_type": "llama",
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": N_LAYERS,
        "num_attention_heads": N_Q,
        "num_key_value_heads": N_KV,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "tie_word_embeddings": tied,
    }


def random_weight(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return np.ascontiguousarray(rng.normal(0.0, 0.08, shape), dtype=np.float32)


def checkpoint_weights(*, tied: bool) -> list[np.ndarray]:
    """Return one synthetic checkpoint in the emitter's deterministic arg order."""
    rng = np.random.default_rng(858 if tied else 859)
    weights = [
        random_weight(rng, (VOCAB, HIDDEN)),
        np.ascontiguousarray(rng.uniform(0.8, 1.2, HIDDEN), dtype=np.float32),
    ]
    if not tied:
        weights.append(random_weight(rng, (VOCAB, HIDDEN)))
    for _ in range(N_LAYERS):
        weights.extend(
            [
                random_weight(rng, (HIDDEN, INTERMEDIATE)),  # down
                random_weight(rng, (INTERMEDIATE, HIDDEN)),  # gate
                np.ascontiguousarray(rng.uniform(0.8, 1.2, HIDDEN), dtype=np.float32),
                np.ascontiguousarray(rng.uniform(0.8, 1.2, HIDDEN), dtype=np.float32),
                random_weight(rng, (INTERMEDIATE, HIDDEN)),  # up
                random_weight(rng, (N_KV * HEAD_DIM, HIDDEN)),  # wk
                random_weight(rng, (HIDDEN, N_Q * HEAD_DIM)),  # wo
                random_weight(rng, (N_Q * HEAD_DIM, HIDDEN)),  # wq
                random_weight(rng, (N_KV * HEAD_DIM, HIDDEN)),  # wv
            ]
        )
    return weights


def emit_and_compile(*, tied: bool) -> tuple[object, object]:
    tag = "tied" if tied else "untied"
    work = Path(tempfile.mkdtemp(prefix=f"prefill_embeddings_{tag}_"))
    config_path = work / "config.json"
    config_path.write_text(json.dumps(emitter_config(tied=tied)), encoding="utf-8")

    print(f"[emit] {tag}: token + embeddings prefill StableHLO", flush=True)
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

    token_mlir = work / "prefill_logits.mlir"
    embeddings_mlir = work / "prefill_embeddings_logits.mlir"
    token_vmfb = work / "prefill_logits.vmfb"
    embeddings_vmfb = work / "prefill_embeddings_logits.vmfb"
    for label, source, output in [
        ("token", token_mlir, token_vmfb),
        ("embeddings", embeddings_mlir, embeddings_vmfb),
    ]:
        print(f"[compile] {tag}/{label}: llvm-cpu", flush=True)
        compile_file(
            str(source),
            output_file=str(output),
            input_type="stablehlo",
            target_backends=["llvm-cpu"],
        )

    return (
        load_vm_flatbuffer_file(str(token_vmfb), driver="local-task"),
        load_vm_flatbuffer_file(str(embeddings_vmfb), driver="local-task"),
    )


def causal_attention_bias() -> np.ndarray:
    query = np.arange(PREFILL_LP)[:, None]
    key = np.arange(PREFILL_LP)[None, :]
    return np.ascontiguousarray(np.where(key <= query, 0.0, MASKED), dtype=np.float32)


def to_host(value: object) -> np.ndarray:
    host = value.to_host() if hasattr(value, "to_host") else value
    return np.asarray(host, dtype=np.float32)


def run_case(
    *,
    label: str,
    real_len: int,
    padding_token: int,
    weights: list[np.ndarray],
    token_module: object,
    embeddings_module: object,
) -> bool:
    rng = np.random.default_rng(real_len + padding_token)
    tokens = np.full(PREFILL_LP, padding_token, dtype=np.int32)
    tokens[:real_len] = rng.integers(0, VOCAB, real_len, dtype=np.int32)
    embeddings = np.ascontiguousarray(weights[0][tokens], dtype=np.float32)
    positions = np.arange(PREFILL_LP, dtype=np.int32)
    scalar_len = np.asarray(real_len, dtype=np.int32)
    bias = causal_attention_bias()

    print(
        f"[run] {label}: real_len={real_len} padding_token={padding_token}",
        flush=True,
    )
    token_out = token_module.main(*weights, tokens, positions, scalar_len)
    embeddings_out = embeddings_module.main(
        *weights, embeddings, positions, scalar_len, bias
    )

    names = ["logits", "kcache", "vcache"]
    ok = True
    for name, token_value, embeddings_value in zip(
        names, token_out, embeddings_out, strict=True
    ):
        lhs = to_host(token_value)
        rhs = to_host(embeddings_value)
        max_diff = float(np.max(np.abs(lhs - rhs)))
        equal = np.allclose(lhs, rhs, rtol=0.0, atol=ATOL)
        ok = ok and equal
        print(
            f"[compare] {label}/{name}: shape={lhs.shape} "
            f"max|diff|={max_diff:.3e} -> {'PASS' if equal else 'FAIL'}",
            flush=True,
        )
    return ok


def run_checkpoint(*, tied: bool, scenarios: list[tuple[str, int, int]]) -> bool:
    token_module, embeddings_module = emit_and_compile(tied=tied)
    weights = checkpoint_weights(tied=tied)
    return all(
        run_case(
            label=label,
            real_len=real_len,
            padding_token=padding_token,
            weights=weights,
            token_module=token_module,
            embeddings_module=embeddings_module,
        )
        for label, real_len, padding_token in scenarios
    )


def main() -> int:
    tied_ok = run_checkpoint(
        tied=True,
        scenarios=[
            ("tied/real_len_1", 1, 7),
            ("tied/near_capacity", PREFILL_LP - 1, 9),
        ],
    )
    untied_ok = run_checkpoint(
        tied=False,
        scenarios=[("untied/nonzero_padding", 23, 5)],
    )
    ok = tied_ok and untied_ok
    print(f"RESULT: {'PASS' if ok else 'FAIL'} (ATOL={ATOL:g}, local-task)", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
