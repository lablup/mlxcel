#!/usr/bin/env python3
"""Compile and execute the real sparse DeepStack prefill entry with IREE.

The Qwen3 fixture makes every transformer projection zero, keeps all RMSNorm
weights at one, and uses an identity untied LM head. Sparse features injected
after the first, middle, and last language layers therefore accumulate at the
final visual row to [1, 1, 1, 0]. The expected logits are its exact
RMS-normalized value. This exercises the production StableHLO entry and the
same post-language-layer injection point as MLX Qwen3-VL, not a test-only probe.
"""

from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
IREE_HOME = Path(
    os.environ.get(
        "IREE_CUDA_HOME",
        "/home/inureyes/.cache/mlxcel/iree-cuda-3.12.0rc20260721",
    )
)
IREE_COMPILE = Path(
    os.environ.get("IREE_COMPILE", IREE_HOME / "venv/bin/iree-compile")
)
IREE_RUN = Path(
    os.environ.get("IREE_RUN_MODULE", IREE_HOME / "build/tools/iree-run-module")
)

LP = 256
HIDDEN = 4
LAYERS = 3
VISUAL_MAX = 2
EPS = 1e-6


def config() -> dict[str, object]:
    return {
        "model_type": "qwen3",
        "hidden_size": HIDDEN,
        "intermediate_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": HIDDEN,
        "vocab_size": HIDDEN,
        "rms_norm_eps": EPS,
        "rope_theta": 10_000.0,
        "tie_word_embeddings": False,
        "deepstack_language_layer_indices": [0, 1, 2],
        "deepstack_max_visual_positions": VISUAL_MAX,
    }


def write_values(path: Path, code: str, values: list[float | int]) -> None:
    path.write_bytes(struct.pack(f"<{len(values)}{code}", *values))


def input_file(
    work: Path,
    name: str,
    shape: tuple[int, ...],
    dtype: str,
    values: list[float | int],
) -> str:
    path = work / f"{name}.bin"
    write_values(path, "f" if dtype == "f32" else "i", values)
    dimensions = "x".join(str(dim) for dim in shape)
    return f"--input={dimensions}x{dtype}=@{path}"


def weight_inputs(work: Path) -> list[str]:
    zero_matrix = [0.0] * (HIDDEN * HIDDEN)
    one_norm = [1.0] * HIDDEN
    identity = [
        float(row == column)
        for row in range(HIDDEN)
        for column in range(HIDDEN)
    ]
    specs: list[tuple[str, tuple[int, ...], list[float]]] = [
        ("embed", (HIDDEN, HIDDEN), zero_matrix),
        ("final_norm", (HIDDEN,), one_norm),
        ("lm_head", (HIDDEN, HIDDEN), identity),
    ]
    for layer in range(LAYERS):
        specs.extend(
            [
                (f"l{layer}_down", (HIDDEN, HIDDEN), zero_matrix),
                (f"l{layer}_gate", (HIDDEN, HIDDEN), zero_matrix),
                (f"l{layer}_input_norm", (HIDDEN,), one_norm),
                (f"l{layer}_post_norm", (HIDDEN,), one_norm),
                (f"l{layer}_up", (HIDDEN, HIDDEN), zero_matrix),
                (f"l{layer}_wk", (HIDDEN, HIDDEN), zero_matrix),
                (f"l{layer}_wo", (HIDDEN, HIDDEN), zero_matrix),
                (f"l{layer}_wq", (HIDDEN, HIDDEN), zero_matrix),
                (f"l{layer}_wv", (HIDDEN, HIDDEN), zero_matrix),
                (f"l{layer}_q_norm", (HIDDEN,), one_norm),
                (f"l{layer}_k_norm", (HIDDEN,), one_norm),
            ]
        )
    return [
        input_file(work, name, shape, "f32", values)
        for name, shape, values in specs
    ]


def runtime_inputs(work: Path) -> list[str]:
    features = [0.0] * (LAYERS * VISUAL_MAX * HIDDEN)
    for layer in range(LAYERS):
        features[(layer * VISUAL_MAX * HIDDEN) + layer] = 1.0
    return [
        input_file(work, "embeddings", (LP, HIDDEN), "f32", [0.0] * (LP * HIDDEN)),
        input_file(work, "positions", (LP,), "i32", list(range(LP))),
        "--input=i32=2",
        input_file(work, "bias", (LP, LP), "f32", [0.0] * (LP * LP)),
        input_file(work, "visual_positions", (VISUAL_MAX,), "i32", [1, -1]),
        input_file(
            work,
            "layer_features",
            (LAYERS, VISUAL_MAX, HIDDEN),
            "f32",
            features,
        ),
        input_file(work, "layer_indices", (LAYERS,), "i32", [0, 1, 2]),
        f"--input=i32={LAYERS}",
        "--input=i32=1",
    ]


def main() -> int:
    if not IREE_COMPILE.is_file() or not IREE_RUN.is_file():
        raise SystemExit("set IREE_COMPILE and IREE_RUN_MODULE to pinned IREE tools")
    with tempfile.TemporaryDirectory(prefix="mlxcel_deepstack_") as temp:
        work = Path(temp)
        config_path = work / "config.json"
        config_path.write_text(json.dumps(config()), encoding="utf-8")
        subprocess.run(
            [
                os.environ.get("CARGO", "cargo"),
                "test",
                "-p",
                "mlxcel-xla",
                "--lib",
                "emitter::tests::dump_prefill_embeddings_parity_graphs",
                "--",
                "--ignored",
            ],
            cwd=REPO_ROOT,
            env={
                **os.environ,
                "MLXCEL_DUMP_CONFIG": str(config_path),
                "MLXCEL_DUMP_DIR": str(work),
            },
            check=True,
        )
        mlir = work / "prefill_embeddings_deepstack_logits.mlir"
        vmfb = work / "prefill_embeddings_deepstack_logits.vmfb"
        subprocess.run(
            [
                str(IREE_COMPILE),
                str(mlir),
                "--iree-input-type=stablehlo",
                "--iree-hal-target-device=local",
                "--iree-hal-local-target-device-backends=llvm-cpu",
                "--iree-llvmcpu-target-cpu=host",
                "-o",
                str(vmfb),
            ],
            check=True,
        )
        normalized = 1.0 / math.sqrt(0.75 + EPS)
        command = [
            str(IREE_RUN),
            "--device=local-task",
            f"--module={vmfb}",
            "--function=main",
            *weight_inputs(work),
            *runtime_inputs(work),
            f"--expected_output=4xf32={normalized} {normalized} {normalized} 0",
            "--expected_output=(ignored)",
            "--expected_output=(ignored)",
        ]
        subprocess.run(command, check=True)
    print("RESULT: PASS (sparse DeepStack prefill, IREE local-task)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
