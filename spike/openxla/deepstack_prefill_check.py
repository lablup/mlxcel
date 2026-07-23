#!/usr/bin/env python3
"""Compile and execute the real sparse DeepStack prefill entry with IREE.

This ports the fixed `Qwen3VLModel::deepstack_process` regression fixture:
hidden states start at one, visual positions are 1..=3, and each visual feature
component is ten. With zero transformer projections and unit RMSNorm weights,
the first/middle/last post-hook residuals are exactly 11/21/31 at visual rows
and one elsewhere. The production entry's final logits and every K/V element
are also checked. The diagnostic entry reuses the production layer loop and
only exposes those post-hook states for this oracle. Both ordinary 1D and
explicit Qwen M-RoPE `[3, S]` positions execute the same fixed oracle; the root
`xla_prepared_prefill` integration test covers the stateful first decode step.
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
HIDDEN = 6
LAYERS = 3
VISUAL_MAX = 3
EPS = 1e-6


def config(mrope: bool) -> dict[str, object]:
    value: dict[str, object] = {
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
    if mrope:
        value["rope_scaling"] = {
            "rope_type": "mrope",
            "mrope_section": [1, 1, 1],
        }
    return value


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


def runtime_inputs(work: Path, mrope: bool) -> list[str]:
    features = [10.0] * (LAYERS * VISUAL_MAX * HIDDEN)
    if mrope:
        position_shape = (3, LP)
        positions = [
            coordinate
            for axis in range(3)
            for coordinate in (
                list(range(LP))
                if axis == 0
                else [max(0, value - axis) for value in range(LP)]
            )
        ]
    else:
        position_shape = (LP,)
        positions = list(range(LP))
    return [
        input_file(work, "embeddings", (LP, HIDDEN), "f32", [1.0] * (LP * HIDDEN)),
        input_file(work, "positions", position_shape, "i32", positions),
        "--input=i32=4",
        input_file(work, "bias", (LP, LP), "f32", [0.0] * (LP * LP)),
        input_file(work, "visual_positions", (VISUAL_MAX,), "i32", [1, 2, 3]),
        input_file(
            work,
            "layer_features",
            (LAYERS, VISUAL_MAX, HIDDEN),
            "f32",
            features,
        ),
        input_file(work, "layer_indices", (LAYERS,), "i32", [0, 1, 2]),
        f"--input=i32={LAYERS}",
        f"--input=i32={VISUAL_MAX}",
    ]


def compile_module(source: Path, output: Path) -> None:
    subprocess.run(
        [
            str(IREE_COMPILE),
            str(source),
            "--iree-input-type=stablehlo",
            "--iree-hal-target-device=local",
            "--iree-hal-local-target-device-backends=llvm-cpu",
            "--iree-llvmcpu-target-cpu=host",
            "-o",
            str(output),
        ],
        check=True,
    )


def expected_post_hook_states(work: Path) -> str:
    # Exact MLX-derived table, extended with unchanged rows to the static bucket.
    visual_values = [11.0, 21.0, 31.0]
    states: list[float] = []
    for value in visual_values:
        for position in range(LP):
            row_value = value if position in (1, 2, 3) else 1.0
            states.extend([row_value] * HIDDEN)
    path = work / "expected_post_hook_states.bin"
    write_values(path, "f", states)
    return f"--expected_output={LAYERS}x{LP}x{HIDDEN}xf32=@{path}"


def run_module(work: Path, vmfb: Path, include_states: bool, mrope: bool) -> None:
    normalized = 31.0 / math.sqrt(31.0 * 31.0 + EPS)
    expected = [
        f"--expected_output={HIDDEN}xf32="
        + " ".join([str(normalized)] * HIDDEN),
        f"--expected_output={LAYERS}x{LP}x1x{HIDDEN}xf32=0",
        f"--expected_output={LAYERS}x{LP}x1x{HIDDEN}xf32=0",
    ]
    if include_states:
        expected.append(expected_post_hook_states(work))
    subprocess.run(
        [
            str(IREE_RUN),
            "--device=local-task",
            f"--module={vmfb}",
            "--function=main",
            *weight_inputs(work),
            *runtime_inputs(work, mrope),
            *expected,
        ],
        check=True,
    )


def main() -> int:
    if not IREE_COMPILE.is_file() or not IREE_RUN.is_file():
        raise SystemExit("set IREE_COMPILE and IREE_RUN_MODULE to pinned IREE tools")
    with tempfile.TemporaryDirectory(prefix="mlxcel_deepstack_") as temp:
        root = Path(temp)
        for mode_name, mrope in [("one_d", False), ("mrope", True)]:
            work = root / mode_name
            work.mkdir()
            config_path = work / "config.json"
            config_path.write_text(json.dumps(config(mrope)), encoding="utf-8")
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
            production_mlir = work / "prefill_embeddings_deepstack_logits.mlir"
            production_vmfb = work / "prefill_embeddings_deepstack_logits.vmfb"
            diagnostics_mlir = work / "prefill_embeddings_deepstack_diagnostics.mlir"
            diagnostics_vmfb = work / "prefill_embeddings_deepstack_diagnostics.vmfb"
            compile_module(production_mlir, production_vmfb)
            compile_module(diagnostics_mlir, diagnostics_vmfb)
            run_module(work, production_vmfb, include_states=False, mrope=mrope)
            run_module(work, diagnostics_vmfb, include_states=True, mrope=mrope)
            print(f"MODE {mode_name}: PASS")
    print(
        "RESULT: PASS (1D + M-RoPE Qwen3-VL fixed post-hook states, logits, "
        "and every K/V; IREE local-task)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
