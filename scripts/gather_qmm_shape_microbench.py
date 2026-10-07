#!/usr/bin/env python3
"""
Sweep `mx.gather_qmm` across the MoE shape configurations of Gemma 4,
Qwen3, and Mixtral to test the hypothesis that a specific
`(hidden, intermediate)` ratio pushes MLX's quantized gather-matmul
into a slower kernel variant.

We benchmark the exact `gather_qmm` signature mlxcel / mlx-lm drive on
decode (rank-4 LHS, rank-3 RHS, `transpose=true`, affine mode, biases
present, `sorted_indices=false`) at Gemma 4 / Qwen3 / Mixtral shapes,
plus a synthetic sweep that varies intermediate size while holding
hidden fixed.

Usage:
  python scripts/gather_qmm_shape_microbench.py
"""
from __future__ import annotations
import time
import mlx.core as mx

GROUP_SIZE = 64
BITS = 4
WARMUP = 20
ITERS = 200


def _prep(hidden: int, intermediate: int, num_experts: int, top_k: int):
    """Build a (gate-shaped) quantized weight + scales + biases + lhs + indices.

    Weight shape for `gate_proj` / `up_proj`: [E, intermediate, hidden].
    We always transpose=true so rhs is expected in that orientation.

    LHS shape on decode is the [..., 1, 1, hidden] rank-4 produced by
    `x.expand_dims((-2, -3))` in the Python reference (or two sequential
    `expand_dims` in mlxcel's SwitchGeGLU::forward).

    Indices shape is [1, top_k] matching one decoded token.
    """
    # Random fp32 weight we'll quantize.
    raw = mx.random.normal(shape=(num_experts, intermediate, hidden))
    w_q, scales, biases = mx.quantize(raw, group_size=GROUP_SIZE, bits=BITS, mode="affine")

    x = mx.random.normal(shape=(1, 1, 1, hidden))
    indices = mx.arange(top_k, dtype=mx.uint32)[None, :]
    mx.eval(w_q, scales, biases, x, indices)
    mx.synchronize()
    return w_q, scales, biases, x, indices


def bench_one(label: str, hidden: int, intermediate: int, num_experts: int, top_k: int):
    w_q, scales, biases, x, indices = _prep(hidden, intermediate, num_experts, top_k)

    # Warmup
    for _ in range(WARMUP):
        y = mx.gather_qmm(
            x, w_q, scales, biases,
            rhs_indices=indices,
            transpose=True,
            group_size=GROUP_SIZE,
            bits=BITS,
            mode="affine",
            sorted_indices=False,
        )
        mx.eval(y)
    mx.synchronize()

    # Measured run
    start = time.perf_counter()
    for _ in range(ITERS):
        y = mx.gather_qmm(
            x, w_q, scales, biases,
            rhs_indices=indices,
            transpose=True,
            group_size=GROUP_SIZE,
            bits=BITS,
            mode="affine",
            sorted_indices=False,
        )
        mx.eval(y)
    mx.synchronize()
    elapsed = time.perf_counter() - start

    per_call_us = elapsed / ITERS * 1e6
    # Throughput in effective FLOPs (ignoring sparsity): top_k tokens
    # each do a (intermediate x hidden) matmul.
    flops_per_call = 2 * top_k * intermediate * hidden
    gflops = flops_per_call / (per_call_us * 1e-6) / 1e9

    ratio = intermediate / hidden
    print(
        f"{label:<40} "
        f"H={hidden:>5} I={intermediate:>5} "
        f"(I/H={ratio:>5.2f}) "
        f"E={num_experts:>3} k={top_k:>2}  "
        f"per_call={per_call_us:>8.2f}us  "
        f"eff_tflops={gflops/1000:>6.3f}"
    )


def main():
    print("=== gather_qmm shape sweep ===")
    print(f"WARMUP={WARMUP} ITERS={ITERS}  group_size={GROUP_SIZE} bits={BITS} mode=affine transpose=True sorted=False")
    print()

    print("## Real MoE configs (gate/up shape: [E, I, H])")
    # Gemma 4 26B-a4b (text_config): hidden=2816, moe_intermediate=704, E=128, top_k=8
    bench_one("Gemma 4 26B-a4b (gate/up)", 2816, 704, 128, 8)
    # Gemma 4 31B-it: text_config has no MoE; skip.
    # Qwen3 30B-a3b: hidden=2048, moe_intermediate_size ≈ 1408, E=128, top_k=8
    bench_one("Qwen3 30B-a3b (gate/up)", 2048, 1408, 128, 8)
    # Mixtral 8x7B: hidden=4096, intermediate=14336, E=8, top_k=2
    bench_one("Mixtral 8x7B (gate/up)", 4096, 14336, 8, 2)

    print()
    print("## down_proj (shape: [E, H, I])")
    bench_one("Gemma 4 26B-a4b (down)", 704, 2816, 128, 8)
    bench_one("Qwen3 30B-a3b (down)", 1408, 2048, 128, 8)
    bench_one("Mixtral 8x7B (down)", 14336, 4096, 8, 2)

    print()
    print("## Synthetic sweep (hidden=2816, E=128, top_k=8, vary intermediate)")
    for inter in (352, 704, 1056, 1408, 2112, 2816, 5632):
        bench_one(f"H=2816 I={inter}", 2816, inter, 128, 8)

    print()
    print("## Synthetic sweep (hidden=2048, E=128, top_k=8, vary intermediate)")
    for inter in (352, 704, 1056, 1408, 2048, 4096):
        bench_one(f"H=2048 I={inter}", 2048, inter, 128, 8)


if __name__ == "__main__":
    main()
