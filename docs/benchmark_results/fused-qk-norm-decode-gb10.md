# Fused QKV+QK-norm+RoPE decode path on GB10 (CUDA)

Validation of the generalized fused decode primitive (#326, merged in #341) on
**NVIDIA GB10 / CUDA**. The primitive `FusedQKVLinear::forward_split_norm_rope_quantized`
collapses the Qwen3 / Qwen3-MoE decode QKV projection, split, Q/K `RMSNorm`, and
RoPE into one C++ call. It is **default-OFF** (opt-in via `MLXCEL_FUSED_QK_NORM=1`)
because it cuts Rust<->C++ FFI crossings rather than MLX op count, and on M1 Ultra
it already measured slower than the graph path. The merge left one open question:
whether CUDA, with different op-dispatch cost, would be the per-backend win that
justifies enabling it. This benchmark answers that question. It does not.

## Environment

| Item | Value |
|------|-------|
| **Hardware** | NVIDIA GB10 (DGX Spark), 122 GB unified LPDDR5x |
| **OS** | Linux aarch64, kernel 6.17 |
| **Backend** | CUDA 13.0, SM 12.1 |
| **Build** | `MLX_CUDA_ARCHITECTURES=121 cargo build --release --features cuda` |
| **mlxcel version** | 0.3.1 (HEAD includes #341) |
| **MLX pin** | `a6ec7123` |
| **Harness** | same-process `mlxcel-bench-decode`, warm prefill, `--no-chat-template` |
| **Prompt** | narrative continuation: `"Once upon a time, in a small village nestled between two great mountains, there lived an old clockmaker who"` |
| **Tokens** | 256 measured, 32 warmup |
| **Method** | best-of-3, `MLXCEL_FUSED_QK_NORM=1` (fused) vs `=0` (graph / split) on the same binary |

The continuation prompt with 256 tokens avoids the early-EOS short generations
(9-34 tokens) that the chat-template prompt produces, so the decode rate is
measured over enough steps to resolve the small per-op effect. The fused kernel
JIT-compiles on first use: the first fused run of qwen3-0.6b read 248 tok/s cold,
then 361-373 warm; best-of-3 reports the warm figure.

## Decode throughput (tok/s, best-of-3)

| Model | Role | fused ON | graph OFF | fused / graph |
|-------|------|---------:|----------:|--------------:|
| qwen3-0.6b-4bit | wired dense | 372.8 | 387.2 | 0.96x (-3.7%) |
| qwen3-8b-4bit | wired dense | 52.6 | 52.4 | 1.00x (+0.4%) |
| qwen3-30b-a3b-4bit | wired MoE | 83.8 | 90.7 | 0.92x (-7.6%) |
| apertus-8b-2509-4bit | not wired (QK-norm) | 43.0 | 43.2 | 1.00x (flag inert) |
| llama-3.1-8b-4bit | regression guard (no QK-norm) | 48.0 | 48.2 | 1.00x (flag inert) |

The fused path is slower on every wired model, worst on the 48-layer MoE where
the per-layer reorder accumulates. The two control models confirm the harness:
apertus has Q/K norms but is not wired to the primitive, and llama has no QK-norm
at all, so on both the flag is inert and the on/off delta is within run noise.
That is the "regression guard shows no change" check: the change does not touch
the common attention path.

## Why fusion does not help here

`forward_split_norm_rope_quantized` is not a single custom kernel. It assembles
the same MLX ops as the graph path
(`quantized_matmul -> slice -> reshape -> transpose -> fast::rms_norm -> fast::rope`)
inside one C++ call. The GPU work is identical; only two things change: the
Rust<->C++ FFI crossing count (about 14 down to 1) and the norm/transpose order
(the fused path norms after the head transpose). Decode is GPU/bandwidth-bound
(the MoE decode-gap study measured roughly 95% GPU-bound, not FFI/dispatch), so
collapsing FFI calls buys nothing and norming the transposed layout costs a
little. This is the same outcome as M1 Ultra (1-3.4% slower) and is consistent
with M5 Max, which already prefers the split decode path for Gemma3n and where
the Turbo4 kernel measured 2.0x slower than its graph fallback.

## Correctness and determinism

- **RMS < 5e-3** vs the graph reference is covered by the
  `fused_qkv_split_norm_rope_standard_rmsnorm_matches_graph` unit test (the
  reduction is over the transpose-invariant `head_dim` axis).
- **Greedy temp-0 over 256 tokens:** qwen3-8b stays byte-identical to the graph
  path; qwen3-0.6b and qwen3-30b diverge at a near-tie argmax after about 64-140
  tokens into coherent, equal-quality continuations (`"the"` vs `"these"`,
  `"world"` vs `"land"`).
- **That divergence is not introduced by the fused path.** On CUDA the graph
  path is itself non-deterministic run-to-run at temp-0: two graph runs of
  qwen3-0.6b diverge after about 130 tokens, from non-deterministic GPU
  floating-point reduction order. The fused path is deterministic run-to-run, so
  it is more stable than the graph baseline and its divergence stays inside the
  graph's own run-to-run envelope. The parity result is therefore documented
  jitter, not a correctness regression.

## Conclusion

CUDA is not the per-backend win the fused QK-norm primitive was waiting for: on
GB10 it is 3.7-7.6% slower on the wired models with no upside. The primitive
stays default-OFF and ships as the shared building block for the deferred QK-norm
families (#326). The graph / split path remains the decode default on every
measured backend (M1 Ultra, M5 Max, GB10 / CUDA).
