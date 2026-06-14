# Fused decode-MoE kernel: design and roadmap (#268)

Follow-up to the [MoE decode gap investigation](moe-decode-gap-investigation.md).
That report localized the gap: at batch=1 decode the GPU is ~95% of the time but
only ~16-20% bandwidth-utilized, idling between many small kernels on the MoE
path. This doc scopes the kernel work that closes it and the harness that
directs it.

## Why the obvious small fusions do not help

It is tempting to start by fusing the cheap, self-contained ops. They are not
worth it:

- **Combine** (`moe_weighted_sum`: `expand_dims + astype + multiply + sum`):
  MLX already fuses the elementwise multiply at eval, so this is ~1-2 kernels.
  Collapsing it to one custom kernel saves ~1 dispatch/layer ≈ 48/token. At
  ~0.8 µs/dispatch that is ~38 µs against a ~21 ms token (~0.18%). Negligible,
  and matching the Rust path's accumulation dtype to stay greedy-identical is
  fiddle for no gain.
- **Router post-processing** (sigmoid + bias + top-k + gather + normalize):
  contains `argpartition` (a sort, not kernel-fusable) and small gathers; same
  marginal dispatch-count story.

Per-dispatch overhead is not the bottleneck (Step 0 showed the Rust graph build
is 0.8 ms/tok vs 20.3 ms GPU). The GPU **time** is the target, and it is spent
idling between the expert matmuls.

## The target: the expert path

Within the MoE block, the expert `gather_qmm` (gate/up/down) is ~39% of decode
and the surrounding small ops leave the GPU idle between them. The win is a
single Metal kernel for the **single-token decode** expert computation:

```
out[h] = sum_k scores[k] * down_k( swiglu( gate_k(x), up_k(x) ) )[h]
```

where `gate_k/up_k/down_k` are the affine-quantized (4-bit, and 6-bit for
dots.llm1) projections of the k-th selected expert. Doing the gather + dequant +
three matmuls + swiglu + weighted-sum in one launch keeps the GPU saturated for
the whole expert step instead of bubbling between `gather_qmm` calls.

### Approach

- Build via `mlx::core::fast::metal_kernel(name, inputs, outputs, source)` (the
  same JIT path as `ssm_update_kernel` / the gated-delta kernels in
  `mlx_cxx_bridge.cpp`). Template args carry `T` (activation dtype), `K`
  (top-k), `Din`/`Dff` (hidden / intermediate), `bits`, `group_size`.
- Inputs: `x [1, Din]`, `indices [K]`, packed `gate/up/down` weights + scales
  (+ biases for 6-bit), `scores [K]`. Output: `out [1, Din]`.
- The hard part is the in-kernel affine dequant-and-matmul (the `gather_qmm`
  equivalent): unpack `bits`-wide weights per `group_size`, scale, accumulate in
  f32. Start from MLX's `quantized_matmul` Metal source as the reference for the
  unpack/scale math; the kernel only needs the GEMV (L=1) case.
- Single-token decode only (`seq_len == 1` and the kernel is available);
  fall back to the existing `SwitchGLU` path for prefill and any unsupported
  bit width. Mirror the `seq_len == 1 && ssm_kernel_available()` dispatch used by
  the Mamba2 mixers.

### Risks

- Quantized unpack correctness across 4-bit and 6-bit (dots.llm1 mixed) and
  group sizes. Mitigated by the RMS gate below and per-bit-width unit tests.
- Register/threadgroup pressure for large `Dff`; may need tiling over `Dff`.
- Mixed per-tensor bits within one expert set (dots.llm1) may need a per-proj
  bit template arg.

## Validation harness (gate every change)

1. **Unit test** (`switch_layers` tests): fused kernel vs the existing
   `SwitchGLU::forward` on random inputs, RMS < 5e-3, for 4-bit and 6-bit.
2. **Greedy parity**: `mlxcel generate` temp-0 output byte-identical (or within
   the documented f16 jitter class) before/after, on qwen3-30b-a3b and dots.llm1.
3. **Decode bench**: `mlxcel-bench-decode` vs `scripts/bench_mlxlm.py` on
   qwen3-30b-a3b (cheap, 16 GB) each step; confirm the win carries to dots.llm1
   and nemotron-h.
4. **Trace check**: capture a Metal System Trace before/after (see
   `scripts/capture_moe_decode_trace.sh`) and confirm the inter-kernel GPU idle
   on the expert path shrank. The skill notes per-shader GPU time needs the
   Xcode GUI; the bundle Summary (Command Buffers / Compute Encoders / Dispatch
   Calls) is readable without Profile and is enough to confirm fewer/larger
   dispatches.

## Results

### Step 2a (correctness): one threadgroup per token

The first kernel computed the whole `out[Din]` in a single threadgroup (one
thread per output element, serial dequant-GEMV). Byte-identical to `SwitchGLU`
on qwen3-30b-a3b, but **~88x slower** (0.54 vs 47.4 tok/s): one threadgroup
runs on one GPU core, so 63 of the M1 Ultra's 64 cores idle. Correctness-first,
off by default.

### Step 2b (performance): SIMD reduction, then break the barrier

Two changes, each benched on qwen3-30b-a3b (M1 Ultra, best-of-5 decode tok/s,
greedy output byte-identical throughout):

1. **SIMD-cooperative GEMV** (one simdgroup per output row, 32 lanes stride the
   contraction dim + `simd_sum`), still one threadgroup per expert: **0.54 ->
   21.6 tok/s** (40x over 2a), but still 0.46x of gather_qmm. An occupancy sweep
   (redundant down-output tiling to force K*R threadgroups) confirmed the kernel
   is occupancy-bound, not tuning-bound:

   | threadgroups | 8 (K) | 16 | 32 | 64 |
   |---|--:|--:|--:|--:|
   | decode tok/s | 21.6 | 24.9 | 26.4 | 27.4 |

   Even at 64 threadgroups (all cores) it tops out at 27.4 (0.58x), because the
   redundant gate/up reads it needs to fill those threadgroups cost ~5.7x the
   weight bandwidth. A fully-fused single launch cannot escape this: the
   gate/up -> down dependency runs through the per-expert activation, so keeping
   it on-chip pins each expert to one threadgroup.

2. **Two-kernel non-redundant split** (stage the swiglu activation in global
   memory: kernel A = gate/up + swiglu -> `act_g[K, Dff]`, kernel B = down *
   score -> `partial[K, Din]`, summed over K). Every GEMV output row is now an
   independent simdgroup across all cores, each weight read exactly once:
   **49.0 vs 47.3 tok/s, +3.5%** over gather_qmm. This is the shipped kernel
   (`MLXCEL_FUSED_MOE`, off by default), `MLXCEL_FUSED_MOE_SGY` tunes simdgroups
   per threadgroup (default 8).

The win is modest because the expert GEMV is already a small, equally-efficient
fraction on both runtimes (the op-count investigation found mlxcel emits *fewer*
QuantizedMatmul than mlx-lm); the gain comes from folding swiglu/score into the
GEMV epilogues and reading each weight once across all cores. On a smaller, much
faster MoE (qwen1.5-moe-a2.7b, ~140 tok/s) the kernel is at parity (no
regression): the expert path is a smaller share there and gather_qmm already
runs it efficiently.

## Roadmap (one PR per step)

1. **Done** (#274): design + trace harness (`capture_moe_decode_trace.sh`).
2. **Done** (#275 = 2a, this PR = 2b): fused 4/8-bit expert kernel behind
   `MLXCEL_FUSED_MOE`, wired into `SwitchGLU` decode; two-kernel split beats
   gather_qmm by ~3.5% on qwen3-30b-a3b, byte-identical.
3. 6-bit / mixed-bit support; validate on dots.llm1 (its `down_proj` is 6-bit,
   so it falls back today). MLX packs non-power-of-2 bits with
   `get_bytes_per_pack` (3 bytes for 6-bit) rather than a clean shift-unpack.
4. Broader-fleet validation (phi-3.5-moe, qwen3-vl, mixtral, deepseek-v3,
   nemotron-h MoE) and a decision on flipping `MLXCEL_FUSED_MOE` on by default.
