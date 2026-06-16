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
   (`MLXCEL_FUSED_MOE`, on by default as of #282), `MLXCEL_FUSED_MOE_SGY` tunes simdgroups
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
2. **Done** (#275 = 2a, #276 = 2b): fused 4/8-bit expert kernel behind
   `MLXCEL_FUSED_MOE`, wired into `SwitchGLU` decode; two-kernel split beats
   gather_qmm by ~3.5% on qwen3-30b-a3b, byte-identical.
3. **Done** (#278): 6-bit / mixed-bit support; dots.llm1 (4-bit gate/up, 6-bit
   `down_proj`) wired. MLX packs non-power-of-2 bits with `get_bytes_per_pack`
   (3 bytes / 4 weights for 6-bit), so the down kernel reads the row as bytes
   and reconstructs with the `quantized.h` `qdot` bit layout.
4. **Done** (#279 = qwen3_next/qwen3.5/3.6, #280 = squared-ReLU kernel preserved
   for nemotron-class behind `MLXCEL_FUSED_MOE_RELU2`, #281 = GeGLU/gemma4).
5. **Done** (#282): validated the fused path on M5 (Neural Accelerator) and
   flipped `MLXCEL_FUSED_MOE` on by default. The documented M5-Max mixed-dtype
   NaN risk does not materialize in the new kernels: across the wired MoE set the
   greedy decode output is byte-identical or within the f16 jitter class with no
   NaN, and decode is parity-to-faster (no regression). Per-layer kernel
   execution on M5 was confirmed with a dispatch trace. See the M5 results below.

## Usage and flags

The fused decode-MoE path is **on by default** as of #282 and applies only to
single-token decode with affine 4/8-bit gate/up and 4/6/8-bit down. Anything
else (prefill, non-affine, unsupported bit widths, mismatched gate/up bits)
falls back to the proven `gather_qmm` / `SwitchGLU` path automatically. Set
`MLXCEL_FUSED_MOE=0` to force that fallback everywhere.

| Env var | Default | Effect |
|---------|---------|--------|
| `MLXCEL_FUSED_MOE` | unset (on) | On by default. Set to `0` (also `false`/`off`/`no`, case-insensitive) to force the proven `gather_qmm` / `SwitchGLU` path; any other value, or leaving it unset, keeps the kernel on. |
| `MLXCEL_FUSED_MOE_SGY` | 8 | Simdgroups per threadgroup (one output row each). Tune per hardware. |
| `MLXCEL_FUSED_MOE_RELU2` | unset (off) | Enable the squared-ReLU (fc1/relu²/fc2) fused path used by nemotron-class experts. Correct but measured performance-neutral on nemotron-h; kept for a future MoE-dominated squared-ReLU model. |
| `MLXCEL_FUSED_MOE_MAX_DFF` | 4096 | Expert-intermediate (Dff) upper bound. Above it, `forward_fused_kernel` declines and the caller falls back to `gather_qmm`. The fused path wins only while `gather_qmm` underutilizes the GPU (small experts); for large experts `gather_qmm` already saturates and the extra dispatch plus global-memory activation staging is a net loss. M1 Ultra measurements: Dff 704..2560 gain +3.5% to +15.4%, Dff 6400 (phi-3.5-moe) loses 5.9%, Dff 14336 (mixtral) loses 21%. The break-even is hardware-dependent, so the bound is tunable. |

### Measured decode gains (M1 Ultra, `MLXCEL_FUSED_MOE=1`)

| Model | activation | bits | decode tok/s | gain | greedy parity |
|-------|-----------|------|-------------:|-----:|---------------|
| gemma-4-26b-a4b-it | GeGLU | 4 | 73.8 → 83.2 | **+13%** | byte-identical (use the chat template) |
| qwen3.5/3.6-35b-a3b | SwiGLU | 4 | 68.7 → 74.7 | +8.7% | within f16 jitter class |
| dots.llm1 | SwiGLU | 4 + 6 | 13.1 → 13.7 | +4.7% | byte-identical |
| qwen3-30b-a3b | SwiGLU | 4 | 47.3 → 49.0 | +3.5% | byte-identical |
| qwen1.5-moe-a2.7b | SwiGLU | 4 | ~140 | ~par | byte-identical |
| nemotron-h-30b | relu² | 4 | 54.9 → 54.7 | ~0% | byte-identical (off the default path) |

The gain tracks how MoE-dominated the decode is and how inefficient the
baseline was: gemma4 wins most (small experts, and its compiled-SwitchGeGLU
baseline was three `gather_qmm`), while nemotron-h barely moves because its
decode is dominated by Mamba2 + attention.

### Measured decode gains (M5 Max, `MLXCEL_FUSED_MOE=1`, #282)

Validated on M5 Max (Neural Accelerator, macOS 26.5) against the true `gather_qmm`
baseline (env unset, since `MLXCEL_FUSED_MOE=0` only began disabling the kernel in
#282). Greedy temp-0; best-of-3 decode tok/s, dots.llm1 single-pass.

| Model | activation | bits | decode tok/s | gain | greedy parity |
|-------|-----------|------|-------------:|-----:|---------------|
| gemma-4-26b-a4b-it | GeGLU | 4 | 133.9 → 142.2 | +6.2% | within f16 jitter class |
| qwen3.5-35b-a3b | SwiGLU | 4 | 153.6 → 161.3 | +5.1% | within f16 jitter class |
| qwen3-30b-a3b | SwiGLU | 4 | 60.5 → 61.7 | +1.9% | byte-identical |
| dots.llm1 | SwiGLU | 4 + 6 | 14.1 → 14.0 | ~par | byte-identical |

The M5 gains are smaller than M1 Ultra's but positive with no regression, and no
run produced NaN or garbage. The Neural Accelerator narrows the gap because it
already accelerates the baseline expert matmul, leaving less inter-kernel idle for
the fusion to recover. Per-layer kernel execution on M5 was confirmed with a
dispatch trace (30 / 40 / 48 / 61 dispatches per decode token for gemma4 /
qwen3.5 / qwen3-30b / dots.llm1).

**Parity caveat.** The kernel accumulates the GEMV in f32 with a different
reduction order than `gather_qmm`'s tiling, so it is within the f16 jitter
class (RMS < 5e-3), not bit-exact, for every model. Many models happen to be
byte-identical for typical prompts; on a deep hybrid (qwen3.5/3.6) the ~1e-4
per-block perturbation can flip a near-tie greedy token. This is the design
harness's accepted "byte-identical or within f16 jitter class" outcome, and is
the expected numerical consequence of the fusion.

### Models covered

The fused single-token decode dispatch is wired into eight model paths: qwen3_moe
(Qwen3 MoE), qwen3_next (qwen3.5/3.6), dots.llm1 (mixed 4/6-bit), gemma4 (GeGLU),
qwen2_moe (qwen1.5-moe / Qwen2-MoE; migrated from its local `SwitchGLU` to the
shared one, which gained a per-expert stacking loader for the `experts.{idx}`
checkpoint layout), mixtral (Mixtral 8x7B; SwiGLU, softmax-routed with no shared
expert; migrated from its local `SwitchGLU` to the shared one, which gained an
overridable projection-leaf-name loader for the `w1`/`w2`/`w3` checkpoint
convention, mapping gate=w1, up=w3, down=w2; Mixtral's expert intermediate is
14336, above `MLXCEL_FUSED_MOE_MAX_DFF`, so the kernel declines and decode stays
on `gather_qmm` (the migration removes duplication; the dispatch arms only if the
bound is raised)), lfm2 (LFM2-MoE; sigmoid-routed,
optional expert_bias and norm_topk_prob), and qwen3_vl_moe (Qwen3-VL MoE; imports
`SwitchGLU` from qwen3_moe, SwiGLU activation, text-only decode path). Other MoE
families reuse the shared `SwitchGLU` for the expert matmul but were not wired with
the fused decode dispatch, so they stay on `gather_qmm` regardless of
`MLXCEL_FUSED_MOE`: olmoe, minimax, and phimoe. nemotron-h's MoE runs through the
separate C++ `fused_moe_forward` and is wired behind `MLXCEL_FUSED_MOE_RELU2` only.
