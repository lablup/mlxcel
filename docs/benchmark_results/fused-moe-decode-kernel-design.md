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
6. **Done** (#319): ported the kernel to the CUDA backend via
   `mx.fast.cuda_kernel` (the `simd_sum` reduction becomes `__shfl_down_sync`,
   precise `expf` for greedy parity), runtime-selected by `metal::is_available()`.
   Before this the kernel was Metal-only and aborted on CUDA (`[metal_kernel] No
   Metal back-end`) for the small-expert MoE models the dispatch selected; they
   now run on the fused path, byte-identical and faster (qwen3-30b-a3b +55% on
   GB10). fused stays default-on for both backends. See the GB10 results below.

## Usage and flags

The fused decode-MoE path is **on by default** as of #282 (Metal) and #319
(CUDA, via `mx.fast.cuda_kernel`), and applies only to single-token decode with
affine 4/8-bit gate/up and 4/6/8-bit down. Anything
else (prefill, non-affine, unsupported bit widths, mismatched gate/up bits)
falls back to the proven `gather_qmm` / `SwitchGLU` path automatically. Set
`MLXCEL_FUSED_MOE=0` to force that fallback everywhere.

| Env var | Default | Effect |
|---------|---------|--------|
| `MLXCEL_FUSED_MOE` | unset (on) | On by default. Set to `0` (also `false`/`off`/`no`, case-insensitive) to force the proven `gather_qmm` / `SwitchGLU` path; any other value, or leaving it unset, keeps the kernel on. |
| `MLXCEL_FUSED_MOE_SGY` | 8 | Simdgroups per threadgroup (one output row each). Tune per hardware. |
| `MLXCEL_FUSED_MOE_RELU2` | unset (off) | Enable the squared-ReLU (fc1/relu²/fc2) fused path used by nemotron-class experts. Correct but measured performance-neutral on nemotron-h; kept for a future MoE-dominated squared-ReLU model. |
| `MLXCEL_FUSED_MOE_MAX_DFF` | 4096 | Expert-intermediate (Dff) upper bound. Above it, `forward_fused_kernel` declines and the caller falls back to `gather_qmm`. The fused path wins only while `gather_qmm` underutilizes the GPU (small experts); for large experts `gather_qmm` already saturates and the extra dispatch plus global-memory activation staging is a net loss. M1 Ultra measurements: Dff 704..2560 gain +3.5% to +15.4%, Dff 6400 (phi-3.5-moe) loses 5.9%, Dff 14336 (mixtral) loses 21%. On CUDA (GB10) the crossover is much higher, ~13-14k (Dff 768 +60%, 1792 +13%, 6400 +2%, 14336 -2%), so 4096 is conservative there but kept as the shared default. The break-even is hardware-dependent, so the bound is tunable. |

### Measured decode gains (M1 Ultra, `MLXCEL_FUSED_MOE=1`)

decode tok/s pairs are `MLXCEL_FUSED_MOE=0` (gather_qmm baseline) followed by the
default fused path. The first four rows and nemotron-h are from the kernel
development passes; qwen3-vl-30b-a3b, lfm2-8b-a1b, and qwen1.5-moe are from the
`--profile` decode benchmark (prompt "Hello, how are you today?", 100 tokens,
median of 3) added when those families were wired (epic #307).

| Model | activation | bits | decode tok/s | gain | greedy parity |
|-------|-----------|------|-------------:|-----:|---------------|
| qwen3-vl-30b-a3b (text path) | SwiGLU | 4 | 69.3 → 82.3 | **+18.8%** | within f16 jitter class |
| gemma-4-26b-a4b-it | GeGLU | 4 | 73.8 → 83.2 | +13% | byte-identical (use the chat template) |
| qwen3.5/3.6-35b-a3b | SwiGLU | 4 | 68.7 → 74.7 | +8.7% | within f16 jitter class |
| dots.llm1 | SwiGLU | 4 + 6 | 13.1 → 13.7 | +4.7% | byte-identical |
| qwen3-30b-a3b | SwiGLU | 4 | 47.3 → 49.0 | +3.5% | byte-identical |
| lfm2-8b-a1b (LFM2-MoE) | SwiGLU | 4 | 168.9 → 174.7 | +3.4% | byte-identical |
| qwen1.5-moe-a2.7b | SwiGLU | 4 | 143.0 → 146.1 | +2.2% | within f16 jitter class |
| nemotron-h-30b | relu² | 4 | 54.9 → 54.7 | ~0% | byte-identical (off the default path) |

The gain tracks how MoE-dominated the decode is and how inefficient the baseline
was: qwen3-vl-30b-a3b and gemma4 win most (small experts, MoE-dominated decode;
gemma4's compiled-SwitchGeGLU baseline was three `gather_qmm`), while nemotron-h
barely moves because its decode is dominated by Mamba2 + attention. Three wired
families do not gain and stay on `gather_qmm`: mixtral (Dff 14336) and phi-3.5-moe
(Dff 6400) sit above `MLXCEL_FUSED_MOE_MAX_DFF`, so the kernel declines and decode
is unchanged (mixtral ~53 tok/s, phi-3.5-moe ~75 tok/s); olmoe (Dff 1024, top_k=8)
dispatches but is perf-neutral (~272 tok/s on and off). All three remain within
the f16 jitter class of their baselines, with no regression.

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

### Measured decode gains (GB10 / CUDA, `MLXCEL_FUSED_MOE=1`, #319)

The kernel was ported to the CUDA backend via `mx.fast.cuda_kernel` (#319), the
CUDA analogue of `metal_kernel`: the same two dispatches, with the `simd_sum`
warp reduction expressed as `__shfl_down_sync` and a precise `expf` in the
SwiGLU. `run_fused_moe_two_kernel` picks the cuda_kernel port when
`metal::is_available()` is false. Before #319 the kernel was Metal-only, so the
selected small-expert MoE models aborted on CUDA with `[metal_kernel] No Metal
back-end`; they now run on the fused path. Greedy output is byte-identical to
`gather_qmm` on qwen3-30b-a3b. Measured on GB10 (DGX Spark, CUDA 13.0) against
the `gather_qmm` baseline (`MLXCEL_FUSED_MOE=0`):

| Model | activation | bits | decode tok/s | gain | greedy parity |
|-------|-----------|------|-------------:|-----:|---------------|
| qwen3-30b-a3b | SwiGLU | 4 | 58.2 → 90.3 | **+55%** | byte-identical |
| qwen3.5-35b-a3b | SwiGLU | 4 | 45.8 → 64.7 | +41% | within f16 jitter class |
| lfm2-8b-a1b | SwiGLU | 4 | 140.9 → 158.2 | +13% | byte-identical |
| qwen1.5-moe-a2.7b | SwiGLU | 4 | 113.0 → 124.7 | +10% | byte-identical |

The CUDA gains are larger than Apple Silicon's: at batch=1 `gather_qmm` leaves
more of the GB10 GPU idle, so the all-warps fused path (one warp per output row)
recovers more. The break-even with expert size is also far higher than Metal's.
Forcing the fused path past the cap (`MLXCEL_FUSED_MOE_MAX_DFF=20000`):

| Dff | model | fused vs gather_qmm |
|----:|-------|--------------------:|
| 768 | qwen3-30b-a3b | +60% |
| 1792 | lfm2-8b-a1b | +13% |
| 6400 | phi-3.5-moe | +2% |
| 14336 | mixtral | −2% |

The CUDA crossover is ~13–14k (vs Metal's ~4096, where phi-3.5-moe already loses
5.9%). The 4096 cap is therefore conservative on CUDA but kept as the shared
default: the meaningful wins are all below it, and the 4096–14k range gains at
most ~2% (phi-3.5-moe) while mixtral (14336) slightly prefers `gather_qmm`. CUDA
users with mid-size experts can raise `MLXCEL_FUSED_MOE_MAX_DFF`.

**Parity caveat.** The kernel accumulates the GEMV in f32 with a different
reduction order than `gather_qmm`'s tiling, so it is within the f16 jitter
class (RMS < 5e-3), not bit-exact, for every model. Many models happen to be
byte-identical for typical prompts; on a deep hybrid (qwen3.5/3.6) the ~1e-4
per-block perturbation can flip a near-tie greedy token. This is the design
harness's accepted "byte-identical or within f16 jitter class" outcome, and is
the expected numerical consequence of the fusion.

### Models covered

The fused single-token decode dispatch is wired into eleven model paths: qwen3_moe
(Qwen3 MoE), qwen3_next (qwen3.5/3.6), dots.llm1 (mixed 4/6-bit), gemma4 (GeGLU),
qwen2_moe (qwen1.5-moe / Qwen2-MoE; migrated from its local `SwitchGLU` to the
shared one, which gained a per-expert stacking loader for the `experts.{idx}`
checkpoint layout), mixtral (Mixtral 8x7B; SwiGLU, softmax-routed with no shared
expert; migrated from its local `SwitchGLU` to the shared one, which gained an
overridable projection-leaf-name loader for the `w1`/`w2`/`w3` checkpoint
convention, mapping gate=w1, up=w3, down=w2; Mixtral's expert intermediate is
14336, above `MLXCEL_FUSED_MOE_MAX_DFF`, so the kernel declines and decode stays
on `gather_qmm` (the migration removes duplication; the dispatch arms only if the
bound is raised)), lfm2 (LFM2-MoE; sigmoid-routed, optional expert_bias and
norm_topk_prob), qwen3_vl_moe (Qwen3-VL MoE; imports `SwitchGLU` from qwen3_moe,
SwiGLU activation, text-only decode path), phimoe (Phi-3.5-MoE; migrated from
its local `SwitchGLU`/`SwitchLinear` to the shared ones; checkpoints pre-stacked
under `block_sparse_moe.switch_mlp.{gate,up,down}_proj`; `sanitize_weights` still
handles the unstacked `experts.{i}.w1/w2/w3` layout for community checkpoints; the
expert intermediate is 6400, above `MLXCEL_FUSED_MOE_MAX_DFF`, so like mixtral the
kernel declines and decode stays on `gather_qmm`), olmoe (OLMoE; migrated from
its local `SwitchGLU`/`SwitchLinear` to the shared ones; SwiGLU, softmax-routed
with optional `norm_topk_prob`, weights stacked under `switch_mlp.{gate,up,down}_proj`
or joined from the per-expert `experts.{i}` layout by `sanitize_weights`; expert
intermediate is 1024, below `MLXCEL_FUSED_MOE_MAX_DFF`, so the kernel dispatches at
decode and delivers a real throughput gain), and minimax (MiniMax-M2; sigmoid-routed
with e_score_correction_bias, unbiased scores selected, SwiGLU/SiLU activation,
no shared expert; the dispatch uses the default swiglu act=0 template; wired in
#304; runtime validation is hardware-blocked because MiniMax-Text-01 at ~456B does
not fit 128 GB unified memory, so greedy temp-0 throughput and output-parity
measurements are deferred until a smaller variant or larger-memory machine is
available; a runtime smoke on minimax-m2-3bit (M1 Ultra, 16 tokens at 17.6 tok/s)
confirmed the dispatch wiring is non-breaking: 3-bit quant is not a supported
fused-kernel input, so `forward_fused_kernel` returned `None` and decode fell back
to `gather_qmm`, generating coherent output with no crash or OOM; fused-path
throughput and output-parity validation remain blocked on a fitting 4-bit or 8-bit
minimax checkpoint). nemotron-h's MoE runs through the separate C++ `fused_moe_forward` and is wired
behind `MLXCEL_FUSED_MOE_RELU2` only.
