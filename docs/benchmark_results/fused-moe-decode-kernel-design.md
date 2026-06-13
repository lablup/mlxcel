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

## Roadmap (one PR per step)

1. **This PR**: design + trace harness (`capture_moe_decode_trace.sh`). No
   model behavior change.
2. Fused 4-bit expert kernel behind a dispatch guard + unit test; wire into
   `SwitchGLU` decode; validate + bench on qwen3-30b-a3b.
3. 6-bit / mixed-bit support; validate on dots.llm1.
4. Fold the router scoring + combine into the same launch if the trace shows
   residual idle; re-bench across qwen3-30b-a3b / dots.llm1 / nemotron-h.
