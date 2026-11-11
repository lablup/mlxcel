# PR #2960: A narrow tuning matcher for the GQA decode context contraction

**Date**: 2026-11-11
**Status**: Completed (draft PR; closes #2956)

## Executive Summary

The decode attention item of #2921 split into two dispatches that cost about 100 us each at capacity 256. The context matmul turned out to be a lowering problem: because `group` is 4, ROCm selects enough wave-level tiling that IREE's compiler leaves the reduction at one tile and the batch dimension at one, which was a quarter of the tuned score dispatch per MAC. A third `foreach_match` branch in the ROCm tuning spec tiles all four query rows and all 64 output columns per workgroup and steps the key axis in 64-element chunks, dropping the context dispatch from `6.343 us` to `1.806 us` at capacity 256.

The KV read convert is a negative result. `elementwise_transpose_256x8x64_f32xf16` moves 768 KB per call in 3.2 us, which is about 240 GB/s, so it is bandwidth-bound under the f32 KV caches that `xla_validate_kv_pair` pins. No lowering change can remove it; it needs an f16 cache, which is an ABI and numerics decision the issue excluded, so it is recorded here rather than implemented.

The combined step saving is `72.6 us` at capacity 256 (about 35 percent of the 206 us item, against the issue's 30 percent), but reported wall time is flat there (`140.5` to `140.0 tok/s` over five interleaved rounds) and improves only at larger capacities, where the dispatch dominates: `123.8` to `130.0 tok/s` at 512 and `82.6` to `87.4 tok/s` at 1024. The honest statement is that the dispatch time is fixed and the end-to-end benefit depends on the measured capacity.

## 1. Problem Statement

Issue #2956 asks for two things at the last decode-attention profile:

1. The KV read convert (`elementwise_transpose_256x8x64_f32xf16`), 32 calls per step at 3.2 us, 102.9 us per step.
2. The context matmul (`main_dispatch_11_batch_matmul_8x4x64x256_f16xf16xf32`), 16 calls per step at 6.343 us, 101.5 us per step.

The target was a 30 percent cut of the combined 205 us, about 62 us. The profile was taken on `llama-3.2-1b-instruct-4bit` at `gfx1151`, `MLXCEL_XLA_CONTEXT_CAPACITY=256`, `n=128` minus `n=1` so the per-step attribution is not diluted by startup.

## 2. Root Cause

### 2.1 The context matmul lowering

The contraction is `[8, 4, K] x [8, K, 64] -> [8, 4, 64]`: batch 8 (the KV heads), `M = 4` (the query group), `N = 64` (the head width) and `K` = keys. Only 256 multiply-accumulates per output element exist at capacity 256, and the compiler default assigns one workgroup per output element with no reuse across the four query rows: grid `256x8`, workgroup `32x8` (256 threads), VGPR 104.

That is the same output-element-per-workgroup layout the score matmul had before #2921's tile, except here the work per element is 64 MACs against the score's 2048. The score matcher does not fire because the roles are swapped: score reduces over `head_dim` and writes `keys`, context reduces over `keys` and writes `head_dim`.

### 2.2 The KV read convert

Each call reads a 512 KB f32 block and writes a 256 KB f16 block. At 3.2 us that is 768 KB in 3.2 us, about 240 GB/s, which is a plausible fraction of the achievable rate on a generic local-memory host build and certainly not an obviously mis-lowered access pattern. Changing it means changing the KV cache dtype, which the issue excludes.

## 3. Change

`src/lib/mlxcel-xla/assets/rocm/bigk_vecmat_tuning.mlir` gains a third named sequence, `@match_gqa_context`, wired into `@__kernel_config` after `@match_gqa_score`:

```mlir
transform.iree.match.contraction %arg0,
  lhs_type = f16, rhs_type = f16, output_type = f32,
  indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                   affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
                   affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
transform.iree.match.dims_equal %batch_dims, [8]
transform.iree.match.dims_equal %m_dims, [4]
transform.iree.match.dims_equal %n_dims, [64]
transform.iree.match.dim_bounds %rhs[1], umin = 128, none
#iree_gpu.lowering_config<{reduction = [0, 0, 0, 64], thread = [1, 1, 2, 0],
                           workgroup = [1, 4, 64, 1]}>
pipeline = #iree_gpu.pipeline<TileAndFuse> workgroup_size = [32, 4, 1] subgroup_size = 32
```

The four predicates are what keep it from firing elsewhere:

- `batch = 8` is `n_kv` for the single-sequence graph. Ragged and uniform-B graphs collapse `bsz * n_kv` into the batch of the same rank-3 shape, so without this they would pick up a tile measured for one row. This is the same load-bearing predicate #2921 established for the score contraction, and the header comment says so.
- `M = 4` is the query group. Prefill's M axis carries the prompt length, so prefill cannot match.
- `N = 64` is the head width; other widths keep the compiler default.
- `K >= 128` keeps the tile from a very short KV, where the reduction predication would matter more than the tiling.

`src/lib/mlxcel-xla/src/tuning.rs` documents the third matcher and adds `tuning_spec_matches_the_gqa_decode_context_shape`, which pins the batch, M and N predicates, the reduction bound and the workgroup size against accidental widening. The existing score test now also asserts that the context line is still present in the entrypoint.

## 4. Measurements

Protocol: `llama-3.2-1b-instruct-4bit`, `gfx1151`, exclusive GPU lock, `rocprofv3` traces at `n = 128` minus `n = 1` for per-step attribution; wall time from five interleaved rounds at capacity 256 and three rounds at 512 and 1024.

### 4.1 Per-call dispatch time

| Capacity | Default us/call | Default us/step (16) | New us/call | New us/step | Saved |
|----------|-----------------|----------------------|-------------|-------------|-------|
| 256 | 6.343 | 101.5 | 1.806 | 28.9 | 72.6 |
| 512 | 11.967 | 191.5 | 2.813 | 45.0 | 146.4 |
| 1024 | 24.890 | 398.2 | 5.923 | 94.8 | 303.5 |

The default rows at 512 and 1024 are the same protocol at those capacities; the new rows are the runtime dumps and traces at those shapes.

### 4.2 End to end

| Capacity | Default tok/s | New tok/s | Rounds |
|----------|---------------|-----------|--------|
| 256 | 140.5 | 140.0 | 5 |
| 512 | 123.8 | 130.0 | 3 |
| 1024 | 82.6 | 87.4 | 3 |

At capacity 256 the 72.6 us per step is about 1 percent of a 7 ms step, which is why the medians are flat and the 5 percent at 1024 shows. The change is not a throughput claim at this shape; it removes dispatch time that the profile attributed to the item.

## 5. Evaluation

### 5.1 What the matcher fires on

From compiled configuration dumps plus trace call counts:

| Graph | bigk | score | context | Calls per step |
|-------|------|-------|---------|----------------|
| Single-sequence decode | 1 | 1 | 1 | 16 each |
| Prefill | 0 | 0 | 0 | - |
| Ragged B=4 | 0 | 0 | 0 | - |
| Ragged B=8 | 0 | 0 | 0 | - |
| f16-weight decode graph | 1 | 1 | 1 | - |

The f16-weight row is the control that the edit did not disable the existing matchers: with f16 projections the bigk vecmat is live, and it still matches.

### 5.2 Verification

- `cargo fmt --all -- --check` clean.
- `cargo clippy -p mlxcel-xla --features iree --lib --tests -- -D warnings` clean.
- `cargo test -p mlxcel-xla --lib tuning::` 3 passed.
- `single_decode_score_contraction_keeps_the_tuning_spec_shape` passed.
- `validation::tests::emit_all_reproduces_registered_goldens` passed (no emitter or asset change).
- Greedy output identical to the default spec on two prompts.
- `xla_batch_bench` with `MLXCEL_XLA_PRECISION=f16`: `REFERENCE-EXACT PASS` at B=4 (N=8, maxcap 24) and B=8 (N=16, maxcap 48).

## 6. Remaining

The KV read convert stays at 102.9 us per step because both caches are f32 and the transpose moves 768 KB per call. An f16 KV cache would remove the convert, but it changes the numerical path and the `xla_validate_kv_pair` contract, so it belongs in its own issue with its own precision argument.

## 7. Change Summary

| Item | Value |
|------|-------|
| Files changed | 2 |
| Lines added | +108 |
| Lines deleted | -18 |
| Commits | 1 |

Related: #2921 (the score matcher this mirrors), #2960 (this PR), #2956 (this issue).
