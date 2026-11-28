# PR #3405: Byte-exact speculative verify for Inkling 2

**Date**: 2026-11-25
**Status**: Draft PR open (refs #3280). Inkling 2 and core suites pass; validated on an M3 Ultra (512 GB) on `models/inkling-2-mini`, `inkling-2-mini-4bit` and `inkling-2-small`

## Executive Summary

Before this PR, the block-versus-chain exactness probe declined the DSpark drafter bundled with Inkling 2 Mini and Small at every width, so the drafter stayed off unless `MLXCEL_MTP_ALLOW_INEXACT=1`. A per-visit comparison found five sources of difference. Fixing them makes an 8-row verify block equal eight decode steps bit for bit, with contexts up to 8000 tokens. The drafter-on text now equals classic decode byte for byte, including a 2006-token prompt that crosses both the sliding window and the prefill chunk.

The main piece is a new core module, `mlxcel_core::row_invariant`, backed by a Metal matvec whose summation order does not depend on the row count. Inkling 2 is its first user. Decode with the kernel is 1.4% faster than MLX gemv on Mini and 0.9% faster on Mini 4-bit. The drafter gives 1.60x on a code prompt and 1.04x on prose on Mini, and 1.54x and 0.90x on Small.

## 1. Problem Statement

Speculative decoding at temperature 0 promises the classic-decode text. mlxcel enforces that with a probe: it runs K single-token steps and one K-row block from the same cache and compares the logits bytes. If they differ, the drafter is declined. On Inkling 2 the probe failed at width 2 and width 8 on both checkpoints. The failure also showed with `qmv_wide` disabled, so the quantized expert matvec was not the cause.

## 2. Diagnosis

`exactness_diag_tests.rs` (ignored, real checkpoint) captures the hidden state entering every layer visit for each block row and each chain step, and prints the differing element count per row. It found the causes one at a time:

| # | Cause | First visible | Fix |
|---|---|---|---|
| 1 | MLX runs `matmul` with one row through gemv and with several rows through gemm. The two sum in different orders. MLX also folds a shared-weight batch axis into M, so reshaping the rows into a batch does not help. | rows 1+ from layer 0; row 0 after layer 6 | every dense product goes through `row_invariant::dense` |
| 2 | The per-query zero sink column sat after the newest key: index N+1 in a step, N+K in a block. This moves the SDPA lane assignment even for row 0. | layer 13 SDPA, 1-element differences | sink column first |
| 3 | The message read `query @ bank^T` in the looped core is a batched matmul over the block length. | visit 18 | per position inside a verify block |
| 4 | Past a sliding layer's 1024-token window, row r of a block also sees the r oldest keys, masked out, which decode step r has already dropped. | rows 1+ after a prompt longer than 1024 | per-row attention from each row's window start |
| 5 | The DSpark prefill ran the prompt in one forward. Classic decode prefills in 2048-token chunks. | prompts over 2048 | prefill in the classic chunks |

Causes 4 and 5 were invisible to the probe, which draws 32-token prompts. Cause 4 showed up in the tiny-model test, whose window is 4 tokens. Cause 5 showed up in an end-to-end run with a long prompt.

Ruled out along the way: the NVFP4/MXFP8 expert kernels (both row-invariant already), the #3359 Mini kernels, the short convolution and the speculative append scope.

## 3. Design

### 3.1 `row_invariant_matvec` (`src/lib/mlxcelverse/kernels/row_invariant_matvec.{h,cpp}`)

The kernel computes `x [M, K] @ w [N, K]^T` for 1 to 16 rows, bf16, f16 or f32, with `K % 4 == 0`. Each row has its own f32 accumulator, and each lane walks `k = 4 * lane + 128 * j` in a fixed order, so the M-row output equals the M one-row outputs. The weight is read once per block rather than once per row.

There are two variants. Both are chosen from N alone, never from M:

- **Wide** (N ≥ 512): one SIMD group per output column. This tiling was the fastest of those tried; 2, 4 and 8 columns per group were slower in real decode.
- **Split** (N < 512, the router and the controller down projection): eight SIMD groups share a column and add their partials in group order. The first, single-variant version cost 2.5 to 3.4% of decode on these narrow sites.

### 3.2 `mlxcel_core::row_invariant`

A model opts in by entering `Scope::enter(Mode { kernel, verify_block })` around every forward, decode and verify alike, and building its products with:

- `dense` / `linear`
- `per_row`, for quantized projections, which the kernel does not take
- `per_position`, for other products whose kernel depends on the block length

Outside a scope these are plain `matmul` and `forward`, so no other model changes. This opt-in boundary is deliberate. Wiring the kernel into `UnifiedLinear::forward` would change the decode bits and speed of every model at once. With `kernel` off, or off Metal, a verify block multiplies one row at a time. That is exact but reads the weight K times; an 8-row Mini verify takes 80.6 ms instead of 51.2 ms.

### 3.3 Inkling 2 wiring

- `Inkling2Model` caches `row_kernel = kernels::enabled()` at build and enters the scope in `hidden_rows` and `project`.
- The seven attention projections, the router, the message controllers, the dynamic conv projection and the LM head use the helpers.
- Dense (unquantized) experts are split per token inside a verify block, because `gather_mm` also switches kernel with the row count.
- The sliding-window rule is in `attention.rs::attend`. The shared SDPA body moved to `sdpa()`.
- `Inkling2Model::prefill_speculative` serves both the text and the VL speculative targets. It evaluates each chunk, keeps the last chunk's logits and concatenates the captures.

### 3.4 Build fix

`mlxcel-core/build.rs` did not list the Inkling 2 kernel `.cpp` files in `rerun-if-changed`, so editing them did not trigger a rebuild. This had been the case since #3369. Two rounds of kernel measurements in this PR silently ran stale code before this was found.

## 4. Results

M3 Ultra, serialized under the GPU lock, ABAB order, 256 greedy tokens. Full tables are in `docs/benchmark_results/inkling2-verify-exactness-m3ultra-2026-11-24.md`.

| Measure | Result |
|---|---|
| Probe, widths 2/4/8 | equal on Mini, Small, Mini 4-bit |
| Block vs chain after a long prompt | equal at 1100, 3000, 8000 (Mini); 1100 (Small, 4-bit) |
| Decode, kernel vs MLX gemv | Mini +1.4%, 4-bit +0.9% |
| Mini 8-row verify | 51.2 ms (inexact before: 52.1 ms) |
| DSpark Mini, prose / code | 1.04x / 1.60x, text equals classic |
| DSpark Small, prose / code | 0.90x / 1.54x, text equals classic |
| DSpark Mini, 2006-token prompt | 1.00x, text equals classic |
| mlx-vlm text trace, Mini 4-bit | 31/1024 top-1, 0/372 decided (before: 33, 0) |

On Small, a verify round costs about 120 ms against a 51 ms decode step, and the expert path dominates that cost. A prose round emits 2.3 tokens, which does not pay for the round.

## 5. Tests

- `row_invariant_tests.rs` (core):
  - every row of a kernel product equals its one-row product, on both variants and in f32 and bf16;
  - the values agree with an f64 dot product;
  - the contract refuses unsupported shapes;
  - the scope semantics hold.
- `the_verify_block_matches_the_chain_on_metal`: the tiny model passes the probe at widths 2, 4 and 8, looped and not. The tiny `head_dim` went from 8 to 64 so attention takes the fused SDPA kernel that real checkpoints use. At 8, MLX falls back to an unfused path no checkpoint takes.
- Inkling 2 lib suite: 173 passed.

## 6. Limits and Follow-ups

- **Probe prompt length.** The probe still draws short prompts. It cannot see a long-context regression, but the diagnostic test covers that by hand (`INKLING2_DIAG_PROMPT`).
- **Server prefill.** The server prefills with `--prefill-chunk-size`. The DSpark prefill uses `MLXCEL_PREFILL_CHUNK`, so a server started with a non-default chunk size keeps the old difference for prompts longer than one chunk.
- **Quantized attention projections** (affine 4-bit checkpoints) take the row-at-a-time path inside a verify block. Those checkpoints have no bundled drafter, so only prompt lookup pays for it.
- **Generalization.** Other drafter-capable models will be surveyed in a follow-up issue: which ones the probe declines, and whether `row_invariant` fixes them. Small's verify cost will be part of the same follow-up.
