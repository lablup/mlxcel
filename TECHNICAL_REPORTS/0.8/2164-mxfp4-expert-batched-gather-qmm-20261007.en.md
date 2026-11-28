# Technical Report: PR #2164 - Batch Sorted mxfp4 gather_qmm Prefill by Expert

**Date**: 2026-10-07

**Status**: Implemented and verified on the gfx1151 host; head `e87ffa7d` (up to date with origin/main `5c851fc4`), PR open, pending merge.

**Languages**: HIP C++ (`src/lib/mlx-cpp/patches-rocm/mlx/backend/rocm/quantized/qmm.hip`), Rust (`tests/rocm_gather_qmm_expert_batched.rs`), Markdown (`LOCAL_FIXES.md`, two benchmark pages)

**Risk Level**: Low to medium (one kernel arm and one dispatch condition in the ROCm overlay, on by default; no Metal or CUDA file is touched, and `MLX_ROCM_GATHER_QMV_EXPERT_BATCHED=0` restores the previous dispatch)

## Executive Summary

Issue #2106 (part of #1814, epic #1801, split out of #2066) reported that `gpt-oss-20b-MXFP4-Q4` prefilled 512 tokens on gfx1151 in about 67.8 s, with 99.78% of GPU time in the per-row non-affine `gather_qmv_kernel<hip_bfloat16, unsigned char, 4, 32, false>` at about 940 ms per call. `GatherQMM::eval_gpu` sent only affine weights to `gather_qmv_expert_batched_kernel`, the kernel that reads each expert's weights once per four routed rows, so mxfp4 reread every expert's weights for every routed row.

The PR gives that kernel an `AFFINE=false` arm for mxfp4 (group size 32, one E8M0 scale byte per group, no biases) and lets bf16 mxfp4 reach it under the existing gate and switch. It adds one template instantiation; `qmm.hip` compile time is unchanged. On gfx1151 the 512-token prefill median went from 7.53 to 526.3 tok/s (69.9x) with decode unchanged, and `logit_trace` against the per-row path found 0 of 88 and 0 of 81 decided-position mismatches. The arm is on by default under the rule #2066 set, and is recorded as `LOCAL_FIXES.md` item 31 without an upstreaming mark.

## 1. Problem Statement

### 1.1 Where the prefill time went

gpt-oss-20b has 32 experts and top-4 routing, so a 512-token prefill gives `B = 2048` sorted rows, about 64 rows per expert. Every faster gather path in `GatherQMM::eval_gpu` (expert-batched, wide, tiled, warp-shared) was gated on `mode_ == QuantizationMode::Affine`. mxfp4 therefore fell to the generic dispatch that `LOCAL_FIXES.md` item 10 had fixed for correctness: `gather_qmv_kernel<T, uint8_t, 4, 32, false>`, one thread-block row per batch element. Each MoE call in the prefill reread each expert's weights once per routed row, about 64 times. The `rocprofv3` profile on the #2066 benchmark page put 67.7 s over 72 calls in that kernel.

### 1.2 What the issue asked for

A sorted expert-batched mxfp4 path like the affine one, with only the instantiations needed (the ahead-of-time compile cost of `qmm.hip` is noted in `LOCAL_FIXES.md` item 17), a test against the unsorted path and a dequantized f32 reference at gpt-oss's shape (`K = 2880`, `E = 32`, top 4), guarded before/after prefill numbers, and unchanged decode. The issue's fourth criterion (mark the LOCAL_FIXES entry as an upstreaming candidate for #1813) was superseded by the 2026-10-06 fork policy; see section 6.

## 2. Change Summary

| Area | Change |
|---|---|
| `qmm.hip`, decode helper | New `fp4_e2m1_to_float_scaled` (branch-free nibble decode through fp16 bits) and `kFp4HalfBitsScale = 16384.0f` |
| `qmm.hip`, `gather_qmv_expert_batched_kernel` | `static_assert` widened to allow `!AFFINE && BITS == 4 && GROUP_SIZE == 32`; `if constexpr` arms for the nibble decode and the accumulate step |
| `qmm.hip`, `GatherQMM::eval_gpu` | Gate split into `eb_affine` and `eb_mxfp4` (bf16 activations, group size 32, 4 bits); one new launch of `<hip_bfloat16, uint8_t, 4, 32, false, 16>` with a null bias pointer |
| `tests/rocm_gather_qmm_expert_batched.rs` | `Scheme` type instead of fixed affine constants, optional biases, `MXFP4_SHAPES`, a new mxfp4 test, and a dispatch check at `K >= 2048` |
| `LOCAL_FIXES.md` | New item 31; items 9 and 10 point to it |
| `docs/benchmark_results/` | New `rocm-moe-prefill-mxfp4-gfx1151-2026-10-06.md`; the #2066 page's "Not measured" line links to it |

Two commits: the kernel, dispatch, tests and docs (`826f9d61`), and the review follow-up that adds the dispatch check, SAFETY comments and LOCAL_FIXES wording (`e87ffa7d`). The diff against origin/main is 5 files, 302 insertions and 61 deletions.

## 3. Design

### 3.1 The mxfp4 arm of the expert-batched kernel

The affine kernel's inner loop already had the right shape for mxfp4: each lane loads one packed 32-bit word, applies it to four rows of the expert's run (`TOKENS = 4`) before the next load, and the 16 lanes of a column reduce once per row. At 4 bits a word holds eight values, and since the group size is 32 and lanes step in whole words, all eight nibbles of a word sit in one group. The mxfp4 arm changes two things inside that loop and nothing about the schedule.

**Decode.** An e2m1 nibble is one sign bit, two exponent bits and one mantissa bit. `fp4_e2m1_to_float_scaled` places them into an fp16 bit pattern:

```cpp
const uint16_t h =
    static_cast<uint16_t>(((nibble & 0x8u) << 12) | ((nibble & 0x7u) << 9));
return static_cast<float>(__builtin_bit_cast(_Float16, h));
```

The sign goes to fp16 bit 15, the exponent bits to the two lowest fp16 exponent bits (10 and 11), and the mantissa bit to the top fp16 mantissa bit (9). For an e2m1 exponent `e` of 1 to 3, the fp16 value is `2^(e-15) * (1 + m/2)` against e2m1's `2^(e-1) * (1 + m/2)`; for `e = 0` the fp16 value is the subnormal 0 or 2^-15, against e2m1's 0 or 0.5. Every code point is therefore exactly its e2m1 value times 2^-14, with no branch and no lookup table, and `v_cvt_f32_f16` widens it exactly. The existing `fp4_e2m1_to_float` in the same file switches on the value instead.

**Rescale.** Per row, the arm accumulates `qx = sum(x_j * w_j)` over the word's eight decoded values, then does `acc[t] = fmaf(scale, qx * kFp4HalfBitsScale, acc[t])`. Multiplying by 2^14 is exact in f32, so it undoes the decode's factor without rounding, and `scale` is the group's E8M0 byte converted by the kernel's existing `load_scale_value<ScaleT, GROUP_SIZE, AFFINE>`. The affine arm keeps `fmaf(scale, qx, fmaf(bias_val, xs, acc[t]))`; the mxfp4 arm discards `xs` and `bias_val`, and the launch passes a null bias pointer with `has_bias = false`.

### 3.2 Why only bf16 mxfp4, and the compile cost

`qmm.hip` is compiled ahead of time for each target, and `LOCAL_FIXES.md` item 17 records that every instantiation adds to that cost. gpt-oss runs bf16 activations and is the only mxfp4 MoE checkpoint on the host, so the PR adds exactly one instantiation, `gather_qmv_expert_batched_kernel<hip_bfloat16, uint8_t, 4, 32, false, 16>`. f16 mxfp4 and mxfp8 still take the per-row kernel. Compiled alone with the build's `hipcc` flags and alternated with main's file, `qmm.hip` took 37.9 and 37.8 s against 38.1 and 37.9 s for main: no measurable change.

The `static_assert` encodes the same scope: the kernel compiles for affine at 4 or 8 bits, or for the non-affine case at 4 bits and group size 32, and nothing else.

### 3.3 Dispatch and the default rule from #2066

The gate is unchanged in shape (sorted, transposed, `M == 1`, `B >= 64`, `E > 0`, `E <= 64`, `B / E >= 4`); the scheme condition is now `eb_affine || eb_mxfp4`. A decode step has `B = top_k = 4`, so decode never reaches the kernel, and neither does a prefill shorter than 32 tokens on gpt-oss (`B / E < 4`).

The kernel stays behind `MLX_ROCM_GATHER_QMV_EXPERT_BATCHED`, read on every call and on by default. #2066 set the rule for that default: on only if every eligible dtype matches the unsorted path within that path's own error against a dequantized f32 reference, and 512-token prefill improves on every eligible model measured with no decode change. bf16 is the one eligible mxfp4 dtype and gpt-oss-20b the one eligible mxfp4 checkpoint on the host, both conditions hold (section 4), so mxfp4 is on by default with affine. Setting the variable to `0` turns the kernel off for every scheme; there is no mxfp4-only switch.

## 4. Verification

### 4.1 Kernel tests

`mxfp4_expert_batched_gather_qmm_matches_unsorted_and_reference` forces the kernel on and compares sorted mxfp4 `gather_qmm` with the unsorted path (the per-row kernel) and with per-expert dense f32 matmuls of the dequantized weights. Shapes: gpt-oss-20b's expert layer (32 experts, top 4, `K = N = 2880`) and a narrow 32-expert layer at `K = 512`, `N = 516`, whose last column block is partial. Each shape runs four index layouts (flat `Gathered` and `Shared`, broadcast `BroadcastRows` and `BroadcastIndices`), bf16 activations, 256 rows per call: 8 mxfp4 cases.

Relative L2 error against the reference (expert-batched / unsorted):

| Shape | Gathered | Shared | BroadcastRows | BroadcastIndices |
|---|---|---|---|---|
| gpt-oss-20b experts | 1.654e-3 / 1.654e-3 | 1.654e-3 / 1.654e-3 | 1.657e-3 / 1.657e-3 | 1.657e-3 / 1.657e-3 |
| `K = 512`, `N = 516` | 1.653e-3 / 1.653e-3 | 1.617e-3 / 1.617e-3 | 1.673e-3 / 1.673e-3 | 1.660e-3 / 1.660e-3 |

The two paths agree to four digits in every case, and two sorted runs are bit-identical. Two further checks show the test exercises the new code:

- **Deliberate break.** With the sign bit dropped from the nibble decode, the first case fails at 1.377 while the unsorted path stays at 1.654e-3, so the comparison reaches the new arm.
- **Dispatch check** (added in `e87ffa7d`). The per-row fallback is also correct, so the error comparison alone would pass if the gate stopped sending mxfp4 to the kernel. At `K = 2880` the test also runs the sorted call with the kernel switched off and requires its bytes to differ from the kernel-on output somewhere; the different summation order over a long reduction guarantees some difference when the kernel runs. At `K = 512` both kernels round to identical bf16 outputs (measured), so the narrow shape skips this check.

The 48 affine cases in the same file are unchanged and pass.

### 4.2 Prefill and decode on gfx1151

`mlxcel-bench-decode` at `scripts/bench_decode.sh`'s shape (512-token prompt, 128 generated tokens, 20-token warmup, `--ignore-eos`), one binary, `MLX_ROCM_GATHER_QMV_EXPERT_BATCHED=0` and `=1` alternated run by run, six rounds, every run under `scripts/rocm_gpu_guard.sh` and accepted on its first attempt:

| | Off (per-row) | On (expert-batched) |
|---|---|---|
| Prefill median, tok/s | 7.53 | 526.28 |
| Prefill range, tok/s | 7.41 to 7.92 | 524.78 to 529.38 |
| Decode median, tok/s | 8.45 | 8.49 |
| Decode range, tok/s | 8.34 to 8.65 | 8.40 to 8.64 |

Prefill is 69.9x faster, about 0.97 s instead of 65 to 69 s, and every on run beats every off run by about two orders of magnitude. Decode medians are 0.4% apart with overlapping ranges. After three rounds the off median led decode by 1.3%, which is why three more rounds were run. As a control, a 16-token prompt (`B = 64`, `B / E = 2`, so neither arm reaches the kernel and both run identical code) alternated three times gave 8.46 / 8.50 / 8.47 off and 8.55 / 8.47 / 8.48 on: about 1% spread from the host alone. Decode is unchanged.

A `rocprofv3` trace with the kernel on puts the expert-batched kernel at 12.2 ms per call (144 calls) against about 940 ms for the per-row kernel before.

### 4.3 Logit traces

No gpt-oss trace exists in `benchmarks/logit_traces/`, so the reference is the per-row path on the same binary (`MLX_ROCM_GATHER_QMV_EXPERT_BATCHED=0`) and the candidate is the default, compared with `scripts/compare_logit_traces.py --decided 2.0` over `tests/fixtures/wikitext2_excerpt.txt`:

| Window | Top-1 disagreement | Decided mismatches | Perplexity off / on |
|---|---|---|---|
| `w8` (512-token prefill reaches the kernel) | 29 / 640 | 0 / 88 | 161.96 / 159.95 |
| `w256` (two 256-token chunks, `B = 1024`) | 29 / 512 | 0 / 81 | 76.60 / 76.88 |

Every disagreement sits where the reference was undecided (top-two gap under 2.0), which the script classifies as rounding rather than behavior. gpt-oss has few decided positions on this corpus (40% and 44% of positions have a top-two gap under 0.5). The traces are not committed.

### 4.4 Gates

From the PR: `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt verify-rocm-overlay` and `cargo test --test dead_doc_pointers` pass. The unit's own first full `make verify-rocm` (on `826f9d61` plus the review commit's test change) passed every step through clippy and the smoke test, but `verify-test-rocm` gave 11,962 passed and 1 failed. The failure was the first version of the dispatch check, which also ran at `K = 512`, where both kernels produce identical bf16 bytes and the "must differ" assertion cannot hold. `e87ffa7d` restricts the check to `K >= 2048`; the test file then passed 2/2 with clippy clean.

From the orchestrator, on head `e87ffa7d` (up to date with origin/main `5c851fc4`): `make verify-rocm` passed every step, with 11,963 tests passed, 0 failed, 380 ignored, and the smoke test OK.

Not verified: Metal and CUDA (not available on the host; the change touches no Metal or CUDA file).

## 5. Technical Decisions

- **Extend the expert-batched kernel rather than give mxfp4 an arm of the warp-shared kernel.** The issue allowed either. The expert-batched kernel is the one that removes the per-row reread, and its loop needed only a decode and an accumulate change, so the schedule #2066 tuned is reused as is.
- **Decode through fp16 bits instead of a branch or a table.** The bit layout of e2m1 maps onto fp16 with a constant 2^-14 factor, including the subnormal code point. That keeps the decode to two masks, two shifts and one conversion per nibble, and the compensating 2^14 multiply is exact.
- **Rescale per word, not per value.** The 2^14 factor and the E8M0 scale are both constant across a word (one group), so applying them once to the word's partial dot product costs one extra multiply per word per row.
- **One instantiation.** Adding f16 mxfp4 or mxfp8 would cost compile time for dtypes and schemes no checkpoint on the host uses; they keep the correct per-row path.
- **Same switch, same default rule.** A separate variable for mxfp4 would add a configuration surface without a use; the #2066 rule gives an evidence bar the mxfp4 arm met.
- **Dispatch check by output bytes.** It needs no profiler or kernel counter in the test and fails if a future gate change silently drops mxfp4, which the error comparison alone cannot catch. The cost is a dependency on summation order: it is valid only where the reduction is long enough to change some bf16 result, which is why it is limited to `K >= 2048`.

## 6. Fork Policy

The change is in the ROCm overlay (`patches-rocm/`) applied over the forked MLX ROCm backend. Under the maintainer's 2026-10-06 fork policy, ROCm fork fixes stay in mlxcelverse, so `LOCAL_FIXES.md` item 31 ends "kept in mlxcelverse under the 2026-10-06 fork policy, not proposed there" and is not marked as an upstreaming candidate for #1813. The issue's fourth acceptance criterion is struck through with that explanation. Item 9 still carries its own earlier upstreaming mark; the PR does not change older entries' status.

## 7. Residual Risks and Follow-ups

- **gpt-oss decode is now the bottleneck.** With prefill fixed, the profile shows the per-row mxfp4 `gather_qmv_kernel<hip_bfloat16, unsigned char, 4, 32, false>` at 1.39 ms per call, 504 calls (72 per decode step), about 100 ms of a 115 ms step. That is the unsorted `B = 4` path, outside #2106's scope and the next target for gpt-oss decode on this host.
- **Out-of-range sorted indices leave rows unwritten.** Block z covers expert z's run, found by binary search over the sorted `rhs_indices`. A row whose index is `E` or more belongs to no block, so its output is never stored. This was already true for affine, and routing never produces such an index, but nothing in the kernel or dispatch rejects it.
- **The environment variable is undocumented for users.** `MLX_ROCM_GATHER_QMV_EXPERT_BATCHED` appears only in `LOCAL_FIXES.md`, the benchmark pages and code comments, not in any user-facing environment-variable reference.
- **Coverage limits.** f16 mxfp4 and mxfp8 do not reach the kernel. Wave64 (CDNA) parts are untested; the 16-lane reduction stays inside a wave on both widths. The four-rows-per-load factor was not retuned for mxfp4. No other mxfp4 MoE checkpoint was available, so "every eligible model" in the default rule is one model.
- **Dispatch check depends on reduction length.** If the per-row kernel ever summed in the same order as the expert-batched one, the byte comparison would fail even with correct dispatch; the test comment states the reason for the `K` threshold.

## 8. Learning Points

- **Small float formats can be decoded by bit placement.** When a narrow format's exponent and mantissa fit inside a wider one's low exponent and high mantissa bits, a shift and a constant rescale replace a lookup, and subnormals come out right without special cases.
- **Fold constants to the coarsest level that holds them.** Here the decode factor and the group scale both apply per word, so neither touches the per-value multiply-add.
- **A correct fallback hides a dead fast path.** When the slow path is also correct, accuracy tests pass whether or not dispatch works; a test needs an observable that differs between the two paths, and that observable must be checked where it actually differs (the `K = 512` failure).
- **Alternate arms and add a control.** A three-round decode median that leaned 1.3% one way disappeared at six rounds, and a control where both arms run identical code measured the host's own spread.

Refs: #2106, #2066, #1814, #1813, #1801, #2161.
