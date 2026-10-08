# Technical Report: PR #2235 - An mxfp4 Arm for the Warp-Shared gather_qmv Kernel

**Date**: 2026-10-08

**Status**: Implemented and verified on the gfx1151 host (ROCm/HIP); code head `16be6bf9` on main `6cd0139d`, PR open, pending merge. Closes #2178.

**Languages**: C++/HIP (ROCm overlay `qmm.hip`), Rust (new test in `tests/rocm_mxfp4_quant.rs`, edited `tests/rocm_gather_qmm_expert_batched.rs`), Markdown (`patches-rocm/LOCAL_FIXES.md`, benchmark page)

**Risk Level**: Low to medium. The new path is on by default for bf16 mxfp4 `gather_qmm` at group size 32 with `K % 32 == 0`, so every gpt-oss decode step runs it. It sums in a different order than the per-row kernel, so a few bf16 outputs differ in the last bit. `MLX_ROCM_GATHER_QMV_USE_WARP=0` restores the previous kernel. f16 and f32 mxfp4, mxfp8 and affine dispatch are unchanged. Metal and CUDA code paths are untouched.

## Executive Summary

After #2164 made gpt-oss-20b prefill fast, decode stayed at about 8.5 tok/s. A decode step routes one token to `top_k = 4` experts, so `B = 4`, far below the expert-batched kernel's `B >= 64` gate. Every other fast arm in `GatherQMM::eval_gpu` was affine-only, so bf16 mxfp4 fell to the per-row `gather_qmv_kernel`: one thread per output column walking all of `K`, 1.39 ms per call, 72 calls per step, about 100 ms of a 115 ms step. The #2164 report named this kernel as the next target.

The fix gives `gather_qmv_warp_shared_kernel` an mxfp4 path. Each lane takes one packed 32-bit word (eight e2m1 nibbles, all in one group of 32) per step across the whole `shared_x` chunk, reads that group's E8M0 scale, decodes without a branch and scales the word's dot product back once. The old per-group loop gave work to lanes 0 to 3 of 16 at group size 32. A new dispatch arm sends bf16 mxfp4 there by default.

On gfx1151, `rocprofv3` shows 504 calls of the new instantiation at 108.3 us each and no `gather_qmv_kernel`. Over six alternated, guarded rounds decode goes from a median 8.32 to 63.62 tok/s (7.65x, ranges disjoint) and 512-token prefill from 511.73 to 510.67 tok/s (inside the host spread). A teacher-forced `logit_trace` comparison against the per-row kernel has 0 of 89 decided mismatches. The unit's `make verify-rocm` passes (161 suites, 12,188 passed, 0 failed, 398 ignored). Metal, CUDA and wave64 were not verified.

## 1. Problem Statement

### 1.1 Why decode never reached the expert-batched kernel

The kernel from #2164 handles a sorted, transposed `gather_qmm` with `M == 1`, `B >= 64`, `E <= 64` and `B / E >= 4`, which is a prefill of enough tokens. A decode step has one token and `top_k = 4`, so `B = 4`, and a prefill under 32 tokens misses it too. The affine fast arms (wide and warp-shared) are affine-only. A bf16 mxfp4 call therefore went to the generic non-affine route (LOCAL_FIXES item 10), `gather_qmv_kernel<T, uint8_t, 4, 32, false>`.

### 1.2 What that cost

The per-row kernel assigns one thread to each output column and loops over all of `K` alone. On gfx1151 a guarded `rocprofv3` trace (the #2106 and #2164 profiles) put it at 1.39 ms per call, 72 calls per decode step (24 layers, three gather projections), 504 calls over seven steps. That is about 100 ms of a 115 ms step, so decode ran at about 8.5 tok/s.

### 1.3 Why the existing warp-shared kernel did not help mxfp4

`gather_qmv_warp_shared_kernel` stages a chunk of `x` in shared memory and has `THREADS_PER_COL` lanes (16) cooperate on one output column. Its inner loop walks groups, and inside a group each lane starts at `lane * 8`. At group size 32 that leaves lanes 0 to 3 with work and lanes 4 to 15 idle, 12 of 16. Its decode also went through the branchy `fp4_e2m1_to_float` switch. The kernel was only instantiated for affine.

## 2. Change Summary

| Area | Change |
|---|---|
| `qmm.hip`, `gather_qmv_warp_shared_kernel` | New `kMxfp4WordPath` (`!AFFINE && BITS == 4 && GROUP_SIZE == 32`): a word walk over the whole chunk. The per-group loop is skipped for that case and kept for the others. |
| `qmm.hip`, `GatherQMM::eval_gpu` | New arm after the affine warp-shared arm for bf16 mxfp4, `group_size_ == 32`, `bits_ == 4`, `K % 32 == 0`, thread count 16 or `WARP_SIZE`. Two new instantiations. Comments on the generic non-affine route updated. |
| `tests/rocm_mxfp4_quant.rs` | New `mxfp4_warp_shared_gather_qmv_matches_per_row_and_reference` (section 5.1). |
| `tests/rocm_gather_qmm_expert_batched.rs` | Per-row dispatch check now sets `MLX_ROCM_GATHER_QMV_USE_WARP=0` (section 5.2). |
| `patches-rocm/LOCAL_FIXES.md` | Item 41 (new), item 10 (updated). The issue called the new entry item 32, which was taken; 40 went to #2220. |
| `docs/benchmark_results/rocm-moe-decode-mxfp4-gfx1151-2026-10-08.md` | New results page. |

One commit, 5 files, 428 insertions, 6 deletions.

## 3. Design

### 3.1 The word path

For mxfp4 each lane handles one packed `uint32_t` (eight 4-bit values) per step:

```cpp
for (int k = chunk_start + lane * 8; k < chunk_end; k += THREADS_PER_COL * 8) {
  const uint32_t packed = *reinterpret_cast<const uint32_t*>(&w_row[k / 2]);
  const float scale = load_scale_value<ScaleT, GROUP_SIZE, false>(scales_row[k / GROUP_SIZE]);
  const float* xs = &shared_x[k - chunk_start];
  float qx = 0.0f;
  for (int j = 0; j < 8; ++j)
    qx = fmaf(xs[j], fp4_e2m1_to_float_scaled((packed >> (4 * j)) & 0xFu), qx);
  acc = fmaf(scale, qx * kFp4HalfBitsScale, acc);
}
```

- **All lanes busy.** Consecutive lanes take consecutive words, so 16 lanes cover 128 values per step instead of 4 lanes covering 32 per group.
- **One scale per word.** Eight divides 32, so a word never straddles a group. The scale is loaded once per word and applied to the word's dot product.
- **Branchless decode.** `fp4_e2m1_to_float_scaled` returns the value times 2^-14 by bit placement, with no switch. The word's sum is multiplied back by 2^14 (`kFp4HalfBitsScale`) once. This is the decode the expert-batched kernel uses (LOCAL_FIXES item 31).
- **No chunk straddling.** The dispatch sends only `K % 32 == 0`, and `CHUNK_SIZE` is a multiple of 32, so no word crosses a chunk end.

The summation order (per-lane partial sums over words, then the lane reduction) matches the expert-batched kernel and differs from the per-row kernel's single serial walk.

### 3.2 Dispatch arm and its guard

The arm sits after the affine warp-shared arm and before the generic non-affine route. It takes bf16 mxfp4 with `group_size_ == 32`, `bits_ == 4` and `K % 32 == 0`, unsorted and sorted calls alike (`use_sorted_rhs_schedule` is passed as the implicit-lhs flag, as the affine arm does). It also requires `fast_threads_per_col` to be 16 or `WARP_SIZE`.

That extra guard exists because `MLX_ROCM_GATHER_QMV_THREADS_PER_COL` can set any thread count, while only two instantiations exist and the kernel's lane reduction assumes the block's x dimension equals `THREADS_PER_COL`. A value that matches neither stays on the per-row kernel instead of failing to find a kernel or reducing wrongly. Only bf16 is instantiated (gpt-oss runs bf16 activations) to hold the file's compile time, so f16 and f32 mxfp4 and mxfp8 keep the per-row kernel.

### 3.3 The A/B switch

`MLX_ROCM_GATHER_QMV_USE_WARP=0`, read on every call, skips the arm and sends the call to the per-row kernel. It is the "off" arm of the benchmark and the reference arm of the tests, and a way out if a model shows a regression.

### 3.4 Rejected

- **A new kernel.** The warp-shared kernel already stages `x` and reduces across lanes; only the inner loop needed a different shape for mxfp4.
- **Instantiating f16 and f32.** Not needed for gpt-oss and it grows compile time.
- **Opt-in default.** The issue's rule makes it default if decode improves by more than the host spread and prefill stays within it. Both held (section 6.3).

## 4. Risks Specific to the Change

- The new path changes the summation order for decode, so outputs can differ from the per-row kernel in the last bf16 bit. Section 5.1 and 6.4 bound this.
- Every bf16 mxfp4 gather call with `K % 32 == 0` that misses the expert-batched gate now takes it, including prefills under 32 tokens and sorted calls with `B < 64`.

## 5. Verification

### 5.1 The new test

`mxfp4_warp_shared_gather_qmv_matches_per_row_and_reference` in `tests/rocm_mxfp4_quant.rs` runs gpt-oss-20b's expert layer (32 experts, top 4, `K = 2880`, two `shared_x` chunks) at `N = 2880` and `N = 516`, bf16. Cases: unsorted with 1 token (`B = 4`) and 8 tokens (`B = 32`), and sorted with 16 tokens (`B = 64`, which misses the expert-batched gate). It reports the relative L2 error of the default and the per-row paths against per-expert dense f32 matmuls of the CPU-dequantized weights.

| Case | `N = 2880`, default / per-row | Outputs differing | `N = 516`, default / per-row | Outputs differing |
|---|---|---|---|---|
| unsorted, 1 token | 1.6358e-3 / 1.6358e-3 | 0 of 11520 | 1.6588e-3 / 1.6588e-3 | 0 of 2064 |
| unsorted, 8 tokens | 1.6609e-3 / 1.6609e-3 | 5 of 92160 | 1.6827e-3 / 1.6827e-3 | 1 of 16512 |
| sorted, 16 tokens | 1.6569e-3 / 1.6569e-3 | 7 of 184320 | 1.6606e-3 / 1.6606e-3 | 0 of 33024 |

The two paths agree to five digits, and two default runs are bit-identical.

**Deviation from the issue.** The issue asked that, at `N = K = 2880`, the default and per-row outputs differ in at least one byte. That holds over the three `N = 2880` cases together (288,000 outputs) but not for the 1-token case alone, whose sums round to identical bf16 outputs: over 2880 f32 terms the two summation orders rarely straddle a bf16 rounding boundary. The test therefore accumulates the differing-output count over the `N = 2880` cases and asserts it is above zero, which still proves the dispatch left the per-row kernel. The count is 12 of 288,000 (0 in the 1-token case, 5 in the 8-token case, 7 in the sorted 16-token case), as the test printed it on the measured run; the benchmark page's table, the PR body and LOCAL_FIXES item 41 carry the same figures.

**Mutation check.** With the sign bit dropped from the new decode the first case fails at a relative L2 error of 1.211, against 1.636e-3 for the per-row path.

### 5.2 The expert-batched test change

`tests/rocm_gather_qmm_expert_batched.rs` proves its gate reaches the expert-batched kernel by switching it off and requiring the sorted call's bytes to differ from the kernel-on bytes; the switched-off call used to run the per-row kernel, which sums in a different order. With this PR a call that misses the gate takes the warp-shared word path, which sums in the same order as the expert-batched kernel, so the bytes would match and the check would fail with correct dispatch. The test now also sets `MLX_ROCM_GATHER_QMV_USE_WARP=0` (through a new `force_gather_warp_off` helper) around that call so the per-row kernel is selected explicitly, and updates its comments. This is the "dispatch check depends on reduction length" risk the #2164 report listed.

### 5.3 Kernel trace

`rocprofv3 --kernel-trace --stats` of `mlxcel-bench-decode` with the default dispatch (`--prompt-tokens 512 -n 8 --warmup-tokens 1`), gpt-oss-20b-MXFP4-Q4:

| Kernel | Calls | Average per call |
|---|---|---|
| `gather_qmv_expert_batched_kernel<hip_bfloat16, unsigned char, 4, 32, false, 16>` (prefill) | 144 | 12.29 ms |
| `gather_qmv_warp_shared_kernel<hip_bfloat16, unsigned char, 4, 32, false, 16>` (decode) | 504 | 108.3 us |
| `gather_qmv_kernel` (any instantiation) | 0 | |

The decode calls are the same 504 the #2106 trace attributed to the per-row kernel at 1.39 ms: 12.8x per call.

## 6. Results

### 6.1 Environment

Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`), HIP 7.15.26333, rocprofv3 1.3.5. Every GPU run went through `scripts/rocm_gpu_guard.sh`; all 15 guarded attempts were accepted on the first try. The benchmark page records the baseline binary as `main` at `ad844354` plus this change; the PR base is `6cd0139d`.

### 6.2 Decode and prefill

One binary, 512-token prompt, 128 generated tokens, 20-token warmup, `--ignore-eos`, `MLX_ROCM_GATHER_QMV_USE_WARP=0` (off) and the default (on) alternated run by run, six rounds:

| Metric | Off (median) | On (median) | Ratio |
|---|---|---|---|
| Decode tok/s | 8.32 | 63.62 | 7.65x |
| Prefill tok/s | 511.73 | 510.67 | 0.998x |

Decode: every on run (63.42 to 63.79) is faster than every off run (8.23 to 8.39); a 128-token decode takes 2.0 s instead of 15.3 s. Prefill: ranges overlap (509.9 to 512.5 off, 509.2 to 512.2 on), inside the about 1% host spread #2106 measured with identical code on both arms. The 512-token prefill reaches the expert-batched kernel on both arms.

### 6.3 Decision

Decode improved 7.65x and prefill moved 0.2%, so the arm is on by default, with no opt-in.

### 6.4 Logit trace

`examples/logit_trace` over `tests/fixtures/wikitext2_excerpt.txt` (`w8`: each 8-token chunk runs `B = 32` unsorted, which takes the new path), per-row against default on the same binary, `scripts/compare_logit_traces.py --decided 2.0`: top-1 disagreement 23 of 640, decided mismatches 0 of 89, largest gap at a disagreement 0.250, perplexity 159.95 per-row and 158.99 default. Every disagreement sits where the reference's top-two gap is under 0.5; the verdict is the rounding class, not a behaviour change. The traces are not committed.

### 6.5 Gates

- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt verify-rocm-overlay`, `cargo test --test dead_doc_pointers`, and clippy `-D warnings` on both test targets: pass.
- `rocm_mxfp4_quant` (5 tests) and `rocm_gather_qmm_expert_batched` (2 tests): pass.
- Unit's `make verify-rocm` (with `MLXCEL_ROCM_SMOKE_MODEL=models/mlx/Qwen3-0.6B-4bit`): OK, 161 suites, 12,188 passed, 0 failed, 398 ignored.
- Orchestrator gate (`make verify-rocm` on `16be6bf9`): passed, 12,188 passed, 0 failed, 398 ignored.

Not verified: Metal and CUDA (not available on this host; the change touches only the ROCm overlay and ROCm-gated tests). Wave64 (CDNA) is untested: the 16-lane instantiation reduces inside a wave on both widths, and the `WARP_SIZE` instantiation is selected only for `K >= 16384` with one routing entry or through `MLX_ROCM_GATHER_QMV_THREADS_PER_COL`.

## 7. LOCAL_FIXES Items 41 and 10

Item 41, "mxfp4 in the warp-shared gather qmv", records the fall-through to the per-row kernel and its cost, the word path, the dispatch condition and switch, the two instantiations, the test and mutation result, and the trace and benchmark numbers. Item 10 now says that bf16 mxfp4 at group size 32 reaches the warp-shared kernel since item 41, and that f16 and f32 mxfp4, mxfp8 and `MLX_ROCM_GATHER_QMV_USE_WARP=0` still reach the per-row kernel. Item 41 ends with the fork-policy wording: "Applies to the fork; kept in mlxcelverse under the 2026-10-06 fork policy, not proposed there."

## 8. Technical Decisions

- **Reshape the loop, not the kernel.** The staging and reduction were fine; the loop shape left 75% of lanes idle at group size 32.
- **Word granularity.** A word is the largest unit that stays inside one group, so one scale load and one rescale serve eight values.
- **Reuse the item 31 decode.** The same branchless decode and constant already pass the expert-batched tests.
- **Guard on thread count.** Unsupported `MLX_ROCM_GATHER_QMV_THREADS_PER_COL` values keep a working path.
- **Aggregate byte check.** The test asserts what the data supports (a difference over the full-width cases) and relies on the L2 errors and the mutation for correctness.

## 9. Residual Risks

- **Coverage limits.** f16 and f32 mxfp4 and mxfp8 do not reach the path. No other mxfp4 MoE checkpoint is on this host, so "default" rests on one model.
- **Wave64 untested.**
- **Last-bit differences.** Different summation order means decode outputs are not byte-identical to the previous kernel; the bounds are the L2 errors and 0 of 89 decided mismatches.
- **The dispatch check in the expert-batched test is again environment-dependent.** It relies on the switch to pick the per-row kernel.
- **Baseline binary.** The benchmark page names `ad844354` as the base of the measured binary, not the PR base `6cd0139d`.

## 10. Learning Points

- **A fast path that covers one quantization mode leaves the others on the slowest kernel.** Decode in a new mode went through a path nobody had profiled until prefill was fixed.
- **Count idle lanes against the group size.** A loop shaped for group size 64 or 128 wastes most of a wave at 32.
- **A byte-difference check needs data that can differ.** Sums that round to the same bf16 values everywhere make "must differ" fail on correct dispatch, so assert it over enough outputs.
- **Changing a kernel's summation order can invalidate another test's observable.** The expert-batched test needed an explicit switch.

## 11. Follow-up

The dense mxfp4 `qmv_warp_shared_kernel` has the same per-group lane pattern, with lanes 4 to 15 idle at group size 32, and was not changed here.
