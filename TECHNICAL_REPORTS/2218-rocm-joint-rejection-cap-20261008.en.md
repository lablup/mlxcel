# Technical Report: PR #2218 - Recalibrating the Joint Top-k + Top-p Rejection Vocab Cap on ROCm

**Date**: 2026-10-08

**Status**: Implemented and verified on the gfx1151 host (ROCm/HIP); code head `7d737d49` rebased on main `ae343d84`, PR open, pending merge. Closes #2157, part of #1801.

**Languages**: C++ (`mlx_cxx_bridge.cpp`, `mlx_cxx_bridge.h`), Rust (`lib.rs`, `sampling_rejection_tests.rs`, `examples/rejection_sampling_microbench.rs`, `src/bin/bench_decode.rs`), shell (`scripts/bench_decode.sh`), Markdown (benchmark page, `installation.md`, `environment-variables.md`)

**Risk Level**: Low. Only the value of one routing constant changes, and only in ROCm builds: a ROCm build now sends top-k together with top-p to the rejection kernel up to vocab 152064 instead of 32768. `MLXCEL_SAMPLING_REJECTION=0` restores the stock chain. Metal and CUDA builds keep 32768, but that arm was not run on this host.

## Executive Summary

The rejection sampler routes top-k together with top-p only up to `REJECTION_JOINT_VOCAB_MAX`, which was 32768. That number was measured on M1 Ultra, where joint top-k + top-p won 1.27x to 1.64x at 32K and lost (0.71x to 0.83x) at 152K. The HIP port (#2064) inherited it without a measurement. On ROCm, `--top-k 40 --top-p 0.95` on a vocab-128256 model such as Llama 3.1 therefore never reached the kernel.

On gfx1151 the microbenchmark alone would have kept 32768. Its batch 4 and 8 cells win at every vocabulary (pipelined 1.05x to 2.02x), but the synthetic batch-1 cell reads 0.90x to 0.98x pipelined above 32768. The end-to-end decode run, which the issue names as the confirmation at 128256 and above, disagrees with that cell. With the joint case routed, Meta-Llama-3.1-8B-Instruct-4bit (128256) goes from 32.67 to 37.57 tok/s (1.15x) and Qwen2.5-7B-Instruct-4bit (152064) from 41.91 to 46.92 tok/s (1.12x), medians of three interleaved guarded pairs, with every kernel run faster than every chain run. On the final branch build, on the engine decode path, Llama goes from 33.15 to 37.73 tok/s (1.14x).

The ROCm value is now 152064, the largest vocabulary measured both ways, chosen by a build flag with no runtime backend branch. A new bridge accessor reports the active value and a test pins it per build. Why the synthetic batch-1 cell does not predict real decode was not isolated. Vocabularies above 152064 were not measured and are unchanged. Metal and CUDA were not verified. The orchestrator's `make verify-rocm` on the rebased head passes (161 suites, 12,189 passed, 0 failed, 398 ignored).

## 1. Problem Statement

### 1.1 A cap calibrated on another GPU

`REJECTION_JOINT_VOCAB_MAX` in `src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp` is read by `sampling_rejection_routes` once top-p is active: a top-k that can bind routes only if `vocab <= REJECTION_JOINT_VOCAB_MAX`. The reason top-k loses at large vocabularies on M1 Ultra is that it needs more rejection rounds as the vocabulary grows, and each round is a full-row sweep on one threadgroup. gfx1151 has a different per-threadgroup bandwidth and launch cost, so the crossover can sit elsewhere. Nobody had measured it: `rocm-samplers-gfx1151-2026-10-05.md` noted that `--top-k 40 --top-p 0.95` on a vocab-128256 model never reaches the kernel and that whether the cap is right on ROCm was not measured.

### 1.2 The microbenchmark could not run on ROCm

The tool that measures the matrix, `examples/rejection_sampling_microbench.rs`, gated on `custom_kernels_available()`. That function is true on Metal or CUDA by definition, so on a ROCm build the benchmark refused to run. The port that made the kernel run on ROCm never had its crossover measured because the measuring tool was not ported with it. Its vocabulary list also lacked 128256 (Llama 3) and 151936 (Qwen3).

### 1.3 No way to benchmark the joint filter end to end

`mlxcel-bench-decode` and `scripts/bench_decode.sh` had no `--top-k` option, so the case the issue was opened for could not be timed through the normal decode benchmark.

## 2. Change Summary

| Area | Change |
|---|---|
| `mlx_cxx_bridge.cpp` | `REJECTION_JOINT_VOCAB_MAX` is 152064 under `#ifdef MLXCEL_BRIDGE_ROCM_BACKEND` and 32768 otherwise. The gfx1151 tables are in the comment above it. The not-routed message names the build's cap. |
| `mlx_cxx_bridge.h`, `lib.rs` | New accessor `sampling_rejection_joint_vocab_max()`. |
| `sampling_rejection_tests.rs` | The cap is pinned per build; the routing matrix covers 65536, 128256, 151936 and 152064 per build; the large-vocabulary decline test picks a vocabulary above the build's cap. |
| `examples/rejection_sampling_microbench.rs` | Vocabularies 128256 and 151936 added; `--dtype f32\|bf16\|f16` and `--config <label>` options; prints the build's cap; port gate fixed. |
| `src/bin/bench_decode.rs`, `scripts/bench_decode.sh` | New `--top-k` option (filename tag `_k<K>`, validated). |
| Docs | New results page `docs/benchmark_results/rocm-rejection-joint-cap-gfx1151-2026-10-07.md`, linked from `rocm-samplers-gfx1151-2026-10-05.md`. The ROCm sampled-decode row in `installation.md` lists which filter combinations route at the Llama 3 and Qwen vocabularies. The `MLXCEL_SAMPLING_REJECTION` row in `environment-variables.md` is updated. |
| `benchmarks/` | Raw CSVs: three f32 microbenchmark runs, three bf16 runs, the end-to-end pairs of the measurement build, and the before/after pairs of the final build. |

## 3. Design

### 3.1 The per-build constant

```cpp
#ifdef MLXCEL_BRIDGE_ROCM_BACKEND
constexpr int32_t REJECTION_JOINT_VOCAB_MAX = 152064;
#else
constexpr int32_t REJECTION_JOINT_VOCAB_MAX = 32768;
#endif

int32_t sampling_rejection_joint_vocab_max() {
    return REJECTION_JOINT_VOCAB_MAX;
}
```

This is the build-flag pattern of the fused-MoE SGY default (#2098) and the `cfg!(feature = "rocm")` defaults of #2189. A ROCm build has no Metal or CUDA backend, so no runtime backend comparison is needed and `verify-kernel-port-dispatch` stays satisfied. The constant has one reader, `sampling_rejection_routes`, and the accessor exposes the same value, so a unified per-row sampling step (epic #2166 Phase 2) should call `sampling_rejection_routes` or the accessor instead of copying the number.

### 3.2 The test that pins it

`the_joint_vocab_cap_is_the_measured_crossover` in `sampling_rejection_tests.rs` expects 152064 when built with `feature = "rocm"` and 32768 otherwise, so the constant cannot move without a new measurement. It then checks that the policy turns exactly at the cap (routes at `cap`, declines at `cap + 1`, with and without min-p), that top-p alone still routes above the cap, and that a top-k that cannot bind (`top_k >= vocab`) does not bring the cap in.

### 3.3 The microbenchmark port gate

The gate now checks `sampling_rejection_backend_supported()`, the predicate production routing uses, instead of `custom_kernels_available()`. The harness measures what `fused_sample` does, so it must agree with `fused_sample` on which backends it applies to.

### 3.4 Rejected

- **Keep 32768 on ROCm.** It follows the issue's literal rule (section 7) but leaves the configuration the issue was opened for 12% to 15% slower.
- **A runtime backend comparison for the cap.** The build flag is enough and keeps the kernel-port-dispatch check passing.
- **A cap above 152064.** Larger vocabularies (gpt-oss 201088, Gemma 262144) were not measured, and the existing comment requires that unmeasured vocabularies never be included.

## 4. Risks Specific to the Change

- On ROCm, decode for top-k + top-p at vocab 65536 up to 152064 changes path. The rejection kernel is distribution-equivalent to the stock chain but uses a different random stream, so sampled tokens at a fixed seed differ from before.
- The synthetic batch-1 cell says the kernel is slower than the chain on a single row at 128256 and up. Real decode says the opposite, and the reason is not known. A different model, batch size or GPU could land on the other side of that cell.
- The cap is a single number per build, not per GPU. Another ROCm part (for example a discrete GPU) may have a different crossover.

## 5. Verification

### 5.1 Gates

- Orchestrator gate, `make verify-rocm` on the rebased head `7d737d49`: 161 suites, 12,189 passed, 0 failed, 398 ignored.
- From the PR body, on the earlier rebase onto `ad844354`: `sampling_rejection_tests` and `sampling_fixed_key_tests` (34 passed, covering support, distribution and fixed-key tests), the `sampling_gumbel_kill_switch` and `sampling_rejection_kill_switch` tests, and `cargo test --test dead_doc_pointers` pass.
- Dispatch logs on the final build: `mlxcel generate` logs `rejection kernel (batch 1, vocab 128256, ... top_k 40, top_p 0.95 ...)` by default and `argpartition chain: pinned by MLXCEL_SAMPLING_REJECTION` with the switch set, so each arm of the before/after pairs ran the path it is labelled with.
- Every benchmark ran through `scripts/rocm_gpu_guard.sh`.

Not verified: Metal and CUDA are not available on this host. Their value (32768) is unchanged and the routing code is shared, but the non-ROCm arm of the per-build tests was not run here.

## 6. Results

### 6.1 Environment

AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`), 96 GiB VRAM carve-out, ROCm 10.0.0 (HIP 7.15.26333), MLX pin `81ba1c6a`, ROCm overlay `75915908`, `cargo build --release --features rocm`. The measurement build is `main` at `84d3a7bc` plus this change with the ROCm cap raised to `INT32_MAX`, so `fused_sample` routes every joint cell. Its CSVs record `mlxcel_commit` as `unknown` because the binaries were copied out before the source moved on. The final-build CSVs record `79ea5814`. Checkpoints are `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (128256) and `mlx-community/Qwen2.5-7B-Instruct-4bit` (152064).

### 6.2 Microbenchmark grid

`rejection_sampling_microbench`, 200 timed and 30 warmup iterations per cell, three runs, batch 1/4/8, vocabularies 32768, 65536, 128256, 151936 and 152064, top-k 40 + top-p 0.9, f32 logits (every configuration) and bf16 logits (joint only, because both checkpoints are bf16). Every joint cell reads `routed=yes`. The pipelined arm goes through `fused_sample`, so on a cell production does not route it would time the chain against itself, which is why the measurement build lifted the cap. `pipe_x` is the pipelined speedup (stock chain time over kernel time), the number the issue says to read, not the isolated `iso_x`. Medians of three, f32:

| vocab | batch 1 pipe_x | batch 4 pipe_x | batch 8 pipe_x | batch 1 iso_x |
|---|---|---|---|---|
| 32768 | 1.00 (0.96-1.04) | 1.17 | 1.44 | 2.23 |
| 65536 | 0.98 (0.95-1.05) | 1.18 | 1.64 | 1.59 |
| 128256 | 0.98 (0.96-0.99) | 1.28 | 1.80 | 0.96 |
| 151936 | 0.90 (0.88-0.92) | 1.36 | 1.90 | 0.92 |
| 152064 | 0.93 (0.88-0.97) | 1.29 | 2.02 | 0.92 |

Batches 4 and 8 win at every vocabulary. Batch 1 is the only losing cell, and from 128256 up the isolated cell is also below 1.0 (0.92 to 0.96). The bf16 rerun reads the same way: batch 1 pipe_x is 0.98 at 128256, 0.89 at 151936 and 0.86 at 152064, batch 4 is 1.05 to 1.23 and batch 8 is 1.38 to 1.54. The dtype is therefore not the cause of the disagreement with real decode. The full tables with `rounds` and times are in the benchmark page.

How much the batch-1 pipelined cells can say is limited. Rows that production does not route (top-k alone and min-p alone, chain against chain) scatter between 0.90x and 1.11x by median, so a batch-1 `pipe_x` between 0.95 and 1.05 is inside the noise either way. The synthetic forward measured 2110 to 2659 us per step, and the cap-overflow counter stayed at 0.

### 6.3 End-to-end decode

`scripts/bench_decode.sh` at pp512/tg128 with `--temperature 0.8 --top-k 40 --top-p 0.95`, kernel arm against the same binaries with `MLXCEL_SAMPLING_REJECTION=0`. Each pair ran inside one guard attempt, alternating which arm went first. Medians of three pairs:

| Model (vocab) | Stock chain tok/s | Rejection kernel tok/s | Ratio |
|---|---|---|---|
| Meta-Llama-3.1-8B-Instruct-4bit (128256), measurement build | 33.94 / 31.64 / 32.67, median 32.67 | 37.70 / 37.37 / 37.57, median 37.57 | 1.15x |
| Qwen2.5-7B-Instruct-4bit (152064), measurement build | 42.68 / 41.91 / 40.79, median 41.91 | 46.92 / 46.40 / 47.18, median 46.92 | 1.12x |
| Meta-Llama-3.1-8B-Instruct-4bit (128256), final build, engine path | 33.15 / 35.76 / 33.09, median 33.15 | 37.61 / 38.45 / 37.73, median 37.73 | 1.14x |

Every kernel run is faster than every chain run on every model. Within each pair the gain is 11% to 18% on Llama (measurement build), 10% to 16% on Qwen and 8% to 14% on the final Llama build. The chain arm spreads more than the kernel arm, as the top-p chain did on Llama 3.1 in #2064. The final build is the branch rebased on main, where `mlxcel-bench-decode` decodes through the engine (#2224, #2225; the CSV's `decode_path` column reads `engine`). Its "before" arm is `MLXCEL_SAMPLING_REJECTION=0`, which is the path main takes for this configuration. Qwen2.5-7B was not re-run on the final build because the shared GPU was contended for hours at a time.

## 7. The Decision Rule and Why It Was Not Applied Literally

The issue's rule takes the largest measured vocabulary V such that every batch cell at V and at every measured vocabulary below V has `pipe_x >= 1.0`, confirmed end to end when V is 128256 or more. Read literally, the synthetic batch-1 cells stop it at 32768: the batch-1 cell reads 0.90x to 0.98x pipelined above 32768 (and 0.98x already at 65536), while 32768 itself sits at 1.00x. The rule would keep the old cap.

The end-to-end run is the confirmation the same rule asks for, and it contradicts the cell by a wide margin: 1.15x on Llama 3.1 and 1.12x on Qwen 2.5, with no overlap between arms. The microbenchmark's own batch 4 and 8 cells also win everywhere. Keeping the configuration the issue was opened for on the slower path because of a synthetic cell would be the wrong call, so the ROCm value is 152064, the largest vocabulary measured both ways. The deviation is stated in the PR body and the benchmark page.

## 8. Technical Decisions

- **Decide on decode, not on the synthetic cell.** Real decode is what users run. The synthetic batch-1 cell is inside the noise at several vocabularies and disagrees with the measured end-to-end result.
- **152064, not a round number.** It is the largest vocabulary measured both ways, so the cap never covers an unmeasured size.
- **Build flag, not runtime branch.** It matches the existing per-build defaults and passes the kernel-port-dispatch check.
- **Fix the tool with the port.** The microbenchmark gate now follows the production predicate, so a future backend port does not hide behind the same bug.
- **Add `--top-k` to the decode benchmark.** The joint filter can now be benchmarked end to end without a custom harness.

## 9. Residual Risks

- **Cause not isolated.** Why the synthetic batch-1 cell does not predict real decode on this GPU is unknown. The bf16 rerun ruled out the logits dtype only.
- **Vocabularies above 152064 are unmeasured and unchanged.** gpt-oss (201088) and Gemma 3 (262144) stay on the stock chain.
- **Metal and CUDA are unchanged and not verified here.** The non-ROCm arm of the per-build tests was not run.
- **Qwen2.5-7B was measured on the measurement build only.** The final build was checked on Llama 3.1.
- **One GPU.** The cap is calibrated on gfx1151 only.

## 10. Learning Points

- **A constant measured on one GPU is not a property of the algorithm.** A port that inherits a calibrated cap needs its own measurement, and the harness has to be able to run on the port.
- **A measuring tool with its own backend gate can silently exclude a backend.** The gate should reuse the production predicate.
- **A synthetic cell is a proxy.** When the proxy and the end-to-end run disagree, the end-to-end run decides, and the disagreement is recorded as unexplained.
- **Interleaved guarded pairs make small differences readable.** The chain arm's wider spread would have hidden a 5% difference in sequential runs.

## 11. Follow-up

The stock chain's `top_k_filter` calls `argpartition` without checking `top_k` against the vocabulary size. This is pre-existing and not changed here; it affects the chain arm when `top_k` is at or above the vocabulary.
