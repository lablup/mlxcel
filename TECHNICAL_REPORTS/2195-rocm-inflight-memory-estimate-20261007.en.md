# Technical Report: PR #2195 - Reserve the In-Flight Budget in the ROCm Pre-Load Memory Estimate

**Date**: 2026-10-07

**Status**: Implemented and verified on the gfx1151 host (ROCm/HIP); head `9374ae72` on origin/main `87538835`, PR open, pending merge. Closes #2155 (part of #1801).

**Languages**: Rust (`src/execution/memory_estimate.rs`, new `src/execution/memory_estimate_inflight_tests.rs`, `src/commands/generate.rs`), Markdown (README, `docs/environment-variables.md`, `docs/installation.md`, `docs/benchmark_results/rocm-memory-gfx1151-2026-09-30.md`), text (raw peak data file)

**Risk Level**: Low. The change only raises a reserve in the estimate, and only on ROCm builds; it does not touch what runs on the GPU. Metal and CUDA builds compile a stub that returns 0, so their totals are unchanged. The visible effects on ROCm are a 1 GiB larger estimate at the default budget, a correspondingly smaller paged KV `auto` budget, and a lower prompt-cache capacity ceiling that did not change any chosen capacity on this host.

## Executive Summary

After PR #2084 bounded in-flight command batches with `MLX_ROCM_MAX_INFLIGHT_MB` (default 1024), the ROCm results page recorded that the pre-load estimate for Meta-Llama-3.1-8B-Instruct-4bit at 640 tokens (5.58) was below the measured peak (6.14 GB). Nothing in the estimator modeled the transients that committed but unfinished batches may hold, mostly the f16 weight copies of the dequantize-and-GEMM qmm path during prefill.

Issue #2155 argued that most of that gap was a units mix-up, leaving only about 0.15 GB. That claim was wrong. The unit measured it: the 5.58 is `total_bytes / 1e9`, `mlxcel inspect` prints the same estimate as 5.20 GiB, and the byte count is 5,579,686,809. Both figures were already in decimal GB, so the 0.56 GB gap was real.

The fix adds the in-flight budget as its own additive term on ROCm builds. `rocm_inflight_reserve_bytes` parses `MLX_ROCM_MAX_INFLIGHT_MB` exactly as the overlay's `max_inflight_bytes()` does, and the term is selected at compile time with `#[cfg(feature = "rocm")]`. It enters `total_bytes` and `runtime_headroom_bytes`, appears as `Backend in-flight` in `format_estimate` and as `backend_inflight_bytes` in `inspect --json` and the generate log, and is subtracted in `auto_kv_budget_bytes`.

Across 18 guarded measurements (three models, two contexts, three budgets), the estimate before the change was below the peak in 9 rows; now it covers all 18, the smallest margin being 221,877,772 bytes. A test pins the table and fails if the term is removed. The unit's `make verify-rocm` at `9374ae72` passed with 12,004 passed, 0 failed and 384 ignored; the orchestrator re-ran it on the same head before merge.

## 1. Problem Statement

### 1.1 The gap the results page left open

`estimate_total_memory` sums weights, architecture-aware KV, `(DEFAULT_HEADROOM_FACTOR - 1) * (weights + kv)` with the factor at 1.20 (calibrated on Apple Silicon only), and a workload activation term. On ROCm, the backend lets committed but unfinished batches hold up to `MLX_ROCM_MAX_INFLIGHT_MB` MiB of newly allocated memory on top of the steady state (`device.cpp` in the ROCm overlay, `LOCAL_FIXES.md` item 28). The estimate had no term for it.

`docs/benchmark_results/rocm-memory-gfx1151-2026-09-30.md` recorded the consequence at the default budget: an estimate of 5.58 for the 8B model against a 6.14 GB peak, and 20.72 against an 18.58 GB peak for Qwen3-30B-A3B-4bit. It left recalibrating the 1.20 factor for ROCm out of scope. A preflight that under-reserves can pass a load that then exceeds the budget.

### 1.2 Correction to the issue: the units

The issue's "Problem / Background" section stated that `bench_decode` prints the peak as `bytes / 1e9` while `format_bytes` prints the estimate in GiB, so that 5.58 read as GiB is 5.99e9 bytes and the estimate is only about 0.15 GB under the peak.

That reading is wrong, and the PR states so. The 5.58 on the results page is `total_bytes / 1e9`, not a `format_bytes` value. `mlxcel inspect` prints the same estimate as 5.20 GiB, and in bytes it is 5,579,686,809. Against the 6.14 GB peak (6,140,000,000 bytes at its printed precision) the estimate was short by about 560 million bytes, the 0.56 GB the page reported. The new results section and every comparison in the PR use bytes for this reason.

The correction matters for the design. A 0.15 GB gap could have been closed by a small factor nudge; a 0.56 GB gap at the default budget that grows with the budget (the sweep on the same page went from 5.28 to 9.36 GB of peak for the 8B between 256 and 4096 MiB) points at the in-flight budget, not at the factor.

### 1.3 Why not a larger factor

The issue rejected raising the factor on ROCm, and the PR keeps that. The factor scales with weights, so it would over-reserve for large models: the 30B MoE was already above its peak because its 17.17 GB of weights make the 1.20 factor alone worth 3.4 GB. The measured excess tracks the in-flight budget, not model size, so the reserve is an additive term equal to the budget.

## 2. Change Summary

| Area | Change |
|---|---|
| `src/execution/memory_estimate.rs` | New `ROCM_MAX_INFLIGHT_ENV`, `ROCM_DEFAULT_MAX_INFLIGHT_MB = 1024`, `pub fn rocm_inflight_reserve_bytes(raw: Option<&str>) -> u64` and its strtoull-mirroring helper; `backend_inflight_reserve_bytes()` with a `#[cfg(feature = "rocm")]` body and a `#[cfg(not(feature = "rocm"))]` stub returning 0; new `backend_inflight_bytes` field on `MemoryEstimate` and `InspectReport`; the term added to `runtime_headroom_bytes` (and so `total_bytes`); `auto_kv_budget_bytes` subtracts it; `format_estimate` keeps the allocator line factor-only and prints `Backend in-flight` when non-zero |
| `src/execution/memory_estimate_inflight_tests.rs` (new) | Parser tests, an overlay-source drift test, the `auto` budget inversion test, compile-time gate tests for both build kinds, a saturation test, and the `measured` module that pins the 18-row gfx1151 table |
| `src/commands/generate.rs` | `log_estimate_vs_actual_delta` logs `backend_inflight_bytes` |
| Docs | README `inspect --json` field list; `docs/environment-variables.md` (`MLXCEL_HEADROOM_FACTOR` row, the `inspect --json` paragraph, and the `MLX_ROCM_MAX_INFLIGHT_MB` paragraph); `docs/installation.md` ROCm memory paragraph; new "Estimate against peak" section on the results page |
| Data | `docs/benchmark_results/data/rocm-memory-gfx1151-2026-09-30/estimate-vs-peak-2026-10-07.txt`, the raw guarded `[Memory]` and peak lines for all 18 runs |

Three commits: the estimator change with tests and docs (`e387531f`), a docs correction of which estimate consumers see the reserve (`160f7e02`), and a test fix that restores non-UTF-8 environment values in the in-flight tests' guard (`9374ae72`). 8 files, 661 insertions and 13 deletions.

## 3. Design

### 3.1 Parsing like the backend

The estimate has to reserve what the backend will enforce, so `rocm_inflight_reserve_bytes` mirrors `max_inflight_bytes()` in the overlay's `device.cpp` rather than using a Rust-idiomatic parse:

- Unset or empty: `ROCM_DEFAULT_MAX_INFLIGHT_MB` (1024 MiB).
- Otherwise the `strtoull(e, &end, 10)` behavior: leading C whitespace and one optional `+` or `-` are skipped, then decimal digits. The value is kept only when the whole string was consumed, the first byte is not `-`, the digits fit in 64 bits, and the MiB count still fits once shifted by 20.
- Anything else (`1G`, `8abc`, `-1`, an overflow) falls back to the default, as the backend does.
- `0` turns the bound off in the backend and reserves 0 here. An unbounded in-flight set is not modeled; the page measured 20.60 GB peaks for the 8B model that way, and the env-var docs say so.

One strtoull detail is reproduced deliberately: a `-` after leading whitespace is not rejected by the overlay's first-byte check, and strtoull negates it in unsigned arithmetic, so the helper uses `wrapping_neg` and then lets the shift check reject the huge result.

`default_budget_and_env_name_match_the_overlay_source` reads `device.cpp` and asserts it still declares `constexpr size_t default_max_inflight_mb = 1024;` and still reads `std::getenv("MLX_ROCM_MAX_INFLIGHT_MB")`. If the backend's default or variable name drifts, this test fails instead of the estimate silently disagreeing.

On the Rust side the variable is read with `std::env::var(..).ok()`, so a non-UTF-8 value reads as unset and takes the default. The last commit fixes the test guard so it restores such values faithfully after a test, rather than changing the parser.

### 3.2 Compile-time gating

The term is chosen with `#[cfg(feature = "rocm")]`, the same pattern as the ROCm cache-limit default in `src/execution/runtime.rs`. The issue required this over a runtime backend query because the estimator is documented as side-effect-free with respect to MLX state: it must not bring up a runtime to learn which backend it is on. A ROCm build compiles no Metal or CUDA backend, so the build feature already answers the question. The non-ROCm stub returns 0, which keeps Metal and CUDA totals byte-identical to before; `non_rocm_estimate_reserves_nothing_for_inflight` asserts that on those builds.

### 3.3 Where the term appears

- **`total_bytes` and `runtime_headroom_bytes`.** `runtime_headroom_bytes` was already documented as the full reserve beyond `weights + kv`; it is now allocator overhead plus activation plus in-flight, and `total_bytes` follows. `assert_total_is_the_sum_of_its_terms` recomputes the factor overhead independently so a term dropped from the total cannot hide inside the headroom figure.
- **`format_estimate`.** The allocator-overhead line now subtracts both activation and in-flight from the headroom, so it still shows only the factor term. A separate `Backend in-flight: <size>  (MLX_ROCM_MAX_INFLIGHT_MB)` line is printed only when the reserve is non-zero, so Metal, CUDA and ROCm with `0` print nothing new.
- **`inspect --json`.** `InspectReport` gains `backend_inflight_bytes`, documented in the README and env-var page as non-zero only on ROCm and part of `headroom_bytes`. Recipe tooling that already sums the fields keeps working because `headroom_bytes` and `total_bytes` include it.
- **The generate log.** `log_estimate_vs_actual_delta` adds `backend_inflight_bytes`, so the estimate-versus-actual delta can be attributed per term.
- **`auto_kv_budget_bytes`.** See 3.4.

`--recommend-quant` is not affected: it reads only the estimate's weight and KV figures. The first revision of the env-var page said otherwise; commit `160f7e02` corrected the `MLX_ROCM_MAX_INFLIGHT_MB` paragraph during review.

### 3.4 Why the auto KV budget subtracts the reserve

`--kv-cache-budget auto` inverts the estimate's fit inequality. With `total = factor × (weights + kv) + activation + inflight` and `total ≤ available`, solving for KV gives `kv ≤ (available − activation − inflight) / factor − weights`. If the inversion ignored the new term, it would choose a KV size whose own estimate exceeds the available memory by up to the budget, so the preflight would reject the configuration the auto policy just picked. `auto_kv_budget_leaves_room_for_the_inflight_reserve` checks the arithmetic on a fixed estimate (25e9 available, 1e9 activation, 1e9 in-flight, 10e9 weights gives 9,166,666,666 bytes) and that the resulting total fits.

### 3.5 Saturation

The largest budget the parser accepts is `u64::MAX >> 20` MiB, whose byte value is close to `u64::MAX`. All additions into the headroom and total use `saturating_add`, and `rocm_largest_accepted_budget_saturates_the_total` checks that such a value saturates rather than wrapping into a small total that would wrongly "fit".

## 4. The 18-Configuration Measurement

Peaks were re-measured on gfx1151 with a release build of main at `97f35bca` plus this change (the change touches only the estimate, not what runs). Each row is one `mlxcel-bench-decode` run (`-n 128 --warmup-tokens 20 --ignore-eos --prompt-tokens N`, N = context minus 128) inside `scripts/rocm_gpu_guard.sh --idle-secs 30`, accepted only from a clean attempt; 4 attempts were rejected as contended and rerun. The estimates come from `mlxcel inspect --max-tokens <context> --json`. The peak is printed in GB to two decimals, so the peak bound adds the 0.005 GB the rounding can hide. All figures are bytes.

| Model | Context | Budget (MiB) | Peak bound | Estimate before | Estimate now | Now minus peak |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct-4bit | 640 | 1024 (default) | 6,145,000,000 | 5,579,686,809 | 6,653,428,633 | 508,428,633 |
| Llama-3.1-8B-Instruct-4bit | 640 | 256 | 5,185,000,000 | 5,579,686,809 | 5,848,122,265 | 663,122,265 |
| Llama-3.1-8B-Instruct-4bit | 640 | 4096 | 9,235,000,000 | 5,579,686,809 | 9,874,654,105 | 639,654,105 |
| Llama-3.1-8B-Instruct-4bit | 4096 | 1024 (default) | 6,955,000,000 | 6,103,135,948 | 7,176,877,772 | 221,877,772 |
| Llama-3.1-8B-Instruct-4bit | 4096 | 256 | 6,025,000,000 | 6,103,135,948 | 6,371,571,404 | 346,571,404 |
| Llama-3.1-8B-Instruct-4bit | 4096 | 4096 | 10,115,000,000 | 6,103,135,948 | 10,398,103,244 | 283,103,244 |
| Qwen3-30B-A3B-4bit | 640 | 1024 (default) | 18,665,000,000 | 20,717,224,703 | 21,790,966,527 | 3,125,966,527 |
| Qwen3-30B-A3B-4bit | 640 | 256 | 17,845,000,000 | 20,717,224,703 | 20,985,660,159 | 3,140,660,159 |
| Qwen3-30B-A3B-4bit | 640 | 4096 | 21,815,000,000 | 20,717,224,703 | 25,012,191,999 | 3,197,191,999 |
| Qwen3-30B-A3B-4bit | 4096 | 1024 (default) | 18,915,000,000 | 21,109,811,558 | 22,183,553,382 | 3,268,553,382 |
| Qwen3-30B-A3B-4bit | 4096 | 256 | 18,785,000,000 | 21,109,811,558 | 21,378,247,014 | 2,593,247,014 |
| Qwen3-30B-A3B-4bit | 4096 | 4096 | 19,265,000,000 | 21,109,811,558 | 25,404,778,854 | 6,139,778,854 |
| Qwen2.5-7B-Instruct-4bit | 640 | 1024 (default) | 5,915,000,000 | 5,240,405,811 | 6,314,147,635 | 399,147,635 |
| Qwen2.5-7B-Instruct-4bit | 640 | 256 | 4,855,000,000 | 5,240,405,811 | 5,508,841,267 | 653,841,267 |
| Qwen2.5-7B-Instruct-4bit | 640 | 4096 | 9,125,000,000 | 5,240,405,811 | 9,535,373,107 | 410,373,107 |
| Qwen2.5-7B-Instruct-4bit | 4096 | 1024 (default) | 6,175,000,000 | 5,469,414,809 | 6,543,156,633 | 368,156,633 |
| Qwen2.5-7B-Instruct-4bit | 4096 | 256 | 5,415,000,000 | 5,469,414,809 | 5,737,850,265 | 322,850,265 |
| Qwen2.5-7B-Instruct-4bit | 4096 | 4096 | 9,455,000,000 | 5,469,414,809 | 9,764,382,105 | 309,382,105 |

**Before**: the estimate was below the peak bound in 9 of 18 rows: every dense row except the 256 MiB ones, plus the MoE model at 640 tokens and 4096 MiB.

**After**: the estimate covers all 18. The smallest margin is 221,877,772 bytes (Llama-3.1-8B, 4096 tokens, default budget). Each "now" figure is the "before" figure plus exactly the budget in bytes (for example 5,579,686,809 + 1,073,741,824 = 6,653,428,633), which confirms the term is purely additive and the rest of the estimate did not move. The MoE margin stays large (2.59 to 6.14 billion bytes) because the factor already over-reserves on its weights.

**The pinned test.** `memory_estimate_inflight_tests::measured::rocm_estimate_covers_every_measured_gfx1151_peak` (ROCm builds) rebuilds each of the 18 estimates from the checkpoint's config and weight size, which equal the `inspect` column byte for byte, sets the row's budget, and asserts estimate >= peak bound. It also counts the rows the estimate would miss without the term and asserts that count is non-zero, so the test fails both when a row is uncovered and when the term stops being load-bearing. With the term forced to 0, this test and two others fail.

## 5. Side Effect: Prompt-Cache Capacity Ceilings

The server's model-aware prompt-cache defaults (`src/server/prompt_cache/snapshot_sizing.rs`, for the snapshot store and the KV store) are sized from the estimate's slack (`slack_bytes()`), with a ceiling of a quarter of the slack. The issue flagged this downstream and required the PR to state it. On ROCm the reserve lowers the slack by the budget, so the ceiling falls by a quarter of the budget: 268,435,456 bytes at the default.

At the 8192-token representative length on this host (76.80 GiB available):

| Model | Ceiling before (GB) | Ceiling now (GB) | Six representative entries (GB) | Capacity changed |
|---|---:|---:|---:|---|
| Llama-3.1-8B-Instruct-4bit | 18.93 | 18.66 | 6.44 | No |
| Qwen3-30B-A3B-4bit | 15.22 | 14.95 | 4.83 | No |
| Qwen2.5-7B-Instruct-4bit | 19.18 | 18.91 | 2.82 | No |

No capacity changed because the ceiling does not bind here: six representative entries need 2.82 to 6.44 GB, well under ceilings of 14.95 to 18.91 GB, so the capacity is set by the entry count and not by the ceiling. In general a capacity can change only where the ceiling binds, by at most the larger of a quarter of the budget and one entry, and never below the compiled-in default. On a smaller-memory ROCm host, or with a large `MLX_ROCM_MAX_INFLIGHT_MB`, the ceiling could bind and a capacity could drop by that bound. That is the intended behavior: the cache should not be sized into memory the backend may claim for in-flight batches.

## 6. Verification

On gfx1151:

- **Guarded peaks**: the 18 rows in section 4, raw lines in `estimate-vs-peak-2026-10-07.txt`.
- **Pinned table**: `rocm_estimate_covers_every_measured_gfx1151_peak` passes, and fails (with two other tests) when the term is forced to 0.
- **Targeted tests**: `cargo test --features rocm --lib -- execution::memory_estimate server::prompt_cache::snapshot_sizing`, 70 passed.
- **Lints and gates**: clippy on lib and tests with `-D warnings`, `dead_doc_pointers`, and the fast `make verify-*` script gates pass.
- **Unit's `make verify-rocm`** at `9374ae72` (on main `87538835`): OK, 12,004 passed, 0 failed, 384 ignored over 153 test binaries.
- **Orchestrator's `make verify-rocm`**: re-run on the same head before merge.

Not verified: Metal and CUDA, which are not available on this host. On those builds the code path is the `#[cfg(not(feature = "rocm"))]` stub that returns 0, plus the new zero-valued field and JSON key. `non_rocm_estimate_reserves_nothing_for_inflight` was not compiled or run here.

## 7. Technical Decisions

- **An additive term, not a ROCm factor.** The excess tracks the in-flight budget, not weights; a larger factor would over-reserve further for large models that are already over-estimated.
- **Reserve exactly the budget the backend enforces.** Mirroring `strtoull` and the overlay's acceptance checks, including the fallback for invalid values, keeps the estimate and the backend reading one number. The drift test against `device.cpp` makes that agreement a checked contract instead of a comment.
- **`0` reserves nothing.** The bound off means no bound to reserve. The estimate does not try to model an unbounded set and the docs say so, rather than inventing a figure.
- **Compile-time selection.** `#[cfg(feature = "rocm")]` keeps the estimator side-effect-free and leaves non-ROCm totals unchanged by construction.
- **Surface the term separately.** A distinct `Backend in-flight` line, JSON field and log field let users and tooling see why a ROCm estimate is larger, and keep the allocator line meaning only the factor.
- **Make the auto budget consistent with the estimate.** Subtracting the reserve in the inversion prevents the auto policy from choosing a KV size the preflight then rejects.
- **Pin the measurement, not just the formula.** The 18-row test turns a one-time benchmark into a regression check, and its "load-bearing" assertion keeps the test from passing vacuously.

## 8. Residual Risks and Follow-ups

- **Leftover doc error (pre-existing).** The `MLXCEL_HEADROOM_FACTOR` row in `docs/environment-variables.md` still lists `--recommend-quant` among the consumers of the factor. `--recommend-quant` reads only the estimate's weight and KV figures, so it does not use the factor. The row predates this PR; the PR appended a sentence to it but left the consumer list as it was. A one-line follow-up should remove `--recommend-quant` from that row.
- **One host, one GPU generation.** All peaks are on gfx1151 (RDNA 3.5, unified memory). The reserve is the budget by design, so it should transfer, but the margin on other ROCm targets is unmeasured.
- **One run per row.** Each peak is a single clean run. The smallest margin (about 0.22 GB on the 8B at 4096 tokens) is comfortable for the rounding but not proven against run-to-run variance of the peak.
- **Workloads outside the sweep.** The measurement covers batch 1 decode at 640 and 4096 tokens. Larger batches or contexts grow the KV and activation terms, which the estimator already models, but the interaction with in-flight transients at those sizes was not measured.
- **`MLX_ROCM_MAX_INFLIGHT_MB=0`.** The estimate is known to be far below the peak in that mode (20.60 GB measured for the 8B). That is documented, not fixed.
- **Metal and CUDA untested here.** The stub is trivial, but the non-ROCm test has not run on this host.

## 9. Learning Points

- **Check the units claim against the bytes before designing around it.** The issue's GiB reading looked plausible and would have shrunk the problem by about three quarters. Printing the estimate in bytes settled it and changed which fix was appropriate.
- **When a reserve tracks a budget, model the budget.** A multiplicative factor encodes "overhead proportional to size"; the in-flight set is bounded by a configured number, so an additive term equal to that number is both simpler and more accurate.
- **Mirror the parser of the component you are predicting.** An estimate that parses the variable differently from the backend can disagree on exactly the malformed values users are most likely to set. Matching `strtoull` and testing against the C++ source keeps them aligned.
- **Every inversion of a formula must change with the formula.** Adding a term to the estimate without updating `auto_kv_budget_bytes` would have made the auto policy pick configurations its own preflight rejects.
- **A measurement test should prove it is load-bearing.** Asserting that some row fails without the term keeps the regression test from passing after a refactor that removes what it was written to protect.
