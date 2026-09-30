# Technical Report: PR #2059 - ROCm Correctness Matrix Rows, Paged-Attention Port Predicates, and a Strided-Scan Grid Fix

**Date**: 2026-09-30

**Status**: Implemented and validated on the gfx1151 host; pending merge.

**Languages**: C++ (ROCm overlay header, paged-attention launchers, cxx bridge), Rust (production gates, tests), Python (CI checker), Markdown, TSV traces

**Risk Level**: Low to medium. The overlay change alters the launch grid of every strided scan on ROCm, which is a correctness fix but touches a hot primitive. The production gate changes answer the same on every backend today by construction. Metal and CUDA were not run.

## Executive Summary

Issue #1809 asks for two things: a ROCm correctness matrix against a Metal reference, and a green `make verify-test-rocm` gate. The first run (#1826) covered four checkpoints; the gate failed on 37 tests at `origin/main` `4595b06f`.

This PR takes the gate to the three failures that belong to other work. The 34 paged-attention tests failed only because those kernels have no ROCm port (#1814). They now skip in one place, through predicates that read each kernel's own `KernelPorts` table, and only on ROCm, so they run again the moment a port lands. The production paths in front of these kernels ask the same predicates.

It also adds the five missing matrix rows on the ROCm side: a second dense model, a sliding-window model, two SSM hybrids and a VLM. The Metal side is pending because no Metal host was reachable; the exact commands are committed. Running the SSM hybrids found a real ROCm defect. A strided scan's launch grid was sized by an outdated local copy of `get_2d_grid_dims`, which launched up to 16 times the blocks it needed and wrote past the array, faulting the GPU. That is fixed and covered by a test that faults without the fix.

Finally it settles a question #2029 left open: the Nemotron-H `use_fused` guard is a confirmed fix for the opt-in `MLXCEL_FUSED_MOE_RELU2` path (the pre-guard build aborts) and defensive for the default path (the pre-guard build generates).

## 1. Problem Statement

### The gate was red for a missing port, not a wrong number

`make verify-test-rocm` failed with 37 tests. 34 of them exercised the fused paged-attention kernels (v1 decode, v2 partial, merge), which have Metal and CUDA ports and `.rocm = nullptr`. They either launched the kernels and got the launcher's refusal, or went through a production path that declined on the missing port before reaching the decision the test checks. None was a numerics defect. A red gate whose failures are all known makes a new failure easy to miss.

There was no port predicate to skip on. `paged_attention.cpp` had a `KernelPorts` table but exported no `*_available()`, which is also why `ffi_tests.rs` sat in the dispatch checker's `BACKEND_ENUMERATION_TODO` (#1814 plan item 5). The production gates asked `custom_kernels_available()`, which is Metal-or-CUDA by definition, so a future HIP port would have stayed unreachable behind them.

### Five matrix rows had never run

The sliding-window, SSM-hybrid and VLM rows, and `Qwen2.5-7B-Instruct-4bit`, had been blocked on #1803 and #1805, both since closed. None of the checkpoints was on the host.

### The Nemotron-H guard was traced, not executed

PR #2029 added `&& custom_kernels_available()` to Nemotron-H's `use_fused` without running a Nemotron-H checkpoint on ROCm, and recorded the term as defensive.

## 2. Change Summary

- **Port predicates.** `paged_attention_decode_available`, `paged_attention_v2_partial_available` and `paged_attention_merge_states_available` in `src/lib/mlx-cpp/turbo/` each return `has_kernel_port` on their table. The bridge exposes `paged_attention_kernels_available` (all three), `paged_attention_decode_available`, `paged_attention_v2_available` (partial and merge) and `paged_attention_merge_available`.
- **Production gates.** `PagedBlockPool`'s batched paged decode (`cache/paged.rs`) asks the all-kernels predicate, MLA split-KV (`mla/mod.rs`) the merge predicate, `paged_decode_backend` (`layers.rs`) the v1 predicate, and `run_sparse_decode` (`paged_v2/sparse.rs`) the v2 predicate after its shape and sparsity declines. The sparse path had no port check before; on ROCm each layer of each step built its inputs, had the launch refused, and reported a rejected plan.
- **Test skips.** `src/lib/mlxcel-core/src/test_support/kernel_ports.rs` holds `require_paged_attention_port!`, `require_paged_decode_port!`, `require_paged_v2_port!` and `require_paged_merge_port!`. Each returns early only when the backend is ROCm and the predicate for the kernels that test needs is false, and prints `skipping <module>:<line>: ROCm has no <kernels> kernel port yet (lablup/mlxcel#1814) ...` straight to the process's stderr. 34 failing tests and two `ffi_tests` tests that used to return silently off Metal and CUDA use them.
- **Checker.** The two `ffi_tests` gates spelled `!metal_is_available() && !cuda_is_available()` now use the macro plus a no-GPU-backend return, and `ffi_tests.rs` leaves `BACKEND_ENUMERATION_TODO`.
- **Scan grid fix.** `get_2d_grid_dims(shape, strides, divisor)` in `patches-rocm/mlx/backend/rocm/kernel_utils.hpp` delegates to `get_2d_grid_dims_common` from `mlx/backend/common/utils.cpp`, as the CUDA backend does. `LOCAL_FIXES.md` item 22; `tests/rocm_strided_scan.rs`.
- **Traces.** `benchmarks/logit_traces/rocm_gfx1151_c5fe9a16/`: sixteen traces (five checkpoints at w1, w8, w256, plus `gemma-3-4b-it` at `w8ctx1536`), METADATA, RUNS, SHA256SUMS and a README with the Metal commands.
- **Docs.** `docs/benchmark_results/rocm-correctness-gfx1151-2026-09-30.md`, pointers from the 2026-09-12 run and `docs/installation.md`, and the Makefile note on the `verify-rocm` order.

## 3. Technical Decisions

### Skip on the table, only on ROCm, and say so

The user's condition for skipping was that the skip name #1814, sit in one visible place, and reappear when a port lands. Reading the port table meets the third directly: the predicate and the dispatch share `port_for`, so a filled `.rocm` entry turns the predicate true. Restricting the skip to ROCm means a predicate that wrongly answers false on Metal or CUDA fails the test there instead of passing it. Printing through `std::io::stderr()` rather than `eprintln!` matters because libtest captures `eprintln!` for passing tests and discards it; the first version of the helper printed nothing in the gate log.

### Per-kernel predicates, not one

The first version used one all-kernels predicate for every test. Review pointed out that #1814 may port the kernels one at a time, and then a v2-only test would stay skipped with a message that had become false. Each test now names the kernels it launches (v1, v2, merge, or all for the batched decode, which may take either path).

### Fix the scan grid rather than file it

The SSM-hybrid rows could not produce a trace without it, and the cause was a small, local divergence from upstream: the overlay carried its own body for one overload while the other overload already delegated to the shared helper. Delegating restores upstream's behavior exactly. The shared helper takes the gcd of each dimension against the remaining divisor; for a contiguous input the product of dimensions is a multiple of `axis_size * stride`, so the gcd pass always ends with a divisor of 1 and the grid is exactly `size / divisor`.

### Detect the scan defect by its fault

An oversized grid leaves the in-bounds output correct, because the extra blocks only touch memory past the array. A value comparison cannot see it. The test therefore includes a shape whose old grid overshoots by 16x on a 25 MB array, which leaves mapped memory and faults; with the header reverted it does, on gfx1151. The value comparison is kept for what it can check, the scan itself, on small integers stored as f32 so the GPU and CPU agree bit for bit. Review caught that the first version of the input generator produced only zeros; it was fixed before merge.

### Add a sliding-window shape

The standard widths never give `gemma-3-4b-it` more than 520 tokens of context, inside its 1024-token window. A `w8ctx1536` shape with `MLXCEL_TRACE_START_TOKEN=1536` gives every scored chunk 1536 tokens of context, so the sliding-window mask and rotating cache are exercised past the window.

### Do not fabricate the Metal half

No Metal host was reachable and no Metal trace for these checkpoints exists in the repository. The ROCm traces are committed with the revisions, commit and binary hash, and the README gives the exact commands, and notes the pairings that compare a kernel on Metal against a graph on ROCm (the hybrids' `w1`, all of Nemotron-H).

## 4. Validation

All on gfx1151 (Radeon 8060S, ROCm 10.0.0 packages, HIP 7.15.26333).

- Paged-attention tests: `cargo test -p mlxcel-core --profile test-fast --features rocm --lib -- --test-threads=1 paged_v2 mla:: cache::paged ffi_tests::test_fused_paged` after the per-kernel refinement: 266 passed, 0 failed, with 36 skip lines (the 34 that failed at `4595b06f` and the two `ffi_tests` that used to return silently). mlxcel-core clippy `--lib --tests -D warnings`, the fast script gates and `dead_doc_pointers` pass.
- `tests/rocm_strided_scan.rs`: 3 passed with the fix. With the old header the `[64, 96, 32, 32]` case faults in `strided_scan`.
- Traces: sixteen, all exit 0, no NaN (`RUNS.txt`). Before the fix, granite w8 and Nemotron-H w8 and w256 faulted in `strided_scan`.
- Generation: all five checkpoints generate coherent text with `-t 0`; the two VLMs describe `tests/fixtures/test_image_shapes.png` correctly.
- Nemotron-H guard: at `680eb064` the default path generates and `MLXCEL_FUSED_MOE_RELU2=1` aborts with `[metal_kernel] No Metal back-end.` (exit 134); at `4595b06f` both generate.
- `make verify-rocm` on the branch before the review follow-ups (`bae40d24`): versions, kernel dtype keys, kernel port dispatch, llama-compat, fmt, workspace clippy and the ROCm smoke passed; `verify-test-rocm` had 11736 passed, 3 failed, 377 ignored. The three are the known failures owned elsewhere. The final gate on the rebased branch is recorded in the PR body.

## 5. Learning Points

- **A matrix row is also a crash test.** The SSM-hybrid rows produced no cross-backend number, because the Metal side is pending, and still found the most important defect in this PR. Running a new model family end to end on a backend exercises primitives no unit test had combined in that shape.
- **A local copy of an upstream helper drifts silently.** One overload had been updated to delegate and the other had not. Whether a shape was hit depended on how the head count and chunk length shared factors, so short prompts and the one-token width worked and hid it.
- **A skip needs a visible trace.** `eprintln!` in a passing test vanishes under libtest's capture. A skip that cannot be seen in the gate log reads as a pass.
- **Check a test's input, not only its assertion.** The first scan test compared GPU and CPU exactly, and was fed only zeros. It passed with the fix and would have faulted without it, so the revert check alone did not reveal that the value comparison checked nothing.
- **Execute the claim.** The Nemotron-H guard was recorded as defensive from a trace. Running both builds showed it is a real fix, on a path the trace did not follow.

## 6. Caveats and What Is Not Verified

- **Metal and CUDA were not run.** The new predicates are true on both by construction (same table, same `port_for`), so the gates and skips do not change there, but that was not executed. The overlay fix is in `patches-rocm/`, which Metal and CUDA builds never copy.
- **The Metal side of the five rows** is pending; there is no cross-backend number for them.
- **The VLM trace is text-only.** The vision tower is covered only by the two image generation checks.
- **Only gfx1151** was run, and the strided-scan test's revert check relies on the fault the overshoot causes on this host's allocator layout.
- **Traces shared the GPU** with another unit's processes; they are correctness traces, not timings.

## 7. Remaining Work

- #1809 stays open: the Metal side of the five rows, and the three remaining gate failures (`layers::tests::prefill_dense_gemm_matches_qmm_bytes_where_eligible` in bf16, and #2037's `gelu_approx_matches_mlx_nn_bit_for_bit` and `family_order_is_exhaustive`).
- #1814: the paged-attention ports that end the 36 skips, and the other port tables.
- #1813: propose `LOCAL_FIXES.md` item 22 to the fork.

Refs: #1809, #1801, #1814, #1813, #1826, #2029, #2037, #1785.
