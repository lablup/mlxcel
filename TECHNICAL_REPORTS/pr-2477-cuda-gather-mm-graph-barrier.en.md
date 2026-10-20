# PR #2477: CUDA sorted gather_mm batch stride and captured-graph barriers

**Date**: 2026-10-20
**Status**: Implemented and verified on CUDA (GB10, sm_121); PR open

## Summary

Closes #2462. Fixes two CUDA overlay defects that produced silently wrong values on GB10, each hidden behind a model-level workaround: (1) the sorted `gather_mm` branch (`gather_mm_rhs`) read the stride of a size-1 axis as its grouped-GEMM leading dimension, so rows after the first of an expert group read shifted activations when a producer such as a per-expert bias add handed down a unit stride; (2) the first execution of a freshly captured large CUDA graph intermittently returned wrong values in about 40% of fresh processes (GLiNER2.5's 12-layer DeBERTa-v2 encoder), because upstream orders a stream-captured child graph only by the buffers it declares. The workarounds (`PrivacyFilterMoe.sort_routed` restricted to quantized planes, `DebertaV2Encoder` ending a graph after every layer) are removed.

## Problem

Both defects shipped in PRs #2460 and #2461 with workarounds and a note that they deserve their own issue. The stride defect made the Privacy Filter's dense planes (7.5k-token inputs) take the unsorted gather, about 1.5x the 8-bit forward, and risked any bf16 MoE that feeds f32 activations into sorted `gather_mm` with `M == 1`. The graph defect hid behind per-layer graph breaks, which lose the capture amortization the encoder was captured for in the first place, and the same trigger (kernel-kernel edges between stream-captured child graphs) can hit any workload that captures multiple cuBLASLt matmuls into one graph.

## Technical decisions

- **A1: layout-correct leading dimension.** `gather_mm_rhs` now runs `ensure_row_contiguous(a)` and takes `lda = a.shape(-1)`, the batch stride, instead of `a.strides(-2)` (the stride of a size-1 axis, donation-dependent). This matches the `GatherQMM` grouped path and upstream Metal, which the bug's bisection showed is exact in every order. Marked `[mlxcel #2462]` next to the upstream base.
- **A2: captured child graphs as barriers.** New overlay `mlx/backend/cuda/device.cpp` (the 23rd CUDA overlay file, based on upstream `243bdf09c`): a captured child graph joins every open branch of the graph before it, gates every node added after it, and chains to the previous captured child graph, on top of the data dependencies MLX derives from buffers. `MLXCEL_CUDA_GRAPH_CAPTURE_BARRIER=0` restores upstream's data-only ordering for A/B; `MLXCEL_CUDA_GRAPHS_DOT_VERBOSE=1` dumps each graph with verbose DOT flags so node parameters show which buffers concurrent nodes touch.
- **Concurrent batches inherit the capture.** A `ConcurrentContext` batch whose nodes include a captured child graph makes its join node the barrier and clears the open-branch state afterwards, so cuBLASLt matmuls grouped under a concurrent section gate later nodes the same way single nodes do.
- **Env flag grammar.** The two new switches parse with the convention `docs/environment-variables.md` documents (trimmed, case-insensitive, `0`/`false`/`off`/`no` and `1`/`true`/`on`/`yes`) via a new `mlxcel_env_flag` helper, instead of MLX's `env::get_var`, whose `atoi` reads `on` as 0 and would have disabled the barrier by default (review finding).
- **Barrier bookkeeping leaked on purpose.** The per-encoder barrier state is a leaked `thread_local` map: `~CommandEncoder` runs `commit` and `clear_graph_state` during thread exit, after a `thread_local` with a destructor may already be gone. The comment records that and the `new_thread_unsafe_stream` caveat that would move the map under a mutex.
- **Workarounds removed.** `PrivacyFilterMoe.sort_routed` returns true unconditionally again, so dense planes take the sorted path; `DebertaV2Encoder` no longer ends a CUDA graph after every layer on CUDA (`graph_break_per_layer` dropped). Metal is untouched.

## Changes

- `src/lib/mlxcelverse/overlays/cuda/mlx/backend/cuda/matmul.cpp` (A1), `device.cpp` (new, A2), `README.md` (23 files, device.cpp provenance and the pin-bump drift check).
- `src/lib/mlxcelverse/CMakeLists.txt` (device.cpp in the overlay set).
- `src/models/openai_privacy_filter_layers.rs` (`sort_routed` restored), `deberta_v2.rs` (`graph_break_per_layer` removed), `switch_layers.rs` (test module decls), `openai_privacy_filter_tests.rs`, `deberta_v2_tests.rs`.
- New tests: `switch_layers_sorted_gather_layout_tests.rs` (3), `switch_layers_sorted_gather_history_tests.rs` (3), `deberta_v2_tests.rs::real_encoder_first_run_equals_replay`, `switch_layers_sorted_gather_test_support.rs` (shared harness: symmetric RNG, `bf16_values`, reference).
- `src/lib/mlxcel-core/cpp/mlx_cxx_bridge.{cpp,h}`, `src/lib/mlxcel-core/src/lib.rs` (`array_strides` binding).
- `docs/environment-variables.md` (the two switch rows and their grammar), `docs/token-classification.md`, `CHANGELOG.md` (the full family list that sorts its routed rows).

## Verification

GB10, `--profile test-fast --features cuda`, test binaries under `gpu-lock` with `--test-threads=1`, parity runs with `MLX_ENABLE_TF32=0`.

| Check | Result |
|---|---|
| Sorted gather layout tests (unit-stride singleton axis, transposed plane, biased expert add) | 3/3, and all fail on the unfixed overlay (direct layout construction) |
| Allocation-history tests (fresh-process reference, 8-bit MoE predecessor, other sorted shapes) | 3/3 |
| Privacy Filter checkpoint gates on the sorted dense path (incl. the bf16, 8-bit, bf16 order #2460 saw drift in) | 10/10 |
| `real_encoder_first_run_equals_replay`, 20 fresh processes, barrier on | 20/20 (first encode == replay bit for bit) |
| Same, barrier off + `CUDA_MODULE_LOADING=EAGER` | 13/20 FAILED, so lazy module loading is not the trigger; the barrier is what fixes it |
| Decode A/B, barrier on vs off, same binary, 4 interleaved rounds per arm | qwen3-moe-4bit ON 76.46 vs OFF 75.96 tok/s (+0.7%), phi-3.5-moe-4bit ON 43.21 vs OFF 42.61 (+1.4%); paired deltas alternate sign and sit inside the ~1.5% host noise floor, so the barrier costs nothing measurable on MoE decode |
| Full workspace gate on `5e29de95` | build 0 (161 bins), clippy 0, doctests 0, contract set 0; 160/161 bins pass, `mllama_parity` failed once with the known 1-ulp `ragged_multi_image_states_concatenate_real_rows` flake and passes on rerun (8/8); issue #2467 |

The EAGER result is recorded for the upstream report: chaining the graph in program order removed the corruption, chaining only kernel-kernel edges did not, and EAGER module loading does not fix it.

## Follow-ups

- MLX upstream bug report with the trigger data above (stream-captured child graphs ordered only by declared buffers overlap unrelated kernel nodes on GB10).
- The producers that hand the branch a unit stride today are the per-expert bias adds of GPT-OSS, Phixtral and the Privacy Filter; the CHANGELOG entry lists every family that sorts its routed rows, so a future family can check the same pattern.
- The GLiNER head's own `take` calls were observed clean in 3 processes x 4 rounds, consistent with the barrier fix; not separately proven.

## Statistics

| Item | Value |
|------|-------|
| Files changed | 18 |
| Lines added | 1519 |
| Lines deleted | 73 |

| Commit | Message |
|--------|---------|
| `f5e14115` | fix(cuda): take the batch stride in sorted gather_mm, make captures barriers |
| `5e29de95` | fix(cuda): parse barrier flags per docs, extend barriers to batches |

## Related

- Issue #2462; workaround PRs #2460 (sort_routed) and #2461 (per-layer graph breaks).
- #2174 (Llama fresh-process divergence that persists with graphs off) is related but distinct.
- The one-ulp mllama parity flake seen in the gate is issue #2467, unrelated.
