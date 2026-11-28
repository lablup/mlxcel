# PR #2933: Native 4-bit GEMV in the ragged batch engine

**Date**: 2026-11-10
**Status**: Completed (PR open as a draft; closes #2922)

## Executive Summary

The ragged continuous-batching engine now runs the checkpoint's packed 4-bit projections through the native HIP GEMV, the same mechanism that took single-sequence decode from 24.6 ms to 7.1 ms per step. Six new exports share the existing HSACO: `qmv4_multi_kernel`, `qmv4_multi_row_kernel`, and fused q/k/v and gate/up multi-row variants. One workgroup owns one output row, dequantizes each weight word once, and accumulates it into one accumulator per activation row, so the weight stream is read once per row instead of once per row per request. On `gfx1151` with llama-3.2-1b-instruct-4bit, ragged decode at B=8 measures 74.5 and 76.7 tok/s against 59.8 for the f16 path, and B=4 measures 60.0 and 62.2 against 65.6, a roughly 5 percent regression from the fixed eight-accumulator layout that is recorded as a follow-up. Every run prints REFERENCE-EXACT PASS, the GEMV launch count per step falls from 113 to 65 independently of B, and an untied Qwen2.5-7B checkpoint is also reference-exact. 9 files, +982/-51.

## 1. Problem Statement

`XlaBatchEngine` still streamed f16-resident projection weights. The single-sequence graph had used the packed 4-bit weights since #2890 and #2892, but the batch graph is emitted with `ProjectionLayout::OutIn` and f16 weights (`src/lib/mlxcel-xla/src/iree.rs`, `src/lib/mlxcel-xla/src/emitter/model.rs`), and `linear_seq_proj` fell through to the f16 `dot_general` because no packed operand was registered for that graph. Reading the checkpoint's 4-bit weights instead of the f16 expansion cuts the bytes streamed per step by roughly 4x, which is the lever that moved single decode from 24.6 ms to 7.1 ms per step and the gap #2886 identified between the batch and single paths.

The kernel problem is different from single decode. `qmv4_kernel` maps one warp to one output row and both existing kernels read a single activation row `x[k]`. A batch step has B activation rows that all need the same weight rows. Calling the single-row kernel once per request would multiply launches by B and read the weight stream B times, which removes most of the byte saving. The batch path needs one launch that reads each weight word once and applies it to every activation row.

## 2. What Changed

### 2.1 Kernel (`src/lib/mlxcel-xla/csrc/qmv4.hip`)

- `qmv4_multi_kernel` keeps the warp-per-row layout with constants `K`, `N`, `B` and ABI `(x[B,K], w[N,K/8], scales[N,K/64], biases[N,K/64], y[B,N])`. Each lane loads a weight word once, dequantizes it once, and multiplies it into `B` accumulators. The warp-shuffle reduction is repeated per accumulator and writes `y[b*N + row]`.
- `qmv4_multi_row_kernel` is the block-per-row counterpart used when `K >= ROW_KERNEL_MIN_K`, with one shared-memory pass and a single barrier for all B accumulators.
- Fused `qkv` and `gateup` multi-row variants reuse the cumulative-N ABI from #2907: the row selects its set by cumulative output width, so q, k and v (and gate and up) still take one launch per group, not one per projection per row.
- Per-row accumulation order is identical to the single-row kernels. That is what keeps the results numerically identical to the f16 path and makes the benchmark print REFERENCE-EXACT rather than merely approximate.

### 2.2 Emitter and loader

- `emitter/builder.rs` gained `native_qmv_rows` over a rank-2 `[B, K]` activation for one to three operands. A rank-2 `linear_seq_proj` routes through it, and `gate_up_proj` picks the rank-2 or rank-1 form. The rank-1 assert on `native_qmv` was kept rather than relaxed.
- `emitter/model.rs` registers packed operands for the `OutIn` layout, fuses q/k/v and gate/up, and carries the logical shape per layout. `emit_decode_ragged_with_native` splices the executable sources into the ragged module; when no native module is passed, the f16 graph is byte-identical to before.
- `iree.rs` resolves the native mode once in `load_inner`, before `runtime_ffi_dimensions` and the weight upload, because the argument list and the resident weight set differ between the packed and f16 paths. The ragged gate requires a dense 4-bit group-64 checkpoint and a packed-supported layout; anything else logs one warning and stays on f16.
- `native_qmv.rs` selects the multi-row exports for the ragged graph and keeps the single-row group for the single-sequence graph. `module_parts` and `ragged_module_parts` are separate so each graph embeds only the exports it dispatches.

### 2.3 Fallback

With `HIPCC` unset or pointing at a missing compiler, with an unsupported architecture, or with a checkpoint that is not 4-bit group-64, the engine emits the previous f16 graph and logs one warning per load. Verified with `HIPCC=/nonexistent/hipcc` and a fresh artifact cache: one warning, f16 path, REFERENCE-EXACT PASS, 66.1 tok/s.

## 3. Measurements

All runs are on `gfx1151` under `scripts/rocm_gpu_guard.sh --hold`, llama-3.2-1b-instruct-4bit unless noted, same binary for the A/B.

| Setting | Native 4-bit | f16 path | Result |
|---------|--------------|----------|--------|
| B=8, run 1 | 74.5 tok/s | 59.8 tok/s | REFERENCE-EXACT PASS |
| B=8, run 2 | 76.7 tok/s | 59.8 tok/s | REFERENCE-EXACT PASS |
| B=4, run 1 | 60.0 tok/s | 65.6 tok/s | REFERENCE-EXACT PASS |
| B=4, run 2 | 62.2 tok/s | 65.6 tok/s | REFERENCE-EXACT PASS |
| B=4, untied Qwen2.5-7B-Instruct-4bit | 14.2 tok/s | not compared | REFERENCE-EXACT PASS |

- GEMV launches per ragged decode step: 113 to 65, independent of B.
- The issue's baseline for B=8 was 69 to 71 tok/s, and this work measures 74.5 to 76.7.
- B=4 native is about 5 percent below f16 in both runs. The kernels predicate a fixed eight accumulators for every B, so a B=4 launch does twice the accumulator work it needs. The acceptance gate in the issue is B=8 and every run is reference-exact, so this was left as a follow-up rather than hidden.

## 4. Technical Decisions

### 4.1 Fixed eight accumulators instead of per-B specialization

**Context:** the ABI must serve the ragged engine's supported B values (4 and 8) from one export group.

**Alternatives:**

| Option | Pros | Cons |
|--------|------|------|
| Per-B specialization (`MAX_B` variants or a compile-time B switch) | No wasted accumulator work, recovers the B=4 regression | More exports, more pipeline layouts, more code to keep in sync |
| **Chosen: fixed eight accumulators with a runtime B** | One export group for every supported B, simpler ABI and tests | About 5 percent regression at B=4 |

**Rationale:** the issue's target is B=8, where the multi-row form wins, and a wrong row cannot produce a wrong result because `y[b*N + row]` is indexed only for `b < B`. The specialization is a contained follow-up that does not change the graph.

### 4.2 Per-graph export embedding

The single-sequence graph and the ragged graph dispatch different export sets. Embedding the whole HSACO in both modules would carry unused exports into every vmfb. `module_parts` and `ragged_module_parts` emit only the group each graph needs, at the cost of two source builders over one artifact. The security review raised the shared artifact cache separately (see follow-ups).

### 4.3 Reuse the cumulative-N fused ABI

The batch engine would otherwise multiply the q/k/v and gate/up launch count by B. Folding the multi-output form into the same kernels keeps the ragged launch count at 65 for any B and avoids a second fusion contract that would drift from the single-sequence one from #2907.

## 5. Verification and Limits

- `cargo test -p mlxcel-xla --lib`: 399 passed, 0 failed, 19 ignored at the final tip.
- The targeted emitter test `native_ragged_decode_emits_multi_row_dispatches` and a B=4 launch-structure test cover the new dispatch shape and the fallback.
- `cargo fmt --all -- --check`, `cargo check -p mlxcel-xla --features iree --lib --tests`, and `cargo clippy -p mlxcel-xla --features iree --lib --tests -- -D warnings` are clean at the final tip. The `iree` feature has to be built from the root crate; the crate alone cannot link it.
- `hipcc --genco -O3 --offload-arch=gfx1151 csrc/qmv4.hip` compiles and the resulting HSACO contains all twelve symbols, including the six multi-row kernels.
- Not verified: the OpenXLA serve worker end to end with native mode at B>1. The bench harness is the same engine, but the worker admission path was not exercised in this run.
- `mistral-7b-instruct-v0.3-4bit` could not serve as a second checkpoint check because the OpenXLA emitter rejects `model_type = "mistral"` before any weight handling. Qwen2.5-7B-Instruct-4bit was used for the untied check instead.

## 6. Change Summary

| Item | Value |
|------|-------|
| Files changed | 9 |
| Lines added | +982 |
| Lines deleted | -51 |
| Tests added | 2 focused emitter/kernel tests |

| Category | Summary |
|----------|---------|
| Performance | Multi-row packed GEMV and fused multi-row q/k/v and gate/up for the ragged engine |
| Correctness | Reference-exact at B=4 and B=8, tied and untied checkpoints, fallback unchanged |
| Documentation | README, environment variable docs, CHANGELOG |

## 7. Follow-ups

1. Per-B accumulator specialization to recover the B=4 regression (about 5 percent, both runs).
2. The single-sequence native gate does not warn on an unsupported packed layout, while the ragged gate now does.
3. The shared `<temp>/mlxcel-xla-qmv4` HSACO cache is world-accessible and its key omits the hipcc version. This is pre-existing and repository-wide, and it was not changed in this PR.
