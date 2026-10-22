# PR #2537: Make the qmm determinism gate fail when it cannot run

**Date**: 2026-10-22
**Status**: Completed (PR open; refs #1558, which stays open)

## Executive Summary

PR #2537 reworks `tests/cuda_qmm_determinism.rs`. The test used to report `ok` on every host without running anything: its checkpoint was missing everywhere, and a missing checkpoint was a silent early return. Pointed at gemma-4-12B on sm_70 (#1558), it failed, but not because of the kernels. Gemma keeps its K/V in model-owned state, and the test never reset that state between iterations. The real-model test is now `#[ignore]`, panics under `MLXCEL_REQUIRE_MODELS=1` when no checkpoint resolves, calls `reset_runtime_state` before each iteration, and has a make target. Four checkpoint-free kernel tests were added. Nothing was run on sm_70, so #1558 stays open and the PR uses `Refs`. 6 files, +393/-57, of which about 320 lines are in the test file.

## 1. Change Summary

| Area | Change |
|---|---|
| `tests/cuda_qmm_determinism.rs` | Real-model test is `#[ignore]`, resolves its checkpoint through `tests/common`, panics under `MLXCEL_REQUIRE_MODELS=1` when none resolves, resets runtime state per iteration, reports distinct-logit count per step; four kernel tests |
| `tests/common/mod.rs` | New `repo_model_dir_any`; asserts a non-empty name list instead of falling back to the models root |
| `Makefile` | `verify-test-cuda-qmm-determinism` runs the model test for real; `verify-test-cuda` reports it as `ignored` but still runs the kernel tests |
| Docs | `docs/environment-variables.md`, `docs/installation.md`, `CONTRIBUTING.md` |

## 2. Technical Decisions

### 2.1 The sm_70 failure was a test defect

Gemma 3 and Gemma 4 keep their K/V in `ModelOwnedSequenceState` and return an empty slice from `make_caches`. The test's "fresh caches" were therefore never a fresh sequence: each iteration prefilled on top of the previous ones. This was already true at `07ea6b1f` (2026-08-31), the build the V100 report used, and the test never called `reset_runtime_state`.

GB10 (sm_121, driver 580.178.04, `MLX_ENABLE_TF32=0`) reproduces the reported symptom with the old loop. The first divergence is at the prefill, with 10 distinct logits per step over 10 iterations and iteration 0 unique, on gemma-4-12b-it-4bit, gemma-4-e2b-it-4bit and gemma-3-4b-it-4bit. `MLXCEL_SDPA_DETERMINISTIC=1`, `MLX_USE_CUDA_GRAPHS=0`, the fused-kernel switches off and attention chunking off all left it unchanged. llama-3.2-1b-4bit, qwen3-0.6b-4bit and qwen2.5-0.5b-4bit passed. With a reset before each iteration, all three Gemma checkpoints, gemma-4-26b-a4b-it-4bit and gemma-4-12b with cuDNN SDPA off pass. This also fits the report's own evidence: iteration 0 hashed the same on both builds, and greedy `generate`, one process per run, was reproducible.

### 2.2 No race found in `qmm_naive`

Code reading found no race in the `qmm_naive` mainloop. It has one shared-memory buffer with a barrier on each side of the store, prefetches into registers, writes the epilogue from registers straight to global memory, and uses `UniversalCopy` (no cp.async), with no atomics and no split-K. The #910 mechanism has nothing to act on. The k-residue path is unreachable with `transpose=true`. The Volta `SM70_8x8x4` atom (#1678, added after the report) was not run.

### 2.3 The gate fails loudly

A plain run reports the real-model test as `ignored` instead of a false `ok`. Under `MLXCEL_REQUIRE_MODELS=1` it panics when neither `llama-3.2-1b-instruct-4bit` nor `llama-3.2-1b-4bit` resolves, independent of the working directory. An `MLXCEL_TEST_QMM_MODEL_DIR` override that does not exist always fails, with or without the variable. The make target sets `MLX_ENABLE_TF32=0` because the `ctor` pin is `cfg(test)` in the lib crates and absent from integration test binaries.

### 2.4 Kernel tests without a checkpoint

Every Llama-3.2-1B projection has N % 128 == 0, so that model reaches only `qmm_sm80`. Four synthetic tests cover the other paths at M in {8, 17, 64, 128, 333}, require byte-identical output, quantize weights in the activation dtype and assert the output dtype. Paths were traced with `MLXCEL_TRACE_ARCH` / `MLXCEL_TRACE_QMM_TILE`:

| Arm | Traced kernel |
|---|---|
| bf16, N=256 | `qmm_sm80` |
| f16, N=200 | `qmm_naive`, `MmaPath::Sm80`, `half_t` |
| f32, N=256 | `qmm_naive`, `MmaPath::Sm80`, `float`; `make_tiled_mma<Sm80, float>` falls back to `UniversalFMA`, so this is the scalar arm despite the label |
| f16, transpose=false, K=200 | `qmm_naive` k-residue |

`MLXCEL_QMM_FORCE_SCALAR` only changes tile selection on sm_80+; the MMA path is chosen from the compute capability alone.

### 2.5 Changes during review

- Doc corrections: the f32 kernel test traces as `MmaPath::Sm80` while running `UniversalFMA`; the TF32 reason for the make target; the full `qmm_sm80` predicate; `CONTRIBUTING.md` mentions the target.
- `repo_model_dir_any` asserts a non-empty name list instead of falling back to the models root.

## 3. Verification

All measured on GB10 (sm_121, `MLX_ENABLE_TF32=0`).

| Check | Result |
|---|---|
| Plain run | 4 passed, 1 ignored; a bogus override fails with and without `MLXCEL_REQUIRE_MODELS=1` |
| Model test, llama-3.2-1b-4bit (`qmm_sm80`) | 10 iterations, 1 distinct per step |
| Kernel arms | 20 shapes, 1 distinct output over 10 calls each |
| Old loop (no reset), gemma-4-12b, gemma-4-e2b, gemma-3-4b | 10 distinct logits per step |
| With the reset, same three plus gemma-4-26b-a4b and gemma-4-12b with cuDNN SDPA off | Pass |
| Final tree, `--include-ignored --test-threads=1` under `gpu-lock` with `MLXCEL_REQUIRE_MODELS=1` (the make target's flags) | 5 passed |
| `cargo fmt --all -- --check` | Clean |
| clippy `-D warnings` on `cuda_qmm_determinism` and `sampling_top_k_above_vocab_e2e` test crates | Clean; run as `clippy-driver` on cargo's captured rustc invocation because `cargo clippy` re-ran the MLX CUDA CMake build on a loaded host |

## 4. Unverified and Open Items

- **Not run:** anything on sm_70, including the Volta `SM70_8x8x4` atom. This is why #1558 stays open.
- **Not run:** `make verify-test-cuda-qmm-determinism` invoked through make; its exact flags were run against the test binary.
- **Follow-up:** the target is not wired into `nightly-verify.yml`.

## 5. Learning Points

- A test that returns early when its input is missing reports `ok` for a check that never ran. Make the skip visible (`#[ignore]`) and give CI a switch that turns a missing input into a failure.
- Models that hold K/V in model-owned state ignore the "fresh caches" a harness builds. A repeat-run determinism test must reset runtime state explicitly, or iteration N measures a different sequence than iteration 0.
