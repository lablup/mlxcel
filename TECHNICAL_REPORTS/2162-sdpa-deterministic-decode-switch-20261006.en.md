# Technical Report: PR #2162 - Add MLXCEL_SDPA_DETERMINISTIC for reproducible decode

**Date**: 2026-10-06

**Status**: Implemented and verified on GB10; pending merge.

**Languages**: C++ (MLX CUDA source overlay), Rust (integration tests), Markdown (docs)

**Risk Level**: Low. With the switch unset, the dispatch is identical to `main`. With it set, decode takes an existing MLX kernel, and the one new failure path (no stream-K-off plan builds) falls back to the default selection instead of aborting.

## Executive Summary

Qwen3 temp-0 output on GB10 changed from run to run, and even between two repeats in one process, once the context passed 256 tokens (#2128). From 256 KV positions, MLX sends a one-row decode SDPA to cuDNN. The engine cuDNN's heuristics pick for the head_dim 128 graph does not round the same way on every call: about 1 call in 100 at 300 keys and 22 percent of calls at 1500. That engine is also the fast long-context decode, so by maintainer decision the default stays as it is. A new opt-in switch, `MLXCEL_SDPA_DETERMINISTIC=1`, routes decode to MLX's `sdpa_vector` kernel, and with it every model tested gave byte-identical output across processes.

## 1. Problem Statement

The project's verification practice compares greedy outputs byte for byte, before and after a change. PR #2117 could not do that for Qwen3-1.7B because the unchanged binary's output differed between three runs. Two earlier GB10 nondeterminism reports had turned out to be real kernel defects, a `cp.async` race (#910) and an RMSNorm overlay reading past shared memory (#831), so this one needed a root cause rather than a shrug.

Measured on the release CLI (800 tokens, temp 0): Qwen3-1.7B gave 4 distinct outputs in 4 runs and Qwen3-0.6B gave 2 in 3; Llama-3.2-1B and Gemma-3-1B were byte-identical. `MLX_USE_CUDA_GRAPHS=0` changed nothing.

## 2. Root Cause

Each step narrowed the search space before any kernel was touched:

1. **In-process, at a fixed position.** A probe that repeats the real greedy decode in one process and hashes the logits of every step showed prefill and the first 224 decode steps bit-identical, then divergence between steps 225 and 247 in every comparison. With a 34-token prompt, that is just past 256 cache positions.
2. **The attention dispatch.** `MLX_CUDA_USE_CUDNN_SDPA=0` removed it (6 sequences of 400 steps, bit-identical); `MLXCEL_KV_INPLACE_WRITE=0` did not. In the overlay, `use_cudnn_for_decoding` sends a one-row SDPA to cuDNN once `k.shape(2) >= 256` and k and v are slices of a cache buffer whose extent is a multiple of 256, unslicing them to the buffer and passing the true lengths in a padding mask. Gemma-3 never reaches it (cuDNN SDPA takes head_dim ≤ 128).
3. **The kernel in isolation.** A model-free probe repeating one SDPA call (q `[1,16,1,128]`, k and v 300 rows of a 512-row buffer) gave 2 distinct outputs in 4000 calls, differing by at most 7.6e-6. That is ULP level, so summation order rather than wrong data.
4. **The engine.** cuDNN offered two candidates for this graph. The first, `eng8_k40=0_k41=2_k24=2_k38=1_k27=0`, carries knob 38 = `CUDNN_KNOB_TYPE_STREAM_K` and reproduced the defect; forcing the second, `eng8_k40=2_k41=1_k24=1_k38=0_k27=0`, gave 1 output in 4000. `deselect_numeric_notes({NONDETERMINISTIC})` changed nothing, so cuDNN does not tag the engine.

The two candidates also differ in tile and kernel-config knobs, so knob 38 is the measured correlate, not an isolated cause. A contributor's analysis on #2128 (Jaeyeong-CHOI, with a standalone cuDNN reproducer on 9.13.0 and 9.24.1) goes further. It locates the order dependence in an unordered cross-warp FP32 reduction of the softmax denominator through a shared-memory CAS loop, and confirms it by fixing that order inside the kernel.

## 3. Technical Decisions

### Opt-in rather than default

Two deterministic options were measured against the default on GB10 (decode tok/s, median of 3 interleaved runs, quiet host):

| Model | Keys | cuDNN, stream-K barred | `sdpa_vector` |
|---|---|---|---|
| Qwen3-1.7B | ~400 / 2826 / 7366 / 15653 | 0.99 / 0.92 / 0.90 / 0.88 | 1.01 / 1.01 / 0.96 / 1.00 |
| Qwen3-4B | ~800 / 2826 / 7366 / 15653 | 0.97 / 0.94 / 0.92 / 0.87 | 1.00 / 0.91 / 0.97 / 0.94 |
| Llama-3.2-1B | ~870 / 2717 / 7189 / 15122 | 0.99 / 0.89 / 0.77 / 0.67 | 1.01 / 0.95 / 0.90 / 0.87 |

The first version of the branch barred stream-K and stayed on cuDNN. With a single query row, the key axis is the decode's only parallelism, and without stream-K cuDNN loses up to a third of its throughput. `sdpa_vector` splits the key axis into fixed blocks and reduces them in a fixed order, and costs less, but still up to 13 percent at 15K keys. The choice between reproducibility and long-context speed was put to the maintainer, who chose to keep the fast path as the default and make determinism opt-in.

### What the switch does

`sdpa_deterministic()` reads `MLXCEL_SDPA_DETERMINISTIC` once (MLX's `env::get_var`, so only a non-zero integer enables it). With it on:

- `use_cudnn_for_decoding` returns false. It answers the same at graph-build time and at eval time, so a one-row decode always lands on `sdpa_vector` (or the ops fallback for unsupported shapes) and never builds a cuDNN plan.
- As a precaution, `build_sdpa_graph` bars stream-K candidates on the other forward graphs (prefill, a bucketed verify block). Those graphs were not shown to vary. Barring them measured within noise on prefill throughput.

### Failing safe

`select_behavior_notes` and `check_support` can rule out the remaining stream-K-off candidates, and the review found that the process would then abort on `CHECK_CUDNN_ERROR`. The bar therefore builds through `build_sdpa_graph_with(bar_stream_k, ...)`. A failed build throws a dedicated `StreamKOffUnbuildable`, and only that exception falls back to a fresh graph built with the default selection, because barred candidates stay barred in the graph that failed. A one-time warning names the switch whenever a graph keeps a stream-K engine. Genuine cuDNN refusals keep their existing path, including the #1820 bucketing fallback.

### Tests in their own binary

The switch is latched in a C++ static, so the tests set it before `main` with `ctor` (the repository's existing pattern for `MLX_ENABLE_TF32`). They live in `tests/cuda_sdpa_determinism.rs` so that `tests/cuda_qmm_determinism.rs` keeps exercising the default dispatch for #910.

## 4. Validation

- `tests/cuda_sdpa_determinism.rs` passes 3 of 3 with the switch: decode SDPA at 300 of 512 keys and at 1500 of 2048 keys (the two-pass kernel), and Qwen3-0.6B decoding 400 steps past 256 positions. With `MLXCEL_SDPA_DETERMINISTIC=0` it fails 3 of 3 (11 and 446 of 2000 calls differ; logits diverge at step 204). Neither SDPA case needs a checkpoint, so the CUDA test gate runs them.
- **CLI, 5 separate processes, CUDA graphs on and off:** with the switch, Qwen3-0.6B, 1.7B and 4B, Llama-3.2-1B and Gemma-3-1B each gave one output. A 7366-token prompt was identical over 3 runs at 114.3 tok/s, against 119.0 for the default; 119.0 also matches the pre-change figure.
- **Server:** `mlxcel-server`'s pool-backed decode runs mlxcel's paged-attention kernels, not MLX SDPA. One greedy request gave the same output across restarts in both modes.

## 5. Residual Risks and What Was Not Verified

- **The default path is still nondeterministic for affected models**, by decision. Parity and determinism work must set the switch.
- **Shapes are sampled, not covered:** head_dim 128 with 16/8 and 32/8 heads, keys up to 16K, three dense 4-bit model families. Other heads, dtypes, cuDNN versions and GPUs may pick different engines.
- **The prefill bar is a precaution.** Prefill determinism was observed (identical prefill hashes) but not proven for every shape.
- **A server repeat served from the prompt-prefix cache differs from the first, uncached run**, in both modes. That is a separate path difference and is untracked.
- **The contributor reports Llama variability** that was not reproduced here.
- The switch accepts integers only; `true` silently reads as off (documented).

## 6. Learning Points

- **Localize in-process before suspecting the process.** The step-level logits hash turned "differs run to run" into "differs from step 225", and the position pointed straight at the 256-key dispatch boundary.
- **A library's determinism metadata is not evidence.** cuDNN offered the nondeterministic engine first and did not flag it; only repeated identical calls showed it.
- **Price every deterministic option before choosing one.** The obvious fix (stay on cuDNN, bar the engine) was the expensive one.
- **Read the issue thread before concluding.** A contributor had already traced the mechanism inside the kernel, and the documentation needed to say so.

## 7. Related

- Issue #2128.
- PR #2117 (where the nondeterminism blocked a parity check).
- #910 and #831 (earlier GB10 determinism defects).
- #1558 (making the determinism test fail instead of skip when its checkpoint is missing).
- #1820 and #1799 (SDPA plan-cache work in the same overlay).
- `docs/upstream/mlx-cuda-sdpa-stream-k-nondeterminism.md`.
