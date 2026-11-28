# PR #3408: Nemotron-H hybrid state carry

**Date**: 2026-11-25

**Status**: CUDA and ROCm acceptance passed

## Executive Summary

This PR adds native OpenXLA/IREE execution support for the Nemotron-H hybrid schedule: NoPE attention layers use compact KV rows, Mamba2 layers carry recurrent state, and relu2-only layers omit both mixers. The same schedule now controls configuration parsing, weight ordering, checkpoint validation, prefill, single-sequence decode, and continuous batching.

## Problem Statement

The recurrent engine previously assumed that every layer was either attention-only or Mamba2-only. Nemotron-H interleaves Mamba2 (`M`), attention (`*`), and feed-forward-only (`-`) blocks, with attention projection width and SSM inner width independent of the hidden width. Treating this layout as a dense KV model wastes state and misorders weights; treating it as a pure recurrent model loses attention state and cannot admit or recycle batched slots atomically.

The official `nvidia/Nemotron-H-4B-Base-8K` checkpoint also fixes the SSD chunk size at 256. Preserving that value matters for matching the reference reduction partition and its fp32 token trajectory.

## Change Summary

The emitter parses the hybrid schedule and builds independent compact indices for attention KV and recurrent state. NoPE attention uses the checkpoint's 128-wide heads even though its projected width is 4096 and hidden width is 3072. Mamba2 uses the checkpoint's 7168-wide inner state, grouped gated RMSNorm, and f32 recurrence. Feed-forward-only blocks emit the checkpoint's nongated `relu2` transform. MoE `E` blocks remain rejected until #3345 supplies the sigmoid router.

The runtime carries KV and SSM rows together and marks a slot valid only after all required transfers complete. Failed admits, malformed rows, and partial stores clear validity before reuse. Prompt caching and APC remain disabled for recurrent families. Checkpoint validation and weight-name mapping follow the same schedule used by graph emission.

Registered StableHLO goldens cover a synthetic `M*-M-` model with unequal hidden and attention widths, two KV heads, and two SSM groups. The IREE integration target covers single-reference parity, staggered admission, slot recycling, batch occupancy, allocation reuse, failed decode invalidation, failure after a partial KV store, and failure after complete KV plus the first SSM tensor.

## Technical Decisions

### Use separate compact indices for KV and recurrent state

A layer's original position remains the source of checkpoint names and residual ordering, while attention and SSM tensors are stored only for layers that need them. This avoids allocating KV for recurrent or feed-forward-only blocks and avoids allocating recurrent state for attention blocks.

### Commit recurrent state atomically with KV

Hybrid slots are invalidated before writes and become visible only after all KV and SSM transfers succeed. The test hook has distinct failure points after K and after KV plus the first SSM tensor, so both torn-store boundaries exercise recovery instead of relying only on code inspection.

### Preserve checkpoint numerical parameters

The parser accepts the canonical `chunk_size`, retains the legacy `mamba_chunk_size` fallback, rejects invalid values, and preserves the real checkpoint's value of 256. The synthetic fixture sets 128 explicitly so the correction does not rewrite established goldens.

## Validation

At commit `93ddebad382478540fdfc987fbca655744be941d`, all registered byte-exact goldens, five Nemotron-H structural tests, and 27 recurrent structural regressions passed. All five synthetic execution tests passed with IREE local-task and CUDA on NVIDIA GB10 with driver 580.178.04; CUDA GPU work ran under `gpu-lock`. Focused XLA-library clippy passed with `-D warnings`, and modified Rust files passed formatting.

An independent Transformers 5.12.1 CPU fp32 run loaded all 311 tensors from checkpoint revision `faba3b731ad7ea5781b9518ae75fb610a94affcf` without missing or unexpected-weight warnings and produced a committed 40-token oracle with provenance hashes. This establishes the reference artifact, not XLA agreement.

The real 4B CUDA check at the same commit matched all 40 committed HF fp32 oracle tokens using the original checkpoint, context 256, f32 execution, NVIDIA GB10, and driver 580.178.04. It exited successfully under `gpu-lock`; the full log is `/tmp/epic564-plan/3344-cuda-real-oracle.log`. The executable used the `test-fast` build profile, so this result establishes token correctness and makes no release-build performance claim.

The real-checkpoint CUDA batch gate also served eight requests at B=4 and matched all 170 generated tokens against per-request single-sequence references with context 256 and fp32. Its test-fast informational throughput was 4.974 tok/s batched versus 6.02 tok/s sequential (0.8x); this is not a release benchmark or a performance-improvement claim. Raw oracle and batch outputs are included under `spike/openxla/nemotron_h_reference/`.

The test-fast CLI matched all 40 HF oracle token IDs exactly. Four HTTP requests overlapped for 16.958 seconds and each returned 24 completion tokens whose text and counts matched a separately captured CLI reference. The server API does not expose token IDs, so HTTP acceptance establishes rendered-text and count parity. The successful server used `--ctx-size 2048`, B=4, explicit XLA capacity 256, and prompt cache/APC disabled; a prior `--ctx-size 1024` attempt was correctly rejected because its 256 tokens per slot fell below the 512-token minimum.

On CUDA, `xla_ssm_single_seq` passed six tests, `xla_ssm_state_carry` passed five tests, and the separate `xla_hybrid_nemotron_h` target passed five tests. The Nemotron-H prompt-cache exclusion unit test also passed. The validation binaries came from `93ddebad`; CLI/server evidence was captured from the documentation-only checkout `d7e8a040`. Raw evidence and binary hashes are included under `spike/openxla/nemotron_h_reference/`.

ROCm gfx1151 acceptance also passed at executable source `93ddebad382478540fdfc987fbca655744be941d`: independently regenerated HF oracle, IREE-HIP and CLI 40/40 token parity, B4/N8 170-token parity, four concurrent streaming HTTP text comparisons, five hybrid tests on HIP and local-task, eleven SSM probe cases per backend, and Mamba2 single-sequence/state-carry and real-checkpoint regressions. HTTP token IDs and usage counts were not collected on ROCm, so its HTTP evidence is rendered-text parity. All builds and device work used the ROCm guard; release runtime binaries were used without a performance claim. The [ROCm acceptance record](https://github.com/lablup/mlxcel/pull/3408#issuecomment-7138393314) identifies the commands and log directory. Remote CI was cancelled under the run's local-validation policy; cancellation is not passing CI evidence.

## Learning Points

Hybrid model support needs one authoritative schedule across configuration, tensor names, checkpoint shapes, graph emission, and runtime state. Validating those surfaces independently can produce a graph that compiles but reads the wrong layer's weights or exposes only half of a recycled slot's state.

An internal single-sequence reference proves batch consistency but cannot prove architecture correctness. Token acceptance therefore uses a separately generated HF fp32 oracle, fixed checkpoint revision, and exact token IDs.

## Follow-up Actions

- [x] Record real-model, batching, server, and Mamba2 evidence on ROCm gfx1151 at the exact tested source SHA ([acceptance record](https://github.com/lablup/mlxcel/pull/3408#issuecomment-7138393314)).
- [ ] Implement and validate `E` blocks in #3345 before enabling Nemotron-H MoE schedules.

## References

- Issue #3344
- Epic #564
- `spike/openxla/NEMOTRON_H_VALIDATION.md`
- `spike/openxla/nemotron_h_reference/`
