# PR #3408: Nemotron-H hybrid state carry

**Date**: 2026-11-25

**Status**: Partial; CUDA CLI/server and ROCm acceptance pending

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

The following acceptance evidence is still pending: CLI and concurrent-server parity, CUDA Mamba2 hardware regressions, and all ROCm gfx1151 gates. Remote CI was cancelled under the run's local-validation policy; cancellation is not passing CI evidence. The PR remains a draft and #3344 remains open at `status:review`.

## Learning Points

Hybrid model support needs one authoritative schedule across configuration, tensor names, checkpoint shapes, graph emission, and runtime state. Validating those surfaces independently can produce a graph that compiles but reads the wrong layer's weights or exposes only half of a recycled slot's state.

An internal single-sequence reference proves batch consistency but cannot prove architecture correctness. Token acceptance therefore uses a separately generated HF fp32 oracle, fixed checkpoint revision, and exact token IDs.

## Follow-up Actions

- [ ] Run CLI and four-concurrent-request server parity on CUDA.
- [ ] Run the Mamba2 single-sequence and state-carry regressions on CUDA hardware.
- [ ] Record the same real-model, batching, server, and Mamba2 evidence on ROCm gfx1151 at the exact tested PR SHA.
- [ ] Implement and validate `E` blocks in #3345 before enabling Nemotron-H MoE schedules.

## References

- Issue #3344
- Epic #564
- `spike/openxla/NEMOTRON_H_VALIDATION.md`
- `spike/openxla/nemotron_h_reference/`
