# Technical Report: PR #2912 - feat(moe): stream qwen4_exp routed experts from the SSD

**Date**: 2026-11-10
**Reviewer**: -
**Status**: Needs Follow-up
**Languages**: Rust
**Risk Level**: Medium

---

## Executive Summary

`qwen4_exp` can now keep its routed experts on the SSD and load them on demand into a fixed per-layer slot arena, with everything else resident. The mode is off unless routed experts exceed 60% of the Metal working set, so a machine that can hold the model keeps the existing path. Output matched resident execution on a 512 GB M3 Ultra with forced 24 GB and 8 GB arenas, but the target case, a Mac with less memory than the checkpoint, has not been run.

---

## 1. Problem Statement

### 1.1 Background

The Qwen3.8-Flash-Next checkpoint spends most of its bytes on 512 routed experts per MoE layer, and a token only touches its top-k experts. Holding every expert resident requires more memory than most Macs have. Issue #2882 asks for running the model from the SSD where it cannot stay resident.

### 1.2 Existing Issues

- **All or nothing**: experts were resident tensors, so a checkpoint larger than the working set could not run usefully.
- **Router result is a graph node**: choosing what to read requires the top-k on the host, which forces an `eval` inside each layer of the forward pass.
- **Numerics depend on batching**: the resident sorted kernel groups rows by expert, and the kernel and tile choice follow from `E = 512`. Any change in grouping can change low bits.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Output differs from resident execution without any error | High | Medium, seen in the first version |
| A slot is rewritten while a pending command still reads it | High | Low, prevented by the per-layer sync |
| A hostile checkpoint drives file reads or allocations through unchecked header fields | High | Medium before the security fixes |

---

## 2. Technical Review

### 2.1 Security

The checkpoint and manifest are untrusted input to the new reader. Security review reported 4 HIGH and 8 MEDIUM findings, fixed in `f37da5c4`.

| Issue | Severity | Status |
|-------|----------|--------|
| `row_bytes` and shape not tied together, integer wrap in sizes | High | Fixed: checked `out * in * element_size`, non-integer shapes refused |
| Shard paths and manifest could leave the model directory | High | Fixed: all go through `resolve_contained` |
| Tensor end offset not checked against the shard | High | Fixed: checked arithmetic against the opened handle's length |
| Duplicate `(layer, suffix)`, zero or excessive layers | Medium | Fixed: refused, `MAX_LAYERS` is 4096 |
| `prepare-expert-stream` writing through existing files | Medium | Fixed: unlink, `create_new`, manifest written last |
| Panic on an I/O error while streaming | Low | Open |
| No free-space check in `prepare-expert-stream` | Low | Open |
| `shard_names` reads `model.safetensors.index.json` without `resolve_contained`, so a FIFO at that path blocks the load | Low | Open |

### 2.2 Performance

Measured with a sibling job on the host (load 9 to 13), OptiQ 2-bit, 128 tokens, four interleaved runs per arm. The per-layer sync arm evaluates the router top-k on resident weights, so it isolates synchronisation from I/O.

| Area | Resident | Per-layer sync | Change |
|------|----------|----------------|--------|
| Decode | 42.0 tok/s (23.8 ms per token) | 25.6 tok/s (39.1 ms per token) | About +15.3 ms per token, 1.64x |

About 0.32 ms per layer. The numbers come from a loaded host and are not a benchmark; see `docs/benchmark_results/versions/unreleased.md#expert-streaming-sync-cost-m3ultra-2026-11-10`. Streamed throughput with real misses has no paired measurement.

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: none. The default decision keeps resident execution on machines that fit the routed experts.
- **New Dependencies**: none.
- **Switches**: `MLXCEL_EXPERT_STREAM` (`1` or `0` overrides the automatic rule), `MLXCEL_EXPERT_ARENA_BYTES` (arena size).
- **New command**: `mlxcel prepare-expert-stream` writes the contiguous file, which is hash-checked and used automatically.

### 2.4 Code Quality

- `qwen4_exp` library tests: 199 passed, including the stream tests for slot map, LRU, batching, decision rule, hostile checkpoints and the contiguous writer round trip.
- `src/models/qwen4_exp_stream_source.rs` (589 lines) and `src/models/qwen4_exp_stream_tests.rs` (592 lines) are over the 500-line guideline.

---

## 3. Technical Decisions

### 3.1 Keep the MLX graph and sync once per layer

**Context:** The host must know each layer's top-k before it can read experts, and a slot must not change while a command reads it.

| Option | Pros | Cons |
|--------|------|------|
| Custom Metal handshake | Could lower the per-layer cost | Large native surface, no evidence yet that it pays off |
| **Chosen: `eval` of the top-k per layer in the MLX graph** | No new native code, the sync point also makes slot reuse safe | About 15.3 ms per token at a 100% hit rate |

**Rationale:** The mode engages only where the checkpoint cannot stay resident, so the alternative to a fixed per-token cost is not running.

**Trade-offs:** Decode speed under streaming is capped by the sync cost even with no misses.

### 3.2 Reproduce the resident kernel choice in prefill

**Context:** The first version split oversized prefill windows by token rows. Greedy output at an 8 GB arena diverged from resident on all five prompts, deterministically, while short prompts matched, which ruled out eviction.

**Decision:** Keep each expert's rows in one batch and plan batches so MLX picks the kernel and tile the resident `E = 512` tensor would. A window that cannot be planned this way logs a one-time warning that low bits may differ. The warning did not fire in any run.

### 3.3 Off by default on machines that fit

**Context:** Streaming trades decode speed for memory. The automatic rule engages only above 60% of the Metal working set. `On` fails with a reason on a non-streamable layout, while `Auto` falls back to resident.

---

## 4. Implementation Details

```
[Before]
router top-k -> gather_qmm over 512 resident experts

[After, when streaming]
router top-k -> eval to host -> read misses into LRU victims -> gather_qmm over slot indices
                                    ^ page-aligned host memory wrapped by MLX without a copy
```

Key modules:

- `src/models/qwen4_exp_stream.rs`: mode decision, `ExpertStream::resolve`, `check_streamable`.
- `src/models/qwen4_exp_stream_arena.rs`, `qwen4_exp_stream_slots.rs`: arena, slot map and LRU. Sizing uses `host_memory` and `AvailabilityGate` from #2883.
- `src/models/qwen4_exp_stream_source.rs`: checkpoint and contiguous file reading with the trust checks.
- `src/commands/prepare_expert_stream.rs`: the contiguous file writer.
- `src/server/routes/metrics.rs`: `mlxcel_expert_arena_*`.

A drafter configured while streaming below 96 GB logs a warning; it is not disabled.

---

## 7. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed (before this report) | 21 |
| Lines added | +2552 |
| Lines deleted | -6 |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `eb930a1d` | feat | stream qwen4_exp routed experts from the SSD |
| `b9f14d00` | fix | keep the resident sorted kernel on the expert arena |
| `986cb570` | fix | keep the expert arena stats line after the model is dropped |
| `f37da5c4` | fix | harden expert streaming against hostile checkpoints |
| `98759ec9` | docs | record qwen4_exp expert streaming in the changelog |

---

## 8. Follow-up Actions

### Required

- [ ] Run OptiQ 2-bit on a Mac with less memory than the checkpoint, with the external PLE table, and confirm no swap-outs. No such machine was available, and streaming was never combined with the external PLE table.
- [ ] Paired five-run medians for at least two arena sizes, with and without the contiguous file, and for 4K-token prefill, on an idle host.
- [ ] `prepare-expert-stream` on the real checkpoint. Only the unit test round trip has run.

### Monitoring Required

- `[Expert arena: ...]` hit rate and I/O, and `mlxcel_expert_arena_*`.
- The one-time window planning warning. If it fires, output may differ from resident in low bits.

### Future Improvements

- Not implemented from the issue: read-ahead, warm-start usage profile, prefill chunk size by available memory. Wiring arena buffers was not measured. Batched decode with streaming is untested.
- `preflight_resident_bytes` does not run `check_streamable`, so a layout that `auto` sends to the resident path is estimated at arena size and under-counted.

---

## Appendix

### A. Test Results

Forced streaming, 512 GB M3 Ultra, release build:

- Five prompts, 256 tokens: byte-identical to resident at 24 GB and 8 GB arenas.
- Logit trace after 4608 context tokens in one forward: 0 differing rows at widths 32 and 1, at printed logit precision, for both arenas.
- 3286-token prompt through default prefill chunking: identical at both arenas, but generation stopped after 16 tokens, so this is short evidence.
