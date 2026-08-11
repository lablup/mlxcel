# Technical Report: PR #1101 - feat(models): add Meta Muse Glimmer VLM support

**Date**: 2026-08-12
**Status**: Completed
**Languages**: Rust, C++, Markdown
**Risk Level**: Medium

---

## Executive Summary

PR #1101 adds first-class Meta Muse Glimmer 30B support to mlxcel for the pinned dense BF16 checkpoint. It integrates the mixed-cache text decoder, vision preprocessing and fusion, CLI and continuous-batching server routes, Muse recipient/reasoning channels, and bounded ATEM tool calls while rejecting unsupported execution modes at startup.

The final review found one stale CLI help assertion after the documented GB10 qualification replaced the earlier pending-qualification message. Commit `d4ba28ac` aligns that contract test; no blocking correctness, security, or performance issue remains in the Muse scope.

---

## 1. Problem Statement

### 1.1 Background

Muse Glimmer is a 30B vision-language model whose released checkpoint combines a 52-layer text stack, a 50-layer vision tower, mixed sliding/full attention, a checkpoint-specific multimodal prompt layout, recipient-oriented reasoning channels, and ATEM tool calls. Existing generic VLM paths did not encode these contracts.

### 1.2 Existing Limitations

- The model type and published weight namespaces were not recognized.
- The text stack needed model-owned per-sequence state because sliding layers rotate a 2,048-token window while full-attention layers continue growing.
- Image markers had to expand to the exact merged visual-token count and preserve multi-image order.
- Streaming APIs needed to suppress Muse reasoning and ATEM structure from visible content without losing token-position accounting.
- Quantization, video, adapters, speculative decoding, TP/PP, XLA, and distributed modes were not qualified for this baseline and had to fail closed.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood without the change |
|------|--------|-------------------------------|
| Incorrect mixed-cache state crosses requests or corrupts long prompts | High | High |
| Placeholder/feature mismatch silently associates the wrong image | High | Medium |
| Reasoning or tool payload leaks into streamed content | High | Medium |
| Unsupported modes start and fail late inside native kernels | High | Medium |

---

## 2. Technical Review

### 2.1 Correctness and Security

- Weight loading classifies all 1,436 tensors in the pinned index and rejects unknown or quantization sidecar roots.
- Configuration validation pins the layer schedule, vision geometry, RoPE behavior, dense BF16 baseline, and unsupported capabilities.
- Image preparation rejects marker/cardinality and feature-row mismatches rather than falling back to text-only generation.
- Model-owned sequence IDs isolate mixed-cache state across admission, batching, release, reset, snapshot, and restore.
- ATEM parsing caps input size, calls, parameters, names, and argument bytes; tool allowlists are applied before route responses are built.
- The streaming filter tracks ATEM nesting, handles byte-split delimiters and malformed EOF, and keeps reasoning/tool markup out of visible deltas.

No authentication or persistent-data boundary changed. No new external dependency was introduced.

### 2.2 Performance and Memory

The baseline intentionally targets the 59,553,253,376-byte BF16 checkpoint on a GB10-class 128 GB unified-memory host. Sliding layers retain a bounded 2,048-token rotating cache while full-attention layers preserve the long-context contract.

The 2026-08-11 real-checkpoint gate recorded:

| Scenario | Result |
|----------|--------|
| Greedy text decode | 4.25 tokens/s |
| 2,204-token long prompt prefill | 46.47 tokens/s |
| Single and two-image prompts | Grounded the orange test fixture |
| Scheduler | Isolated answers at parallelism 1 and 2 |
| Cold two-image concurrency | At most 59.608 GiB decrease in system `MemAvailable`; 4.136 GiB process `VmHWM` |

CUDA allocator counters are unavailable on this GB10 backend, so the report preserves the OS-level memory measurements rather than treating allocator zeroes as evidence.

### 2.3 Compatibility

- Qualified: Linux/aarch64, NVIDIA GB10, CUDA 13.0, driver 580.173.02.
- Supported routes: CLI, OpenAI Chat Completions, Responses, Anthropic-compatible APIs, streaming, text, single image, multi-image, and ATEM replay.
- Explicitly unsupported: video, quantized weights, Turbo/INT8 KV, DFlash/speculative decoding, LoRA/adapters, TP, PP, XLA/IREE/OpenXLA, and distributed/disaggregated serving.
- Apple Silicon/Metal remains unqualified for this checkpoint.

---

## 3. Technical Decisions

### 3.1 Model-Owned Mixed Cache

**Decision:** Keep Muse sequence state inside the model and address it with scheduler sequence IDs.

**Rationale:** A generic homogeneous cache cannot represent the checkpoint's alternating sliding and full-attention layers. Model ownership preserves the exact schedule and makes release/reset semantics explicit.

**Trade-off:** Generic paged/disaggregated cache paths are disabled until they can preserve the same mixed-state contract.

### 3.2 Strict Multimodal Cardinality

**Decision:** Expand each checkpoint marker to the computed visual-token count and reject any marker, grid, or feature-row mismatch.

**Rationale:** Silent truncation or text-only fallback could associate features with the wrong image. Failing before generation keeps multi-image ordering auditable.

### 3.3 Bounded ATEM Parsing and Streaming Suppression

**Decision:** Use a dedicated bounded parser plus a depth-aware streaming filter instead of adapting an existing JSON tool-call format.

**Rationale:** Muse emits attribute-bearing XML-like tags and recipient channels. The dedicated path can preserve typed parameter values, parallel-call order, allowlists, malformed-output behavior, and API-specific streaming events.

### 3.4 Fail-Closed Baseline

**Decision:** Reject every unqualified feature at CLI/server startup.

**Rationale:** The dense BF16 path has real-checkpoint evidence; the rejected modes do not. Early actionable errors are safer than late kernel failures or silently incorrect output.

---

## 4. Implementation Overview

```text
Checkpoint config/index
        |
        +--> Muse text decoder --> per-sequence mixed rotating/full KV state
        |
Images --> Muse processor --> 50-layer vision tower --> 2x2 pixel shuffle
                                                     --> adapter/projection
        |                                                     |
Prompt template --> exact patch-marker expansion -------------+
        |
CLI / continuous scheduler --> generation --> recipient/reasoning split
                                           --> bounded ATEM parser/filter
                                           --> Chat / Responses / Anthropic events
```

The implementation also adds model detection and metadata, checkpoint fixtures, generation defaults, startup guards, scheduler admission coverage, API round-trip tests, documentation, and CLI support descriptions.

---

## 5. Learning Points

### 5.1 Cache Shape Is Part of the Model Architecture

Mixed sliding/full attention is not only a memory optimization. It changes state lifetime and therefore must be represented in batching, sequence identity, snapshot, restore, and long-context tests.

### 5.2 Stream Filtering Must Preserve Position Accounting

Suppressing structural tokens is insufficient if the scheduler loses track of consumed token positions. The Muse filter records suppressed and consumed positions across fragmented delimiters so streaming and non-streaming results remain equivalent.

### 5.3 Real-Checkpoint Qualification Must Remain Fail-Closed

Synthetic tests establish shape and invariant coverage, but large-model readiness also needs a pinned revision, hardware, memory, throughput, route, long-context, tool, and concurrency evidence set. Help text and documentation tests must be updated together when qualification status changes.

---

## 6. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 85 |
| Lines added | 11,437 |
| Lines deleted | 84 |
| Added Rust test attributes | 127 |

### Major Areas

| Area | Summary |
|------|---------|
| Model/runtime | Detection, configuration, weight loading, 52-layer decoder, mixed caches |
| Vision | Processor, layout/position logic, 50-layer tower, fusion and ordered scatter |
| Serving | Scheduler admission, expanded usage, startup guards, three API families |
| Tools/streaming | Muse recipients, reasoning split, bounded ATEM parsing and replay |
| Documentation | Supported-model contract, adding-model guidance, GB10 qualification |
| Review correction | Updated stale top-level help assertion to the qualified GB10 message |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `38307107` | feat | add Muse Glimmer 30B support |
| `7fc1bdd0` | fix | harden Muse streaming and slot admission |
| `d4ba28ac` | test | align Muse qualification help assertion |

Related issue: #1100.

---

## 7. Validation and Follow-up

### Passed

- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets --features cuda -- -D warnings`
- Focused CLI qualification test: 1 passed
- Muse regression set on host GPU with the pinned checkpoint: 95 passed
- ATEM regression set: 37 passed
- PR-hosted cheap gates before the final push passed; the final push re-queued them
- Earlier real-checkpoint gates covered text, single image, multi-image, long prompt, ATEM replay, streaming, and scheduler parallelism 1/2

### Repository-Wide Gate State

The authoritative CUDA command completed with `--no-fail-fast`. The `mlxcel` library target reported 5,528 passed, 5 failed, and 113 ignored; all later workspace targets and doctests completed without another failure. The five failures reproduce individually and their implementation/test paths are unchanged from `origin/main`:

- `execution::memory_estimate::tests::resolve_block_budget_explicit_bytes_floors_to_block_count`
- `models::bailing_moe_linear::tests::chunked_gla_matches_the_sequential_recurrence`
- `models::deepseek_v2::tests::absorbed_mla_attention_matches_the_decompressed_block_step_for_step`
- `models::florence2::florence2_tests::incremental_decode_matches_full_sequence`
- `models::klear::tests::the_prefill_is_causal_without_being_handed_a_mask`

These are repository-wide pre-existing CUDA gate failures, not Muse regressions. They should be repaired in separate focused work so their memory-budget arithmetic and numerical tolerances remain reviewable.

### Remaining Qualification Boundary

- Apple Silicon/Metal validation is not claimed.
- Unsupported baseline modes remain intentionally blocked.
