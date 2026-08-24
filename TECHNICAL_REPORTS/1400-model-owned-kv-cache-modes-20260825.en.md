# Technical Report: PR #1400 - Apply KV Modes to Model-Owned Caches

**Date**: 2026-08-25
**Author**: mlxcel contributors
**Status**: Completed
**Languages**: Rust, Markdown
**Risk Level**: High

---

## Executive Summary

PR #1400 makes the resolved KV-cache policy reach model families that allocate heterogeneous sequence state internally. It also aligns CLI/server observability, cache statistics, snapshot refusal, and dense prompt-cache adoption with the mode that is actually active, eliminating a silent configuration no-op across Gemma, Qwen, LFM2, Llama 4, AFMoE, and related VLM wrappers.

---

## 1. Problem Statement

### 1.1 Background

The CLI generator and server cache pool already resolved `--kv-cache-mode`, Boundary-V, and `--kv-bits` into per-layer policies, but this policy was applied only to the external `Vec<KVCache>` passed through `LanguageModel::forward`. Model-owned families ignored that homogeneous slice and constructed their own attention caches with FP16 defaults because they also needed recurrent, convolutional, sliding-window, or other heterogeneous state.

### 1.2 Existing Issues

- **Silent no-op**: Operators could request Int8 or Turbo modes while model-owned families continued to allocate FP16 attention caches without an error.
- **Misleading observability**: Banners and statistics could describe the requested or legacy mode even when batch KV quantization selected another effective mode.
- **Unsafe reuse boundary**: Snapshot serializers that copied only ordinary keys and values could truncate quantization sidecars, and dense prompt-cache adoption did not reject per-layer KV-mode mismatches.
- **Incomplete family coverage**: The documented symmetric Turbo4 allowlist named Qwen families whose internal caches never consumed the resolved policy.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood before fix |
|------|--------|-----------------------|
| Production memory sizing differs from announced configuration | High | High |
| Prompt or snapshot reuse crosses incompatible KV representations | High | Medium |
| Operators tune performance from inaccurate cache statistics | Medium | High |

---

## 2. Change Summary

### 2.1 Policy Injection

`LanguageModel` now exposes hooks for installing and reading a resolved per-layer KV-mode table. A shared `KvCacheLayerModes` holder stores the table next to model-owned sequence state, while `LoadedModel` and applicable VLM wrappers delegate the hooks to their underlying text models.

Both execution surfaces inject policy after effective-mode resolution: `CxxGenerator` does so for offline generation, and the batch scheduler does so for server inference. Boundary-V and batch KV quantization continue to use their existing resolution logic, so this change extends one source of truth rather than introducing another requested-mode path.

### 2.2 Family Coverage

Gemma 3, AFMoE, Gemma 4, Qwen 3.5, Qwen3-Next, Bailing MoE Linear, LFM2, and Llama 4 regular attention constructors now select the configured mode by layer index. Recurrent, convolutional, gated-delta, and Llama 4 `ChunkedKVCache` state remain FP16; Llama 4 emits a one-time warning when a non-FP16 policy reaches an unsupported chunked layer.

### 2.3 Reuse Safety

Dense detached prompt-cache entries are compared against the resolved mode for every layer. A mismatch is rejected through the named `kv_mode_mismatch` reason and its metric instead of being adopted into a live scheduler with a different representation.

Model-owned exact-prefix snapshot paths now reject non-FP16 state unless they can preserve the required representation. The PR intentionally chooses explicit refusal over silently restoring only keys and values while losing Int8 or Turbo sidecars.

### 2.4 Observability

The CLI banner and server startup log report how many layers received the effective mode. `/v1/cache/stats` exposes `kv_cache_mode_effective`, and the review fix makes it prefer `BatchKvQuantConfig::base_mode()` when `--kv-bits` is enabled rather than incorrectly reporting the legacy FP16 field.

---

## 3. Technical Decisions

### 3.1 Inject a Resolved Table, Not Raw Flags

**Decision:** Store a vector of resolved modes indexed by model layer.

**Rationale:** A scalar raw flag cannot represent Boundary-V or per-layer batch KV policy and would reintroduce the requested-versus-effective divergence fixed immediately before this issue. The table also lets heterogeneous model wrappers apply the policy only to attention-bearing layers while leaving recurrent state unchanged.

**Trade-off:** Models now carry a small piece of configuration state and wrapper delegation must remain complete when new VLM families are added.

### 3.2 Refuse Unsupported Snapshots

**Decision:** Fail closed for non-FP16 model-owned snapshots instead of partially serializing them.

**Rationale:** A complete Turbo or Int8 snapshot requires packed tensors, norms, scales, seeds, offsets, and thresholds. Copying only ordinary keys and values creates a cache that appears valid but has a different representation. Explicit refusal preserves correctness while leaving full sidecar serialization for a future change.

**Trade-off:** Some non-FP16 exact-prefix cache hits fall back to cold prefill.

### 3.3 Keep Chunked KV State in FP16

**Decision:** Quantize only Llama 4 regular attention caches and warn for unsupported `ChunkedKVCache` layers.

**Rationale:** Chunked caches do not yet implement quantized sidecar storage. Partial, explicit application is safer than either silently claiming complete coverage or changing the chunked cache format in this issue.

---

## 4. Review and Quality Findings

### 4.1 Implementation Review

The implementation review found one HIGH issue: cache statistics still read the legacy effective-mode field when server-side batch KV quantization was active. Commit `01773044a` fixed the route to prefer the batch quantization base mode and added Int8, Turbo, and legacy fallback coverage.

### 4.2 Security and Performance Review

No unresolved CRITICAL or HIGH security/performance findings remained. The security review confirmed that the new reuse boundaries fail closed, the mode table is derived from validated runtime configuration, and no untrusted input is used to index beyond the resolved layer count.

### 4.3 Compatibility

- **Breaking changes**: None to CLI flags or public HTTP request formats.
- **New dependencies**: None.
- **Behavior change**: Existing non-FP16 flags now take effect for model-owned attention caches; unsupported model-owned snapshot reuse may cold-prefill instead of accepting incomplete state.

---

## 5. Validation

- `cargo test --workspace --profile test-fast --features metal,accelerate` passed after final review changes.
- `cargo clippy --workspace --all-targets -- -D warnings` passed.
- `cargo fmt --all -- --check` and `git diff --check` passed.
- Focused tests passed for generator mode forwarding, Qwen3-Next snapshot mismatch refusal, dense prompt-cache KV-mode rejection, and effective cache-stat reporting.
- Real Gemma 3 generation passed on Metal with both FP16 and Int8. Both banners reported `applied to 26 of 26 layers`, both outputs were fluent and finite, and the greedy token streams differed, demonstrating that Int8 reached the internal attention caches.

---

## 6. Change Statistics

| Item | Value |
|------|-------|
| Files changed | 26 |
| Lines added | 1,009 |
| Lines deleted | 204 |
| Implementation commits | 2 |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `b3536bff9` | fix | Apply KV modes to model-owned caches |
| `01773044a` | fix | Report batch KV mode in cache stats |

---

## 7. Follow-up Considerations

- Implement full quantized sidecar serialization for model-owned exact-prefix snapshots if cache-hit performance justifies the additional format surface.
- Add quantized representations for Llama 4 `ChunkedKVCache` and ring-sliding caches in separately scoped work.
- Monitor `mlxcel_prompt_cache_reject_total{reason="kv_mode_mismatch"}` to distinguish expected configuration changes from unexpected policy drift.
- Preserve delegation of the `LanguageModel` KV-mode hooks when introducing new VLM wrappers or model-owned families.

---

## References

- Issue #1330: model-owned KV-cache mode no-op and reuse safety requirements
- PR #1400: final implementation and review fix
- `docs/turbo-kv-cache.md`: supported modes, model-owned behavior, and current exceptions
