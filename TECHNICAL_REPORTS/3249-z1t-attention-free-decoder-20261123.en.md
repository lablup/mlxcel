# Technical Report: PR #3249 - Z1T-0 (z1t) attention-free AFT-conv decoder

**Date**: 2026-11-23
**Reviewer**: -
**Status**: PR open (closes #2364); verified on M3 Ultra against the vendored mlx-vlm z1t module
**Languages**: Rust
**Risk Level**: Low (new model family; shared code touched only in registries, memory classification and `Used by:` annotations)

---

## Executive Summary

This PR adds Extropic Z1T-0 (`model_type: z1t`), a 4-layer decoder with no attention. Each block mixes tokens with AFT-conv (a causal depthwise conv plus a causal cumulative pool), Dynamic Tanh is the only normalization, and every internal projection is a fixed fan-in-4 sparse linear wrapped in `tanh`. The model runs through `mlxcel generate`, `mlxcel-server` and `mlxcel arch`, keeps O(1) per-token decode state in `ModelOwnedSequenceState` as RWKV-7 does, and joins native batched decode. On `models/z1t-0-mlx` a teacher-forced trace over 512 tokens disagrees with mlx-vlm on 0 of 113 decided positions with a maximum logit delta of 0.0000, while an arm that drops the cumulative pool disagrees on 62 of 113 and fails the gate. Greedy decode runs at about 226 tok/s.

---

## 1. Problem Statement

### 1.1 Background

Z1T-0 was ported to mlx-vlm in Blaizzy/mlx-vlm#2166 and published as `mlx-community/Z1T-0-mlx` (fp32, 4.97 GB, GPT-2 BPE tokenizer). mlxcel had no `z1t` detection arm and nothing structurally similar: there is no K/V tensor to cache, and the per-sequence state is the running pool sums plus the last `ksize - 1` conv inputs.

### 1.2 Existing Issues

- **No model family**: `detect_model_type` returned nothing for `z1t`, so the checkpoint could not load.
- **No sparse-linear primitive**: the projections store `weight [out, k]`, `bias [out]` and an int32 `indices [out, k]` table, and compute `y[o] = sum_k W[o,k] * x[indices[o,k]] + b[o]`. No existing layer implements this.
- **CI rules for new families**: since #3090 a new `impl LanguageModel` must implement `last_row_logits` (or be listed), and must either join batched decode or be pinned as delegating or per-row in `batched_decode_coverage_tests.rs`.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Streaming state drifts from the flat forward (window or pool off by one) | High: silent quality loss on every generated token | Medium for a hand port; covered by the flat-vs-streaming test and the trace |
| A malformed checkpoint drives an out-of-bounds GPU gather or a huge allocation | High: crash or out-of-memory at load or first forward | Low, but checkpoint configs are untrusted input |
| Memory estimate treats the model as a 1.5 GiB-per-context KV model | Medium: wrong preflight refusals and inflated prompt-cache budgets | Was certain before `e36757d6` |

---

## 2. Technical Review

### 2.1 Security

The implementation and security reviews treated the checkpoint's `config.json` and tensors as untrusted.

| Issue | Severity | Status |
|-------|----------|--------|
| `max_position_embeddings` near 2^32 truncated through `as i32`, so a 256-row `pe` table passed the shape check while positions were clamped only at the real value, leading to an unchecked `take` past the table | High | Fixed (`44e0247f`): `(max_positions + 1) * hidden` must fit in i32 elements |
| `aft_ksize` had no upper bound: 2^32 truncated to an empty `w_conv` and panicked on the first forward; a large value made every sequence's conv window enormous | High | Fixed: capped at `MAX_AFT_KSIZE = 1024` (Z1T-0 uses 4) |
| Sparse gather chunking measured only `L * out` (upstream's measure), ignoring batch size and fan-in, so a large fan-in allocated `8M * k` elements per projection | High | Fixed: measure is `B * L * out * k`, saturating, capped at 32M elements, which equals upstream's chunk size for Z1T-0 (B = 1, k = 4) |
| The lazy running sums kept each layer's full-length `ekv`/`ek` alive after a long prefill until the first decode step | Medium | Fixed: scheduled with `schedule_prefill_state`; output bit-identical |
| Index tables out of range | High if missed | Range-checked at load since the first commit |
| Dense-path weights, embedding and head not shape-checked | Medium | Fixed (`5ac895ad`): `expect_dense_shape` checks `[out, in]` when no `.scales` tensor is present |
| An unknown sequence id panics | Low | Open; same pattern as RWKV-7 |

### 2.2 Performance

| Area | Before | After | Change |
|------|--------|-------|--------|
| Greedy decode, `z1t-0-mlx`, M3 Ultra | 2.72 tok/s (first build) | 225.7 tok/s | 83x |
| Same prompt, vendored mlx-vlm | 139.6 tok/s | 139.6 tok/s | reference |
| `mlxcel inspect` KV estimate | 1.50 GiB at full context (196608 B/token) | 0 (pure-SSM class) | no context-proportional state |

The first build spent 93 ms per layer per decode step in MLX's strided `cumsum` over a time axis of length 1. A one-token chunk is its own inclusive cumsum, so `inclusive_cumsum_time` now returns the input when `len == 1`. This is exact, and the decode output is unchanged.

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: none. Other families are untouched; the shared edits are registry entries, the `z1t` arm in `kv_arch::classify`, the APC opt-out lists and `Used by:` annotations on `materialized_state_tail` and `schedule_prefill_state`.
- **New Dependencies**: none.
- **Weights**: kept in fp32. The loader accepts both the raw JAX key names the checkpoint ships (`embedding.*`, `blocks.N.*`, `norm.*`, `clf.*`, `pe.pe`) and upstream's sanitized names.

### 2.4 Code Quality

- **Tests**: 13 tests in `src/models/z1t_tests.rs` plus one in `hybrid_ssm_tests.rs` and one in `kv_arch`.
- **Structure**: the family is split into `z1t.rs`, `z1t_config.rs`, `z1t_layers.rs`, `z1t_batch.rs` and `z1t_tests.rs`, each under 500 lines, following `docs/code-guidelines.md`.
- **Technical debt**: RWKV-7 has the same KV-estimate gap that `e36757d6` fixed for Z1T. It was left alone as out of scope.

---

## 3. Technical Decisions

### 3.1 Model-owned state, not `KVCache`

**Context:** a layer's state is `cum_eKV [B, heads, D/heads]`, `cum_eK [B, heads]` and the last `ksize - 1` rows of `eKV` and `eK`. Its size is fixed, about 1 MB per sequence on Z1T-0.

| Option | Pros | Cons |
|--------|------|------|
| Store the state in `KVCache` tensors | Reuses the cache plumbing | Misrepresents the state as context-proportional; offsets and trimming do not fit a running sum |
| **Chosen: `ModelOwnedSequenceState<Vec<Z1tLayerState>>`** | Same contract as RWKV-7 and Mamba; per-sequence lifecycle already handled by the server | Prefix caching of the state needs separate work (out of scope; APC is opted out) |

### 3.2 Join native batched decode

**Context:** the coverage test requires every model-owned family to either batch natively or declare a fallback.

**Chosen:** follow RWKV-7. `z1t_batch.rs` stacks each row's layer state along the batch axis using `recurrent_batch` helpers, runs one forward with per-row position offsets, and splits the state back. Mixed fresh and initialized rows, multi-token calls and masked calls fall back to per-row forwarding. A B=4 trace matches B=1 with 0 of 512 top-1 disagreements and a maximum logit delta of 0.0000.

**Trade-off:** every step re-stacks and copies the per-row state. At about 1 MB per sequence this is small, and the other recurrent families do the same.

### 3.3 Chunk the sparse gather by total elements

**Context:** the gather materializes `[B, L, out, k]` before the reduction. For `mlp.proj1` (out = 49152, k = 4) a 2048-token prefill is 400M f32 elements.

**Chosen:** split along the sequence axis so that `B * L * out * k` stays under 32M elements. Splitting along the sequence axis is exact because each output row depends only on its own input row. For Z1T-0 this reproduces upstream's 8M `L * out` chunk; for wider fan-ins or batches it stays bounded where upstream's measure does not.

### 3.4 Position table zero-extension

The shipped `pe [256, 12288]` table is numerically zero, but the code follows upstream: rows past 256 read a zero row (`pe_ext` has one extra zero row and positions clamp to it). The original JAX `greedy_generate` slides a 256-token window instead; this port follows mlx-vlm's streaming semantics, which is what the trace compares against.

---

## 4. Implementation Details

### 4.1 Forward pass per block

```
x ─ DyT(norm1) ─ qkv = tanh(sparse(qkv_proj))  → Q [D], V [D], K [heads]
                  eK = exp(tanh(K)),  eKV = eK * V (per head)
                  kernel = exp(w_conv) - 1
                  num = conv(window ++ eKV) + (cum_eKV + cumsum(eKV))
                  den = conv(window ++ eK)  + (cum_eK  + cumsum(eK))
                  Y = tanh(Q) * num / (den + 1e-6)
                  out = tanh(sparse(out_proj(Y)))          ─ x += out
x ─ DyT(norm2) ─ tanh(sparse(proj2(tanh(tanh(sparse(proj1)))))) ─ x += mlp
```

The conv follows upstream's tap order: `out[t] = sum_j kernel[:, j] * ext[t + j]` with `ext = window(ksize - 1) ++ new`, so tap `ksize - 1` is the current token. A fresh state is zeros, which equals the flat forward's left padding.

### 4.2 Wiring

`ModelType::Z1t`, the `"z1t"` detection arm, `LoadedModel`, the special and nonstandard loaders, `model_metadata`, the registry, the memory estimate (`kv_arch` classifies Z1T as pure SSM), the tensor-parallel model lists, the APC opt-out (`HYBRID_SSM_MODEL_TYPES` and `HYBRID_SSM_ARCH_FRAGMENTS`), `docs/supported-models.md`, `docs/CONTINUOUS_BATCHING.md`, `docs/turbo-kv-cache.md` and `CHANGELOG.md`.

`last_row_logits` selects the sampled row, applies the final DyT, then the head with its bias, so prefill never projects every row through the 50257-wide head.

### 4.3 Load-time refusals

`aft_kind` other than `conv`, `hidden_size` not divisible by `aft_heads`, `aft_ksize` of 0 or above 1024, position tables whose element count overflows i32, index tables out of range, and dense weights whose shape does not match the config.

---

## 5. Learning Points

### 5.1 A length-1 scan is not free in MLX

MLX lowers `cumsum` to a strided scan kernel even when the scanned axis has length 1. In a decode loop that runs 4 layers times 2 scans per token, that cost dominated the step. When an op is the identity for a degenerate shape, test for the shape and skip the op. Upstream mlx-vlm avoids the same trap with a separate `_decode_step` that drops the time axis.

### 5.2 Check sizes in the type you index with

Both HIGH security findings came from validating a value after narrowing it with `as i32`. Validate in the wide type (`usize` or `i64`, with checked or saturating arithmetic) and narrow only after the bound holds.

### 5.3 Prove exactness with a negative arm

The trace gate passes at 0 of 113 decided positions, which on its own does not show that the trace can detect a wrong model. The conv-only arm (pool removed) fails at 62 of 113, so the gate does separate a correct port from a structurally wrong one.

---

## 6. Further Learning

### Key Terms

| Keyword | Description | Relevance |
|---------|-------------|-----------|
| AFT (Attention Free Transformer) | Token mixing through exponentiated keys weighted over positions, without q·k | The Z1T token mixer is an AFT-conv variant |
| DyT (Dynamic Tanh) | `weight * tanh(alpha * x) + bias` in place of LayerNorm | The only normalization in Z1T |
| Fixed-sparsity linear | Each output reads `k` inputs chosen by a fixed index table | Every internal projection, fan-in 4 |

### Related PRs/Issues

- Issue #2364: the requirements this PR implements.
- #3090: `last_row_logits` coverage rule.
- #2685, #2672: batched decode across model-owned sequence state, the pattern `z1t_batch.rs` follows.
- Blaizzy/mlx-vlm#2166: the upstream port used as the numerical reference.

---

## 7. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 25 |
| Lines added | +1766 |
| Lines deleted | -9 |
| Tests added | 15 |

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| Model | 5 files | `z1t.rs`, `z1t_config.rs`, `z1t_layers.rs`, `z1t_batch.rs`, `z1t_tests.rs` |
| Wiring | 12 files | Detection, loaders, registry, metadata, memory estimate, TP lists, APC opt-out |
| Security | 3 commits | Config bounds, gather transient bound, dense shape checks |
| Documentation | 4 files | supported models, continuous batching, turbo KV cache, CHANGELOG |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `7b5d502` | feat | add Z1T-0 (z1t) attention-free AFT-conv decoder |
| `4ead407` | docs | document Z1T-0 support |
| `e36757d` | fix | size Z1T as having no KV cache |
| `44e0247` | fix | bound Z1T config values and sparse gather transients |
| `5ac895a` | fix | validate Z1T dense shapes and exempt Z1T from APC by arch name |

---

## 8. Follow-up Actions

### Future Improvements

- RWKV-7 has the same KV-estimate gap that `e36757d6` closed for Z1T.
- Prefix caching of the streaming state (APC is opted out for `z1t`).
- Dense variants (`linear_fan_in: null`) are accepted by the code path and covered by a unit test, but no real dense checkpoint exists to validate them.
- An unknown sequence id panics instead of returning an error (shared with RWKV-7).

---

## Appendix

### A. Verification on `models/z1t-0-mlx` (M3 Ultra)

- `mlxcel generate -p "The meaning of life is" -n 64 --temp 0`: "The meaning of life is not a thing of the past. It is not a question of meaning. ..." at 225.7 tok/s. The vendored mlx-vlm module produces the same 64 tokens.
- `mlxcel-server`, one `/v1/completions` request (32 tokens, temperature 0): the same continuation, `finish_reason: length`, usage 5 + 32.
- Teacher-forced trace (`examples/logit_trace`, 8 chunks of 64, top-8, `compare_logit_traces.py --decided 2`):

| Reference | Candidate | Top-1 disagreement | Decided disagreement | Max logit delta |
|---|---|---|---|---|
| mlx-vlm z1t | mlxcel | 0 / 512 | 0 / 113 | 0.0000 |
| mlx-vlm z1t | mlx-vlm, conv only | 325 / 512 | 62 / 113 (fails) | n/a |
| mlxcel B=1 | mlxcel batched B=4 | 0 / 512 | 0 / 113 | 0.0000 |

Perplexity: 130.56 for both full arms, 320.92 for the conv-only arm.

### B. Unit tests

`models::z1t` (13), `coverage_tests` (18), `prompt_cache::hybrid_ssm` (14), `execution::kv_arch` (57), and clippy on lib and tests with `-D warnings`.
