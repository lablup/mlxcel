# Technical Report: PR #1412 - ModernBERT Encoder for /v1/embeddings

**Date**: 2026-08-26
**Author**: mlxcel contributors
**Status**: Completed
**Languages**: Rust, Markdown
**Risk Level**: Medium

---

## Executive Summary

PR #1412 ports the ModernBERT encoder (alternating local/global attention, RoPE, GeGLU) so `nomic-ai/modernbert-embed-base` serves through `POST /v1/embeddings` and `mlxcel embed`, and adds the `ModernBertForSequenceClassification` head that `/v1/rerank` (#1356) consumes. It is the first family forward pass to land on the embedding foundation from #1353, so it also establishes the pattern the remaining family ports follow: family module under `src/models/`, one arm in the embedding loader dispatcher, real-checkpoint gates that soft-skip.

---

## 1. Problem Statement

### 1.1 Background

Since #1353, mlxcel detected ModernBERT checkpoints and routed them to `/v1/embeddings`, but no forward pass existed. Loading `nomic-ai/modernbert-embed-base` reported `ModernBERT encoder is detected as an embedding checkpoint, but this embedding family is not yet supported`. The detection table, the `ModelType::ModernBert` variant, the `mlxcel arch` entry, and the `ModelKind::Embedding` registration were all already in place; only the model was missing.

### 1.2 What the Architecture Demands

ModernBERT is not a BERT variant with different hyperparameters. Five properties have no precedent in the existing encoder code:

- **Alternating attention.** Two out of every three layers use a bidirectional sliding window; the third sees the whole sequence. The layer index alone decides which.
- **Two RoPE bases.** A local layer rotates with `local_rope_theta` (10000), a global layer with `global_rope_theta` (160000). The bases differ by 16x.
- **No position table.** RoPE replaces absolute positions, so `max_position_embeddings` does not cap input length the way it does for BERT, XLM-RoBERTa, and SigLIP.
- **Fused projections.** `Wqkv` is one `[3 * hidden, hidden]` tensor and `Wi` one `[2 * intermediate, hidden]` tensor.
- **An absent norm in layer 0.** Upstream makes `layers.0.attn_norm` an `nn.Identity()`, so the checkpoint ships no such tensor.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood before this change |
|------|--------|-------------------------------|
| Local/global parity or RoPE base applied wrongly, producing plausible but degraded embeddings | High | High, because no shape assertion catches it |
| Fused projections split by slicing the weight, silently breaking quantized exports | High | Medium |
| `layers.0.attn_norm` treated as a load error, rejecting every valid checkpoint | High | High |
| Reranker head advertising a label count that disagrees with its own logits | Medium | Medium, and invisible until #1356 consumes it |

---

## 2. Technical Decisions

### 2.1 Alternate the RoPE Base With the Mask, Not Just the Mask

`ModernBertArgs::is_local_layer` and `ModernBertArgs::rope_base` are a single source of truth consulted at layer construction. A reader who only ported the mask would produce a model that loads, runs, returns finite unit-norm vectors, and retrieves measurably worse. `layer_parity_selects_local_and_global_rope_base` pins both facets together across nine layer indices, and also covers the `local_rope_theta: null` fallback to the global base, which upstream expresses as `if config.local_rope_theta is not None`.

### 2.2 Split the Fused Projections After the Matmul, Never the Weight

`Wqkv` and `Wi` load through `UnifiedLinear` and are split with `slice_axis` on the projection output. Slicing the weight tensor instead would have been simpler to read and would have worked on the dense f32 checkpoints available here, but a quantized export packs each fused tensor as one unit with its own scales and biases; slicing the packed plane produces garbage. This decision costs nothing on the dense path and is the only reason a future quantized ModernBERT loads unchanged.

### 2.3 Two Epsilon Fields Instead of a serde Alias

The issue specified `#[serde(alias = "layer_norm_eps")]` on `norm_eps`. Both published checkpoints carry **both** keys in the same `config.json`, and serde reports a duplicate-field error when an aliased field is supplied under two of its names, so the alias would have failed to parse every real checkpoint. `norm_eps` and `layer_norm_eps` are independent `Option<f32>` fields resolved by a `norm_eps()` accessor. `norm_eps_accepts_either_spelling_and_both_together` locks all four combinations so a future simplification back to an alias fails loudly.

### 2.4 Derive `num_labels` From the Tensor, Not the Config

The first implementation took `num_labels` from `config.json` (`num_labels`, else `len(id2label)`, else 1). A synthetic test exposed the consequence: a config that understates the label count leaves `num_labels()` disagreeing with the real width of `logits`. Since #1356 will size its output on that accessor, the disagreement would surface as a downstream shape bug far from its cause. `classifier.weight`'s row count is now authoritative. Row counts are never packed by quantization, so this is correct on both paths; the column count is checked against `hidden_size` on the dense path only, and a config/tensor mismatch is logged rather than silently accepted.

### 2.5 Reach the Reranker Head by Directory, Not by Detection

`is_embedding_checkpoint` deliberately returns `Ok(None)` for any `architectures[0]` ending in `ForSequenceClassification`: a reranker is not an embedder, and #1353's tests assert that `Alibaba-NLP/gte-reranker-modernbert-base` does not detect as an embedding variant. Rather than weaken that rule, `ModernBertSequenceClassifier::load` reads and validates `config.json` itself and is reached by directory. `/v1/rerank` wiring in #1356 can therefore adopt the head without touching detection.

---

## 3. Implementation Details

### 3.1 Forward Pass

```
h = LayerNorm(tok_embeddings[input_ids])                    # embeddings.norm, no bias
global_mask  = create_bidirectional_padding_mask(mask)      # [B, 1, 1, L]
sliding_mask = create_bidirectional_window_mask(mask, local_attention / 2 + 1)   # [B, 1, L, L]
for i in 0..num_hidden_layers:
    local = i % global_attn_every_n_layers != 0
    x = h if i == 0 else LayerNorm(h)                       # layers.0 has no attn_norm
    q, k, v = split(Wqkv(x))                                # each [B, heads, L, head_dim]
    q, k = fast_rope(.., head_dim, traditional=false, base = local ? local : global)
    h = h + Wo(sdpa(q, k, v, head_dim^-0.5, local ? sliding_mask : global_mask))
    inp, gate = chunk(Wi(LayerNorm(h)))
    h = h + Wo_mlp(gelu(inp) * gate)
last_hidden_state = LayerNorm(h)                            # final_norm
```

The window argument is `local_attention / 2 + 1` because `create_bidirectional_window_mask` blocks at `|q - k| >= window`, while ModernBERT attends `|q - k| <= local_attention / 2`. With the published `local_attention: 128` the bound is 65, verified directly against the additive mask in `sliding_mask_attends_within_64_and_blocks_beyond`.

### 3.2 Weight Layout and Sanitize

`sanitize_modernbert_weights` strips an optional leading `model.` (present in the MLM and classifier exports, absent in `ModernBertModel`), always drops `decoder.*` and `pooler.*`, and drops `head.*` / `classifier.*` unless the caller is building the classification head. `ModernBertForMaskedLM` therefore loads as a plain embedder with its MLM head discarded.

The `Wqkv` block order is Q, K, V, each holding all heads, which follows from upstream's `qkv.view(bs, -1, 3, num_heads, head_dim)`: the 2304-wide axis splits into three 768-wide blocks before the head split. `Wi`'s halves are (input, gate) in that order, so the activation lands on the first half and the gate multiplies the second.

---

## 4. Test-Harness Correctness Findings

Three findings during validation were about the tests rather than the model, and each would have shipped a misleading gate.

### 4.1 MLX Driven From Several Threads at Once

`EmbeddingModel` is documented in `src/embeddings/model.rs` as used from exactly one thread, and the server backs that with a dedicated MLX-owning worker per model. `cargo test` runs tests in parallel by default, so the suite was the one component violating that contract. Two distinct symptoms appeared on this CUDA host:

- **Silently wrong numbers.** One parallel run in three scored two byte-identical rows of a single batch at cosine 0.99991 instead of 1.0, and moved a reranker logit by 0.05, while the first row of every batch stayed bit-identical at 3.0652723. Non-associative float reduction cannot make two identical rows of one batch disagree by 9e-5.
- **An abort.** `cudaStreamEndCapture ... operation failed due to a previous error during capture`, SIGABRT, inside MLX's CUDA graph capture.

A shared module lock now serializes every MLX-touching test in both modules. An initial fix guarded only the five real-checkpoint gates and missed the nine synthetic tests that also build encoders, which is exactly what the abort then hit; the final audit confirms all fourteen MLX-touching tests take the lock and the five that do not are pure config-parsing or filesystem tests.

### 4.2 A Tolerance Fitted to an Artifact

The padding-invariance gate first failed with a worst per-component drift of 1.2e-3, which was rationalized as f32 reduction-order noise and given a 5e-3 bound. That explanation was wrong: the drift was the cross-talk above. Once serialized, the measured drift is 1.3e-7, and the bound is 1e-4. A tolerance sized around an artifact is worse than no tolerance, because it silently widens the gate while looking rigorous.

### 4.3 A Gate That Assumed Its Own Premise

The long-document gate was named for 4096 tokens but only repeated a sentence 220 times without checking the result. The document is actually 4187 tokens. The test now asserts both that the input exceeds 4096 tokens and that it stays under `max_length` so no truncation occurs, meaning an edit to the sentence cannot silently reduce it to a short-sequence test that still passes.

---

## 5. Real-Checkpoint Results

Gate values were byte-identical across 15 consecutive runs after serialization.

| Gate | Observed | Requirement |
|------|----------|-------------|
| Identical inputs, cosine | 0.999999583 | within 1e-6 of 1.0 |
| Query vs relevant document | 0.625270 | must beat the unrelated one by 0.15 |
| Query vs unrelated document | 0.144859 | below 0.5 |
| Padded batch vs single input | cosine 0.999999642, worst component 1.3e-7 | within 1e-3 |
| Document of 4187 tokens, batched vs solo | cosine 0.999999821 | within 1e-3 |
| Vector L2 norms | 1.000000004 / 1.000000025 | within 1e-5 of 1.0 |
| gte-reranker logits (relevant, irrelevant) | 3.0652723, -1.1471016 | finite `[B, 1]` |

End to end, `mlxcel embed` returns a cosine matrix of 1.0000 on the diagonal, 0.6253 query vs the t-SNE document, and 0.1448 query vs the Eiffel document at dim 768 and max_length 8192. `mlxcel-server` serves `GET /v1/models` as `modernbert-embed-base` and `POST /v1/embeddings` as two unit-norm 768-dim vectors at cosine 0.62521, with `dimensions: 256` returning 256 re-normalized components.

---

## 6. Validation Summary

| Command | Result |
|---------|--------|
| `cargo test --profile test-fast --features cuda --lib models::modernbert` | 19 passed, 0 failed |
| `cargo test --profile test-fast --features cuda --lib embeddings::` | 62 passed, 0 failed |
| `cargo clippy --profile test-fast --features cuda --lib --bins --tests -- -D warnings` | exit 0 |
| `cargo fmt --all -- --check` | exit 0 |
| `cargo build --profile test-fast --features cuda --bins` | exit 0 |
| `mlxcel embed` / `mlxcel-server` + `POST /v1/embeddings` | verified against the real checkpoint |
| `mlxcel arch` / `mlxcel list` | ModernBERT encoder listed under Embedding; both checkpoints listed |

Clippy caught five real issues, all mechanical: two `manual_is_multiple_of` (which also removes a latent divide-by-zero shape, since `is_multiple_of(0)` returns cleanly where `%` panics), one `neg_multiply`, and two `excessive_precision` constants in the erf reference that f32 cannot represent.

---

## 7. Change Summary

| File | Change |
|------|--------|
| `src/models/modernbert.rs` | New. Config parsing and validation, weight sanitize, GeGLU split, layer and encoder. |
| `src/models/modernbert_heads.rs` | New. `EmbeddingModel` implementation and the sequence-classification head. |
| `src/models/modernbert_tests.rs` | New. 14 unit tests covering parity, masks, GeGLU, sanitize, config, detection, and the classifier. |
| `src/models/modernbert_real_checkpoint_tests.rs` | New. 5 soft-skipping real-checkpoint gates. |
| `src/embeddings/loader.rs` | Only the `ModelType::ModernBert` arm split out of the `not yet supported` match. |
| `src/models/mod.rs` | Module and test-module registration. |
| `docs/supported-models.md` | Embedding table row. |
| `docs/embeddings.md` | ModernBERT family section and status update. |

Total: 8 files, +1866 / -2.

---

## 8. Follow-up Actions and Unverified Areas

These are recorded because a future maintainer should not assume them covered.

- **No quantized ModernBERT checkpoint exists to test.** The split-after-projection design in 2.2 is what makes the quantized path work, but no quantized export was available, so that path is reasoned rather than exercised.
- **No element-wise parity against transformers.** PyTorch and transformers are not installed on this host, so numeric agreement with the reference implementation was not checked directly. Validation rests on the issue's published acceptance thresholds (relative ordering, the 0.15 margin, unit norms, padding invariance) rather than tensor-level parity. A parity harness would be the strongest remaining gate.
- **The MLM path is covered synthetically only.** `sanitize_modernbert_weights` drops `head.*` and `decoder.*` and has a synthetic test, but no real `ModernBertForMaskedLM` checkpoint was loaded.
- **Length coverage stops at 4187 tokens**, not the full 8192 the checkpoint allows.
- **Reranker ordering is one pair.** The head ranks a relevant document above an irrelevant one, but full ordering validation belongs to #1356.
- **Pre-existing teardown race, outside this PR.** The combined `models::modernbert` filter sometimes aborts at process exit with `Destroy(handle_) failed: driver shutting down` after `test result: ok`, turning a passing run into exit 101. Observed twice in 17 post-fix runs, with 0 in the most recent 10 consecutive runs and 0 in 9 isolated single-module runs, and both occurrences coincided with sibling units loading the same GPU. This is a load-dependent MLX CUDA context-teardown race already recorded from unmodified main and reproduced on the untouched `embeddings::` suite. The test command was deliberately not reshaped to hide it.

---

## References

- Issue #1332 (this port), epic #1348 (embedding families)
- PR #1408 / issue #1353 (embedding foundation: pooling, masks, `ModelKind::Embedding`, `/v1/embeddings`)
- Issue #1356 (`/v1/rerank`, consumes `ModernBertSequenceClassifier`)
- `docs/embeddings.md`, `docs/supported-models.md`
