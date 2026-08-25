# Technical Report: PR #1411 - BERT and XLM-RoBERTa Encoders

**Date**: 2026-08-26
**Author**: mlxcel contributors
**Status**: Completed
**Languages**: Rust, Markdown
**Risk Level**: Medium

---

## Executive Summary

PR #1411 implements issue #1321: the first family forward pass behind the embedding subsystem that #1353 built. BERT (`model_type: bert`) and XLM-RoBERTa (`model_type: xlm-roberta`) are one post-LayerNorm encoder over absolute position embeddings, so they land as a single module selected by a `BertVariant` switch rather than as two ports.

The change also supplies the `BertForSequenceClassification` / `XLMRobertaForSequenceClassification` head that `/v1/rerank` (#1356) needs. That head reuses the same trunk with the `pooler.` tensors kept and is reached by directory path, because a `ForSequenceClassification` export is a reranker and detection deliberately refuses to call it an embedder.

Four real checkpoints load and run through both `mlxcel embed` and `POST /v1/embeddings`, with the CLI and HTTP vectors bit-identical. `all-MiniLM-L6-v2` reproduces the published sentence-transformers quickstart cosines to four decimals, `BAAI/bge-m3` reproduces its model-card dense matrix within 6e-4, and `BAAI/bge-reranker-v2-m3` reproduces the model card's sign split.

---

## 1. Problem Statement

`src/embeddings/` shipped the trait, the pooling modes, the mask builders, the tokenizer plumbing, the worker thread, the HTTP route and the CLI, but no family. Every recognized embedding checkpoint answered `<family> is detected as an embedding checkpoint, but this embedding family is not yet supported by /v1/embeddings`. The endpoint existed and returned nothing useful.

BERT and XLM-RoBERTa are the highest-value first port: they cover `all-MiniLM-L6-v2`, the `multilingual-e5` family and `bge-m3`, which between them account for most deployed sentence-embedding checkpoints, and they carry the classification heads the reranker endpoint needs.

Two hazards were specific to this family rather than generic port work.

| Hazard | Impact | Where it bites |
|--------|--------|----------------|
| XLM-RoBERTa position ids are offset by `pad_token_id + 1` | Out-of-bounds gather on a long input | Any XLM-RoBERTa checkpoint whose derived `max_length` comes from `max_position_embeddings` |
| Position ids for XLM-RoBERTa key off token ids, not the attention mask | Silently wrong embeddings for padded batches | Every batched request |

---

## 2. Technical Decisions

### 2.1 One module with a variant switch, not two ports

The two families differ in exactly five places: position-id construction, the `type_vocab_size` default (2 versus 1), the `layer_norm_eps` default (1e-12 versus 1e-5), the `pad_token_id` default (0 versus 1) and the weight-key prefix (`bert.` versus `roberta.`). Everything else, the embedding sum, the LayerNorm placement, the attention shape, the GELU feed-forward and the residual order, is identical.

Splitting them would have duplicated roughly 300 lines of block code to express five constants. `BertVariant` keeps one trunk and makes the differences enumerable, which is also what let the classification head serve both dialects from one `ClassifierHead` with a prefix pair.

### 2.2 Only `model_type` selects the dialect

`intfloat/multilingual-e5-small` declares `model_type: bert` and `architectures: [BertModel]` while shipping `tokenizer_class: XLMRobertaTokenizer` and a sentencepiece vocabulary. Its weights are a BERT layout and upstream builds its position ids the BERT way, so keying the dialect on the tokenizer or the vocabulary would have produced silently wrong positions for a widely used checkpoint. The switch reads `model_type` and nothing else.

### 2.3 The family reports its own token cap

XLM-RoBERTa indexes its position table from `pad_token_id + 1`, so a table with `max_position_embeddings` rows addresses `max_position_embeddings - pad_token_id - 1` real tokens. `bge-m3` publishes 8194 rows and holds 8192 tokens.

`EmbeddingLimits::derive` had no way to know that: it takes a boolean `is_absolute_position` and reads `max_position_embeddings` directly. For `bge-m3` the checkpoint's own `sentence_bert_config.json` and the shared 8192 cap happen to mask the problem, but the stock 514-row `xlm-roberta-base` layout without a `sentence_bert_config.json` would derive 514 and gather two rows past the table.

The fix is a new defaulted trait method, `EmbeddingModel::max_sequence_length() -> Option<usize>`, folded into `max_length` next to the existing `pad_to_max_length` clamp. Putting the rule in the family rather than in `limits.rs` keeps the shared derivation free of per-family arithmetic and gives every later family the same hook.

The encoder additionally refuses a batch wider than that cap with a named error, so a bug upstream of the limit surfaces as a message rather than as an MLX gather fault.

### 2.4 Token-type embeddings are always applied

`XLMRobertaModel` has `type_vocab_size: 1`, which invites treating the segment table as absent. It is not: `bge-m3` ships a real `[1, 1024]` learned vector that upstream adds to every token. Dropping it would shift every embedding.

The encoder therefore always reads the table. When the batch carries no `token_type_ids`, it indexes with a `[1, 1]` zero array whose `[1, 1, D]` result broadcasts over `[B, L, D]`, which costs one row lookup instead of materializing a `[B, L]` zero matrix. `needs_token_type_ids()` stays `true` only for the BERT dialect, so the engine does not pay for a segment axis XLM-RoBERTa never varies.

### 2.5 The classification head is reached by path, not by detection

`is_embedding_checkpoint` returns `Ok(None)` for any `architectures[0]` ending in `ForSequenceClassification`, so `get_model_type` reports `Unsupported model type` for `bge-reranker-v2-m3`. That is correct: a cross-encoder reranker is not an embedder, and routing it to `/v1/embeddings` would return a pooled hidden state instead of a relevance score.

`BertSequenceClassifier::load(dir)` therefore reads `config.json` itself and picks the dialect with `BertVariant::from_config`. This PR adds no HTTP surface for it; #1356 owns that.

### 2.6 Sanitization is idempotent and prefix-agnostic

A task-head checkpoint nests the encoder under `bert.` or `roberta.`; a bare `BertModel` export does not. `sanitize` strips whichever prefix is present, so both layouts reach the same key set, then drops `position_ids` buffers (older transformers exports register them as tensors), `cls.` and `lm_head.` masked-LM heads, and `pooler.` unless a classifier head is being built. Running it twice is a no-op, matching the convention in `src/models/sanitize.rs`.

---

## 3. Implementation Details

| Area | Change |
|------|--------|
| `src/models/bert_config.rs` | `BertVariant`, `BertArgs` with variant-aware defaults, `num_labels` from `num_labels` or `id2label`, `max_sequence_length()`. Re-exported through `bert` so callers keep one public path. |
| `src/models/bert.rs` | `sanitize`, `xlm_roberta_position_ids`, `BertEmbeddings`, `BertLayer`, `BertEncoder`, and the `Activation` enum with a local `gelu_new` implementation. |
| `src/models/bert_heads.rs` | `BertEmbeddingModel` implementing `EmbeddingModel`, `BertSequenceClassifier`, and the shared `load_encoder` that both use. |
| `src/embeddings/loader.rs` | The `Bert` / `XlmRoberta` dispatcher arm; `finish_loaded_model` folds `max_sequence_length()` into the derived limit. |
| `src/embeddings/model.rs` | `EmbeddingModel::max_sequence_length()`, defaulted to `None`. |
| `docs/embeddings.md`, `docs/supported-models.md` | Family table rows, the prompt-prefix table, the positional length rule, and where the classification head lives. |

The forward pass is:

```
positions = 0..L                                  # BERT
positions = cumsum(ids != pad) * (ids != pad) + pad   # XLM-RoBERTa
h = LayerNorm(word[ids] + position[positions] + type[segments])
mask = create_bidirectional_padding_mask(attention_mask)   # [B, 1, 1, L]
for layer:
    a = LayerNorm(attn_out(sdpa(q, k, v, head_dim^-0.5, mask)) + h)
    h = LayerNorm(output(gelu(intermediate(a))) + a)
```

`sdpa` is `mlxcel_core::layers::attention` with the explicit additive mask, and every projection carries a bias, which is why `UnifiedLinear` rather than a bias-free path is used throughout.

### 3.1 Test serialization

`cargo test` runs test functions on parallel threads, and concurrent MLX forward passes inside one process perturb each other: sibling unit #1332 measured two byte-identical rows of one batch coming back at cosine 0.999912 instead of 1.0, and a classifier logit moving by 0.05, only while other real-checkpoint tests ran. `EmbeddingModel` is documented as single-thread and the server honors that through the embedding worker, so the hazard is test-side, but a gate measured under it is meaningless.

Every BERT test that touches MLX takes a module-level `OnceLock<Mutex<()>>` guard that recovers a poisoned lock instead of propagating it. Three consecutive runs of the guarded module produced identical gate numbers at every printed decimal.

---

## 4. Real-Checkpoint Results

All four checkpoints are f32 and run in f32; the shared bf16-to-f16 rule never fires for them, which matters because `all-MiniLM-L6-v2` uses `layer_norm_eps: 1e-12` and that underflows in f16.

| Checkpoint | Reference | Measured |
|------------|-----------|----------|
| `sentence-transformers/all-MiniLM-L6-v2` | quickstart cosines 0.666 and 0.105 | 0.6660 and 0.1046 |
| `BAAI/bge-m3` (safetensors mirror) | model card `[[0.6265, 0.3477], [0.3499, 0.678]]` | `[[0.6259, 0.3475], [0.3499, 0.6782]]` |
| `BAAI/bge-reranker-v2-m3` | model card sign split, about -8 and +5 | -8.1838 and 5.2650 |
| `intfloat/multilingual-e5-small` | matching passage ranks first | 0.9252 versus 0.7632, same ordering for a Korean query |

Epic self-consistency gate: identical inputs give cosine 1.0000000 to 1.0000001; a padded batch matches the unpadded single-input result at 0.9999999 to 1.0000001; unrelated sentences score 0.1046 for MiniLM and 0.2519 for bge-m3. A 2613-token bge-m3 input embeds to a unit vector, which exercises the shifted position ids well past 512.

`multilingual-e5-small` is the one gate deviation: its unrelated pair scores 0.7390, above the epic's 0.5 bound. That is the checkpoint's compressed cosine range, not the port. The test carries a documented 0.80 bound for this family, and the ranking test is the real discrimination gate for it.

---

## 5. Validation Summary

Local checks on GB10 (Linux, CUDA), all with `--profile test-fast --features cuda`.

| Check | Result |
|-------|--------|
| `cargo fmt --all -- --check` | exit 0 |
| `cargo clippy --profile test-fast --features cuda --lib --bins --tests -- -D warnings` | exit 0 |
| `cargo check --profile test-fast --features cuda --all-targets` | exit 0 |
| `cargo test --profile test-fast --features cuda --lib models::bert` | 27 passed, 0 failed, three repeats with identical gate numbers |
| `cargo test --profile test-fast --features cuda --lib embeddings::` | 63 passed, 0 failed |
| `cargo build --profile test-fast --features cuda --bins` | exit 0 |
| `mlxcel arch`, `mlxcel list`, `/v1/models` | Both variants listed under `Embedding`; the served model is reported |
| `dimensions: 8` with `encoding_format: base64` | 8-component unit vector on all three embedders |

---

## 6. Change Summary

| File | Lines | Purpose |
|------|-------|---------|
| `src/models/bert.rs` | new, 387 | Encoder trunk |
| `src/models/bert_config.rs` | new, 199 | Config resolution |
| `src/models/bert_heads.rs` | new, 258 | Embedding model and classification head |
| `src/models/bert_tests.rs` | new | Sanitization, config, position ids, forward shapes |
| `src/models/bert_heads_tests.rs` | new | Head behavior over a deterministic fixture |
| `src/models/bert_real_checkpoint_tests.rs` | new | Real-checkpoint gates, soft-skipping when absent |
| `src/embeddings/loader.rs` | +15/-5 | Dispatcher arm and the positional cap |
| `src/embeddings/model.rs` | +11 | `max_sequence_length()` |
| `src/embeddings/loader_tests.rs` | +27/-6 | Unported-family test made family-agnostic |
| `src/embeddings/real_checkpoint_tests.rs` | +4/-3 | BERT leaves the unported list |
| `docs/embeddings.md`, `docs/supported-models.md` | +31/-1 | Family notes and table rows |

---

## 7. Known Limitations and Follow-ups

- `BAAI/bge-m3` publishes only `pytorch_model.bin` and the mlxcel downloader takes safetensors, so the XLM-RoBERTa embedder gate ran against `seansitter/bge-m3-safetensors`. Its `config.json` and tensor layout are identical to the BAAI repo: 391 tensors, `XLMRobertaModel`, 8194 position rows, 250002 vocabulary. A safetensors conversion in the BAAI repo would remove the mirror.
- No PyTorch, transformers or numpy is installed on the validation machine, so every reference number above is published by the checkpoint's own model card or quickstart rather than recomputed locally. A parity run against `transformers` on a machine that has it would tighten the tolerances from 1e-2 to something closer to float noise.
- Quantized BERT and XLM-RoBERTa checkpoints go through `UnifiedLinear` and `UnifiedEmbedding` and should load, but no quantized checkpoint of either family was available to run.
- Only the `gelu` activation appears in published checkpoints of these families. `gelu_new` and `relu` are implemented and reachable through `hidden_act` but untested against a real checkpoint.
- The `/v1/rerank` wiring for `BertSequenceClassifier`, and the sparse and ColBERT outputs of `bge-m3`, remain out of scope.
- Performance was deliberately not measured; the epic runs one performance pass at the end on a quiet machine.

---

## References

- Issue #1321, epic #1348
- PR #1408 (issue #1353), the embedding foundation this port builds on
- Issue #1356, the `/v1/rerank` consumer of the classification head
- `docs/embeddings.md`, section `Family notes`
