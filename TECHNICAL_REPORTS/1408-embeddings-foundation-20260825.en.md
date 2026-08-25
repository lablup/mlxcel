# Technical Report: PR #1408 - Embeddings Foundation: Pooling, Masks, Embedding Kind and /v1/embeddings

**Date**: 2026-08-25
**Author**: mlxcel maintainers
**Status**: Completed; family forward passes deferred to the epic #1348 sub-issues
**Languages**: Rust, Markdown
**Risk Level**: Medium

---

## Executive Summary

PR #1408 lands the shared embedding foundation requested by issue #1353 (epic #1348). Before this change mlxcel had no embedding path at all: every checkpoint was either a text generator or a VLM, detection dispatched on `model_type` alone, the only attention masks were causal, sentence-transformers module subfolders were never read, and no `/v1/embeddings` route existed. The PR adds a `src/embeddings/` subsystem (the `EmbeddingModel` trait, four pooling modes with a `1_Pooling/config.json` reader, L2 normalization, `dimensions` truncation, `max_length` derivation, right-padded batch tokenization, a family dispatcher and a length-sorted micro-batching engine), three padding-aware mask builders and a subfolder-aware safetensors loader in `mlxcel-core`, `ModelKind::Embedding` with thirteen `ModelType` variants and layout-based detection, a dedicated embedding worker thread, the OpenAI-compatible `POST /v1/embeddings` route, five `--embedding-*` server flags, the `mlxcel embed` command and `docs/embeddings.md`.

No family forward pass is included. Every recognized family reports `not yet supported` from the loader until its own port merges; the route, worker, batching, pooling and encoding are exercised end to end through a test-only stub model, and detection, pooling-config parsing, length derivation and tokenization were verified against real `all-MiniLM-L6-v2` and `Qwen3-Embedding-0.6B` checkpoints.

---

## 1. Problem Statement

### 1.1 Background

Epic #1348 ports a set of embedding families (BERT and MiniLM, XLM-RoBERTa, ModernBERT, SigLIP text, EmbeddingGemma, Qwen3-Embedding, Qwen3-VL-Embedding, LFM2, Ministral 3 and Llama bidirectional embedders, Llama-Nemotron-VL, ColIdefics3, ColQwen2.5). Each family needs the same infrastructure: a way to be recognized as an embedder rather than a generator, padding-aware attention masks that stay finite in f16, pooling and normalization, a batching engine, a worker thread and an HTTP surface. Issue #1353 isolates that infrastructure so the family ports only add a forward pass.

### 1.2 Existing limitations

- `ModelKind` had only `Text` and `Vlm`; `load_model` always returned a `LanguageModel`, so an embedding checkpoint failed with a missing-tensor symptom (no `lm_head`) instead of a clear message.
- `get_model_type` read `model_type` only. Qwen3-Embedding (`model_type: qwen3`) and EmbeddingGemma (`gemma3_text`) were routed to the causal generators.
- `mlxcel_core::utils` had causal masks only (`create_causal_mask`, the window variant, the left-padding variant). Encoder families need a mask that blocks padding keys in both directions.
- `load_weights_from_dir` listed the shard index or top-level `*.safetensors` only, so the `2_Dense/model.safetensors` projection of a sentence-transformers export was silently dropped.
- The server had an `audio_model` slot and workers for Whisper and Kokoro but nothing equivalent for embeddings, and no `/v1/embeddings` route.
- `MlxcelTokenizer::encode` returned ids only: no batch padding, no `token_type_ids`, no control over the fixed padding some `tokenizer.json` files bake in.

### 1.3 Risk assessment

| Risk | Impact | Likelihood before fix |
|------|--------|-----------------------|
| Embedding checkpoints misrouted to causal generation and producing garbage or a loader panic | High | High for Qwen3-Embedding and EmbeddingGemma |
| Each family port reimplementing pooling, masks and batching with divergent semantics | High | High without a shared module |
| f16 masks built as `(1 - m) * C` producing NaN for every attended position | High | Medium |
| A silent fallback on an unsupported pooling mode returning wrong vectors without an error | Medium | Medium |
| Embedding inference sharing the chat worker's MLX stream and stalling generation | Medium | Medium |

---

## 2. Technical Decisions

### 2.1 Recognize embedding checkpoints by layout before the `model_type` dispatch

`is_embedding_checkpoint` runs right after the Kokoro check in `get_model_type` and before the `model_type` match. A checkpoint is an embedder when any of the following holds: `model_type` is an encoder-only family (`bert`, `xlm-roberta`, `modernbert`, `siglip`); `architectures[0]` is an embedding architecture (`BertModel`, `XLMRobertaModel`, `ModernBertModel`, `SiglipModel`, `SiglipTextModel`, `LlamaBidirectionalModel`, `LlamaNemotronVLModel`, `Lfm2BidirectionalModel`, `ColIdefics3`, `ColQwen2_5`, `ColQwen2ForRetrieval`, or the flag-gated `Gemma3TextModel` with `use_bidirectional_attention: true` and `Ministral3Model` with `is_causal: false`); `modules.json` lists a module whose `type` ends with `.Pooling`; or `1_Pooling/config.json` exists.

Two negative rules protect the reranker issue (#1356) and the generators: an `architectures[0]` ending in `ForSequenceClassification` is never an embedder, and a `modules.json` whose only extra module is `1_LogitScore` (Qwen3-VL-Reranker) does not qualify. The resolved variant is keyed on `model_type`; a pooling layout on a `model_type` with no embedding family is returned as an error naming the `model_type` rather than falling through to a causal generator. `Qwen3ForCausalLM` without a pooling layout still detects as `Qwen3`.

Alternatives considered: keying on `architectures[0]` alone misses sentence-transformers exports that keep the base architecture name (`Qwen3-Embedding` ships `Qwen3ForCausalLM` plus `1_Pooling`); keying on `1_Pooling` alone misses plain `BertModel` exports. The union of rules covers every checkpoint in the detection sweep listed in section 5.

### 2.2 Additive f32 `0 / -inf` masks built from boolean comparisons

The three builders (`create_bidirectional_padding_mask -> [B, 1, 1, L]`, `create_causal_padding_mask(mask, offset) -> [B, 1, L, L + offset]`, `create_bidirectional_window_mask(mask, window) -> [B, 1, L, L]`) share one tail, `additive_from_allowed`, that turns a boolean "attend" array into f32 `0.0 / -inf` with `where_cond`, exactly like `create_causal_mask`. The output is f32 on purpose: `fast_scaled_dot_product_attention` casts a floating mask to the query dtype and `-inf` survives that cast, so the same mask is safe for f16 and bf16 activations (4-bit checkpoints dequantize to f16). A mask built as `(1 - m) * C` with a finite `C` in the activation dtype is explicitly ruled out, because in f16 `-1e9` overflows to `-inf` and `0 * -inf` is NaN.

The causal and window builders apply the same rescue as `create_causal_mask_with_left_padding`: a query row whose allowed key set is empty (a leading padding row in a left-padded batch) keeps its diagonal column, so every softmax row is finite. Padding-row outputs are garbage-but-finite and never consumed because those key positions stay blocked for every real query. The bidirectional padding mask needs no rescue since padding query rows still see every real key. `utils_embedding_mask_tests.rs` checks each rule and casts the masks to f16 and bf16 to confirm the softmax stays finite.

### 2.3 Families own the forward pass and pooling; the engine owns everything else

`EmbeddingModel` returns already-pooled `[B, D]` vectors (or the padded `[B, L, D]` token matrix with zeroed padding rows for multi-vector families). The engine in `src/embeddings/engine.rs` owns per-family text formatting (`format_text`), tokenization, length-sorted micro-batching, the f32 cast, L2 normalization, `dimensions` truncation with re-normalization, shape validation and readback. The trait deliberately requires neither `Send` nor `Sync`: implementors hold MLX array handles and are used from exactly one thread (the worker thread or the `mlxcel embed` main thread).

The shared helpers are what make the family ports small: `pool(hidden, mask, mode)` implements `cls` (argmax of the mask, left-padding aware), `mean` (mask-weighted sum over `max(count, 1e-9)`), `max` (`-inf` fill under padding) and `lasttoken` (highest real index, `L - 1` for an all-padding row); `resolve_pooling_mode` layers `1_Pooling/config.json`, the family default and the `MLXCEL_EMBEDDING_POOLING` override; `load_embedding_weights` applies the subfolder loader and the text bf16-to-f16 rule (non-quantized bf16 converts on Apple Silicon, quantized scales and biases stay bf16); `quantization_params` reads `config.quantization` for `UnifiedLinear::from_weights`.

### 2.4 Explicit errors instead of silent fallbacks

`PoolingConfig::parse` accepts both the new-style `"pooling_mode"` string and the legacy `pooling_mode_*` booleans, but `weightedmean`, `mean_sqrt_len_tokens`, `include_prompt: false` and more than one legacy flag set are load errors naming the mode. `build_family_model` returns `<family> is detected as an embedding checkpoint, but this embedding family is not yet supported by /v1/embeddings` for every recognized variant, and the generation loader bails for `ModelKind::Embedding` with a message pointing at `/v1/embeddings` and `mlxcel embed`. `load_embedding_model` on a generator checkpoint reports `not an embedding checkpoint`. The trade-off is that a checkpoint using an unsupported pooling mode cannot be loaded at all, which is preferable to serving vectors computed with the wrong pooling.

### 2.5 A dedicated worker thread with the audio worker's admission design

`EmbeddingWorker` owns one thread that installs a thread-local default stream, loads the `LoadedEmbeddingModel` on itself and serves commands from a bounded `mpsc::SyncSender`. `try_send` on a full queue returns `EmbeddingError::QueueFull` (HTTP 503), `recv_timeout` returns `EmbeddingError::Timeout` (HTTP 504, the in-flight MLX work is not cancelled), and every engine call runs under `catch_unwind` followed by a stream synchronize so one panicking request does not kill the worker. The model, tokenizer and every array live on the worker thread; the HTTP side only sees `Vec<f32>`. Queue depth and timeout default to the audio values (8 and 120 s). This mirrors `AudioWorker` rather than adding embedding work to the chat scheduler, so chat and embeddings never share a stream.

### 2.6 Startup semantics for one or two checkpoints

`resolve_embedding_source` combines `--embedding-model` with the `-m` detection. `-m <embedding checkpoint>` alone serves the route while the chat worker's `load_model` bails and logs (the Whisper pattern); `-m <chat> --embedding-model <embed>` serves both on separate workers; both together (an embedding checkpoint in `-m` plus `--embedding-model`) is a startup error before the listener binds. A load failure of an explicit `--embedding-model` is fatal; a load failure of an `-m` embedding checkpoint logs and leaves the slot empty, so the route answers a structured 501. `/v1/models` lists the embedding model next to the chat model only when the two ids differ. `MLXCEL_EMBEDDING_MODEL` is layered over clap's `LLAMA_ARG_EMBEDDING_MODEL` with the existing precedence rule (flag or `LLAMA_ARG_*` wins, a conflicting alias is logged and ignored).

### 2.7 Integrate the route before any family exists

`src/embeddings/stub.rs` is a `#[cfg(test)]` model returning mean-pooled one-hot token embeddings, built through the same `finish_loaded_model` tail as production. The route tests, engine tests and worker tests run against it, so string, list, token-id, base64, `dimensions`, empty input, 501, model mismatch, usage counting, image rejection and multi-vector output were all verified before the first family port. This is the reason the family sub-issues can be validated with `mlxcel embed` alone.

### 2.8 Length limits and tokenizer hygiene

`max_length` is the smallest of `sentence_bert_config.json` `max_seq_length`, `tokenizer_config.json` `model_max_length` when `0 < value < 1_000_000` (the HuggingFace unset sentinel is ignored), `config.json` `max_position_embeddings` for absolute-position families only (BERT, XLM-RoBERTa, SigLIP), the hard cap 8192 and `--embedding-max-length`. `strip_padding_and_truncation` removes the fixed padding and truncation some `tokenizer.json` files bake in (MiniLM pads to 128), because the engine pads per micro-batch and truncates per checkpoint limit. Truncation keeps the trailing special token the tokenizer appended, so a BERT input keeps `[SEP]` and a Qwen3-Embedding input keeps `<|endoftext|>`; verbatim token-id inputs are truncated from the right without special-token bookkeeping.

---

## 3. Implementation Details

### 3.1 Request flow

```
POST /v1/embeddings
  -> routes/embeddings.rs: parse body, 501 if no provider, model id check,
     encoding_format, dimensions, per-item validation, image fetch + decode
  -> spawn_blocking(embed_items)
       -> EmbeddingModelProvider (EmbeddingWorkerProvider)
            -> bounded SyncSender -> embedding worker thread
                 -> EmbeddingEngine: format_text, encode_row, length sort,
                    micro-batches of --embedding-batch-size, EncodedBatch
                    -> EmbeddingModel::embed (family forward pass)
                    -> astype f32, normalize_l2, truncate_dimensions,
                       normalize_l2 again, array_to_vec_f32
       <- EmbedReply { vectors, prompt_tokens } written back in request order
  <- EmbeddingsResponse { data, model, usage }
```

### 3.2 Changes by module

| Area | Change |
|------|--------|
| `src/lib/mlxcel-core/src/utils.rs` | `create_bidirectional_padding_mask`, `create_causal_padding_mask`, `create_bidirectional_window_mask`, the shared `additive_from_allowed` tail, and the `array_to_vec_f32` readback helper. |
| `src/lib/mlxcel-core/src/weights.rs` | `load_weights_from_dir_with_subfolders`: top-level shards as before, then every non-hidden immediate subdirectory in sorted order with tensor names prefixed `<folder>.`; `consolidated.safetensors` and `adapter_model.safetensors` are skipped at both levels. |
| `src/embeddings/model.rs` | `EmbeddingModel` trait, `EmbeddingBatch` (`input_ids`, `attention_mask`, optional `token_type_ids`, optional images), `EmbeddingOutput`, `ImageInput`. |
| `src/embeddings/pooling.rs` | `PoolingMode`, `PoolingConfig::read` and `parse`, `resolve_pooling_mode`, `pool`, `normalize_l2`, `truncate_dimensions`. |
| `src/embeddings/limits.rs` | `EmbeddingLimits::derive`, `derive_max_length`, `resolve_pad_token_id` (`pad_token`, then `eos_token`, then 0), `resolve_vocab_size` (`vocab_size`, then `text_config.vocab_size`, then the tokenizer), `config_normalize_flag`. |
| `src/embeddings/tokenize.rs` | `EncodedRow`, `EncodedBatch::from_rows` (right padding, optional fixed width), `encode_row` with HuggingFace special-token masks or an inferred mask for other tokenizers, `encode_pair_row` and `encode_pairs` for the reranker issue, `strip_padding_and_truncation`. |
| `src/embeddings/loader.rs` | `load_embedding_model[_with_options]`, `build_family_model` (the per-family dispatcher), `load_embedding_weights`, `quantization_params`, `finish_loaded_model`. |
| `src/embeddings/engine.rs` | `EmbeddingEngine` with `embed_texts`, `embed_tokens`, `embed_image`, `run_rows`, `forward_batch`, `postprocess`; `EmbedOptions`, `EmbedReply`, `EmbeddingVector`, `EmbeddingEngineError`. |
| `src/model_metadata.rs`, `src/models/mod.rs` | `ModelKind::Embedding`, `is_embedding_model_type`, thirteen `ModelType` variants under the `Embedding` family in `mlxcel arch`, registrations with `weight: None` and an adapter rejection message. |
| `src/models/detection.rs` | `is_embedding_checkpoint` and the rules in section 2.1, called before the `model_type` match. |
| `src/loading/mod.rs` | `load_model` bails for `ModelKind::Embedding` with the route pointer. |
| `src/distributed/tensor_parallel/inference.rs` | Placeholder arms so the fallback architecture table stays total. |
| `src/server/embedding_model.rs` | `EmbeddingModelProvider` trait and `EmbeddingError { QueueFull, Timeout, InvalidInput, Internal }`. |
| `src/server/embedding_worker.rs` | `EmbeddingWorker`, `worker_loop`, `run_guarded`, `EmbeddingWorkerProvider`. |
| `src/server/routes/embeddings.rs`, `src/server/types/embeddings.rs` | The handler, validation order, error mapping, the untagged `EmbeddingInput` enum, `EmbeddingEncoding`, base64 helpers, `EmbeddingData::from_vector`. |
| `src/server/app.rs`, `routes/models.rs`, `state.rs`, `startup.rs`, `config.rs`, `cli_input.rs` | Route mounting for `/v1/embeddings` and `/embeddings`, the `/v1/models` entry, the `embedding_model` slot, `resolve_embedding_source` and `resolve_embedding_provider`, the five config fields and their defaults, `env_fallback_embedding_model`. |
| `src/bin/mlx_server.rs`, `src/main.rs`, `src/commands/serve.rs` | `--embedding-model`, `--embedding-batch-size`, `--embedding-max-length`, `--embedding-queue-depth`, `--embedding-request-timeout-secs` on both binaries; `--embedding-model` resolves through the same store lookup and auto-download as `-m`. |
| `src/commands/embed.rs` | `mlxcel embed -m <path> -p ... [--image ...] [--instruction] [--dimensions] [--max-length] [--batch-size] [--json]`, one vector per line plus the cosine matrix (MaxSim averaged over query rows for multi-vector output). |
| `docs/embeddings.md`, `docs/supported-models.md`, `docs/environment-variables.md`, `docs/README.md`, `README.md`, `CONTRIBUTING.md` | The endpoint, detection rules, pooling, limits, masks, flags, CLI and the "Adding a family" checklist; an empty Embedding models table the family ports fill in. |

### 3.3 HTTP contract

`input` accepts a string, a list of strings, a token-id array, a list of token-id arrays, or a list of typed parts (`{type: text}` and `{type: image_url}`). Token-id inputs are used verbatim (no special tokens added) and every id must be below `vocab_size`. `encoding_format: base64` encodes little-endian f32 bytes with standard padding; for multi-vector models the float form is a list of rows and the base64 form carries a sibling `shape: [num_real_tokens, D]`. `usage.prompt_tokens` counts real tokens including special tokens and `total_tokens` equals it. `image_url` items go through the shared media limits (`try_read_image_url_with_limits`, `decode_request_images_with_limits`) and are embedded one at a time.

| Status | Type | Trigger |
|--------|------|---------|
| 400 | `invalid_request_error` | Malformed body, empty `input`, empty string or token list, token id at or above `vocab_size`, image for a text-only model, `dimensions` outside `1..=D`, unsupported `encoding_format`, `model` not matching the served id |
| 501 | `not_implemented` | No embedding model loaded |
| 503 | `server_busy` | Worker queue full |
| 504 | `server_timeout` | Worker reply timeout |
| 500 | `server_error` | Forward pass failure or a recovered worker panic |

---

## 4. Security, Performance and Quality Review

### 4.1 Input handling

Every request field is validated on the HTTP side before the worker is touched, in a fixed order, and again inside the engine (empty strings, out-of-vocabulary ids, `dimensions`, images for text-only families), so the CLI and the server share the same rejections. Image payloads reuse the existing size and dimension limits of the chat image path. The worker is the only place MLX arrays are created for embedding requests, and its bounded queue plus timeout give the route the same load-shedding behavior as the audio routes. No credentials, file writes or new network paths are introduced beyond the existing model download used by `-m`.

### 4.2 Performance characteristics

Text items are sorted by token length and cut into micro-batches so padding waste is bounded by the length spread inside one batch rather than the whole request; the write-back restores request order. The forward pass output is cast to f32 once and normalized and truncated on the device before a single readback per micro-batch. Images are not batched. The `mean` and `lasttoken` pooling paths avoid data-dependent host loops by using `argmax`, `where_cond` and `take_along_axis`. Detection now reads `modules.json` and probes `1_Pooling/config.json` for every checkpoint; both are small file operations at load time.

### 4.3 Compatibility

- Breaking changes: none for existing generation and audio paths. Embedding checkpoints that previously misrouted to a generator (and failed) now detect as embedders and fail with a `not yet supported` message until their port lands.
- New dependencies: none; `base64`, `image`, `thiserror` and `safetensors` were already in the workspace.
- New `ModelType` variants extend `ALL_MODEL_TYPES`, the `mlxcel arch` family order and the tensor-parallel fallback table, so every exhaustive match remains total.

### 4.4 Test coverage

The PR adds 84 test functions (70 synchronous and 14 async). Coverage by area: five mask tests in `mlxcel-core` including the f16 and bf16 finiteness check; pooling tests for every mode under left and right padding, both config key styles, each unsupported mode, the resolution order and the env override; tokenizer tests for padding, trailing-special truncation, `token_type_ids` and the stripped built-in padding; loader tests for the subfolder walker, quantization params, `max_length` derivation, pad and vocab resolution and the three rejection paths; engine tests for order preservation across micro-batches, `dimensions`, multi-vector rows and image rejection; worker tests for readiness, loader failure, panic recovery, timeout and queue-full; route tests for every input shape and error code; detection tests for each rule and each negative rule; startup and CLI tests for the config plumbing; and eight real-checkpoint gates that soft-skip when the checkpoint is absent.

---

## 5. Verification Evidence

From the PR's test plan and real-checkpoint section (executed on a CUDA build with `--profile test-fast`):

| Validation | Result |
|------------|--------|
| `cargo test --lib` over `server::embedding_worker`, `server::routes::embeddings`, `models::detection_tests`, `model_metadata_tests`, `models::tests`, `embeddings::`, `server::cli_input_tests`, `server::startup_tests` | 115 passed |
| `cargo test --lib -- embeddings::real_checkpoint_tests` | 8 passed with both checkpoints present |
| `cargo test -p mlxcel-core --lib embedding_mask_tests` | 5 passed |
| `cargo test --bin mlxcel -- embed serve_tests` | 3 passed |
| `cargo check --all-targets`, `cargo clippy --lib --bins --tests -- -D warnings`, `cargo fmt --all -- --check` | Passed |
| Workspace test run on the Metal CI runner | Not run at merge time |

Real-checkpoint observations recorded in the PR:

- `sentence-transformers/all-MiniLM-L6-v2` detects as `Bert` via `model_type: bert` and `1_Pooling/config.json`; pooling parses to `mean`; `max_length` derives to 256 from `sentence_bert_config.json`; the subfolder loader reads `embeddings.word_embeddings.weight` `[30522, 384]` and prefixes a synthetic `2_Dense/model.safetensors` as `2_Dense.linear.weight`; the batch tokenizer right-pads with `[PAD]` (id 0), no longer pads to the fixed 128 baked into `tokenizer.json`, and keeps `[SEP]` under truncation; `load_model` bails with the `/v1/embeddings` message.
- `Qwen/Qwen3-Embedding-0.6B` detects as `Qwen3Embedding` via `modules.json` and `1_Pooling` with `pooling_mode_lasttoken`; `max_length` derives to the 8192 cap; the pad token resolves to `<|endoftext|>` (151643) and truncation keeps it.
- A detection sweep over twelve locally present embedding checkpoints (including `multilingual-e5-small`, `bge-m3`, `modernbert-embed-base`, `siglip-base-patch16-224`, `embeddinggemma-300m-4bit`, `Qwen3-VL-Embedding-2B`, `LFM2.5-Embedding-350M`, the Nemotron embedders and the ColSmolVLM and ColQwen2.5 retrievers) resolves each to its family, and five rerankers never route to an embedding variant.
- `mlxcel-server -m all-MiniLM-L6-v2` end to end: chat stays unloaded, `/v1/models` lists the checkpoint, `/v1/embeddings` and `/embeddings` answer the structured 501, a malformed body answers 400; the two-embedding-models and unported-family startup errors fire as designed.

No review threads were recorded on the PR before the squash merge.

---

## 6. Learning Points

- Embedding exports frequently reuse a generator's `model_type`, so a detector that only reads `model_type` cannot tell Qwen3-Embedding from Qwen3. Layout signals (`1_Pooling`, `modules.json`, `architectures[0]`, bidirectional flags) must be checked first, and the reranker exclusions must be explicit.
- An additive attention mask must be built from booleans with `where_cond` into f32 `0 / -inf`. The arithmetic form `(1 - m) * C` breaks in f16 because the large constant overflows and `0 * -inf` is NaN.
- Any mask that can fully block a query row needs a rescue (keep the diagonal) so softmax stays finite; padding rows then produce finite garbage that is never consumed.
- Sentence-transformers tokenizers can bake fixed padding and truncation into `tokenizer.json`; a batching engine must strip those or every input is padded to the fixed width.
- Truncating an embedding input must preserve the trailing special token the model was trained to pool from (`[SEP]`, `<|endoftext|>`), which a plain right cut would discard.
- A test-only model implementing the production trait lets the full request path be integrated and verified before the first real family lands, and keeps every family port's diff limited to its forward pass.

---

## 7. Change Summary

| Item | Value |
|------|-------|
| Files changed | 52 |
| Lines added | 6,689 |
| Lines deleted | 23 |
| New modules | `src/embeddings/` (6 modules plus stub and 5 test siblings), `src/server/embedding_model.rs`, `src/server/embedding_worker.rs`, `src/server/routes/embeddings.rs`, `src/server/types/embeddings.rs`, `src/commands/embed.rs`, `docs/embeddings.md` |
| Test functions added | 84 |
| New dependencies | 0 |

| Category | Summary |
|----------|---------|
| Core (`mlxcel-core`) | Three padding-aware mask builders, `array_to_vec_f32`, subfolder safetensors loading |
| Model registry | `ModelKind::Embedding`, 13 `ModelType` variants, layout-based detection, generation-loader bail |
| Embeddings subsystem | Trait, pooling, limits, tokenization, loader dispatcher, micro-batching engine |
| Server | Provider trait, worker thread, `/v1/embeddings` and `/embeddings`, `/v1/models` entry, five flags and env aliases, startup source resolution |
| CLI | `mlxcel embed` |
| Documentation | `docs/embeddings.md`, supported-models and environment-variables sections, README and CONTRIBUTING links |

| Commit | Purpose |
|--------|---------|
| `3634b9a60` (PR head), squash-merged as `b6f77857` | feat(embeddings): pooling, masks, Embedding kind and /v1/embeddings |

---

## 8. Follow-up Actions

### Required

- Land the family forward passes under epic #1348, replacing each `not yet supported` arm in `build_family_model` and filling the Embedding models table in `docs/supported-models.md`.
- Run the workspace test gate on the Metal CI runner; the PR's checks were executed on a CUDA build.
- Implement the reranker path (#1356) on top of `encode_pairs`, which is shipped here but has no consumer yet.

### Remaining unverified

- The mask builders are verified for shape, blocking rules and f16 finiteness in isolation, not yet inside a real attention layer.
- `pool`, `normalize_l2` and the engine are exercised through the stub and synthetic arrays; numerical parity with the Python reference for each family is the responsibility of the family ports.
- Quantized embedding checkpoints exercise `quantization_params` only; `UnifiedLinear::from_weights` integration is deferred to the families.

### Monitoring

- Watch the 503 and 504 rates on `/v1/embeddings` once a family is served; the defaults (`--embedding-queue-depth 8`, `--embedding-request-timeout-secs 120`) are inherited from the audio worker and were not tuned for embedding workloads.
- The reply timeout frees the HTTP thread but does not cancel the in-flight MLX work, the same limitation as the audio worker.

### Future improvements

- Batch image inputs instead of embedding them one at a time.
- Consider `weightedmean` pooling if a target checkpoint requires it; today it is rejected at load.
- The `MLXCEL_EMBEDDING_POOLING` override applies to the server and `mlxcel embed`; it is a debugging aid and should not be relied on in production configurations.

## References

- Issue #1353: pooling, bidirectional masks, Embedding model kind and /v1/embeddings.
- Epic #1348: embedding model family ports.
- Issue #1356: reranker support.
- PR #1408: the implementation.
- `docs/embeddings.md`: endpoint, detection, pooling, limits, flags and the family checklist.
