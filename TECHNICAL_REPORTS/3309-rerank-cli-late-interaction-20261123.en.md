# PR #3309: Rerank with a multi-vector embedder in `mlxcel rerank`

**Date**: 2026-11-23
**Status**: PR open (closes #3300). Unit and real-checkpoint suites pass; CLI and `/v1/rerank` scores are bit-identical on LFM2.5-ColBERT-350M and the merged colSmol-256M on an M3 Ultra

## Executive Summary

Since #3267, `mlxcel-server -m <multi-vector embedder>` serves `/v1/rerank` by MaxSim when no reranker is configured. `mlxcel rerank -m` on the same checkpoint failed with "is not a reranker checkpoint", because its loader only knew the cross-encoder and the two Qwen3 generative kinds. The server and the offline validation tool therefore disagreed on which checkpoints can rerank.

This PR moves the scoring half of the server's `MaxSimRerankProvider` into a shared library function, `rerank::score_late_interaction`, and runs it from both sides: over the embedding worker in the server and over an `EmbeddingEngine` on the calling thread in the CLI. `mlxcel rerank` now loads a checkpoint as an embedder when it is not a reranker and, when the embedder is multi-vector, scores by MaxSim with `kind: "late_interaction"`. Every other checkpoint keeps today's error.

## 1. Problem Statement

`src/commands/rerank.rs` called `load_reranker_with_options`, whose `detect_reranker_kind` accepts `ForSequenceClassification` heads on BERT, XLM-RoBERTa and ModernBERT, plus `qwen3` and `qwen3_vl`. LFM2.5-ColBERT (`model_type: lfm2`), ColIdefics3 (`idefics3`) and ColQwen2.5 fell through to `Unsupported reranker model type`. The server reached these checkpoints through a separate startup branch (`resolve_late_interaction_reranker`), so there was no single place where "this checkpoint can rerank" was decided, and the CLI could not be used to check a `/v1/rerank` result offline for this kind.

## 2. Change Summary

| Area | Change |
|---|---|
| `src/rerank/late_interaction.rs` (new) | `MultiVectorEmbedder` trait (`supports_images`, `embed_texts`, `embed_image`), `score_late_interaction` (moved from the server provider), `LateInteractionError`, and `LateInteractionReranker`, a `Reranker` over an `EmbeddingEngine`. `EmbeddingEngine` implements the trait. |
| `src/server/rerank_maxsim.rs` | `MaxSimRerankProvider::rerank` calls `score_late_interaction` through a `WorkerEmbedder` adapter over `EmbeddingModelProvider`. `embed_items` and `single_vector` are removed (moved, not duplicated). Error mapping and messages are unchanged. |
| `src/rerank/loader.rs` | `load_reranker_or_late_interaction` and the testable tail `late_interaction_from_embedder`. |
| `src/commands/rerank.rs` | Uses the new loader. `--instruction` refusal names the kind. Help text for `-m`, `--batch-size` and `--max-length` covers the embedder case. |
| Tests | `rerank/late_interaction_tests.rs` (9 tests) and two real-checkpoint gates in `server/rerank_maxsim_checkpoint_tests.rs`. |
| Docs | `docs/embeddings.md` (source map, Late-interaction scoring, `mlxcel rerank`), `docs/supported-models.md` (late-interaction row). |

## 3. Technical Decisions

### 3.1 One scoring function behind a small trait

The server's provider owned the whole rule: put every text item in one `embed_texts` call, embed each image in its own call, check the reply counts, then run `embeddings::maxsim` per document. The rule depends only on three embedder operations, so it now takes a `MultiVectorEmbedder` with an associated error type. The server adapter returns `EmbeddingError` and maps it onto `RerankError` exactly as before (queue full to 503, timeout to 504, invalid input to 400, internal to 500); the CLI's engine returns `EmbeddingEngineError`. The alternative, a CLI-side copy of the loop, would have let the two paths drift in batching, which changes scores (see 3.3).

### 3.2 Detection order and cost

`load_reranker_or_late_interaction` tries reranker detection first, so every reranker checkpoint loads exactly as it did. When detection fails, it asks `get_model_type` plus `is_embedding_model_type`, the same pair the embedding loader uses. A chat checkpoint fails that check from `config.json` alone and returns the original detection error, with no weight read. An embedding checkpoint is loaded, and the multi-vector flag is read from the loaded limits; a single-vector embedder is then refused with the original error. That costs a full load on a path that errors, which is acceptable for an offline tool and matches the server, which also decides from the loaded model. If the embedder itself fails to load, the error says both that the checkpoint is not a reranker and that loading it as a multi-vector embedder failed, rather than hiding the real cause behind the detection message.

### 3.3 Matching the server's batching

Length-sorted micro-batching pads rows inside a batch, so batch composition can move scores slightly. The CLI defaults the embedding batch width to `DEFAULT_EMBEDDING_BATCH_SIZE` (16), the server's `--embedding-batch-size` default, rather than the reranker kinds' default of 8. With the same width and the same single `embed_texts` call, both paths run identical forwards, and the measured scores agree to the last bit. `--batch-size` still overrides the width. Embed options are the defaults, which equal a server started without `--embd-normalize`.

## 4. Verification

| Check | Result |
|---|---|
| `cargo clippy --lib --tests --bins --features metal,accelerate -- -D warnings` | clean |
| `rerank::` and `server::rerank_maxsim` lib tests (serial, under the GPU lock) | 84 passed |
| Real-checkpoint gate, LFM2.5-ColBERT-350M, CLI loader vs server provider | `[5.6823516, 6.4644585]` on both, prompt tokens 33 on both |
| Real-checkpoint gate, merged colSmol-256M, text query vs one text and two page documents | `[8.537709, 22.494436, 19.177414]` on both, prompt tokens 1804 on both |
| Release binaries: `mlxcel rerank --json` vs `curl /v1/rerank` on `mlxcel-server -m` | LFM2.5-ColBERT: identical scores for three documents (5.6823516, 6.4644585, 6.31933), 41 tokens. colSmol merged: identical scores for one text and two image documents (22.494436, 19.177414, 8.537709), 1804 tokens. `kind` is `late_interaction` |
| Refusals | `--instruction` on colSmol: refused naming MaxSim. `--image` on LFM2.5-ColBERT: "text-only". `sentence-transformers/all-MiniLM-L6-v2` (single-vector) and `k2-horizon-0.9b` (chat): both fail with the existing "is not a reranker checkpoint ... Unsupported reranker model type" message |

The colSmol adapter (`vidore/colSmol-256M`) is LoRA-only and is not loadable by itself; the gates use the merged directory written by `tests/fixtures/generate_colsmol_rerank_fixture.py`, as the server documentation already requires.

## 5. Limits and Follow-ups

- The CLI has no `--embd-normalize` or `--pooling`; the checkpoint's own normalization applies. Pooling does not affect the multi-vector families.
- A single-vector embedder is loaded in full before it is refused (3.2).
- The new gates soft-skip when the checkpoints are absent, like the existing gates in that file.
