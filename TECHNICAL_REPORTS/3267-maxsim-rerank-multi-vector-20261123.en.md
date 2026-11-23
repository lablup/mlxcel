# Technical Report: PR #3267 - Score `/v1/rerank` by MaxSim over a multi-vector embedder

**Date**: 2026-11-23
**Status**: Completed
**Risk Level**: Low (active only when the served embedder is multi-vector and no reranker is configured; the probability reranker kinds keep their existing checks)

## Executive Summary

When the served embedding model is multi-vector (LFM2.5-ColBERT, ColIdefics3, ColQwen2.5) and no `--reranker-model` is configured, `POST /v1/rerank` and its aliases (`/rerank`, `/v1/reranking`, `/reranking`) now score documents with late-interaction MaxSim (#1426, follow-up to #1414 and epic #1348). There is no new route and no new response schema. A new `RerankerKind::LateInteraction` (`late_interaction` on `/props`) reports raw MaxSim sums, not probabilities. On M3 Ultra, LFM2.5-ColBERT scores 6.4645 vs 5.6824 (relevant vs unrelated), equal to `mlxcel embed` for both the bf16 and HF layouts. colSmol-256M with its adapter merged scores 19.1788 vs 8.5367 on two rendered pages.

## 1. Problem Statement

#1414 added `crate::embeddings::maxsim`, and `mlxcel embed` printed a MaxSim matrix for the col* embedders. HTTP had no scoring endpoint. A client that wanted "which document answers this query" had to download every document's `[num_tokens, 128]` matrix from `/v1/embeddings` (1776 rows for the two-page ColIdefics3 example) and run MaxSim itself. #3245 added LFM2.5-ColBERT, a text-only multi-vector embedder that is cheap enough to serve as a reranker, which made the gap more visible.

## 2. What Changed

- `src/server/rerank_maxsim.rs` (new): `MaxSimRerankProvider` implements `RerankModelProvider` over the existing `EmbeddingModelProvider`. It embeds the query and documents, then scores each document with `maxsim(query_rows, doc_rows)`. `usage.prompt_tokens` is the sum of the embedding replies, and the server's `--embd-normalize` is applied so the scored rows are the rows `/v1/embeddings` returns. Errors map as follows: queue full to 503, timeout to 504, invalid input to 400, other errors to 500.
- Startup (`resolve_late_interaction_reranker`, called from `startup.rs` after `resolve_rerank_provider`): installs the provider only for a multi-vector embedder with no configured reranker. A configured reranker wins even when it failed to load, and the log then warns that `/v1/rerank` answers 501. The provider shares the embedding worker, so no second `SideWorkerExit` is registered. `check_serving_mode` counts it, so `--reranking` works with a multi-vector `-m`.
- `src/rerank/mod.rs`: `LateInteraction` kind. `accepts_instruction()` is false, and a new `scores_are_probabilities()` is false only for this kind. The reranker loader refuses the kind, because it never comes from a reranker checkpoint.
- `src/server/routes/rerank.rs`: the score check is now `score_is_valid(score, probabilities)`. Every kind must return finite scores; only probability kinds must stay in `[0, 1]`. The `instruction` 400 message names the loaded kind, and the 501 message mentions multi-vector embedders.
- Docs: a "Late-interaction scoring" section in `docs/embeddings.md` covers the raw-score scale, the caller-side `[Q] ` / `[D] ` prefixes for LFM2.5-ColBERT and the adapter merge for colSmol. The rerank error table is updated, and the reranker table in `docs/supported-models.md` gets a new row.
- Tests: stub-embedder route tests check scores against `maxsim` of known rows, `top_n`, `return_documents`, the TEI `texts` envelope, the instruction 400, error mapping, per-kind score ranges, one call for 500 documents, a wrong-count reply and a NaN row. Real-checkpoint gates cover LFM2.5-ColBERT (bf16, HF) and merged colSmol-256M. `tests/fixtures/generate_colsmol_rerank_fixture.py` merges the LoRA adapter (`W + (alpha/r) B A` plus the trained `1_Dense`) and renders the two pages.

## 3. Technical Decisions

**Extend `/v1/rerank` instead of adding `/v1/score`.** Request validation, image fetching, `top_n`, `return_documents` and both response envelopes already exist behind `RerankModelProvider`. A provider plus a kind reuses all of it, and the route needed only the kind-aware messages and range check.

**Raw MaxSim sums, not normalized.** The score scales with the query's token count. Normalizing would hide the margin, and the CLI's numbers would no longer match. The doc comment on `RerankResult.relevance_score` and the docs say so. This is also why the route's `[0, 1]` check had to become per kind: before the change, every MaxSim score (for example 19.18) would have returned 500.

**One batched text call per request (security review).** The first revision made one `embed_texts` call per document. The worker's reply timeout applies per call, and `/v1/rerank` limits document count only through the body size. A single request with thousands of short documents could therefore hold the shared embedding worker for thousands of timeouts back to back, and could hit `QueueFull` partway through after earlier forwards had already run. All text items now go in one call, as `/v1/embeddings` does: one queue slot, one timeout, length-sorted micro-batching. Image items stay one call each, bounded by the existing per-request image cap. A side effect is that LFM2.5 scores now equal the CLI's exactly. The per-item revision differed by 8e-4 because bf16 batched prefill accumulates differently.

**A configured reranker always wins.** If an explicit reranker fails to load and MaxSim takes over silently, the operator gets a different scoring model than the one they asked for. So the failed case answers 501 and the startup log says why.

## 4. Measurements

M3 Ultra, Metal, release `mlxcel-server` and the checkpoint gates.

| Checkpoint | Relevant | Unrelated | `mlxcel embed` MaxSim | prompt_tokens |
|---|---|---|---|---|
| lfm2.5-colbert-350m-bf16 | 6.4645 | 5.6824 | 6.4645 / 5.6824 | 33 |
| lfm2.5-colbert-350m (HF layout) | 6.4645 | 5.6824 | 6.4645 / 5.6824 | 33 |
| colSmol-256M merged, PR #1414 query | 19.1788 | 8.5367 | 19.1788 (per-image path) | 1776 |

## 5. Not Measured

- PR #1414's page images were never committed, so its 18.7396 vs 8.6879 cannot be reproduced. The gate uses rendered stand-in pages and asserts the ordering, a 10% margin and equality with the embed path. The 1776 prompt tokens match #1414's row total, so the page geometry is the same.
- ColQwen2.5 was not run on `/v1/rerank`; it uses the same provider path as ColIdefics3.
- Throughput under concurrent rerank and embeddings load was not measured.

## 6. Learning Points

- A per-item loop over a shared worker turns a bounded per-call timeout into an unbounded per-request hold. Batch at the request boundary, or cap the item count.
- A validity check written for one score domain (probabilities) becomes wrong when a new kind joins the same route. The check belongs to the kind, not to the route.

## 7. Follow-up Actions

- `/v1/rerank` has no document-count or total-token cap for any reranker kind beyond the body limit. A shared cap would bound worst-case request cost for every kind.
- `maxsim` uses `f32::max`, which skips NaN rows. A document with some NaN rows gets a finite score; only an all-NaN document is rejected (as `-inf`).
- Surfaced but not in scope: `mlxcel rerank` accepting a multi-vector embedder, and a `rerank` capability on a side embedder's `/v1/models` entry.
