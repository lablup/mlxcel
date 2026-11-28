# Technical Report: PR #2210 - KV attention dispatch inside the cache

**Date**: 2026-10-07

**Status**: Implemented and verified on GB10; pending merge.

**Languages**: Rust, Markdown (ADR, docs)

**Risk Level**: Medium. Every Qwen3, Llama 3, Helium and DeepSeek V2 attention step now goes through one cache entry. FP16 dense and pool-backed routes run the same ops as before, and real-checkpoint parity is unchanged. Two batched routes (FP16 boundary layers under a non-FP16 paged layout, and Turbo rows) now compute the same attention in a different op order.

## Executive Summary

Before this PR, each model forward decided which attention kernel to run by asking the cache what storage it had (`is_paged_backed()`) and reading the scheduler's `DecodeBatchContext`. Epic #2166 needs one engine where storage is a property of the sequence, not a branch in every model. This PR, Phase 4a (#2171), adds `KVCache::attend` and `cache::attend_batched`, which append the step's K/V and choose the kernel from the storage behind the cache, plus `MlaLatentCache::attend` for the MLA latent cache. Qwen3, Llama 3 (and the families that reuse their attention), Helium and DeepSeek V2's absorbed decode call these entries. ADR 0008 records the design and defers Gemma 3, Llama 4 and the `model_owned` helpers to Phase 4b.

## 1. Problem Statement

- The kernel choice was repeated in each model's `forward_split_attention` and batched forward: pool-backed caches took the pooled paged kernel, dense caches under a paged layout took a dense-pointer paged route (`paged_decode_attention_dense_compat`), Turbo modes took dequant-first variants, everything else took SDPA.
- The CLI and the server ran different kernels for the same model and request because each call site chose its own route.
- Adding or changing a storage meant editing every family that branched on it.

## 2. Change Summary

- **`src/lib/mlxcel-core/src/cache/attend.rs`:**
  - `KVCache::attend(q, k, v, scale, mask)`. Pool-backed cache at a single unmasked token: the #899 pooled entry, which declines before writing. Pool-backed otherwise: pool intercept plus SDPA. Dense Turbo: the dequant-first variants (ADR 0002). Dense FP16/Int8: dense update plus fused SDPA.
  - `attend_batched(q, k, v, caches, scale, mask)`: one pooled launch for the batch when `caches[0]` is pool-backed and the step is a single unmasked token; rows it declines go through `attend` one by one.
- **`MlaLatentCache::attend`** holds the absorbed MLA decode policy: split-KV when enabled and the plan accepts, else `absorbed_decode`.
- **Models:** qwen3, llama3 and helium call `attend` / `attend_batched` and drop the `DecodeBatchContext` parameter from their batched forwards. deepseek_v2's single-token absorbed decode calls `MlaLatentCache::attend`.
- **ADR 0008** and docs (`CONTINUOUS_BATCHING.md`, `turbo-kv-cache.md`, `environment-variables.md`, `architecture.md`).
- **Tests:** `attend_tests.rs` (dispatch per storage, Turbo batched rows vs single-sequence, empty, one-row and mixed-storage batches), MLA `attend` against the decompressed reference.

## 3. Technical Decisions

- **A match on the cache's fields, not a `KvStorage` trait object.** The cache already resolves its quantization mode by matching; a virtual call per layer per token is what ADR 0004 rejected for the hot loop.
- **Storage is chosen where the cache is built.** `CachePool::allocate_with_layout` gives a dense-natural family on the paged backend pool-backed caches only for an FP16 layout. That allocation is the per-sequence storage policy, so `DecodeBatchContext` no longer selects kernels for the families on `attend`. The ADR 0007 measured default (dense for a lone sequence) holds wherever the worker runs at `max_batch_size=1`; the server's `--decode-storage-backend` policy is unchanged, and moving a sequence between storages as the batch grows is out of scope.
- **Remove the dense-pointer route from Qwen3 and Llama 3.** It only served the FP16 layers a non-FP16 paged layout keeps dense (Turbo Boundary-V layers, the last layer under `--kv-skip-last-layer`). The removed C++ entry was a per-row loop of block slices, a concat and SDPA; per-row dense update plus SDPA is the same attention without the concat copy. The kernels stay for the model-owned families until Phase 4b.
- **Accept the Turbo batched numerics change.** Rows the pooled launch declines now take the dequant-first variants, the same route single-sequence decode already used. This moves batched output toward ADR 0007's "same input, same output" invariant instead of away from it; a test pins batched rows bit-identical to the single-sequence call.
- **Keep the per-row retry after a refused batch.** A variant that skipped the pooled attempt per row would fall back to the pool intercept plus gather, which ADR 0001 measured at 2 to 3 times SDPA cost past 4K tokens.

## 4. Validation

- Unit tests: mlxcel-core `cache::attend` (11), `mla::` (42), `cache::paged_batch_decode`, `cache::decode_undo`, `cache::paged_detach`; root modules `models::qwen3::`, `models::llama3::`, `models::helium`, `models::deepseek_v2`, scheduler, prompt-cache, model-owned lookahead, `distributed::disaggregated` (189), each run alone.
- clippy (`-D warnings`, root and mlxcel-core tests), `cargo fmt --check`, the license-header check, `vlm_wrapper_capability_delegation`, `dead_doc_pointers`.
- Real checkpoints, release `mlxcel-engine-parity` under `MLXCEL_SDPA_DETERMINISTIC=1`, before and after, and again after merging `main` (#2204): Qwen3-1.7B 4-bit and Llama-3.2-1B 4-bit, CLI vs server dense vs server paged identical; the only divergences are the documented prompt-cache-hit pairs, unchanged.

## 5. Residual Risks

- **Turbo plus paged batched serving was not run on a real checkpoint.** The boundary-layer and Turbo-row routes are covered by unit tests only. The epic's end-of-run measurement should include one Turbo paged batch for throughput.
- **DeepSeek absorbed decode was not run on a real checkpoint.** It is opt-in (`MLXCEL_MLA_ABSORBED`), deepseek-v2-lite is known broken on GB10, and the change moves existing calls. The block-level unit test covers it.
- **The ignored real-checkpoint paged parity tests** (`paged_scheduler_parity`, `serving_handoff_parity_tests`) compile but were not run.
- **Model-owned families still read `DecodeBatchContext`** until Phase 4b.

## 6. Learning Points

- **"Unreachable" needs a check against every layout, not the default one.** The first draft called the dense-compat route unreachable because FP16 paged layouts get pool-backed caches. Mixed layouts, where most layers are quantized and a few are kept at FP16, still reached it.
- **A new entry point is only finished when production code calls it.** `MlaLatentCache::attend` passed its own tests while DeepSeek still did its own dispatch; the review caught it by searching for callers.

## 7. Related

- Epic #2166, issue #2171, ADR 0007, ADR 0008.
- #2172 (Phase 4b, the deferred families), ADR 0001 (paged gather cost), ADR 0002 (Turbo dequant-first), ADR 0004 (no per-op dynamic dispatch), #899 (pooled paged entry), #2182 (rotating decode-undo log, untouched).
