# PR #2748: Keep NemotronNAS and LongCat caches model-owned

**Date**: 2026-10-30 (generated before merge; the PR was an open draft at the time)
**Status**: In review
**Languages**: Rust, Markdown
**Risk Level**: Low to medium (both families were unusable on the engine path before, so nothing working can regress; the changes add persistent per-sequence state to two models and alter LongCat's MoE router and n-gram embedding math)

## Executive Summary

PR #2748 (closes #2544) makes NemotronNAS and LongCat-Flash / LongCat-Flash-Ngram honour the contract for models that cannot batch: the engine hands `forward` an empty cache slice, so the model keeps its own caches and resets them in `make_caches()`. NemotronNAS panicked with an index out of bounds on its first attention layer through `Engine::prefill`, so neither `mlxcel generate` nor `mlxcel-server` produced text. LongCat rebuilt fresh caches and an empty n-gram context on every call, so each decode step attended to its own token at RoPE position 0. Both now hold their state in a `RefCell` on the model. Getting LongCat to run at all also required two pre-existing forward fixes: a rank-3 router slice that aborted the process on every forward, and an n-gram context truncation that dropped one neighbour per decode step. NemotronNAS was verified end to end on the 49B 4-bit checkpoint on GB10; LongCat has no checkpoint that fits a 128 GB host and is gated by tiny-config equivalence tests only.

## 1. Problem Statement

### The engine contract for non-batching models

A model with `supports_batching() == false` gets a `ModelOwned` pool entry whose cache vector is empty (`SequenceCacheSet::model_owned` in `src/lib/mlxcel-core/src/cache.rs`), and `Engine::prefill` passes that empty slice to `forward`. Such a model must keep its caches on `&self` and reset them when `make_caches()` is called, which the scheduler does at prefill and `DirectEngine::open_sequence` does when a raw completion opens (#2494, PR #2542). ExaOne 4, GPT-OSS, Ministral 3, Step 3.5, Mellum and Laguna already follow this pattern. The security review of PR #2542 found two families that did not.

### NemotronNAS indexed an empty slice

`NemotronNASModel::forward` indexed `caches[cache_idx]` for each block with an attention subblock. The mask code tolerated an empty slice through `caches.first()`, but the layer loop did not, so every prefill through `Engine::prefill` panicked at `nemotron_nas.rs:731`. Since #2225 moved `mlxcel generate` onto the engine, the CLI hit this as well as the server. #1688 shows the family generating text before #2225, when the CLI built and passed its own caches.

### LongCat kept no KV between calls

`LongcatFlashNgramModel::forward` built a fresh `[KVCache; 2]` per layer and set `ngram_context = None` on every call. On a decode step `offset` was 0, so RoPE ran at position 0, `update_and_fetch` returned only the current token's latent K/V, and the n-gram hashes saw only the current token. A single-piece prefill was correct, but prompts longer than the 2048-token prefill chunk lost every earlier piece, and decode was never correct on any path. One struct serves both `longcat_flash` and `longcat_flash_ngram`, so both model types were affected.

### Two LongCat bugs hidden behind the first

Once the tests could reach a LongCat forward, they exposed two older bugs that no prior test had executed:

- **Router slice**: `LongcatFlashTopkRouter` sliced the argpartitioned indices with `slice(&indices, &[0, 0], &[shape[0], top_k])`. The router input is `[B, L, hidden]`, so the indices are rank 3 and MLX threw `[slice] Invalid number of indices`. An MLX exception crossing the cxx bridge is an abort, so every LongCat forward killed the process. On a rank-2 input the same call would have cut the token axis instead of the expert axis.
- **N-gram truncation**: `NgramEmbedding::forward` joined the carried tail with the new tokens and then cut the joined context to `n - 1` tokens before hashing. On a decode step with one new token the context was then `n - 1` long, `shift_right` by `n - 1` returned zeros, and the oldest neighbour of the largest n-gram was lost on every step.

## 2. Technical Decisions

### Model-owned `RefCell` state, not batch support

The issue considered making NemotronNAS batch-capable and rejected it, and the PR follows that. `make_caches` yields one cache per block with an attention subblock (49 on the 49B checkpoint), while `num_layers()` is `num_hidden_layers` (80). `num_layers()` drives the server's KV-mode table and the paged layout, so flipping `supports_batching` would describe the wrong layout and would also enable APC and paged decode for a family that had no checkpoint on the dev hosts when the issue was filed. The PR keeps `supports_batching() == false`, adds `caches: RefCell<Vec<KVCache>>` filled by a shared `fresh_caches(&layers)` helper, and updates `docs/CONTINUOUS_BATCHING.md` to say that a batch-capable layout is a follow-up.

LongCat gets a private `LongcatState { caches: Vec<[KVCache; 2]>, ngram_context: Option<UniquePtr<MlxArray>> }` in a `RefCell`. Keeping the KV caches and the n-gram tail in one struct means one reset clears both; resetting one without the other would let a new request hash against the previous request's last tokens. `forward` destructures the borrowed state and passes both fields to the existing `internal_forward`, which did not change signature.

Both families implement `reset_runtime_state()` and call it from `make_caches()`. `make_caches` is what the engine and scheduler call per sequence; `reset_runtime_state` is the trait hook for a bare single-row run with no `SequenceId` (#2176), and delegating wrappers forward it. The vector `make_caches` returns keeps its old shape but is never read.

### NemotronNAS mask from the first full-attention slot

The issue proposed sizing the causal mask from the internal first cache. The PR does not, because the first slot is not always written. `has_attention()` is true for both full attention and `replace_with_linear` blocks, so a linear replacement block owns a cache slot it never touches. With `caches.first()`, a checkpoint whose first attention-bearing block is a linear replacement reads `live_len() == 0` on every call; the first prefill chunk works, and the second builds an `(L, L)` mask against `(L, L + offset)` scores and aborts with `broadcast_shapes (3,3) vs (1,4,3,6)`. The new `first_attention_slot()` finds the slot index of the first `AttentionSubblock::Attention` block among attention-bearing blocks and reads `live_len()` there, falling back to 0 when no full attention block exists. A mutant that reverts this (together with the n-gram fix below) aborts `nemotron_nas_linear_first_chunked_prefill_matches_full_forward` with that broadcast error.

### LongCat mask from `live_len()`, padded prefill refused

`internal_forward` sized the mask from `caches[0][0].offset`. Offset is monotonic, while `update_and_fetch` returns the live window, so the PR switches to `live_len()` (the #421 behaviour NemotronNAS already had) and also tolerates an empty cache list. `supports_padded_prefill()` now returns `false` for LongCat, and NemotronNAS keeps `false` with a corrected reason: with persistent model-owned caches, pad positions would land in state that the caller's trim never reaches (#1755). The old NemotronNAS comment blamed Mamba recurrent state, which this family does not have; the same mistake in `scheduler_tests.rs`, which listed NemotronNAS among hybrid SSM families, is fixed too.

### Fix the router with `slice_axis` on the last axis

`slice_axis(&indices, -1, 0, top_k)` expresses `[..., :top_k]` on the expert axis regardless of rank. A rank-specific `slice` with three index pairs would have fixed the abort but kept the rank assumption that caused it.

### Cut only the saved n-gram tail

The joined context is now hashed whole, and only the copy saved into `context` for the next call is cut to `n - 1` tokens. The hashes of this call's tokens need up to `n - 1` tokens of history before the first new token, so the joined context must keep all of them. `new_ids` already selects the last `seq_len` positions after hashing, so the longer context does not change output length.

## 3. Change Summary

8 files, +770 / -49. Two commits (`9630310f` for the fix, `942761b4` merging a duplicated `impl NemotronNASModel` block) plus a merge of origin/main (`48071260`).

| File | Change |
|---|---|
| `src/models/nemotron_nas.rs` | `caches: RefCell<Vec<KVCache>>`, `fresh_caches`, `first_attention_slot`; `forward` ignores the caller's slice; `make_caches` and `reset_runtime_state` reset; comments on padded prefill and batching corrected. |
| `src/models/longcat_flash_ngram.rs` | `LongcatState` in a `RefCell`; `forward` uses it; `make_caches` and `reset_runtime_state` reset it; mask from `live_len()`; `supports_padded_prefill() == false`; router `slice_axis` fix; n-gram context truncation fix. |
| `src/models/nemotron_nas_engine_tests.rs` (new) | 4 tests on tiny f32 configs covering full attention at three GQA widths, a `no_op` block, and a `replace_with_linear` block, in a mixed order and a linear-first order. |
| `src/models/longcat_flash_ngram_engine_tests.rs` (new) | 6 tests over both variants, built through `sanitize_weights` with deterministic fixture weights inserted in a fixed order (#1265). |
| `src/lib/mlxcel-core/src/engine/direct.rs`, `shared_cache_tests.rs` | Both families added to the model-owned family lists in the doc comments. |
| `src/server/batch/scheduler_tests.rs` | Doc comment no longer calls NemotronNAS a recurrent-state family. |
| `docs/CONTINUOUS_BATCHING.md` | The class D exclusion now says both families keep one model-owned cache set and names the NemotronNAS layout follow-up. |

## 4. Trade-offs and Residuals

- **LongCat decode now costs what it should.** Each decode step attends over every cached position in both sub-layers of each layer, and the KV grows by `kv_lora_rank + qk_rope_head_dim` values per token per sub-layer. Before, each step attended to one key. Tok/s will fall with context length; nothing measured it because no checkpoint fits.
- **LongCat is unverified on real weights.** The 560B checkpoint does not fit a 128 GB host. The tiny-config tests prove incremental, chunked and engine paths match a full-sequence forward, but they cannot catch a loader or numerics problem specific to the real checkpoint.
- **NemotronNAS batching stays off.** The cache-per-attention-block versus `num_layers()` mismatch is documented, not fixed.
- **Shared with every model-owned family** (ExaOne 4, GPT-OSS, Mellum, and others), and out of scope: `--kv-cache-mode` and `--max-kv-size` do not reach the internal caches, and the last request's KV stays resident until the next prefill resets it. For LongCat this is new memory residency, since the old code dropped its caches at the end of each call.
- **Single-sequence by construction.** The `RefCell` holds one sequence. That is safe only because `supports_batching() == false` keeps the scheduler single-row; a future change that enables batching for either family must replace this state first.
- **Out of scope from the issue**: `NASMLP::forward` maps every activation other than `silu` / `relu` / `gelu` to `silu`, tracked with #2423.

## 5. Validation

The new tests drive each model the way the engine does: through `DirectEngine::run`, and through `forward` with an empty slice. The reference is a full-sequence forward on fresh state, compared at max abs diff 1e-3 on f32 weights, and a greedy decode that recomputes from scratch each step.

| Test | main 8385f40b | PR |
|---|---|---|
| `nemotron_nas_direct_engine_run_completes` | panic, index out of bounds | ok |
| `nemotron_nas_consecutive_direct_engine_runs_match_recompute` | same panic | ok |
| `nemotron_nas_incremental_logits_match_full_forward` | same panic | ok |
| `nemotron_nas_linear_first_chunked_prefill_matches_full_forward` | same panic | ok |
| `longcat_flash{,_ngram}_incremental_logits_match_full_forward` | abort, `[slice] Invalid number of indices` | ok |
| `longcat_flash{,_ngram}_consecutive_direct_engine_runs_match_recompute` | same abort | ok |
| `longcat_flash_ngram_chunked_engine_prefill_matches_single_piece` | same abort | ok |
| `longcat_flash_refuses_padded_prefill` | assert fails | ok |

The tests isolate each fix. With only the router fix applied, the cache bug shows on its own: incremental versus full max abs diff is 5.65 (flash) and 6.76 (ngram). A mutant that reverts only the n-gram truncation fix and the linear-first mask fix fails `longcat_flash_ngram_*` (diff 5.14) and aborts the NemotronNAS linear-first test.

Local gate on GB10 (`--profile test-fast --features cuda`, test binaries under `gpu-lock`, `MLX_ENABLE_TF32=0`): `nemotron_nas` 6/6, `longcat_flash` 8/8 at the final head, `hybrid_ssm_workers_never_use_paged_decode_backend`, mlxcel-core `engine::` 71/71, `cargo fmt --all -- --check`, and `cargo clippy --lib --tests -- -D warnings` on `mlxcel` and `mlxcel-core`. Metal and ROCm were not run locally.

Real checkpoint: `mlx-community/Llama-3_3-Nemotron-Super-49B-v1-mlx-4bit` (`model_type: nemotron-nas`, 80 layers, 26.1 GB resident) on GB10. `mlxcel generate` with `--temp 0` produced a coherent 96-token answer on Rayleigh scattering at 7.06 tok/s (one run, not a benchmark). Two identical greedy `/v1/chat/completions` requests to `mlxcel-server` returned byte-identical, correct text, which shows the per-request reset on the server path. On main both paths panicked on the first attention layer.

## 6. Learning Points

- **An empty cache slice is a contract, not a bug.** For a non-batching model the engine's pool entry is model-owned and empty by design. Any model that sets `supports_batching() == false` and still indexes the caller's slice will panic on the engine path, even if it worked when the CLI built its own caches. Copy the ExaOne 4 pattern: `RefCell` state, borrow it in `forward`, reset it in `make_caches` and `reset_runtime_state`.
- **Pick the mask's reference cache from a slot that is written.** In architectures where some layers own a cache slot they never fill (linear replacements, `no_op` attention, sliding-window layers with a different live length), `caches.first()` is not a safe proxy for the sequence length. The failure appears only on the second prefill chunk, so a test must include a chunk after the first.
- **Test what the model computes, not only that it runs.** LongCat's router abort and n-gram truncation were in code no test had executed. Equivalence between an incremental run and a full-sequence forward on fresh state caught all three LongCat bugs, and per-fix mutants showed each test detects its own bug.
- **MLX exceptions abort across the bridge.** A shape error inside an MLX op is not a recoverable Rust error here; it terminates the process. Rank assumptions in `slice` calls deserve `slice_axis` on a named axis.
