# Technical Report: PR #1390 - stop the prompt cache donating and adopting KV-less shadow paged entries

**Date**: 2026-08-23
**Author**: Jeongkyu Shin
**Status**: Completed
**Languages**: Rust
**Risk Level**: High before the fix (silent history loss on turn 2 for two model families under the default server config); the change itself is narrowly scoped

---

## Executive Summary

Under the default server configuration a Gemma 3 or Llama 4 sequence is allocated on the paged backend purely for shadow block-table accounting, while its real K/V lives in the model's `ModelOwnedSequenceState`. The prompt cache did not know that. It detached the shadow block table as a paged entry, and the next request extending the conversation adopted it: the scheduler reported `cached=160/190`, skipped prefill for those 160 tokens, and the model, whose internal caches for the new sequence were fresh, decoded from the 30-token suffix alone.

The user-visible symptom is that **turn 2 answers without its history**, with no error and no warning. On `gemma3-1b-4bit` the second turn of a three-turn conversation returned "I understand. You're repeating yourself. It's a very frustrating process" where the same conversation with `--no-prompt-cache` returned a coherent reply.

This unit is unusual in that the acceptance test existed before the fix did. A multi-turn differential harness built earlier in this batch already reproduced the defect on two checkpoints, so turning those runs green is the proof, and no argument from unit tests was needed.

## 1. Problem Statement

### 1.1 Background

Three facts had to line up for the corruption:

1. **Allocation.** Under the paged override a model-owned family gets backend `PagedKvCache` with `model.make_caches()` placeholders, which for Gemma 3 and Llama 4 is an empty vector. `pool.append_tokens` still allocates real pool blocks for the shadow table, so `retained_block_count() > 0` and `seq_len() > 0`.
2. **Donation.** The only model-owned exit in `donate_finished_sequence_cache` was `supports_snapshot_reuse()`, which Gemma 3, AFMoE and Llama 4 all return `false` from, followed by a check of the *allocated* backend, which reads `PagedKvCache` rather than `ModelOwned`.
3. **Emptiness.** `DetachedKvSet::is_empty` for the paged arm tested `seq_len() == 0 || retained_block_count() == 0`. Both are false for a shadow entry, so it was inserted and later adopted.

### 1.2 Existing Issues

The failure is silent by construction. Nothing throws, nothing logs, and the output is fluent, so it reads as the model being weak at following a conversation rather than as the server dropping its context.

### 1.3 Risk Assessment

High before the fix, on the default configuration, for two families that are both in the project's recommended-test set. The change itself is narrow: the review established that for a natural-`ModelOwned` family to have donated before this PR, its allocated backend had to be non-`ModelOwned`, which requires the paged override, which requires `supports_batching() && supports_paged_decode_backend()`. That set is exactly Gemma 3, Llama 4, and Qwen 3.5, and Qwen 3.5 exits earlier through the snapshot branch. There is no configuration in which this gate takes away a working cache.

## 2. Change Summary

12 files, roughly 710 insertions.

| Area | Change |
|---|---|
| `src/server/batch/scheduler.rs` | Donate gate and adopt gate on the model's natural backend |
| `src/lib/mlxcel-core/src/cache/paged_detach.rs` | `detach_paged` declines a handle-less paged sequence before pinning any block; `clone_eligible` requires dense handles |
| `src/server/prompt_cache/entry.rs` | `is_empty` treats a zero-handle paged set as empty, via an extracted `paged_set_is_empty` |
| `src/server/prompt_cache/metrics.rs`, `routes/cache.rs`, `routes/metrics.rs` | New `PromptCacheRejectReason::ModelOwnedState`, surfaced as `reject_model_owned_state` and a `/metrics` label |
| `docs/turbo-kv-cache.md` | Records that model-owned families are routed through the paged backend for accounting only |
| Tests | `scheduler_model_owned_cache_tests.rs` (new), `paged_detach_tests.rs`, `store_tests.rs`, `cache_tests.rs` |

## 3. Technical Decisions

### 3.1 The predicate is the model's natural backend, not the allocated one and not the override

This is the whole fix, and picking the wrong one of the three fails in opposite directions: gating on the allocated backend changes nothing (it is the value that lies), and gating on the override disables the cache for families that should keep it.

The choice was settled by reading the code rather than by argument, and there were two in-tree precedents:

- `CachePool::allocate_with_layout` already reads `model.sequence_state_layout().backend`, names it `natural_backend`, and uses it to decide pool-backing precisely because the override may disagree.
- `BatchScheduler::handoff_supported` gates the disaggregated pool-block KV handoff on the same underlying fact, for the same reason, from #708.

`sequence_state_layout_override` was ruled out on inspection: it is built from server config alone (`decode_storage_backend`, `num_layers`, the KV cache mode, the block size) and never asks the model what it does with K/V. It is also `None` on the dense backend, where a model-owned family still needs the same refusal.

The disagreement between the two values is now pinned by a test rather than described in a comment: `paged_override_does_not_change_the_model_owned_natural_backend` builds a real `Gemma3Wrapper`, asserts the model reports `ModelOwned` while the override reports `PagedKvCache`, and asserts the allocated sequence carries zero per-layer caches.

### 3.2 Defense in depth, stated accurately

The change adds five terms, and it is worth being precise about what each buys, because the first draft of the PR body overstated it as "any one of them stops the corruption".

Three are independently sufficient: the donate gate, the adopt gate, and `detach_paged` declining a handle-less sequence.

Two are jointly sufficient on the adopt side rather than individually. `clone_eligible` alone does not stop adoption, because a clone-ineligible paged set falls through to the take path where `is_empty` is what stops it; and `is_empty` alone does not stop the clone path, which never consults it.

`detach_paged` declines inside the read-only borrow, before `self.active.remove` and well before the `retain_block` loop, so the decline is inert on the pool budget. The test asserts both the active count and unchanged refcounts, which locks that property in rather than leaving it to the reading.

### 3.3 A mis-declared layout now fails in the safe direction

A consequence worth recording: a family that wrongly reports `model_owned` while actually using the external caches now loses its prompt cache instead of returning wrong answers. The codebase already treats this declaration as load-bearing (`phi4mm_vl.rs` and `falcon_ocr.rs` both carry explicit dense overrides with comments describing exactly that hazard), so the gate inherits an existing invariant rather than inventing one.

## 4. Verification

### 4.1 The harness that already reproduced the defect

A multi-turn differential harness built earlier in this batch runs a three-turn conversation twice, once with the prompt cache on and once with `--no-prompt-cache`, and compares every generated token, excusing a divergence only when that step's own top-2 logprob margin is below a jitter floor of 0.05.

| Checkpoint | Before | After |
|---|---|---|
| `models/gemma3-1b-4bit` | FAIL, turn 2 step 0, margin 0.516 | **PASS, all turns identical** |
| `models/gemma-3-4b-it-4bit` | FAIL, turn 3 step 0, margin 0.203 | **PASS, all turns identical** |
| `models/llama-4-scout-17b-4bit` | not measured | **PASS** |
| `models/llama-3.1-8b-4bit` | PASS | PASS, unchanged |
| `models/internlm3-8b-4bit` | PASS, tie excused | PASS, same tie, same margin 0.00000 |

The harness was not modified to obtain this. The internlm3 excusal is the same exact-tie flip at the same margin it showed before the change, which is the control confirming the comparison itself did not move.

`llama-4-scout-17b-4bit` was a bonus: the brief assumed no local checkpoint for the second family the issue names, and there was one.

### 4.2 The cache is still live where it should be

The characteristic failure of this fix is disabling the prompt cache for everyone, so that was measured rather than assumed. Same three-turn conversation per model, then `GET /v1/cache/stats`:

| Model | Family | inserts | hits | snapshot ins/hit | reject_model_owned_state |
|---|---|---|---|---|---|
| `llama-3.1-8b-4bit` | dense-natural | 3 | 2 | 0 / 0 | 0 |
| `gemma-4-e2b-it-4bit` | snapshot reuse | 0 | 0 | **5 / 2** | 0 |
| `gemma3-1b-4bit` | model-owned | 0 | 0 | 0 / 0 | **3** |
| `llama-4-scout-17b-4bit` | model-owned | 0 | 0 | 0 / 0 | **3** |

Server logs corroborate: Llama 3.1 still adopts (`cached=96/133`, `cached=128/165`), Gemma 3 now logs `cached=0/29`, `0/52`, `0/84`.

Gemma 4 is the named still-donating snapshot family, and the review independently enumerated all 13 `supports_snapshot_reuse() == true` families and confirmed every one is model-owned-natural, so none of them falls into the new gate.

### 4.3 Counterfactual proof that the tests are not vacuous

The five guard terms were neutralised, the suite re-run, and the files restored from backups. With the guards off, `model_owned_paged_family_never_donates_or_adopts` failed with `a shadow paged sequence must never reach the store; left: 1, right: 0` plus `DetachedPagedCacheSet dropped with 2 retained blocks`, which is the exact #1346 shape minted through the real scheduler rather than a mock.

### 4.4 Gate

`cargo test --workspace --profile test-fast --features metal,accelerate`: 8367 passed, 0 failed. Local CI (8 of the 10 `ci.yml` jobs, GitHub Actions being unavailable this session): 7 pass, 0 fail, 2 skip. The two skips are `OpenXLA feature compile` and `link`, which need a CUDA toolchain; this change touches nothing under an `xla` or `iree` path, so they are not relevant here.

One gate run failed on the known flake `text_only_forward_produces_finite_logits` in `tests/hunyuan_vl_parity.rs` (#997). Resolved by the standing procedure: the branch touches no file a hunyuan path reaches, the target passed 3 of 3 in isolation, the same commit range had passed the gate earlier, and the re-run was green. Occurrence data from this batch was added to #997.

## 5. Findings from Review

Nothing above MEDIUM. Two MEDIUM and six LOW findings; eight were applied.

**The most valuable finding was that the adopt gate had no test.** Five guards and five tests looked like a clean mapping, but one of the five is a premise test covering no guard, and every assertion in the donate/adopt test stays green with the adopt gate deleted, because the donate gate already left the store empty so the lookup it skips would have missed anyway. Even the counterfactual check above did not catch this, since it was valid for the other four. The fix is a single assertion on `store.stats().lookups == 0`, which separates "returned before the store lookup" from "looked, missed, and moved on", and which also pins the intentional metrics change below into a test instead of prose.

The other MEDIUM: `docs/turbo-kv-cache.md` named AFMoE as one of the families allocated on the paged backend. It is not one. `afmoe.rs` returns `false` from `supports_batching()`, which `effective_decode_storage_backend` requires, so AFMoE stays on the dense backend where the allocated backend already read `ModelOwned`. The issue body was itself wrong here from a different direction, blaming `supports_paged_decode_backend()`, and the doc inherited the error.

Also applied: corrected `detach_paged`'s `Used by:` roster, which named a scheduler request-boundary handoff caller that does not exist (the prompt-cache donate path is the only caller outside tests); documented two observability consequences; removed four em dashes introduced in doc comments; and corrected two overstated claims in the PR body.

## 6. Intentional Observability Changes

Two, both now documented on `CacheStatsResponse::reject_model_owned_state` rather than left in the PR body:

- A `supports_snapshot_reuse()` family no longer advances the KV `lookups` counter on a snapshot miss, because the adopt gate returns before `store.lookup_longest_prefix`. That lookup could only ever miss for those families, so `hit_rate` becomes more honest, but `/v1/cache/stats.lookups` and the Prometheus `mlxcel_prompt_cache_misses_total` now disagree by design where they used to agree, because the miss counter is recorded at the caller.
- Because the decline fires on every healthy completion for Gemma 3 and Llama 4, `last_reject_reason` stays pinned to `model_owned_state` and will mask an earlier, more interesting decline such as `oversized`. The per-reason counters stay separable and are the ones to read for these families.

## 7. What Remains Unverified

AFMoE on a real checkpoint. `models/afm-4.5b` exists locally but was not run. It is the third model-owned family without snapshot reuse, but it stays on the dense backend where the allocated backend already read `ModelOwned` and donation was already skipped, so the gate replaces one early return with an equivalent earlier one. Behavior is unchanged by construction, but unmeasured.

The two OpenXLA CI jobs, which need CUDA. Not relevant to this diff.

## 8. Learning Points

- When three values could serve as a predicate and one of them is the value that lies, the fix is choosing correctly among them, not adding a check. Look for an in-tree precedent that already made the same choice for the same reason; there were two here.
- Guard count and test count matching is not coverage. Check that each test maps to a distinct guard, and specifically that deleting each guard fails at least one test. A test can be green for a reason unrelated to the guard it appears to cover.
- A counterfactual check is only as complete as the guard-to-test mapping it walks. This one validated four guards and silently skipped the fifth.
- Measure the characteristic failure of the fix, not just the failure being fixed. For a gate that declines work, that means proving the work still happens where it should.
- State defense-in-depth claims precisely. "Any one of these five stops it" was false for two of the five, and an inaccurate safety claim is worse than a modest one, because the next reader may remove a term believing another covers it.
