# Technical Report: PR #1288 - A total order for eviction candidate sorts

## Executive Summary

Three eviction paths built a candidate list from a `HashMap`, sorted it, and consumed a prefix. Sorting looks like it makes the outcome deterministic and does not: `slice::sort_by_key` is a **stable** sort, so entries comparing equal keep their input order, and the input order is `HashMap` iteration order. Among candidates tied on the sort key, which ones landed in the consumed prefix varied per run.

This is the subtle member of the defect class fixed by #1265, #1267, #1276 and #1277. Those four had no sort at all, so the defect was visible on inspection. These three do sort, which is precisely why they read as safe.

## 1. Problem Statement

- `src/distributed/tensor_parallel/cache_manager.rs`: `select_eviction_candidates` collects `allocations.values()` and sorts on `last_accessed` (LRU) or `current_offset` (LeastTokens). Its caller `check_pressure` iterates and breaks as soon as `projected_used <= target_bytes`, making it a prefix consumer. Ties decide which sequences actually lose their cache, and under LeastTokens a shared token count is an ordinary occurrence rather than a corner case.
- `src/distributed/pipeline/cache_manager.rs`: the same shape across three `PreemptionPolicy` arms, published as `PreemptionSignal.sequence_ids` and documented as eviction priority order. No prefix consumer exists in-tree today, so what was broken was the published contract rather than observed behavior.
- `src/distributed/request_tracker.rs`: `evict_if_needed` collects terminal requests, sorts on `created_at`, then takes an explicit prefix with `.take(to_remove)`.

Nothing corrupts and nothing crashes: the eviction relieves the pressure it set out to relieve and the counts are right. What was lost is reproducibility of **which** sequence was sacrificed, which is exactly what an operator needs when a request is preempted and they are working out why.

## 2. Technical Decisions

### 2.1 Make the sort key a total order

Each sort now includes the unique id as a tiebreaker, so no two elements compare equal and stability stops mattering. Both the comparator form and the tuple-key form are correct; the request tracker uses the comparator form specifically so its `String` key is not cloned into a tuple on every comparison, while the two cache managers use the tuple form over `Copy` fields.

`sort_unstable_by` was explicitly not the fix. It would replace one arbitrary tie order with another rather than remove the arbitrariness.

### 2.2 Pin the pipeline contract at its boundary anyway

The issue required an end-to-end assertion only for the tensor-parallel site, since it is the one with a live prefix consumer. The pipeline site got one too, asserting on the published `PreemptionSignal.sequence_ids` rather than only on the private helper, so the ordering is already pinned at the boundary a future consumer will read.

## 3. Change Summary

| File | Change |
| --- | --- |
| `src/distributed/tensor_parallel/cache_manager.rs` | Both policy arms sort on `(key, sequence_id)` |
| `src/distributed/pipeline/cache_manager.rs` | All three policy arms sort on `(key, sequence_id)`; the garbled `PreemptionSignal.sequence_ids` doc corrected to `first = evict first` |
| `src/distributed/request_tracker.rs` | `created_at` sort tie-breaks on the request key via a comparator |
| the three matching `*_tests.rs` | 8 new tests |

No public setter and no `#[cfg(test)]` accessor was added. All three test files attach via `#[path]` inside the module under test, so the private helpers and the private state were already reachable, and widening the API to make testing easier would have been the wrong trade.

## 4. Review Findings

The tests were written first and run against the still-unfixed sources, rather than the fix being written and then reverted to check. Three ways to write a vacuous test here were identified in advance and all three turned out to matter:

1. A reused map passes without the fix, because `RandomState` randomizes per `HashMap` instance. Each test builds a fresh map per iteration and loops 32 or 64 times.
2. Test data with distinct sort keys passes without the fix, because distinct keys already give a total order. Every test constructs a real tie.
3. `Instant::now()` resolves to nanoseconds on this platform and never collides on its own, so the LRU and `created_at` tests write equal `Instant` values in deliberately.

Pre-fix output, where `left` is raw hash order passed through by the stable sort:

```
---- eviction_candidates_lru_tie_break_is_deterministic ----
  left: [5, 3, 1, 4, 7, 2, 8, 6]
 right: [1, 2, 3, 4, 5, 6, 7, 8]
---- check_pressure_prefix_is_deterministic_under_ties ----
  left: [15, 11, 14]
 right: [11, 12, 13]
---- eviction_tie_break_is_deterministic ----
  left: ["req-0", "req-1", "req-4", "req-5"]
 right: ["req-2", "req-3", "req-4", "req-5"]
```

One trap not named in the issue, recorded here because it will catch the next person: `evict_if_needed` runs inside `submit_with_id` before the insert, so a test that completes each request as it submits triggers an unintended eviction on the sixth submit. The test submits all six, then completes them, then calls `evict_if_needed` directly.

## 5. Validation

Measured on GB10 (DGX Spark, CUDA sm_121, Linux aarch64), rebased onto `main` at `8fcc01f2` before gating so the gate ran on the tree that merges.

- `make verify-test-cuda`: **8246 passed, 0 failed, 311 ignored**, 101 suites, exit 0. That is +8 against `main` (8238), and the diff adds exactly 8 `#[test]` functions and removes none.
- The three module filters: 48, 47 and 21 passed, exit 0 each, and green again across five further separate processes, which reseeds `RandomState` per run.
- `cargo fmt --all -- --check`: exit 0. `cargo clippy --lib --tests --features cuda -- -D warnings`: exit 0 (the `err_expect` that previously reddened this was fixed by #1283 / PR #1285).

The gate log was scanned for process-level aborts as well as failing tests. A per-suite tally cannot see a teardown crash: an earlier run of this same gate reported 0 failed and still exited 101, because every test passed and the process then aborted with `Destroy(handle_) failed: driver shutting down` while a second cargo process saturated the GPU. Run alone, both this gate and #1283's are clean.

## 6. Related Work

- #1286: the issue this closes, filed from a sweep of `src/distributed/` during #1277.
- #1265 and PR #1266, #1267 and PR #1269, #1276 and PR #1281, #1277 and PR #1284: the sibling instances.
- #1287: the proposal to record this class in `docs/code-guidelines.md` and decide whether a static check can catch it.

Two near-misses were checked and left alone deliberately. `src/distributed/routing.rs` stable-sorts and takes `online[0]`, and an idle cluster ties on every component, but PR #1284 already cured it upstream by giving `Registry::all_nodes` a defined order, so that path silently depends on the accessor fix. And `nodes_at_stage` / `nodes_at_rank` sort on `Option<u32>` keys through `unwrap_or(u32::MAX)`, which looks tie-able, but `ClusterConfig::validate` rejects a PPTP node with either field unset before it can reach the registry, and neither accessor has a non-test consumer.
