# Technical Report: PR #3268 - Count each prompt-cache lookup once in Prometheus

**Date**: 2026-11-23
**Status**: Completed
**Risk Level**: Low (metrics bookkeeping only; no request-path behavior, cache contents, metric names or HELP text change)

## Executive Summary

`/metrics` reported roughly twice the real prompt-cache hit, miss and reused-token totals, because the prompt-cache store and the batch scheduler both wrote the same `BatchMetrics` counters (#3246). Each counter now has exactly one writer. The store's `BatchMetricsCacheAdapter` forwards only evictions and snapshot misses; the scheduler writes dense hits, dense misses, snapshot hits and both reused-token totals. The same double count existed on the snapshot-restore and state-checkpoint paths, and the same adapter change fixes them. Three production-wired regression tests fail on main (2 vs 1 dense hits, 2 vs 1 snapshot hits, 8 vs 4 checkpoint-path snapshot hits) and pass on this branch. The full macOS gate passed: 16227 tests, 0 failures.

## 1. Problem Statement

`startup.rs` builds the store with `PromptCacheStore::with_metrics(.., BatchMetricsCacheAdapter::new(batch_metrics))` and hands the same `Arc<BatchMetrics>` to the scheduler. The store called `record_lookup` on every lookup, and the adapter turned that into `record_prompt_cache_hit(matched_len)` or `record_prompt_cache_miss()`. The scheduler then recorded the same event again: `finish_prompt_cache_adopt` on a successful adoption and the admission miss site when nothing was adopted. The issue measured 18 hits on `/metrics` against 9 on `/v1/cache/stats`, and 111840 reused tokens against about 55.9k summed `cached_tokens`.

The two writers also measured different things. The store recorded its raw match length; the scheduler records `adopted_len`, which is capped by `whole_prompt_reuse_cap`, floored to the block size, and returned as `cached_tokens`. A store hit the scheduler declined was counted once as a hit and once as a miss. Ratios came out right by accident; absolute counts and token savings were off by 2x.

No existing test could see this: every scheduler test built its store with `PromptCacheStore::with_config`, which has no metrics adapter.

## 2. What Changed

- `src/server/state.rs`: `BatchMetricsCacheAdapter::record_lookup` is a no-op, matching `record_insert`. `record_snapshot_lookup` forwards only misses. The type's doc comment states the one-writer rule.
- `src/server/batch/scheduler/prompt_cache.rs`: removed the "also increment BatchMetrics" comment; the scheduler's call sites are unchanged because they already record the reported `cached_tokens` value.
- `src/server/startup.rs`: the wiring comment now says the adapter carries evictions and snapshot misses only.
- `docs/turbo-kv-cache.md`: one sentence stating that the snapshot counters equal the `snapshot_*` fields of `/v1/cache/stats` and the reported `cached_tokens`.
- Tests:
  - `src/server/batch/scheduler_prompt_cache_metrics_tests.rs` (new): a dense Llama run (KV adopt) and a tiny Gemma 3 run (snapshot restore), each with the store wired as in production and a cold plus a warm turn through admission. They assert that the `/metrics` counters equal the store's `hits` and `lookups - hits`, and that reused tokens equal the summed `cached_tokens`.
  - `src/server/batch/scheduler_state_checkpoint_tests.rs`: `checkpoint_rebuild_counts_once_in_prometheus_with_production_wiring` covers four turns plus the `edited(2)` checkpoint rebuild. The existing `scheduler()` helper now delegates to `scheduler_with_metrics(store, metrics)`.
  - `src/server/state_tests.rs`: the adapter test now asserts that the adapter leaves the dense counters alone, and a new test covers the snapshot-miss-only forwarding.
  - Two helpers (`cycle_llama`, `EOS_ID`) became `pub(super)` for reuse.

## 3. Technical Decisions

**The scheduler is the writer, not the store.** The rejected alternative was to keep the adapter and drop the scheduler calls. That would count declined adoptions as hits and report the store's raw match length instead of `cached_tokens`. The scheduler is the only component that knows whether a lookup was adopted and how many tokens were actually reused, and the `/metrics` HELP text already says "Successful prompt-prefix cache adoptions".

**Snapshot misses stay with the adapter.** The store is the only place that knows a snapshot lookup found nothing; the scheduler never sees a snapshot miss as a distinct event. Moving it would need a new scheduler signal for no gain, so the rule is one writer per counter, not one writer per subsystem.

**Tests wire the store as production does.** The defect lived in the wiring, so a test on the existing `with_config` harness would have passed on main. Each new test was confirmed to fail with main's `state.rs` before being accepted.

## 4. Measurements

Unit level, macOS Metal, `--profile test-fast`, `--test-threads=1`.

| Test | main | this PR |
|---|---|---|
| dense lookup, `/metrics` hits vs store hits | 2 vs 1 (fail) | 1 vs 1 |
| snapshot restore, snapshot hits vs store | 2 vs 1 (fail) | 1 vs 1 |
| state checkpoint rebuild, snapshot hits vs store | 8 vs 4 (fail) | 4 vs 4 |
| state checkpoint rebuild, reused tokens vs summed adopted lengths | 192 vs 96 (reviewer throwaway run) | 96 vs 96 |

Gate: `make verify-test` passed with 16227 tests, 0 failed, 499 ignored across 192 test binaries.

## 5. What Was Not Measured

- No manual server run against a real checkpoint comparing `/metrics`, `/v1/cache/stats` and summed `cached_tokens`; the validation machine had no checkpoint. The production-wired scheduler tests exercise the same comparison on tiny models.
- Linux/CUDA was not run; the change is backend-independent bookkeeping.

## 6. Learning Points

- When two components share an `Arc` sink, the wiring is part of the behavior. A harness that builds the components differently from `startup.rs` can hide a defect that lives only in production wiring.
- A metric with two writers can look correct as a ratio while every absolute value is wrong, so assertions should compare absolute counts against an independent source (`/v1/cache/stats`, the `cached_tokens` in responses).

## 7. Follow-up

- Warmup snapshot misses: `run_next_prompt_cache_warmup` calls `lookup_snapshot_prefix`, so a warmup miss still increments `prompt_cache_snapshot_misses_total`. A warmup hit no longer does. Excluding warmup needs a store API change that also touches `/v1/cache/stats`.
- Snapshot families and the dense hit rate (pre-existing): for models with `supports_snapshot_reuse()`, admission increments `prompt_cache_misses_total` on every cold request, while restores reach only `prompt_cache_snapshot_hits_total`. A dense hit rate computed from `/metrics` reads 0% for those models.
- A store snapshot hit that the scheduler declines (`ModeMismatch`, restore error, slot exhaustion) now moves neither snapshot counter, which matches the "successful restores" HELP text. Prometheus snapshot hits plus misses can therefore be lower than the store's `snapshot_lookups`.
