# Technical Report: PR #1509 - Model-aware snapshot cache capacity

**Date**: 2026-08-30
**Status**: Completed
**Languages**: Rust, Markdown
**Risk Level**: Medium

## Executive Summary

PR #1509 fixes issue #1167 by replacing the fixed 512 MiB implicit snapshot-cache capacity with a bounded, model-aware startup default for snapshot-capable large hybrid models. Explicit operator capacity settings still win. The PR also makes capacity thrash observable by exposing per-entry snapshot bytes and same-session self-evictions in `/v1/cache/stats`, and by logging one WARN per affected session when snapshot LRU pressure evicts the session that just donated a snapshot.

## 1. Problem Statement

Qwen3.8-27B 4-bit can produce prompt-cache snapshot entries around the old 512 MiB default at agent-scale prompts. Under that default, the snapshot store could accept a new entry and immediately LRU-evict another live entry from the same session, leaving later turns with a 0% multi-turn hit rate.

The failure mode was hard to diagnose: counters showed inserts and LRU evictions, but operators had to manually divide `snapshot_bytes / snapshot_entries` and infer that a fixed default was capacity one for the loaded model. The issue also asked for the `--kv-bits` / `--kv-cache-mode` behavior on model-owned snapshot families to be explicit rather than silently accepted.

## 2. Technical Decisions

### 2.1 Keep operator capacity authoritative

`PromptCacheConfig` now tracks whether `snapshot_capacity_bytes` came from an explicit CLI/env/builder value. Startup applies the model-aware default only when this bit is false. This preserves existing deployment tuning and avoids surprising operators who intentionally set a smaller or larger cap.

### 2.2 Derive the implicit default from `config.json`

The new snapshot sizing module reads model metadata and reuses the existing architecture-aware KV estimator. The default sizes six representative snapshots at `min(context_size, 8192)` tokens, using FP16 snapshot serialization because non-FP16 model-state sidecars are not serialized today. For Qwen3-Next / Qwen3.5-family configs it adds the fixed gated-delta recurrent/conv state estimate from linear-attention dimensions.

The derived raise is clamped to one quarter of detected available memory. This gives large models room for multi-turn reuse without letting metadata alone consume an unbounded share of a constrained host.

### 2.3 Separate healthy supersede from capacity thrash

`PromptCacheStats` and `/v1/cache/stats` now expose `snapshot_self_evictions`. The counter advances when snapshot LRU enforcement evicts an entry from the same session chain as the snapshot insert being admitted. Healthy same-session strict-prefix replacement remains counted under `snapshot_supersedes`.

The WARN suppression set is capped at 4096 distinct session keys so malicious or accidental high-cardinality session identifiers cannot grow memory without bound. The public counter remains exact even after duplicate warning suppression reaches the cap.

### 2.4 Make snapshot byte sizing visible

`/v1/cache/stats` now includes `snapshot_bytes_per_entry`, computed as `snapshot_bytes / snapshot_entries` with zero for an empty store. This makes the operator sizing advice a direct field lookup.

### 2.5 Document the KV-mode contract

The documentation now states that model-owned snapshot families do not silently serialize non-FP16 attention KV sidecars. Today those modes are applied to live attention layers and reported through startup/log stats, but snapshot donation skips with a named warning until sidecar snapshot support exists.

## 3. Change Summary

| Category | Count | Summary |
|---|---:|---|
| Capacity sizing | 1 | Added model-aware snapshot default derivation and startup wiring. |
| Config authority | 1 | Added explicit-capacity tracking for CLI/env/builder overrides. |
| Observability | 2 | Added `snapshot_bytes_per_entry` and `snapshot_self_evictions` to cache stats. |
| Runtime warning | 1 | Added bounded once-per-session WARN for same-session snapshot self-eviction. |
| Documentation | 3 | Updated env var docs, Turbo KV docs, and the issue #1167 validation record. |
| Tests | 5 | Added deterministic sizing, override, stats, self-eviction, and supersede coverage. |

## 4. Validation

- `cargo test --lib server::prompt_cache::snapshot_sizing::tests`: 3 passed.
- `cargo test --lib snapshot_capacity_self_eviction_is_counted_and_warned_once_per_session`: 1 passed.
- `cargo test --lib supersede`: 7 passed.
- `cargo test --lib cache_stats`: 3 passed.
- `cargo test --lib build_stats_response_reports_snapshot_bytes_per_entry`: 1 passed.
- `cargo test --lib prompt_cache_snapshot_limits`: 2 passed.
- `git diff --check`: passed.

The Apple Silicon / Metal Qwen3.8-27B benchmark from issue #1167 was not rerun in this Linux/aarch64 worktree. The PR preserves that external evidence as issue context and validates the implementation with deterministic local tests.

## 5. Review Notes

- **Correctness**: Model-aware sizing is skipped when the capacity is explicit. Standard full-attention models do not get a snapshot-capacity raise. Qwen3-shaped tests pin the computed KV and fixed-state bytes.
- **Security**: The WARN suppression map is bounded, preventing unbounded memory growth from high-cardinality session keys. No prompt contents or token vectors are logged.
- **Performance**: Startup reads `config.json` and the existing memory estimator metadata only; no MLX model state or tensors are loaded for sizing. Hot insert overhead is limited to a same-session key comparison during snapshot LRU enforcement.
- **Compatibility**: The stats response gains additive fields. Existing CLI/env knobs retain their names and precedence.

## 6. Follow-up Actions

- Rerun the agent-scale Qwen3.8-27B Apple Silicon / Metal benchmark on qualified hardware and record before/after hit-rate medians.
- Add real checkpoint coverage for model-owned non-FP16 snapshot sidecar serialization if those modes become supported instead of fail-closed.
