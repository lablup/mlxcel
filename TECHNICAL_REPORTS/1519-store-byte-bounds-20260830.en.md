# Technical Report: Store Byte Bounds

## Summary

PR #1519 resolves issue #1248 by bounding the in-memory Responses API response store and conversation transcript store by approximate retained bytes as well as entry count and TTL.

The change adds saturating JSON-byte accounting, per-entry size snapshots, running totals and byte-budget configuration for both server binaries. It also replaces the previous map-wide LRU victim scan with a `BTreeSet` LRU index so each live entry owns one bounded metadata node and each eviction removes the oldest entry in O(log n).

## Problem

The prior stores capped only the number of retained entries. Each response entry can retain the complete request input and response object, and each conversation entry can retain a growing transcript, so a small number of large multimodal requests could pin much more memory than the entry-count limit implied.

The response store also selected LRU victims with a linear scan over the full map on each eviction. Under byte pressure one insert can evict multiple entries, making the scan cost repeat while memory pressure is already high.

## Implementation

- Added `ResponsesStoreConfig::max_bytes` and `ConversationStoreConfig::max_bytes` with defaults of 256 MiB and 64 MiB.
- Added `store_budget::serialized_json_len_saturating`, which counts serialized JSON bytes through a writer and returns `usize::MAX` on serialization failure so accounting fails closed.
- Added one `LruKey` per live entry and a `BTreeSet` index to both stores, with removal on delete, refresh, replacement, TTL sweep and eviction so stale LRU metadata cannot grow independently.
- Evicted until both `max_entries` and `max_bytes` are satisfied, including the oversized-single-entry case where the new entry evicts itself.
- Added `--responses-store-max-bytes` / `MLXCEL_RESPONSES_STORE_MAX_BYTES` and `--conversation-store-max-bytes` / `MLXCEL_CONVERSATION_STORE_MAX_BYTES` to both `mlxcel serve` and `mlxcel-server`.
- Documented the new byte-budget knobs in `docs/environment-variables.md` and `docs/responses-api.md`.

## Correctness

The stores compute size at insertion or transcript replacement time and maintain a running total with saturating add and subtract. Replacements first remove the old entry, then insert the updated entry, so totals and LRU metadata cannot double-count an id.

Reads refresh access order without advancing the LRU sequence on a miss. TTL sweeps remove both map entries and index keys. A zero byte budget keeps the route/store surface enabled when entry count is nonzero, but retained entries immediately self-evict and retrieval returns a normal miss.

## Security and Resource Bounds

The byte limits reduce the memory retention blast radius for well-formed large inputs, including base64 image data carried inside stored request items. Serialization errors and extreme values are treated as maximum-size entries, causing eviction rather than under-accounting.

Initial `HashMap` allocation is capped even when an operator configures an extreme `max_entries` value, avoiding a large upfront allocation from the capacity knob itself. TTL sweep remains O(n) over live entries, but LRU victim selection is no longer a repeated O(n) scan under byte pressure.

## Validation

- `cargo test --lib responses_store -- --nocapture` passed: 16 tests, including byte-only eviction, simultaneous count and byte pressure, oversized single entry, exact boundary, replacement, access-order refresh, TTL, zero and extreme budgets, and running-total consistency.
- `cargo test --lib conversation_store -- --nocapture` passed: 13 tests, including the same byte, count, oversized, exact-boundary, update, LRU, TTL, zero/extreme and total-consistency cases for transcripts.
- `cargo test --lib store_byte_budgets_round_trip_through_into_startup_config -- --nocapture` passed.
- `cargo test --bin mlxcel serve_store_byte_budget -- --nocapture` passed.
- `cargo test --bin mlxcel-server store_byte_budget -- --nocapture` passed.
- `cargo test --bin mlxcel-server settings_cli_mlxcel_server -- --nocapture` passed after preserving runtime-settings PR #1516 while rebasing onto main through PR #1518, the chat-template-cache head.
- `cargo test --bin mlxcel settings_cli_mlxcel_serve -- --nocapture` passed after preserving runtime-settings PR #1516 while rebasing onto main through PR #1518, the chat-template-cache head.
- `cargo test --bin mlxcel settings_cli_build_startup_input_defaults_off_and_propagates_enablement -- --nocapture` passed after preserving runtime-settings PR #1516 while rebasing onto main through PR #1518, the chat-template-cache head.
- `cargo test --lib settings_cli -- --nocapture` passed after preserving runtime-settings PR #1516 while rebasing onto main through PR #1518, the chat-template-cache head.
- `cargo test --test llama_compat_manifest manifest_option_claims_hold_on_both_server_binaries -- --nocapture` passed.
- `python3 scripts/ci/check_llama_compat_manifest.py` passed.
- `cargo fmt --all --check` passed.
- `cargo clippy --lib --bin mlxcel --bin mlxcel-server -- -D warnings` passed.
- `git diff --check` passed.
- Static scan found no conflict markers or old `min_by`, `min_by_key` or `evict_to_capacity` victim-scan code in the touched stores.

## Skipped Validation

Broad cargo workspace tests, serial all-tests, workspace clippy and cold release builds were intentionally skipped under the wave-runner watchdog guard. No real checkpoint, ffmpeg or hardware qualification was run because this change is confined to in-memory server-store accounting and CLI/config wiring.

## Risk Notes

The byte accounting is approximate by design. It now counts JSON escaping for strings and serialized JSON for response/output structures, but Rust allocation overhead and container internals are not included in the budget.

Conversation updates replace the old transcript accounting with the newly appended transcript. If the updated transcript exceeds the byte limit, it self-evicts; this is the intended fail-closed behavior for over-budget retained state.
