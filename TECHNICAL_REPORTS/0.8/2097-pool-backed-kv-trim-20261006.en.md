# PR #2097: Pool-backed `KVCache::trim` rewinds the block table

**Date**: 2026-10-06
**Status**: Reviewed and verified on CUDA (GB10); M5 tile-aligned path verified by the contributor only
**Risk**: Low

## Summary

`KVCache::trim` returned `0` for a pool-backed cache (`paged_backing.is_some()`), so every caller that dropped prefill pad rows with `trim(excess)` left those rows in the pool and left `offset` at the padded length. Decode then attended over the pad rows and took its RoPE position from the padded `offset`. The PR makes `trim` call `PagedBlockPool::rewind_tokens` through the cache's own backing and subtract the removed count from `offset`, and routes the decode lookahead teardown (`apply_lookahead_trim`) through the same call instead of its pool-only `rewind_paged_tokens` branch, which rewound the pool but left `offset` one ahead.

Closes #2096. Contributor: rapsealk.

## Review

- Every production `trim` caller on a pool-backed cache (the four pad trims in `scheduler/prefill.rs` and the lookahead teardown) expects a real trim. No caller pairs `trim` with a separate pool rewind, so there is no double rewind. `trim_to`/`DetachedCacheSet::truncate_to` have no production caller.
- `sync_paged_state_with_dense` returns early for pool-backed sequences, so the `sync_sequence_storage` call after the teardown does not re-trim the block table.
- The `live_len()` clamp still runs before the pool branch, and `offset` moves by the count the pool actually removed, so pool `len` and `offset` cannot diverge through this path.
- The pool branch uses `.expect` where the old lookahead branch logged a warning. This matches the neighbouring `write_paged` and `update_and_fetch_paged` calls on the same pool, and a rewind failure there means the block table is already inconsistent. Recorded as MEDIUM, not changed.
- `CachePool::rewind_paged_tokens` has no non-test caller left; it was kept.

No CRITICAL or HIGH findings; no code changes were made on top of the contributor's commits.

## Verification on GB10 (CUDA, release profile, `--features cuda`)

Branch head merged with `origin/main` at `2bbf192f`.

- New test `trim_drops_padded_prefill_rows_from_a_pool_backed_cache`: passes with the fix. With `cache.rs` and `decode_tick.rs` reverted to `origin/main` it fails at the first assertion (`left: 0, right: 27`).
- Neighbouring mlxcel-core suites, `--test-threads=1` under `gpu-lock`: `cache::paged_batch_decode` 23 passed, `cache::paged` 137 passed, `cache::tests` 83 passed, `cache::detach` 51 passed, `speculative` 175 passed. Main crate: `lookahead` 10 passed (including the teardown position tests), `block_reclaim` 23 passed, `server::batch::scheduler` 167 passed (7 ignored).
- `cargo fmt --all -- --check`, `cargo clippy -p mlxcel-core --lib --tests --features cuda -- -D warnings`, and the same for `-p mlxcel`: clean.
- Server: `mlxcel-server -m llama-3.2-1b-4bit --no-prompt-cache`, five distinct chat prompts (40, 42, 51, 52, 64 prompt tokens) sent concurrently, three rounds, `temperature 0`, `max_tokens 32`. Default backend (auto, resolves to pool-backed paged on this host) compared with `--decode-storage-backend dense`. Debug logs confirm the batched padded prefill ran (`batched prefill: 3 requests, padded to 52`, etc.).

| Build | Default vs dense, identical continuations | First batched decode gather |
|---|---|---|
| `origin/main` (fix reverted) | 5 of 15 | 224 visible KV tokens across 4 requests (pad rows retained) |
| PR #2097 | 15 of 15 | 189 visible KV tokens across 4 requests (real rows plus one each) |

Before the fix the default backend answered `Repeat: one two three` with `One!Two!Three!` and the France prompt with a rambling `The capital of France is... Paris! The City of Light, ...`, matching the contributor's M5 report. The dense reference output was byte-identical between the two builds. This confirms the issue's claim that the batched-prefill longest-row padding corrupts pool-backed output on hardware without a Neural Accelerator.

## Not verified on this host

- The M5 tile-aligned prefill path (`should_align_prefill()` true) needs Apple M5 hardware. The contributor's M5 Pro results (Llama-3.2-1B, SmolLM2-135M, Qwen3-0.6B, before 0-1 of 5, after 5 of 5) are taken as reported.
- The #1760 whole-prompt replay abort on M5 is out of scope for this PR.
