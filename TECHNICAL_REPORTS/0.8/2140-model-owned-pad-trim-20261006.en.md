# PR #2140: Sequence-aware pad trim for model-owned families

**Date**: 2026-10-06
**Status**: Implemented and verified on CUDA (GB10) with forced tile alignment; M5 manual check pending
**Risk**: Medium (changes every scheduler prefill pad-trim site; the padded path itself only runs on M5 or with `MLXCEL_FORCE_PADDED_PREFILL`)

## Summary

A tile-aligned padded prefill writes pad positions after the real tokens, and the scheduler removed them by trimming the `CachePool` entry's `KVCache`s. A family whose own `sequence_state_layout()` is model-owned keeps its K/V inside the model, so that entry is empty and the trim reached nothing: the pad positions stayed and `offset` ran ahead of the token count by the pad width. PR #1752 opted Gemma 3, AFMoE and Llama 4 out of padded prefill as a stopgap.

The PR adds `LanguageModel::trim_sequence_state(seq_id, excess) -> Result<(), String>`. A new scheduler helper (`scheduler/pad_trim.rs`) trims the pool caches and, when the model's natural layout is model-owned, calls the hook; all four prefill sites (batched, full, chunked first chunk, continuation) use it, and an `Err` aborts the request. The default hook delegates to `trim_internal_caches` for non-batching models (their internal state is the running sequence) and fails closed for batching ones.

Gemma 3 implements the hook and `trim_internal_caches` through `Cache::rewind_padded_prefill` and answers `supports_padded_prefill() == true` again, except when a sliding layer is stored as Turbo4Asym. `deepseek_v4`, `bailing_moe_linear`, `qwen3_next` and `PipelineServerModel` opt out explicitly with the mechanism named. Closes #1755.

## Design notes

- `RotatingKVCache::trim` only rewinds `offset`/`idx`. After a padded chunk longer than the window, the next single-token update pre-trims by the physical buffer length and keeps the last `max_size` slots, which would include pad keys. `RotatingKVCache::rewind_padded_prefill` (`cache/prefill_rewind.rs`) also slices the physical buffer to `idx`. It refuses a non-chronological buffer (speculative buffering, a ring wrapped by a decode write, physical length not equal to `idx`) and Turbo4Asym storage, leaving the cache untouched.
- Gemma 3's text forwards ignore the caller's padding mask and build their own causal and sliding-window masks, so padding needs no mask change: pad positions trail the real ones and causal attention keeps them out of every real row.
- Prompt lookup still refuses Gemma 3 through its model-owned layout check, independently of `supports_padded_prefill`.
- `MLXCEL_FORCE_PADDED_PREFILL` forces alignment on any hardware (CLI and server); the server now honors `MLXCEL_NO_PADDED_PREFILL` too. `LoadedModel` previously did not delegate `trim_internal_caches`, so no family's CLI internal trim ran through it; it now delegates both hooks.
- The continuation-chunk mask offset read the first pool cache for batching models and fell back to 0 for an empty entry; it now falls back to the prefill cursor.

## Verification on GB10 (CUDA, release profile, `--features cuda`)

Before this PR no model-owned family reached padded prefill on CUDA: only qwen3_5 combines a model-owned layout with `supports_batched_prefill()`, and it declines padding, so the longest-row padded batched site is unreachable; the single-sequence sites were hardware-gated with no override.

- `cache::prefill_rewind` (5 tests): padded append plus rewind equals an unpadded append byte for byte, within and past the window and on continuation chunks, including after three decode steps; refusals leave the cache untouched.
- `scheduler_model_owned_pad_trim_tests` (4 tests, tiny Gemma 3 with one sliding and one global layer): offsets, the sliding layer's physical buffer, global keys and decoded tokens match the unpadded run for full prefill (dense and paged backends) and 16-token chunked prefill; a padded turn donates a snapshot that turn 2 adopts. With the Gemma 3 rewind disabled the first three fail (offset 32 vs 13, 32 vs 5, 96 vs 37).
- Suites: `server::batch::scheduler` 178 passed (7 ignored), `block_reclaim` 23, `lookahead` 11, `speculative` 175, `generate::tests` 36, `cache::rotating` 11, `models::afmoe` 28, `muse_glimmer` 109, `models::gemma3` and `models::gemma4` pass. Clippy on both crates with `-D warnings` and `cargo fmt --check`: clean.
- Server, gemma-3-4b-it-4bit, `--prefill-chunk-size 500`, greedy, 48 tokens, prompts of 176, 730 and 1852 tokens:

| Comparison | short (176) | mid (730) | long (1852) |
|---|---|---|---|
| unpadded p1 vs padded p1 | identical | diverges at token 15, top-2 gap 0 | diverges at token 11, top-2 gap 0 |
| unpadded p1 vs unpadded p4 (concurrent) | identical | identical | diverges at token 11, top-2 gap 0 |
| unpadded p4 vs padded p4 (concurrent) | identical | diverges at token 15, top-2 gap 0 | identical |
| unpadded p1 vs padded p1 with the rewind disabled | diverges at token 1, gap 1.0 | diverges at token 1, gap 1.5, output degenerates to `texId` | diverges at token 11 |

Gaps are measured at the reported logprob resolution (0.25). Every divergence with the fix sits where the top-2 logprobs are equal, and the long prompt diverges at the same position between two unpadded runs.

- CLI: `mlxcel generate` with gemma-3-4b-it-4bit, forced padding vs unpadded, identical 40-token output.

## Not verified on this host

- The M5 manual check from the issue and Metal numerics of the padded path.
- AFMoE and Llama 4 remain opted out (no checkpoint-validated rewind; Llama 4's `ChunkedKVCache` needs its own).
- `cudaStreamEndCapture` aborts occurred in roughly one of four repeated runs of several suites, including the gemma4 suite whose source change is comment-only; each suite passed on rerun.

## Follow-up

Decode lookahead stays disabled for model-owned families (PR #2139). The hook does not cover that teardown: the rotating rewind requires a chronological buffer, and a single-token decode write into a wrapped ring is not one.
