# Technical Report: PR (issue #2088), residual paged-budget gaps after #2077

**Date**: 2026-10-02

**Status**: Implemented and validated on GB10 with meta-llama-3.1-8b-instruct-4bit under a forced 4444-block budget; not yet merged.

**Languages**: Rust

**Risk level**: Medium (changes paged admission and reclaim decisions; adds a default-on admission watermark of 0.01)

## Executive summary

PR #2077 left three gaps in the paged KV block budget: the free-block figure was pool-wide while blocks are reused per layer, a parked chunked prefill's next chunk had no reservation of its own, and nothing kept decode headroom at admission. This PR closes all three and adds `--kv-admission-watermark` (default 0.01). Measuring the watermark on GB10 exposed two more accounting gaps from #2077: adopted prompt-cache prefixes were charged twice at admission (which wedged the queue), and blocks pinned by queued adoptions could not be reclaimed (which shed a running row). Both are fixed here as well.

## Problem statement

1. `PagedBlockPool::free_block_budget` returned `budget - live`. A freed block goes back to its own layer's free list, and `acquire_block` for layer `L` reuses only `free_lists[L]` before minting under the pool-wide cap. Once the free lists diverge, one layer can fail to acquire while the figure is positive.
2. A parked chunked prefill is protected only by subtracting its remaining blocks from the figure other consumers read. That figure saturates at 0, so when the pool fell below the reservation every other consumer deferred but nothing restored it, and the chunk's forward could hit an exhausted pool. The reservation also ignored the tile padding a padded chunk writes before trimming.
3. Under a tight budget, admitting into exactly the free blocks leaves running rows no room to grow, so decode preempts them and they re-prefill (12 preemptions for 9 turns in the #2077 pressure run).

## Change summary

- Per-layer accounting (`mlxcel-core` `paged.rs`): `free_block_budget` returns `n * num_layers` for the largest `n` such that every layer can acquire `n` blocks from its own free list plus a share of the mint headroom `budget - allocated`, capped at `budget - live`. Every scheduler demand is uniform across layers, so with balanced free lists the figure is unchanged. Re-tagging free blocks across layers was rejected: pool rows are per-layer slabs, so a moved block needs a new row and the budget would stop bounding memory. Layers that have never minted a block are left out of the constraint.
- Chunk reservation (`block_reclaim.rs`, `prefill.rs`): `continue_chunked_prefill` calls `reserve_prefill_chunk_blocks` before the forward. It checks the pool figure (not the saturating set-aside figure), evicts cold prefixes, then preempts with floor 0 and no priority limit, since the chunk's admission already checked the whole prompt. If even that leaves too few blocks, the request fails with "KV cache budget exhausted" instead of panicking the worker. This covers the `MLXCEL_MIXED_STEP` tick and the #1011 grant path. `chunked_prefill_reserved_blocks` now includes the next chunk's padding.
- Admission watermark (`admission_watermark.rs`, `--kv-admission-watermark`, env `MLXCEL_KV_ADMISSION_WATERMARK`, range 0.0 to 0.5): while a decode batch is live, admission requires free blocks minus `ceil(F * budget)` to cover the request and reclaims toward that target before deferring: cold prefixes first, then (for a higher-priority request) strictly lower-priority rows, as admission already did for the request itself. With an empty batch it does not apply, so a request that fits the budget is never refused. `paged_prefill_room` applies it, so the batched window and the single-sequence gate charge the same figure.
- Adopted prefix charge (`admission.rs`, `CachePool::paged_blocks_to_reach`): admission charges the blocks the suffix will mint, measured from the sequence's paged state, and at least one block per layer after an adoption (a full-prompt hit re-runs its last token into a shared tail block, which forks). Before this, an adopted prefix larger than half the budget could never be admitted, even into an empty batch.
- Queued adoption reclaim (`queued_adoption.rs`, `PrefillQueue::find_lowest_first`/`get_mut`/`remove`): adoption happens at enqueue and pins the prefix's blocks while the request waits. Decode and chunk reclaim now drop one queued text request's adoption (lowest priority, newest first) after evicting cold prefixes and before preempting; the request re-prefills cold and keeps its prompt-cache context for donate-back. Admission does not do this; a queued request waits instead of costing another one its hit. Multimodal requests are skipped because an adopted VLM request no longer holds its full-prompt embeddings.

## Validation

- Unit tests: `paged_detach_tests.rs` adds 3 per-layer cases; on a diverged state (layer 0 owns 2 free blocks, layer 1 none, budget fully minted) the old figure is 2 while a layer-1 append fails, and the new figure is 0; with 2 blocks of headroom the figure is 4 and exactly that much is acquirable. `paged_append_need_tests.rs` adds the suffix charge. `block_reclaim_tests.rs` adds MixedStep tick-order cases (decode defers and the chunk fits; mid-block rows leave the pool below the reservation, the chunk append fails without the reservation and succeeds after it preempts one row; a cold prefix is evicted before any row), the watermark deferral and eviction cases, and the queued-adoption cases (a lone boundary row survives by dropping a queued adoption; order is prefix, adoption, row; admission never drops one). `scheduler_prompt_cache_tests.rs` adds the adopted-prefix charge on a real pool.
- `server::batch`, `server::prompt_cache`, `runtime_settings`, `cli_input`, `config_tests`, `commands::serve` filters: 822 passed. mlxcel-core `paged` and `budget` filters: 282 passed. `cargo clippy --release --features cuda --lib --tests -- -D warnings` (root and `-p mlxcel-core`) and `cargo fmt --all` are clean.
- GB10 pressure run (`--kv-cache-budget 600000000`, 4444 blocks, paged, `--max-batch-size 4`, a prefix parked first, then three concurrent 3-turn conversations of 600-token generations):

| Build | Watermark | Preemptions | Decode reclaims | Turns | Errors |
|-------|-----------|-------------|-----------------|-------|--------|
| main (#2077 report) | n/a | 12 | 16 | 9 | none |
| gaps 1-3 only | 0 | 12 | 16 | 9 | none |
| gaps 1-3 only | 0.15 | 2 | 4 | 4 | 2 shed rows, 1 request hung 900 s |
| this PR (x2) | 0 | 10 | 43 | 9 | none |
| this PR (x2) | 0.01 | 6 | 6 | 9 | none |
| this PR (x2) | 0.02 | 6 | 6 | 9 | none |
| this PR (x2) | 0.05 | 5 | 5 | 9 | none |
| this PR | 0.10 | 1 | 2 | 9 | none |
| this PR | 0.15 | 1 | 3 | 9 | none |

  The 0.15 failure before the last two fixes: both sheds had `preempted=0` with one row in the batch and nothing evictable, so the remaining blocks were held by a queued adoption, and the hang was a request whose adopted prefix exceeded half the budget, deferred forever with an empty batch (an aborted 0.20 run showed the same pattern with the worker spinning at 96% CPU). The default 0.01 is the smallest grid value that lowered preemptions without losing a turn. At 0 the branch's 43 reclaim passes are 33 one-tick deferrals of a row short only by a parked chunk's reservation.
- `MLXCEL_MIXED_STEP=1` at 0.01, `--metrics`: 21 mixed steps and 55 prefill chunks, 9 of 9 turns, no panics, 7 preemptions.
- Default launch (auto budget, 753620 blocks) at 0.01, two 3-turn conversations: no reclaim or preemption, turns 2 and 3 hit the prompt cache (1280 and 1504 tokens).

## Not validated

- The per-layer figure trusts that every refcount-0 block sits on exactly one layer free list. Its `debug_assert!` never ran (all tests were release builds); instead every mutation site was read: `release_block` pushes on reaching 0, `acquire_block` and `restore_block` pop or remove before setting refcount 1, `retain_block` refuses a refcount-0 block, and no path removes a record from `blocks`.

- The chunk reservation's reclaim path did not trigger in any real-server run (`Chunked prefill reclaimed` never logged); it is covered by unit tests only.
- The queued-adoption drop triggered once each at 0.10 and 0.15; its re-prefill of a dropped request was checked only through those runs completing all turns.
- The padded-chunk reservation applies only on Metal M5+ (`should_align_prefill`), which this host cannot run.
- The watermark default was measured on one model and one budget; under the auto budget it holds back 1% of the pool while rows decode.
