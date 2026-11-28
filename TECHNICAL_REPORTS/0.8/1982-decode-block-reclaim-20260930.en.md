# Technical Report: PR (issue #1982), paged block reclaim during decode

**Date**: 2026-09-30

**Status**: Implemented and validated on GB10 with meta-llama-3.1-8b-instruct-4bit under a forced 4444-block budget; not yet merged.

**Languages**: Rust

**Risk level**: Medium (changes scheduler preemption behavior under paged KV budget pressure)

## Executive summary

With the paged backend and a KV block budget (the default `--kv-cache-budget auto`), a decoding row that crossed a block boundary while the pool was full reached `PagedBlockPool::acquire_block`, which returned "block budget exhausted". The write happens inside the model forward (`write_paged` calls `.expect`), so the model worker panicked and the server returned 503 for every request. Decode now reserves the blocks a tick will mint before the forward runs, reusing the admission reclaim loop (cold prompt-cache prefixes first, then preemption). Real-server testing exposed five more ways the same budget crashed, wedged, or livelocked the worker (longest-first preemption, equal-priority admission thrash, a spinning deferral, batched prefill without a budget gate, and chunked prefill losing its blocks to decode), and this PR fixes those as well.

## Problem statement

Prefill admission (`admit_paged_prefill`) reclaimed blocks, but nothing else in the scheduler did. Prompt-cache paged entries pin blocks that count against the budget, and #1978 raised the default store capacity, so decode growth hit the limit more often.

## Change summary

- `mlxcel-core`: `PagedBlockPool::blocks_to_append` and `CachePool::paged_blocks_to_append` count the blocks an append would acquire per layer: new tail blocks plus one copy-on-write fork when the first new token lands in a shared partial tail block.
- New `scheduler/block_reclaim.rs`: the reclaim loop moved out of `admission.rs` into `reclaim_blocks`, written against a small `PagedBlockReclaimer` trait so it can be tested over a real `CachePool` and `PromptCacheStore`. Admission and decode both call it; there is still one loop.
- `execute_decode_step` calls `reserve_decode_step_blocks` first. When the pool has room this is a read-only check with no side effects. Under pressure it discards any prebuilt lookahead, reclaims with a preemption floor of 1 (the last row is never preempted to make room for itself), drops rows a preemption removed, and then either defers or sheds rows that still cannot grow. A shed row finishes with a "KV cache budget exhausted" error, so the forward never runs against an exhausted pool. The lookahead prime is skipped when its extra position would need a block.
- Victim order for block reclaim (`select_block_reclaim_victim_from`): lowest priority, then fewest generated tokens, then newest id. This is a deliberate change from the configured slot-preemption policy. `LongestFirst` preempts the row closest to finishing, and in the first GB10 run it did so every time: 179 preemptions and one completion in ten minutes, with every victim near 250 generated tokens. Slot preemption keeps its configured policy.
- Admission now preempts only strictly lower-priority rows. Before this, two equal-priority requests that could not both fit preempted each other right after each prefill (over 1,600 preemptions of rows with 1 generated token). When admission defers the head request, the same tick runs a decode step. Without that step, `decide_action` kept choosing Prefill, the deferral spun, and no blocks were ever freed (a 98% CPU stall).
- Batched prefill had no budget gate: three 1095-token prompts were admitted together and the worker panicked. A head that does not fit now goes to the single-sequence admission path, and the window drain stops at the first row that would overflow the budget.
- A parked chunked prefill's remaining blocks are set aside (`available_paged_blocks`), because its admission checked the whole prompt but it acquires blocks chunk by chunk. Decode growth used to consume them and the next chunk panicked. A row short only by that reservation is deferred for one tick instead of shed; the prefill becomes preemptible once it joins the batch.
- `src/models/sanitize_tests.rs`: removed a `cfg(not(feature = "cuda"))` gate on a test helper the test now calls unconditionally. Without this, `cargo test --features cuda --lib` does not compile on current main.

## Validation

- Unit tests: `block_reclaim_tests.rs` has 11 cases. A pool filled by cached prefixes, where a decode row crosses a boundary: the append fails without reclaim ("block budget exhausted"), and with reclaim it succeeds after exactly one LRU eviction. With free budget nothing is evicted or preempted. With no evictable entries, reclaim preempts. With nothing reclaimable, the row is shed and nothing panics. The remaining cases cover a mid-block row surviving shedding, admission's priority limit, the chunked-prefill reservation (decode evicts instead of taking the reserved blocks), deferral, and the batched-window budget charge. `paged_append_need_tests.rs` has 3 cases, including the copy-on-write fork. `scheduler_tests.rs` has 3 selector cases.
- `cargo test --release --features cuda --lib -- server::batch server::prompt_cache models::sanitize_tests --test-threads=1`: 674 passed. mlxcel-core `cache::paged`: 132 passed. `cargo clippy --release --features cuda --lib --tests -- -D warnings` and `cargo fmt --all` are clean.
- GB10 with `--kv-cache-budget 600000000 --decode-storage-backend paged --max-batch-size 4` (4444 blocks), three concurrent 3-turn conversations with 600-token generations after a prefix had been parked: all 9 turns completed with no errors or panics, 16 decode reclaim passes, and 12 preemptions. The same load on earlier iterations of this branch produced each failure mode listed above.
- Same server, one 1099-token prompt generating 2200 tokens with cached prefixes resident: the log shows three decode reclaim passes, each with `evicted_prefixes=1` and `preempted=0`, and the request completed.
- Default launch (auto budget, 753620 blocks), two 3-turn conversations: turns 2 and 3 hit the prompt cache (1376 and 1696 cached tokens), with zero reclaim or preemption lines.

## Remaining work

- `free_block_budget` is a pool-wide count, but free rows sit on per-layer free lists. Uniform paged layouts keep the lists balanced, so the global count holds in practice. A non-uniform layout could still fail on one layer.
- The prefill chunk in a MixedStep tick is covered by the chunked-prefill reservation, not by a per-tick reservation. Speculative slice rounds use model-owned caches and are outside the paged pool.
- Under a budget this tight, preemption costs recomputation: 12 preemptions for 9 turns in the pressure run. An admission watermark that leaves decode headroom would reduce this. It is not implemented here.
