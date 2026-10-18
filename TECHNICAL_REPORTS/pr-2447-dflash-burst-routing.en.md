# PR #2447: Route queued DFlash heads to the speculative burst

**Date**: 2026-10-18
**Status**: Items 2 and 3 of #2442 implemented and verified on CUDA (GB10) with a real checkpoint; item 1 (per-row accepts) remains open on #2442

## Summary

Refs #2442 (items 2 and 3 only). Under the default `--max-batch-prefill 4`, a queue of 2 or more requests no longer loses speculative decoding: a head the burst would serve now takes `execute_prefill`, which reaches `try_speculative_burst`. The batched DFlash window becomes opt-in through `MLXCEL_ENABLE_DFLASH_BATCH` (default off), because on GB10 its output diverged from sequential requests at temperature 0. `run_dflash_burst_batched` now applies the same exactness probe as the B = 1 arm before loading the drafter.

## 1. Problem Statement

### 1.1 Routing (item 2)

`execute_prefill_action` sent any queue of 2+ to `execute_batched_prefill` under the default `--max-batch-prefill 4`. Only `execute_prefill` calls `try_speculative_burst`, so queued requests got no speculative decoding at all, B = 1 or batched. `--max-batch-prefill 1` was needed to reach the burst.

The history says this was not intended: the batched prefill (`60e50b0c`) predates the burst (`6a816947`), the burst's window collector drains exactly these queued rows, and `docs/benchmark_results/gemma4-mtp-batched-gb10-2026-10-07.md` lists the skip as a pre-existing gap it worked around.

### 1.2 Exactness probe (item 3)

`run_dflash_burst` gates on `exactness_allows(bs)` before the drafter loads. `run_dflash_burst_batched` checked only `supports_batched()`. The doc comment still called Qwen 3.5 permissive, although `dflash_exactness_allows` now probes.

### 1.3 Batched burst divergence found during verification

With the routing fix alone, default-flag concurrent requests reached the batched DFlash burst, and on GB10 its output diverged from sequential requests at temperature 0. The divergence is pre-existing: the unchanged main binary with `--max-batch-prefill 1` produces the same diverging bytes. A window of two identical prompts, where no round clamps, diverges too, so the batched verify forward itself is not byte-identical to the single-row one. The B = 1 exactness probe cannot see that.

## 2. Technical Decisions

### 2.1 Route a burst-eligible head through `execute_prefill`

While no classic rows are decoding, a head the burst would serve takes `execute_prefill`. The new `queued_head_takes_speculative_burst` mirrors the burst's gates without side effects: empty active batch, shared KV budget, `should_burst_for_sequence`, a busy slice slot, and for DFlash the pre-drafter gates in `dflash_burst_would_engage` (adopted prefix, target variant, exactness verdict, greedy-only drafter). A head the burst would only decline keeps the batched prefill.

- Eligibility is the head's alone. Rows behind a declined head are drained into the batched prefill with it, as before.
- MTP keeps classic batched prefill unless `MLXCEL_ENABLE_MTP_BATCH` is set, matching its existing window decline.
- A server without a drafter is unchanged.
- The decision is recorded in the `execute_prefill_action` doc comment.

### 2.2 Require an empty active batch

A run-to-completion burst admits nothing to the active batch, and the tick policy keeps choosing Prefill while the batch has room and the queue is non-empty. Bursting every queued head while classic rows were decoding would hold their decode off for as long as the queue stays non-empty. With classic rows active, queued rows keep the batched prefill as before. This came from a read-only review and was added in commit `9aeffffd`.

### 2.3 Batched DFlash window is opt-in

`MLXCEL_ENABLE_DFLASH_BATCH` defaults to off. By default queued DFlash heads each run the exact B = 1 burst, one per tick. Setting the flag restores batched windows. A B = 1-only DFlash family (LFM2, Muse Glimmer, Laguna) also stops collecting a window it could only decline, so its head gets the B = 1 burst. The PR marks this as going beyond the issue text and as the one decision to review.

### 2.4 Shared exactness helper

`run_dflash_burst_batched` now applies the same `exactness_allows(bs)` gate and decline warning as `run_dflash_burst` before `ensure_loaded`, through a shared `dflash_target_exactness_allows` helper. The batched variant gate uses `dflash_target_supports_batched`. The stale doc is fixed.

## 3. Changes

- `src/server/batch/scheduler/request_memory.rs`: routing in `execute_prefill_action` and its doc comment.
- `src/server/batch/scheduler/speculative_finalize.rs`, `src/server/batch/queue.rs`, `src/server/batch/scheduler/mod.rs`: the eligibility check and its wiring.
- `src/server/batch/speculative_burst.rs`, `src/server/batch/dflash_target.rs`: shared exactness helper, batched exactness gate, opt-in window policy.
- `src/models/speculative_exactness.rs`: `#[cfg(test)] seed_decision_for_test`, which writes the memoized verdict directly. It bypasses the probe, the `qmv_wide` retry and `MLXCEL_MTP_ALLOW_INEXACT`.
- `src/models/qwen3_5_tests.rs`, `src/models/mod.rs`: `qwen3_5_tests::two_layer_model` is now `pub(crate)`.
- `docs/environment-variables.md`: new `MLXCEL_ENABLE_DFLASH_BATCH` row; the `MLXCEL_ENABLE_MTP_BATCH` row no longer says a window needs `--max-batch-prefill 1`. `docs/CONTINUOUS_BATCHING.md` updated.

## 4. Verification

### 4.1 Tests

`scheduler_dflash_routing_tests` (new) runs the scheduler over the two-layer synthetic Qwen 3.5 with default `max_batch_prefill`, two equal-length queued requests, and a DFlash dispatch whose drafter path does not exist (a burst that reaches the drafter load fails its rows, which shows which burst ran):

- `queued_equal_length_dflash_requests_reach_the_batched_burst_as_one_window`: with the batched window enabled, both rows reach `try_run_burst_batched` as one window in one tick.
- `queued_dflash_head_runs_the_b1_burst_by_default`: the head runs the B = 1 burst, the second request stays queued.
- `queued_dflash_requests_keep_batched_prefill_when_the_burst_would_decline`, `queued_dflash_requests_keep_batched_prefill_while_classic_rows_decode` (a mirostat row is decoding) and `queued_requests_without_speculative_dispatch_keep_batched_prefill`: the rows still take the batched prefill.

`qwen35_batched_dflash_declines_a_failing_exactness_verdict_before_drafter_io` (new): a failing verdict returns the window untouched, sends no client event and leaves the drafter unloaded.

Revert checks:

- Removing the routing condition fails the first two routing tests (`no row may take the classic prefill: 2 active`).
- Removing the batched exactness gate fails the exactness test (`Ok` with drafter-load errors instead of `Err`).
- Forcing the DFlash window policy on fails `queued_dflash_head_runs_the_b1_burst_by_default`.
- Removing the active-batch condition fails `queued_dflash_requests_keep_batched_prefill_while_classic_rows_decode`.

All pass with the fix.

Local runs: `cargo fmt --all -- --check`, `cargo clippy --profile test-fast --features cuda --lib --tests -- -D warnings`, the lib test binary filtered to `server::batch::` (551 passed, 9 ignored), `speculative_exactness` (16 passed) and `qwen3_5_tests` (20 passed), all with `--test-threads=1` under `gpu-lock`. The five contract checks pass.

### 4.2 Real checkpoint (GB10, CUDA)

Target `~/models_archive/mlx/qwen3.5-4b-4bit`, drafter `~/models_archive/mlx/qwen3.5-4b-dflash`, `MLXCEL_SDPA_DETERMINISTIC=1`, `RUST_LOG=info,mlxcel::server::batch=debug`, test-fast `mlxcel-server --model <target> --model-draft <drafter> --draft-kind dflash` with otherwise default flags (`max_batch_size=4`, `max_batch_prefill=4`, prompt cache on, block 4 from the measured sm_121 default). Driver: a primer request, then two equal-length (23 prompt tokens) temperature-0 `/v1/chat/completions` with `seed: 1` and `max_tokens: 96` sent 0.25 s later so they queue together, then the same two sent one at a time. Compared text is reasoning plus content. Correctness only, no throughput numbers.

| Binary, flags | Concurrent rows served by | Concurrent == sequential |
|---|---|---|
| main 74f9ae1d, default | classic batched prefill and decode, no burst lines | yes, yes |
| main, `--max-batch-prefill 1` | one `DFlash batched burst ... batch_size=2` | no (row 0 diverges after about 230 chars), yes |
| main, `--max-batch-prefill 1 --no-prompt-cache` | batched burst | no, yes |
| main, `--max-batch-prefill 1 --draft-block-size 16` | batched burst | no, yes |
| main, `--max-batch-prefill 1`, two identical prompts | batched burst, `clamped_rounds=0` | no, no |
| this PR, default | two `DFlash accept lengths` B = 1 bursts | yes, yes |
| this PR, default + `MLXCEL_ENABLE_DFLASH_BATCH=1` | one `DFlash batched burst ... batch_size=2` | no, yes (same bytes as main with `--max-batch-prefill 1`) |
| this PR, no drafter | classic | yes, yes |

With this PR and default flags the concurrent rows get speculative decoding, and their text equals the sequential requests, the main binary's classic output, and the no-drafter server's output. The no-drafter server's concurrent and sequential texts are byte-identical to main's. Both checks were repeated on the final head (after the review follow-up) with the same results.

## 5. Review Notes

A read-only review found the gate mirroring and the B = 1 refactor correct. Changes made from it: the active-batch condition (section 2.2), the test seed helper no longer runs the production gate, and the doc comments now state the head-only decision and the MTP behaviour. Points left as they are:

- Default concurrent DFlash traffic is now served as back-to-back B = 1 bursts rather than classic batched decode, so a later request waits for earlier ones and receives its tokens when its burst finishes. This is the speculative decoding the issue asks for queued requests to get, and it is what `--max-batch-prefill 1` already did; the classic alternative is exact too. No latency or throughput comparison was taken on this shared box. Keeping main's classic behaviour for queues of 2+ is a one-line change in `queued_head_takes_speculative_burst` (require `dispatch_runs_batched_window()` in the DFlash arm) if that trade is not wanted.
- With `MLXCEL_ENABLE_MTP_BATCH=1`, a window that collapses or declines sends the head to classic single prefill and re-enqueues siblings at the back. That is pre-existing `try_speculative_burst` behaviour (reachable before with `--max-batch-prefill 1`) and is not changed here.

## 6. Not Addressed

- Item 1 of #2442 (per-row accepts, the B = 4 versus B = 1 parity gate, throughput) stays open. The divergence in section 1.3 means that gate currently fails on this host for these prompts, independent of the clamp; the opt-in flag should come off once a batched exactness check holds.
- No throughput comparison of one B = 1 burst per tick against classic batched decode was taken.

## 7. Related

- Issue #2442: follow-up to #2353 and PR #2440 (`TECHNICAL_REPORTS/pr-2440-dflash-batched-uniform-accept.en.md`). Its comment records the batched-burst divergence as a finding relevant to item 1.

## 8. Change Summary

13 files changed, 727 insertions, 62 deletions.

| Hash | Type | Message |
|------|------|---------|
| `a6c0dd7e` | fix | route queued DFlash heads to the burst (#2442) |
| `9aeffffd` | fix | keep batched prefill while classic rows decode |
