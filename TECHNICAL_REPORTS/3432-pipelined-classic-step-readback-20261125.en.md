# PR #3432: pipelined classic steps keep the decode overlap (issue #3430)

**Date**: 2026-11-25

**Status**: Draft PR open (closes #3430). Validated on an M3 Ultra (512 GB).

## Executive Summary

PR #3426 left a gap open: a classic decode step run inside an Inkling-Small MTP session cost about 22.3 ms, against 21.7 ms for the engine's own decode step, and pure classic decode inside the session ran at about 41 tok/s against 46. Eleven knobs had been tried without effect.

The cause is the token read. The session's classic steps are pipelined, so step `i + 1` is queued before step `i`'s token is read. The read went through `array_to_raw_bytes`, which queues a `contiguous()` copy of the token and evaluates it. That copy sits behind step `i + 1` on the same stream, so the read waited for the whole next forward and the pipeline ran as synchronous decode.

The fix reads through `array_evaluated_bytes`, which waits on the token's own completion event and queues nothing. The engine has read its tokens this way since #632. Pure classic inside the session now runs at 0.97x to 0.99x the engine, the default MTP width goes from 0.81x to 0.87x classic, and width 2 goes from 0.89x to 0.95x. Output is identical to classic decode in every arm.

## 1. How the cause was found

The search started by comparing the two paths call by call. The graphs matched. Both paths run the same Inkling forward, the same row-invariant scope and a one-row argmax. The difference was in timing. Scratch `Instant` instrumentation gave these per-step numbers (128 tokens, prose prompt):

| Path | Graph build | `async_eval` | Token read wait | tok/s |
|---|---|---|---|---|
| engine classic decode | 1.5 ms | 20.0 ms | about 0 | 46.3 |
| engine, forced synchronous after each step | 1.1 ms | 16.3 ms | 6.7 ms | 41.6 |
| session classic step, before | 1.1 ms | 18.5 ms | 4.1 ms | 41.6 |
| session classic step, after | 1.1 ms | 20.6 ms | 0.27 ms | 45.0 |

The engine's `async_eval` blocks for almost the whole step. MLX stops encoding while more than 10 command buffers are in flight, so the next step's encoding overlaps the GPU, and the token read is free. The session's classic steps ran exactly as fast as the engine forced synchronous. That points to the read, not to the graph.

These variations each left the session at 41.4 to 41.9 tok/s, which ruled them out:

- building the step from the engine's own forward, inside the speculative append scope;
- evaluating only the token;
- `MLXCEL_KV_INPLACE_WRITE=0`;
- runs of 16 or unlimited steps per call.

The engine reads with `try_tokens_to_host`, which calls `array_evaluated_bytes`. The session read with `materialize_argmax_i32_vec`, which calls `array_to_raw_bytes`. The second one queues a `contiguous()` copy and evaluates it. MLX signals that evaluation's event at the end of the stream's queue, which is after step `i + 1`. A no-op `reshape` has the same effect: the DFlash classic steps (#3417) read `reshape(token)` and waited the same way.

## 2. The change

- `drafter::dflash::materialize_argmax_i32_vec` reads through `array_evaluated_bytes`. A non-row-contiguous array still takes the `contiguous()` copy. For the helper's other callers (drafter proposals, the verify argmax, the engine verify round), the argmax is not yet scheduled, so the read evaluates it as before, minus a copy.
- `drafter::dflash::classic_steps` passes the token itself instead of `reshape(token, [1])`.
- `drafter/dflash/readback_tests.rs` adds a Metal test. It schedules a token, queues about 76 ms of matmuls behind it, and requires the read to return while most of that work is still running. With the old read it failed 3 of 3 runs (read 76 ms, later work under 1 us). With the fix it passed 3 of 3.

## 3. Measurements

M3 Ultra, greedy, `--show-reasoning`, chat template, ABAB in one locked block per table. The "before" and "after" arms come from one build that kept the old read behind a scratch switch.

Pure classic inside the session (drafting disabled by a scratch switch), load average 7.5 to 12.5:

| Tokens | Prompt | Engine classic | Old read | New read | New read, no 16-step cap |
|---|---|---|---|---|---|
| 128 | prose | 46.48, 45.38 | 40.41, 41.58 | 44.49, 45.18 | 45.52, 45.69 |
| 128 | code | 46.53, 46.15 | 41.59, 41.84 | 45.37, 44.95 | 44.85, 45.34 |
| 256 | prose | 45.26, 45.65 | 41.38, 41.25 | 44.56, 44.63 | 45.00, 44.89 |
| 256 | code | 45.57, 45.46 | 41.23, 41.31 | 44.59, 44.82 | 44.89, 44.86 |

What remains is the first step of each round-loop call, which has no queued predecessor: 34 ms of read wait over 128 tokens at the 16-step cap, and 4 ms without the cap. The cap stays so that a server tick returns as often as a drafted round does.

MTP with the fallback, 128 tokens, load average 6.2 to 8.4:

| Prompt | Classic | Width 10, before | Width 10, after | Width 2, before | Width 2, after |
|---|---|---|---|---|---|
| prose | 46.12, 46.05 | 37.01, 37.71 | 40.30, 40.44 | 41.04, 41.04 | 43.95, 44.11 |
| code | 45.86, 45.81 | 37.06, 37.28 | 39.66, 39.76 | 40.95, 41.16 | 43.77, 43.86 |

Inkling 2 Mini DSpark with `MLXCEL_DFLASH_ADAPTIVE_WIDTH=1`, 128 tokens:

- Prose went from 51.7 to 52.8 and 53.1 tok/s. The first old-read run, 38.9 tok/s, was excluded as warm-up.
- Code went from 49.8 and 50.0 to 51.8 tok/s.
- Output was identical in every run.

## 4. Risks

- The read now waits on the array's own event instead of the stream's tail. For an array that is not yet scheduled, `array::eval` evaluates it, which matches the old behavior without the copy.
- A failed read now surfaces as a panic with the bridge's error message. Before, the C++ exception aborted the process.
- Inkling-Small MTP still runs below classic decode. Acceptance is the limit, as #3426 found.
