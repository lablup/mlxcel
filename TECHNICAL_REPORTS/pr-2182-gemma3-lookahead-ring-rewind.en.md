# PR #2182: Gemma 3 decode lookahead via a ring-aware teardown rewind

**Date**: 2026-10-07
**Status**: Implemented and verified on CUDA (GB10); Metal not measured
**Risk**: Medium (re-enables pipelined decode for Gemma 3, adds state to every Gemma 3 sliding-window cache on the server path)

## Summary

PR #2139 turned the decode lookahead off for every model-owned family, because the teardown trim reached only `CachePool` caches and the speculative appends stayed in the model's own state. Gemma 3 4B lost about 10 tok/s on GB10. A cursor trim cannot serve Gemma 3 either: once the sliding window wraps, each decode write overwrites the K/V of the position that just left the window, and a trim leaves the speculative K/V in that slot.

The PR adds an opt-in undo log to `RotatingKVCache` (`cache/decode_undo.rs`), two `LanguageModel` hooks (`supports_decode_lookahead_rewind`, `rewind_decode_appends`, delegated in `LoadedModel` and the VLM wrapper), the Gemma 3 implementation, and the scheduler wiring: the gate admits a model-owned family only with the capability, every teardown calls the rewind, and a failed rewind finishes the request with an error. Closes #2159.

## Design notes

- Each logged write records its starting `offset` and `idx` (post over-window pre-trim, pre-wrap), its slot, and whether the slot was valid. Inside a `DecodeLookaheadAppendScope` an overwrite also copies the slot's `[B, H, 1, D]` K and V rows. The scheduler enters the scope only around the prime forward, which makes every append a teardown can unwind; synchronous steps copy nothing and are not rewindable.
- `rewind_decode_writes(n)` validates first (buffering off, FP16, log enabled, `n <= offset`, consecutive entries, rows present for every overwrite), then restores rows newest first, re-zeroes warmup slots, and sets the exact cursors. Writes older than the log are accepted only while the ring is still chronological. Any other mutation clears the log.
- Lazy row copies pin the pre-write buffer, which stops MLX from donating it to the write's `slice_update`: the whole window is then copied every step. Gemma 3 therefore schedules all pending copies of a forward in one `async_eval` before the forward is evaluated. The issue's rejected alternative (retaining pre-write handles) pins the same buffers and would cost the same.
- Gemma 3 logs only on scheduler sequence caches (depth 2, the shared `DECODE_LOOKAHEAD_MAX_SPECULATIVE_APPENDS`), and only when every sliding layer is FP16. The CLI fallback slot keeps no log.
- At `--parallel 1` Gemma 3 is allocated `ModelOwned`, not `DenseKvCache`; the per-sequence backend check admits that allocation when the model has the capability.
- On a failed rewind the sequence is set to `Finished(Error)` directly, since a same-tick `length` or `stop` finish cannot transition to `Error` and would otherwise donate the desynchronized state.

## Verification on GB10 (CUDA, release, `--features cuda`)

- Unit and scheduler tests: `cache::` 556, `server::batch` 495, `models::gemma3` 32, `vision::` 545 pass. The new lockstep scheduler tests compare pipelined against force-sync state bit for bit after the wrap; with the row restore disabled both fail.
- Server, gemma-3-4b-it-4bit, greedy: short prompt, 1906-token prompt, and 4 concurrent requests are byte-identical between lookahead and `MLXCEL_FORCE_SYNC=1` at `--parallel 4` and `--parallel 1`. Concurrent requests are queued behind a blocker prefill so both arms see the same batch history; without it force-sync alone gave two different outputs across runs.
- Throughput, interleaved rounds, null arm (byte-identical binary), host-idle CPU gate:

| Gemma 3 4B, 200 tokens | new | null | pre-#2139 | force-sync | main |
|---|---|---|---|---|---|
| short prompt, tok/s | 87.3 | 87.2 | 87.6 | 76.2 | 75.7 |
| after 1906-token prompt, tok/s | 72.7 | 72.9 | 77.1 | 67.3 | 67.4 |

- Llama 3.2 1B dense lookahead: 270.4 vs 270.0 tok/s on main, identical output.

Measurement history worth keeping:

- Without the CPU gate (other agents compiling on the host) both lookahead arms showed 40 to 60 tok/s rounds while force-sync did not; the gate removed them, and the null arm then stayed within 1%.
- Wrapped regime, cost isolation: unflushed row copies 64.5 tok/s, flushed 72.4, no copies (unsound) 76.9.
- The first version copied rows on every write, which slowed force-sync in the wrapped regime from 67.3 (main) to 65.1; the scope removed that (67.3 vs 67.4).

## Not verified on this host

- Metal throughput and numerics of the row copy and flush.
- The branch is based on `0ee13dfe`; #2162 (CUDA SDPA switch) landed later and was not rebuilt here.

## Follow-up

- The wrapped regime still runs 5.7% below the unsound pre-#2139 pipeline: the cost of the row copies and their extra `async_eval`. Folding the copies into the forward's own evaluation, with an ordering that keeps donation, is the candidate.
- Gemma 4, Llama 4, AFMoE, Muse Glimmer, the SSM/hybrid families, and INT8/Turbo4Asym sliding storage stay synchronous through the default capability.
