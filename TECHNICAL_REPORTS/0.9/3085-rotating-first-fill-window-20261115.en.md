# Technical Report: PR #3085 - Keep only the window on a rotating cache's first fill

**Date**: 2026-11-15
**Status**: Completed
**Risk Level**: Medium

## Executive Summary

On a multi-token first fill, `RotatingKVCache` stored every prompt token, and only the first decode step trimmed it to the window. Every sliding-window layer therefore held the whole prompt through a single-pass prefill. The first fill now stores the newest `max_size + 32` positions as a fresh buffer, scheduled through the shared deferred-error `async_eval`, while attention still sees the full K/V for that layer. On an M5 Max at 28.7K tokens, gemma-4-12b's single-pass peak drops from 27.38 to 18.67 GB and gemma-3-4b's from 15.14 to 7.03 GB. Greedy output is byte-identical and prefill time is unchanged.

## 1. Problem Statement

gemma-4-12b has 40 sliding layers with a 1024-token window. At 28.7K tokens they held about 8.75 GiB of K/V that no later layer reads (#3062). Every family that uses `RotatingKVCache` for sliding layers had the same issue.

## 3. Technical Decisions

- **The kept length is `max_size + 32`, not `max_size`.** Speculative verify and padded-prefill rewind trim up to one NA tile (32 positions) after a fill. With 32 extra positions, the stored buffer has exactly the chronological layout a chunked-prefill continuation leaves. Decode, continuation appends, `trim`, `rewind_padded_prefill`, `enable_speculative_buffer`, snapshot restore and the Turbo4Asym sidecars therefore return exactly what the full fill returned. Int8, Turbo4, Turbo4Delegated and Turbo3Asym go through the FP16 path.
- **The ragged batched-prefill cohort keeps the full fill.** The server rewinds its shorter rows by the whole length difference, which can exceed 32. `FullFirstFillScope`, a thread-local scope entered only when some row needs more than 32 positions of padding, keeps the old full first fill for that forward. `rewind_padded_prefill` now refuses, with an error, a rewind that would expose dropped positions, instead of silently corrupting the window.
- **One helper, not a copy.** `materialized_axis_tail(x, axis, n_keep)` generalizes #3063's `materialized_state_tail` to the cache's time axis (2), and shares its deferred-error handling.

## 7. Change Summary

| model | prefill | peak, main | peak, branch | output |
|---|---|---|---|---|
| gemma-4-12b-it-4bit | single pass | 27.38 GB | 18.67 GB | identical |
| gemma-4-12b-it-4bit | 2048 chunks | 14.17 GB | 14.17 GB | identical |
| gemma-3-4b-it-4bit | single pass | 15.14 GB | 7.03 GB | identical |
| gemma-3-4b-it-4bit | 2048 chunks | 5.25 GB | 5.25 GB | identical |

Tests:
- 9 new `cache::first_fill_tests` drive a trimmed cache and a full-fill cache through the same operations and compare the K/V exactly in FP16, Int8 and Turbo4Asym.
- Two memory tests check that the trimmed first fill releases the prompt and that the scoped control keeps it.
- `server::batch` passes (610 tests), and so do the gemma3, gemma4, kv_snapshot and afmoe model tests.

Details: `docs/benchmark_results/rotating-first-fill-window-m5max-2026-11-15.md`.

## 8. Follow-up Actions

- No unit test drives the scheduler's ragged-cohort scope through a model, and no server or MTP end-to-end run used a real checkpoint.
