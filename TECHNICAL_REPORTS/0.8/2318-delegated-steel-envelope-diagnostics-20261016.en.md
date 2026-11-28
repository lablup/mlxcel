# PR #2318: Exercise the steel kernel in the delegated envelope tests and classify #1712 failures

**Date**: 2026-10-16
**Status**: Implemented and verified on CUDA (GB10) and Metal (Nightly verify); issue #1712 stays open
**Risk**: Low (test-only change: one new test-support module and a rewrite of two existing macOS-gated tests, no production code)

## Summary

Issue #1712 is a rare, non-reproducible failure of the macOS-gated `delegated_steel_envelope_*` tests: the output RMS differs from the graph reference by `4.6824e-1`. This PR classifies the failure as far as the evidence allows, makes the next recurrence self-classifying, and fixes a coverage gap found on the way. The root cause is not established, so the PR uses `Refs #1712` and the issue stays open.

1. The "steel envelope" tests have not exercised the steel kernel since May. `update_and_turbo4_delegated_attention` tries the dequant-first SDPA route first, and that route has been default-on since f77e98bc (2026-05-08, two days after the test landed in 09d9cc69). Under the default env the wrapper returns before `attention_turbo4_delegated_steel`, and nothing else calls it. The tests, and the launcher CONTRIBUTING.md names for post-pin-bump re-validation, compared dequant-first SDPA against the cold-only composition. The `4.6824e-1` came from that pairing, not from `turbo4_delegated_steel_sdpa`.
2. Both tests now also call `attention_turbo4_delegated_steel` directly and still check the wrapper's output, so the pairing that failed keeps its coverage.
3. A new test-only module `cache/delegated_parity_diag.rs` reports route gates, a cache-to-cache comparison, per-output deviation from the graph reference, dropped-contribution signatures and an in-process re-run of every route when a parity check fails. Thresholds are unchanged.

## Problem

The failure was seen twice, both times in multi-threaded workspace gates, and left no data beyond one RMS number. Without data the only options were to retry or to guess at the kernel.

### What the wrong value says

Step 0 was rebuilt on CUDA (cold_offset=8, hot_offset=1, same seeds) and candidate wrong outputs were measured against the graph reference:

| Candidate wrong output | RMS vs reference |
|------------------------|------------------|
| Cold V rows zeroed | 4.6820e-1 |
| K read at full buffer capacity | 4.03e-1 |
| All-zero output | 5.12e-1 |
| Hot V dropped | 5.15e-2 |
| Missing inverse rotation | 7.85e-1 |

The reported `4.6824e-1` is 4e-5 from the cold-V-dropped candidate (the two correct paths differ by about 1.4e-4 from each other). So on one side of the comparison the cold contribution was all zeros: the bulk rotated-dequant output (dequant-SDPA route), the cold weighted-sum output (cold-only route), or a zero `v_rescale`. A never-written fresh Metal buffer reads as zeros, which explains why the value repeated to five digits while the failure itself was rare. The value alone cannot say which side or which of the three.

### Recurrence record

Nightly verify since 2026-09-08: 48 runs. 44 ran the test and all 44 passed (0accedd90 through 537ebe10d, four Mac minis). One run failed before the test step (38430123889) and one was cancelled. Six passes ran on MLX 243bdf09. Every nightly passes `--test-threads=1`, while both failures came from multi-threaded workspace gates. That fits the shared-device hazard of #163 and #1008 but does not prove it. In this window the delegated kernel source changed only by #2029 (dispatch through `select_kernel_port`, kernel bodies untouched), the #2305 move and the #1796 header update. A code read of the three launchers found no unwritten output region, out-of-bounds index or missing barrier.

## Technical decisions

- Diagnostics are test-only and run only on failure. Rejected: adding runtime checks or zero-fill of output buffers in the launchers. That could mask the bug, and it changes production code on the strength of an unconfirmed hypothesis.
- Thresholds are unchanged. Loosening them would hide the very failure being tracked.
- The in-process re-run of every route on both caches separates a persistent defect (re-run fails again) from a transient one (re-run passes), and the cross-process comparison against the nightly record separates a threading hazard from a kernel bug.
- The wrapper pairing is kept next to the new direct kernel call instead of replaced, because the wrapper pairing is the one that failed in the field.
- The CUDA candidate table came from a throwaway test on the same binary and is not committed.

## Changes

- `src/lib/mlxcel-core/src/cache/delegated_parity_diag.rs` (new, 460 lines, test-only): failure report with route gates, comparison of the two caches' K, packed V, rescale and hot V, each observed output against the graph reference with the max-abs location, dropped-contribution signatures, and the route re-runs.
- `src/lib/mlxcel-core/src/cache/delegated_parity_diag_tests.rs` (new, 2 tests): run the full report path on CUDA.
- `src/lib/mlxcel-core/src/cache/turbo_tests.rs`: the two steel envelope tests call the steel kernel directly and still check the wrapper output; they report through the new module.
- `src/lib/mlxcel-core/src/cache.rs`: module registration.

## Verification

- CUDA (GB10): `cache::delegated_parity_diag_tests` (2 tests) pass and run the full report path.
- The two macOS-gated tests compile on Linux with their gate lifted (clippy `-D warnings`, `--lib --tests`). The gate was restored before commit.
- Metal: Nightly verify run 38777513435 (macOS Apple Silicon, `workflow_dispatch` on head 821c343a) passed fmt, clippy, the full test-fast lib suite (9005 passed, 0 failed, 168 ignored) and the video tests. Both `delegated_steel_envelope_*` tests, now calling the steel kernel directly, passed on Metal. This was the first Metal run of the steel kernel through these tests since May, so a pass is also a finding: the kernel is not broken in the single-threaded nightly configuration.

## Not verified and follow-up

- The root cause of #1712 is not established. A single Metal pass does not rule out the shared-device hazard, since the original failures only appeared in multi-threaded workspace gates.
- Issue #1712 stays open. On the next recurrence the failure output answers three questions: which route produced the zeros, whether the two caches differed in K, packed V, rescale or hot V, and whether a re-run in the same process reproduced it.
- If the next recurrence points at an unwritten output buffer, the fix belongs in the launcher that owns that buffer, and the diagnostics module should be kept as the regression harness.
