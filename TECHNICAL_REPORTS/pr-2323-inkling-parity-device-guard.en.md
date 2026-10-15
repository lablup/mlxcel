# PR #2323: Hold the device guard in the Inkling exact-parity tests

**Date**: 2026-10-16
**Status**: Implemented and verified on CUDA (GB10) before merge; issue #1622 stays open until the orchestrator's post-merge loop result is posted
**Risk**: Low (test-only change: three tests take an existing guard and one test helper reports mismatches, no production code)

## Summary

Issue #1622 is an intermittent failure of `vision::inkling_vl::tests::mixed_prefill_scatter_order_is_normalized_text_then_image_then_audio` under the full multi-threaded lib suite. It passes in isolation. This PR classifies it as a test-harness race on MLX's process-global default device, not a defect in the scatter, merge or tower path, and not load-sensitive numerics in the production sense. The PR uses `Refs #1622` because the issue's acceptance criteria also ask for 20 full `--lib` runs, which the orchestrator runs after merge.

1. Sibling tests such as `vision::merge_tests::cpu_device` move the default device to the CPU under `lock_default_device()`. The Inkling-VL zero-tolerance parity tests took no lock, so in a parallel libtest run one side of a comparison could be built on the CPU and the other on the GPU. The two backends differ in the last bits (`0xbfcd2ef1` vs `0xbfcd2ef0`), and the comparison uses `allclose` with rtol = atol = 0.
2. The two exact tests in `vision::inkling_vl` and `models::inkling::tiny_tests::mtp_partial_accept_restores_then_replays_kv_and_conv_state_exactly` now hold `mlx_test_guard()`, which takes the default-device lock.
3. `arrays_equal` now returns a mismatch report (shapes, first differing flat index with both values and bit patterns, mismatch count, max difference) instead of a bare bool, which is what the issue asked for.

## Problem

The test asserts that one call sequence returns bit-identical results across two invocations in one process: `expected` is `prepare_input_embeddings` followed by `merge_audio_embeddings`, and `actual` is `prepare_input_embeddings_with_audio`, which is those same two calls. An ordering defect in the scatter would therefore fail on every run, including in isolation. It did not. The single observation (2026-09-04, `cargo test --workspace`, 7722 passed, 1 failed) left no data beyond a boolean, so the cause could be a harness race, load-sensitive numerics in the tower, or a rare scatter defect, and each calls for a different fix.

The per-test CPU guards in `vision::merge::tests` landed in #1595 on 2026-09-03, one day before the observation. Since MLX's default device is a process-global static, that change made the race possible. It is the same class as #2058 (llmjp_vl, fixed in #2060).

### Evidence

| Check | Result |
|-------|--------|
| Same inputs on CPU vs GPU | Differ bitwise: embed RMSNorm `0xbfcd2ef1` vs `0xbfcd2ef0` at index 0, audio row differs when only the audio merge runs on the CPU, max abs diff 1.6e-7 |
| 1000 in-process repeats, both call paths, plus 50 fresh models | 0 mismatches |
| 25 separate processes | 25 of 25 passed, one output hash across all probe processes |
| Probe thread switching the device like `cpu_device()`, no guard | 882 of 1000 failed (175 of 400 in a first run), each a last-bit diff at index 0 |
| Same probe with `mlx_test_guard()` held | 0 of 1000 failed |
| `vision::` with `--test-threads=1` | 25 of 25 runs passed (549 passed, 0 failed) |
| Nightly verify since 2026-09-04 (Metal, `lablup-macmini-12`) | 49 runs, test ran in 48 and passed in all 48, one run failed before tests ran |

The out-of-process and in-process repeats rule out nondeterminism in the scatter, merge or tower path: the fixture is keyed by name, so no HashMap order is involved, and that path iterates no maps. The nightly record fits too, since every nightly passes `--test-threads=1`, which closes the race, and only the issue's multi-threaded `--workspace` run hit it.

## Technical decisions

- Hold the existing `mlx_test_guard()` instead of adding `serial_test`. The issue forbids a `#[serial]` attribute, a `--test-threads=1` change or a loosened tolerance before classification, and the guard is the lock the sibling tests already use, so it adds no dependency.
- Keep the tolerance at exactly zero. The comparison is meant to detect scatter ordering errors, and the CPU/GPU difference is a harness artifact, not a reason to accept drift.
- Guard `image_prefill_keeps_hmlp_prepared_embeddings_on_classic_wrapper_path` and the MTP rollback test as well. Both compare two computations exactly and have the same exposure, even though only the mixed-prefill test was observed to fail.
- Extend `arrays_equal` in place, as the issue's reuse note asks, rather than adding a second helper. It uses `!=` on host copies to match `allclose` with `equal_nan = false`. NaN never equals anything and -0.0 equals 0.0.
- The device-switching probe is not committed. It reproduces the race but depends on timing, so it is evidence and not a regression test.

## Changes

- `src/vision/inkling_vl.rs`: `arrays_equal` returns `Result<(), String>` with the mismatch report; the two exact tests hold `mlx_test_guard()` and panic with the report.
- `src/models/inkling/tiny_tests.rs`: `mtp_partial_accept_restores_then_replays_kv_and_conv_state_exactly` holds `mlx_test_guard()`.

## Verification

- CUDA (GB10, `--profile test-fast`): `models::inkling` (28 tests) and `vision::inkling_vl` (5 tests) pass; clippy `-D warnings`, fmt and the five contract checks are clean.
- Revert check: the probe thread fails the unguarded comparison 882 of 1000 times and the guarded one 0 of 1000 times.
- No real checkpoint was used because only test code changed.

## Not verified and follow-up

- A multi-threaded `vision::` run cannot be measured on CUDA. All 6 runs, with or without the fix, aborted after 76 to 293 tests with `cudaStreamEndCapture ... failed`, before reaching this test, so the fix is shown by the probe and not by an end-to-end multi-threaded pass.
- The 20-run full `--lib` loop from the acceptance criteria is left to the orchestrator. Issue #1622 stays open until its result is posted.
- Any new test that compares two computations at zero tolerance should take `mlx_test_guard()` for the same reason. The mismatch report makes a recurrence distinguishable at a glance: a last-bit diff at index 0 points back at the device, a permutation or partial write points at the scatter.
