# PR #2530: Drop the Metal compiled.cpp overlay that zeroed fused imag()

**Date**: 2026-10-22
**Status**: Completed (PR open, closes #2333)

## Executive Summary

PR #2530 removes the Metal `compiled.cpp` source overlay. The overlay cast every fused op input whose type differed from the op's output type. For `Imag` (complex64 in, float32 out) it emitted `Imag()(static_cast<float>(tmp_z))`. Metal's `complex_t` conversion keeps `.real`, so a compiled `imag()` of a complex64 value returned 0. `Real` happened to stay correct. At the current MLX pin the cast is not needed, because MLX ops promote their own inputs, so the fix is to retire the overlay and use upstream emission. The PR also makes the overlay step restore a retired file in a cached MLX checkout, and adds four compile-probe graphs with two tests. Metal was verified on the self-hosted runner in two arms: green with the fix, and only the imag test failing with the old overlay restored. 7 files, +178/-544, of which 509 deletions are the overlay file.

## 1. Problem Statement

The overlay copy of `mlx/backend/metal/compiled.cpp` was added in 1923da32 so that fused kernels with mixed operand types compiled. It cast any input whose type differed from the output type to the output type. That is wrong whenever the output type is narrower than the input in a way the conversion does not preserve. `Imag` takes complex64 and returns float32, and the cast to `float` drops the imaginary part. The same cast also broke comparisons (#1792, covered by earlier tests). Issue #2333 asked whether the overlay was still needed (option b) or should be fixed in place (option a).

## 2. Change Summary

| Area | Change |
|---|---|
| `overlays/metal/mlx/backend/metal/compiled.cpp` | Deleted (509 lines) |
| `src/lib/mlxcelverse/CMakeLists.txt` | Overlay list drops `compiled.cpp`; new restore step for retired overlays |
| `overlays/metal/README.md` | Overlay inventory updated |
| `cpp/mlx_cxx_compile_probe.cpp`, `cpp/mlx_cxx_bridge.h`, `src/lib.rs` | Probe graphs 7-10 |
| `compiled_dtype_cast_tests.rs` | `compiled_real_of_complex64_matches_eager`, `compiled_imag_of_complex64_matches_eager` |

## 3. Technical Decisions

### 3.1 Remove the overlay instead of fixing the cast (option b)

At the pin (`bff10220`), upstream `compiled.cpp` differs from the overlay only in the header comment and this cast block. The cast existed for mixed operand types, and `mlx/ops.cpp` no longer produces them:
- binary ops and comparisons astype both inputs to the promoted dtype;
- `logical_and` and `logical_or` cast to bool;
- `where` casts the condition to bool and both branches to the output dtype;
- `Real` and `Imag` take `complex64_t`;
- fused `AsType` is emitted upstream as `cast_to<T>` (ml-explore/mlx#4351).

So no fused op sees mixed operand types. CUDA has no such overlay and already runs the same graphs with upstream emission. Keeping a 509-line copy of an upstream file for a cast that is both unnecessary and wrong would also add a refresh burden at every pin move.

### 3.2 Restore retired overlays in a cached checkout

The overlay is copied into the fetched `_deps/mlx-src`. `build.rs` reuses that tree while the pin is unchanged, so deleting the file from the list alone would leave the patched `compiled.cpp` on every warm build. `mlx_apply_source_overlays` now runs `git diff --name-only -- mlx/backend/metal` in the fetched tree and runs `git checkout` on every modified file the overlay list no longer names. It logs `Restored upstream <path> (no longer overlaid)`. If the tree is not a git checkout, or `git diff` fails, it warns and skips. A failed `git checkout` of a retired file aborts configure, so a build never silently uses the old overlay.

### 3.3 Probe graphs that exercise the failure

Graphs 7-10 compute `real` and `imag` of `z = a + i*b` with nonzero imaginary parts. `z` is built inside the compiled graph in one pair and passed in as an eagerly built complex64 input in the other. Each result feeds a multiply and an add so the op fuses. Each test compares compiled and compiled-shapeless output against eager, and eager against host arithmetic. The #1792 tests are unchanged apart from comments.

## 4. Verification

### 4.1 Local (Linux, default features, CPU backend)

| Check | Result |
|---|---|
| Warm-cache restore, before | `git status --short mlx/backend/metal` in the cached `_deps/mlx-src` listed `compiled.cpp`, `device.cpp`, `kernels/utils.h`, `quantized.cpp` as modified |
| Warm-cache restore, after `cargo test ... --no-run` on the same cache | only `device.cpp`, `kernels/utils.h`, `quantized.cpp` modified; log shows `Restored upstream mlx/backend/metal/compiled.cpp (no longer overlaid)`; file byte-identical to upstream at the pin |
| `cargo test --profile test-fast -p mlxcel-core --lib compiled_dtype_cast_tests` | 7 passed |
| `cargo fmt --all -- --check`, `cargo clippy ... -D warnings` | Clean |

The CPU run validates the probe, the eager oracle against host arithmetic, and CPU compile. It does not exercise the Metal fix.

### 4.2 Metal (self-hosted macOS runner, nightly-verify dispatch)

| Arm | Run | Result |
|---|---|---|
| With the fix | [39703647781](https://github.com/lablup/mlxcel/actions/runs/39703647781) | Full `make verify` green. All seven `compiled_dtype_cast_tests` pass, including both new tests and the five #1792 tests |
| Reverted (throwaway branch with the new tests and the old overlay) | [39707309473](https://github.com/lablup/mlxcel/actions/runs/39707309473) | Only `compiled_imag_of_complex64_matches_eager` fails: `compiled [0.2, -0.5, 0.7, 0.0, 3.0] != eager [1.6, 0.100000024, 1.1, 2.0, 9.0]`. `compiled_real_of_complex64_matches_eager` passes, as predicted |

The reverted arm confirms the new test fails for the stated reason and the fix is what turns it green. The throwaway branch is deleted and its auto-filed tracking issue #2536 is closed.

## 5. Unverified and Open Items

- **Not run:** the fix was not checked on a Metal build of a real model, only through the probe tests.
- **Warm-build restore only observed on Linux.** The restore ran against a Linux cache. The same CMake code serves macOS, but no warm macOS cache was exercised in this PR.
- **Pin moves:** the overlay README no longer lists `compiled.cpp`. If a later pin reintroduces mixed-type fused inputs, the #1792 and complex64 probe tests are the detectors.

## 6. Learning Points

- An overlay is a copy of upstream code frozen at one pin. When the workaround targets behavior upstream has since fixed or restructured, re-diff the overlay against the pin before refreshing it. Here the diff was header-only plus the cast, which made deletion the smaller change.
- Removing a file from an overlay list is not enough while the fetched source tree is cached. Retire an overlay together with a step that restores the file.
- A fix for a backend-specific bug needs a test arm on that backend that fails without the fix. The CPU suite passed before and after, so only the reverted Metal run showed the test is meaningful.
