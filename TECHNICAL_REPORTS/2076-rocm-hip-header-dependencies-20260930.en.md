# Technical Report: PR #2076 - Rebuild ROCm HIP objects when an included header changes

**Date**: 2026-09-30

**Status**: Implemented and validated on the gfx1151 host; pending merge.

**Languages**: CMake (ROCm overlay), Markdown

**Risk Level**: Low (build glue only; no kernel or runtime code changes. The first build after the change recompiles every HIP object once. Metal and CUDA builds never copy `patches-rocm/`)

## Executive Summary

Issue #2075 (part of epic #1801) found that the ROCm overlay compiles each `.hip` file with its own `add_custom_command` whose only dependency is the `.hip` file itself. CMake never learned which headers an object includes, so a warm build did not recompile any `hip_objs/*.o` when only a header changed. Fresh builds were correct. Reused build trees were not, and in this epic reused build trees are the merge gate: with hosted CI down, `make verify-rocm` in a long-lived worktree is the only check a PR passes before it merges. A header-only overlay fix could therefore be "verified" against kernels compiled from the old header.

The PR has `hipcc` write a depfile per object (`-MD -MF <obj>.d`) and hands it to CMake with `DEPFILE`, so the build tool tracks every header an object reads, directly or transitively. On gfx1151 an edit to `kernel_utils.hpp` now recompiles the 39 objects that include it (0 of 41 before), an edit to `reduce/reduce.hpp` recompiles exactly its 7 includers, and a no-change rebuild recompiles nothing. The change is recorded as `LOCAL_FIXES.md` item 26 and documented in `docs/installation.md`.

## 1. Why This Matters to Epic #1801

The defect did not produce wrong kernels in a clean build. What it broke was the reliability of the local gate the epic depends on. Three cases in this epic ran into it:

- **#2059 (`kernel_utils.hpp`).** PR #2059 (d1128266) fixed `get_2d_grid_dims` with a divisor (`LOCAL_FIXES.md` item 22), the grid-size bug behind the `HSA_STATUS_ERROR_MEMORY_FAULT` in `strided_scan` on 48- and 64-head Mamba2 layers. Among overlay files it changed only `kernel_utils.hpp`. In any worktree whose `target/` predated that commit, `scan.hip` was never recompiled, so the tree kept running the old grid code while its tests reported the fix as merged.
- **#2051 (`lru_cache.h`).** The fix for `MLX_ROCM_FFT_CACHE_SIZE` (merged as PR #2073, item 25) lives in `lru_cache.h`. Its first run still behaved like the old code. It took effect only because `fft.hip` was edited as well (a comment stating the parsing rule), which made that one object stale by its own dependency. Item 25 originally said so explicitly; this PR rewrites that sentence to point to item 26.
- **#2072 (bisect).** A warm-tree bisect of the intermittent `rocm_mxfp4_quant` failure could not be trusted across commits that differed only in headers, because stepping between them rebuilt no HIP object.

The common thread: any merge gate that runs in a reused tree could test stale kernels, and nothing in the build output said so. After this PR, header-only overlay changes reach the HIP objects like any other change, and a later bisect in a warm tree compiles what each commit actually contains.

## 2. Problem Statement

`src/lib/mlx-cpp/patches-rocm/mlx/backend/rocm/CMakeLists.txt` compiles HIP sources outside CMake's HIP language support: a `foreach(hip_src ${HIP_SOURCES})` loop emits one `add_custom_command` per file that runs `hipcc -c`. A custom command is opaque to CMake. It knows only the dependencies listed in `DEPENDS`, which was `${hip_src}`. The `.cpp` sources added through `target_sources(mlx ...)` were never affected, because CMake scans their includes itself.

The objects live under `target/<profile>/build/mlxcel-core-*/out/build/_deps/mlx-build/mlx/backend/rocm/hip_objs/`. Headers such as `kernel_utils.hpp`, `device.h`, `lru_cache.h` and `quantized/*.hpp` are shared by many `.hip` files (39 include `kernel_utils.hpp` alone), so a header fix is exactly the kind of change that touches the most kernels while triggering the fewest rebuilds.

## 3. Change Summary

- **`patches-rocm/mlx/backend/rocm/CMakeLists.txt`**: under `if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.20)`, each command adds `-MD -MF ${hip_obj}.d` to the `hipcc` line and passes `DEPFILE ${hip_obj}.d`, with policy `CMP0116` set to `NEW` when it exists. Below 3.20, every object instead depends on a `file(GLOB_RECURSE ... CONFIGURE_DEPENDS)` list of every `*.h`, `*.hpp` and `*.cuh` under `mlx/`. A comment block explains the mechanism and the fallback.
- **`patches-rocm/LOCAL_FIXES.md`**: a new section, "Fixes to the fork's build", with item 26 (mechanism, dependency chain, measurements, limits; upstreaming candidate for #1813). Item 25's sentence about per-source dependencies now reads in the past tense and points to item 26.
- **`docs/installation.md`** (ROCm, Build): incremental builds track headers; a bare `touch` of an overlay header rebuilds nothing; ROCm and system headers are tracked, device-only includes and the compiler are not; and how to force a clean HIP rebuild (delete `hip_objs/` for the profile and touch an overlay file so Cargo reruns the build script, or `cargo clean -p mlxcel-core`).

Commits: `9d17bb1e` is the fix, docs and item 26; `b8d8b437` is the review follow-up that widened the fallback glob from the backend directory to all of `mlx/` and stated the depfile limits.

## 4. Technical Decisions

### Depfile from the compiler, not a header glob

The issue ranked the two options: `DEPFILE` first, a header glob only if `DEPFILE` is unusable. A glob makes every object depend on every header, so one edit to `reduce/reduce.hpp` would recompile all 41 objects instead of 7. A depfile is also the only option that follows transitive includes and headers outside the backend directory without maintaining a list. The glob is kept as a fallback because it is never stale for MLX headers, only slow.

### Why 3.20 and `CMP0116`

`build.rs` builds through `cmake::Config` without setting a generator, so the build uses Unix Makefiles unless `CMAKE_GENERATOR` is set. `DEPFILE` on `add_custom_command` works with the Makefile generators only from CMake 3.20. `CMP0116` NEW makes Ninja resolve relative depfile paths against the binary directory, as the Makefile generators do. The overlay's top-level `CMakeLists.txt` requires 3.25, which already implies `CMP0116` NEW and makes the fallback branch unreachable today. The explicit guard and policy set keep the file correct if that minimum is ever lowered.

### Fallback glob covers all of `mlx/`

The issue proposed globbing the rocm backend directory. The review follow-up widened it to every header under `mlx/`, because HIP sources also include MLX headers outside the backend (for example `mlx/primitives.h`). A backend-only glob would have been stale for exactly those headers.

### Track the build-tree copies, not the overlay originals

The overlay reaches the build tree through `configure_file(... COPYONLY)` in `src/lib/mlx-cpp/CMakeLists.txt`, and `hipcc` includes the copies under `_deps/mlx-src` (the `.hip` file's own directory and `-I${PROJECT_SOURCE_DIR}`). The depfiles therefore name the copies by absolute path; 0 of 41 depfiles mention `patches-rocm`. The full chain is:

1. An overlay edit changes a file under `patches-rocm/`, which `build.rs` watches with `rerun-if-changed`, so Cargo reruns the build script.
2. CMake reconfigures, and `configure_file(COPYONLY)` rewrites the copy only when its content differs.
3. make compares the copy's mtime with the object's, through the depfile.

This is the correct thing to track, since the copies are what the compiler reads. The PR does not try to point depfiles back at the originals.

## 5. Measured Rebuilds

gfx1151, CMake 3.31.6, Unix Makefiles, warm `test-fast` tree. Each row rebuilt with `cargo test --profile test-fast --features rocm --test rocm_strided_scan --no-run` and counted `hip_objs/*.o` newer than a marker file.

| Change | Before | After |
|---|---|---|
| No change | 0 | 0 |
| Comment appended to `kernel_utils.hpp` | 0 of 41 (the build-tree copy was rewritten) | 39: every object whose depfile names it (all but `event.o` and `quantized/qmv_tiled_kernel.o`) |
| Revert of that edit | 0 | the same 39 |
| Comment appended to `reduce/reduce.hpp`, then reverted | not run | 7 each time: `reduce.o`, `reduce/{all,col,init,row}_reduce.o`, `layer_norm.o`, `rms_norm.o` |

Both includer sets also match an independent walk of the `#include` graph over the build-tree sources, so the depfiles are not just self-consistent.

Caveats that come with these numbers:

- **`touch` alone does nothing.** Because `configure_file(COPYONLY)` rewrites a copy only when its content changed, touching an overlay header leaves the copy's mtime alone and rebuilds no object. A reproduction has to change content. This is by design and stated in the docs.
- **Headers included only on the device side are not tracked.** The depfile comes from the host compilation. It lists ROCm and system headers (so updating ROCm rebuilds every HIP object) but not device-only includes. Today that is `rocwmma/rocwmma.hpp` in `flash_attention_wmma.hip`; `quantized/qmm.hip` includes it on the host side too, so its depfile lists it. The compiler binary itself is not tracked either. After upgrading `hipcc` or rocWMMA, force a clean HIP rebuild as `docs/installation.md` describes.
- **The first build after picking up the change recompiles all 41 objects once.** The `hipcc` command line changed, so every object is out of date one time. This is what heals existing warm trees, including any still holding pre-#2059 `scan.hip` objects, but it costs a full HIP compile on the first gate run in each reused worktree.

## 6. Validation

PR author (gfx1151):

- The rebuild table above, plus the `#include` graph cross-check.
- Ninja was not available for the full mlxcel build. The same pattern (a `configure_file` copy, a depfile, `CMP0116` NEW) was checked with Ninja 1.13 and Unix Makefiles on a two-file probe project: an edit recompiles only the includer, a touch or no change rebuilds nothing, and the depfile names the build-tree copy.
- The documented forced clean rebuild: deleting `hip_objs/` without a touch rebuilt nothing (Cargo skipped the build script); after touching the overlay `CMakeLists.txt`, 41 objects and 41 depfiles were rebuilt. On that fresh HIP build, `rocm_strided_scan` (3 passed) and `rocm_slice_update_reduce` (8 passed), both with `--test-threads=1`, confirmed that #2059's header-only `get_2d_grid_dims` fix holds.
- After the review follow-up commit (a comment and the fallback glob only), a rebuild reconfigured and recompiled no HIP object.
- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt` and `cargo test --test dead_doc_pointers` passed.

Orchestrator verification (gfx1151):

- `make verify-rocm` on the branch at base `0d17303d` ran every step. Versions, kernel dtype keys, kernel port dispatch, llama-compat, fmt, workspace clippy with `--features rocm`, and the ROCm smoke run (32 tokens) passed.
- `verify-test-rocm` failed in four targets, none related to this PR: the three known baseline failures (`layers::tests::prefill_dense_gemm_matches_qmm_bytes_where_eligible`, and #2037's `gelu_approx_matches_mlx_nn_bit_for_bit` and `tests::family_order_is_exhaustive`), and the pre-existing intermittent `rocm_mxfp4_quant::quantized_matmul_matches_dequantized_reference` tracked in #2072.
- After the branch was rebased onto `a6e07542`, the orchestrator resolved a `LOCAL_FIXES.md` conflict by keeping item 24 from #2070, then this PR's edited item 25 and new item 26. It then re-ran the fast script gates and the `rocm_slice_update_source` and `rocm_strided_scan` tests.

## 7. Learning Points

- **A custom command knows only what you tell it.** Compiling with `add_custom_command` trades CMake's built-in include scanning for control over the compiler line. Any build that compiles this way needs a `DEPFILE` (or an explicit header list), or header edits silently stop propagating.
- **A merge gate is only as good as its build's staleness.** The gate here reported green on a reused tree that had not compiled the change under test. When CI is replaced by local runs in long-lived trees, incremental build correctness becomes part of the verification story, not a convenience.
- **Editing the including file hid the bug.** #2051's fix worked only because `fft.hip` was also edited. A fix that "works after touching the caller" is a signal to check the build graph, not just the code.
- **Copy-if-different layers change what "changed" means.** With `configure_file(COPYONLY)` between the source and the compiler, mtime-based tracking only sees content changes. Reproductions and forced rebuilds have to account for that, which is why the documented clean rebuild deletes `hip_objs/` and touches an overlay file to make Cargo rerun the build script.

## 8. Caveats and What Is Not Verified

- **Ninja on the full build** was not run; only the probe project.
- **Metal and CUDA** were not run (not available on this host). The change is confined to the ROCm overlay, which those builds never copy.
- **Device-only includes and compiler upgrades** are still not tracked; the documented clean rebuild is the answer for those.
- **The fallback branch** is unreachable with the 3.25 minimum and was not exercised on an older CMake.
- **Post-rebase re-runs**: the orchestrator re-ran the fast script gates and the two ROCm tests after the rebase onto `a6e07542`; this report does not restate their outcome.

## 9. Remaining Work

- #1813: propose item 26 to the fork with the other upstreaming candidates.
- #2072: the intermittent `rocm_mxfp4_quant` failure; its bisect can now be repeated in a warm tree for header-only differences.
- The three baseline `verify-test-rocm` failures (`prefill_dense_gemm_matches_qmm_bytes_where_eligible`, and #2037's two) are tracked outside this PR.

Refs: #2075 (closed by this PR), #1801, #2059, #2051, PR #2073, #2072, #2070, #1813, #2037.
