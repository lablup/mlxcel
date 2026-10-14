# ROCm fork fixes as patch packages (historical reference)

mlxcel's ROCm backend is vendored from the `rocm-support` branch of [NripeshN/mlx](https://github.com/NripeshN/mlx/tree/rocm-support) (see `src/lib/mlx-cpp/patches-rocm/UPSTREAM`). Fixes made here that apply to the fork itself are listed in `src/lib/mlx-cpp/patches-rocm/LOCAL_FIXES.md`.

**Nothing here has been submitted, and nothing will be.** Since 2026-10-06 fixes to the vendored ROCm fork stay in mlxcelverse and no PRs go to `NripeshN/mlx` `rocm-support` (lablup/mlxcel#2144, lablup/mlxcel#1813). The packages remain as a self-contained patch, rationale and reproduction per fix against fork `75915908`.

## Packages

State checked on 2026-09-30. The fork's `rocm-support` head is `75915908` (merged 2026-07-19), which is the commit `UPSTREAM` records, and the fork has no open PRs; so none of these fixes has landed in the fork, and every package was built against that head.

| Package | LOCAL_FIXES items | Applies cleanly to `75915908` | Depends on | Status |
|---|---|---|---|---|
| [01 quantized matmul dispatch](01-quantized-matmul-dispatch/pr.md) | 8, 10, 12, 13 | yes | none | not submitted (historical) |
| [02 ArgReduce grid limit](02-arg-reduce-grid-limit/pr.md) | 11 | yes | none | not submitted (historical) |
| [03 Scatter size argument and narrow index dtypes](03-scatter-gather-index-types/pr.md) | 14, 17 | yes | none | not submitted (historical) |
| [04 SliceUpdate tail and source donation](04-slice-update-tail-and-donation/pr.md) | 21, 24 | yes | none | not submitted (historical) |
| [05 Hadamard](05-hadamard/pr.md) | 16 | yes | none | not submitted (historical) |
| [06 blocking-sync before the first HIP queue](06-blocking-sync-before-first-queue/pr.md) | 20 | yes | none | not submitted (historical) |
| [07 FFT through hipFFT](07-fft-hipfft/pr.md) | 18, 25 | no: applies after 05 and 06 (adjacent-line conflicts with 05 in `CMakeLists.txt` and `primitives.cpp`) | 06 at runtime, 05 for a clean apply | not submitted (historical) |
| [08 launch-geometry helpers](08-launch-geometry-helpers/pr.md) | 19, 22 | yes | none | not submitted (historical) |
| [09 SDPA fallback on CPU streams](09-sdpa-cpu-stream-fallback/pr.md) | 23 | yes | none | not submitted (historical) |
| [10 HIP header dependencies](10-hip-header-dependencies/pr.md) | 26 | yes | none | not submitted (historical) |
| [11 CPU-stream BLAS over fine-grained memory](11-cpu-blas-finegrained/pr.md) | 27 | yes (also on top of 01 to 10) | none | not submitted (historical) |

Packages are grouped the way the fork would review them: all of 01 is `quantized/qmm.hip`, 03 and 04 are the two halves of `indexing.hip` (scatter/gather and slice update), 08 is the two `kernel_utils.hpp` grid helpers. All eleven patches also apply in sequence (01 to 11) with `git am`, so the series applies in order.

Each directory holds the patch (`git format-patch` output, authored by the maintainer, no AI attribution), `pr.md` (the PR title and body drafted at the time), and the reproduction (`repro.py`, `repro.hip` or `repro.sh`).

## LOCAL_FIXES entries not packaged

| Item | Why |
|---|---|
| 1 to 6 | Retarget-only: core-file merges and API drift between the fork's MLX base (`39886de4`) and mlxcel's pin. They follow the fork when it merges newer MLX, and they are recorded in `CORE_RESIDUAL.diff` and `LOCAL_FIXES.md`, not here. |
| 7 (GPU fault reporting) | Deferred. It is built on `Event::error()`, which ml-explore/mlx added in #3742 (`06f154bc`, 2026-08-17); the fork's MLX base predates it, so the patch does not apply. The wait changes that end a hang on a faulted queue (`HipEvent::wait`, `AtomicEvent`, `CommandEncoder::synchronize`) could be split out for the current fork, but that would be a different change from the one mlxcel ships and verified. Not packaged, and not planned. |
| 9 (expert-batched gather qmv: strided index reads, register reuse) | Not packaged (root-caused and rewritten 2026-10-05 with lablup/mlxcel#2066). The fork runs this kernel by default, and it reads `lhs_indices[b]` and `rhs_indices[b]` as flat arrays, which is wrong (and reads past both arrays) when MLX broadcasts the indices; the fix and the faster inner loop are confined to `quantized/qmm.hip`. A package needs a repro that calls sorted `gather_qmm` with `x` `[T, 1, K]` and `rhs_indices` `[T, 1]` on the unpatched fork, and should carry the strided reads apart from the rewrite so the fork can take the correctness fix alone. |
| 15 (`SearchSorted`) | Not applicable yet: the primitive arrived in ml-explore/mlx#4035 (`5ec30acd`, 2026-08-10), after the fork's MLX base, so the fork has no `SearchSorted` to implement. Not packaged, and not planned. |
| 29 (bf16 GEMMs leave the WMMA kernel from 128 rows on RDNA 3.5) | Not packaged. The row ceiling was measured on gfx1151 only, and the exported `quantized_matmul_runs_dequant_gemm` exists for mlxcel's dense prefill check; a fork PR would carry the ceiling alone, with measurements from at least one more RDNA 3 or RDNA 3.5 device. |
| 30 (custom-kernel `_shape` / `_strides` by value) | Not packaged (added 2026-10-04 with lablup/mlxcel#2064). It is confined to `custom_kernel.cpp`: the two declarations, header structs with an `elem_to_loc` overload, and the matching launch condition; a package needs a repro that launches a `hip_kernel` reading `<name>_shape` on the unpatched fork, which faults the queue. |

## What was verified

Verified while preparing (2026-09-30):

- Every patch was produced by a 3-way merge of the mlxcel commit that made the fix onto the fork file, with retarget-only hunks and item 9 left out and mlxcel-specific comments rewritten. `git apply --check` passes for each package on `75915908` (07 on `75915908` plus 05 and 06), and `git am` of all eleven in order succeeds.
- After applying 01 to 10, the fork's `mlx/backend/rocm/` differs from mlxcel's overlay only by the retarget items (2 to 6), items 7, 9 and 15, and comment wording.
- The evidence in each PR body comes from the mlxcel change and its tests, measured on gfx1151 (Radeon 8060S, HIP 7.15) with the backend retargeted onto MLX `81ba1c6a`; each body says so.

Not verified:

- **Build the fork with the patch.** The patches were not compiled against the fork head: preparing them did not build the fork. The fork's MLX base is older than mlxcel's pin, so check at least that it compiles; the fork's `build_rocm.yml` builds a wheel with `CMAKE_ARGS="-DMLX_BUILD_ROCM=ON -DMLX_ROCM_ARCHITECTURES=gfx1151 -DBLA_VENDOR=OpenBLAS"`. Package 03 adds a compile-time argument-size check to `add_kernel_node`; a fork call site that mlxcel never compiled would fail there, which is the point of the check but should be fixed in the same PR.
- **Run the reproduction** on the unpatched fork and on the patched one. None of the scripts has been run against a build of the fork; they follow the mlxcel tests that verified each fix. Several unpatched cases fault the GPU queue, so those scripts run each case in a child process with a timeout.
- **Run `pre-commit run --all-files`** (clang-format, black, cmake-format), which the fork's PR template asks for. It was not run.
- **Tick the checklist honestly.** The PR bodies end with the fork's template checklist, unticked.

## Keeping the packages

The packages are not maintained. When a fork sync shows the fork fixed one of them, delete that package together with its LOCAL_FIXES entry (as `docs/mlxcelverse/rocm-overlay.md` sync step 3 says).
