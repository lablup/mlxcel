# ROCm fixes prepared for NripeshN/mlx `rocm-support`

mlxcel's ROCm backend is vendored from the `rocm-support` branch of [NripeshN/mlx](https://github.com/NripeshN/mlx/tree/rocm-support) (see `src/lib/mlx-cpp/patches-rocm/UPSTREAM`). Fixes made here that apply to the fork itself are listed in `src/lib/mlx-cpp/patches-rocm/LOCAL_FIXES.md`. This directory turns them into pull requests for the fork, one directory per PR, so the overlay stops drifting from its source (lablup/mlxcel#1813).

**Nothing here has been submitted, and nothing is submitted by automation.** The maintainer's rule: fork-side fixes become PRs to `NripeshN/mlx` `rocm-support`, prepared here and submitted by hand. Automation never pushes to, forks, or opens issues, PRs or comments on NripeshN/mlx or ml-explore/mlx; only read-only fetches and API reads were used to prepare this.

## Packages

State checked on 2026-09-30. The fork's `rocm-support` head is `75915908` (merged 2026-07-19), which is the commit `UPSTREAM` records, and the fork has no open PRs; so none of these fixes has landed in the fork, and every package was built against that head.

| Package | LOCAL_FIXES items | Applies cleanly to `75915908` | Depends on | Status | Upstream link |
|---|---|---|---|---|---|
| [01 quantized matmul dispatch](01-quantized-matmul-dispatch/pr.md) | 8, 10, 12, 13 | yes | none | ready for manual submission | not submitted |
| [02 ArgReduce grid limit](02-arg-reduce-grid-limit/pr.md) | 11 | yes | none | ready for manual submission | not submitted |
| [03 Scatter size argument and narrow index dtypes](03-scatter-gather-index-types/pr.md) | 14, 17 | yes | none | ready for manual submission | not submitted |
| [04 SliceUpdate tail and source donation](04-slice-update-tail-and-donation/pr.md) | 21, 24 | yes | none | ready for manual submission | not submitted |
| [05 Hadamard](05-hadamard/pr.md) | 16 | yes | none | ready for manual submission | not submitted |
| [06 blocking-sync before the first HIP queue](06-blocking-sync-before-first-queue/pr.md) | 20 | yes | none | ready for manual submission | not submitted |
| [07 FFT through hipFFT](07-fft-hipfft/pr.md) | 18, 25 | no: applies after 05 and 06 (adjacent-line conflicts with 05 in `CMakeLists.txt` and `primitives.cpp`) | 06 at runtime, 05 for a clean apply | ready for manual submission, after 05 and 06 | not submitted |
| [08 launch-geometry helpers](08-launch-geometry-helpers/pr.md) | 19, 22 | yes | none | ready for manual submission | not submitted |
| [09 SDPA fallback on CPU streams](09-sdpa-cpu-stream-fallback/pr.md) | 23 | yes | none | ready for manual submission | not submitted |
| [10 HIP header dependencies](10-hip-header-dependencies/pr.md) | 26 | yes | none | ready for manual submission | not submitted |
| [11 CPU-stream BLAS over fine-grained memory](11-cpu-blas-finegrained/pr.md) | 27 | yes (also on top of 01 to 10) | none | ready for manual submission | not submitted |

Packages are grouped the way the fork would review them: all of 01 is `quantized/qmm.hip`, 03 and 04 are the two halves of `indexing.hip` (scatter/gather and slice update), 08 is the two `kernel_utils.hpp` grid helpers. All eleven patches also apply in sequence (01 to 11) with `git am`, so they can be submitted in any order that respects 07's dependency.

Each directory holds the patch (`git format-patch` output, authored by the maintainer, no AI attribution), `pr.md` (PR title, a header table for the submitter, and the PR body below a rule), and the reproduction (`repro.py`, `repro.hip` or `repro.sh`).

## LOCAL_FIXES entries not packaged

| Item | Why |
|---|---|
| 1 to 6 | Retarget-only: core-file merges and API drift between the fork's MLX base (`39886de4`) and mlxcel's pin. They go upstream when the fork itself merges newer MLX, and they are recorded in `CORE_RESIDUAL.diff` and `LOCAL_FIXES.md`, not here. |
| 7 (GPU fault reporting) | Deferred. It is built on `Event::error()`, which ml-explore/mlx added in #3742 (`06f154bc`, 2026-08-17); the fork's MLX base predates it, so the patch does not apply. The wait changes that end a hang on a faulted queue (`HipEvent::wait`, `AtomicEvent`, `CommandEncoder::synchronize`) could be split out for the current fork, but that would be a different change from the one mlxcel ships and verified. Package it once the fork merges MLX at or past #3742. |
| 9 (expert-batched gather qmv opt-in) | Held. It disables a path that returns wrong bf16 results on gfx1151 but has no root cause yet; LOCAL_FIXES.md says to propose it only after one. |
| 15 (`SearchSorted`) | Not applicable yet: the primitive arrived in ml-explore/mlx#4035 (`5ec30acd`, 2026-08-10), after the fork's MLX base, so the fork has no `SearchSorted` to implement. Package it once the fork merges MLX at or past #4035. |
| 29 (bf16 GEMMs leave the WMMA kernel from 128 rows on RDNA 3.5) | Not packaged yet. The row ceiling was measured on gfx1151 only, and the exported `quantized_matmul_runs_dequant_gemm` exists for mlxcel's dense prefill check; a fork PR would carry the ceiling alone, with measurements from at least one more RDNA 3 or RDNA 3.5 device. |

## What was verified, and what the submitter still has to do

Verified while preparing (2026-09-30):

- Every patch was produced by a 3-way merge of the mlxcel commit that made the fix onto the fork file, with retarget-only hunks and item 9 left out and mlxcel-specific comments rewritten. `git apply --check` passes for each package on `75915908` (07 on `75915908` plus 05 and 06), and `git am` of all eleven in order succeeds.
- After applying 01 to 10, the fork's `mlx/backend/rocm/` differs from mlxcel's overlay only by the retarget items (2 to 6), items 7, 9 and 15, and comment wording.
- The evidence in each PR body comes from the mlxcel change and its tests, measured on gfx1151 (Radeon 8060S, HIP 7.15) with the backend retargeted onto MLX `81ba1c6a`; each body says so.

Not verified, to do before submitting each PR:

- **Build the fork with the patch.** The patches were not compiled against the fork head: preparing them did not build the fork. The fork's MLX base is older than mlxcel's pin, so check at least that it compiles; the fork's `build_rocm.yml` builds a wheel with `CMAKE_ARGS="-DMLX_BUILD_ROCM=ON -DMLX_ROCM_ARCHITECTURES=gfx1151 -DBLA_VENDOR=OpenBLAS"`. Package 03 adds a compile-time argument-size check to `add_kernel_node`; a fork call site that mlxcel never compiled would fail there, which is the point of the check but should be fixed in the same PR.
- **Run the reproduction** on the unpatched fork and on the patched one. None of the scripts has been run against a build of the fork; they follow the mlxcel tests that verified each fix. Several unpatched cases fault the GPU queue, so those scripts run each case in a child process with a timeout.
- **Run `pre-commit run --all-files`** (clang-format, black, cmake-format), which the fork's PR template asks for. It was not run.
- **Tick the checklist honestly.** The PR bodies end with the fork's template checklist, unticked.

## Contribution guidelines and the fork's stance on automated PRs

Read on 2026-09-30, read-only.

- **The fork's `CONTRIBUTING.md`** (at `75915908`) is MLX's older generic text: fork and open a PR, add tests for code that should be tested, run benchmarks for efficiency-sensitive changes, update docs for API changes, every PR needs passing tests and at least one review, format with `pre-commit` (clang-format, black). Issues are disabled on the fork (`has_issues: false`), so a PR is the only channel.
- **The fork's PR template** (at `75915908`) has "Proposed changes" and a four-box checklist (read CONTRIBUTING, ran pre-commit, added tests, updated docs). It says nothing about AI or automation. The `pr.md` bodies follow its shape.
- **Recent PR history.** 13 PRs, all against `rocm-support` except one, 8 merged, from 4 contributors; the last merge was #13 on 2026-07-19 and nothing has merged since. Merges happen without formal GitHub reviews (0 reviews on the merged PRs sampled, a few comments each). Titles follow `fix(rocm): ...` / `ROCm: ...`, which the patches match.
- **Upstream MLX's policy matters too.** The fork's branch is the head of ml-explore/mlx#2300 (open), so what merges into the fork is headed for ml-explore review. On 2026-08-20 ml-explore/mlx added an AI usage policy (#4331): its PR template now opens with "I understand it is strictly prohibited to use AI to write PR description" and an "AI usage disclosure" field, and its `CONTRIBUTING.md` says contributors stay accountable for AI-assisted work, which must meet the same standards, and that written material should be in the contributor's own voice. The fork's branch predates this and its template does not carry it.
- **What that means for these packages.** The patches, commit messages and `pr.md` bodies were drafted with AI assistance for the maintainer to review. Nothing in the fork's own rules forbids that, but ml-explore's policy forbids AI-written PR descriptions for its repository, and the fork's maintainer may apply the same expectation. Decide before submitting; the conservative path is to treat each `pr.md` as notes, write the PR description in your own words, and state AI assistance in the description if the fork asks for a disclosure.

## Submitting one package by hand

1. Fork NripeshN/mlx under your account (a manual step; automation does not create forks) and branch from `rocm-support`.
2. `git am docs/mlxcelverse/upstream/<package>/*.patch` (for 07, apply 05 and 06 first or wait until they merge).
3. Build, run the reproduction before and after, run `pre-commit`.
4. Open the PR against `NripeshN/mlx:rocm-support` with the title and a description based on `pr.md`.
5. Record the link: set the "Upstream link" cell above and add "Proposed upstream: <link>" to the matching `LOCAL_FIXES.md` items.

## Keeping the packages current

The packages target the fork head named above. After a fork sync (`scripts/mlxcelverse/sync_from_fork.sh`, see `docs/mlxcelverse/rocm-overlay.md`), check which packages the fork has absorbed, drop them here and from the "applies to the fork" wording in LOCAL_FIXES.md, and regenerate the rest against the new head. A new LOCAL_FIXES entry that applies to the fork gets a directory here in the same shape.
