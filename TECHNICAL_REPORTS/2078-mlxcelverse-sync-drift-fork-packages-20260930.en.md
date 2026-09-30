# Technical Report: PR #2078 - mlxcelverse ROCm sync, drift checks and fork PR packages

**Date**: 2026-09-30

**Status**: Implemented and validated on the gfx1151 host (branch `update/issue-1813-mlxcelverse-sync` at `53c8f122` on `origin/main` `c5a71cfa`); pending merge.

**Languages**: Python and Bash (tooling), Make, Markdown, git patches; one comment-only change in a HIP file

**Risk Level**: Low for the build (no functional change to compiled code; the only overlay edit is a comment revert in `rope.hip`). Medium for process: `make verify` and `make verify-rocm` gain a new gate that fails any pin bump which skips the ROCm retarget, on every host.

## Executive Summary

Issue #1813 (phase 3 of epic #1801) asked for tooling and a written procedure for the two upstreams of mlxcel's ROCm overlay, the MLX pin (ml-explore/mlx) and the fork the backend is vendored from (NripeshN/mlx `rocm-support`), plus sending the overlay's fork-side fixes back to the fork.

The PR adds `scripts/mlxcelverse/rocm_overlay.py` with five subcommands (`records`, `drift`, `sync`, `retarget`, and the `export-tree`/`api-report` helpers behind `check_api_drift.sh`), the entry points `sync_from_fork.sh` and `check_api_drift.sh`, and a 29-check synthetic-history test. The central addition is a drift check that rebuilds each of the 15 core overlay files as "MLX pin plus the fork's own change to that file" and requires what remains, the overlay's residual, to match a reviewed record, `patches-rocm/CORE_RESIDUAL.diff`, in which every hunk carries a note naming its `LOCAL_FIXES.md` item. Unlike the old +/- line-count comparison, this sees an overlay that was already wrong before a bump. The offline half runs as `make verify-rocm-overlay`, now part of both `make verify` and `make verify-rocm`.

The PR also prepares eleven PR packages for the fork under `docs/mlxcelverse/upstream/` (patch, PR notes, reproduction), covering LOCAL_FIXES items 8, 10 to 14 and 16 to 27. None were submitted. The maintainer's rule is that automation prepares them and he submits them by hand, and ml-explore/mlx's AI usage policy (#4331) means each `pr.md` is notes for him to rewrite in his own words, not a finished description.

Building fork code was denied by the auto-mode permission check in the implementing session, so `check_api_drift.sh` end to end, compiling the packages against the fork, and running their reproductions on a fork build were not done. They are listed as steps before submission.

## 1. Problem Statement

The ROCm overlay (`src/lib/mlx-cpp/patches-rocm/`) is 108 backend files under `mlx/backend/rocm/` plus 15 whole-file copies of MLX core files merged with the fork's hooks. `UPSTREAM` records three commits: the fork commit `75915908`, the fork's merge base with MLX `39886de4`, and the MLX pin the core files were retargeted to, `81ba1c6a`. Before this PR:

- **No tooling** existed for either upstream. `scripts/mlxcelverse/` did not exist; the pin-bump procedure was three lines in `patches-rocm/README.md` and a paragraph in `CONTRIBUTING.md`. The retarget in #1801 had needed six hand fixes after 360 upstream commits.
- **The bump check could not see a wrong overlay.** Comparing an overlay's +/- line counts against the old and new upstream base confirms that upstream's changes arrived and the local ones stayed. It cannot tell whether the overlay was correct to begin with. A Metal overlay once carried half of an upstream fix for months until a reviewer compared it line by line (TECHNICAL_REPORTS/1772, section 3).
- **Nothing had gone back to the fork.** `LOCAL_FIXES.md` had grown to 27 entries, many marked "applies to the fork; to be proposed there", with no upstream link on any of them.
- **The records had drifted.** The README said 106 backend files (there are 108), several LOCAL_FIXES entries did not name the files they touch, and `rope.hip` differed from the fork only by a rewrapped comment that no entry recorded.

## 2. Change Summary

- **`scripts/mlxcelverse/rocm_overlay.py`** (new, about 1100 lines): `records`, `drift`, `sync`, `retarget`, `export-tree`, `api-report`. Git objects come from a cache repository (`~/.cache/mlxcel/mlxcelverse/mlx.git`) holding both remotes; the tool only fetches, never pushes.
- **`scripts/mlxcelverse/sync_from_fork.sh`** and **`check_api_drift.sh`** (new): the entry points the issue named. `check_api_drift.sh` configures MLX standalone with the ROCm options `build.rs` passes, builds with `make -k` so every failing translation unit is listed, and compares defined and undefined symbols across objects, since a primitive without a ROCm `eval_gpu` only appears as an undefined `mlx::core` symbol in a static libmlx.
- **`scripts/mlxcelverse/rocm_overlay_test.sh`** (new): 29 checks on a synthetic upstream and fork history built in a temporary repository.
- **`src/lib/mlx-cpp/patches-rocm/CORE_RESIDUAL.diff`** (new, generated): the residual of the 4 core files that do not rebuild byte for byte, each with a note.
- **`Makefile`**: new `verify-rocm-overlay` target, added to `verify` and `verify-rocm`.
- **`docs/mlxcelverse/rocm-overlay.md`** (new): the records, the tools, the pin-bump and fork-sync procedures, and a "ROCm overlay" section template for bump PR bodies and reports. `CONTRIBUTING.md`, `patches-rocm/README.md`, `docs/architecture.md`, `docs/README.md` and the overlay comment in `src/lib/mlx-cpp/CMakeLists.txt` point to it or list the new record.
- **`docs/mlxcelverse/upstream/`** (new): an index and eleven package directories.
- **Records**: README backend count 106 to 108; LOCAL_FIXES now names the files of items 2, 7, 11 and 20 and states the naming rule; `rope.hip`'s comment is reverted to the fork's wording.

Outside scripts and docs, the code diff is the comment revert in `rope.hip` and a reproduction `.hip` under `docs/` (package 06), which no build compiles.

## 3. How the drift check catches an overlay that was already wrong

### 3.1 The reconstruction

For each core file, `drift` builds the file the overlay should be:

```
reconstruct(path) = git merge-file  pin:path  fork-merge-base:path  fork:path
```

That is the pin's copy with the fork's own change to the file merged in, using the fork's merge base with upstream as the 3-way base, conflict markers kept. Using the fork's diff against its own merge base, applied on the pin, is what keeps upstream changes the fork picked up by merging MLX from being mistaken for fork changes. If the fork does not have the file, the reconstruction is the pin's copy.

The overlay is then diffed against the reconstruction. What remains is exactly what the overlay adds on its own: conflict resolutions and local fixes. At `81ba1c6a`/`75915908`, 11 of the 15 core files rebuild byte for byte. The other four carry a residual, all explained:

| Core file | Residual explained by |
|---|---|
| `mlx/backend/common/compiled.cpp` | item 1: keeps upstream's `negative_strides` tracking and the fork's constant-input skipping |
| `mlx/fast_primitives.h` | item 1: keeps upstream's `compile_options` and the fork's `output_input_aliases` |
| `mlx/io/safetensors.cpp` | item 1: keeps upstream's empty-tensor guard and the fork's ROCm `staged_write` |
| `mlx/backend/gpu/primitives.cpp` | item 24: `DynamicSliceUpdate` copies through `copy_gpu` and drops the fork's forced donation |

### 3.2 The reviewed record

The residual is committed as `CORE_RESIDUAL.diff`. Its header records the three commits and a SHA-256 of every core overlay file; each file with a residual needs a `# note <path>:` line naming its LOCAL_FIXES item. `drift` fails when the recomputed residual differs from the record, prints the difference hunk by hunk, and requires `--write` to accept it; `--write` keeps existing notes and flags new files as needing one.

This is how a wrong overlay surfaces. The residual is "everything the overlay does that neither upstream nor the fork does". An overlay that dropped half of an upstream change has a hunk there that deletes an upstream line, and the only way past the check is for a reviewer to see that hunk and write a note tying it to a LOCAL_FIXES entry. The initial generation of the record was itself that review of the existing overlay: every residual hunk was traced to items 1 and 24, and no half-applied upstream change was found. From then on:

- any edit to a core file changes its hash, so `records` (offline) fails with "changed since the residual was generated" until `drift` is re-run and reviewed;
- a pin bump or fork sync changes the header commits, so `records` fails until the residual is regenerated for the new commits.

The synthetic test covers the case directly: it replaces an upstream line of a core file with its pre-change form, and asserts that `records` fails on the hash, `drift` fails with "core residual differs", and the drift output contains the dropped upstream line as a new hunk.

### 3.3 Backend files and uncarried hooks

The backend files have no pin to reconstruct against, so `drift` compares each against the fork commit instead. Every file that is modified, local-only (`hadamard.hip`, `fft.hip`) or dropped relative to the fork must be named in `LOCAL_FIXES.md`. The check also lists fork changes to MLX files outside the backend that the overlay does not carry, and requires LOCAL_FIXES to say why.

Review found the first version too lenient: a bare file name anywhere counted as naming, so `mlx/device.cpp` named `mlx/backend/rocm/device.cpp` and the top-level `CMakeLists.txt` named the backend's. A short name now counts only on its own, and a bare file name only when no other overlay file shares it (`bd5de658`, with a regression case that fails on the earlier tool). This rule is also why the PR had to touch the records: LOCAL_FIXES items 2, 7, 11 and 20 now name their files, and `rope.hip`, whose only difference from the fork was a rewrapped comment, was reverted rather than recorded as a local fix.

### 3.4 What `sync --check` does not prove

`sync_from_fork.sh --check` against the recorded fork commit reproduces `patches-rocm/` byte for byte, which was one of the issue's acceptance items. It holds by construction: `sync` carries local fixes as the overlay's own difference from the old fork commit, so syncing to the same commit is an identity. It cannot detect an unrecorded local change; the procedure doc says so, and points to `drift` as the check that can.

## 4. The `make verify-rocm-overlay` gate

```make
verify-rocm-overlay:
	@python3 scripts/mlxcelverse/rocm_overlay.py records
	@bash scripts/mlxcelverse/rocm_overlay_test.sh
```

`records` reads only files in the repository and takes about two seconds together with the test. It checks that `UPSTREAM` is well formed and names the build's MLX pin (from `scripts/ci/mlx_pinned_commit.sh`), that the README's backend and core counts and core file list match the tree, that LOCAL_FIXES entries 1 to N each appear once, and that `CORE_RESIDUAL.diff` was generated for the current commits and current core-file hashes, with a note for every residual and no note without one. On this branch it reports 108 backend and 15 core files, 27 LOCAL_FIXES entries, 4 core files with a noted residual, pin `81ba1c6a`.

The target sits in **both** `make verify` and `make verify-rocm`. Pin bumps are usually made on a Mac, where `verify-rocm` never runs; with the gate in `verify`, editing `GIT_TAG` without running `rocm_overlay.py retarget` fails on the Mac immediately instead of first breaking an AMD build. The online checks (`drift`, `sync_from_fork.sh`, `check_api_drift.sh`) fetch or build and stay manual steps in the procedure.

## 5. The eleven upstream packages

Each directory under `docs/mlxcelverse/upstream/` holds a `git format-patch` against fork head `75915908` (authored by the maintainer, no AI attribution), `pr.md` (title, a header table for the submitter, and a draft body in the shape of the fork's PR template), and a reproduction. On 2026-09-30 the fork head was still `75915908`, the commit `UPSTREAM` records, and the fork had no open PRs, so none of these fixes had landed there.

| Package | LOCAL_FIXES items | Applies cleanly to `75915908` | Status |
|---|---|---|---|
| 01 quantized matmul dispatch | 8, 10, 12, 13 | yes | ready for manual submission, not submitted |
| 02 ArgReduce grid limit | 11 | yes | same |
| 03 Scatter size argument and narrow index dtypes | 14, 17 | yes | same |
| 04 SliceUpdate tail and source donation | 21, 24 | yes | same |
| 05 Hadamard | 16 | yes | same |
| 06 blocking-sync before the first HIP queue | 20 | yes | same |
| 07 FFT through hipFFT | 18, 25 | only after 05 and 06 (adjacent-line conflicts with 05 in `CMakeLists.txt` and `primitives.cpp`) | ready after 05 and 06, not submitted |
| 08 launch-geometry helpers | 19, 22 | yes | ready for manual submission, not submitted |
| 09 SDPA fallback on CPU streams | 23 | yes | same |
| 10 HIP header dependencies | 26 | yes | same |
| 11 CPU-stream BLAS over fine-grained memory | 27 | yes, also on top of 01 to 10 | same |

Package 11 was added after the branch was rebased onto `c5a71cfa`, which merged item 27 (#2072, PR #2079). The index records that each patch passes `git apply --check` on the fork head (07 after 05 and 06) and that all eleven apply in sequence with `git am`.

Items not packaged, with the reason recorded in the index: 1 to 6 are retarget-only and go upstream when the fork merges newer MLX; 7 (GPU fault reporting) is built on `Event::error()` from ml-explore/mlx#3742, which the fork's MLX base predates; 9 disables a wrong-result path that has no root cause yet; 15 (`SearchSorted`) implements a primitive the fork's base does not have (ml-explore/mlx#4035).

Packages are grouped as the fork would review them (all of 01 is `quantized/qmm.hip`; 03 and 04 are the two halves of `indexing.hip`; 08 is the two grid helpers in `kernel_utils.hpp`), and each patch was built by 3-way merging the mlxcel commit that made the fix onto the fork file, leaving out retarget hunks and item 9 and rewriting mlxcel-specific comments.

## 6. Contribution situation and the AI usage policy

### 6.1 Prepared by automation, submitted by hand

The maintainer's direction, restated in the issue thread and at the top of the index, is that fork-side fixes become PRs to NripeshN/mlx `rocm-support`, prepared under #1813 and submitted manually. Automation never pushes to, forks, or opens issues, PRs or comments on NripeshN/mlx or ml-explore/mlx; only read-only fetches and API reads were used. The index ends with the manual steps: fork, `git am`, build, run the reproduction before and after, `pre-commit`, open the PR, then record the link in the index and in the matching LOCAL_FIXES items.

### 6.2 The fork

Read-only on 2026-09-30:

- The fork's `CONTRIBUTING.md` is MLX's older generic text (tests, benchmarks for performance changes, docs for API changes, passing tests and one review, `pre-commit`). Issues are disabled, so a PR is the only channel.
- Its PR template has "Proposed changes" and a four-box checklist, and says nothing about AI or automation. The `pr.md` drafts follow its shape and leave the checklist unticked.
- 13 PRs from 4 contributors, 8 merged, the last on 2026-07-19; merges happen without formal GitHub reviews. Titles follow `fix(rocm): ...`, which the patches match. The fork has been quiet for over two months, so response time is unknown.

### 6.3 ml-explore/mlx's AI usage policy

The fork's branch is the head of ml-explore/mlx#2300, so what merges into the fork is headed for ml-explore review. On 2026-08-20 ml-explore/mlx adopted an AI usage policy (#4331): its PR template opens with "I understand it is strictly prohibited to use AI to write PR description" and adds an "AI usage disclosure" field, and its `CONTRIBUTING.md` holds contributors accountable for AI-assisted work and asks for written material in the contributor's own voice. The fork's branch predates this.

The patches, commit messages and `pr.md` bodies were drafted with AI assistance. The fork's own rules do not forbid that, but its maintainer may apply the upstream expectation. The index therefore treats each `pr.md` as **notes for the maintainer to rewrite in his own words**, recommends stating AI assistance if a disclosure is asked for, and leaves the decision to him. That is why every package is "ready for manual submission" rather than submitted, and why the acceptance item "the scale-dispatch fix is proposed upstream with a link recorded" stays open after this PR.

## 7. Technical Decisions

### Reconstruct against the fork's merge base, not the fork commit

Diffing the overlay against the fork's copy of a core file would mix in every upstream change between the fork's MLX base and the pin. Taking the fork's delta against its own merge base and merging it onto the pin isolates the three sources (upstream, fork, local), so the residual is only the local part. The same idea drives `sync` for core files: a 3-way merge from "pin plus old fork delta" to "pin plus new fork delta".

### A committed, reviewed residual instead of a zero-residual rule

Items 1 and 24 are legitimate local differences, so "the overlay must equal the reconstruction" would never pass. Recording the residual and requiring a note per file makes every legitimate difference explicit and makes any new one a visible diff in review. The file hashes in the header let the offline `records` check catch a core-file edit without fetching anything.

### Carry local fixes as the overlay's own difference

`sync` 3-way merges old fork commit, overlay and new fork commit for each backend file, so the difference between the overlay and the old fork commit (the full local fix list) is carried automatically, including local-only files. The alternative, replaying a separate patch list, would need that list kept in step with the overlay by hand. The trade-off, discussed in section 3.4, is that `sync --check` proves nothing about unrecorded changes; `drift`'s naming rule covers that.

### Offline checks in the gate, online checks manual

Only `records` and the synthetic test run in `make verify`: they need no network and no ROCm, and they finish in about two seconds on any host. `drift` fetches, and `check_api_drift.sh` needs an AMD host and a cold MLX build. Putting those in a gate would make `make verify` depend on the network and on hardware most contributors do not have.

### Treat fetched trees as untrusted

Git stores a `..` tree entry verbatim and a fetch does not fsck by default, so a hostile fork commit (or a `--fork-url`) could have made `sync` or `export-tree` write outside the overlay, for example `mlx/backend/rocm/../../../../.bashrc`. Security review reproduced it with a crafted tree. Every write or unlink of a tree-derived path now goes through `safe_dest`, tree listings use `-z`, and option-like fetch arguments are refused (`51b142fc`, `0119aee9`), with a test that builds such a tree through `git mktree`.

## 8. Validation

PR author (gfx1151):

- `sync_from_fork.sh --check` at the recorded fork commit, with a freshly fetched cache, reproduced `patches-rocm/` byte for byte. Because that is an identity (section 3.4), the author also ran a forward sync of a copy claiming the older fork commit `53cbdf8c` (34 files changed since): 30 files merged cleanly and 4 were flagged as real interactions with local fixes.
- `drift` passed against the recorded commits: 11 of 15 core files byte for byte, residual items 1 and 24.
- `git apply --check` of each package on `75915908` (07 after 05 and 06) and `git am` in sequence. The PR body's verification line says "all ten"; package 11 was added afterwards, and its commit records that it applies on its own and after 01 to 10.
- `cargo test --features rocm --test dead_doc_pointers` passed.

Orchestrator verification (gfx1151, branch at `53c8f122` on `c5a71cfa`):

- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt verify-rocm-overlay` passed. The records check reported 108 backend and 15 core files, 27 LOCAL_FIXES entries, 4 core files with a noted residual, pin `81ba1c6a`.
- `scripts/mlxcelverse/rocm_overlay_test.sh` passed 29 checks: good-overlay records and drift, `sync --check`, a forward sync, a fork conflict, a fork ahead of the pin, a `..` tree entry for both `sync` and `export-tree`, a pin bump and retarget, the already-wrong overlay, an unrecorded backend change (including one named only by a same-named file elsewhere), `.DS_Store`, gap-numbered LOCAL_FIXES, a wrong README count, a stray record, and `api-report` on a synthetic build log.
- The full test suite was not rerun for this PR. Outside scripts and docs the diff is a comment-only change in `rope.hip` and a `.hip` reproduction under `docs/` that no build compiles, so a suite run would only have exercised unchanged code, while two GPU-measuring units were running on the host.
- The orchestrator found the unit's rebase onto `c5a71cfa` and the item-27 commit unpushed, pushed them with `--force-with-lease`, and corrected the package count in the PR body.

## 9. Learning Points

- **Count checks confirm movement, not correctness.** A +/- line comparison across a bump can only say that the delta was preserved. Checking correctness needs a reference for what the file should be, and here that reference is constructible: pin plus the fork's delta against its own base.
- **Make the reviewed exception list the artifact.** The residual is small (4 files, 2 LOCAL_FIXES items) and each hunk is annotated, so the review a human would otherwise redo on every bump is recorded once and re-triggered only when something changes.
- **An identity check needs to be named as one.** `sync --check` at the recorded commit is byte for byte by construction. The first version of the procedure doc claimed it could detect an overlay changed outside the procedure; review corrected that, and the useful sync evidence became the forward sync from an older commit.
- **Name matching is a security-relevant heuristic.** Accepting a bare file name as "recorded" silently exempted the two most often edited backend files. Checks that decide whether a change is documented need the same care as checks that decide whether it is allowed.
- **Paths from fetched git trees are input.** Tooling that only fetches still writes files derived from someone else's tree; treat those paths as untrusted.

## 10. What Is Not Verified

The auto-mode permission check in the implementing session denied building fork code. As a result:

- **`check_api_drift.sh` was not run end to end.** The acceptance item "the drift check lists the six known breaks against the fork's merge base plus current upstream" is unchecked. The command to run is `scripts/mlxcelverse/check_api_drift.sh --rocm-from fork:75915908dfe5028335d318b10340313744fd3a8d` on an AMD host. `api-report` is covered only by the synthetic build log in the test.
- **No package was compiled against the fork.** The fork's MLX base is older than mlxcel's pin, so each patch must at least compile there. Package 03 adds a compile-time argument-size check to `add_kernel_node`, which may flag a fork call site mlxcel never compiled.
- **No reproduction was run on a fork build**, unpatched or patched. The scripts follow the mlxcel tests that verified each fix; several unpatched cases fault the GPU queue, so those scripts run each case in a child process with a timeout.
- **`pre-commit run --all-files`**, which the fork's template asks for, was not run on the patches.

The index lists all four as steps before submitting each PR. Separately: Metal and CUDA were not run (not available on this host); the PR touches no Metal or CUDA code, and `verify-rocm-overlay` is pure Python and git.

## 11. Remaining Work

- Before each submission: build the fork with the patch, run the reproduction before and after, run `pre-commit`, rewrite `pr.md` in the maintainer's own words, and decide on AI disclosure.
- After each submission: record the link in the index and add "Proposed upstream: <link>" to the LOCAL_FIXES items. The issue's acceptance item for the scale-dispatch fix (package 01) is met only then.
- Run `check_api_drift.sh --rocm-from fork:75915908...` once on an AMD host to confirm the six known breaks.
- Package item 7 once the fork merges MLX at or past #3742, item 15 at or past #4035, and item 9 once it has a root cause.
- After any fork sync, drop the packages the fork absorbed and regenerate the rest against the new head.

Refs: #1813 (closed by this PR), #1801, #1802, PR #1818, PR #1819, #1811, PR #2079, #2072, TECHNICAL_REPORTS/1772, ml-explore/mlx#2300, ml-explore/mlx#4331, ml-explore/mlx#3742, ml-explore/mlx#4035.
