# PR #2533: Rename aux.rs and guard against Windows device names

**Date**: 2026-10-22
**Status**: Completed (PR open at report time; closes #2531)

## Executive Summary

`src/lib/mlxcel-xla/src/aux.rs` made the repository impossible to check out on Windows. `AUX` is a reserved device name with any extension and in any case, and Git for Windows (with its default `core.protectNTFS=true`) refuses to write the path. `git clone` aborted mid-checkout and left an index showing every file as deleted, and `git reset --hard main` failed with `invalid path`. The file had been in the tree since `f128d1a5` (#922). Nothing noticed, because every CI job runs on macOS or Linux. This PR renames the file and the Rust module to `auxiliary`, and adds `scripts/ci/check_windows_reserved_paths.py` as a CI job and a `make` target, so a reserved name fails on any platform. The Rust change is a pure rename with no behavior change. Review and security found no CRITICAL, HIGH, or MEDIUM issue. Three LOW findings were fixed during review.

## 1. Problem Statement

Windows reserves `CON`, `PRN`, `AUX`, `NUL`, `COM0`-`COM9`, `LPT0`-`LPT9`, and `COM`/`LPT` followed by a superscript `¹`, `²`, or `³`. The rule applies to the part of a name before its first dot, so `aux.rs`, `Aux.md`, and `aux.tar.gz` are all the AUX device. macOS and Linux accept these names without complaint. A single such path anywhere in the tree breaks the whole checkout on Windows, not just one crate. The only workaround is a sparse checkout that leaves `mlxcel-xla` unbuildable.

## 2. Change Summary

| Area | Change |
|---|---|
| `src/lib/mlxcel-xla/src/aux.rs` → `auxiliary.rs` | `git mv`, content unchanged |
| `src/lib/mlxcel-xla/src/lib.rs` and 10 sibling modules | `mod auxiliary;`, every `crate::aux::` import rewritten, rustfmt import reorder |
| `src/loading/vlm_gemma_xla_tests.rs` | `include_str!` path follows the file |
| `scripts/ci/check_windows_reserved_paths.py` (new) | Fails on any tracked path component with a reserved stem |
| `scripts/ci/check_windows_reserved_paths_test.sh` (new) | 16 throwaway-repo cases plus the real tree |
| `.github/workflows/ci.yml` | `windows-reserved-paths` job after `binary-assets`, outside the `changes` filter |
| `Makefile` | `make verify-windows-paths`, listed under Maintenance Targets |

## 3. Technical Decisions

### 3.1 Rename the module, not just the file

A `#[path = "auxiliary.rs"] mod aux;` shim would have fixed the checkout with a one-line diff, but it keeps `aux` as the module name, and any later refactor that maps modules back to files would recreate the bad path. The PR renames the module, so the 12 `crate::aux::` call sites change and the name stops existing. The `aux_*.rs` siblings (`aux_manifest.rs`, `aux_smoke.rs`) keep their names: their stem is `aux_manifest`, which is not reserved.

`TECHNICAL_REPORTS/1381-openxla-warning-backlog-20260822.*` still mentions `aux.rs`. Reports are dated records and were left as they are.

### 3.2 The matching rule

For each component of each `git ls-files -z` path, the checker takes the text before the first `.`, strips trailing spaces, casefolds it, and looks it up in a fixed set. Each part of the rule exists for a reason:

- **First dot, not last.** Windows reserves `aux.tar.gz` too. Splitting on the last dot would read the stem as `aux.tar` and miss it.
- **Every component, not only the basename.** A directory named `nul` breaks every file under it (`a/nul/b.txt`).
- **Trailing spaces stripped.** Windows trims them, so `aux .txt` opens the AUX device. This was found in the security pass and added with a test case.
- **Exact stem match.** `aux_manifest.rs`, `console.rs`, `nullable.rs`, `com10.rs`, and `data.aux` must pass, and the self-test asserts each one.
- **UTF-8 decoding is explicit.** The superscript digits are non-ASCII, so matching them should not depend on the runner's locale. A path that is not valid UTF-8 raises and fails the job instead of being skipped.

`COM0` and `LPT0` are in the set as the issue specifies, although some Windows versions may not reserve them. That can only produce a false alarm on a name nobody is likely to use, never a miss.

### 3.3 CI placement

The job copies the shape of `binary-assets`: `ubuntu-latest`, `contents: read`, `actions/checkout@v7` with `persist-credentials: false`, no secrets, and no `${{ }}` expressions inside `run:`. It is outside the `changes` filter because it needs no toolchain, runs in seconds, and the defect is silent on every platform CI uses. Gating it on Rust changes would let a doc or fixture named `con.md` through. A Windows runner that clones the repo would also catch this, but the workflow has no Windows job, and adding one for a checkout test would cost minutes per run against seconds for the static check.

### 3.4 Make target name

The target is `verify-windows-paths`, not `verify-windows-reserved-paths`. `make help` sorts targets into sections by grepping their names and descriptions, and the Run Targets filter is `(run|serve|generate)`. "reserved" contains "serve", so the longer name showed up under Run Targets as well as Maintenance. The description avoids the word for the same reason. Like `verify-binary-assets`, it is not part of the aggregate `make verify`.

### 3.5 Review findings

| Severity | Source | Finding | Fix |
|---|---|---|---|
| LOW | review | Failure text and CI comment listed COM0-9 and LPT0-9 but not the superscript names the rule rejects | `975047a2` |
| LOW | review | `git ls-files` output decoded with the locale encoding | `975047a2`: `encoding="utf-8"` |
| LOW | security | `aux .txt` passed because the stem kept its trailing space | `d3eae581`: `rstrip(" ")` and a test case |
| (format) | finalizer | rustfmt reordered imports after `aux` became `auxiliary` | `23ff2ac4` |

## 4. Verification

| Check | Result |
|---|---|
| `cargo check --all-targets` (root crate) | Passed |
| `cargo check -p mlxcel-xla --all-targets`, `cargo fmt -p mlxcel-xla -- --check` | Passed |
| `cargo test -p mlxcel-xla` | 363 passed, 0 failed, 19 ignored |
| `check_windows_reserved_paths.py` on this branch | OK |
| Same checker against the tree of `origin/main` | Fails, names `src/lib/mlxcel-xla/src/aux.rs` |
| `check_windows_reserved_paths_test.sh` | 16 cases plus the real tree, all passed |
| `git grep -nE '\baux::\|\baux\.rs' -- ':!TECHNICAL_REPORTS'` | No matches |

## 5. Unverified and Open Items

- **Not built locally:** `diagnostics_thread_overrides_stay_out_of_production_iree_paths` in `vlm_gemma_xla_tests.rs` is gated on `xla-iree` and needs `IREE_DIST`, which this host does not have. The `include_str!` target was checked by hand. The `xla-compile` CI job runs `cargo check --features cuda,xla-iree --all-targets`, which compiles it.
- **Not tested on Windows:** the acceptance step "`git clone` on Windows completes and `git status` is clean" needs a Windows machine.
- **Out of scope:** other names Windows cannot create pass the checker: the characters `<>:"|?*`, backslashes, control characters, components ending in a dot or space, and paths that differ only by case. None exist in the tree today. If one appears, the checker is the place to add it.

## 6. Learning Points

- A platform restriction that no CI runner has goes unnoticed until someone on that platform tries to use the repository. A static check over `git ls-files` costs seconds and covers the platform without a runner for it.
- Windows device names are matched on the stem before the first dot, in any case, with trailing spaces trimmed, and in every path component. A check that looks only at the basename's last extension misses most of them.
- When `make help` groups targets by substring, check the target name against every section's filter. A word like "reserved" lands in the wrong section.
