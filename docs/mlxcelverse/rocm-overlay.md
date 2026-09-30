# Maintaining the ROCm overlay

The ROCm part of mlxcelverse (`src/lib/mlx-cpp/patches-rocm/`, described in its [README](../../src/lib/mlx-cpp/patches-rocm/README.md)) has two moving upstreams: the MLX pin (ml-explore/mlx, bumped regularly) and the fork its backend is vendored from (NripeshN/mlx `rocm-support`). This page is the procedure for following both, the tools that support it, and how fixes go back to the fork. Tooling and procedure come from lablup/mlxcel#1813.

## The records

Four files next to the overlay are records, never copied into MLX:

| File | What it holds |
|---|---|
| `UPSTREAM` | The fork commit (`commit`), the fork's merge base with upstream MLX (`fork_upstream_merge_base`), and the MLX pin the core files were retargeted to (`retargeted_to_mlx_pin`). Every tool starts from these three commits. |
| `LOCAL_FIXES.md` | Every change the overlay carries relative to the fork commit, numbered, with the reason. Each backend file that differs from the fork must be named in an entry (the drift check enforces it). |
| `CORE_RESIDUAL.diff` | Generated. For each of the MLX core files, what the overlay carries beyond "MLX pin + the fork's own change to that file", with a note naming the LOCAL_FIXES item that explains it. |
| `README.md` | Layout, file counts, provenance. |

The overlay itself is `CMakeLists.txt` plus `mlx/`: the backend under `mlx/backend/rocm/` and the MLX core files it hooks.

## The tools

All under `scripts/mlxcelverse/`. The engine is `rocm_overlay.py`; `sync_from_fork.sh` and `check_api_drift.sh` are the entry points the issue named.

| Command | Network | What it does | Where it runs |
|---|---|---|---|
| `rocm_overlay.py records` | no | Checks the records against each other and the tree: `UPSTREAM` is well formed and names the build's MLX pin, the README counts are the real ones, `LOCAL_FIXES.md` is numbered without gaps, `CORE_RESIDUAL.diff` was generated for the current commits and current core-file contents and every entry has a note. | `make verify-rocm-overlay`, part of `make verify` and `make verify-rocm` |
| `rocm_overlay_test.sh` | no | The tool's own coverage on a synthetic fork and pin history, including the negative cases. | same target |
| `rocm_overlay.py drift` | fetches once | Rebuilds each core file as "pin + fork delta" and diffs the overlay against it; the result must match `CORE_RESIDUAL.diff`. Also requires every backend file that differs from the fork, and every fork hook outside the overlay, to be named in `LOCAL_FIXES.md`. `--write` rewrites the record, keeping its notes. | manual, after any core-file change, pin bump or fork sync |
| `rocm_overlay.py retarget [--to <sha>]` | fetches once | 3-way merges each core file from the old pin to the new one (`git merge-file overlay old-pin new-pin`) and updates `retargeted_to_mlx_pin`. Default target: the build's pin. | manual, pin bump |
| `sync_from_fork.sh [<commit>] [--check \| --out DIR]` | fetches once | Moves the overlay to a new fork commit (details below) and updates `UPSTREAM`. `--check` syncs into a scratch copy and requires it to equal the committed overlay byte for byte. | manual, fork sync |
| `check_api_drift.sh [--mlx-commit <sha>] [--rocm-from overlay\|fork:<sha>]` | fetches, builds | Builds MLX at a candidate commit with the ROCm files over it, `make -k`, and lists every compile error and every `mlx::core` symbol that is referenced but defined nowhere (a primitive without a ROCm `eval_gpu` only shows up that way, because libmlx is a static archive). | manual, AMD host with ROCm |

Git objects come from a cache repository, `~/.cache/mlxcel/mlxcelverse/mlx.git` by default (`--git-dir` or `MLXCELVERSE_GIT_DIR` to use another clone; `--no-fetch` to stay offline). It holds both remotes and is filled on first use (a full MLX history, under 100 MB). The tools only fetch.

Only `records` and the test are in the Makefile gates, because only they are offline and fast (about two seconds). `records` is in `make verify` as well as `make verify-rocm` because pin bumps are usually made on a Mac: a bump that skips the ROCm retarget now fails there, on any host.

### Why a residual record rather than line counts

The overlay's core files are whole-file copies: upstream at the pin with the fork's hooks merged in. The earlier bump check compared each overlay's +/- line counts against the old and the new upstream base. That confirms upstream's changes came in and ours stayed, but it cannot see an overlay that was already wrong before the bump; a Metal overlay once carried half of an upstream fix for months until a reviewer compared it line by line (TECHNICAL_REPORTS/1772, section 3).

`drift` makes the line-by-line comparison mechanical. It rebuilds the file the overlay should be, the pin with the fork's own change to that file merged in (a 3-way merge with the fork's merge base as the base, conflict markers kept), and diffs the overlay against it. What is left is exactly what the overlay adds on its own: conflict resolutions and local fixes. At `81ba1c6a`/`75915908` that is 4 of the 15 core files, all explained by LOCAL_FIXES items 1 and 24; the other 11 reproduce byte for byte. A half-applied upstream change shows up as a new hunk that deletes the upstream line, and it has to be reviewed and noted before `records` passes again.

## Bumping the MLX pin: the ROCm part

The general procedure is "Bumping the MLX upstream pin" in `CONTRIBUTING.md`. For the ROCm overlay:

1. Edit `GIT_TAG` as described there. From this point `make verify-rocm-overlay` fails until the overlay is retargeted.
2. `python3 scripts/mlxcelverse/rocm_overlay.py retarget`. It 3-way merges each core file onto the new pin and records the new pin in `UPSTREAM`. Resolve any conflicts it reports (they are left in the files with markers).
3. On an AMD host: `scripts/mlxcelverse/check_api_drift.sh`. Upstream API changes show up as compile errors in `mlx/backend/rocm/`, new primitives as undefined `eval_gpu` symbols; give those a `NO_GPU` stub in `mlx/backend/rocm/primitives.cpp` or an implementation. Re-run until the report is empty. A plain `cargo build --features rocm` finds the same things one error at a time.
4. `python3 scripts/mlxcelverse/rocm_overlay.py drift`. Review every hunk of the residual change it prints. A hunk that only moved is fine; a hunk that drops or alters an upstream line needs a reason. Then `drift --write`, and add or update the `# note` lines in `CORE_RESIDUAL.diff`.
5. Record every fix in `LOCAL_FIXES.md` (retarget fixes under "Retarget onto the MLX pin"), naming the files touched.
6. `make verify-rocm` on the AMD host: it runs `verify-rocm-overlay`, clippy, the `verify-rocm-smoke` generation smoke (lablup/mlxcel#1811) and the ROCm test gate.
7. In the PR body, add a "ROCm overlay" section (below). If the bump gets a technical report, give it the same section.

## Syncing to a newer fork commit

1. See what moved on the fork since the `UPSTREAM` commit, on GitHub or in the cache (`git -C ~/.cache/mlxcel/mlxcelverse/mlx.git log --oneline <UPSTREAM commit>..fork/rocm-support` once a run without a commit argument has fetched the branch).
2. `scripts/mlxcelverse/sync_from_fork.sh <commit> --out /tmp/rocm-sync` for a dry run, then without `--out` to sync in place. What it does:
   - Backend files (`mlx/backend/rocm/`): 3-way merge of old fork commit, overlay and new fork commit. The difference between the overlay and the old fork commit is exactly the local fix list, so every LOCAL_FIXES entry is carried without replaying it by hand. Local-only files (`hadamard.hip`, `fft.hip`) are kept; new fork files are added; a file the fork deleted is deleted only if the overlay never changed it.
   - Core files: 3-way merge from "pin + old fork delta" to "pin + new fork delta". The fork's diff against its own merge base is applied on the pin, so upstream changes the fork picked up by merging MLX are not mistaken for fork changes.
   - It refuses when the new fork commit has merged MLX newer than the pin (retarget first, or pass `--allow-fork-ahead`), reports fork changes to MLX files the overlay does not carry, and updates `commit` and `fork_upstream_merge_base` in `UPSTREAM`.
   - Conflicts are left in the files with markers and make the exit status 1.
3. Resolve conflicts. Where the fork changed code a local fix also changed, decide whether the fork now fixes the problem (then drop the LOCAL_FIXES entry and its upstream package) or the fix still applies.
4. `check_api_drift.sh` (or a `--features rocm` build), then `drift`, review, `drift --write`.
5. Update `LOCAL_FIXES.md` (its title names the fork commit) and `docs/mlxcelverse/upstream/README.md`.
6. `make verify-rocm`, and the same "ROCm overlay" section in the PR body.

`sync_from_fork.sh --check` against the recorded commit reproduces the committed overlay byte for byte. That holds by construction, because the local fixes are carried as the overlay's own difference from the fork rather than replayed from a separate patch list, so it confirms only that a sync to an unchanged fork commit is a no-op; it cannot detect an unrecorded local change. That is `drift`'s job: every backend file that differs from the fork must be named in `LOCAL_FIXES.md`, and every core-file residual must be recorded in `CORE_RESIDUAL.diff`. `--check` without a commit compares against the head of the fork branch and fails once the fork has moved past `UPSTREAM`.

## The "ROCm overlay" section of a bump or sync PR

Put this in the PR body, and in the technical report when there is one (as section 3 of `TECHNICAL_REPORTS/1772-mlx-pin-mxfp8-round-up-20260911.en.md` does for the Metal and CUDA overlays):

```markdown
## ROCm overlay

- Pin: <old> -> <new> (retarget) / Fork: <old> -> <new> (sync)
- Core files merged: <n> of 15, conflicts in <files, or none>
- Residual (`CORE_RESIDUAL.diff`): unchanged / changed: <each hunk and the LOCAL_FIXES item behind it>
- API drift (`check_api_drift.sh`): <compile errors and undefined symbols found, and the fix for each, with LOCAL_FIXES item numbers>
- `make verify-rocm` on <gfx target>: <result, including known baseline failures>
```

## Sending fixes back to the fork

Fixes that apply to the fork itself (LOCAL_FIXES entries marked "Applies to the fork") are prepared as PR packages under [`docs/mlxcelverse/upstream/`](upstream/README.md): a `git format-patch` against the fork head, a PR title and body, and a reproduction. They are submitted by hand by the maintainer; no tool here pushes to, forks, or opens anything on NripeshN/mlx or ml-explore/mlx. The index there lists each package, its LOCAL_FIXES items, whether it applies cleanly, and its link once submitted, and summarizes the fork's contribution rules and ml-explore/mlx's AI usage policy.
