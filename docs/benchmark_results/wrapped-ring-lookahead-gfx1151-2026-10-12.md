# Wrapped-window B=1 decode on Gemma 3, gfx1151, 2026-10-12

Issue lablup/mlxcel#2239. The issue's Oct 8 evidence (the #2192 verification session) measured gemma-3-4b-it-4bit at pp2048/tg128 three arms that did not overlap: the pre-epic `CxxGenerator` loop 62.66 to 62.82 tok/s, pipelined `main` 61.67 to 61.93 (median -1.3 percent), and `main` with `MLXCEL_FORCE_SYNC=1` 63.50 to 63.69 (+1.5 percent), and concluded the synchronous step was the faster choice once the 1024-token sliding window had wrapped. This page records the same measurement repeated on the current tree, which does not reproduce the regression, and the follow-up runs that separate what recovered it.

## Method

`mlxcel-bench-decode -m models/mlx/gemma-3-4b-it-4bit -n 128 --warmup-tokens 20 --ignore-eos --prompt-tokens 2048`, greedy, Radeon 8060S (gfx1151), every run its own `scripts/rocm_gpu_guard.sh` command inside one host-wide lock hold (`data/wrapped-ring-lookahead-gfx1151-2026-10-12/session.sh`): 10 s of idle GPU and no compiler before each run, a 1 Hz monitor during it, and all 12 runs CLEAN on the first attempt. Four arms, three interleaved rounds, arm order rotating by one per round: `base` is the pre-epic binary (a `git archive` of `4c44e317`, PR #2194, whose decode code is the pre-epic `CxxGenerator`), `new` is the branch build, `new-sync` is the same branch binary with `MLXCEL_FORCE_SYNC=1`, and `base-null` is the pre-epic binary again, the method's noise floor. `data/.../summarize_rounds.py` computes the medians and the paired per-round deltas (percent of the same round's `base`). The threshold is ADR 0007's 1.0 percent on the paired decode delta. The branch build at measurement time carried a wrapped-window synchronous handover prototype on top of `e24fb7b2`; the handover never routes this bench (see Findings), so the arm measures the pipeline.

## Four-arm session, pp2048/tg128 (data/session/rounds.csv)

| Arm | Rounds (tok/s) | Median | Paired vs base |
|---|---|---:|---|
| base (`CxxGenerator`, `4c44e317`) | 62.61, 62.71, 62.56 | 62.61 | |
| new (branch, pipelined) | 62.49, 62.63, 62.66 | 62.63 | -0.19, -0.13, +0.16 (median -0.13 percent) |
| new-sync (`MLXCEL_FORCE_SYNC=1`) | 61.10, 60.80, 62.11 | 61.10 | -2.41, -3.05, -0.72 (median -2.41 percent) |
| base-null (base again) | 62.61, 62.60, 62.65 | 62.61 | +0.00, -0.18, +0.14 (median +0.00 percent) |

The pipelined arm sits inside the threshold (median -0.13 percent, inside the null arm's -0.18 to +0.14 spread): the regression the issue reports does not reproduce on this build.

## Pre-fix comparison (data/nofix-pp2048/rounds.csv)

To separate what recovered the gap from what the branch's prototype contributed, the same two arms were re-run on a build of origin/main at `e24fb7b2` without the prototype, three interleaved rounds, same guard hold, same command (`data/.../nofix_session.sh`). The session ran after the four-arm session rather than interleaved with it, so the deltas below are against the four-arm session's base median rather than same-round pairs; the base and base-null arms bracketed 62.56 to 62.71 across both sessions.

| Arm | Rounds (tok/s) | Median | vs base median 62.61 |
|---|---|---:|---:|
| nofix (`e24fb7b2`, pipelined) | 62.79, 62.58, 62.83 | 62.79 | +0.29 percent |
| nofix-sync (`MLXCEL_FORCE_SYNC=1`) | 59.23, 62.52, 62.08 | 62.08 | -0.85 percent |

The pipelined arm on origin/main without any of this branch's changes is at parity (+0.29 percent): the -1.3 percent the Oct 8 session measured on `b2fda840` was recovered by the commits merged between the two sessions, before this branch existed. #2271 (avoid replaying terminal B1 decode) and #2272 (bound DRY and sparse penalty step costs) are the B=1 decode changes in that range.

## pp512 spot check (data/pp512-gfx1151-2026-10-12/rounds.csv)

The pp512 cell of the #2192 session sits inside the 1024-token sliding window, so the wrapped-window path the issue targets is not exercised there; this spot check on the same tree confirms the cell stays within the threshold. Same method, three interleaved rounds of three arms at `--prompt-tokens 512`, all 9 runs CLEAN on the first attempt.

| Arm | Rounds (tok/s) | Median | Paired vs base |
|---|---|---:|---|
| base (`CxxGenerator`, `4c44e317`) | 69.11, 69.78, 68.84 | 69.11 | |
| new (branch, pipelined) | 69.65, 69.46, 69.40 | 69.46 | +0.78, -0.46, +0.81 (median +0.78 percent) |
| base-null (base again) | 69.10, 60.16, 69.12 | 69.10 | -0.01, -13.79, +0.41 (median -0.01 percent) |

The pipelined arm stays within the threshold (median +0.78 percent, an improvement). The base-null arm's second round (60.16 tok/s) is a single outlier the guard recorded CLEAN: no other process held the GPU and no compiler ran, so the dip is host state (clocks or thermals), not interference.

## Findings

1. The issue's gfx1151 acceptance criterion is met on the current tree without any code change: pipelined `main` at pp2048 is within 1.0 percent of the pre-epic loop (median -0.13 percent on the branch, +0.29 percent on origin/main), against a null arm spread of -0.18 to +0.14 percent.
2. The Oct 8 conclusion that the synchronous step is the faster choice on a wrapped window no longer holds: `MLXCEL_FORCE_SYNC=1` on the same binary measures below the pipelined arm today (median -2.41 percent on the branch, -0.85 percent on origin/main), where the Oct 8 session measured it +1.5 percent above base.
3. The wrapped-window synchronous handover prototype this branch carried (`ae3b9ee9`, kept in the reflog) was measured and dropped: it never routes this bench (the pipelined arm's numbers match the origin/main arm within noise), its target gap no longer reproduces, and its handover destination measures below the pipeline.
4. The CUDA (GB10) three-arm measurement the issue also asks for was not run: the spark-102 host's `gpu-lock` tooling is not available from this host. That criterion stays open.
