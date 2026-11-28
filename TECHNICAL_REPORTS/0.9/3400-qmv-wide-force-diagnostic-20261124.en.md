# Technical Report: PR #3400 - `MLXCEL_QMV_WIDE=force` to measure qmv_wide below generation 15

**Date**: 2026-11-24
**Status**: Open (written before merge)
**Risk Level**: Low (a new opt-in value; unset, `0` and `1` are unchanged)

## Executive Summary

Gemma 4 12B with its MTP assistant ran slower than classic decode on an M1 Ultra for prose, because a block-4 verify costs about three classic decode steps there. The question was whether MLX's `qmv_wide` kernel, which amortizes a few-row quantized matmul on generation 15 and later, would help on generation 13. Upstream excludes affine products below generation 15, and `MLXCEL_QMV_WIDE` could only switch qmv_wide off, so answering it took a patched rebuild. This PR adds `MLXCEL_QMV_WIDE=force`, which admits those products, and records the answer: on an M1 Ultra the forced kernel makes the verify slower (about 68 to 94 ms per round) and breaks byte-identity, so upstream's threshold holds there (#3398).

## 1. Problem Statement

`use_qmv_wide` in the Metal overlay of `quantized.cpp` returns `mode != "affine" || arch_gen >= 15`, with upstream's note that affine qmv_wide only beats qmv from generation 15. On M1 (generation 13) a 4-row MTP verify therefore runs plain `qmv` row by row, since `qmv_mma` is disabled on Ultra parts and mlxcel's own `mlxcel_affine_qmm_t_rows` starts at 5 rows. The threshold is an upstream tuning constant that can go stale after a pin bump, and generation 14 (M2) has not been measured here.

## 2. What Changed

- `quantized.cpp` overlay: `mlxcel_qmv_wide_forced()` reads `MLXCEL_QMV_WIDE` once and returns true for `force`; `use_qmv_wide` accepts affine below generation 15 when it is set. The existing flag already treats `force` as "on", so the exactness gate's runtime `mlxcel_set_qmv_wide` still works. The overlay header documents the value and the measurement.
- `overlays/metal/README.md` and the `src/lib/mlxcelverse/CMakeLists.txt` overlay note list it as a retained local change, which the MLX pin-bump procedure re-applies.
- `docs/environment-variables.md`: the value, its scope, the measurement, and "do not use it for serving".

## 3. Technical Decisions

**A value on the existing variable, not a new one.** `MLXCEL_QMV_WIDE` already owns this gate. A second variable would give one decision two knobs with an undefined combination (for example `MLXCEL_QMV_WIDE=0` together with a separate force flag).

**Only the qmv/qmv_wide branch.** In `dispatch_qmm_t`, `qmv_mma` and `mlxcel_affine_qmm_t_rows` are tried before the qmv branch. `force` leaves their row ranges alone, so it measures exactly the kernel upstream gates and nothing else.

**No Rust change.** `qmv_wide_pinned_by_operator()` in `src/models/speculative_exactness.rs` counts any set value as an operator pin, so under `force` the gate skips its narrow retry and declines MTP when the probe fails. That is the right default for a diagnostic: output can only change when `MLXCEL_MTP_ALLOW_INEXACT=1` is also set, which already warns loudly.

## 4. Measurements

M1 Ultra, `gemma-4-12b-it-4bit` (HF revision `73bcf09`) with `gemma-4-12b-it-assistant-bf16`, block 4, temperature 0, 400-token enumeration prompt, one run per value after a warm-up:

| `MLXCEL_QMV_WIDE` | Probe | MTP | verify ms / 100 rounds | decode tok/s |
|---|---|---|---:|---:|
| unset | pass | on | 6771 | 50.3 |
| `0` | pass | on | 6768 | 50.2 |
| `1` | pass | on | 6900 | 49.6 |
| `force` | fail (layer 0 attention) | declined | n/a | classic |
| `force` + `MLXCEL_MTP_ALLOW_INEXACT=1` | overridden | on | 9397 | 37.1 |

An earlier interleaved run with a throwaway build gave 91 ms per round under the forced kernel against 68 ms, and on a prose prompt 26 against 33 tok/s with the output diverging at character 156. Classic decode is unaffected because `M = 1` never takes `qmv_wide`.

## 5. Limitations

- Generation 14 (M2) is still unmeasured; this PR makes that a one-variable run.
- One checkpoint and block width. The probe result is per geometry, so other models may differ in where they diverge, not in whether qmv_wide is faster.
- The M1 slowdown for MTP itself is not addressed. Extending `mlxcel_affine_qmm_t_rows` down to 4 rows would be the candidate, and it needs its own measurement.
