# MTP verify early-exit walk: decided by measurement, not built (issue #1179)

Issue #1179 asked whether a device-side early-exit verify walk beats the
batched full-logits verifier, with a stated prior: the LM head is
weight-read-bound, so skipping tail positions should save almost nothing.
The measurement below answers no, the walk should not be built, but for a
sharper reason than the prior: the projection cost is *not* flat in the
projected width, and the very effect that makes tail positions expensive
also makes a sequential walk pay the full weight read once per walked
position, which loses to the batched projection outright at every
production block size.

## Environment

| Field | Value |
|---|---|
| Host | Apple M5 Max, 128 GB unified memory, macOS 26.6.1 |
| Build | `cargo build --release --features metal,accelerate`, branch `perf/issue-1179-verify-early-exit-decision` (main at `77c71402` plus this branch's cleanup) |
| Harness (a) | `examples/mtp_projection_width_bench` 16 5: T=16 projections per batch folded into one eval, best of 5 rounds, weights round-robined over 2 copies (566 to 715 MB each) so every read hits DRAM |
| Harness (b) | offline CLI, 300 tokens, temperature 0, `MLXCEL_MTP_BLOCK_CONTROLLER=requested` (block width pinned, #1207), `MLXCEL_MTP_ALLOW_INEXACT=1` (widths above the narrow batch limit cannot pass the exactness probe; this sweep measures cost, not output), 15 s cooldowns, Time Machine stopped and verified stopped at both ends, indexers paused |
| Guard | each width row asserts and prints its logits shape `[1, W, vocab]`; the two arms of the question differ by that shape and nothing else |

## (a) LM-head projection cost against projected width

Projection + argmax at the real head shapes, affine 4-bit group 64.

Default kernel selection (`qmv_wide` on, what a non-MTP process runs):

| W | gemma4-12b tied head (262144 x 3840) | GB/s | qwen3.8-27b lm_head (248320 x 5120) | GB/s |
|---:|---:|---:|---:|---:|
| 1 | 0.985 ms | 575 | 1.234 ms | 580 |
| 2 | 1.020 ms | 555 | 1.301 ms | 550 |
| 3 | 1.149 ms | 493 | 1.419 ms | 504 |
| 4 | 1.352 ms | 419 | 1.903 ms | 376 |
| 5 | 1.846 ms | 307 | 2.246 ms | 318 |
| 8 | 3.406 ms | 166 | 4.299 ms | 166 |
| 16 | 2.561 ms | 221 | 3.266 ms | 219 |
| 32 | 2.452 ms | 231 | 3.111 ms | 230 |

Narrow kernel selection (`MLXCEL_QMV_WIDE=0`, what the MTP verify path
actually runs on generation 15+ after the #1199 exactness retry):

| W | gemma4-12b tied head | GB/s | qwen3.8-27b lm_head | GB/s |
|---:|---:|---:|---:|---:|
| 1 | 0.985 ms | 575 | 1.225 ms | 584 |
| 2 | 1.243 ms | 456 | 1.453 ms | 492 |
| 3 | 1.811 ms | 313 | 1.960 ms | 365 |
| 4 | 2.331 ms | 243 | 2.849 ms | 251 |
| 5 | 3.141 ms | 180 | 3.353 ms | 213 |
| 8 | 4.510 ms | 126 | 5.030 ms | 142 |
| 16 | 2.559 ms | 221 | 3.256 ms | 220 |
| 32 | 2.450 ms | 231 | 3.110 ms | 230 |

W = 1 is identical in both selections (`M = 1` takes `qmv` regardless),
and so are W = 16 and 32 (the matrix-matrix kernel ignores the flag). The
`M` in between is where the narrow pin costs, which is the #1261/#1278
collateral measured at the head shape.

Two structural facts, visible in both selections:

- The curve is not flat. W = 1 runs at the memory-bandwidth roof
  (575 to 580 GB/s); the `M >= 2` kernels degrade steadily to about
  166 GB/s at W = 8; the matrix-matrix kernel takes over at the batch
  limit and W = 16 and 32 cost *less* than W = 8. The issue's prior
  (projection cost invariant in W, ceiling near zero) was wrong in the
  letter but right in the verdict.
- The early-exit ceiling `t(K) - t(1)` at the widths anyone runs is
  0.74 to 0.83 ms per round at K = 3 and 1.35 to 1.62 ms at K = 4 under
  the production (narrow) selection, against verify forwards of 31 to
  40 ms per round: about 2 to 4% of the verify forward, before paying
  anything for the walk itself. And the walk cannot collect it, per the
  next section.

## Why the walk loses even that ceiling

A device-side early-exit walk projects position by position and stops at
the first mismatch, so it pays `E[A] x t(1)` where `E[A]` is the expected
number of positions walked, and every one of those positions re-reads the
full weight matrix. At the measured production operating points:

`E[A]` below is `E[accepted] + 1` under a per-position independence
approximation of the measured acceptance rate, capped at K. `t(K)` is the
narrow-selection column, because that is what the MTP verify path runs.

| pairing | K | acceptance | E[A] | walk cost `E[A] x t(1)` | batched `t(K)` | walk wins? |
|---|---:|---:|---:|---:|---:|---|
| gemma4-12b | 3 | 0.876 | 2.64 | 2.60 ms | 1.81 ms | no, loses 1.4x |
| gemma4-12b | 4 | 0.799 | 2.95 | 2.91 ms | 2.33 ms | no, loses 1.2x |
| gemma4-12b | 8 | 0.636 | 2.67 | 2.63 ms | 4.51 ms | ~1.9 ms, see below |
| qwen3.8-27b | 3 | 0.827 | 2.51 | 3.10 ms | 1.96 ms | no, loses 1.6x |

The one width where the walk gets ahead, K = 8, is dominated by a far
simpler lever the width curve exposes: padding the projection to the
matrix-matrix width costs 2.56 ms there, which collects the same ~1.9 ms
with a reshape instead of a sequential dispatch-compare-branch walk, and
K = 8 is not an operating point on either pairing anyway (see below). The
walk estimate also charges nothing for its own K sequential
dispatch/compare steps.

The one cell where the walk is marginally ahead (K = 8, ~0.7 ms against a
57 ms verify forward, ~1.2%) is not an operating point: the production
sweep below puts Gemma 4's throughput optimum at K = 4 and the #1207
controller now finds that optimum by measurement, and the estimate charges
the walk nothing for its K sequential dispatch/compare steps, which a real
implementation would pay.

## (b) Production verify cost against block size

Same protocol as (a)'s harness row. `verify_forward_ms / rounds`:

| K | gemma verify ms/round | gemma tok/s | qwen verify ms/round (relative) | qwen tok/s |
|---:|---:|---:|---:|---:|
| 3 | 31.2 | 83.1 | 107.9 | 22.6 |
| 4 | 35.5 | 89.1 | 136.6 | 22.6 |
| 8 | 57.0 | 85.9 | 255.5 | 15.7 |
| 16 | 61.1 | 75.2 | 301.4 | 12.5 |
| 32 | 155.4 | 28.6 | 377.5 | 8.9 |

The Gemma column is consistent with this host's clean baselines (89.15
tok/s pinned at K = 4 in the #1207 record). The Qwen block ran late in the
sweep and its absolute values carry sustained-load throttling (this host's
known failure mode; its clean K = 3 verify round is ~40 ms), so read only
its ordering: K = 3 and 4 are equivalent, everything wider loses. On both
pairings the operating range stays at K <= 8, where (a) says early exit
has nothing to win.

An observation for whoever revisits wide blocks: the W = 8 anomaly (more
expensive than W = 16) means a K = 8 verify would project cheaper padded
to the matrix-matrix width than early-exited. If wide blocks ever become
an operating point, pad-to-qmm is the lever to measure first, not the
walk.

## Decision

Not built, closed as a measured no (the outcome #1179 names as success):

- At the production widths (K = 3 to 5) the batched projection beats a
  sequential early-exit walk by 1.2 to 1.6x on the walk's own best case,
  because the walk re-reads the weight per walked position.
- The ceiling the walk chases there is 2 to 4% of the verify forward,
  and where a wide block would make it larger, pad-to-qmm collects it
  cheaper.
- The stale gate comment and the `MLXCEL_ENABLE_MTP_DEFERRED` flag, which
  gated a path that split the verify forward from the projection without
  deferring any work (identical compute, one extra bridge crossing), were
  removed with their orphaned helper chain.
