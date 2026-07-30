# Gumbel-max sampling: Apple M1 Ultra, 2026-07-31

Validation run for issue #900 (epic #909).

**Outcome: the fusion wins at every measured point and ships wired on.** Op-level
sampling is 1.13x to 1.40x faster across the whole vocab x batch matrix, and
end-to-end decode at batch 1 shows no regression.

## Environment

| Field | Value |
|---|---|
| Hardware | Apple M1 Ultra, 20 cores (16P + 4E), 128 GB unified memory |
| OS | macOS 26.5.2 (Darwin 25.5.0) |
| Backend | Metal |
| mlxcel | 0.4.3, branch `feature/issue-900-gumbel-max-sampling`, base `be7afe26` |
| MLX pin | `b7c3dd6d27f45b5365b08a840310187dc503f1db` |
| Op-level harness | `examples/gumbel_sampling_microbench.rs`, warmup 30, iters 200 |
| End-to-end model | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (276 MiB) |
| Load average | 4.0 to 5.9 during the sweep, from unrelated concurrent work |

CUDA was not available. The CUDA JIT string is written but has never been
compiled or executed.

## Op-level: sampling step only

Baseline is the existing `categorical` path (temperature scale, then
`mlx::core::random::categorical`, which normalizes over the vocabulary). Gumbel
is the new kernel: add Gumbel noise to `logits / temperature` and take an
index-carrying argmax, with no normalization and no sort.

| vocab | batch | splits | baseline us | gumbel us | speedup |
|---|---|---|---|---|---|
| 32768 | 1 | 32 | 523.62 | 463.92 | **1.13x** |
| 32768 | 4 | 16 | 514.54 | 437.42 | **1.18x** |
| 32768 | 8 | 8 | 581.29 | 431.08 | **1.35x** |
| 65536 | 1 | 64 | 571.46 | 472.83 | **1.21x** |
| 65536 | 4 | 16 | 522.25 | 429.54 | **1.22x** |
| 65536 | 8 | 8 | 621.12 | 474.38 | **1.31x** |
| 152064 | 1 | 64 | 532.96 | 472.67 | **1.13x** |
| 152064 | 4 | 16 | 588.92 | 429.50 | **1.37x** |
| 152064 | 8 | 8 | 687.92 | 491.79 | **1.40x** |

The shape of the result matters more than the magnitude. Baseline cost climbs
with both vocabulary and batch, from 523.62us to 687.92us, because it does
normalization work proportional to the vocabulary for every row. The Gumbel path
stays essentially flat, 429.50us to 491.79us across a 4.6x range of vocabulary
and an 8x range of batch, because one index-carrying max reduction does not care
how large the vocabulary is beyond the read itself. That is precisely the
property the issue set out to obtain, and it is why the advantage grows with
batch: 1.13x at batch 1 and 152K vocab, 1.40x at batch 8.

**Below the issue's prediction.** Issue #900 anticipated ">= 2x at batch >= 4".
The measured range at batch 4 and above is 1.18x to 1.40x. The gain is real,
consistent, and monotone in the right variables, but it is not 2x on this
hardware. Recorded as measured rather than reconciled to the estimate.

## End-to-end decode, batch 1

`mlxcel generate -n 256 -t 0.8 --seed 4242` on Qwen2.5-0.5B-Instruct-4bit, three
repetitions per condition.

| Condition | Runs (tok/s) | Median |
|---|---|---|
| Gumbel (default) | 319.60, 363.50, 361.79 | 361.79 |
| `MLXCEL_SAMPLING_GUMBEL=0` | 353.56, 353.31, 354.31 | 353.56 |

**+2.3%**, so the batch-1 no-regression requirement is met with a small gain. The
first Gumbel run (319.60) is cold-start model load and is excluded from the
median; the two steady-state runs agree to 0.5%. The baseline condition is
unusually tight, spanning 0.3%, which makes the comparison trustworthy despite
the host's background load.

The modest end-to-end figure is expected: at batch 1 on a 0.5B model, sampling is
a small fraction of a decode step, so a 1.13x sampling speedup cannot move total
throughput far. The op-level table is where the kernel's effect is visible.

## Not measured

- **End-to-end at batch 4.** `mlxcel generate` has no batch flag, so this needs
  `mlxcel-server --parallel 4` with concurrent clients. The issue's "measurable
  improvement at batch 4" outcome is therefore still open. The op-level table
  predicts it should be larger than the batch-1 figure, since the batch-wide
  launch collapses the per-row Rust loop into one dispatch, but that is a
  prediction and not a measurement.
- **CUDA, entirely.** Never compiled, never run.
- Larger models. Only the 0.5B checkpoint was exercised end to end.

## Correctness context

The performance result is not the interesting part of this issue's validation.
The kernel initially computed its uniform as `((word >> 8) + 0.5f) * 2^-24`,
which rounds to exactly `1.0` at the maximum input, giving
`-log(-log(1.0)) = +inf` and making that element win the argmax regardless of its
logit. That is roughly one uniformly-random token every 110 decode steps at a
152K vocabulary, and on a masked token it would have produced `NaN`, whose
behaviour in a max reduction is implementation-defined. The chi-square suite
passed with the bug present; it was caught by a dedicated test that separates
logits by 100 and fails on a single wrong token. Taking 23 bits instead of 24
fixes it.

Chi-square goodness-of-fit runs 1e6 samples per shape (peaked, flat, bimodal,
`-inf`-masked, plus vocab 4096 for the strided path) at a 1e-6 upper-tail
threshold, with temperature 0.5 and 1.5 covered separately.
