# B=1 MTP gate re-measurement on M3 Ultra, 2026-08-20

The static per-hardware gate for the singleton MTP burst, `mtp_b1_default` in
`src/server/batch/speculative_burst.rs`, ran batch-capable targets only where
`has_neural_accelerator` held, which is M5 and nothing else. That policy rested
on two numbers taken in the founding measurement (#165): about 1.2 to 1.4x on
M5 Max, against a 0.75 to 0.96x regression on M1 Ultra. Both predate #1194,
#1199, #1203, #1208 and #1215, and M3 Ultra had never been run on the pairing
at all. Issue #1217 asked what the numbers say now.

Headline: the gate was wrong about M3 Ultra by a wide margin. The batch-capable
Gemma 4 31B target with its bf16 assistant measures **1.95x to 2.65x** there, on
a host the gate declined, and it beats the M5 Max figure the gate was built to
enable. A verify round costs 1.51 classic decode steps on this host and the
rounds emit 2.96 to 3.99 tokens, so the pairing clears break-even by roughly
double on every prompt.

The discriminator is not the Neural Accelerator. It is the `use_qmv_wide` split
documented in `src/models/speculative_exactness.rs`: from Apple GPU generation
15 an affine-quantized projection at `M >= 2` runs as one wide pass, and below
it the verify block runs as `K` narrow passes whose cost grows with the block.
M3 Ultra is generation 15 and the old predicate grouped it with generation 13
anyway.

## Environment

| Field | Value |
|---|---|
| Host | Mac Studio, Apple M3 Ultra, 512 GB unified memory, macOS 26.6.1 (25G76) |
| Build | `cargo build --release --features metal,accelerate` |
| Branch | `update/issue-1217-mtp-b1-gate` at `9e2c6675` (`main` tip, unmodified for the measurement) |
| Harness | `scripts/bench_speculative.sh gemma31b`, six samples per arm, ABBA blocks, two warm-ups discarded |
| Sweep | `scripts/bench_block_width.sh gemma31b`, 8 interleaved rounds, rotating start width |
| Wrapper | `scripts/with_indexers_paused.sh` (17 indexers suspended for the duration) |
| Target | `models/gemma-4-31b-it-4bit` (`model_type: gemma4`, `supports_batching() == true`) |
| Drafter | `models/gemma-4-31b-it-assistant-bf16` |
| Sampling | `--temp 0` |

Neither pairing existed in the bench scripts before this run. `bench_speculative.sh`
covered the 12B and Qwen pairings only, and `bench_block_width.sh` the same two,
so the pairing the gate actually governs had no path through the #1215 protocol.
Both scripts gained a `gemma31b` case as part of this work; that absence is most
of why the founding numbers went stale without anyone noticing.

## Results

Gemma 4 31B + bf16 assistant, block 4, greedy. Round cost is the identity
`emitted per verify / speedup`, in units of this host's own classic decode step,
which is the same quantity the cross-host table in `docs/benchmarks.md` uses.

| Output | Tokens | Block | acceptance | emitted/verify | classic | MTP | speedup | round cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| enumeration | 400 | 4 | 1.000 | 3.990 | 31.5 | 83.6 | **2.65x** | 1.51 |
| source code | 300 | 4 | 0.882 | 3.646 | 31.8 | 76.6 | **2.41x** | 1.51 |
| prose | 400 | 4 | 0.656 | 2.956 | 31.7 | 61.9 | **1.95x** | 1.52 |

Spreads were 0.5% to 1.0% on both arms of all three rows, against the harness
limit of 4%. The contention watch flagged brief spikes in a handful of samples
and nothing sustained enough to report.

Three prompts with nothing in common agree on the round cost to within 0.7%,
which is the control that this is a property of the host and the block width
rather than of the prompt. It also lands on the 1.50 to 1.51 that the 12B
pairing measures at block 4 on this same host. Read that coincidence carefully:
the width sweep below shows the two pairings have different slopes and merely
cross near block 4, so it is not evidence that the drafter's dtype stopped
mattering.

## Block-width sweep

Source-code prompt, 300 tokens, 8 interleaved rounds with a rotating start
width, against the same classic arm of 31.76 tok/s.

| width | decode tok/s | spread | acceptance | emitted per verify | speedup | round cost | vs peak |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 53.3 | 0.7% | 0.942 | 1.942 | 1.68x | 1.16 | -31.2% |
| 3 | 67.3 | 1.2% | 0.897 | 2.794 | 2.12x | 1.32 | -13.0% |
| 4 | 76.6 | 0.9% | 0.882 | 3.646 | 2.41x | 1.51 | -1.0% |
| 5 | 77.4 | 1.7% | 0.873 | 4.096 | 2.44x | 1.68 | **peak** |
| 6 | 75.3 | 1.1% | 0.870 | 4.530 | 2.37x | 1.91 | -2.7% |
| 8 | 73.8 | 0.5% | 0.800 | 4.983 | 2.32x | 2.14 | -4.6% |

The peak is at 5 rather than the drafter's declared 4, but only by 1.0% against
a 1.7% spread on that row, so the two are tied and the declared width stands as
a default. Everything from 3 to 8 gains by more than a factor of two, so no
width in this range is a bad choice; only width 2 gives up real throughput, and
it does so by starving the round of positions to amortise over rather than by
losing acceptance, which is highest there.

Acceptance is nearly flat from width 2 to 6 (0.942 down to 0.870) and only
falls off at 8. That is the bf16 assistant staying accurate deep into a block,
and it is why emitted per verify keeps climbing across the whole table while
throughput turns over at 5: the round is emitting more each time, and past 5 the
verify is simply costing more than the extra emission is worth.

Fitting `cost = a + b K` over the six widths gives:

```
round cost = 0.83 + 0.170 K classic steps    (largest residual 0.064)
```

Beside the two published fits this is the useful comparison, and it corrects a
reading the three-prompt table on its own invites:

| Pairing | Host | fit | cost at K=4 |
| --- | --- | --- | ---: |
| Gemma 4 31B + **bf16** assistant | M3 Ultra | `0.83 + 0.170 K` | 1.51 |
| Gemma 4 12B + 4-bit assistant | M3 Ultra | `1.14 + 0.090 K` | 1.50 |
| Gemma 4 12B + 4-bit assistant | M1 Ultra | `1.35 + 0.346 K` | 2.73 |

The 31B and 12B pairings on this host agree at block 4 to within 0.01 classic
steps, which looks like the drafter's dtype not mattering. It is not: the
per-position slope is 1.9 times steeper for the bf16 drafter, and the two lines
merely cross near K = 4. Either side of that they separate, the 31B pairing
being cheaper per round below the crossing and dearer above it. Anyone
transferring a round cost between these two pairings at some other width will
be wrong in a direction that depends on which side they are on.

## What this does and does not settle

It settles the batch-capable gate for generation 15. The pairing gains 1.95x to
2.65x on M3 Ultra with every row inside a 1.0% spread, so `mtp_b1_default` now
reads `wide_quantized_projections` (Apple GPU generation 15 and up) in place of
`has_neural_accelerator` (M5 only). M4 is grouped with M3 by the shared
`use_qmv_wide` dispatch rather than by measurement, which is an inference and is
labelled as one in the code.

It does not settle generation 13, and the sweep is what makes that worth saying
carefully. The naive move is to take M1 Ultra's published block-4 round cost of
2.71, note that this pairing emits 2.96 to 3.99 tokens per verify, and conclude
that post-#1203 M1 Ultra would now clear break-even. The sweep shows why that is
not sound: 2.71 belongs to the 12B pairing with a 4-bit drafter, and this
pairing's slope is 1.9 times steeper. Carrying that ratio onto generation 13's
`1.35 + 0.346 K` puts a block-4 round near 3.6 classic steps there, which 2.96
to 3.99 emitted tokens would only just cover, and which is consistent with the
0.75 to 0.96x the founding measurement recorded. So the founding decline for
generation 13 reads as sound rather than merely stale, and it stays.

That estimate extrapolates across a pairing and a generation at once, so it is
a reason to leave generation 13 alone, not a result. Measuring it needs an
M1 Ultra host with these two checkpoints, which is tracked separately; the
prediction above is stated in a falsifiable form so that run can settle it.

## Gate exercised through the real dispatch path

Unit tests cover `mtp_b1_default` as a pure function, which is not the same as
showing that the running scheduler consults it. This is the same host, the same
two checkpoints, and the server rather than the offline CLI, with
`MLXCEL_MTP_ADAPTIVE=0` so the static gate is what decides and no
`MLXCEL_ENABLE_MTP_B1` unless stated. One `/v1/completions` request each.

| Binary | `MLXCEL_ENABLE_MTP_B1` | Scheduler outcome |
|---|---|---|
| `main` at `9e2c6675`, before the change | unset | declined, fell back to classic decode |
| this branch | unset | ran the B=1 burst |
| this branch | `0` | declined, fell back to classic decode |

The declines are the scheduler's own log line, `MTP B=1 speculative burst
declined for seq seq-0 ... falling back to classic decode`. The run that
proceeded reports block 4 at effective block 4, 80 tokens over 21 rounds,
acceptance 0.921, 3.762 tokens emitted per verify, and no decline line anywhere
in its log.

That is the before-and-after the gate change is for: an M3 Ultra that declined
this pairing now runs it, and the override still pins the decision in both
directions.

The offline `mlxcel generate` path does **not** exercise this gate, which is
worth recording because it changes how the throughput rows above should be
read. `mtp_b1_default` has exactly one caller, `Scheduler::mtp_b1_should_run`,
and `MtpPolicy` is built only in the server worker; the offline path gates on
exactness alone (`run_offline_mtp`). So the bench harness runs the burst
unconditionally on every host, which is what makes it the right instrument for
deciding the gate, and it is also why `MLXCEL_ENABLE_MTP_B1=1` and
`MLXCEL_MTP_ADAPTIVE=0` are inert there. They are set in the harness so the arm
matches what a server runs with the gate forced on, not because the harness
needs them.

## Not measured here

- **M1 Ultra, this pairing.** No generation-13 host was available. Nothing in
  this record changes M1 or M2 behaviour.
- **M5 Max, re-measured.** Its ~1.2 to 1.4x is the founding figure and is also
  pre-#1203. M5 keeps the path it already had either way, so the gate does not
  depend on it, but the number in the table is old and the M3 Ultra rows above
  now exceed it, which is itself a reason to re-run it.
- **Qwen MTP pairing on M3 Ultra.** Already measured on current main under this
  protocol in #1215 and recorded in `docs/benchmarks.md` (1.67x at block 3, full
  width sweep). Not re-run here.
