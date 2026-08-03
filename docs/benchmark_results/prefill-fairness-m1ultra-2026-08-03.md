# Prefill fairness grant: the TTFT bound and what it costs

Issue #1011 / ADR 0005 cell 2. M1 Ultra (64 GB), macOS 26.6, Metal.
`mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`, `--parallel 8 --prefill-chunk-size 512`.

Issue #908 found that a chunked prefill parked next to a live decode batch never
advances: the tick policy is a pure function of scheduler state, running a decode
changes none of the state it reads, so `Decode` wins every tick until the batch
drains. #1011 adds `--prefill-grant-interval N`: after `N` consecutive decode
ticks the next tick is granted to the parked prefill.

ADR 0005 measured the starvation and deliberately left the price of fixing it
open, because that price belongs to whichever policy ships. This is that
measurement.

## Part 1: the starvation, and that the grant removes it

`scripts/bench/starvation_probe.sh`, which samples `/metrics` every 5 s while
four 400-token decode streams run and one ~10k-token prompt (19 chunks at 512)
is admitted at t≈6 s. Counters, not latencies, so the answer does not depend on
how busy the machine is. Machine load at the start of each arm: 10.0 and 7.9.

| t | grant=0 chunks | grant=0 decode | default (16) chunks | default (16) decode |
|---|---|---|---|---|
| 5 s | 1 | 162 | 3 | 140 |
| 10 s | 1 | 233 | 6 | 183 |
| 15 s | 1 | 288 | 9 | 224 |
| 20 s | 1 | 344 | 12 | 271 |
| 25 s | 3 | 399 | 14 | 316 |
| 30 s | 9 | 399 | 17 | 351 |
| 35 s | 14 | 399 | **19** | 399 |
| 40 s | **19** | 406 | 19 | 399 |

With the grant disabled the prefill holds at chunk 1 for 20 s while decode
advances 162 to 344, then completes only once decode has drained to its 399-step
ceiling, reproducing ADR 0005's cell 1 on the same hardware. With the shipped
default the prefill advances from the first sample, concurrently with decode
throughout.

**Attribution.** `mlxcel_batch_prefill_grants_total` is 0 in the disabled arm and
18 in the default arm, so the arms differ by the thing they claim to differ by.
`mlxcel_batch_mixed_steps_total` is 0 in both, so neither arm is the
`MLXCEL_MIXED_STEP` prototype: this measures the DEFAULT policy.

## Part 2: the ITL price, per interval

`scripts/bench_mixed_step_admission.py`: four streams decode, one ~8k-token
prompt (14 chunks) is admitted after a 5 s settle, and per-stream inter-token
gaps are split into a quiet window and an admission window. Five intervals, three
repeats each, interleaved (`0 4 8 16 32`, then again, then again) so machine-load
drift cannot alias onto the interval. `--expect baseline --expect-grant on|off`
on every run, so a run whose counters do not match its claimed arm prints no
table at all.

Machine load was NOT quiet for the first two rounds: an unrelated build was
running at 300%+ CPU, giving 1-minute load averages of 8.8 to 17.2. It finished
before round 3, which ran at load 3.7 to 5.7. Round 3 is therefore the
quiet-machine reference and the other two are the robustness check. The headline
quantity, inflation within a run, is a ratio of two windows of the same run, so
it survives slowly varying load; the absolute millisecond figures from rounds 1
and 2 do not, and are shown only to demonstrate that the ratios do not move.

### Round 3, quiet machine (load 3.7 to 5.7)

| interval | grants | admitted TTFT | stream ITL p50 quiet → admission | p95 quiet → admission | mean quiet → admission |
|---|---|---|---|---|---|
| 0 (disabled) | 0 | 51.5 s | 66.6 → 37.4 ms | 79.3 → 75.1 ms (0.95x) | 73.0 → 47.6 ms (0.65x) |
| 4 | 13 | 14.7 s | 55.0 → 60.1 ms | 58.1 → 954.3 ms (**16.4x**) | 60.9 → 268.4 ms (**4.4x**) |
| 8 | 13 | 17.8 s | 55.6 → 59.0 ms | 59.6 → 912.7 ms (**15.3x**) | 61.0 → 163.6 ms (**2.7x**) |
| **16 (default)** | 13 | **23.8 s** | 54.7 → 58.4 ms | 58.2 → 833.3 ms (**14.3x**) | 60.7 → 110.9 ms (**1.8x**) |
| 32 | 13 | 29.1 s | 49.5 → 51.3 ms | 50.1 → 53.0 ms (1.06x) | 55.4 → 68.2 ms (1.23x) |

### All three rounds: dispersion

| interval | TTFT median (range) | p95 inflation median (range) | mean inflation median (range) |
|---|---|---|---|
| 0 (disabled) | 50.5 s (44.6 - 51.5) | 0.95x (0.84 - 1.01) | 0.73x (0.65 - 0.96) |
| 4 | 14.7 s (14.5 - 15.4) | 13.3x (13.1 - 16.4) | 3.93x (3.70 - 4.41) |
| 8 | 17.6 s (17.5 - 17.8) | 11.8x (11.8 - 15.3) | 2.34x (2.22 - 2.68) |
| **16 (default)** | **23.8 s (23.8 - 24.1)** | 10.9x (10.7 - 14.3) | **1.60x (1.60 - 1.83)** |
| 32 | 29.9 s (29.1 - 33.5) | 0.97x (0.85 - 1.06) | 1.22x (1.07 - 1.23) |

Every grant arm recorded exactly 13 grants for 14 chunks: chunk 0 runs at
admission, and every one of the 13 continuations was a granted tick. The
disabled arm recorded 0. The TTFT spread within an interval is 0.3 s at interval
16 and under 1 s everywhere below 32, across a 4x range of machine load, which is
what a tick-counted policy should look like.

## Reading the numbers

**The model fits, so the frontier is predictable rather than empirical.** One
grant cycle is `N` decode ticks plus one chunk forward, so a `C`-chunk prompt
takes `C * (N * D + P)` and the streams' mean inter-token latency over the window
is `D + P / N`. Fitting round 3 gives `D` = 58 ms and `P` = 775 ms, i.e. a
512-token chunk costs about 13 decode steps on this hardware. Predicted TTFT at
`C` = 14: 14.1 s / 17.3 s / 23.8 s at intervals 4 / 8 / 16, against measured 14.7
/ 17.8 / 23.8. Predicted mean inflation 4.3x / 2.7x / 1.8x against measured 4.4x
/ 2.7x / 1.8x.

**The disabled arm's 51 s is not a bound, and that is the whole point.** It is
whatever the decode batch takes to drain, so it grows with the streams' length
and with every request admitted behind them. Lengthening the streams from 512 to
900 tokens moved it and left every grant arm's TTFT unchanged. Interval 16's 23.8
s is a genuine ceiling under this offered load: 14 chunks, one per 17 ticks.

**The p95 column is a percentile artifact and should not be used to pick the
interval.** Exactly one gap in `N + 1` carries a chunk forward, so the chunk is
inside the top 5% of gaps whenever `1 / (N + 1) > 0.05`, that is whenever
`N < 19`. That is the entire explanation for a p95 that sits at 11-16x for every
interval from 4 to 16 and then falls to 1.0x at 32. Nothing got better at 32: the
same 13 hiccups of the same size happened, and they moved from p95 to p99. The
lever that shrinks a hiccup rather than hiding it is `--prefill-chunk-size`,
which sets `P` directly. The honest cost of the grant is the mean column.

**p50 is untouched at every interval** (54.7 → 58.4 ms at the default). A
decoding stream's typical token is unaffected; the cost is entirely in how often
it waits a whole chunk.

## Why the default is 16

The mean-inflation curve has its knee there. Each halving below 16 buys much
less TTFT than the ITL it costs (16 → 8 saves 6 s of TTFT for +0.7x mean ITL;
8 → 4 saves 3 s for +1.6x), and each doubling above buys much less ITL than the
TTFT it costs (16 → 32 pays 6 s measured, 13 s modelled once the batch stops
draining early, to save 0.6x). Interval 16 roughly halves the observed baseline
TTFT for 1.6 to 1.8x mean inter-token latency during the admission window only,
and turns an unbounded wait into a stated one.

The interval is counted in ticks, which is what makes it hardware-independent to
test and free to evaluate, and also what stops it from being a portable statement
about time: `P / D` is 13 here and will differ on other hardware, other models,
and other chunk sizes, so the same interval grants a different share of wall
clock. Deployments that care about the exact share should read `P / D` off their
own run of this harness and set the flag accordingly.

## Not measured

- **CUDA / GB10.** The policy is scheduler-level and backend-agnostic, and no
  GB10 hardware was available. `P / D` is much larger on a machine with fast
  prefill and slow decode, which would shift the knee.
- **Multiple concurrent long prompts.** Not a gap in coverage: the scheduler
  holds at most one parked chunked prefill (a single `Option`), and the tick
  policy's chunked branch short-circuits above the admission branch, so a second
  long prompt cannot be admitted while the first is parked. There is no
  prefill-versus-prefill contention to measure.
- **The suppressed per-chunk `clear_memory_cache()`.** #1011 generalises #908's
  `mixed_tick` suppression to "a decode batch is live", because a granted chunk
  now runs next to one by default. No A/B was run: this is not a claimed
  speedup, it is avoidance of a pattern the decode path already avoids
  deliberately (`cache_clear_interval()`, ml-explore/mlx#2358). Treat it as
  unmeasured.
- **Interaction with a speculative slice.** #734's strict alternation halves the
  classic tick rate, so the grant fires at half the wall-clock rate and the bound
  doubles. This is pinned as a unit test
  (`a_speculative_slice_cannot_starve_the_parked_prefill`) and not measured on
  hardware; it needs a Gemma 4 MTP checkpoint alongside a long prompt.

## Reproduce

```bash
# Structural: does the parked prefill advance while decode runs?
GRANT=0 scripts/bench/starvation_probe.sh baseline    # starved
scripts/bench/starvation_probe.sh baseline            # shipped default

# ITL price at one interval. Serve with --parallel above --streams or the
# admitted request never leaves the queue.
./target/release/mlxcel-server -m models/llama-3.1-8b-4bit \
    --parallel 8 --prefill-chunk-size 512 --prefill-grant-interval 16 \
    --metrics --port 8080
python3 scripts/bench_mixed_step_admission.py \
    --stream-max-tokens 900 --expect baseline --expect-grant on
```

`--stream-max-tokens 900` matters: with shorter streams the decode batch drains
before a large interval's prefill finishes, the harness clamps the window, and
the arm reads like the disabled one.
