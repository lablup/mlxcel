# Benchmarks

This page documents how benchmark claims should be recorded for `mlxcel`. It is
intentionally conservative: do not publish aggregate speedup numbers without the
raw per-model rows and the exact software/hardware versions used to produce them.

## What to record

For every benchmark run, include:

- hardware model and memory size;
- operating system version;
- `mlxcel` version or commit;
- pinned MLX commit/version;
- comparison runtime version (`mlx-lm`, `mlx-vlm`, or another baseline);
- model checkpoint name and quantization format;
- prompt length, requested decode length, batch size, and warmup policy;
- cache mode and server/generation flags;
- raw per-model prefill and decode throughput where available;
- for op-level microbenchmarks, the memory mode (warm or cold last-level cache) and the rotation count, per the section below.

Averages are useful only after the raw rows are available. Avoid statements such
as "faster than X" unless the comparable model set and exclusions are explicit.

## Current result snapshot

Keep public result summaries in a single place so aggregate numbers do not drift
between documents. The current Apple Silicon benchmark report is:

- [Benchmark Report - 2026-05-19](benchmark_results/benchmark-report.md)

Use that report and its linked raw per-hardware tables for release notes,
README updates, or capacity planning. This page should stay focused on
methodology, required metadata, and caveats.

Decode-gap investigations (root-cause analyses of where mlxcel trails the
reference runtime) live alongside the snapshot:

- [MoE decode gap investigation](benchmark_results/moe-decode-gap-investigation.md)
- [Fused decode-MoE kernel: design and roadmap](benchmark_results/fused-moe-decode-kernel-design.md)
- [Gemma3n decode profile: is a compiled fusion justified?](benchmark_results/gemma3n-decode-profile.md)
- [Gemma3n decode profile on M5 Max](benchmark_results/gemma3n-decode-profile-m5max.md)

## Suggested benchmark commands

The repository contains benchmark helper scripts under `scripts/`. The exact
arguments may evolve, so inspect each script before publishing results.

```bash
# Single-model decode benchmark shape.
./scripts/bench_decode.sh -m models/<checkpoint> --runs 3

# Multi-model suite shape.
./scripts/bench_all_models.sh --hardware <name> --cooldown 45 --big-cooldown 60

# Sampling step, no model attached. Gumbel-max (#900) covers the no-filter
# path; the rejection kernel (#901) covers top-k / top-p / min-p.
cargo run --release --features metal,accelerate --example gumbel_sampling_microbench
cargo run --release --features metal,accelerate --example rejection_sampling_microbench
```

Both sampling harnesses print, before the table, the dispatch outcome each arm
recorded, from the same record `mlxcel-server` announces at INFO. Read those
lines before you read the numbers. Issue #899 shipped a production benchmark
that compared the fallback against itself across a full sweep and returned a
clean-looking null result, because nothing said which path had run; a sampling
sweep whose two arms report the same path is measuring nothing.

### An op-level number is not a decode number (issue #901)

`examples/rejection_sampling_microbench.rs` reports two speedups per row, and
the second one is the one to read.

`iso_x` is the classic op-level measurement: build, `eval`, `synchronize`, once
per iteration. `pipe_x` reproduces what a decode loop does, which is a software
pipeline: build the next forward and the next sample, `async_eval` both, then
read the PREVIOUS step's token, with one synchronize at the end of the run.

The two disagree whenever the operation under test forces a synchronization on
its caller, because `iso` has already paid for a sync at every iteration and is
structurally blind to one more. The first cut of #901 read a device flag back to
host inside the sampler; `iso` scored it 1.14x to 1.17x faster at vocab 152064
while end-to-end decode on Qwen3-0.6B measured 1.7x SLOWER. A large `iso_x` with
a poor `pipe_x` means the operation is synchronizing.

The general rule this leaves behind: an op-level harness measures an operation
in isolation, and "in isolation" silently includes "with the caller's pipeline
already drained". Any operation that touches host memory, reads a flag, or
branches on device state needs a pipelined arm before its number means
anything. `the_production_sampling_call_never_synchronizes` in
`src/lib/mlxcel-core/src/sampling_rejection_tests.rs` is the cheaper form of the
same check: it enqueues a large chain of matmuls and asserts the sampler returns
before that chain drains, so a regression fails a test rather than a benchmark.

## Fused decode kernels: the measure-then-keep gate (issue #905)

The two fused decode kernels from issue #905, residual-add + RMSNorm and q/k
RoPE + KV-append layout, land under a measure-then-keep policy: each keeps its
wiring only if it beats the graph it replaced at op level on the adopting
backend and does not lose on the other. `examples/fused_norm_rope_microbench.rs`
produces that evidence. It loads no model, sweeps hidden sizes 2048 / 4096 /
8192 against batch 1 / 4 / 8, and prints both a human-readable table and a CSV
block.

```bash
caffeinate -i cargo run --release --features metal,accelerate \
    --example fused_norm_rope_microbench

caffeinate -i cargo run --release --features cuda \
    --example fused_norm_rope_microbench
```

A speedup below 1.00 for either op is the signal to flip
`FUSED_ADD_RMSNORM_DEFAULT` or `FUSED_ROPE_APPEND_DEFAULT` in
`src/lib/mlxcel-core/src/layers.rs`, which leaves the kernel available and
opt-in through `MLXCEL_FUSED_ADD_RMSNORM=1` / `MLXCEL_FUSED_ROPE_APPEND=1`
instead of removing it.

The op-level number is a lower bound on the end-to-end effect. Both fusions also
remove a full-width intermediate from the MLX graph per call, which shows up as
allocator and dependency-tracking pressure rather than as kernel time, so the
end-to-end decode sweep (one dense model and one MoE model, batch 1 and 4, with
each kill switch flipped for the A/B) is the deciding measurement. Record both
in `docs/benchmark_results/fused-norm-rope-<hw>-<date>.md`.

## Warm vs cold last-level cache (issue #906)

An op-level microbenchmark that allocates its inputs once and reuses them on
every timed iteration measures a warm cache. After the first iteration the
working set is resident in the last-level cache (Apple's System Level Cache,
NVIDIA's L2), so the remaining iterations read at cache bandwidth. For a
bandwidth-bound kernel that is a different measurement from the one production
takes: the KV pool is far larger than any last-level cache and is touched once
per decode step, so the representative read comes from DRAM.

The gap matters most for exactly the kernels this epic touches. Paged KV
gather, paged decode attention, and rmsnorm are bandwidth-bound, so a warm-cache
number for them is an upper bound rather than an estimate. Compute-bound
kernels (large-M quantized GEMM) barely move between the two modes, which is
itself a useful signal: a kernel whose warm and cold numbers agree is not
limited by memory.

### How the harnesses do it

`mlxcel_core::bench_rotation` allocates several copies of the input and
advances one copy per timed iteration. The rotation count is
`ceil(2 * last_level_cache / per_iteration_read_bytes)`, clamped to 64, so the
whole rotation set exceeds the cache and a buffer has been evicted by the time
the rotation returns to it. The 2x headroom covers the cache being shared with
the rest of the system and with the kernel's own output traffic.

Cache sizing is an estimate by device family, because macOS exposes no SLC size
through `sysctl`: 8 MiB for a base M-series, 24 MiB for a Pro, 48 MiB for a
Max, 96 MiB for an Ultra (two dies, two SLCs). The estimates are biased high,
since over-estimating only costs memory for extra rotation buffers while
under-estimating silently reintroduces the warm-cache bias. Reading the CUDA L2
size needs `cudaDeviceProp::l2CacheSize` through an FFI helper that does not
exist yet, so on a CUDA host set `MLXCEL_BENCH_LLC_BYTES` to the device's real
L2 size. Set the same variable on Apple Silicon when the published SLC figure
for the specific chip is known.

Note that a large working set needs no rotation at all: once a single iteration
reads more than the last-level cache holds, the rotation count collapses to 1 and
the cold mode costs nothing. On an M1 Ultra (96 MiB SLC) that crossover lands at
batch 4 / context 16384. The two modes diverge at small batch and short context,
which is also where a warm measurement is most misleading.

### What it actually measured on Apple Silicon

Measured before assuming, because the size of the effect turned out to matter
less than its shape. On an M1 Ultra at batch 1 / context 4096 (rotation 12),
medians over five repetitions each:

| Path | Warm | Cold | Delta |
|---|---|---|---|
| `contig_sdpa` | 438.3 us | 433.7 us | -1.0% |
| `gatherA_sdpa` | 509.4 us | 535.0 us | +5.0% |

The median barely moves. What moves is the spread: warm `gatherA_sdpa` ranged
425.6 to 542.1 us (27%, including one 708 us first-iteration outlier), while cold
ranged 525.1 to 546.0 us (4.0%).

So on this part the case for cold mode is reproducibility, not a large correction
to the number. Unified memory with very high bandwidth blunts the cache cliff that
motivates the technique. Do not carry that conclusion to CUDA: a discrete GPU with
a private L2 behind PCIe has a much sharper cliff, and the same rotation there
should be expected to move the median considerably more. Full numbers and the
load conditions they were taken under are in
[autotuner-m1ultra-2026-07-30](benchmark_results/autotuner-m1ultra-2026-07-30.md).

### Running and recording

```bash
# Warm (historical default, unchanged).
caffeinate -i cargo run --release --features metal,accelerate \
    --example page_gather_microbench

# Cold last-level cache.
caffeinate -i cargo run --release --features metal,accelerate \
    --example page_gather_microbench -- --cold-l2
```

The harness prints `memory mode=...` in its header, `mode=` and `rotation=` per
config, and appends `mode` and `rotation` as the last two columns of its `CSV:`
rows. Every recorded result must state which mode produced it; a warm number
and a cold number for the same kernel are not comparable and must never appear
in the same column of a report.

`examples/qmm_gemv_microbench.rs` has carried its own ad-hoc version of this
since it landed (a fixed 128 MiB target, 2 to 12 weight copies round-robin).
`bench_rotation` is the generalization of that idea with the cache size
detected rather than assumed.

## Speculative decoding (MTP)

MTP speculative decoding pairs a decode target with a small assistant drafter
that proposes a block of tokens, which the target then verifies in a single
forward pass. At `temperature 0` the accelerated output is byte-identical to
classic decode **where the runtime exactness probe says it is**, so on a probe
that passes the only metric that moves is decode throughput; confirm correctness
by diffing the two completions. The probe is not a formality: whether a `T = K`
verify block is bit-equal to `K` single-token steps depends on which MLX kernel
each quantized projection dispatches to at `M = K` versus `M = 1`, which varies
by Apple GPU generation, quantization mode and block width. The Qwen 3.5 family
declines to classic decode when the probe fails (#1186). The Gemma 4 arms are not
probed yet and their byte-identity has not been measured on Apple GPU generation
15 or newer (#1188), so treat the claim there as unverified on M3, M4 and M5
rather than established.

For each pairing, record both the baseline (no drafter) and the MTP run:

- decode tok/s for each, and the speedup ratio (MTP divided by baseline);
- mean acceptance length (accepted draft tokens per verify), read from the
  `MTP round-loop diagnostics` log line;
- the block size (`--draft-block-size`), and whether the singleton burst
  engaged or was declined.

Measure with the `speculative_bench` harness or the server:

```bash
# In-process harness: baseline vs MTP on the same target.
./target/release/speculative_bench --target <target_dir> --kind none --max-tokens 256
./target/release/speculative_bench --target <target_dir> --draft <drafter_dir> --kind mtp --max-tokens 256

# Server (production path): time a fixed temperature-0 completion with and
# without the drafter. The server logs decode tok/s and acceptance per request.
mlxcel serve -m <target> --draft-model <drafter> --draft-kind mtp
```

The offline `mlxcel generate` command also supports the MTP round-loop path (issue #166) for Gemma 4 text, VLM, and Unified targets. Pass `--draft-model <drafter> --draft-kind mtp` explicitly; without `--draft-kind mtp` the command keeps the classic speculative path for backward compatibility even when the drafter auto-detects as MTP.

Parity: at `temperature 0` with no repetition, frequency, presence, or DRY penalties, the offline MTP output matches the non-speculative path within the documented f16 / #203 batched-kernel jitter class (on some hardware, notably M1 Ultra, near-tie token choices can differ). When any sampling penalty is active, only the first bonus token samples from the penalized distribution; subsequent tokens in each verify window are accepted or rejected greedily, so non-greedy or penalized requests are not byte-identical to the non-speculative path. This matches the known limitation of the server burst path.

### Judging a change that moves the numbers

Byte-identity answers one question well and nothing else: is this arithmetic
path bit-equal to its reference. Once the answer is no, and on Apple GPU
generation 15 and newer it is no for reasons the caller did not choose, the
tool is spent. Perplexity answers a different question, whether a model's
predictive distribution got worse on a corpus, and a kernel reordering can
leave it unmoved while flipping percents of the greedy tokens a user sees.

`examples/logit_trace` covers the gap. It is teacher-forced, so both arms are
scored over the same token stream and no comparison is lost to divergence; a
free-running comparison collapses at the first flipped token, and on a real
pairing that left 16 comparable positions out of 250. Each configuration
writes its own trace, which is what lets process-global switches like
`MLXCEL_QMV_WIDE` be compared at all, and
`scripts/compare_logit_traces.py` reads two traces.

```bash
cargo build --release --features metal,accelerate --example logit_trace
./target/release/examples/logit_trace  MODEL CORPUS.txt 5 60 8 512 > a.tsv
MLXCEL_QMV_WIDE=0 \
./target/release/examples/logit_trace  MODEL CORPUS.txt 5 60 8 512 > b.tsv
python3 scripts/compare_logit_traces.py b.tsv a.tsv
```

The metric to gate on is **disagreement on decided positions**: the fraction
of positions where the reference's own top two were more than a stated gap
apart and the candidate still picked something else. A position the reference
was indifferent about has no right answer to get wrong, and pooling those with
decided ones hides the only distinction that matters. Byte-identity is the
limit case, zero disagreements at every gap.

Two things decide whether the answer means anything.

**Trace at the width the code under test runs at.** A forward over `N`
positions runs the quantized projections at `M = N`, and MLX picks a different
kernel per `M`, so the chunk width selects what is measured rather than how
much. The same `MLXCEL_QMV_WIDE` comparison on gemma-4-12b-it-4bit reads 20.6%
top-1 disagreement at width 8, 19.5% at 16, and 0.0% at both 32 and 256,
because `use_qmv_wide` splits at `M >= 2` while the batch limit sends larger
`M` to a matrix-matrix kernel both arms share. An MTP verify block is the block
size, a decode step is 1, a prefill is the prompt length.

**Separate context length from forward width.** The sixth argument prefills a
context whose rows are not traced, so a narrow forward can be measured against
a realistic history. It matters: the same comparison at width 5 with no context
and at width 5 behind 512 tokens are different measurements, and only the
second one is the shape a verify actually runs at. Behind 512 tokens the two
kernels disagree on 4.0% of positions overall but 0.585% of decided ones, and
three quarters of the disagreements are the reference's runner-up.

### Gemma 4 Unified (12B) + 4-bit assistant

`mlx-community/gemma-4-12b-it-4bit` as the target and
`mlx-community/gemma-4-12B-it-assistant-4bit` as the drafter, `temperature 0`,
warm, arms alternated with a warm-up discarded, eight samples per arm, the
host's background indexers suspended, spreads at or under 1.9% of the median
on M5 Max, 2.9% on M3 Ultra and 1.9% on M1 Ultra.

| Host | Output | Prompt | Tokens | Block | acceptance | classic | MTP | speedup |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M5 Max (128 GB) | enumeration | "Count from 1 to 200, one number per line, with no other text." | 400 | 4 | 0.997 | 43.1 | 135.4 | **3.14x** |
| M5 Max (128 GB) | source code | "Write a Python function that computes the nth Fibonacci number, with a docstring and type hints." | 300 | 5 | 0.784 | 43.5 | 121.0 | **2.79x** |
| M5 Max (128 GB) | prose | "Explain how speculative decoding accepts or rejects draft tokens." | 400 | 5 requested, 4 effective | 0.489 | 43.3 | 82.4 | **1.90x** |
| M3 Ultra (512 GB) | enumeration | "Count from 1 to 200, one number per line, with no other text." | 400 | 4 | 0.980 | 63.4 | 165.5 | **2.61x** |
| M3 Ultra (512 GB) | source code | "Write a Python function that computes the nth Fibonacci number, with a docstring and type hints." | 300 | 5 | 0.733 | 64.2 | 138.5 | **2.16x** |
| M3 Ultra (512 GB) | prose | "Explain how speculative decoding accepts or rejects draft tokens." | 400 | 5 requested, 4 effective | 0.544 | 64.0 | 111.2 | **1.74x** |
| M1 Ultra (128 GB) | enumeration | "Count from 1 to 200, one number per line, with no other text." | 400 | 4 | 0.997 | 34.2 | 50.4 | **1.48x** |
| M1 Ultra (128 GB) | source code | "Write a Python function that computes the nth Fibonacci number, with a docstring and type hints." | 300 | 5 | 0.815 | 34.9 | 43.5 | **1.25x** |
| M1 Ultra (128 GB) | prose | "Explain how speculative decoding accepts or rejects draft tokens." | 400 | 5 requested, 4 effective | 0.525 | 34.5 | 32.7 | **0.95x** |

Acceptance is the column that explains the rest within a host. The enumeration
row accepts almost every draft (3.990 tokens emitted per verify against a
block of 4 on M5 Max, 3.912 on M3 Ultra, 3.990 on M1 Ultra) and the prose row
accepts about half (2.463, 2.625 and 2.574). The prompt is the only thing that
differs between those rows.

Read the hosts down the columns rather than across the speedups. All three
M3 Ultra rows sit below their M5 Max twins while both arms are faster in
absolute terms: classic decode runs at about 63 to 64 tok/s there against 43,
so the baseline the ratio divides by gained more than the MTP arm did. The
speedup is a property of the pair, not a ranking of the hosts, and comparing a
ratio against one measured on other silicon says nothing. The Qwen pairing
below moves the other way on the same two hosts, which is the point: the size
of the host effect is not transferable between pairings either.

The M1 Ultra rows say the arithmetic above is not the mechanism. Both of its
arms are *slower* than M5 Max, not faster, and the ratios fall further than
M3 Ultra's did, to an outright regression on prose. What all three hosts do
share is a single quantity, and it is the one worth measuring: the cost of a
verify round in units of that host's own classic decode step, which is the
round's wall time divided by `1 / classic tok/s`.

| Host | Block | round cost, in classic steps | emitted per verify to break even |
| --- | ---: | ---: | ---: |
| M5 Max (128 GB) | 4 | 1.27 (enumeration), 1.29 (prose) | ~1.28 |
| M3 Ultra (512 GB) | 4 | 1.50 (enumeration), 1.51 (prose) | ~1.51 |
| M1 Ultra (128 GB) | 4 | 2.70 (enumeration), 2.72 (prose) | ~2.71 |
| M1 Ultra (128 GB) | 5 | 3.15 (source code) | ~3.15 |

On each host two prompts with nothing in common agree on the round cost to
within 1%, which is the control that this is a property of the host and the
block width rather than of the prompt. It also orders the three hosts exactly
as the speedups do, and it explains the M3 Ultra reading above without
appealing to which arm gained more: the verify simply costs relatively more
there than on M5 Max.

The break-even column is what the regression comes from. A round has to emit
more tokens than it costs in classic steps, so M5 Max needs 1.28 tokens per
verify and clears it on every prompt here by a wide margin, M3 Ultra needs
1.51, and M1 Ultra needs 2.71 at block 4. The prose row's 2.574 lands just
under that last one, which is the whole of its 0.95x.

The mechanism is the `use_qmv_wide` split documented in
`src/models/speculative_exactness.rs`: from Apple GPU generation 15 a
quantized projection at `M >= 2` runs one wide pass, while generation 13 has
no such path and runs the block as narrow passes whose cost grows with the
width. M1 Ultra is generation 13 and pays nearly per-position for the verify
that the other two amortise.

Acceptance also moves between hosts on an identical prompt and pairing (0.784,
0.733 and 0.815 on the code row), though not always: the enumeration row reads
0.997 on both M5 Max and M1 Ultra, to three digits. Sampling is not the
source, since at `temperature 0` both arms are greedy. The explanation
consistent with everything else on this page is that the hosts resolve the
target's own near-tie positions differently, which changes the continuation
and therefore what there is to accept at all. That is the divergence the
exactness probe reports below, seen from the acceptance side. It has not been
traced per host, so treat it as the reading of the numbers rather than a
measured cause, and expect acceptance to be a per-host figure rather than one
carried between rows.

Reproduce or extend the table with `scripts/bench_speculative.sh`. It carries
the prompts, the block widths and the protocol, detects the host, prints rows
in the shape above, and refuses to start until nothing else is using the GPU.
It keeps watching while it measures, because the entry gate guards only the
start: a load that arrives later is otherwise left to the spread check alone,
and a steady load evades that check by depressing every sample equally. A run
whose spread exceeds 4% of the median, or that spent over a fifth of its time
contended, is reported as untrustworthy rather than averaged, because a
contaminated median is indistinguishable from a real regression once it
reaches a document.

On a Mac that is also somebody's desktop, the thing most likely to fail those
checks is the machine's own housekeeping. Spotlight indexing, Photos analysis
and cloud sync are idle-triggered, so they start up exactly when a host is
left alone to measure something, and they reach several hundred percent CPU
without `pmset -g therm` reporting anything. Run the sweep through
`scripts/with_indexers_paused.sh`, which suspends them for the length of one
command and resumes them however it ends:

```bash
./scripts/with_indexers_paused.sh ./scripts/bench_speculative.sh --reps 4
```

It uses SIGSTOP and SIGCONT only, so the suspended work continues from where
it left off, and three separate paths resume it, including one that survives
a SIGKILL of the wrapper itself. `INDEXER_EXTRA_NAMES` takes a list, one name
per line, of anything else this particular host needs quiet, since which chat
and mail clients sit on top of the indexers is a property of the machine.
Time Machine is the one contender it deliberately does not touch: end a
running backup with `tmutil stopbackup` before the sweep, which lets it resume
incrementally later, rather than freezing a backup session for an hour.

The M1 Ultra rows were measured this way on 2026-08-19. The record is
`benchmark_results/speculative-decoding-m1ultra-2026-08-19.md`, including the
two rows the guard rejected and remeasured, and the derivation of the
round-cost figures above.

The block-width tables further down come from `scripts/bench_block_width.sh`,
run the same way:

```bash
./scripts/with_indexers_paused.sh ./scripts/bench_block_width.sh gemma
./scripts/with_indexers_paused.sh ./scripts/bench_block_width.sh qwen
```

It visits every width once per round and rotates which width starts the round,
because measuring one width to completion before the next puts any drift over
the run onto whichever widths were measured late — indistinguishable from
those widths being slower, which is the question the sweep is asking.

**Record the host and the prompt.** Both move the ratio by more than most code
changes do. The prompt decides acceptance, and the host decides which kernel
each quantized projection dispatches to and therefore what a verify block
costs, which is why the same pairing can pay on one generation and regress on
another. A row without both cannot be reproduced or compared, and rows from
different protocols do not belong in the same table.

A speculative-decoding figure without its prompt cannot be reproduced or
compared.

The block width is not a tuning knob worth much, but where it peaks is a
per-host fact rather than a constant. On M5 Max, measured on the code row,
throughput peaks at width 5 and falls at 6, 8, 10 and 12. On M3 Ultra it peaks
at 4, and what follows is a plateau rather than a fall:

| width | decode tok/s | spread | acceptance | emitted per verify | vs peak |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 129.1 | 0.7% | 0.843 | 2.670 | -10.1% |
| 4 | 143.6 | 0.6% | 0.803 | 3.398 | **peak** |
| 5 | 138.0 | 0.9% | 0.733 | 3.477 | -3.9% |
| 6 | 138.8 | 0.5% | 0.738 | 3.737 | -3.3% |
| 8 | 135.1 | 0.6% | 0.683 | 3.934 | -5.9% |
| 10 | 126.2 | 2.7% | 0.628 | 4.041 | -12.1% |
| 12 | 127.7 | 2.8% | 0.658 | 4.333 | -11.1% |

Widths 5 to 8 sit within 6% of the peak, and the ordering inside that band is
not resolved by these samples: 6 reads 0.6% above 5, less than the spread
either was measured with, and 12 reads 1.2% above 10 against spreads of 2.7
and 2.8%. Only three things separate cleanly — width 4 above the band, and
3, 10 and 12 below it — so this host says "peak at 4, then a plateau", not
the ranking the M5 Max sentence gives. Width 5 reads 138.0 here against the
138.5 the table above measured at block 5, a 0.4% agreement inside the
spread, so the sweep and the published row are the same measurement twice.

The mechanism is in the last two columns. Emitted per verify climbs towards
`1 / (1 - acceptance)` and flattens — 2.670, 3.398, 3.477, 3.737, 3.934,
4.041, 4.333 across the range — while the verify forward keeps costing more
per position, so past the peak each widening buys less than it pays for.

One consequence for the row above: on this host the adaptive controller does
not land on the peak. The code row runs at effective width 5 and leaves 3.9%
on the table against width 4, which is well outside the 0.6 to 0.9% spreads
either width was measured with. That is small enough to ignore and too large
to call noise, so it is recorded rather than tuned away here.

The accelerated output is byte-identical to classic decode where the startup
exactness probe says it is, which the runtime measures rather than assumes: an
affine 4-bit target on Apple GPU generation 13/14 is byte-identical for block
widths below 12, while generation 15 and newer (M3, M4, M5) diverge from block
width 2 because MLX routes `M >= 2` quantized projections to a different
reduction. A declining probe falls back to classic decode unless
`MLXCEL_MTP_ALLOW_INEXACT=1` is set. B=1 (single-request)
MTP runs by default for every MTP target; the Gemma 4 Unified target cannot batch
at all, so B=1 is also its only decode path. The batch-capable 31B + bf16
assistant measures ~1.2 to 1.4x on the same host. Set `MLXCEL_ENABLE_MTP_B1=0` to
opt out on hardware where the B=1 verify forward does not pay for itself.

Gemma 4 is not probed yet (#1188), so the rows above are the fast kernel rather
than the byte-identical one. Keeping byte-identity on the code row, by dropping
`qmv_wide`, measures 93.2 tok/s instead of 121.0 on M5 Max, or 2.14x instead of
2.79x, and 117.5 tok/s instead of 138.5 on M3 Ultra, 1.83x instead of 2.16x.
That is 23% of throughput on one host and 15% on the other, which is not the
same quantity as the 17 to 20% the probe quotes for the Qwen pairing: the
probe is costing the verify forward, while these figures are end-to-end decode,
where the drafter step and the accepted-token emission are unaffected. Quote
whichever one the question is about, and not the other.

On M1 Ultra there is no such cost, because generation 13 never takes
`qmv_wide` in the first place, but the missing probe still lets a divergence
through. Diffing the two arms at `temperature 0` on the three prompts above
gives byte-identical output on source code and on enumeration, and a
**reproducible divergence on prose**, 892 bytes into a 1755-byte generation:

```
classic: ...tokens in parallel (as long as they are provided as inp
MTP:     ...tokens in parallel (the attention mechanism allows this
```

Each arm is byte-identical to itself across three runs and the two arms
disagree with each other every time, so this is the block-versus-chain path
and not run-to-run noise. The Qwen pairing below, which *is* probed and whose
probe passes on that host, comes out byte-identical on the same test. That
contrast is the argument for #1188: the probe catches what the unconditional
Gemma 4 gate does not, and one 400-token generation per arm reproduces it. It
also bounds the rule of thumb above, which holds at the op level and does not
carry to the model level here: two of three prompts agree and one does not.

### Qwen 3.5 / 3.6 / 3.8 with the model's own MTP head

Qwen ships the MTP head as part of the family rather than as a companion
checkpoint, either split out (`Qwen3.8-27B-MTP-bf16`, `-4bit`) or carried inside
the target. Same protocol, `qwen3.8-27b-4bit` with `qwen3.8-27b-mtp-4bit`, the
code prompt above, 300 tokens, eight samples per arm:

| Host | Path | decode tok/s | speedup |
| --- | --- | ---: | ---: |
| M5 Max (128 GB) | classic decode (no drafter) | 32.7 | 1.00x |
| M5 Max (128 GB) | MTP, block 3 (the drafter's declared width) | 53.4 | **1.63x** |
| M3 Ultra (512 GB) | classic decode (no drafter) | 35.7 | 1.00x |
| M3 Ultra (512 GB) | MTP, block 3 (the drafter's declared width) | 59.5 | **1.67x** |
| M1 Ultra (128 GB) | classic decode (no drafter) | 23.8 | 1.00x |
| M1 Ultra (128 GB) | MTP, block 3 (the drafter's declared width) | 23.4 | **0.98x** |

On M5 Max the MTP figure is a median over eight samples that ranged from 50.8
to 55.2, an 8.2% spread against 0.3% on the classic arm of the same run.
Contention would have moved both arms, so this belongs to the pairing rather
than the host: acceptance and effective block come out identical every run at
`temperature 0`, and what varies is where the adaptive B=1 controller (#333)
lands when it profiles the opening bursts. The median is stable even so,
reading 1.61x, 1.64x and 1.63x across three independent sweeps. A single run
of this pairing on that host is worth roughly 1.55x to 1.69x, so a move
smaller than that is not a result.

That spread is the one part of this pairing that did not carry over. The same
sweep on M3 Ultra measured 0.7% on both arms, so the width of the interval
above is a property of the pairing *on that host*, not of the pairing alone.
Read the M3 Ultra row at its face value and re-measure the spread before
quoting an interval for any third host: the controller has a different set of
opening bursts to profile on each one.

Two things about those two ratios. They are measured **with** the
byte-identity guarantee: the exactness probe fires on M5 Max and on M3 Ultra
and drops `qmv_wide`, which costs the verify forward about 17 to 20%. Both
Qwen rows are therefore already paying what the Gemma rows above do not, which
is most of why they sit lower:
on M3 Ultra the byte-identical Gemma code row is 1.83x against Qwen's 1.67x,
where the fast-kernel Gemma row reads 2.16x. And the block width is genuinely
optimal at 3 to 4 rather than merely default: 48 of the target's 64 layers are
GatedDeltaNet, a recurrence that processes tokens in sequence, so the verify
cost grows nearly linearly with the block instead of amortising the way an
attention-only target's does. Widths 5, 6 and 8 measured 48.0, 46.8 and 35.7
in an earlier run on M5 Max, where the gap between widths 3 and 6 sat inside
that host's run-to-run range and only width 8 was clearly outside it.

The M3 Ultra sweep separates what that range swallowed:

| width | decode tok/s | spread | acceptance | emitted per verify | vs peak |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 54.6 | 0.5% | 0.875 | 1.869 | -8.2% |
| 3 | 59.5 | 0.9% | 0.753 | 2.492 | **peak** |
| 4 | 57.4 | 0.7% | 0.689 | 3.051 | -3.5% |
| 5 | 52.1 | 0.6% | 0.607 | 3.398 | -12.4% |
| 6 | 49.4 | 0.4% | 0.546 | 3.691 | -17.0% |
| 8 | 40.7 | 0.3% | 0.412 | 3.833 | -31.6% |

"Optimal at 3 to 4" is a measurement here rather than a restatement of the
declared width: 3 is the peak, 4 is 3.5% back, and every gap in the table is
far outside the 0.9% spread it was measured with, including the 3-to-6 one
M5 Max could not resolve. The drop is also steeper than the Gemma pairing's
over the same widths, which is what the GatedDeltaNet recurrence predicts —
a verify cost growing with the block rather than amortising across it.

M1 Ultra is a wash on this pairing, and for the reason the Gemma table gave.
Its acceptance is the highest of anything on this page (0.855, 2.694 tokens
emitted per verify at block 3) and it still does not clear that host's round
cost. Neither caveat above applies there: generation 13 never takes
`qmv_wide`, so the probe passes as it stands and there is no 17 to 20% being
paid, and both arms measured 0.2% and 1.3% rather than M5 Max's 8.2%. This is
the same pairing that measured 0.59x to 0.70x on that host in
`benchmark_results/qwen38-mtp-m1ultra-2026-08-16.md` with the bf16 drafter;
quantizing the drafter to 4-bit (#1185 Phase 3) is what moved it to
break-even, and it did not move it past. Nothing measured here argues for
enabling this pairing on generation 13.

### Gemma 4 31B + bf16 assistant

The 31B text target is batch-capable, and its MTP speedup comes from batched
(B>1) verify windows rather than the singleton path. The scheduler declines B=1
MTP there because the bf16 assistant's single-stream acceptance is too low to
offset the extra drafter forward per token. This pairing is wired into
`speculative_bench` (`REACHABLE_PAIRINGS`) and runs once the
`gemma-4-31b-it-4bit` and `gemma-4-31B-it-assistant-bf16` checkpoints are present
in the model store.

### Adaptive B=1 MTP policy

Since issue #333 the server no longer decides the B=1 MTP path from the static
per-hardware gate alone. It profiles the first few B=1 bursts of each (target,
drafter, hardware) pairing (acceptance length, verify latency, drafter latency,
batch size, prompt shape) and settles to a data-driven verdict: a clearly
favorable profile enables MTP even where the static gate would decline, a
clearly unfavorable one declines it, and an ambiguous profile keeps the static
per-hardware default above. The settled verdict (enable/decline plus the coarse
acceptance rate, never prompt data) is cached under
`${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/mtp-policy/`, so profiling is a
one-time cost per pairing and survives restarts. MTP stays mathematically
exact: the policy only chooses when to run it, so it neither creates nor removes
the byte-identity the exactness probe above establishes. `MLXCEL_ENABLE_MTP_B1` pins the decision in
either direction (and suppresses profiling); `MLXCEL_MTP_ADAPTIVE=0` restores
the pre-#333 static gates. When recording benchmark numbers, discard the
profiling window and report the settled-verdict steady state.

## Recommended output layout

Add benchmark artifacts under a dedicated directory before publishing a release,
for example:

```text
benchmarks/
  2026-05-08_m1-ultra_text.csv
  2026-05-08_m1-ultra_vlm.csv
  README.md
```

Each CSV should be machine-readable and accompanied by a short Markdown note
that describes methodology, exclusions, and known failures.

## Caveats

- **Thermals matter.** Apple Silicon decode throughput changes with sustained
  load; record cooldown and run order.
- **MLX pin matters.** Kernel selection can change when the pinned MLX commit
  changes.
- **VLM comparisons are separate from text comparisons.** Vision preprocessing,
  image resolution, and prompt construction differ by family.
- **CUDA numbers are not interchangeable across GPUs.** Publish the SM target
  and driver/toolkit versions with the result.
