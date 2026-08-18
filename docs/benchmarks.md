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
warm, arms alternated with a warm-up discarded, spreads under 1% of the median.

| Host | Output | Prompt | Tokens | Block | classic | MTP | speedup |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| M5 Max (128 GB) | source code | "Write a Python function that computes the nth Fibonacci number, with a docstring and type hints." | 300 | 5 | 43.5 | 121.8 | **2.80x** |
| M5 Max (128 GB) | prose | "Explain how speculative decoding accepts or rejects draft tokens." | 400 | 5 requested, 3 to 4 effective | 43.3 | 84.3 | **1.95x** |

**Record the host and the prompt.** Both move the ratio by more than most code
changes do. The prompt decides acceptance, and the host decides which kernel
each quantized projection dispatches to and therefore what a verify block
costs, which is why the same pairing can pay on one generation and regress on
another. A row without both cannot be reproduced or compared, and rows from
different protocols do not belong in the same table.

 The two rows differ in nothing else, and the ratio moves
by half again between them, because acceptance is a property of how predictable
the continuation is. A speculative-decoding figure without its prompt cannot be
reproduced or compared.

The block width is not a tuning knob worth much either. Measured on the code
row, throughput peaks at width 5 and falls at 6, 8, 10 and 12: tokens emitted
per round saturate near `1 / (1 - acceptance)` while the verify forward keeps
getting more expensive, and the adaptive controller already lands on the peak.

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
`qmv_wide`, measures 93.2 tok/s instead of 121.8, or 2.14x instead of 2.80x.

### Qwen 3.5 / 3.6 / 3.8 with the model's own MTP head

Qwen ships the MTP head as part of the family rather than as a companion
checkpoint, either split out (`Qwen3.8-27B-MTP-bf16`, `-4bit`) or carried inside
the target. Same protocol, `qwen3.8-27b-4bit` with `qwen3.8-27b-mtp-4bit`, the
code prompt above, 300 tokens:

| Host | Path | decode tok/s | speedup |
| --- | --- | ---: | ---: |
| M5 Max (128 GB) | classic decode (no drafter) | 32.5 | 1.00x |
| M5 Max (128 GB) | MTP, block 3 (the drafter's declared width) | 47.0 | **1.45x** |

Two things about that ratio. It is measured **with** the byte-identity
guarantee: the exactness probe fires on this host and drops `qmv_wide`, which
costs the verify forward about 17 to 20%. And the block width is genuinely
optimal at 3 to 4 rather than merely default: 48 of the target's 64 layers are
GatedDeltaNet, a recurrence that processes tokens in sequence, so the verify
cost grows nearly linearly with the block instead of amortising the way an
attention-only target's does. Widths 5, 6 and 8 measure 48.0, 46.8 and 35.7.

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
