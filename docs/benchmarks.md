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
```

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
classic decode, so the only metric that moves is decode throughput; confirm
correctness by diffing the two completions.

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

### Gemma 4 Unified (12B) + 4-bit assistant

Measured on Apple M5 Max (128 GB) with `mlx-community/gemma-4-12b-it-4bit` as the
target and `mlx-community/gemma-4-12B-it-assistant-4bit` as the drafter, block
size 4, `temperature 0`, 200 decode tokens:

| Path                        | decode tok/s | speedup |
| --------------------------- | -----------: | ------: |
| classic decode (no drafter) |         ~39  |  1.00x  |
| MTP                         |         ~74  | ~1.87x  |

The accelerated output is byte-identical to classic decode. B=1 (single-request)
MTP runs by default for every MTP target; the Gemma 4 Unified target cannot batch
at all, so B=1 is also its only decode path. The batch-capable 31B + bf16
assistant measures ~1.2 to 1.4x on the same host. Set `MLXCEL_ENABLE_MTP_B1=0` to
opt out on hardware where the B=1 verify forward does not pay for itself.

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
exact: the policy only chooses when to run it, so temperature-0 output is still
byte-identical to classic decode. `MLXCEL_ENABLE_MTP_B1` pins the decision in
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
