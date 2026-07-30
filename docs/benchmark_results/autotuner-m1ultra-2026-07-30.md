# Shape-bucketed kernel autotuner: Apple M1 Ultra, 2026-07-30

Validation run for issue #906 (epic #909). Covers the autotuner outcome gate and
the cold last-level-cache benchmark methodology.

## Environment

| Field | Value |
|---|---|
| Hardware | Apple M1 Ultra, 20 cores (16P + 4E), 128 GB unified memory |
| Last-level cache | 96 MiB (100663296 bytes), detected by device family |
| OS | macOS 26.5.2 (Darwin 25.5.0) |
| Backend | Metal (`--features metal,accelerate`) |
| mlxcel | 0.4.3, commit `64a1dfd0` (branch `feature/issue-906-shape-bucketed-autotuner`, base `14412c13`) |
| MLX pin | `b7c3dd6d27f45b5365b08a840310187dc503f1db` |
| rustc | 1.93.1 (01f6ddf75 2026-02-11) |
| Build | `cargo build --release --features metal,accelerate` |

**No CUDA hardware was available for this run.** The qmm CTA tile and multirow-qmv
row-window tuners ship wired but unvalidated: their kernels live under
`src/lib/mlx-cpp/patches/mlx/backend/cuda/`, which a Metal build never compiles.
They need a GB10 compile and re-run before any claim is made about them.

## Measurement caveat: the host was not idle

This is recorded first because it bounds everything below. During these runs the
machine carried a load average of 12.8 (1 min) to 18.0 (15 min) from unrelated
concurrent workloads on the same box. Timings here are therefore upper bounds
with real dispersion, not quiet-machine figures.

The practical consequence is a floor on what is measurable. Short-context cells
of the paged-decode matrix are dominated by graph construction and dispatch
rather than kernel work, and under this load their samples spread 5% to 19% per
candidate. Differences smaller than that are not resolvable on this host, and the
tuner correctly declines to act on them. The long-context cells are a different
regime: their samples spread 1% to 5% even under the same load, and their result
reproduced across seven independent sweeps.

Read the long-context rows as the result of this report. Read the short-context
rows as "no signal here", not as "no difference exists".

## Autotuner outcome gate

Command, repeated seven times across two independent sets:

```
mlxcel tune --op paged-decode-splits
```

Matrix: `head_dim=128 q_heads=32 kv_heads=8 block_size=32`, batch {1,2,4,8} x
context {1024, 4096, 16384}. Profiling uses a wall-clock warmup budget, cost-scaled
repetitions (median of N, N >= 5, typically 70 to 600), round-robin candidate
interleaving, and a switch rule requiring the win to clear both a 2% floor and the
combined measured spread of candidate and default.

The shipped default is `num_splits = min(28672 / (head_dim * 4), 32)`, which at
`head_dim=128` saturates the threadgroup-memory ceiling at 32.

| batch | context | selected | speedup vs default | candidate spread | verdict |
|---|---|---|---|---|---|
| 1 | 1024 | default 32 | 1.000x | 4.5% - 8.5% | no measurable win |
| 1 | 4096 | default 32 | 1.000x | 10.0% - 17.1% | no measurable win |
| 1 | 16384 | **16** | **1.218x - 1.230x** | 2.7% - 3.7% | win |
| 2 | 1024 | default 32 | 1.000x | 4.5% - 14.8% | no measurable win |
| 2 | 4096 | default 32 | 1.000x | 7.9% - 16.9% | no measurable win |
| 2 | 16384 | **16** | **1.200x - 1.219x** | 3.8% - 4.7% | win |
| 4 | 1024 | default 32 | 1.000x | 3.5% - 5.6% | no measurable win |
| 4 | 4096 | default 32 | 1.000x | 5.7% - 16.5% | no measurable win |
| 4 | 16384 | **16** | **1.206x - 1.215x** | 2.9% - 3.3% | win |
| 8 | 1024 | default 32 | 1.000x | 3.4% - 7.9% | no measurable win |
| 8 | 4096 | default 32 | 1.000x | 5.2% - 18.9% | no measurable win |
| 8 | 16384 | **16** | **1.151x - 1.162x** | 1.1% - 4.1% | win |

The gate the issue mandates ("the autotuned tactic must be >= the current
manual/default configuration on the tuning matrix") holds: no cell selected a
tactic slower than the default, and cells where nothing beat the default keep it.
`tests/autotuner_outcome.rs` asserts this property directly on a reduced GPU
matrix and passes.

### What the long-context result means

Taking the threadgroup-memory maximum is the wrong choice at long context, by a
consistent 15% to 23%. The v1 kernel's `NumSplits` is the SIMD-group count that
stripes the KV range within one threadgroup, and the shipped formula picks the
largest value the 28 KB threadgroup-memory budget allows, without reference to
how many tokens there are to stripe. At 16384 tokens, 32 stripes oversubscribe
the reduction: each SIMD group's online-softmax partial is smaller, and the
cross-group combine at the end costs more than the extra parallelism returns.
Sixteen is the better split at every batch size measured.

This is exactly the class of decision the issue argues a tuner should own rather
than a static formula, and it is the strongest available evidence that the
autotuner earns its place, since it is the one consumer that could be validated
on this hardware at all.

### Determinism

Repeated sweeps converge. Across seven runs every 16384 cell selected
`num_splits=16`, with one exception: a single run measured `batch 1 / ctx 16384`
at 9.5% spread (against 2.7% to 3.7% in every other run) and the variance-aware
guard declined to switch, falling back to the default. That is the guard working
as designed. It fails toward the configuration that ships today, so a noisy host
degrades the tuner to current behavior rather than to a coin flip.

Short-context selections are stable at "keep the default" across all seven runs,
but stable for a weak reason: the guard is far wider than any candidate's apparent
win there, so it absorbs the noise rather than the noise being absent.

## Cold last-level-cache methodology

Command:

```
page_gather_microbench --batch-sizes 1,4 --context-lengths 1024,4096,16384,32768 \
    --block-sizes 32 --warmup 20 --iters 50 [--cold-l2]
```

Rotation sizing behaves as designed. Rotation count falls as the working set grows
(48 at b1/ctx1024, 12 at b1/ctx4096, 3 at b1/ctx16384, 2 at b1/ctx32768) and reaches
1 once the working set exceeds the 96 MiB cache, at which point the harness reports
`warm` because rotation cannot change what is already uncacheable. On this machine
that crossover lands at batch 4 / ctx 16384, one step earlier than
`docs/benchmarks.md` estimated.

### Warm versus cold delta, five repetitions each

Cell: batch 1, ctx 4096, block 32, rotation 12. Medians of five runs.

| Path | Warm median | Cold median | Delta |
|---|---|---|---|
| `contig_sdpa` | 438.3 us | 433.7 us | -1.0% |
| `gatherA_sdpa` | 509.4 us | 535.0 us | +5.0% |

The latency delta is small, smaller than the framing in `docs/benchmarks.md`
implies. The reproducible effect is on **dispersion**, not on the median:

| Path | Warm range | Cold range |
|---|---|---|
| `gatherA_sdpa` | 425.6 - 542.1 us (27%) | 525.1 - 546.0 us (4.0%) |

Warm mode's first repetition was a 708 us outlier; cold mode produced no
comparable outlier. Rotating the inputs removes the run-to-run lottery of partial
cache residency, which is what makes a number worth recording, even though on this
unified-memory part it does not move the median far.

Do not generalize the small median delta to CUDA. A discrete GPU with a private
L2 and a PCIe-attached host has a much sharper cache cliff than Apple's unified
memory, and the same rotation on that hardware should be expected to move the
median substantially more.

## Reproduction

```
export DEVELOPER_DIR=/Applications/Xcode-<version>.app/Contents/Developer
cargo build --release --features metal,accelerate --bin mlxcel --example page_gather_microbench
./target/release/mlxcel tune --op paged-decode-splits
./target/release/examples/page_gather_microbench --batch-sizes 1 --context-lengths 4096 \
    --block-sizes 32 --warmup 20 --iters 50 --cold-l2
```

`DEVELOPER_DIR` is required whenever `xcode-select -p` points at
`/Library/Developer/CommandLineTools`, which carries no Metal compiler.

## Open items

- qmm CTA tile and multirow-qmv tuners are unvalidated pending GB10 access. The
  `qmv.cu` change has not been compiled by any toolchain.
- The `kv_chunk_size` consumer for the fused paged decode v2 kernel is a documented
  seam only; it lands with issue #898.
- Short-context cells need a quiet host to say anything. Re-run this matrix on an
  unloaded machine before concluding that no split-count win exists below 16384.
