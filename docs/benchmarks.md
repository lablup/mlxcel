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
- raw per-model prefill and decode throughput where available.

Averages are useful only after the raw rows are available. Avoid statements such
as "faster than X" unless the comparable model set and exclusions are explicit.

## README reference snapshot

The root README currently summarizes the latest Apple Silicon reference
snapshot:

- Date: 2026-05-18 full sweep, with selected targeted rows refreshed on 2026-05-19
- Hardware: MacBook Pro M5 Max, 128GB
- Runtime: mlxcel 0.0.28
- MLX: 0.31.2
- Baselines: mlx-lm 0.31.3 (`ed1fca4`) and mlx-vlm 0.4.4
- Scope: 98 text model directories plus a separate 98-entry VLM prompt pass
- Comparable text pairs: 67; average mlxcel/mlx-lm decode throughput 95%, median 97%, with 50 / 67 rows at or above 90% parity
- Comparable VLM pairs: 17; average mlxcel/mlx-vlm decode throughput 94%, median 95%, with 12 / 17 rows at or above 90% parity

An earlier 2026-05-08 M1 Ultra snapshot reported about 119% of the mlx-lm
baseline across 37 comparable 4-bit text checkpoints. That number is kept only
as historical context; use the current raw rows and rerun the suite on the
target hardware before relying on README numbers for release notes or capacity
planning.

## Suggested benchmark commands

The repository contains benchmark helper scripts under `scripts/`. The exact
arguments may evolve, so inspect each script before publishing results.

```bash
# Single-model decode benchmark shape.
./scripts/bench_decode.sh -m models/<checkpoint> --runs 3

# Multi-model suite shape.
./scripts/bench_all_models.sh --hardware <name> --cooldown 45 --big-cooldown 60
```

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
