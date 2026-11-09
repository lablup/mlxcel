# macOS memory guardrails

How mlxcel behaves when a model takes most of unified memory on a Mac, and which switches control it (issue #2883).

## Why total RAM is the wrong number

GPU wiring of a buffer ends a few seconds after its last command buffer. While a server sits idle, macOS may compress or swap the model's memory when anything else wants memory, and the next request pays for it. Total RAM and Metal's recommended working set say what a machine could hold, not what it can give now without compressing.

## Available now and the reserve

`mlxcel inspect` and `--estimate-memory` print, on macOS only:

| Line | Meaning |
|------|---------|
| Total RAM | `hw.memsize` |
| Metal working set | Metal's recommended working set |
| Available now | `free + purgeable + min(inactive + speculative, file-backed)` pages from `host_statistics64`, with `free` excluding speculative pages (the kernel's `free_count` includes them). Inactive anonymous memory of other programs is excluded, because macOS frees it only by compressing or swapping. |
| macOS reserve | `max(7 GiB, 15% of RAM)` |
| This load | weights + KV cache + activations |

When the load exceeds available-now minus the reserve, a line starting `WARNING: memory available now` gives the figures. It never refuses a load and does not change the `FITS` verdict.

The figure and the reserve are in `mlxcel_core::host_memory` (`host_memory_status`, `HostMemoryStatus`, `AvailabilityGate`) for any store that grows on demand.

## Footprint reporting

After a load the server logs `Process footprint after load`. With `--metrics`, `/metrics` adds these gauges on macOS:

| Metric | Meaning |
|--------|---------|
| `mlxcel_process_phys_footprint_bytes` | `phys_footprint` of the process |
| `mlxcel_process_compressed_bytes` | Bytes of the process in the compressor or swap. Growth while idle means idle compression. |
| `mlxcel_mlx_active_bytes`, `mlxcel_mlx_cache_bytes` | MLX allocator counters |
| `mlxcel_weights_wired_bytes` | Bytes wired by `MLXCEL_WIRE_WEIGHTS` at the last load |

## Switches

- `MLXCEL_WIRED_LIMIT` is unchanged (default `max`).
- `MLXCEL_WIRE_WEIGHTS=1` wires the evaluated weights with `mlock` after load, macOS only, default off. Weights are wired in 1 GiB chunks. Before the first chunk and after each one the system wired total is compared with RAM minus the reserve, and a chunk that crosses it is unwired and the pass stops. The result, including failures, is logged. It needs the default eager `MLXCEL_WEIGHT_MATERIALIZATION`. Multimodal towers are not wired, and a repack that runs at the first token (expert stacking, dtype conversion) replaces wired sources with unwired results.

## Growing stores

The external PLE row cache of `qwen4_exp` checks availability before each 64 MiB growth step and stops growing, evicting instead, rather than push macOS below the reserve. The check reuses one query for 250 ms. The prompt-cache stores are not gated: an entry's arrays already exist when it is inserted and the store is bounded by a byte budget, so there is no growth step to refuse.

The prompt-cache snapshot store grows the same way: while it is below its byte budget, each snapshot insert that would add to its resident size re-checks availability against the reserve and is skipped instead of stored when the host cannot spare it. A store at its budget only replaces entries, so it is never refused.

## What was measured

Nothing yet. No soak has been run, and these guardrails are not adopted as defaults on evidence: `MLXCEL_WIRE_WEIGHTS` stays off, `MLXCEL_WIRED_LIMIT` keeps its default, and no load is refused. To measure on a quiet machine:

```bash
cargo build --release --features metal,accelerate
scripts/macos_memory_soak.sh models/Qwen3.8-Flash-Next-OptiQ-2bit 40G
scripts/macos_memory_soak.sh --wire-weights --wired-limits default models/Qwen3.8-Flash-Next-OptiQ-2bit 40G
```

The first command covers `MLXCEL_WIRED_LIMIT` at its default, at `none` and at `60G`, and writes one CSV per setting plus a `summary.md` with time to first token before and after the write under `docs/benchmark_results/`. `--dry-run` prints the plan without loading a model or writing a file, `--allocate 16G` holds an allocation beside the write. On a 64 GB Mac use a proportionally smaller model and a lower explicit limit, for example `--wired-limits default,none,30G`.
