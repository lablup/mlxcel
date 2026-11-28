# Benchmark decision archive — unreleased

[Benchmark index](../README.md)

This groups records by the first stable release tag containing the report, verified from Git trees, not by filename dates. It is **not** a benchmark of that release binary. Each linked frozen record preserves the actual measured commits (including dirty/branch builds), hardware, model, flags, repetitions, dispersion, attribution and unmeasured cells. Later corrections present at the consolidation snapshot are included; decisions describe the experiment, not necessarily today's defaults. `unreleased` means no stable tag yet contains the report.

<a id="dflash-per-row-accepts-m3ultra-2026-11-06"></a>

## Batched DFlash per-row accepts on M3 Ultra, 2026-11-06

B=4 per-row DFlash acceptance raised median throughput from 83.06 to 116.05 tok/s (1.397x), with arm ranges spanning 0.2% and 0.36%. Rounds fell from 96 to 67 and committed accepts per row-round rose from 0.323 to 1.036. The drafter still sees the shortest shared prefix, so its per-row context lag remains a separate limitation.

[Full record and reproduction](https://github.com/lablup/mlxcel/blob/2dc8ed47b30dc204d892328098e8cc34ec02ddc1/docs/benchmark_results/dflash-per-row-accepts-m3ultra-2026-11-06.md); [#2442](https://github.com/lablup/mlxcel/issues/2442), [#2353](https://github.com/lablup/mlxcel/issues/2353), [#2440](https://github.com/lablup/mlxcel/issues/2440), [#2851](https://github.com/lablup/mlxcel/issues/2851).

<a id="expert-streaming-sync-cost-m3ultra-2026-11-10"></a>

## Per-layer host synchronisation cost of SSD expert streaming (issue #2882)

A sync-only experiment on resident Qwen3.8-Flash-Next weights lowered median decode from 42.0 to 25.6 tok/s: about 15.3 ms/token, or 0.32 ms/layer, without any I/O. Token IDs matched across eight runs. A sibling build and load 9–13 limit absolute interpretation. Keep the graph-based streaming mode as a capacity fallback; the later quiet-host campaign measures actual arena/I/O behavior.

[Full record and reproduction](https://github.com/lablup/mlxcel/blob/2dc8ed47b30dc204d892328098e8cc34ec02ddc1/docs/benchmark_results/expert-streaming-sync-cost-m3ultra-2026-11-10.md); [#2882](https://github.com/lablup/mlxcel/issues/2882).

Related evidence / subsequent correction: [qwen4-exp-deferred-measurements-m3ultra-2026-11-10.md](../qwen4-exp-deferred-measurements-m3ultra-2026-11-10.md).

<a id="fused-gelu-tanh-f32-m1pro-2026-11-03"></a>

## Fused tanh-GELU f32: measure-then-keep evidence

The fused f32 tanh-GELU microbenchmark beat the compiled chain on M1 Pro, with first-sweep fused/compiled ratios 0.77–0.87 across three shapes; repeat sweeps ranged 0.71–0.97. These are minima of seven interleaved trials and op-level evidence only, not a model-level speedup. The source preserves all three sweeps and their differing margins.

[Full record and reproduction](https://github.com/lablup/mlxcel/blob/2dc8ed47b30dc204d892328098e8cc34ec02ddc1/docs/benchmark_results/fused-gelu-tanh-f32-m1pro-2026-11-03.md); [#905](https://github.com/lablup/mlxcel/issues/905).

<a id="switch-gate-up-batch-m3ultra-2026-11-04"></a>

## Switch gate/up kernel batch decode A/B on M3 Ultra, 2026-11-04

The corrected M3 Ultra A/B removed the multi-token switch gate/up attempt; keep only independently supported single-token wiring. Earlier concurrency-5–8 rows were queued behind four active rows, not those batch widths. M4/M5 remain unmeasured and decline at B>=2; a positive future multi-token result must re-establish its own geometry and attribution.

[Full record and reproduction](https://github.com/lablup/mlxcel/blob/2dc8ed47b30dc204d892328098e8cc34ec02ddc1/docs/benchmark_results/switch-gate-up-batch-m3ultra-2026-11-04.md); [#2515](https://github.com/lablup/mlxcel/issues/2515), [#268](https://github.com/lablup/mlxcel/issues/268).
