# Benchmark results

Keep results and decisions, not the history of every attempt. See [methodology and retention rules](../benchmarks.md#result-retention-and-release-consolidation).

## Final reports

Reports keep their own measured versions, workload definitions and caveats. They are not all measurements of the latest release; do not combine their rates without matching conditions. Historical decode-rate definitions are preserved, not silently recalculated.

- [Cross-hardware navigation and compatibility](model_tests.md)
- [Historical Apple Silicon/reference-runtime comparison (2026-05-19)](benchmark-report.md)
- [GB10 text and VLM final sweep](model_tests_gb10.md)
- [M1 Ultra text and VLM final sweep](model_tests_m1ultra.md)
- [M5 Max text and VLM final sweep](model_tests_m5max.md)
- [Strix Halo text, VLM and serving sweep](model_tests_strixhalo-gfx1151.md)
- [GB10 embeddings/rerank](embeddings-rerank-gb10-2026-11-04.md)
- [M1 Ultra embeddings/rerank](embeddings-rerank-m1ultra-2026-09-09.md)
- [M5 Max embeddings/rerank](embeddings-rerank-m5max-2026-09-09.md)
- [GB10 image generation/editing](imagegen-gb10-2026-11-04.md)
- [M3 Ultra final native-batching qualification](native-batched-decode-m3ultra-2026-10-27.md)
- [M1 Ultra final window-SDPA routing evidence](native-window-sdpa-m1ultra-2026-10-26.md)
- [M3 Ultra Qwen3.8-Flash-Next qualification](qwen4-exp-m3-ultra-2026-10-23.md)
- [M3 Ultra final MTP/streaming/memory acceptance](qwen4-exp-deferred-measurements-m3ultra-2026-11-10.md)
- [ROCm OpenXLA/IREE qualification and boundaries](xla_iree_rocm_qualification.md)

## Records with explicit validation gaps

These remain standalone because their reports leave a substantive measurement gate unfilled. This list describes the evidence in the records, not the current GitHub issue status.

- [MLA absorbed decode](mla-absorbed-decode-m1ultra-2026-08-02.md): real-checkpoint token parity was not available; default-on is not justified by synthetic memory savings.
- [Qwen3.8 snapshot capacity](qwen38-snapshot-capacity-default-2026-08-30.md): sizing unit tests are not a rerun of the Apple Silicon multi-turn benchmark.
- [ROCm flash prefill](sdpa-flash-prefill-gfx1151-2026-10-13.md): kernel timings exist, but model-level before/after was not measured.
- [Fused tanh-GELU f32 on CUDA](fused-gelu-tanh-f32-cuda-gb10-2026-11-11.md): the `xla-iree` StableHLO twin re-check was not run on the measuring host, and the new tests were not run on Metal.

## Release decision archives

One concise file per release containing completed investigations, including negative/null results, rejected alternatives and superseding corrections. Placement is the **first stable Git tag containing the original report**, not the filename date and not a claim about the measured binary's version. Actual experiment provenance remains in the frozen full record linked from every section. Reports absent from all stable tags are in `unreleased.md`. No empty files are created for releases with no archived investigations.

- [v0.1.4](versions/v0.1.4.md) — 1 investigation.
- [v0.3.0](versions/v0.3.0.md) — 2 investigations.
- [v0.3.2](versions/v0.3.2.md) — 3 investigations.
- [v0.4.0](versions/v0.4.0.md) — 18 investigations.
- [v0.5.0](versions/v0.5.0.md) — 11 investigations.
- [v0.5.1](versions/v0.5.1.md) — 2 investigations.
- [v0.5.2](versions/v0.5.2.md) — 4 investigations.
- [v0.6.0](versions/v0.6.0.md) — 4 investigations.
- [v0.7.0](versions/v0.7.0.md) — 8 investigations.
- [v0.8.0](versions/v0.8.0.md) — 84 investigations.
- [v0.8.1](versions/v0.8.1.md) — 1 investigation.
- [unreleased](versions/unreleased.md) — 4 investigations.

## Retained measurement evidence

The ordinary model/embedding/image sweeps also retain their CSVs under [`benchmarks/`](../../benchmarks/), linked from the final reports. Below are the smaller evidence sets kept with this directory. CSVs preserve observations, including warmup/status fields where present; do not silently filter or pool them. Compact legacy JSON preserves nested correctness/provenance fields rather than forcing them into a lossy table. Paths to retired captures inside those unchanged JSON objects refer to the frozen tree below.

- [data/native-batched-decode-m3ultra-2026-10-27/gpt-oss-20b.json](data/native-batched-decode-m3ultra-2026-10-27/gpt-oss-20b.json)
- [data/native-batched-decode-m3ultra-2026-10-27/linear-attention.json](data/native-batched-decode-m3ultra-2026-10-27/linear-attention.json)
- [data/native-batched-decode-m3ultra-2026-10-27/mamba2-families.json](data/native-batched-decode-m3ultra-2026-10-27/mamba2-families.json)
- [data/native-batched-decode-m3ultra-2026-10-27/mixed-cache-families.json](data/native-batched-decode-m3ultra-2026-10-27/mixed-cache-families.json)
- [data/native-batched-decode-m3ultra-2026-10-27/model-owned-attention.json](data/native-batched-decode-m3ultra-2026-10-27/model-owned-attention.json)
- [data/native-batched-decode-m3ultra-2026-10-27/recurrent-families.json](data/native-batched-decode-m3ultra-2026-10-27/recurrent-families.json)
- [data/qmm-dequant-lru-off-gfx1151-2026-10-13/decode.csv](data/qmm-dequant-lru-off-gfx1151-2026-10-13/decode.csv)
- [data/rocm-memory-gfx1151-2026-09-30/harness-bounds-off.csv](data/rocm-memory-gfx1151-2026-09-30/harness-bounds-off.csv)
- [data/rocm-memory-gfx1151-2026-09-30/harness-defaults.csv](data/rocm-memory-gfx1151-2026-09-30/harness-defaults.csv)
- [data/rocm-moe-prefill-right-sorted-gfx1151-2026-10-09/moe_prefill.csv](data/rocm-moe-prefill-right-sorted-gfx1151-2026-10-09/moe_prefill.csv)
- [data/rocm-unified-engine-gfx1151-2026-10-08/engine/bench_engine_rounds.csv](data/rocm-unified-engine-gfx1151-2026-10-08/engine/bench_engine_rounds.csv)
- [data/rocm-unified-engine-gfx1151-2026-10-08/gemma3-pp2048/gemma3_pp2048.csv](data/rocm-unified-engine-gfx1151-2026-10-08/gemma3-pp2048/gemma3_pp2048.csv)
- [data/rocm-unified-engine-gfx1151-2026-10-08/moe-prefill/moe_prefill.csv](data/rocm-unified-engine-gfx1151-2026-10-08/moe-prefill/moe_prefill.csv)
- [data/rocm-unified-engine-gfx1151-2026-10-08/rounds/rounds.csv](data/rocm-unified-engine-gfx1151-2026-10-08/rounds/rounds.csv)
- [data/switch-gate-up-batch-m3ultra-2026-11-04/decode.csv](data/switch-gate-up-batch-m3ultra-2026-11-04/decode.csv)
- [data/switch-gate-up-batch-m3ultra-2026-11-04/focus-parallel4.csv](data/switch-gate-up-batch-m3ultra-2026-11-04/focus-parallel4.csv)
- [data/wrapped-ring-lookahead-gfx1151-2026-10-12/nofix-pp2048/rounds.csv](data/wrapped-ring-lookahead-gfx1151-2026-10-12/nofix-pp2048/rounds.csv)
- [data/wrapped-ring-lookahead-gfx1151-2026-10-12/pp512-gfx1151-2026-10-12/rounds.csv](data/wrapped-ring-lookahead-gfx1151-2026-10-12/pp512-gfx1151-2026-10-12/rounds.csv)
- [data/wrapped-ring-lookahead-gfx1151-2026-10-12/rounds.csv](data/wrapped-ring-lookahead-gfx1151-2026-10-12/rounds.csv)
- [fused-moe-decode-dff-sweep-gb10-2026-07-03.csv](fused-moe-decode-dff-sweep-gb10-2026-07-03.csv)
- [gemma4-12b-mtp-exactness-m1-ultra-2026-10-15.json](gemma4-12b-mtp-exactness-m1-ultra-2026-10-15.json)
- [gemma4-31b-mtp-exactness-2026-09-26.json](gemma4-31b-mtp-exactness-2026-09-26.json)
- [macos-memory-soak-lablup-macstudio-02-2026-11-10/samples-60G.csv](macos-memory-soak-lablup-macstudio-02-2026-11-10/samples-60G.csv)
- [macos-memory-soak-lablup-macstudio-02-2026-11-10/samples-default.csv](macos-memory-soak-lablup-macstudio-02-2026-11-10/samples-default.csv)
- [macos-memory-soak-lablup-macstudio-02-2026-11-10/samples-none.csv](macos-memory-soak-lablup-macstudio-02-2026-11-10/samples-none.csv)
- [macos-memory-soak-lablup-macstudio-02-2026-11-10/summary.md](macos-memory-soak-lablup-macstudio-02-2026-11-10/summary.md)
- [macos-memory-soak-lablup-macstudio-02-2026-11-10-wire-weights/samples-default.csv](macos-memory-soak-lablup-macstudio-02-2026-11-10-wire-weights/samples-default.csv)
- [macos-memory-soak-lablup-macstudio-02-2026-11-10-wire-weights/summary.md](macos-memory-soak-lablup-macstudio-02-2026-11-10-wire-weights/summary.md)
- [native-window-sdpa-m1ultra-2026-10-26-full-model.csv](native-window-sdpa-m1ultra-2026-10-26-full-model.csv)
- [native-window-sdpa-m1ultra-2026-10-26-microbench.csv](native-window-sdpa-m1ultra-2026-10-26-microbench.csv)
- [qwen4-exp-2026-10-23/mlxcel-128.csv](qwen4-exp-2026-10-23/mlxcel-128.csv)
- [qwen4-exp-2026-10-23/mlxcel-4608.csv](qwen4-exp-2026-10-23/mlxcel-4608.csv)
- [qwen4-exp-2026-10-23/mlxcel-dense.tsv](qwen4-exp-2026-10-23/mlxcel-dense.tsv)
- [qwen4-exp-2026-10-23/mlxcel-sparse.tsv](qwen4-exp-2026-10-23/mlxcel-sparse.tsv)
- [qwen4-exp-2026-10-23/reference-128.csv](qwen4-exp-2026-10-23/reference-128.csv)
- [qwen4-exp-2026-10-23/reference-4608.csv](qwen4-exp-2026-10-23/reference-4608.csv)
- [qwen4-exp-2026-10-23/reference-dense.tsv](qwen4-exp-2026-10-23/reference-dense.tsv)
- [qwen4-exp-2026-10-23/reference-sparse.tsv](qwen4-exp-2026-10-23/reference-sparse.tsv)

- [data/ssm-bf16-parity-gb10-2026-10-11/granite-parity/comparison.json](data/ssm-bf16-parity-gb10-2026-10-11/granite-parity/comparison.json)

## Recovering a retired record or capture

The pre-consolidation snapshot is [`2dc8ed47`](https://github.com/lablup/mlxcel/tree/2dc8ed47b30dc204d892328098e8cc34ec02ddc1/docs/benchmark_results/). Every archive section links to its exact original report there, whose relative links also resolve against that commit. A historical GitHub issue/PR link using `blob/main` may no longer resolve: replace `main` with `2dc8ed47b30dc204d892328098e8cc34ec02ddc1`, or find the report's basename in `versions/`. No GitHub discussion bodies were rewritten.

For an offline reproduction, restore the old report and its campaign into a separate temporary directory, not into the current result tree:

```sh
REPRO=$(mktemp -d)
git archive 2dc8ed47b30dc204d892328098e8cc34ec02ddc1 docs/benchmark_results | tar -x -C "$REPRO"
```

This is recovery from existing Git history, not a new compressed artifact. Deleting captures reduces checkout/search noise; it does not reduce already-published Git history size.

## Maintenance

- During development, retain the measured commit, host/model, exact command/flags, arm-attribution evidence, repetitions and dispersion, decision and unmeasured cells.
- At release, summarize completed experiments into the release archive, update incoming links and keep only necessary final evidence. An inconclusive measurement is not a speedup; a rejected optimization is still a useful decision.
- Keep raw output in ignored `bench-results/`. Maintained harnesses belong in `scripts/`; do not import tools from result folders. The former host gate now lives in `scripts/bench_hostgate.py`.
- Run `make verify-benchmark-raw-logs`, check local links/anchors and verify retained CSV/JSON evidence before committing cleanup. Preserve original metric definitions and corrections.
