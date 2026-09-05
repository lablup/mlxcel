# Model Compatibility & Performance Tests

Per-hardware benchmark results and cross-hardware comparison for mlxcel.

For a public, data-driven Apple Silicon summary that combines M1 Ultra,
M5 Max, and mlx-lm / mlx-vlm baselines, see
[Benchmark Report - 2026-05-19](benchmark-report.md).

## Per-Hardware Results

| Hardware | File | Status | Last Updated |
|----------|------|--------|-------------|
| Mac Studio M1 Ultra 128GB | [model_tests_m1ultra.md](model_tests_m1ultra.md) | Active | 2026-07-12 |
| MacBook Pro M5 Max 128GB | [model_tests_m5max.md](model_tests_m5max.md) | Active | 2026-09-03/04 at `b2ff1eee`, which is code-identical to `v0.7.0-beta.1` (cooldown-30 full text + VLM sweep, 90 GB weight budget; plus speculative, batched-serving and embedding passes). CSVs record `mlxcel_version` 0.7.0-beta.1, with `mlxcel_commit` `b2ff1eee` and `mlx_commit` `9a795735` |
| NVIDIA GB10 (DGX Spark) | [model_tests_gb10.md](model_tests_gb10.md) | Active | 2026-07-12 (mlxcel 0.4.0-rc.1, full 159-dir sweep; 7 memory-gated skips) |

## Benchmark CSVs

Current source-of-truth data lives in `benchmarks/`:

| CSV | Hardware | Date | Type |
|-----|----------|------|------|
| `metal_m5max_2026-09-03.csv` | M5 Max | 2026-09-03 (mlxcel 0.6.0, MLX pin `9a795735`, `--cooldown 30 --big-cooldown 30`, `BENCH_MEM_OVERHEAD_FACTOR=1.209` for a 90 GB weight budget; version-change full text re-benchmark, 175 dirs, 161 measured; 0 decode regressions vs 0.4.0-rc.1) | Text |
| `metal_m5max_vlm_2026-09-04.csv` | M5 Max | 2026-09-04 (mlxcel 0.6.0, MLX pin `9a795735`, same cooldowns and budget; version-change full VLM re-benchmark, 77 measured rows; Pixtral/Mistral3 image-token counts drop by design after #792) | VLM |
| `metal_m5max_spec_2026-09-04.csv` | M5 Max | 2026-09-04 (mlxcel 0.6.0; `speculative_bench --sweep --max-tokens 128`, 12 rows: 3 baselines, 3 measured Gemma 4 Unified 12B MTP rows at K=2/4/8, 3 DFlash deferred, 3 Gemma 4 31B MTP rows that the harness could not drive. That harness restriction was lifted by #1613; an M5 Max re-run against a binary carrying it has not been made) | Speculative |
| `metal_m5max_batch_2026-09-04.csv` | M5 Max | 2026-09-04 (mlxcel 0.6.0; `bench_serving_concurrency.py` at `--parallel 4 --max-batch-prefill 4`, `--prompt-tokens 512 --max-tokens 128`; 3 models x B=1/2/4, 0 failed requests) | Batch |
| `metal_m5max_embeddings_2026-09-04.csv` | M5 Max | 2026-09-04 (mlxcel 0.6.0; `bench_embeddings.py`, 18 of the 20-model roster present on disk, 90 cells, 0 failures; report in [`embeddings-rerank-m5max-2026-09-04.md`](embeddings-rerank-m5max-2026-09-04.md)) | Embeddings/Rerank |
| `metal_m5max_2026-07-12.csv` | M5 Max | 2026-07-12 (mlxcel 0.4.0-rc.1, MLX pin 57c66cac, `--cooldown 30 --big-cooldown 30`; version-change full text re-benchmark, 175 dirs, 160 measured; no code regressions) | Text |
| `metal_m5max_vlm_2026-07-12.csv` | M5 Max | 2026-07-12 (mlxcel 0.4.0-rc.1, MLX pin 57c66cac, `--cooldown 30 --big-cooldown 30`; version-change full VLM re-benchmark, 75 measured rows) | VLM |
| `metal_m5max_2026-06-15.csv` | M5 Max | 2026-06-15 (mlxcel 0.2.1, MLX pin a6ec7123; full text re-benchmark, 151 rows, 135 measured) | Text |
| `metal_m5max_vlm_2026-06-15.csv` | M5 Max | 2026-06-15 (mlxcel 0.2.1, MLX pin a6ec7123; full VLM re-benchmark, 53 measured rows) | VLM |
| `metal_m5max_2026-05-19.csv` | M5 Max | 2026-05-19 (mlxcel 0.0.28, MLX 0.31.2) | Text |
| `metal_m5max_vlm_2026-05-19.csv` | M5 Max | 2026-05-19 (mlxcel 0.0.28, MLX 0.31.2) | VLM |
| `metal_m5max_vlm_2026-05-20.csv` | M5 Max | 2026-05-20 (mlxcel 0.0.28, MLX 0.31.2; Gemma3n + Molmo v1 + Phi-3.5 vision + Gemma3 4B VLM entries) | VLM |
| `pylm_m5max_2026-05-18.csv` | M5 Max | 2026-05-19 benchmark campaign (mlx-lm 0.31.3 baseline; CSV date crossed midnight) | Text |
| `pylm_m5max_vlm_2026-05-18.csv` | M5 Max | 2026-05-19 benchmark campaign (mlx-vlm 0.4.4 baseline; CSV date crossed midnight) | VLM |
| `metal_m1ultra_spec_2026-09-04.csv` | M1 Ultra | 2026-09-04 (mlxcel 0.7.0-beta.1, MLX pin `9a795735`; `speculative_bench --sweep --batch 1 --max-tokens 128`, 16 rows: 4 baselines, 9 measured MTP rows at K=2/4/8 across Gemma 4 31B, Gemma 4 Unified 12B and Qwen 3.8 27B, 3 DFlash deferred; first sweep with #1613's per-variant MTP dispatch) | Speculative |
| `metal_m1ultra_2026-07-12.csv` | M1 Ultra | 2026-07-12 (mlxcel 0.4.0-rc.1, MLX pin 57c66cac, `--cooldown 30`; version-change full text re-benchmark, 168 rows; decode median 98% vs mlx-lm, flat vs 0.3.3, nvfp4/minicpm-mxfp4 recovered) | Text |
| `metal_m1ultra_vlm_2026-07-12.csv` | M1 Ultra | 2026-07-12 (mlxcel 0.4.0-rc.1, MLX pin 57c66cac, `--cooldown 30`; version-change full VLM re-benchmark, 168 rows) | VLM |
| `metal_m1ultra_2026-07-12_cooldown0.csv` + `_vlm_` | M1 Ultra | 2026-07-12 (mlxcel 0.4.0-rc.1; `--cooldown 0` pass of the same build, kept for the thermal-offset comparison: decode reads ~2% lower than the cooldown-30 canonical) | Text/VLM |
| `metal_m1ultra_2026-07-06.csv` | M1 Ultra | 2026-07-06 (mlxcel 0.3.3; full text re-benchmark post VLM-port batch #660-#664 and fixes #666-#668/#671, 169 rows) | Text |
| `metal_m1ultra_vlm_2026-07-06.csv` | M1 Ultra | 2026-07-06 (mlxcel 0.3.3; full VLM re-benchmark, 14 new VLM families measured) | VLM |
| `pylm_m1ultra_*2026-07-06*_single_*.csv` | M1 Ultra | 2026-07-06 (mlx-lm 0.31.3 / mlx-vlm dev; per-model python baselines for 21 newly added models, 13 measured / 8 python-side FAIL) | Baselines |
| `metal_m1ultra_2026-06-15.csv` | M1 Ultra | 2026-06-15 (mlxcel 0.2.1, MLX pin a6ec712; full text re-benchmark post #289 fix, 151 rows) | Text |
| `metal_m1ultra_vlm_2026-06-15.csv` | M1 Ultra | 2026-06-15 (mlxcel 0.2.1, MLX pin a6ec712; full VLM re-benchmark, 55 measured rows) | VLM |
| `metal_m1ultra_2026-06-15_pre289_regressed.csv` | M1 Ultra | 2026-06-15 (mlxcel pre-#290; bf16-scale decode regression evidence sweep) | Text |
| `metal_m1ultra_2026-06-12.csv` | M1 Ultra | 2026-06-12 (mlxcel 0.1.4, MLX pin a6ec712; full text re-benchmark, 121 rows) | Text |
| `metal_m1ultra_vlm_2026-06-12.csv` | M1 Ultra | 2026-06-12 (mlxcel 0.1.4, MLX pin a6ec712; full VLM re-benchmark, 49 measured rows) | VLM |
| `metal_m1ultra_2026-05-19.csv` | M1 Ultra | 2026-05-19 (mlxcel 0.0.28, MLX commit 84961223; >65GB skipped) | Text |
| `metal_m1ultra_vlm_2026-05-19.csv` | M1 Ultra | 2026-05-19 (mlxcel 0.0.28, MLX commit 84961223; >65GB skipped) | VLM |
| `pylm_m1ultra_2026-05-19.csv` | M1 Ultra | 2026-05-19 (mlx-lm 0.31.3 baseline, https://github.com/ml-explore/mlx-lm @ `df1d3f3`; >65GB skipped) | Text |
| `pylm_m1ultra_vlm_2026-05-19.csv` | M1 Ultra | 2026-05-19 (mlx-vlm baseline, https://github.com/Blaizzy/mlx-vlm @ `d85ca4d`; >65GB skipped) | VLM |
| `cuda_gb10_2026-07-12.csv` | GB10 | 2026-07-12 (mlxcel 0.4.0-rc.1, MLX pin `57c66cac` / 0.32.1, CUDA 13.0 / SM 12.1; full 159-dir sweep, 142 measured / 0 code failures / 7 memory-gate `SKIP:oom_estimate` at `BENCH_MEM_OVERHEAD_FACTOR=2.0` / 10 N.A. FAILs [drafters, speech/TTS, incomplete checkpoints]; 13 rows amended to the same-day post-reboot #755 singles, median-decode run where n=3, overnight originals in git history) | Text |
| `cuda_gb10_2026-07-12_postreboot_single_*.csv` | GB10 | 2026-07-12 (13 files, 21 rows; post-reboot re-measurement for #755: subjects, controls, and the SSM/hybrid cluster, `--cooldown 30`, including the n=3 repeat sets behind the amended sweep rows) | Text |
| `cuda_gb10_vlm_2026-07-12.csv` | GB10 | 2026-07-12 (mlxcel 0.4.0-rc.1; full VLM sweep over all dirs, 63 measured image rows; text-only models FAIL by design; gemma-4-31b-it-nvfp4 image-input FAIL captured here, since fixed by #749) | VLM |
| `cuda_gb10_2026-07-09.csv` | GB10 | 2026-07-09 (mlxcel 0.4.0-rc.1, MLX pin `57c66cac` / 0.32.1, CUDA 13.0 / SM 12.1; 19-model representative subset re-benchmark, 19 pass / 0 fail, includes gemma-4-31b-it-nvfp4 now functional via the ModelOpt NVFP4 direct-transcode path #692/#693/#697) | Text |
| `cuda_gb10_2026-06-17.csv` | GB10 | 2026-06-17 (mlxcel 0.3.1 [CSV relabeled; Cargo.toml 0.3.0 until release], MLX pin a6ec7123, CUDA 13.0 / SM 12.1, post-#319 CUDA fused decode-MoE; full text re-benchmark, 147 models, 136 pass / 0 fail / 9 not-tested-N.A. / 2 too-large) | Text |
| `cuda_gb10_vlm_2026-06-17.csv` | GB10 | 2026-06-17 (mlxcel 0.3.1; full VLM re-benchmark, 54 measured image rows) | VLM |
| `cuda_gb10_2026-05-28.csv` | GB10 | 2026-05-28 (full text re-benchmark, mlxcel 0.1.0, MLX commit 84961223, warm same-process harness `c9a77f2`, `--cooldown 0`; 109 models, 8 fail/skip) | Text |
| `cuda_gb10_vlm_2026-05-28.csv` | GB10 | 2026-05-28 (full VLM re-benchmark, mlxcel 0.1.0; 38 measured VLM rows, 0 image-path failures) | VLM |
| `cuda_gb10_2026-05-19.csv` | GB10 | 2026-05-19 (mlxcel 0.0.27, MLX 0.31.2) | Text |
| `cuda_gb10_vlm_2026-05-19.csv` | GB10 | 2026-05-19 (mlxcel 0.0.27, MLX 0.31.2) | VLM |

## Cross-Hardware Comparison

The table below summarizes the current cross-hardware decode readings for selected models.

### Decode Speed Summary (tok/s, selected models)

| Model | Params | M1 Ultra | M5 Max | GB10 |
|-------|--------|----------|--------|------|
| SmolLM-135M | 135M | 418.85 | 926.25 | 656.73 |
| ERNIE-4.5-0.3B | 300M | 522.31 | 1068.37 | 625.30 |
| Qwen2.5-0.5B (4bit) | 500M | 381.94 | 660.77 | 492.60 |
| Llama-3.2-1B | 1B | 421.40 | 556.36 | 266.04 |
| Qwen3-0.6B | 600M | 228.37 | 601.12 | 283.90* |
| StableLM-1.6B | 1.6B | 263.88 | 428.36 | 203.75 |
| Gemma-3-1B | 1B | 227.38 | 391.42 | 278.52 |
| EXAONE-3.5-2.4B | 2.4B | 199.11 | 284.14 | 141.83 |
| SmolLM3-3B | 3B | 131.45 | 231.61 | 104.24 |
| Nemotron-H-30B | 30B | 91.75 | 176.01 | 87.41¶ |
| Qwen3-MoE-30B | 30B | 83.42 | 173.61 | 89.06† |
| Llama-3.1-8B | 8B | 106.63 | 114.92 | 50.53 |
| Qwen2.5-7B | 7B | 108.47 | 124.11 | 54.56 |
| Mixtral-8x7B | 47B | 51.81 | 65.56 | 28.42 |
| GPT-OSS-120B | 120B (MoE) | 59.29 | 112.83 | 50.48§ |
| Solar-Open-100B | 100B (MoE) | 35.02 | 65.51 | 18.37§ |

*Qwen3-0.6B on GB10 again stopped at 9 tokens before EOS (2026-07-12); the 283.90 tok/s figure is from that short window and is not directly comparable to full-length runs.
†Qwen3-MoE-30B (`qwen3-moe-4bit`) **failed** on GB10 at 0.3.0 (Metal-only fused-MoE kernel aborted on CUDA); the CUDA fused decode-MoE kernel (#319) restored it at 0.3.1, and at 89.06 tok/s it stays ahead of M1 Ultra (83.75).
§GPT-OSS-120B and Solar-Open-100B were excluded from the 2026-07-12 GB10 sweep by the memory gate (weights > ~51 GiB, `SKIP:oom_estimate`); their figures are carried from the 2026-06-17 / 0.3.1 sweep.
¶Nemotron-H-30B doubled vs the 2026-06-17 record (40.32) because the fused single-token SSM decode kernel was ported to CUDA on 2026-07-10 (#727); the post-reboot re-verification (#755) confirmed the gain on a fresh host (87.41, post-reboot single). The whole SSM/hybrid cluster carries the same attribution (see the GB10 file's notable-changes list).

M1 Ultra column is from 2026-07-12 with mlxcel 0.4.0-rc.1 / MLX pin `57c66cac` / `--cooldown 30 --big-cooldown 30`, using the `mlxcel-bench-decode` same-process harness.
M5 Max column is from the 2026-09-03/04 full re-sweep with mlxcel **0.6.0** / MLX pin `9a795735` / `--cooldown 30 --big-cooldown 30`, same-process `mlxcel-bench-decode` harness.
GB10 column is from 2026-07-12 with mlxcel 0.4.0-rc.1 / MLX pin `57c66cac` (0.32.1) / CUDA 13.0 (SM 12.1) / `--cooldown 15 --big-cooldown 45`, using the `mlxcel-bench-decode` same-process warm harness, except the two `§`-marked memory-gated rows carried from 2026-06-17 / 0.3.1 and the `¶`-marked Nemotron-H row, which is the post-reboot single from the same day (#755, `--cooldown 30`).
**The columns no longer share a version.** M5 Max is mlxcel 0.6.0 / MLX pin `9a795735`; M1 Ultra and GB10 are still 0.4.0-rc.1 / `57c66cac`, pending their own re-sweeps. The cross-hardware ratios below therefore mix versions and should be read as indicative until those hosts are re-measured. The mixing is mild in practice: across these 16 rows M5 Max moved between -2.2% and +5.4% from 0.4.0-rc.1 to 0.6.0 (13 of 16 within +/-2%), so the hardware delta still dominates. M5 Max stays roughly 1.73x faster than M1 Ultra on the selected 16 rows (avg ~1.73x, median ~1.77x). The largest MoE rows show the M5 Max advantage: qwen3-moe-30b runs at 175.48 vs 83.42 tok/s (2.10x), gpt-oss-120b at 113.90 vs 59.29 (1.92x), and solar-open-100b at 65.40 vs 35.02 (1.87x). On GB10 the CUDA fused decode-MoE kernel (#319) keeps qwen3-moe-30b (89.06) just ahead of M1 Ultra (83.42).
For Qwen2.5-0.5B the 4-bit row is the directly comparable cross-hardware figure; the bf16 variant runs at 295.65 tok/s on M1 Ultra (0.4.0-rc.1) and 400.41 tok/s on M5 Max (0.6.0).

## Overall Status (M5 Max at mlxcel 0.6.0; M1 Ultra and GB10 still at 0.4.0-rc.1)

| Metric | Count |
|--------|-------|
| Supported model architectures | 89+ ModelType variants |
| Text models tested (M1 Ultra, 2026-06-15) | 136 pass, 2 partial, 4 fail, 9 skip/non-standalone (151 dirs; adds apertus, seed-oss, dots.llm1, granite family, lfm2, plamo-2, falcon-h1, BitNet; diffusiongemma loads via #291) |
| Text models tested (M5 Max, 2026-09-03) | 144 pass, 6 partial, 4 fail/skip across 154 table rows (0.6.0 cooldown-30 full sweep, 175 dirs / 161 measured, 90 GB weight budget; 0 decode regressions vs 0.4.0-rc.1). Table rows are below the dir count because 12 duplicate directories, 4 non-text checkpoints and 6 drafter/dflash variants are benchmarked but not listed separately |
| Text models tested (GB10, 2026-07-12) | 141 pass measured + 5 pass carried (memory-gate skips), 0 code failures, 13 not-tested/N.A. (glm-5 pair incomplete/absent; paligemma2 image-only; docling/granite-speech/whisper/kokoro non-text-gen; 4 MTP/DFlash drafters; glm-4.5v + mistral-small-4-119b memory-gated, never measured) (159 dirs; 0.4.0-rc.1 full sweep) |
| VLM models tested (GB10, 2026-07-12) | 63 measured image rows + 1 carried (llama-4-scout, memory-gate skip); gemma-4-31b-it-nvfp4 image-input FAIL since fixed by #749 (0.4.0-rc.1) |
| VLM models tested (M5 Max, 2026-09-04) | 77 measured VLM rows (0.6.0 cooldown-30 full VLM re-sweep). Pixtral/Mistral3 image-token counts fall sharply by design after #792 (aspect-ratio processing, no upscaling), so their prefill and decode are not comparable with the 0.4.0-rc.1 rows |
| VLM models tested (M1 Ultra, 2026-06-15) | 55 measured VLM rows (53 pass + 2 partial) |
| Speculative MTP on M5 Max (2026-09-04) | Gemma 4 Unified 12B + MTP assistant: 1.39x / **1.57x** / 1.55x at K=2/4/8, acceptance 55.6% / 35.0% / 34.6%. Gemma 4 31B MTP produced no number on this run because the harness drove only a Gemma 4 Unified target; #1613 lifted that and the pairing is measured on M1 Ultra below, but this host has not been re-run. The DFlash rows stay deferred on their own blocker |
| Speculative MTP on M1 Ultra (2026-09-04) | First sweep past the harness's Gemma-4-Unified-only target gate (#1613). Every MTP pairing reads below 1.00x on this Apple GPU generation: Gemma 4 31B + assistant 0.93x / 0.75x / 0.75x, Gemma 4 Unified 12B + assistant 0.94x / 0.74x / 0.76x, Qwen 3.8 27B + MTP head 0.91x / 0.69x / 0.37x at K=2/4/8. Acceptance is not the cause; see the matrix for the round-cost reading |
| Batched serving on M5 Max (2026-09-04) | Aggregate scaling at B=4 vs B=1: qwen2.5-0.5b-bf16 3.17x, llama-3.1-8b-4bit 3.25x, qwen3-30b-a3b-4bit (MoE) 1.55x. 0 failed requests across all 9 cells |
| Embedding / rerank on M5 Max (2026-09-04) | 18 checkpoints (11 text embedders, 2 VL embedders, 5 rerankers), 90 cells, 0 failures. The 2 `local/*-merged` multi-vector entries are merge artifacts and are not fetchable, so they stay unmeasured |
| Beating mlx-lm on M1 Ultra (text, >=100%) | 24/74 (32%, 6-15 vs pinned 5-19 baseline) |
| At 90%+ parity on M1 Ultra (text) | 59/74 (80%, 6-15 vs pinned 5-19 baseline) |
| Average vs mlx-lm on M1 Ultra (text) | 96% decode speed (median 98%, 6-15 vs pinned 5-19 baseline) |
| Beating mlx-lm on M5 Max (text, >=100%) | 27/67 (40%) — **prior 0.0.28 campaign; the mlx-lm baseline has not been re-run since, and is now two mlxcel releases stale** |
| At 90%+ parity on M5 Max (text) | 62/67 (93%) — prior 0.0.28 campaign; pending a fresh baseline |
| Average vs mlx-lm on M5 Max (text) | 98% decode speed (median 99%) — prior 0.0.28 campaign; pending a fresh baseline |
| Average vs mlx-vlm on M5 Max (VLM) | 100% decode speed (median 100%; 17 pairs) — prior 0.0.28 campaign; pending a fresh baseline |

## Generating Benchmarks

```bash
# Full text benchmark (auto-names CSV by hardware+date)
./scripts/bench_decode.sh all

# Full VLM benchmark
./scripts/bench_decode.sh all --vlm

# Single model
./scripts/bench_decode.sh models/<model-name>
```

After benchmarking, update the corresponding `model_tests_<hardware>.md` file from the CSV.

## Prompt cache benchmarks

Feature: cross-request prompt-prefix KV cache. Bench driver:
[`tests/prompt_cache_prefill_bench.rs`](../../tests/prompt_cache_prefill_bench.rs) (run with
`cargo test --test prompt_cache_prefill_bench --release -- --ignored --nocapture`).

### What the bench measures

For each conversation depth in `{1, 2, 4, 8, 16}` the bench issues a warmup
turn against the `/v1/chat/completions` streaming endpoint, then a
measurement turn with an identical prefix. It records:

| Column | Definition |
| --- | --- |
| `cache` | `on` = server started with `--prompt-cache-enabled=true`; `off` = disabled. |
| `prompt_tokens` | `usage.prompt_tokens` from the final streaming chunk. |
| `cached_tokens` | `usage.prompt_tokens_details.cached_tokens` when present; otherwise `-`. |
| `ttft_ms` | Time to first content delta (proxy for prefill latency on a non-speculative decoder). |
| `prefill_ms` | Same quantity as `ttft_ms`; kept as a separate column for compatibility with existing CSV readers. |
| `decode_tps` | `completion_tokens / (total - ttft)`. |
| `total_ms` | End-to-end wall-clock time for the measurement turn. |

### Expected qualitative behavior

On a functioning cache at depths >= 2 the measurement turn reports
`cached_tokens > 0` and `ttft_ms` sits below the matching `cache=off` row
for the same depth. The exact per-depth ratio depends on model and host;
target order-of-magnitude (single-digit billion parameter model, dense
backend) is:

* Depth 1: ratio ≈ 1.0 (no preceding conversation to reuse).
* Depth 2–4: ratio 0.3 – 0.8 (partial prefix reuse).
* Depth 8–16: ratio 0.1 – 0.4 (near-constant cache adopt, linear cold
  prefill on the off row).

Record measured numbers for a specific host under a new sub-heading
(e.g. `### M5 Max, qwen3-0.6b-4bit`) when updating this file.

### Validation scope

The harness itself is end-to-end exercised via the integration test
`tests/prompt_cache_e2e.rs`, which asserts the wire contract
(`cached_tokens == 0` on turn 1, `> 0` and monotonic on turns 2..5) and
the prefill-latency ratio bound (≤ 1.3× turn 1) whenever the server is
able to serve the model. Host-specific prompt-cache throughput numbers
should be appended here after running on M1 Ultra, M5 Max, GB10, or Hopper.

## TurboQuant KV cache benchmarks

Feature: TurboQuant KV cache compression (turbo3 / turbo4 modes). Bench
driver: `tests/turbo_kv_e2e.rs` (run with
`cargo test --test turbo_kv_e2e --release -- --ignored --nocapture`).

For the full config guide, tuning knobs, and architectural description see
[`docs/turbo-kv-cache.md`](../turbo-kv-cache.md).

### Source CSV

`benchmarks/turbo_kv/2026-04-26_Mac.localdomain.csv`

### Measured PPL evaluation throughput — 2026-04-26, Mac.localdomain

The quality gate runs wikitext-2 PPL evaluation and records eval throughput
(tok/s) and wall-clock time over a 4K-token evaluation window. Numbers below
are from the first validated run.

| Model | KV mode | PPL eval tok/s | Wall clock ms | Gate result |
|---|---|---|---|---|
| Meta-Llama-3.1-8B-Instruct-4bit | fp16 | 733.76 | 111,617 | baseline |
| Meta-Llama-3.1-8B-Instruct-4bit | turbo4asym | 490.32 | 167,034 | **pass** |

Notes:
- Llama-3.1-8B-Instruct-4bit passes the turbo4asym PPL gate cleanly.
- The active Qwen2.5 quality-gate fixture is `Qwen2.5-1.5B-4bit` (base variant). Numbers for that row are pending a fresh gate run.
- Gemma-3-4b-it-4bit is ready for a quality-gate run but is not represented in this table yet.
- Decode/prefill tok/s measurements (as distinct from PPL eval throughput) are a follow-up item.

## Speculative drafters

This section records the current parity and perf envelope for the speculative
drafter pairings in the local benchmark setup (Gemma 4 MTP, Qwen 3.5 DFlash).

### Methodology

Driven by `src/bin/speculative_bench.rs` and `tests/speculative_parity.rs`:

- Prompt: 17-token-ish instruction (see `DEFAULT_PROMPT` in the bench source).
- Max new tokens: 96 (matches the upstream `mlx-vlm` README perf-table conditions).
- Sampling: greedy (`temperature = 0.0`).
- Decode-only timing (excludes prefill). Numbers come from `GenerationStats::decode_tok_per_sec`, which divides the generated token count by the decode wall-clock — matches the upstream `_dflash_rounds` / `_mtp_rounds` reporting convention.
- Warm-up: one 4-token generation before the timed run so MLX's lazy Metal kernel compilation doesn't inflate the first measurement.

Invocations:

```bash
# Single pairing:
./target/release/speculative_bench \
    --target models/qwen3.5-4b-4bit \
    --kind none \
    --batch 1 \
    --max-tokens 96 \
    2>&1 | tee /tmp/bench-qwen35-baseline.log

# Full sweep across reachable pairings:
./target/release/speculative_bench --sweep --batch 1 --max-tokens 96 \
    2>&1 | tee /tmp/bench-sweep.log
```

### Hardware + MLX pin

- **Hardware**: Apple M1 Ultra, 128 GB unified memory.
- **MLX upstream commit pin**: `84961223c02925bef6bef95d3a0a046779bde935`
  (the `GIT_TAG` in `src/lib/mlx-cpp/CMakeLists.txt` at the time of measurement,
  which is the single place the pin is written down).
- Re-measure after each MLX pin bump so the perf table reflects the active runtime.

### Reachable pairings

These are the pairings whose target + drafter checkpoints are present on
the M1 Ultra reference host. The no-drafter baseline rows are real numbers
captured on the host; the speculative numerator (tok/s) rows were a perf-bench
follow-up when this table was written, and **correctness parity is verified**
end-to-end by the `#[ignore]`-gated tests in `tests/speculative_parity.rs`.

The follow-up has since been measured: the M1 Ultra matrix below carries real
tok/s for every MTP pairing, and the catalog has grown a Qwen 3.8 27B baseline
and MTP pairing (#1613). The table immediately below is kept as the older
reading it was; read the matrix for current numbers.

| Pairing                       | Kind   | B | block_size | tok/s | speedup vs no-drafter | status                                                                |
|-------------------------------|--------|---|------------|-------|------------------------|-----------------------------------------------------------------------|
| Qwen 3.5 4B (no drafter)      | none   | 1 | —          | 95.4  | 1.00×                  | ok                                                                    |
| Qwen 3.5 4B + DFlash          | dflash | 1 | 16         | —     | —                      | parity verified; tok/s row is a perf-bench follow-up                  |
| Gemma 4 31B (no drafter)      | none   | 1 | —          | 20.4  | 1.00×                  | ok                                                                    |
| Gemma 4 31B + MTP assistant   | mtp    | 1 | 4          | —     | —                      | parity verified; tok/s row is a perf-bench follow-up                  |

### M1 Ultra pairing matrix (2026-09-04, mlxcel 0.7.0-beta.1)

Measured on the Mac Studio M1 Ultra 128 GB (Metal) with
`speculative_bench --sweep --batch 1 --max-tokens 128`. Greedy, decode-only
tok/s, the `DEFAULT_PROMPT` (14 tokens under the Gemma tokenizer, 13 under
Qwen's). Source CSV: `benchmarks/metal_m1ultra_spec_2026-09-04.csv`.

This is the first sweep on any host to carry Gemma 4 31B and Qwen 3.8 27B MTP
rows. #1613 replaced the harness's `LoadedModel::Gemma4Unified` match with
per-variant adapter selection and added the Qwen 3.8 baseline and MTP pairing
to `REACHABLE_PAIRINGS`, so the Gemma 4 31B pairing that reads as a blank in
the M5 Max matrix below, and the Qwen 3.8 pairing that is absent from it
entirely, are both measured here.

| Pairing (M1 Ultra Metal)                | Kind | K | tok/s | speedup vs no-drafter | acceptance | mean accepted len | status |
|-----------------------------------------|------|---|------:|----------------------:|-----------:|------------------:|--------|
| Qwen 3.5 4B (no drafter)                | none | — | 106.1 | 1.00×                 | —          | —                 | ok |
| Qwen 3.5 4B + DFlash                    | dflash | 2/4/8 | — | —                 | —          | —                 | DEFERRED (DFlash loader + public Qwen3NextCache API) |
| Gemma 4 31B (no drafter)                | none | — | 19.9  | 1.00×                 | —          | —                 | ok |
| Gemma 4 31B + MTP assistant             | mtp  | 2 | 18.4  | 0.93×                 | 74.0%      | 0.74              | ok |
| Gemma 4 31B + MTP assistant             | mtp  | 4 | 14.9  | 0.75×                 | 52.9%      | 1.59              | ok |
| Gemma 4 31B + MTP assistant             | mtp  | 8 | 14.9  | 0.75×                 | 52.9%      | 1.59              | ok (effective K=4) |
| Gemma 4 Unified 12B (no drafter)        | none | — | 38.4  | 1.00×                 | —          | —                 | ok |
| Gemma 4 Unified 12B + MTP assistant     | mtp  | 2 | 36.0  | 0.94×                 | 54.5%      | 0.55              | ok |
| Gemma 4 Unified 12B + MTP assistant     | mtp  | 4 | 28.5  | 0.74×                 | 39.6%      | 1.19              | ok |
| Gemma 4 Unified 12B + MTP assistant     | mtp  | 8 | 29.3  | 0.76×                 | 39.6%      | 1.19              | ok (effective K=4) |
| Qwen 3.8 27B (no drafter)               | none | — | 24.9  | 1.00×                 | —          | —                 | ok |
| Qwen 3.8 27B + MTP head                 | mtp  | 2 | 22.5  | 0.91×                 | 76.1%      | 0.76              | ok |
| Qwen 3.8 27B + MTP head                 | mtp  | 4 | 17.2  | 0.69×                 | 50.4%      | 1.51              | ok |
| Qwen 3.8 27B + MTP head                 | mtp  | 8 | 9.3   | 0.37×                 | 21.6%      | 1.51              | ok (proposals past the fourth are all rejected) |

**Every MTP pairing reads below 1.00x on this host, and acceptance is not the
reason.** Gemma 4 Unified 12B accepts *more* here than on M5 Max at the same K
(39.6% and mean accepted length 1.19 at K=4, against 35.0% and 1.05), and still
lands at 0.74x where M5 Max reads 1.57x. The difference is the verify round:
`speculative-decoding-m1ultra-2026-08-19.md` measures a block-4 round at 2.70
classic decode steps on M1 Ultra against 1.27 on M5 Max, so break-even needs
about 2.7 emitted tokens per verify here and roughly 1.3 there. A mean accepted
length near 1.2 emits about 2.2 per verify, which clears the M5 Max bar and not
this one. This is the same reading behind the static `MLXCEL_ENABLE_MTP_B1`
gate declining B=1 MTP on Apple GPU generation 13, so the rows confirm the
shipped default rather than contradicting it.

**K=8 splits by drafter family.** The Gemma 4 assistants clamp to their
configured block size of 4: acceptance and mean accepted length are identical
at K=4 and K=8, so the wider request is a no-op. The `qwen3_5_mtp` head honors
it instead, proposing 7 per round against 3 at K=4 while accepting the same
1.51, which drops acceptance to 21.6% and throughput to 9.3 tok/s (0.37x).
K=4 or narrower is the operating point for that pairing.

Two baselines appear in both this matrix and the older reachable-pairings table
above: Qwen 3.5 4B at 106.1 tok/s against 95.4, and Gemma 4 31B at 19.9 against
20.4. Those older rows were captured at a different mlxcel version and MLX pin,
so read them as context rather than as a controlled run-over-run comparison.

### GB10 CUDA pairing matrix (2026-07-10, issue #638)

Measured on the NVIDIA GB10 (Grace-Blackwell) CUDA host with
`speculative_bench --sweep --k-values 2,4,8`. Greedy, decode-only tok/s, the
14-token `DEFAULT_PROMPT`, `--max-tokens 128`. Full analysis and the policy
tuning derivation are in
[`speculative-pairing-gb10-2026-07-10.md`](speculative-pairing-gb10-2026-07-10.md).

| Pairing (GB10 CUDA)                     | Kind | K | tok/s | speedup vs no-drafter | acceptance | mean accepted len | status |
|-----------------------------------------|------|---|------:|----------------------:|-----------:|------------------:|--------|
| Gemma 4 Unified 12B (no drafter)        | none | — | 14.5  | 1.00×                 | —          | —                 | ok |
| Gemma 4 Unified 12B + MTP assistant     | mtp  | 2 | 19.0  | 1.31×                 | 56.6%      | 0.57              | ok (multirow qmv, #725) |
| Gemma 4 Unified 12B + MTP assistant     | mtp  | 4 | 21.2  | 1.46×                 | 35.8%      | 1.07              | ok (multirow qmv, #725) |
| Gemma 4 Unified 12B + MTP assistant     | mtp  | 8 | 20.4  | 1.41×                 | 34.1%      | 1.02              | ok (effective K=4) |

Pre-#725 record (per-row qmv verify; reproducible with `MLXCEL_QMV_MULTIROW=0`,
which measures 7.7 tok/s at K=4 on the same binary):

| Pairing (GB10 CUDA, pre-#725)           | Kind | K | tok/s | speedup vs no-drafter | acceptance | mean accepted len | status |
|-----------------------------------------|------|---|------:|----------------------:|-----------:|------------------:|--------|
| Gemma 4 Unified 12B + MTP assistant     | mtp  | 2 | 11.1  | 0.77×                 | 55.6%      | 0.56              | regression |
| Gemma 4 Unified 12B + MTP assistant     | mtp  | 4 | 7.6   | 0.52×                 | 35.0%      | 1.05              | regression |
| Gemma 4 Unified 12B + MTP assistant     | mtp  | 8 | 7.5   | 0.52×                 | 35.0%      | 1.05              | regression (effective K=4) |

The pre-#725 regression came from the verify `[1, K]` forward hitting the CUDA
quantized dispatch's `M*B < 8` per-row qmv fallback, which costs roughly K
classic forwards instead of amortizing to one the way it does on Apple Silicon
(the same pairing measured ~1.87× on M5 Max at the time; the 2026-09-04 0.6.0 re-measurement reads 1.57× at K=4, at unchanged acceptance, so the two hosts have converged somewhat since). The multirow qmv path (#725,
`MLXCEL_QMV_MULTIROW`) removes that fallback's weight re-reads, and B=1 MTP on
GB10 now clears the 1.4× target from issue #638 at K=4/K=8 at unchanged
acceptance (see `qmv-multirow-gb10-2026-07-11.md`). K=8 collapses onto K=4
because the drafter's configured block size is 4 and the acceptance never
clears the adaptive block-expansion gate. Serving note (#736, resolved): the
adaptive policy no longer relies on the `sqrt(K)` shape heuristic (issue
#638), which was calibrated against the pre-#725 verify and could wrongly
decline this favourable pairing; it now settles verdicts from a measured
comparison against classic-step probe rounds (hint format v3), so this
pairing profiles to an enable verdict in serving without a manual override
(`sqrt(K)` remains only as a fallback for windows with no probe signal). The
DFlash and 31B rows remain deferred (no checkpoint / wrong target family, see
the dated note).

### M5 Max pairing matrix (2026-09-04, mlxcel 0.6.0)

Measured on the MacBook Pro M5 Max (Metal) with
`speculative_bench --sweep --max-tokens 128`. Greedy, decode-only tok/s, the
14-token `DEFAULT_PROMPT`. Source CSV: `benchmarks/metal_m5max_spec_2026-09-04.csv`.

| Pairing (M5 Max Metal)                  | Kind | K | tok/s | speedup vs no-drafter | acceptance | mean accepted len | status |
|-----------------------------------------|------|---|------:|----------------------:|-----------:|------------------:|--------|
| Qwen 3.5 4B (no drafter)                | none | — | 171.8 | 1.00×                 | —          | —                 | ok |
| Qwen 3.5 4B + DFlash                    | dflash | 2/4/8 | — | —                 | —          | —                 | DEFERRED (DFlash loader + public Qwen3NextCache API) |
| Gemma 4 31B (no drafter)                | none | — | 28.3  | 1.00×                 | —          | —                 | ok |
| Gemma 4 31B + MTP assistant             | mtp  | 2/4/8 | — | —                   | —          | —                 | harness target gate, lifted by #1613; re-run pending |
| Gemma 4 Unified 12B (no drafter)        | none | — | 45.3  | 1.00×                 | —          | —                 | ok |
| Gemma 4 Unified 12B + MTP assistant     | mtp  | 2 | 62.9  | 1.39×                 | 55.6%      | 0.56              | ok |
| Gemma 4 Unified 12B + MTP assistant     | mtp  | 4 | 70.9  | **1.57×**             | 35.0%      | 1.05              | ok |
| Gemma 4 Unified 12B + MTP assistant     | mtp  | 8 | 70.0  | 1.55×                 | 34.6%      | 1.07              | ok (effective K=4) |

**Acceptance matches GB10 exactly** (55.6% / 0.56 at K=2 and 35.0% / 1.05 at
K=4 on both hosts). Acceptance is a drafter-quality property and the run is
greedy over a fixed prompt, so identical figures across two very different
backends are the expected result and confirm the drafter path is doing the same
work. Any difference in the speedup column between hosts is therefore a
kernel-dispatch effect, not a drafter effect.

K=4 is the operating point: K=8 buys nothing (34.6% acceptance, mean accepted
length 1.07, so the extra proposals past the fourth are discarded) and reads
marginally below K=4.

**Two pairings on this host produced no number, for different reasons.** The
`dflash` rows are the known harness deferral and still are. The Gemma 4 31B MTP
rows failed with `MTP bench currently supports a Gemma 4 Unified target;
load_model returned a different variant`, because `run_mtp` hard-matched
`LoadedModel::Gemma4Unified` and the 31B checkpoint loads as another variant.
The `gemma-4-31b-it-assistant-bf16` drafter was on disk throughout, so this was
a harness limitation rather than a missing checkpoint. **That restriction is
gone**: #1613 replaced the match with per-variant adapter selection, and the
pairing is measured in the M1 Ultra matrix above. The three cells here stay
empty because this M5 Max sweep predates the fix and the host has not been
re-run; they are not a statement about the pairing.

The same limitation covered a coverage gap: `qwen3.8-27b-mtp-4bit` and
`qwen3.8-27b-mtp-bf16` were on disk and their target `qwen3.8-27b-4bit` was
measured in the text sweep, but `REACHABLE_PAIRINGS` carried no Qwen 3.8 entry,
and adding one alone would not have helped because the same `Gemma4Unified`
match rejected the target. #1613 closed both halves: the catalog now carries a
Qwen 3.8 27B baseline and an MTP pairing against the `qwen3_5_mtp` head, and
both are measured in the M1 Ultra matrix above. This M5 Max sweep predates them.

### Deferred pairings

These pairings cannot be measured today because the drafter checkpoint is
not on the reference host AND/OR an upstream dependency is unresolved.

| Pairing                          | Drafter checkpoint                              | Status / blocker                                                                  |
|----------------------------------|-------------------------------------------------|-----------------------------------------------------------------------------------|
| Gemma 4 E2B + MTP assistant      | `mlx-community/gemma-4-E2B-it-assistant-bf16`   | drafter checkpoint not on disk; centroid LM head support required                 |
| Gemma 4 E4B + MTP assistant      | `mlx-community/gemma-4-E4B-it-assistant-bf16`   | drafter checkpoint not on disk; centroid LM head support required                 |
| Gemma 4 26B-A4B + MTP assistant  | `mlx-community/gemma-4-26B-A4B-it-assistant-bf16` | drafter checkpoint not on disk                                                  |

### Real-model byte-equality parity test

`tests/speculative_parity.rs` carries two `#[ignore]`-gated real-model
tests — `greedy_parity_dflash_qwen35_4b` and `greedy_parity_mtp_gemma4_31b`
— that verify speculative-decoding **correctness** end-to-end.
Each test runs two phases:

1. **Structural phase** (in-process): load the target, assert the model
   variant, resolve the drafter kind, load the drafter, and — for DFlash —
   `bind()` the drafter to the target.
2. **Byte-equality phase** (subprocess): spawn `mlxcel-server` twice
   against the same target — once with `--model-draft --draft-kind
   {dflash,mtp} --draft-block-size {16,4}` and once without any
   `--draft-*` flag — submit the same fixed prompt to
   `/v1/chat/completions` at `temperature = 0`, and assert the two
   responses are byte-identical (same `message.content` *and* same
   `usage.completion_tokens`). The two servers run sequentially so a
   32–48 GB host only holds one target's weights at a time.

#### CI hardware lane / fixed cadence

These tests are `#[ignore]`-gated so `cargo test` on a dev machine (or a
CI host without the model checkpoints) skips them. They are run on the
**hardware lane** — an Apple Silicon runner with the model checkpoints
mounted under `models/` — on a fixed cadence:

```bash
# Run both speculative real-model parity tests serially (required:
# they share GPU memory and each spawns mlxcel-server subprocesses).
cargo test --test speculative_parity --release -- --ignored --test-threads=1 --nocapture
```

A test whose checkpoints are absent self-skips with a log line, so the
invocation is safe to wire into any Apple Silicon CI lane regardless of
which checkpoints that lane has provisioned.

Once the perf-bench numerators are captured, the speculative tok/s rows in
the table above flip on, and the table grows additional rows for the
`(block_size ∈ {2, 3, 4, 5, 6, 8}, B ∈ {1, 4, 8})` MTP sweep and
`(block_size ∈ {4, 8, 16, 24, 32}, B ∈ {1, 4, 8})` DFlash sweep.

### Expected speedup envelope (per upstream `mlx-vlm` README)

For comparison with the eventual measured numbers — these are the
upstream M3 Max / 96 GB results, NOT mlxcel measurements:

| Pairing               | B | block_size | upstream speedup                                                          |
|-----------------------|---|------------|---------------------------------------------------------------------------|
| Gemma 4 26B-A4B + MTP | 4 | 3          | 3.94×                                                                     |
| Gemma 4 31B + MTP     | 4 | 3          | 2.29×                                                                     |
| Gemma 4 E4B + MTP     | 4 | 4          | 1.56×                                                                     |
| Gemma 4 E4B + MTP     | 16| any        | slower than baseline (overhead > speedup at high B on small target)       |

DFlash speedup envelope is not documented as concretely upstream. mlxcel's
measured numbers will become the reference table once the speculative
perf-bench numerators are captured on the hardware lane.
