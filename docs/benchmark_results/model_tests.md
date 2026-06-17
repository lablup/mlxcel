# Model Compatibility & Performance Tests

Per-hardware benchmark results and cross-hardware comparison for mlxcel.

For a public, data-driven Apple Silicon summary that combines M1 Ultra,
M5 Max, and mlx-lm / mlx-vlm baselines, see
[Benchmark Report - 2026-05-19](benchmark-report.md).

## Per-Hardware Results

| Hardware | File | Status | Last Updated |
|----------|------|--------|-------------|
| Mac Studio M1 Ultra 128GB | [model_tests_m1ultra.md](model_tests_m1ultra.md) | Active | 2026-06-15 |
| MacBook Pro M5 Max 128GB | [model_tests_m5max.md](model_tests_m5max.md) | Active | 2026-06-15 |
| NVIDIA GB10 (DGX Spark) | [model_tests_gb10.md](model_tests_gb10.md) | Active | 2026-06-17 |

## Benchmark CSVs

Current source-of-truth data lives in `benchmarks/`:

| CSV | Hardware | Date | Type |
|-----|----------|------|------|
| `metal_m5max_2026-06-15.csv` | M5 Max | 2026-06-15 (mlxcel 0.2.1, MLX pin a6ec7123; full text re-benchmark, 151 rows, 135 measured) | Text |
| `metal_m5max_vlm_2026-06-15.csv` | M5 Max | 2026-06-15 (mlxcel 0.2.1, MLX pin a6ec7123; full VLM re-benchmark, 53 measured rows) | VLM |
| `metal_m5max_2026-05-19.csv` | M5 Max | 2026-05-19 (mlxcel 0.0.28, MLX 0.31.2) | Text |
| `metal_m5max_vlm_2026-05-19.csv` | M5 Max | 2026-05-19 (mlxcel 0.0.28, MLX 0.31.2) | VLM |
| `metal_m5max_vlm_2026-05-20.csv` | M5 Max | 2026-05-20 (mlxcel 0.0.28, MLX 0.31.2; Gemma3n + Molmo v1 + Phi-3.5 vision + Gemma3 4B VLM entries) | VLM |
| `pylm_m5max_2026-05-18.csv` | M5 Max | 2026-05-19 benchmark campaign (mlx-lm 0.31.3 baseline; CSV date crossed midnight) | Text |
| `pylm_m5max_vlm_2026-05-18.csv` | M5 Max | 2026-05-19 benchmark campaign (mlx-vlm 0.4.4 baseline; CSV date crossed midnight) | VLM |
| `metal_m1ultra_2026-06-15.csv` | M1 Ultra | 2026-06-15 (mlxcel 0.2.1, MLX pin a6ec712; full text re-benchmark post #289 fix, 151 rows) | Text |
| `metal_m1ultra_vlm_2026-06-15.csv` | M1 Ultra | 2026-06-15 (mlxcel 0.2.1, MLX pin a6ec712; full VLM re-benchmark, 55 measured rows) | VLM |
| `metal_m1ultra_2026-06-15_pre289_regressed.csv` | M1 Ultra | 2026-06-15 (mlxcel pre-#290; bf16-scale decode regression evidence sweep) | Text |
| `metal_m1ultra_2026-06-12.csv` | M1 Ultra | 2026-06-12 (mlxcel 0.1.4, MLX pin a6ec712; full text re-benchmark, 121 rows) | Text |
| `metal_m1ultra_vlm_2026-06-12.csv` | M1 Ultra | 2026-06-12 (mlxcel 0.1.4, MLX pin a6ec712; full VLM re-benchmark, 49 measured rows) | VLM |
| `metal_m1ultra_2026-05-19.csv` | M1 Ultra | 2026-05-19 (mlxcel 0.0.28, MLX commit 84961223; >65GB skipped) | Text |
| `metal_m1ultra_vlm_2026-05-19.csv` | M1 Ultra | 2026-05-19 (mlxcel 0.0.28, MLX commit 84961223; >65GB skipped) | VLM |
| `pylm_m1ultra_2026-05-19.csv` | M1 Ultra | 2026-05-19 (mlx-lm 0.31.3 baseline, https://github.com/ml-explore/mlx-lm @ `df1d3f3`; >65GB skipped) | Text |
| `pylm_m1ultra_vlm_2026-05-19.csv` | M1 Ultra | 2026-05-19 (mlx-vlm baseline, https://github.com/Blaizzy/mlx-vlm @ `d85ca4d`; >65GB skipped) | VLM |
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
| SmolLM-135M | 135M | 374.92 | 916.80 | 652.08 |
| ERNIE-4.5-0.3B | 300M | 495.71 | 1072.92 | 654.97 |
| Qwen2.5-0.5B (4bit) | 500M | 343.91 | 678.95 | 485.68 |
| Llama-3.2-1B | 1B | 364.36 | 552.96 | 260.32 |
| Qwen3-0.6B | 600M | 275.55 | 565.88 | 294.34* |
| StableLM-1.6B | 1.6B | 270.88 | 424.38 | 198.28 |
| Gemma-3-1B | 1B | 229.70 | 396.72 | 241.85 |
| EXAONE-3.5-2.4B | 2.4B | 197.73 | 287.70 | 136.93 |
| SmolLM3-3B | 3B | 126.29 | 232.99 | 104.11 |
| Nemotron-H-30B | 30B | 91.54 | 176.10 | 40.32 |
| Qwen3-MoE-30B | 30B | 83.75 | 175.63 | 89.84† |
| Llama-3.1-8B | 8B | 107.89 | 116.61 | 49.10 |
| Qwen2.5-7B | 7B | 111.50 | 126.26 | 53.16 |
| Mixtral-8x7B | 47B | 54.25 | 65.37 | 27.92 |
| GPT-OSS-120B | 120B (MoE) | 58.41 | 113.91 | 50.48 |
| Solar-Open-100B | 100B (MoE) | 32.96 | 65.39 | 18.37 |

*Qwen3-0.6B on GB10 again stopped at 9 tokens before EOS (2026-06-17); the 294.34 tok/s figure is from that short window and is not directly comparable to full-length runs.
†Qwen3-MoE-30B (`qwen3-moe-4bit`) **failed** on GB10 at 0.3.0 (Metal-only fused-MoE kernel aborted on CUDA); the CUDA fused decode-MoE kernel (#319) restores it at 0.3.1, and at 89.84 tok/s it now edges past M1 Ultra (83.75).

M1 Ultra column is from 2026-06-15 with mlxcel 0.2.1 / MLX pin commit `a6ec712` (0.32.0-dev) / no cooldown, using the `mlxcel-bench-decode` same-process harness (post #289 bf16-scale fix and #291 quantized-embedding fix).
M5 Max column is from the 2026-06-15 full re-sweep with mlxcel 0.2.1 / MLX pin `a6ec7123` / same-process `mlxcel-bench-decode` harness (bare run).
GB10 column is from 2026-06-17 with mlxcel 0.3.1 / MLX pin `a6ec7123` / CUDA 13.0 (SM 12.1) / `--cooldown 0`, using the `mlxcel-bench-decode` same-process warm harness. 0.3.1 adds the CUDA fused decode-MoE kernel (#319): the Qwen MoE rows that aborted at 0.3.0 now run, and `qwen3-moe-30b` (89.84) edges past M1 Ultra (83.75). (CSV `mlx_version` relabeled to 0.3.1; the Cargo.toml bump lands at release.)
Both Apple Silicon columns now share mlxcel 0.2.1 and the same MLX pin `a6ec712`, so the gap reflects hardware delta. M5 Max stays roughly 1.76x faster than M1 Ultra on the selected 16 rows (avg ~1.76x, median ~1.88x). The largest MoE rows show the M5 Max advantage: qwen3-moe-30b runs at 175.63 vs 83.75 tok/s (2.10x), gpt-oss-120b at 113.91 vs 58.41 (1.95x), and solar-open-100b at 65.39 vs 32.96 (1.98x). The GB10 column is now on mlxcel 0.3.1 with the CUDA fused decode-MoE kernel (#319); the Qwen MoE rows that failed at 0.3.0 run on CUDA at 0.3.1 and can exceed M1 Ultra (qwen3-moe-30b 89.84 vs 83.75).
For Qwen2.5-0.5B the 4-bit row is the directly comparable cross-hardware figure; on M1 Ultra `qwen2.5-0.5b-bf16` now runs after the #289 fix (298.92 tok/s), and the bf16 variant runs on M5 Max at 404.68 tok/s.

## Overall Status (mlxcel 0.2.1 on M1 Ultra and M5 Max, 0.3.1 on GB10)

| Metric | Count |
|--------|-------|
| Supported model architectures | 89+ ModelType variants |
| Text models tested (M1 Ultra, 2026-06-15) | 136 pass, 2 partial, 4 fail, 9 skip/non-standalone (151 dirs; adds apertus, seed-oss, dots.llm1, granite family, lfm2, plamo-2, falcon-h1, BitNet; diffusiongemma loads via #291) |
| Text models tested (M5 Max, 2026-06-15) | 131 pass, 5 partial, 14 fail/skip (0.2.1 full sweep; post-sweep: qwen2.5-vl-3b-4bit fixed by re-download, oversized bf16 hunyuan dropped; neither a code regression) |
| Text models tested (GB10, 2026-06-17) | 136 pass, 0 fail, 9 not-tested/N.A. (glm-5/glm-5.1 weights not downloaded; paligemma2 image-only; docling/granite-speech non-text-gen; 4 MTP/DFlash drafters), 2 too-large/capacity (qwen3-next-480b, deepseek-v3) (147 total; 0.3.1 with the CUDA fused decode-MoE kernel #319, bitnet CUDA kernel #322, and the VLM mxfp4 loader fix #334) |
| VLM models tested (GB10, 2026-06-17) | 54 measured image rows (0.3.1) |
| VLM models tested (M5 Max, 2026-06-15) | 54 valid VLM rows (0.2.1 full VLM re-sweep; adds qwen3-vl-4b/8b, minicpm-v-4.6-bf16, nemotron-omni, youtu-vl; qwen2.5-vl-3b-4bit restored after re-download) |
| VLM models tested (M1 Ultra, 2026-06-15) | 55 measured VLM rows (53 pass + 2 partial) |
| Beating mlx-lm on M1 Ultra (text, >=100%) | 24/74 (32%, 6-15 vs pinned 5-19 baseline) |
| At 90%+ parity on M1 Ultra (text) | 59/74 (80%, 6-15 vs pinned 5-19 baseline) |
| Average vs mlx-lm on M1 Ultra (text) | 96% decode speed (median 98%, 6-15 vs pinned 5-19 baseline) |
| Beating mlx-lm on M5 Max (text, >=100%) | 27/67 (40%) — **prior 0.0.28 campaign; mlx-lm baseline not re-run for the 0.2.1 sweep** |
| At 90%+ parity on M5 Max (text) | 62/67 (93%) — prior 0.0.28 campaign; pending a 0.2.1 baseline |
| Average vs mlx-lm on M5 Max (text) | 98% decode speed (median 99%) — prior 0.0.28 campaign; pending a 0.2.1 baseline |
| Average vs mlx-vlm on M5 Max (VLM) | 100% decode speed (median 100%; 17 pairs) — prior 0.0.28 campaign; pending a 0.2.1 baseline |

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
  (`src/lib/mlxcel-core/build.rs::MLX_EXPECTED_COMMIT` at the time of measurement).
- Re-measure after each MLX pin bump so the perf table reflects the active runtime.

### Reachable pairings

These are the pairings whose target + drafter checkpoints are present on
the M1 Ultra reference host. The no-drafter baseline rows are real numbers
captured on the host; the speculative numerator (tok/s) rows remain a
perf-bench follow-up, but **correctness parity is verified** end-to-end
by the `#[ignore]`-gated tests in `tests/speculative_parity.rs`.

| Pairing                       | Kind   | B | block_size | tok/s | speedup vs no-drafter | status                                                                |
|-------------------------------|--------|---|------------|-------|------------------------|-----------------------------------------------------------------------|
| Qwen 3.5 4B (no drafter)      | none   | 1 | —          | 95.4  | 1.00×                  | ok                                                                    |
| Qwen 3.5 4B + DFlash          | dflash | 1 | 16         | —     | —                      | parity verified; tok/s row is a perf-bench follow-up                  |
| Gemma 4 31B (no drafter)      | none   | 1 | —          | 20.4  | 1.00×                  | ok                                                                    |
| Gemma 4 31B + MTP assistant   | mtp    | 1 | 4          | —     | —                      | parity verified; tok/s row is a perf-bench follow-up                  |

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
