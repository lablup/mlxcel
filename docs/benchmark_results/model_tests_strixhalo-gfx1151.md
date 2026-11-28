# Model Compatibility & Performance Tests (AMD Strix Halo gfx1151)

> Retention note (2026-11-10): final results are retained here. Superseded experiments are indexed in [benchmark index](README.md); retired run captures and any historical artifact paths resolve in the [frozen evidence tree](https://github.com/lablup/mlxcel/tree/2dc8ed47b30dc204d892328098e8cc34ec02ddc1/docs/benchmark_results/). Measurement values and their original limitations have not been remeasured.

Single-stream decode and prefill for every checkpoint under `models/` on the ROCm backend, plus the batched-serving ladder. All 44 checkpoints measured cleanly in one pass; eight were re-measured once because their first-pass value fell outside the 2026-10-30 range, and those rows are marked †.

## Test Environment

| Item | Value |
|------|-------|
| **Hardware** | AMD Radeon 8060S Graphics (gfx1151, Strix Halo). The hardware string reports 96 GB of device memory; the host has 31 GiB of system RAM |
| **OS** | Debian, Linux 6.18.12 (x86_64) |
| **Backend** | ROCm 10.0.0, HIP 7.15.26333; CSV `backend` = `rocm-10.0.0` |
| **mlxcel version** | 0.8.1 at `f4d5fc5b` |
| **MLX pin** | `7ca40033`, ROCm overlay `75915908` |
| **Build** | `cargo build --release --features rocm` |
| **Harness** | same-process `mlxcel-bench-decode` through `scripts/bench_decode.sh`, `--cooldown 30 --big-cooldown 30` |
| **Shape** | pp512/tg128, 20 warmup tokens, temperature 0, EOS ignored, B = 1, text path (VLM checkpoints run their text decoder) |
| **Repetitions** | one pass per checkpoint; eight checkpoints whose first-pass value fell outside the 2026-10-30 range re-measured once († in the table) |
| **Isolation** | `scripts/rocm_gpu_guard.sh --hold` for the whole sweep; the GPU had no KFD process before the first model |
| **Test Date** | 2026-11-03 15:27 to 16:18 KST (rechecks and batch ladder included) |
| **CSVs** | `benchmarks/rocm_strixhalo-gfx1151_2026-11-03_models44.csv`, `benchmarks/rocm_strixhalo-gfx1151_2026-11-03_recheck.csv`, `benchmarks/rocm_strixhalo-gfx1151_batch_2026-11-03.csv` |

This sweep is one pass per checkpoint, so the table has no per-model range. Prefill on the fastest models completes in 0.1 to 0.25 ms, where a single measurement is noisy: eight rows came back below the 2026-10-30 range and moved back into it on a second pass (for example `qwen3-1.7b-4bit` 2633 to 4420 tok/s, against a 4263 to 4823 range). Those eight rows use the recheck value, are marked †, and are the only rows not taken from the main pass. The recheck rows are in `rocm_strixhalo-gfx1151_2026-11-03_recheck.csv`.

The aggregate CSV was assembled from the four per-root passes: `..._models.csv` (30 top-level checkpoints), `..._models_mlx.csv` (12), `..._models_fp8.csv` (1) and `..._models_hf.csv` (1). The `SKIP:not_a_checkpoint` placeholder rows the top-level pass emits for the three subdirectory roots were dropped.

This host has no mlx-lm or mlx-vlm baseline at this revision, so there is no "vs mlx-lm" column. The numbers do not establish that a checkpoint's output is correct: the harness times decoding and does not grade the text.

## Decode and prefill

Checkpoints are named by their path under `models/`, because `gemma-3-4b-it-4bit` exists in both `models/` and `models/mlx/`. Sorted by decode rate.

| Status | Checkpoint | Prefill tok/s | Decode tok/s |
|--------|------------|--------------:|-------------:|
| ✅ | `qwen2.5-0.5b-4bit` † | 18084.45 | 348.90 |
| ✅ | `qwen3-0.6b-4bit` | 8487.97 | 264.84 |
| ✅ | `mlx/Qwen3-0.6B-4bit` | 8798.82 | 261.52 |
| ✅ | `llama-3.2-1b-instruct-4bit` | 6268.17 | 205.74 |
| ✅ | `qwen3.5-0.8b-4bit` | 2672.47 | 171.05 |
| ✅ | `qwen2.5-1.5b-instruct-4bit` | 6248.52 | 169.86 |
| ✅ | `qwen2-vl-2b-instruct-4bit` | 5617.90 | 153.46 |
| ✅ | `mlx/Qwen3-VL-2B-Instruct-4bit` | 3532.14 | 148.05 |
| ✅ | `qwen3-1.7b-4bit` † | 4420.25 | 141.80 |
| ✅ | `qwen3.5-2b-4bit` | 1889.81 | 111.94 |
| ✅ | `mlx/granite-4.0-h-tiny-4bit` | 944.67 | 103.38 |
| ✅ | `llama-3.2-3b-instruct-4bit` † | 2678.23 | 100.55 |
| ✅ | `ministral-3-3b-instruct-2512-4bit` † | 2530.00 | 93.58 |
| ✅ | `mlx/Qwen2.5-VL-3B-Instruct-4bit` † | 2077.15 | 91.64 |
| ✅ | `gemma-4-e2b-it-4bit` | 3040.20 | 86.98 |
| ✅ | `mlx/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit` | 315.28 | 75.91 |
| ✅ | `qwen3-4b-4bit` | 2128.94 | 75.82 |
| ✅ | `qwen3-vl-4b-instruct-4bit` | 2014.81 | 75.75 |
| ✅ | `qwen3-4b-instruct-2507-4bit` | 2147.82 | 75.71 |
| ✅ | `mlx/gemma-3-4b-it-4bit` † | 2221.71 | 71.49 |
| ✅ | `gemma-3-4b-it-4bit` | 2277.26 | 71.01 |
| ✅ | `mlx/gpt-oss-20b-MXFP4-Q4` | 533.41 | 64.15 |
| ✅ | `mlx/Qwen3-30B-A3B-4bit` | 302.43 | 63.23 |
| ✅ | `qwen3.5-4b-4bit` | 901.75 | 59.04 |
| ✅ | `fp8/ReAligned-Qwen3.5-0.8B-FP8` † | 295.30 | 57.96 |
| ✅ | `mlx/Qwen2.5-7B-Instruct-4bit` † | 1594.29 | 51.39 |
| ✅ | `qwen2.5-coder-7b-instruct-4bit` | 1588.56 | 51.35 |
| ✅ | `mlx/Molmo-7B-D-0924-4bit` | 1545.13 | 51.27 |
| ✅ | `llava-v1.6-mistral-7b-4bit` | 1056.63 | 50.20 |
| ✅ | `llava-1.5-7b-4bit` | 1146.40 | 50.19 |
| ✅ | `mistral-7b-instruct-v0.3-4bit` | 1050.93 | 50.13 |
| ✅ | `qwen3.6-35b-a3b-4bit` | 289.83 | 50.04 |
| ✅ | `gemma-4-26b-a4b-it-4bit` | 356.50 | 48.14 |
| ✅ | `mlx/Meta-Llama-3.1-8B-Instruct-4bit` | 1061.37 | 47.63 |
| ✅ | `gemma-4-e4b-it-4bit` | 1800.42 | 46.01 |
| ✅ | `qwen3-8b-4bit` | 981.06 | 45.54 |
| ✅ | `qwen3.5-9b-4bit` | 574.20 | 38.03 |
| ✅ | `gemma-3-12b-it-4bit` | 831.41 | 28.67 |
| ✅ | `mlx/Mixtral-8x7B-Instruct-v0.1-4bit` | 127.72 | 26.41 |
| ✅ | `hf/Youtu-VL-4B-Instruct` | 2100.17 | 19.48 |
| ✅ | `devstral-small-2-24b-instruct-2512-4bit` | 379.00 | 15.87 |
| ✅ | `qwen3.6-27b-4bit` | 209.56 | 12.81 |
| ✅ | `qwen3.8-27b-4bit` | 206.45 | 12.80 |
| ✅ | `gemma-4-31b-it-4bit` | 315.24 | 10.17 |

## Batched serving (B = 1/2/4)

`mlxcel-server --parallel 4 --max-batch-prefill 4`, one fresh server per model, `scripts/bench_serving_concurrency.py --prompt-tokens 512 --max-tokens 128`, one pass per level under the GPU guard. Levels run in ascending order and share a prompt, so B = 2 and B = 4 start from a warm prompt cache. `qwen2.5-0.5b-4bit` stands in for the usual `qwen2.5-0.5b-bf16`, which is not on this host.

### `qwen2.5-0.5b-4bit`

| B | ok/fail | TTFT mean ms | TTFT p95 ms | Per-request decode tok/s | Aggregate tok/s | Scaling vs B=1 |
|--:|---:|---:|---:|---:|---:|---:|
| 1 | 1/0 | 210.0 | 210.0 | 319.4 | 210.6 | 1.00x |
| 2 | 2/0 | 62.5 | 72.5 | 190.6 | 350.8 | 1.67x |
| 4 | 4/0 | 64.6 | 98.0 | 128.5 | 485.6 | 2.31x |

### `Meta-Llama-3.1-8B-Instruct-4bit`

| B | ok/fail | TTFT mean ms | TTFT p95 ms | Per-request decode tok/s | Aggregate tok/s | Scaling vs B=1 |
|--:|---:|---:|---:|---:|---:|---:|
| 1 | 1/0 | 654.4 | 654.4 | 41.2 | 34.3 | 1.00x |
| 2 | 2/0 | 374.0 | 495.4 | 29.4 | 54.5 | 1.59x |
| 4 | 4/0 | 607.9 | 970.5 | 18.8 | 69.4 | 2.02x |

### `Qwen3-30B-A3B-4bit`

| B | ok/fail | TTFT mean ms | TTFT p95 ms | Per-request decode tok/s | Aggregate tok/s | Scaling vs B=1 |
|--:|---:|---:|---:|---:|---:|---:|
| 1 | 1/0 | 1615.4 | 1615.4 | 67.3 | 36.6 | 1.00x |
| 2 | 2/0 | 1474.7 | 1474.7 | 44.6 | 59.2 | 1.62x |
| 4 | 4/0 | 3082.2 | 3082.9 | 28.8 | 68.4 | 1.87x |

Each level is a single pass lasting seconds, so these figures are noisier than the decode table. `qwen2.5-0.5b-4bit` shows it: its whole ladder ran in about 4 seconds, and its B = 2 aggregate (350.8 tok/s) is much higher than the 2026-10-31 ladder's 208.6 tok/s at the same level, while B = 4 moved from 498.0 to 485.6. The two larger models are stable: `Meta-Llama-3.1-8B-Instruct-4bit` aggregate B = 4 moved 67.3 to 69.4 tok/s and `Qwen3-30B-A3B-4bit` 67.8 to 68.4.

## Observations

- `fp8/ReAligned-Qwen3.5-0.8B-FP8` prefills at 295 tok/s and decodes at 58 tok/s, against 2672 and 171 for `qwen3.5-0.8b-4bit`. Same architecture family, different weight format; this sweep does not explain the gap.
- `qwen3-0.6b-4bit` and `mlx/Qwen3-0.6B-4bit` have byte-identical weights (`cmp` on `model.safetensors`) and now agree within 1.3% (265 and 262 tok/s). The 8% gap between the 2026-10-30 medians did not reproduce, so differences of roughly 10% on small, fast checkpoints are not evidence of a real difference.
- Decode on the fastest rows moves about 10% between passes: `qwen2.5-0.5b-4bit` 343 then 349 tok/s, `qwen3-1.7b-4bit` 149 then 142, and `fp8/ReAligned-Qwen3.5-0.8B-FP8` 48 then 58. Sub-0.25 ms prefill is noisier still, which is why the eight † rows were re-measured.
- Against the 2026-10-30 three-round medians at `2806f500`, decode moved a median +3.6%. Twelve checkpoints improved by more than 10%: `mlx/Mixtral-8x7B-Instruct-v0.1-4bit` 3.0x (8.78 to 26.41 tok/s, the fused-MoE Dff bound raised to 14336 in #2813), `llama-3.2-1b-instruct-4bit` 1.29x, `llama-3.2-3b-instruct-4bit` 1.21x, `qwen3-8b-4bit` 1.20x, and `devstral-small-2-24b-instruct-2512-4bit` 1.18x. The smaller gains on the 7B to 9B dense models are consistent with the `qmv_wide` stride stagger in #2814. Prefill moved a median -1.4%, with no checkpoint beyond 10% in either direction after the rechecks.
- Against the 2026-10-30 popular15 run at `63b1ad9d` (same shape and host), the 15 shared checkpoints moved by -3.1% to +0.8% on decode and -4.5% to +3.5% on prefill. No change exceeds 10%.
