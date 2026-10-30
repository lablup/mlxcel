# Model Compatibility & Performance Tests (AMD Strix Halo gfx1151)

Single-stream decode and prefill for every checkpoint under `models/` on the ROCm backend, plus the batched-serving ladder. All 44 checkpoints passed all three rounds.

## Test Environment

| Item | Value |
|------|-------|
| **Hardware** | AMD Radeon 8060S Graphics (gfx1151, Strix Halo). The hardware string reports 96 GB of device memory; the host has 31 GiB of system RAM |
| **OS** | Debian, Linux 6.18.12 (x86_64) |
| **Backend** | ROCm 10.0.0, HIP 7.15.26333 |
| **mlxcel version** | 0.8.0-beta.1 at `2806f500` |
| **MLX pin** | `7ca40033`, ROCm overlay `75915908` |
| **Build** | `cargo build --release --features rocm` |
| **Harness** | same-process `mlxcel-bench-decode` through `scripts/bench_decode.sh`, `--cooldown 30 --big-cooldown 30` |
| **Shape** | pp512/tg128, 20 warmup tokens, temperature 0, EOS ignored, B = 1, text path (VLM checkpoints run their text decoder) |
| **Repetitions** | 3 rounds, order rotated between rounds, median reported |
| **Isolation** | `scripts/rocm_gpu_guard.sh`: 90 s idle before each round, monitored at 1 Hz, contended attempts discarded |
| **Test Date** | 2026-10-30 22:35 to 2026-10-31 01:54 KST |
| **CSVs** | `benchmarks/rocm_strixhalo-gfx1151_2026-10-30_models44.csv`, `benchmarks/rocm_strixhalo-gfx1151_batch_2026-10-31.csv` |

Rounds 2 and 3 were each rejected once because another `mlxcel` process appeared on the GPU, and were rerun. The rejected attempts are not in the CSV. The `date` column in the decode CSV is per row, so it changes at midnight during the sweep.

This host has no mlx-lm or mlx-vlm baseline at this revision, so there is no "vs mlx-lm" column. The numbers do not establish that a checkpoint's output is correct: the harness times decoding and does not grade the text.

## Decode and prefill

Checkpoints are named by their path under `models/`, because `gemma-3-4b-it-4bit` exists in both `models/` and `models/mlx/`. Sorted by decode rate.

| Status | Checkpoint | Prefill tok/s (median) | Prefill range | Decode tok/s (median) | Decode range |
|--------|------------|-----------------------:|---------------:|----------------------:|--------------:|
| ✅ | `qwen2.5-0.5b-4bit` | 18362.39 | 17260.07 to 19256.10 | 386.15 | 371.53 to 392.87 |
| ✅ | `mlx/Qwen3-0.6B-4bit` | 8772.31 | 8740.34 to 9575.62 | 283.26 | 250.58 to 284.95 |
| ✅ | `qwen3-0.6b-4bit` | 8523.70 | 8348.01 to 8981.58 | 261.42 | 253.47 to 267.76 |
| ✅ | `qwen2.5-1.5b-instruct-4bit` | 6367.22 | 6335.74 to 6516.55 | 177.59 | 172.82 to 179.16 |
| ✅ | `qwen3.5-0.8b-4bit` | 2678.80 | 2674.69 to 2740.68 | 168.20 | 167.05 to 171.05 |
| ✅ | `qwen2-vl-2b-instruct-4bit` | 5662.23 | 5486.51 to 6114.40 | 162.31 | 160.75 to 164.08 |
| ✅ | `llama-3.2-1b-instruct-4bit` | 6014.90 | 5928.64 to 6791.58 | 160.08 | 130.18 to 169.35 |
| ✅ | `mlx/Qwen3-VL-2B-Instruct-4bit` | 3658.14 | 3625.88 to 4005.17 | 130.58 | 126.06 to 134.76 |
| ✅ | `qwen3-1.7b-4bit` | 4784.19 | 4263.16 to 4822.97 | 127.76 | 127.60 to 130.48 |
| ✅ | `mlx/granite-4.0-h-tiny-4bit` | 926.11 | 904.52 to 978.52 | 105.21 | 105.12 to 105.85 |
| ✅ | `qwen3.5-2b-4bit` | 1925.22 | 1862.25 to 2011.98 | 102.12 | 99.77 to 103.89 |
| ✅ | `ministral-3-3b-instruct-2512-4bit` | 2668.47 | 2529.14 to 2676.28 | 90.01 | 89.63 to 90.19 |
| ✅ | `mlx/Qwen2.5-VL-3B-Instruct-4bit` | 1989.45 | 1985.84 to 2278.58 | 86.91 | 86.57 to 86.96 |
| ✅ | `gemma-4-e2b-it-4bit` | 3210.26 | 3152.04 to 3540.27 | 83.81 | 83.17 to 84.86 |
| ✅ | `llama-3.2-3b-instruct-4bit` | 2874.03 | 2627.64 to 2876.65 | 83.38 | 82.99 to 84.22 |
| ✅ | `mlx/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit` | 319.70 | 288.27 to 320.85 | 73.42 | 73.23 to 73.77 |
| ✅ | `qwen3-4b-instruct-2507-4bit` | 2278.70 | 2091.62 to 2298.89 | 72.35 | 72.29 to 72.59 |
| ✅ | `qwen3-4b-4bit` | 2156.07 | 2121.25 to 2157.38 | 72.33 | 72.25 to 72.41 |
| ✅ | `qwen3-vl-4b-instruct-4bit` | 2001.40 | 1982.15 to 2002.29 | 72.22 | 72.20 to 72.23 |
| ✅ | `gemma-3-4b-it-4bit` | 2251.23 | 2243.52 to 2267.45 | 69.55 | 69.36 to 69.67 |
| ✅ | `mlx/gemma-3-4b-it-4bit` | 2263.93 | 2228.66 to 2356.71 | 69.22 | 68.83 to 69.58 |
| ✅ | `mlx/gpt-oss-20b-MXFP4-Q4` | 530.71 | 529.61 to 532.49 | 63.67 | 63.65 to 64.38 |
| ✅ | `mlx/Qwen3-30B-A3B-4bit` | 318.59 | 316.74 to 321.47 | 60.18 | 54.17 to 61.42 |
| ✅ | `fp8/ReAligned-Qwen3.5-0.8B-FP8` | 297.17 | 297.02 to 298.14 | 59.00 | 55.11 to 59.36 |
| ✅ | `qwen3.5-4b-4bit` | 909.03 | 895.67 to 914.56 | 57.60 | 55.39 to 57.68 |
| ✅ | `mlx/Molmo-7B-D-0924-4bit` | 1555.74 | 1543.11 to 1583.11 | 51.80 | 51.79 to 51.92 |
| ✅ | `qwen2.5-coder-7b-instruct-4bit` | 1599.07 | 1577.64 to 1681.77 | 51.67 | 51.55 to 51.80 |
| ✅ | `mlx/Qwen2.5-7B-Instruct-4bit` | 1719.52 | 1659.13 to 1721.33 | 51.59 | 51.53 to 51.62 |
| ✅ | `gemma-4-26b-a4b-it-4bit` | 389.32 | 380.09 to 390.41 | 47.26 | 47.00 to 47.39 |
| ✅ | `qwen3.6-35b-a3b-4bit` | 318.38 | 295.39 to 322.31 | 45.82 | 45.52 to 46.73 |
| ✅ | `gemma-4-e4b-it-4bit` | 1869.23 | 1789.30 to 1871.77 | 45.26 | 45.04 to 45.77 |
| ✅ | `llava-1.5-7b-4bit` | 1145.32 | 1144.62 to 1231.87 | 43.83 | 43.74 to 43.87 |
| ✅ | `llava-v1.6-mistral-7b-4bit` | 1062.01 | 1048.97 to 1064.33 | 42.79 | 42.77 to 42.84 |
| ✅ | `mistral-7b-instruct-v0.3-4bit` | 1062.26 | 1052.27 to 1106.24 | 42.67 | 42.59 to 42.72 |
| ✅ | `mlx/Meta-Llama-3.1-8B-Instruct-4bit` | 1064.25 | 1062.83 to 1124.16 | 40.91 | 40.85 to 40.97 |
| ✅ | `qwen3-8b-4bit` | 983.95 | 976.92 to 1036.97 | 37.98 | 37.86 to 38.09 |
| ✅ | `qwen3.5-9b-4bit` | 593.77 | 591.77 to 598.14 | 32.36 | 31.50 to 32.38 |
| ✅ | `gemma-3-12b-it-4bit` | 879.17 | 850.06 to 881.51 | 27.91 | 27.90 to 27.93 |
| ✅ | `hf/Youtu-VL-4B-Instruct` | 2128.83 | 2066.20 to 2208.62 | 19.48 | 19.40 to 19.51 |
| ✅ | `devstral-small-2-24b-instruct-2512-4bit` | 391.16 | 391.02 to 393.35 | 13.45 | 13.44 to 13.45 |
| ✅ | `qwen3.8-27b-4bit` | 212.80 | 212.61 to 213.41 | 12.77 | 12.75 to 12.77 |
| ✅ | `qwen3.6-27b-4bit` | 215.68 | 213.09 to 216.38 | 12.75 | 12.47 to 12.75 |
| ✅ | `gemma-4-31b-it-4bit` | 322.32 | 316.17 to 323.63 | 9.64 | 9.63 to 9.66 |
| ✅ | `mlx/Mixtral-8x7B-Instruct-v0.1-4bit` | 126.92 | 126.88 to 126.98 | 8.78 | 8.63 to 8.79 |

## Batched serving (B = 1/2/4)

`mlxcel-server --parallel 4 --max-batch-prefill 4`, one fresh server per model, `scripts/bench_serving_concurrency.py --prompt-tokens 512 --max-tokens 128`, one pass per level under the GPU guard. Levels run in ascending order and share a prompt, so B = 2 and B = 4 start from a warm prompt cache. `qwen2.5-0.5b-4bit` stands in for the usual `qwen2.5-0.5b-bf16`, which is not on this host.

### `qwen2.5-0.5b-4bit`

| B | ok/fail | TTFT mean ms | TTFT p95 ms | Per-request decode tok/s | Aggregate tok/s | Scaling vs B=1 |
|--:|---:|---:|---:|---:|---:|---:|
| 1 | 1/0 | 193.6 | 193.6 | 312.0 | 213.1 | 1.00x |
| 2 | 2/0 | 62.5 | 72.4 | 109.1 | 208.6 | 0.98x |
| 4 | 4/0 | 69.1 | 105.8 | 132.7 | 498.0 | 2.34x |

### `Meta-Llama-3.1-8B-Instruct-4bit`

| B | ok/fail | TTFT mean ms | TTFT p95 ms | Per-request decode tok/s | Aggregate tok/s | Scaling vs B=1 |
|--:|---:|---:|---:|---:|---:|---:|
| 1 | 1/0 | 600.9 | 600.9 | 36.2 | 31.2 | 1.00x |
| 2 | 2/0 | 362.6 | 474.8 | 27.2 | 50.9 | 1.63x |
| 4 | 4/0 | 565.5 | 921.6 | 18.1 | 67.3 | 2.16x |

### `Qwen3-30B-A3B-4bit`

| B | ok/fail | TTFT mean ms | TTFT p95 ms | Per-request decode tok/s | Aggregate tok/s | Scaling vs B=1 |
|--:|---:|---:|---:|---:|---:|---:|
| 1 | 1/0 | 1574.6 | 1574.6 | 58.6 | 34.2 | 1.00x |
| 2 | 2/0 | 1565.4 | 1565.9 | 41.3 | 55.2 | 1.61x |
| 4 | 4/0 | 2889.7 | 2890.2 | 27.3 | 67.8 | 1.98x |

Each level is a single pass lasting seconds, so these figures are less stable than the three-round decode table. `qwen2.5-0.5b-4bit` shows it: per-request decode at B = 2 is lower than at B = 4, and its whole ladder ran in about 4 seconds.

## Observations

- `fp8/ReAligned-Qwen3.5-0.8B-FP8` prefills at 297 tok/s and decodes at 59 tok/s, against 2679 and 168 for `qwen3.5-0.8b-4bit`. Same architecture family, different weight format; this sweep does not explain the gap.
- `qwen3-0.6b-4bit` and `mlx/Qwen3-0.6B-4bit` have byte-identical weights (`cmp` on `model.safetensors`), yet their median decode differs by 8% (261 and 283 tok/s). That sets a noise floor for small, fast checkpoints on this host: a gap of this size between two such rows is not evidence of a real difference. The two `gemma-3-4b-it-4bit` copies have different weight files and agree within 1%.
- Decode ranges wider than 10% across rounds: `llama-3.2-1b-instruct-4bit` (130 to 169), `mlx/Qwen3-0.6B-4bit` (251 to 285) and `mlx/Qwen3-30B-A3B-4bit` (54 to 61). Treat small differences on these as noise.
- Against the 2026-10-30 popular15 run at `63b1ad9d` (same shape and host), the 15 shared checkpoints moved by -3.1% to +0.8% on decode and -4.5% to +3.5% on prefill. No change exceeds 10%.
