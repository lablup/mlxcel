# Model Compatibility & Performance Tests (NVIDIA GB10)

Compatibility and decode/prefill performance for mlxcel models on **NVIDIA GB10 (DGX Spark)** running the CUDA backend. This is the **0.3.1** benchmark, run on the merged CUDA fused decode-MoE kernel (#319).

## Test Environment

| Item | Value |
|------|-------|
| **Hardware** | NVIDIA GB10 (DGX Spark), 122 GB unified LPDDR5x |
| **OS** | Linux (aarch64), kernel 6.17 |
| **Backend** | CUDA 13.0 (driver via `libcuda.so.1`, cuDNN 9) |
| **mlxcel version** | 0.3.1 |
| **MLX pin** | mlxcel-core commit `a6ec7123` |
| **CUDA build** | `MLX_CUDA_ARCHITECTURES=121 cargo build --release --features cuda` (GB10 = SM 12.1) |
| **Harness** | same-process `mlxcel-bench-decode`, warm prefill, pre-warm on, `--cooldown 0` |
| **Test Prompt** | "Hello, how are you today?" (text) / "What is in this image?" (VLM) |
| **Max Tokens** | 100 |
| **Test Date** | 2026-06-17 |
| **CSVs** | `benchmarks/cuda_gb10_2026-06-17.csv`, `benchmarks/cuda_gb10_vlm_2026-06-17.csv` |

> Version note: this is the 0.3.1 benchmark, run on the post-#319 code. The `Cargo.toml` bump to 0.3.1 happens at release, so the CSVs were stamped 0.3.0 by the harness and relabeled to 0.3.1 to match the line they measure.
>
> Build note: a default `cargo build --release` on Linux produces a CPU-only binary (default features are `surgery` only), which silently runs MLX on the Grace CPU at ~0.4 tok/s. The GB10 benchmark must be built with `--features cuda`.
>
> The CUDA fused decode-MoE kernel (#319) landed for 0.3.1: nine MoE models that aborted at 0.3.0 on the Metal-only kernel (`[metal_kernel] No Metal back-end`) now run on the CUDA fused path, faster than `gather_qmm`. See the Summary and `fused-moe-decode-kernel-design.md`.

## Legend

- ✅ Pass: model loads and produces tokens within the configured budget
- ❌ Fail: warmup/bench failure or 0 tokens generated. Capacity OOM (weights exceed the 122 GB budget) also shows ❌ but is tallied separately as "Too large" in the summary.
- ⚪ Not tested / not applicable: weights not present, or the checkpoint is not a standalone text generator (an image-only model such as PaliGemma, a speech or document-layout model, or an MTP/DFlash drafter that needs a target); not a code failure

Prefill/Decode are the measured-pass figures from `mlxcel-bench-decode`. Notes record an early-EOS token count (when a model stopped before 100) or the failure cause. Models are grouped by architecture family; VLM-capable models appear once under text (text-prompt pass) and again in the image-input table at the end.

## Basic Transformers

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| apertus-8b-instruct-2509-4bit | ✅ | 494.90 | 42.15 | 31 tok |
| llama-3.1-8b-4bit | ✅ | 1294.78 | 49.10 |  |
| llama-3.1-8b-bf16 | ✅ | 1209.77 | 14.84 | 87 tok |
| llama-3.2-1b-4bit | ✅ | 7793.45 | 260.32 |  |
| meta-llama-3.1-8b-instruct-4bit | ✅ | 1320.16 | 48.93 |  |
| mimo-7b-4bit | ✅ | 355.38 | 52.37 |  |
| minicpm-2b-4bit | ✅ | 476.85 | 120.88 |  |
| olmo-1b-4bit | ✅ | 261.22 | 98.90 |  |
| olmo2-7b-4bit | ✅ | 311.46 | 53.17 | 27 tok |
| olmo3-32b-4bit | ✅ | 306.34 | 11.65 |  |
| phi-2-4bit | ✅ | 132.96 | 35.31 | 1 tok |
| phi-3.5-mini-4bit | ✅ | 290.51 | 91.35 | 40 tok |
| phi-3-mini-4bit | ✅ | 284.33 | 92.24 | 22 tok |
| phi-4-4bit | ✅ | 165.50 | 27.19 |  |
| qwen2-0.5b | ✅ | 4075.86 | 479.62 |  |
| qwen2.5-0.5b-4bit | ✅ | 3784.53 | 485.68 |  |
| qwen2.5-0.5b-bf16 | ✅ | 3416.94 | 199.89 |  |
| qwen2.5-1.5b-4bit | ✅ | 1101.93 | 200.90 |  |
| qwen2.5-1.5b-instruct-4bit | ✅ | 1538.21 | 201.59 |  |
| qwen2.5-7b | ✅ | 594.78 | 53.60 |  |
| qwen2.5-7b-4bit | ✅ | 608.73 | 53.16 |  |
| qwen2.5-7b-8bit | ✅ | 663.57 | 30.18 |  |
| qwen2.5-7b-instruct-4bit | ✅ | 617.78 | 54.28 |  |
| qwen3-0.6b | ✅ | 1764.03 | 294.34 | 9 tok |
| qwen3-0.6b-4bit | ✅ | 1592.92 | 290.52 | 9 tok |
| qwen3-1.7b-4bit | ✅ | 1127.29 | 170.05 | 14 tok |
| qwen3-4b-4bit | ✅ | 532.50 | 80.55 | 36 tok |
| qwen3-8b-4bit | ✅ | 236.45 | 47.55 | 33 tok |
| smollm-135m-4bit | ✅ | 2429.28 | 652.08 |  |
| smollm3-3b-4bit | ✅ | 1645.91 | 104.11 | 51 tok |
| stablelm-1.6b-4bit | ✅ | 1678.51 | 198.28 |  |
| starcoder2-3b-4bit | ✅ | 215.17 | 101.82 |  |

## Gemma Family

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| gemma2-2b-4bit | ✅ | 639.86 | 109.55 | 27 tok |
| gemma-2b-4bit | ✅ | 583.56 | 94.01 | 49 tok |
| gemma3-1b-4bit | ✅ | 1252.95 | 241.85 | 34 tok |
| gemma3-4b-4bit | ✅ | 410.60 | 76.30 | 84 tok |
| gemma-3-4b-it-4bit | ✅ | 411.18 | 76.11 | 84 tok |
| gemma3n-e2b-4bit | ✅ | 428.25 | 79.96 | 72 tok |
| gemma3n-e4b-4bit | ✅ | 281.38 | 53.57 | 74 tok |
| gemma3n-e4b-bf16 | ✅ | 275.17 | 21.35 | 69 tok |

## Gemma 4

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| diffusiongemma-26b-a4b-it-4bit | ✅ | 117.60 | 37.68 | 27 tok |
| gemma-4-12b-it-4bit | ✅ | 202.66 | 14.70 |  |
| gemma-4-12b-it-assistant-4bit | ⚪ | - | - | MTP/DFlash drafter (needs a target; not standalone) |
| gemma-4-26b-a4b-it-4bit | ✅ | 130.81 | 58.59 | 26 tok |
| gemma-4-26b-a4b-it-qat-4bit | ✅ | 136.06 | 50.33 | 26 tok |
| gemma-4-31b-4bit | ✅ | 30.58 | 8.93 |  |
| gemma-4-31b-it-4bit | ✅ | 70.25 | 8.53 | 26 tok |
| gemma-4-31b-it-assistant-bf16 | ⚪ | - | - | MTP/DFlash drafter (needs a target; not standalone) |
| gemma-4-31b-it-nvfp4 | ✅ | 16.26 | 0.89 | 26 tok |
| gemma-4-31b-it-qat-4bit | ✅ | 80.51 | 5.59 | 26 tok |
| gemma-4-e2b-it-4bit | ✅ | 761.69 | 104.25 |  |
| gemma-4-e2b-it-8bit | ✅ | 608.29 | 58.43 |  |
| gemma-4-e2b-it-qat-4bit | ✅ | 503.36 | 66.39 |  |
| gemma-4-e4b-it-4bit | ✅ | 355.22 | 49.39 |  |
| gemma-4-e4b-it-8bit | ✅ | 292.35 | 27.36 | 45 tok |
| gemma-4-e4b-it-qat-4bit | ✅ | 273.71 | 33.67 | 33 tok |

## EXAONE

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| exaone-3.5-2.4b-4bit | ✅ | 1390.18 | 136.93 |  |
| exaone4-1.2b-4bit | ✅ | 1456.06 | 214.72 | 53 tok |

## Cohere / Command R

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| aya-expanse-8b-4bit | ✅ | 178.81 | 47.88 |  |
| command-r7b-4bit | ✅ | 127.27 | 47.78 |  |

## Granite (IBM)

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| granite-3.3-2b-instruct-4bit | ✅ | 1801.88 | 113.75 | 28 tok |
| granite-4.0-h-350m-4bit | ✅ | 1714.62 | 64.00 | 19 tok |
| granite-4.0-h-tiny-4bit | ✅ | 256.82 | 33.84 | 53 tok |
| granite-4.1-3b-4bit | ✅ | 328.96 | 73.74 | 7 tok |
| granite-4.1-8b-4bit | ✅ | 166.20 | 18.45 | 1 tok |
| granite-speech-4.1-2b-nar-mlx | ⚪ | - | - | not a standalone text-gen model |

## MoE (Mixture of Experts)

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| deepseek-v2-lite-4bit | ✅ | 160.07 | 96.81 |  |
| deepseek-v3-4bit | ❌ | - | - | too large for GB10 (671B @ 4bit ~350GB > 122GB); present checkpoint incomplete (layers 0-19 of 61) |
| dots.llm1.inst-mixed-4-6bit | ✅ | 25.42 | 22.04 | 39 tok |
| gpt-oss-120b-4bit | ✅ | 57.75 | 50.48 | 82 tok |
| gpt-oss-20b-mxfp4 | ✅ | 126.16 | 77.25 |  |
| lfm2-8b-a1b-4bit | ✅ | 145.46 | 157.73 | 37 tok |
| llama-4-scout-17b-4bit | ✅ | 28.20 | 20.94 |  |
| minimax-m2-3bit | ✅ | 26.95 | 22.03 |  |
| mixtral-8x7b-4bit | ✅ | 12.46 | 27.92 | 73 tok |
| phi-3.5-moe-4bit | ✅ | 28.87 | 50.13 |  |
| qwen1.5-moe-a2.7b-4bit | ✅ | 261.24 | 125.52 |  |
| qwen3-30b-a3b-4bit | ✅ | 133.40 | 90.70 | 34 tok |
| qwen3-moe-4bit | ✅ | 135.30 | 89.84 | 34 tok |

## MLA (Multi-head Latent Attention)

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| minicpm3-4b-4bit | ✅ | 275.48 | 56.16 |  |

## DeepSeek Family

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| deepseek-coder-1.3b-4bit | ✅ | 4540.35 | 86.86 |  |
| deepseek-r1-distill-7b-4bit | ✅ | 209.88 | 54.56 |  |

## Nemotron Family

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| nemotron-3-nano-omni-30b-a3b-reasoning-4bit | ✅ | 117.76 | 38.45 | 20 tok |
| nemotron-h-30b-4bit | ✅ | 116.21 | 40.32 | 46 tok |
| nemotron-nas-30b-4bit | ✅ | 113.03 | 37.33 | 46 tok |

## SSM / Mamba / Hybrid

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| falcon-h1-tiny-90m-instruct-4bit | ✅ | 1294.33 | 102.99 |  |
| falcon-mamba-7b-4bit | ✅ | 100.28 | 22.26 | 2 tok |
| jamba-v0.1-4bit | ✅ | 541.13 | 85.48 |  |
| lfm2-350m-8bit | ✅ | 2768.23 | 409.01 | 13 tok |
| mamba2-130m | ✅ | 984.73 | 181.05 |  |
| mamba2-1.3b-4bit | ✅ | 283.78 | 81.37 |  |
| plamo-2-1b | ✅ | 189.89 | 34.36 |  |

## Chinese / Asian Language Models

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| baichuan-m1-14b-4bit | ✅ | 74.32 | 22.36 | 7 tok |
| ernie-4.5-0.3b-4bit | ✅ | 6384.30 | 654.97 |  |
| glm4-flash-4bit | ✅ | 106.65 | 53.33 |  |
| glm-5.1-4bit | ⚪ | - | - | not tested (weights not downloaded) |
| glm-5-4bit | ⚪ | - | - | not tested (weights not downloaded) |
| hunyuan-13b | ✅ | 18.78 | 14.80 |  |
| hunyuan-1.8b-4bit | ✅ | 685.98 | 154.72 | 41 tok |
| hunyuan-a13b-instruct-4bit | ✅ | 20.07 | 15.06 |  |
| internlm2-7b-4bit | ✅ | 388.59 | 50.09 |  |
| internlm3-8b-4bit | ✅ | 522.79 | 43.89 |  |
| seed-oss-36b-instruct-4bit | ✅ | 78.84 | 10.23 |  |

## Mistral Family

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| ministral-3b-4bit | ✅ | 6280.07 | 100.54 | 34 tok |
| mistral-small-3.1-24b-4bit | ✅ | 63.68 | 16.08 |  |
| pixtral-12b | ✅ | 35.97 | 32.80 |  |
| pixtral-12b-4bit | ✅ | 36.02 | 32.85 |  |

## BitNet

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| bitnet-b1.58-2b-4t | ✅ | 517.84 | 130.24 | CUDA ternary kernel (#322) |
| bitnet-b1.58-2b-4t-4bit | ✅ | 537.54 | 176.50 | CUDA ternary kernel (#322) |

## VLM-capable Models (text-only pass)

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| aya-vision-8b | ✅ | 124.91 | 46.67 | 87 tok |
| bunny-llama3-8b-4bit | ✅ | 433.97 | 48.81 | 40 tok |
| docling-layout-heron-mlx-bf16 | ⚪ | - | - | not a standalone text-gen model |
| internvl3-1b | ✅ | 4559.06 | 473.46 | 37 tok |
| llava-1.5-7b-4bit | ✅ | 205.07 | 55.06 |  |
| llava-interleave-qwen-0.5b-bf16 | ✅ | 2531.16 | 206.58 | 49 tok |
| llava-next-mistral-7b-4bit | ✅ | 374.38 | 51.61 |  |
| minicpm-v-4.6-bf16 | ✅ | 317.54 | 110.35 |  |
| minicpm-v-4.6-mxfp4 | ✅ | 347.59 | 142.14 |  |
| molmo2-4b | ✅ | 197.01 | 26.56 | 33 tok |
| molmo-7b | ✅ | 203.66 | 33.64 | 24 tok |
| paligemma2-3b-6bit | ⚪ | 160.60 | 0.00 | image-only (PaliGemma): no text-gen without an image; works in the VLM table |
| phi-3.5-vision-4bit | ✅ | 435.53 | 91.05 | 43 tok |
| qwen2.5-vl-3b-4bit | ✅ | 371.22 | 59.93 | 39 tok |
| qwen2-vl-2b | ✅ | 583.13 | 101.35 | 35 tok |
| qwen2-vl-2b-4bit | ✅ | 607.09 | 100.84 | 35 tok |
| qwen3-vl-2b | ✅ | 828.99 | 163.95 | 59 tok |
| qwen3-vl-2b-4bit | ✅ | 840.44 | 165.15 | 59 tok |
| qwen3-vl-30b-a3b-4bit | ✅ | 126.71 | 83.25 | 35 tok |
| qwen3-vl-32b-4bit | ✅ | 84.07 | 10.95 | 37 tok |
| qwen3-vl-4b-4bit | ✅ | 359.60 | 75.66 | 49 tok |
| qwen3-vl-4b-instruct-4bit | ✅ | 350.36 | 75.78 | 49 tok |
| qwen3-vl-8b-4bit | ✅ | 213.04 | 47.57 | 57 tok |
| qwen3-vl-8b-instruct-4bit | ✅ | 227.56 | 47.62 | 57 tok |
| youtu-vl-4b-instruct | ✅ | 134.27 | 21.94 |  |

## Qwen3.5 / Qwen3.6 / Qwen3-next

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| qwen3.5-0.8b-4bit | ✅ | 640.75 | 174.95 | 18 tok |
| qwen3.5-0.8b-optiq-4bit | ✅ | 644.30 | 156.91 | 19 tok |
| qwen3.5-27b-4bit | ✅ | 59.38 | 12.21 | 30 tok |
| qwen3.5-27b-dflash | ⚪ | - | - | MTP/DFlash drafter (needs a target; not standalone) |
| qwen3.5-2b-4bit | ✅ | 514.72 | 124.55 | 32 tok |
| qwen3.5-35b-a3b-4bit | ✅ | 144.28 | 64.26 | 31 tok |
| qwen3.5-4b-4bit | ✅ | 247.95 | 62.05 | 31 tok |
| qwen3.5-4b-dflash | ⚪ | - | - | MTP/DFlash drafter (needs a target; not standalone) |
| qwen3.5-9b-4bit | ✅ | 172.83 | 38.92 | 31 tok |
| qwen3.5-9b-bf16 | ✅ | 169.27 | 13.11 | 31 tok |
| qwen3.6-35b-a3b-4bit | ✅ | 148.18 | 62.99 | 28 tok |
| qwen3-next-480b-4bit | ❌ | - | - | OOM skip (capacity, weights > mem budget) |

## Solar

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| solar-open-100b-4bit | ✅ | 46.76 | 18.37 |  |

## VLM Benchmark (image input)

Models that accept image input and generated tokens under the `"What is in this image?"` prompt with `tests/fixtures/test_image.png`.

| Model | Status | Generated Tokens | Prefill (tok/s) | Decode (tok/s) |
|-------|--------|------------------|-----------------|----------------|
| aya-vision-8b | ✅ | 100 | 632.49 | 45.34 |
| bunny-llama3-8b-4bit | ✅ | 37 | 1681.67 | 45.09 |
| gemma3-4b-4bit | ✅ | 23 | 1018.08 | 73.16 |
| gemma-3-4b-it-4bit | ✅ | 16 | 994.12 | 70.28 |
| gemma3n-e2b-4bit | ✅ | 25 | 2193.58 | 73.10 |
| gemma3n-e4b-4bit | ✅ | 33 | 1467.15 | 52.43 |
| gemma-4-12b-it-4bit | ✅ | 100 | 923.02 | 14.69 |
| gemma-4-26b-a4b-it-4bit | ✅ | 28 | 153.31 | 54.64 |
| gemma-4-26b-a4b-it-qat-4bit | ✅ | 30 | 149.84 | 48.93 |
| gemma-4-31b-4bit | ✅ | 8 | 317.69 | 7.82 |
| gemma-4-31b-it-4bit | ✅ | 25 | 277.27 | 8.26 |
| gemma-4-31b-it-qat-4bit | ✅ | 29 | 302.79 | 5.58 |
| gemma-4-e2b-it-4bit | ✅ | 100 | 2528.92 | 104.17 |
| gemma-4-e2b-it-8bit | ✅ | 100 | 2138.74 | 58.43 |
| gemma-4-e2b-it-qat-4bit | ✅ | 100 | 2235.54 | 66.43 |
| gemma-4-e4b-it-4bit | ✅ | 100 | 1407.62 | 49.50 |
| gemma-4-e4b-it-8bit | ✅ | 51 | 1138.82 | 26.97 |
| gemma-4-e4b-it-qat-4bit | ✅ | 32 | 1228.36 | 33.32 |
| internvl3-1b | ✅ | 8 | 1958.25 | 397.14 |
| llama-4-scout-17b-4bit | ✅ | 100 | 29.15 | 20.54 |
| llava-1.5-7b-4bit | ✅ | 100 | 1721.32 | 52.08 |
| llava-interleave-qwen-0.5b-bf16 | ✅ | 32 | 20633.47 | 188.13 |
| llava-next-mistral-7b-4bit | ✅ | 100 | 2159.71 | 51.07 |
| minicpm-v-4.6-bf16 | ✅ | 23 | 583.76 | 103.61 |
| minicpm-v-4.6-mxfp4 | ✅ | 48 | 625.47 | 138.70 |
| ministral-3b-4bit | ✅ | 100 | 2524.57 | 89.35 |
| mistral-small-3.1-24b-4bit | ✅ | 100 | 504.81 | 15.33 |
| molmo2-4b | ✅ | 46 | 773.43 | 26.21 |
| molmo-7b | ✅ | 2 | 1095.89 | 23.81 |
| nemotron-3-nano-omni-30b-a3b-reasoning-4bit | ✅ | 6 | 147.03 | 32.86 |
| paligemma2-3b-6bit | ✅ | 2 | 1754.82 | 44.34 |
| phi-3.5-vision-4bit | ✅ | 19 | 1413.29 | 80.50 |
| pixtral-12b | ✅ | 100 | 782.19 | 30.29 |
| pixtral-12b-4bit | ✅ | 100 | 772.90 | 30.19 |
| qwen2.5-vl-3b-4bit | ✅ | 64 | 598.32 | 59.70 |
| qwen2-vl-2b | ✅ | 12 | 660.36 | 91.60 |
| qwen2-vl-2b-4bit | ✅ | 12 | 656.09 | 88.14 |
| qwen3.5-0.8b-4bit | ✅ | 100 | 963.74 | 202.03 |
| qwen3.5-27b-4bit | ✅ | 100 | 103.80 | 12.70 |
| qwen3.5-2b-4bit | ✅ | 47 | 737.26 | 127.63 |
| qwen3.5-35b-a3b-4bit | ✅ | 100 | 181.59 | 59.98 |
| qwen3.5-4b-4bit | ✅ | 49 | 425.56 | 62.14 |
| qwen3.5-9b-4bit | ✅ | 100 | 311.34 | 39.81 |
| qwen3.5-9b-bf16 | ✅ | 100 | 338.09 | 13.54 |
| qwen3.6-35b-a3b-4bit | ✅ | 100 | 175.03 | 62.35 |
| qwen3-vl-2b | ✅ | 52 | 2130.17 | 132.22 |
| qwen3-vl-2b-4bit | ✅ | 84 | 2134.98 | 132.22 |
| qwen3-vl-30b-a3b-4bit | ✅ | 73 | 200.25 | 63.56 |
| qwen3-vl-32b-4bit | ✅ | 51 | 150.99 | 10.37 |
| qwen3-vl-4b-4bit | ✅ | 37 | 1001.53 | 66.15 |
| qwen3-vl-4b-instruct-4bit | ✅ | 37 | 1002.80 | 65.02 |
| qwen3-vl-8b-4bit | ✅ | 30 | 596.15 | 40.21 |
| qwen3-vl-8b-instruct-4bit | ✅ | 30 | 577.42 | 40.34 |
| gemma3n-e4b-bf16 | ✅ | 24 | 1878.01 | 20.65 |

---

## Summary

**Test date**: 2026-06-17 | **Hardware**: NVIDIA GB10 (DGX Spark) | **mlxcel**: 0.3.1 (CUDA 13.0, SM 12.1) | **MLX pin**: `a6ec7123`

| Metric | Count |
|--------|-------|
| **Total text models attempted** | 147 |
| **Pass (✅)** | 136 |
| **Fail (❌, code failure)** | 0 |
| **Not tested / N.A. (⚪)** | 9 |
| **Too large for GB10 (capacity, OOM ❌)** | 2 |
| **VLM models measured (image input)** | 54 |

### CUDA fused decode-MoE kernel (#319)

The 0.3.1 line ported the fused decode-MoE kernel to CUDA (#319). Nine MoE models that aborted at 0.3.0 on the Metal-only kernel (`[metal_kernel] No Metal back-end`) now run on the CUDA fused path, faster than the `gather_qmm` fallback:

| Model | 0.3.0 | 0.3.1 |
|-------|-------|-------|
| qwen3-moe-4bit (Qwen3-30B-A3B) | ❌ FAIL | 89.84 |
| qwen3-30b-a3b-4bit | ❌ FAIL | 90.70 |
| qwen3.5-35b-a3b-4bit | ❌ FAIL | 64.26 |
| qwen3.6-35b-a3b-4bit | ❌ FAIL | 62.99 |
| qwen1.5-moe-a2.7b-4bit | ❌ FAIL | 125.52 |
| gemma-4-26b-a4b-it-4bit | ❌ FAIL | 58.59 |
| gemma-4-26b-a4b-it-qat-4bit | ❌ FAIL | 50.33 |
| dots.llm1.inst-mixed-4-6bit | ❌ FAIL | 22.04 |
| diffusiongemma-26b-a4b-it-4bit | ❌ FAIL | 37.68 |

`lfm2-8b-a1b-4bit` (141.8 → 157.7) and `qwen3-vl-30b-a3b-4bit` (57.1 → 83.3) were already passing on the legacy path and got faster on the fused path. Greedy output is byte-identical to `gather_qmm`. The CUDA fused crossover is much higher than Metal's (~13-14k vs ~4096), but the cap stays 4096 (see `fused-moe-decode-kernel-design.md`).

### Not-applicable / skipped models (by cause)

Every model that does not pass falls into one of these buckets; there are no outstanding code-level failures on the 0.3.1 line.

- **Weights not downloaded (⚪):** `glm-5-4bit`, `glm-5.1-4bit`
- **Image-only / not a standalone text generator (⚪):** `paligemma2-3b-6bit` (image-only PaliGemma: 0 text-gen without an image, captions correctly in the VLM image table); `docling-layout-heron-mlx-bf16` (document layout); `granite-speech-4.1-2b-nar-mlx` (speech)
- **MTP/DFlash drafter checkpoints (need a target; not standalone) (⚪):** `gemma-4-12b-it-assistant-4bit`, `gemma-4-31b-it-assistant-bf16`, `qwen3.5-27b-dflash`, `qwen3.5-4b-dflash`
- **Too large for GB10 (capacity, weights exceed the 122 GB budget; shown ❌):** `qwen3-next-480b-4bit`; `deepseek-v3-4bit` (671B @ 4bit ~350GB; the present checkpoint is also an incomplete partial download, layers 0-19 of 61)

`minicpm-v-4.6-mxfp4` previously failed here (mxfp4 warmup) and now passes after the quant-mode/group_size loader fix (#334): 142.14 tok/s text decode and 138.70 tok/s on image input. `qwen2.5-vl-3b` was a broken duplicate of `qwen2.5-vl-3b-4bit` and has been removed. The remaining capacity item (`deepseek-v3-4bit`) is tracked in #315.
