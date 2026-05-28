# Model Compatibility & Performance Tests (NVIDIA GB10)

Compatibility and performance testing for mlxcel models on **NVIDIA GB10 (DGX Spark)**, running the CUDA backend.

## Test Environment

| Item | Value |
|------|-------|
| **Hardware** | NVIDIA GB10 (DGX Spark), 122 GB unified memory, ~273 GB/s LPDDR5x |
| **OS** | Linux (aarch64), kernel 6.17 |
| **Backend** | CUDA 13.0 |
| **mlxcel version** | 0.1.0 |
| **MLX version** | pinned commit `84961223` (via mlxcel-core; CSV `mlx_version` field records 0.31.2) |
| **Harness** | same-process `mlxcel-bench-decode`, warm prefill (PR `c9a77f2`), `--cooldown 0` |
| **Test Prompt** | "Hello, how are you today?" (text) / "What is in this image?" (VLM) |
| **Max Tokens** | 100 |
| **Test Date** | 2026-05-28 |
| **CSV** | `benchmarks/cuda_gb10_2026-05-28.csv`, `benchmarks/cuda_gb10_vlm_2026-05-28.csv` |

## Legend

- ✅ Pass: model loads and produces output within the configured token budget
- ❌ Fail: warmup/bench failure, OOM skip, or 0 tokens generated

## Basic Transformers

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| llama-3.2-1b-4bit | Llama-3.2-1B-4bit | ✅ | 6858.95 | 253.63 | 31 tok |
| llama-3.1-8b-4bit | Llama-3.1-8B-Instruct-4bit | ✅ | 1361.89 | 49.15 | |
| llama-3.1-8b-bf16 | Llama-3.1-8B-Instruct (bf16) | ✅ | 1208.56 | 14.81 | 87 tok |
| phi-2-4bit | phi-2-hf-4bit | ✅ | 135.15 | 36.49 | 1 tok |
| phi-3-mini-4bit | Phi-3-mini-4k-instruct-4bit | ✅ | 280.73 | 93.19 | 25 tok |
| phi-3.5-mini-4bit | Phi-3.5-mini-instruct-4bit | ✅ | 290.58 | 92.50 | 40 tok |
| phi-4-4bit | Phi-4-4bit | ✅ | 161.06 | 27.52 | |
| qwen2-0.5b | Qwen2.5-0.5B (bf16) | ✅ | 3870.91 | 496.44 | |
| qwen2.5-0.5b-4bit | Qwen2.5-0.5B-Instruct-4bit | ✅ | 3705.50 | 502.51 | |
| qwen2.5-0.5b-bf16 | Qwen2.5-0.5B (bf16) | ✅ | 3136.60 | 202.87 | |
| qwen2.5-7b | Qwen2.5-7B (bf16) | ✅ | 634.86 | 54.07 | |
| qwen2.5-7b-4bit | Qwen2.5-7B-Instruct-4bit | ✅ | 617.25 | 53.73 | |
| qwen2.5-7b-8bit | Qwen2.5-7B-8bit | ✅ | 684.10 | 29.98 | |
| qwen3-0.6b | Qwen3-0.6B (bf16) | ✅ | 2021.77 | 317.75 | 9 tok (EOS) |
| qwen3-0.6b-4bit | Qwen3-0.6B-4bit | ✅ | 1956.00 | 314.62 | 9 tok (EOS) |
| qwen3-1.7b-4bit | Qwen3-1.7B-4bit | ✅ | 1134.38 | 167.90 | 14 tok |
| qwen3-4b-4bit | Qwen3-4B-4bit | ✅ | 488.30 | 81.37 | 36 tok |
| qwen3-8b-4bit | Qwen3-8B-4bit | ✅ | 252.73 | 48.71 | 33 tok |
| smollm-135m-4bit | SmolLM-135M-Instruct-4bit | ✅ | 3001.35 | 643.04 | |
| smollm3-3b-4bit | SmolLM3-3B-4bit | ✅ | 1628.18 | 100.66 | 18 tok |
| stablelm-1.6b-4bit | stablelm-2-1_6b-chat-4bit | ✅ | 1546.33 | 197.05 | |
| starcoder2-3b-4bit | starcoder2-3b-4bit | ✅ | 220.65 | 102.42 | |
| olmo-1b-4bit | OLMo-1B-hf-4bit | ✅ | 262.62 | 98.26 | |
| olmo2-7b-4bit | OLMo2-7B-4bit | ✅ | 316.99 | 53.17 | 27 tok |
| olmo3-32b-4bit | OLMo3-32B-4bit | ✅ | 309.94 | 11.70 | |
| minicpm-2b-4bit | MiniCPM-2B-sft-bf16-4bit | ✅ | 434.56 | 122.27 | |
| mimo-7b-4bit | MiMo-7B-RL-4bit | ✅ | 358.62 | 53.33 | |

## Gemma Family

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| gemma-2b-4bit | gemma-2b-it-4bit | ✅ | 601.35 | 100.06 | 41 tok |
| gemma2-2b-4bit | gemma-2-2b-it-4bit | ✅ | 665.31 | 117.38 | 27 tok |
| gemma3-1b-4bit | gemma-3-1b-it-4bit | ✅ | 977.29 | 256.48 | 34 tok |
| gemma3-4b-4bit | gemma-3-4b-it-4bit | ✅ | 392.79 | 80.17 | 72 tok |
| gemma3n-e2b-4bit | gemma-3n-E2B-it-4bit | ✅ | 415.19 | 81.83 | 68 tok |
| gemma3n-e4b-4bit | gemma-3n-E4B-it-4bit | ✅ | 260.59 | 53.53 | 74 tok |
| gemma3n-e4b-bf16 | gemma-3n-E4B-it (bf16) | ✅ | 273.15 | 21.61 | 69 tok |

### Gemma 4

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| gemma-4-e2b-it-4bit | Gemma-4-E2B-it-4bit | ✅ | 707.60 | 98.70 | 28 tok |
| gemma-4-e2b-it-8bit | Gemma-4-E2B-it-8bit | ✅ | 522.63 | 58.24 | |
| gemma-4-e4b-it-4bit | Gemma-4-E4B-it-4bit | ✅ | 325.35 | 47.58 | 33 tok |
| gemma-4-e4b-it-8bit | Gemma-4-E4B-it-8bit | ✅ | 272.71 | 27.22 | 33 tok |
| gemma-4-26b-a4b-it-4bit | Gemma-4-26B-A4B-it-4bit | ❌ | - | FAIL | warmup failure |
| gemma-4-31b-4bit | Gemma-4-31B-4bit | ✅ | 23.32 | 8.79 | |
| gemma-4-31b-it-4bit | Gemma-4-31B-it-4bit | ✅ | 48.97 | 8.06 | 26 tok |
| Gemma-4-31b-it-nvfp4 | Gemma-4-31B-it (NVFP4) | ✅ | 16.52 | 0.90 | 26 tok |

## EXAONE

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| exaone-3.5-2.4b-4bit | EXAONE-3.5-2.4B-Instruct-4bit | ✅ | 1391.24 | 146.48 | |
| exaone4-1.2b-4bit | exaone-4.0-1.2b-4bit | ✅ | 1136.29 | 225.62 | 53 tok |

## Cohere / Command R

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| command-r7b-4bit | c4ai-command-r7b-4bit | ✅ | 124.23 | 52.12 | |
| aya-expanse-8b-4bit | aya-expanse-8b-4bit | ✅ | 159.55 | 52.89 | |

## MoE (Mixture of Experts)

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| mixtral-8x7b-4bit | Mixtral-8x7B-Instruct-v0.1-4bit | ✅ | 12.60 | 28.00 | 73 tok |
| qwen1.5-moe-a2.7b-4bit | Qwen1.5-MoE-A2.7B-Chat-4bit | ✅ | 248.96 | 112.09 | |
| qwen3-moe-4bit | Qwen3-30B-A3B-4bit | ✅ | 133.23 | 57.49 | 33 tok |
| qwen3-30b-a3b-4bit | Qwen3-30B-A3B-4bit | ✅ | 134.11 | 53.55 | 34 tok |
| phi-3.5-moe-4bit | Phi-3.5-MoE-instruct-4bit | ✅ | 28.99 | 51.35 | |
| minimax-m2-3bit | MiniMax-M2-3bit | ✅ | 26.72 | 21.85 | |
| gpt-oss-20b-mxfp4 | gpt-oss-20b-MXFP4 | ✅ | 126.41 | 77.94 | |
| gpt-oss-120b-4bit | gpt-oss-120b-4bit | ✅ | 54.57 | 50.63 | 73 tok |
| deepseek-v2-lite-4bit | DeepSeek-V2-Lite-Chat-4bit | ✅ | 156.31 | 99.07 | |
| deepseek-v3-4bit | DeepSeek-V3-0324-4bit | ❌ | - | FAIL | warmup failure |
| llama-4-scout-17b-4bit | Llama-4-Scout-17B-16E-4bit | ✅ | 27.67 | 20.88 | |

## MLA (Multi-head Latent Attention)

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| minicpm3-4b-4bit | MiniCPM3-4B-4bit | ✅ | 282.58 | 57.55 | |

## DeepSeek Family

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| deepseek-coder-1.3b-4bit | deepseek-coder-1.3b-4bit | ✅ | 4655.97 | 92.61 | |
| deepseek-r1-distill-7b-4bit | DeepSeek-R1-Distill-Qwen-7B-4bit | ✅ | 210.07 | 58.55 | |

## Nemotron Family

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| nemotron-h-30b-4bit | Nemotron-H-30B-4bit | ✅ | 108.15 | 32.92 | 46 tok |
| nemotron-nas-30b-4bit | Nemotron-NAS-30B-A3B-4bit | ✅ | 105.00 | 32.98 | 46 tok |

## SSM / Mamba / Hybrid Models

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| falcon-mamba-7b-4bit | Falcon-Mamba-7B-4bit | ✅ | 83.89 | 22.09 | 2 tok |
| mamba2-1.3b-4bit | mamba2-1.3b-4bit | ✅ | 277.75 | 80.50 | |
| jamba-v0.1-4bit | Jamba-v0.1-4bit | ✅ | 529.88 | 85.42 | |

## Chinese / Asian Language Models

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| baichuan-m1-14b-4bit | Baichuan-M1-14B-Instruct-4bit | ✅ | 75.66 | 24.06 | 7 tok |
| glm4-flash-4bit | GLM-4-Flash-4bit | ✅ | 102.70 | 55.04 | |
| GLM-5.1-4bit | GLM-5.1-4bit | ❌ | - | FAIL | warmup failure |
| internlm2-7b-4bit | InternLM2-7B-4bit | ✅ | 387.86 | 50.26 | |
| internlm3-8b-4bit | internlm3-8b-instruct-4bit | ✅ | 530.75 | 43.89 | |
| ernie-4.5-0.3b-4bit | ERNIE-4.5-0.3B-Instruct-4bit | ✅ | 5403.22 | 682.24 | |
| hunyuan-13b | Hunyuan-Large (bf16, 13B) | ✅ | 19.31 | 15.15 | |
| hunyuan-4bit | Hunyuan-Large-Instruct-4bit | ✅ | 19.87 | 14.86 | |
| hunyuan-dense-4bit | Hunyuan-1.8B-Instruct-4bit | ✅ | 668.84 | 158.90 | 41 tok |
| hunyuan-1.8b-4bit | Hunyuan-1.8B-Instruct-4bit | ✅ | 732.00 | 157.78 | 41 tok |
| hunyuan-large-4bit | Hunyuan-Large-Instruct-4bit | ✅ | 20.04 | 15.10 | |
| hunyuan-moe-a13b-bf16 | Hunyuan-MoE-A13B (bf16) | ✅ | 19.00 | 14.79 | |

## Mistral Family

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| ministral-3b-4bit | Ministral-3B-Instruct-4bit | ✅ | 6316.20 | 101.17 | 34 tok |
| mistral-small-3.1-24b-4bit | mistral-small-3.1-24b-4bit | ✅ | 65.18 | 16.08 | |
| pixtral-12b | pixtral-12b (bf16) | ✅ | 36.69 | 33.09 | |
| pixtral-12b-4bit | pixtral-12b-4bit | ✅ | 36.00 | 33.06 | |

## VLM-capable Models (text-only pass)

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| aya-vision-8b | aya-vision-8b | ✅ | 117.37 | 52.16 | |
| bunny-llama3-8b-4bit | Bunny-Llama-3-8B-V-4bit | ✅ | 380.57 | 52.89 | 40 tok |
| llava-1.5-7b-4bit | llava-1.5-7b-4bit | ✅ | 207.24 | 56.02 | |
| llava-next-mistral-7b-4bit | llava-v1.6-mistral-7b-4bit | ✅ | 371.54 | 52.64 | |
| llava-interleave-qwen-0.5b-bf16 | llava-interleave-qwen-0.5b-bf16 | ✅ | 2448.92 | 212.61 | 49 tok |
| molmo2-4b | molmo2-4b | ✅ | 188.38 | 26.76 | 33 tok |
| molmo-7b | Molmo-7B | ✅ | 212.01 | 33.62 | 24 tok |
| paligemma2-3b-6bit | paligemma2-3b | ❌ | 164.80 | 0.00 | 0 tokens generated |
| phi-3.5-vision-4bit | Phi-3.5-vision-instruct-4bit | ✅ | 426.67 | 91.41 | 43 tok |
| internvl3-1b | InternVL3-1B | ✅ | 4041.24 | 479.58 | 37 tok |
| qwen2-vl-2b | Qwen2-VL-2B (bf16) | ✅ | 670.87 | 101.92 | 35 tok |
| qwen2-vl-2b-4bit | Qwen2-VL-2B-Instruct-4bit | ✅ | 683.57 | 101.28 | 35 tok |
| qwen2.5-vl-3b | Qwen2.5-VL-3B (bf16) | ❌ | - | FAIL | warmup failure |
| qwen2.5-vl-3b-4bit | Qwen2.5-VL-3B-Instruct-4bit | ❌ | - | FAIL | warmup failure |
| qwen3-vl-2b | Qwen3-VL-2B (bf16) | ✅ | 848.87 | 166.97 | 61 tok |
| qwen3-vl-2b-4bit | Qwen3-VL-2B-Instruct-4bit | ✅ | 754.03 | 165.01 | 33 tok |
| qwen3-vl-30b-a3b-4bit | Qwen3-VL-30B-A3B-4bit | ✅ | 126.26 | 56.10 | 34 tok |
| qwen3-vl-32b-4bit | Qwen3-VL-32B-4bit | ✅ | 86.60 | 10.92 | 37 tok |

## Qwen3.5 / Qwen3-next (new architectures)

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| qwen3.5-0.8b-4bit | Qwen3.5-0.8B-4bit | ✅ | 632.45 | 172.27 | 18 tok |
| qwen3.5-2b-4bit | Qwen3.5-2B-4bit | ✅ | 512.33 | 127.49 | 31 tok |
| qwen3.5-4b-4bit | Qwen3.5-4B-4bit | ✅ | 250.83 | 63.05 | 31 tok |
| qwen3.5-9b-4bit | Qwen3.5-9B-4bit | ✅ | 164.08 | 39.19 | 31 tok |
| qwen3.5-9b-bf16 | Qwen3.5-9B (bf16) | ✅ | 163.30 | 13.03 | 31 tok |
| qwen3.5-27b-4bit | Qwen3.5-27B-4bit | ✅ | 59.42 | 12.36 | 30 tok |
| qwen3.5-35b-a3b-4bit | Qwen3.5-35B-A3B-4bit | ✅ | 144.11 | 48.71 | 31 tok |
| Qwen3.5-397B-A17B-4bit | Qwen3.5-397B-A17B-4bit | ❌ | - | SKIP | OOM skip (capacity) |
| qwen3-next-480b-4bit | Qwen3-next-480B-4bit | ❌ | - | SKIP | OOM skip (capacity) |
| solar-open-100b-4bit | Solar-Open-100B-4bit | ✅ | 49.53 | 18.52 | |

## VLM Benchmark (image input)

| Model | Test Model | Status | Generated Tokens | Prefill (tok/s) | Decode (tok/s) |
|-------|------------|--------|------------------|-----------------|----------------|
| aya-vision-8b | aya-vision-8b | ✅ | 100 | 673.71 | 38.94 |
| bunny-llama3-8b-4bit | Bunny-Llama-3-8B-V-4bit | ✅ | 37 | 1680.92 | 38.30 |
| gemma3-4b-4bit | gemma-3-4b-it-4bit | ✅ | 14 | 1012.14 | 69.83 |
| gemma3n-e2b-4bit | gemma-3n-E2B-it-4bit | ✅ | 29 | 2223.72 | 74.87 |
| gemma3n-e4b-4bit | gemma-3n-E4B-it-4bit | ✅ | 33 | 1422.82 | 51.44 |
| gemma3n-e4b-bf16 | gemma-3n-E4B-it (bf16) | ✅ | 24 | 2014.06 | 20.96 |
| gemma-4-31b-4bit | Gemma-4-31B-4bit | ✅ | 8 | 322.22 | 7.82 |
| gemma-4-31b-it-4bit | Gemma-4-31B-it-4bit | ✅ | 27 | 331.15 | 8.46 |
| gemma-4-e2b-it-4bit | Gemma-4-E2B-it-4bit | ✅ | 20 | 2510.79 | 95.14 |
| gemma-4-e2b-it-8bit | Gemma-4-E2B-it-8bit | ✅ | 7 | 2135.14 | 48.47 |
| gemma-4-e4b-it-4bit | Gemma-4-E4B-it-4bit | ✅ | 47 | 1321.27 | 47.15 |
| gemma-4-e4b-it-8bit | Gemma-4-E4B-it-8bit | ✅ | 11 | 1117.64 | 25.43 |
| llama-4-scout-17b-4bit | Llama-4-Scout-17B-16E-4bit | ✅ | 100 | 28.39 | 20.65 |
| llava-1.5-7b-4bit | llava-1.5-7b-4bit | ✅ | 100 | 1704.28 | 53.27 |
| llava-interleave-qwen-0.5b-bf16 | llava-interleave-qwen-0.5b-bf16 | ✅ | 32 | 19737.74 | 191.14 |
| llava-next-mistral-7b-4bit | llava-v1.6-mistral-7b-4bit | ✅ | 100 | 2170.28 | 52.26 |
| ministral-3b-4bit | Ministral-3B-Instruct-4bit | ✅ | 100 | 2529.43 | 90.70 |
| mistral-small-3.1-24b-4bit | mistral-small-3.1-24b-4bit | ✅ | 100 | 473.60 | 15.59 |
| molmo2-4b | molmo2-4b | ✅ | 46 | 762.92 | 26.60 |
| paligemma2-3b-6bit | paligemma2-3b | ✅ | 2 | 1846.20 | 42.61 |
| phi-3.5-vision-4bit | Phi-3.5-vision-instruct-4bit | ✅ | 19 | 1383.21 | 80.90 |
| pixtral-12b | pixtral-12b (bf16) | ✅ | 100 | 562.66 | 29.86 |
| pixtral-12b-4bit | pixtral-12b-4bit | ✅ | 100 | 749.72 | 30.63 |
| qwen2-vl-2b | Qwen2-VL-2B (bf16) | ✅ | 12 | 677.83 | 90.87 |
| qwen2-vl-2b-4bit | Qwen2-VL-2B-Instruct-4bit | ✅ | 12 | 696.50 | 92.67 |
| qwen3-vl-2b | Qwen3-VL-2B (bf16) | ✅ | 84 | 2192.64 | 131.85 |
| qwen3-vl-2b-4bit | Qwen3-VL-2B-Instruct-4bit | ✅ | 80 | 2260.71 | 132.73 |
| qwen3-vl-30b-a3b-4bit | Qwen3-VL-30B-A3B-4bit | ✅ | 72 | 184.85 | 45.15 |
| qwen3-vl-32b-4bit | Qwen3-VL-32B-4bit | ✅ | 55 | 158.46 | 10.40 |
| qwen3.5-27b-4bit | Qwen3.5-27B-4bit | ✅ | 100 | 103.03 | 12.75 |
| qwen3.5-35b-a3b-4bit | Qwen3.5-35B-A3B-4bit | ✅ | 100 | 179.45 | 45.23 |
| qwen3.5-9b-bf16 | Qwen3.5-9B (bf16) | ✅ | 100 | 347.94 | 13.58 |
| qwen3.5-2b-4bit | Qwen3.5-2B-4bit | ✅ | 47 | 719.30 | 123.77 |
| qwen3.5-4b-4bit | Qwen3.5-4B-4bit | ✅ | 49 | 427.82 | 59.77 |
| qwen3.5-9b-4bit | Qwen3.5-9B-4bit | ✅ | 100 | 306.12 | 39.09 |
| qwen3.5-0.8b-4bit | Qwen3.5-0.8B-4bit | ✅ | 100 | 1013.86 | 206.67 |
| internvl3-1b | InternVL3-1B | ✅ | 8 | 1898.25 | 406.33 |
| molmo-7b | Molmo-7B | ✅ | 2 | 1121.48 | 23.66 |

---

## Summary

**Test date**: 2026-05-28 | **Hardware**: NVIDIA GB10 (DGX Spark) | **mlxcel**: 0.1.0 | **MLX**: pin `84961223`

| Metric | Count |
|--------|-------|
| **Total text models attempted** | 109 |
| **Pass (✅)** | 101 |
| **Fail / skip / 0-token (❌)** | 8 (5 fail, 2 OOM skip, 1 zero-token) |
| **VLM models measured (image)** | 38 |
| **VLM Pass (✅)** | 38 |
| **VLM image-path failures (❌, 0 tokens)** | 0 |

The remaining VLM-CSV rows are text-only models that fail warmup under an image prompt; their text-suite results are in the per-family tables above.

### Failing / skipped models

- **Warmup/bench failures:** `deepseek-v3-4bit`, `gemma-4-26b-a4b-it-4bit`, `GLM-5.1-4bit`, `qwen2.5-vl-3b`, `qwen2.5-vl-3b-4bit`
- **Zero tokens generated:** `paligemma2-3b-6bit`
- **OOM-skipped (capacity, not a real failure):** `Qwen3.5-397B-A17B-4bit`, `qwen3-next-480b-4bit`
