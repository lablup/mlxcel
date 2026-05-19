# Model Compatibility & Performance Tests (NVIDIA GB10)

Compatibility and performance testing for mlxcel models on **NVIDIA GB10 (DIGITS)**, running the CUDA backend.

## Test Environment

| Item | Value |
|------|-------|
| **Hardware** | NVIDIA GB10 (DIGITS), 122 GB Unified Memory |
| **OS** | Linux (aarch64), kernel 6.17 |
| **Backend** | CUDA 13.0 |
| **mlxcel version** | 0.0.27 |
| **MLX version** | 0.31.2 (via mlxcel-core) |
| **Test Prompt** | "Hello, how are you today?" (text) / "What is in this image?" (VLM) |
| **Max Tokens** | 100 |
| **Test Date** | 2026-05-19 |
| **CSV** | `benchmarks/cuda_gb10_2026-05-19.csv`, `benchmarks/cuda_gb10_vlm_2026-05-19.csv` |

## Legend

- ✅ Pass: Model generates 100 tokens cleanly
- ⚠️ Partial: Loads but generates fewer tokens than `max_tokens`, or output quality is suspect
- ❌ Fail: Does not work (warmup failure, OOM skip, or 0 tokens generated)

## Basic Transformers

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| llama-3.2-1b-4bit | Llama-3.2-1B-4bit | ⚠️ | 250.04 | 226.87 | 31 tokens |
| llama-3.1-8b-4bit | Llama-3.1-8B-Instruct-4bit | ✅ | 99.90 | 49.46 | 100 tokens |
| llama-3.1-8b-bf16 | Llama-3.1-8B-Instruct (bf16) | ⚠️ | 17.99 | 15.02 | 87 tokens |
| phi-2-4bit | phi-2-hf-4bit | ⚠️ | 8.50 | 3.64 | 1 token (likely EOS) |
| phi-3-mini-4bit | Phi-3-mini-4k-instruct-4bit | ⚠️ | 14.55 | 87.70 | 25 tokens |
| phi-3.5-mini-4bit | Phi-3.5-mini-instruct-4bit | ⚠️ | 13.01 | 55.82 | 40 tokens |
| phi-4-4bit | Phi-4-4bit | ✅ | 5.26 | 27.61 | 100 tokens |
| qwen2-0.5b | Qwen2.5-0.5B (bf16) | ✅ | 104.61 | 459.78 | |
| qwen2.5-0.5b-4bit | Qwen2.5-0.5B-Instruct-4bit | ✅ | 111.01 | 463.31 | |
| qwen2.5-0.5b-bf16 | Qwen2.5-0.5B (bf16) | ✅ | 59.93 | 200.52 | |
| qwen2.5-7b | Qwen2.5-7B (bf16) | ✅ | 18.12 | 53.93 | |
| qwen2.5-7b-4bit | Qwen2.5-7B-Instruct-4bit | ✅ | 17.39 | 54.18 | |
| qwen2.5-7b-8bit | Qwen2.5-7B-8bit | ✅ | 13.72 | 30.71 | |
| qwen3-0.6b | Qwen3-0.6B (bf16) | ⚠️ | 56.07 | 203.04 | 9 tokens (EOS) |
| qwen3-0.6b-4bit | Qwen3-0.6B-4bit | ⚠️ | 59.52 | 206.59 | 9 tokens |
| qwen3-1.7b-4bit | Qwen3-1.7B-4bit | ⚠️ | 37.54 | 139.13 | 14 tokens |
| qwen3-4b-4bit | Qwen3-4B-4bit | ⚠️ | 26.75 | 78.79 | 36 tokens |
| qwen3-8b-4bit | Qwen3-8B-4bit | ⚠️ | 18.17 | 48.01 | 33 tokens |
| smollm-135m-4bit | SmolLM-135M-Instruct-4bit | ✅ | 53.53 | 567.57 | |
| smollm3-3b-4bit | SmolLM3-3B-4bit | ⚠️ | 124.78 | 101.88 | 46 tokens |
| stablelm-1.6b-4bit | stablelm-2-1_6b-chat-4bit | ⚠️ | 42.33 | 186.64 | 71 tokens |
| starcoder2-3b-4bit | starcoder2-3b-4bit | ✅ | 7.55 | 102.47 | |
| olmo-1b-4bit | OLMo-1B-hf-4bit | ✅ | 12.40 | 97.95 | |
| olmo2-7b-4bit | OLMo2-7B-4bit | ⚠️ | 19.67 | 51.63 | 27 tokens |
| olmo3-32b-4bit | OLMo3-32B-4bit | ✅ | 10.15 | 11.63 | |
| minicpm-2b-4bit | MiniCPM-2B-sft-bf16-4bit | ✅ | 18.06 | 120.84 | |
| mimo-7b-4bit | MiMo-7B-RL-4bit | ✅ | 24.58 | 53.19 | |

## Gemma Family

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| gemma-2b-4bit | gemma-2b-it-4bit | ⚠️ | 16.02 | 99.45 | 49 tokens |
| gemma2-2b-4bit | gemma-2-2b-it-4bit | ⚠️ | 16.99 | 73.14 | 18 tokens |
| gemma3-1b-4bit | gemma-3-1b-it-4bit | ⚠️ | 25.58 | 182.97 | 34 tokens |
| gemma3-4b-4bit | gemma-3-4b-it-4bit | ⚠️ | 17.11 | 80.03 | 72 tokens |
| gemma3n-e2b-4bit | gemma-3n-E2B-it-4bit | ⚠️ | 13.05 | 75.41 | 68 tokens |
| gemma3n-e4b-4bit | gemma-3n-E4B-it-4bit | ⚠️ | 10.91 | 52.59 | 74 tokens |
| gemma3n-e4b-bf16 | gemma-3n-E4B-it (bf16) | ⚠️ | 3.28 | 21.64 | 69 tokens |

### Gemma 4

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| gemma-4-e2b-it-4bit | Gemma-4-E2B-it-4bit | ✅ | 48.36 | 99.24 | 100 tokens |
| gemma-4-e2b-it-8bit | Gemma-4-E2B-it-8bit | ✅ | 44.90 | 57.10 | |
| gemma-4-e4b-it-4bit | Gemma-4-E4B-it-4bit | ⚠️ | 46.35 | 46.37 | 36 tokens |
| gemma-4-e4b-it-8bit | Gemma-4-E4B-it-8bit | ⚠️ | 43.04 | 26.79 | 39 tokens |
| gemma-4-26b-a4b-it-4bit | Gemma-4-26B-A4B-it-4bit | ❌ | - | FAIL | warmup failure |
| gemma-4-31b-4bit | Gemma-4-31B-4bit | ✅ | 14.23 | 8.92 | |
| gemma-4-31b-it-4bit | Gemma-4-31B-it-4bit | ⚠️ | 34.84 | 8.47 | 26 tokens |
| gemma-4-31B-it-assistant-bf16 | Gemma-4-31B-it-assistant (bf16) | ❌ | - | FAIL | warmup failure |
| Gemma-4-31b-it-nvfp4 | Gemma-4-31B-it (NVFP4) | ⚠️ | 9.53 | 0.93 | 26 tokens; unusably slow |

## EXAONE

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| exaone-3.5-2.4b-4bit | EXAONE-3.5-2.4B-Instruct-4bit | ✅ | 73.09 | 104.06 | |
| exaone4-1.2b-4bit | exaone-4.0-1.2b-4bit | ⚠️ | 38.87 | 176.69 | 18 tokens |

## Cohere / Command R

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| command-r7b-4bit | c4ai-command-r7b-4bit | ✅ | 7.39 | 52.38 | |
| aya-expanse-8b-4bit | aya-expanse-8b-4bit | ✅ | 7.54 | 52.84 | |

## MoE (Mixture of Experts)

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| mixtral-8x7b-4bit | Mixtral-8x7B-Instruct-v0.1-4bit | ⚠️ | 2.31 | 28.05 | 73 tokens |
| qwen1.5-moe-a2.7b-4bit | Qwen1.5-MoE-A2.7B-Chat-4bit | ✅ | 7.36 | 106.99 | |
| qwen3-moe-4bit | Qwen3-30B-A3B-4bit | ⚠️ | 4.11 | 56.65 | 34 tokens |
| qwen3-30b-a3b-4bit | Qwen3-30B-A3B-4bit | ⚠️ | 3.92 | 56.81 | 34 tokens |
| phi-3.5-moe-4bit | Phi-3.5-MoE-instruct-4bit | ✅ | 1.25 | 50.71 | |
| minimax-m2-3bit | MiniMax-M2-3bit | ✅ | 0.28 | 22.14 | |
| gpt-oss-20b-mxfp4 | gpt-oss-20b-MXFP4 | ⚠️ | 25.77 | 76.30 | 58 tokens |
| gpt-oss-120b-4bit | gpt-oss-120b-4bit | ⚠️ | 1.10 | 48.70 | 65 tokens |
| deepseek-v2-lite-4bit | DeepSeek-V2-Lite-Chat-4bit | ✅ | 4.73 | 95.27 | |
| deepseek-v3-4bit | DeepSeek-V3-0324-4bit | ❌ | - | FAIL | missing weight file |
| llama-4-scout-17b-4bit | Llama-4-Scout-17B-16E-4bit | ✅ | 1.17 | 20.93 | 100 tokens |

## MLA (Multi-head Latent Attention)

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| minicpm3-4b-4bit | MiniCPM3-4B-4bit | ✅ | 19.06 | 50.09 | |

## DeepSeek Family

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| deepseek-coder-1.3b-4bit | deepseek-coder-1.3b-4bit | ✅ | 72.25 | 92.93 | |
| deepseek-r1-distill-7b-4bit | DeepSeek-R1-Distill-Qwen-7B-4bit | ✅ | 5.83 | 58.60 | |

## Nemotron Family

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| nemotron-h-30b-4bit | Nemotron-H-30B-4bit | ⚠️ | 4.34 | 25.75 | 46 tokens |
| nemotron-nas-30b-4bit | Nemotron-NAS-30B-A3B-4bit | ⚠️ | 3.96 | 28.35 | 46 tokens |

## SSM / Mamba / Hybrid Models

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| falcon-mamba-7b-4bit | Falcon-Mamba-7B-4bit | ⚠️ | 8.12 | 21.45 | 2 tokens (EOS) |
| mamba2-1.3b-4bit | mamba2-1.3b-4bit | ✅ | 7.45 | 81.02 | |
| jamba-v0.1-4bit | Jamba-v0.1-4bit | ✅ | 46.49 | 80.91 | 100 tokens |

## Chinese / Asian Language Models

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| baichuan-m1-14b-4bit | Baichuan-M1-14B-Instruct-4bit | ⚠️ | 2.51 | 26.74 | 39 tokens |
| glm4-flash-4bit | GLM-4-Flash-4bit | ✅ | 2.13 | 51.52 | 100 tokens |
| GLM-5.1-4bit | GLM-5.1-4bit | ❌ | - | FAIL | warmup failure |
| internlm2-7b-4bit | InternLM2-7B-4bit | ✅ | 18.72 | 50.65 | |
| internlm3-8b-4bit | internlm3-8b-instruct-4bit | ✅ | 24.59 | 44.14 | |
| ernie-4.5-0.3b-4bit | ERNIE-4.5-0.3B-Instruct-4bit | ✅ | 119.90 | 600.45 | |
| hunyuan-13b | Hunyuan-Large (bf16, 13B) | ✅ | 0.55 | 14.78 | |
| hunyuan-4bit | Hunyuan-Large-Instruct-4bit | ✅ | 0.92 | 14.70 | |
| hunyuan-dense-4bit | Hunyuan-1.8B-Instruct-4bit | ⚠️ | 28.81 | 150.94 | 41 tokens |
| hunyuan-1.8b-4bit | Hunyuan-1.8B-Instruct-4bit | ⚠️ | 27.03 | 149.45 | 41 tokens |
| hunyuan-large-4bit | Hunyuan-Large-Instruct-4bit | ✅ | 0.92 | 14.35 | |
| hunyuan-moe-a13b-bf16 | Hunyuan-MoE-A13B (bf16) | ✅ | 0.49 | 14.84 | |

## Mistral Family

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| ministral-3b-4bit | Ministral-3B-Instruct-4bit | ⚠️ | 922.10 | 91.05 | 34 tokens |
| mistral-small-3.1-24b-4bit | mistral-small-3.1-24b-4bit | ✅ | 1.95 | 16.28 | 100 tokens |
| pixtral-12b | pixtral-12b (bf16) | ✅ | 2.51 | 33.26 | |
| pixtral-12b-4bit | pixtral-12b-4bit | ✅ | 2.48 | 33.45 | 100 tokens |

## VLM-capable Models (text-only pass)

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| aya-vision-8b | aya-vision-8b | ⚠️ | 7.63 | 51.44 | 87 tokens |
| bunny-llama3-8b-4bit | Bunny-Llama-3-8B-V-4bit | ⚠️ | 18.12 | 52.40 | 40 tokens |
| llava-1.5-7b-4bit | llava-1.5-7b-4bit | ✅ | 9.12 | 55.83 | |
| llava-next-mistral-7b-4bit | llava-v1.6-mistral-7b-4bit | ✅ | 16.17 | 52.48 | |
| llava-interleave-qwen-0.5b-bf16 | llava-interleave-qwen-0.5b-bf16 | ⚠️ | 41.50 | 203.89 | 49 tokens |
| molmo2-4b | molmo2-4b | ⚠️ | 7.16 | 26.63 | 33 tokens |
| molmo-7b | Molmo-7B | ❌ | - | FAIL | warmup |
| paligemma2-3b-6bit | paligemma2-3b | ❌ | 6.98 | 0.00 | 0 tokens generated |
| phi-3.5-vision-4bit | Phi-3.5-vision-instruct-4bit | ⚠️ | 20.97 | 56.30 | 43 tokens |
| internvl3-1b | InternVL3-1B | ❌ | - | FAIL | warmup |
| qwen2-vl-2b | Qwen2-VL-2B (bf16) | ⚠️ | 44.43 | 97.37 | 35 tokens |
| qwen2-vl-2b-4bit | Qwen2-VL-2B-Instruct-4bit | ⚠️ | 46.07 | 96.64 | 35 tokens |
| qwen2.5-vl-3b | Qwen2.5-VL-3B (bf16) | ❌ | - | FAIL | warmup |
| qwen2.5-vl-3b-4bit | Qwen2.5-VL-3B-Instruct-4bit | ❌ | - | FAIL | warmup |
| qwen3-vl-2b | Qwen3-VL-2B (bf16) | ⚠️ | 29.83 | 161.91 | 59 tokens |
| qwen3-vl-2b-4bit | Qwen3-VL-2B-Instruct-4bit | ⚠️ | 30.45 | 162.74 | 59 tokens |
| qwen3-vl-30b-a3b-4bit | Qwen3-VL-30B-A3B-4bit | ⚠️ | 2.97 | 55.81 | 34 tokens |
| qwen3-vl-32b-4bit | Qwen3-VL-32B-4bit | ⚠️ | 2.70 | 11.00 | 37 tokens |

## Qwen3.5 / Qwen3-next (new architectures)

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| qwen3.5-0.8b-4bit | Qwen3.5-0.8B-4bit | ⚠️ | 28.15 | 147.92 | 18 tokens |
| qwen3.5-2b-4bit | Qwen3.5-2B-4bit | ⚠️ | 24.00 | 111.43 | 31 tokens |
| qwen3.5-4b-4bit | Qwen3.5-4B-4bit | ⚠️ | 18.97 | 59.28 | 31 tokens |
| qwen3.5-9b-4bit | Qwen3.5-9B-4bit | ⚠️ | 12.64 | 37.93 | 31 tokens |
| qwen3.5-9b-bf16 | Qwen3.5-9B (bf16) | ⚠️ | 3.21 | 13.02 | 31 tokens |
| qwen3.5-27b-4bit | Qwen3.5-27B-4bit | ⚠️ | 3.42 | 12.17 | 30 tokens |
| qwen3.5-35b-a3b-4bit | Qwen3.5-35B-A3B-4bit | ⚠️ | 3.57 | 47.35 | 31 tokens |
| Qwen3.5-4B-DFlash | Qwen3.5-4B-DFlash | ❌ | - | FAIL | warmup failure |
| Qwen3.5-397B-A17B-4bit | Qwen3.5-397B-A17B-4bit | ❌ | - | SKIP | skipped (OOM estimate) |
| qwen3-next-480b-4bit | Qwen3-next-480B-4bit | ❌ | - | SKIP | skipped (OOM estimate) |
| solar-open-100b-4bit | Solar-Open-100B-4bit | ⚠️ | 57.78 | 18.88 | 100 tokens |

## VLM Benchmark (image input)

| Model | Test Model | Status | Generated Tokens | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|------------------|-----------------|----------------|-------|
| aya-vision-8b | aya-vision-8b | ✅ | 100 | 101.13 | 39.64 | |
| bunny-llama3-8b-4bit | Bunny-Llama-3-8B-V-4bit | ⚠️ | 37 | 453.73 | 36.15 | |
| gemma3-4b-4bit | gemma-3-4b-it-4bit | ⚠️ | 16 | 201.51 | 64.42 | |
| gemma3n-e2b-4bit | gemma-3n-E2B-it-4bit | ⚠️ | 29 | 144.98 | 67.64 | |
| gemma3n-e4b-4bit | gemma-3n-E4B-it-4bit | ⚠️ | 33 | 132.96 | 45.18 | |
| gemma3n-e4b-bf16 | gemma-3n-E4B-it (bf16) | ⚠️ | 24 | 52.54 | 20.38 | |
| gemma-4-31b-4bit | Gemma-4-31B-4bit | ⚠️ | 8 | 214.15 | 7.58 | VLM-mode result |
| gemma-4-31b-it-4bit | Gemma-4-31B-it-4bit | ⚠️ | 27 | 225.56 | 8.36 | VLM-mode result |
| gemma-4-e2b-it-4bit | Gemma-4-E2B-it-4bit | ⚠️ | 20 | 486.92 | 82.05 | 20 tokens |
| gemma-4-e2b-it-8bit | Gemma-4-E2B-it-8bit | ⚠️ | 11 | 466.89 | 44.77 | 11 tokens |
| gemma-4-e4b-it-4bit | Gemma-4-E4B-it-4bit | ⚠️ | 47 | 443.17 | 46.52 | |
| gemma-4-e4b-it-8bit | Gemma-4-E4B-it-8bit | ⚠️ | 11 | 414.08 | 23.92 | |
| llama-4-scout-17b-4bit | Llama-4-Scout-17B-16E-4bit | ⚠️ | 60 | 3.59 | 19.66 | |
| llava-1.5-7b-4bit | llava-1.5-7b-4bit | ✅ | 100 | 410.70 | 52.33 | |
| llava-interleave-qwen-0.5b-bf16 | llava-interleave-qwen-0.5b-bf16 | ⚠️ | 32 | 843.42 | 147.49 | |
| llava-next-mistral-7b-4bit | llava-v1.6-mistral-7b-4bit | ✅ | 100 | 388.13 | 51.66 | |
| ministral-3b-4bit | Ministral-3B-Instruct-4bit | ✅ | 100 | 1668.44 | 86.69 | |
| mistral-small-3.1-24b-4bit | mistral-small-3.1-24b-4bit | ✅ | 100 | 317.94 | 15.52 | |
| molmo2-4b | molmo2-4b | ⚠️ | 46 | 169.20 | 26.54 | |
| paligemma2-3b-6bit | paligemma2-3b | ⚠️ | 2 | 545.97 | 21.69 | |
| phi-3.5-vision-4bit | Phi-3.5-vision-instruct-4bit | ⚠️ | 19 | 426.23 | 45.33 | |
| pixtral-12b | pixtral-12b (bf16) | ✅ | 100 | 527.64 | 30.29 | |
| pixtral-12b-4bit | pixtral-12b-4bit | ✅ | 100 | 545.24 | 30.38 | VLM-mode result |
| qwen2-vl-2b | Qwen2-VL-2B (bf16) | ❌ | 0 | 26.01 | 0.00 | 0 tokens generated |
| qwen2-vl-2b-4bit | Qwen2-VL-2B-Instruct-4bit | ❌ | 0 | 25.42 | 0.00 | 0 tokens generated |
| qwen3-vl-2b | Qwen3-VL-2B (bf16) | ✅ | 100 | 34.43 | 131.32 | VLM-mode result |
| qwen3-vl-2b-4bit | Qwen3-VL-2B-Instruct-4bit | ✅ | 100 | 30.85 | 126.26 | VLM-mode result |
| qwen3-vl-30b-a3b-4bit | Qwen3-VL-30B-A3B-4bit | ❌ | 0 | 2.95 | 0.00 | 0 tokens generated (decode timing out) |
| qwen3-vl-32b-4bit | Qwen3-VL-32B-4bit | ⚠️ | 100 | 2.97 | 10.76 | VLM-mode result |
| qwen3.5-27b-4bit | Qwen3.5-27B-4bit | ✅ | 100 | 3.58 | 12.72 | |
| qwen3.5-35b-a3b-4bit | Qwen3.5-35B-A3B-4bit | ✅ | 100 | 3.63 | 48.15 | |
| qwen3.5-9b-bf16 | Qwen3.5-9B (bf16) | ⚠️ | 77 | 2.83 | 13.28 | |
| qwen3.5-2b-4bit | Qwen3.5-2B-4bit | ✅ | 100 | 17.43 | 125.51 | VLM-mode result |
| qwen3.5-4b-4bit | Qwen3.5-4B-4bit | ✅ | 100 | 15.42 | 63.07 | VLM-mode result |
| qwen3.5-9b-4bit | Qwen3.5-9B-4bit | ⚠️ | 50 | 11.22 | 38.30 | VLM-mode result |
| qwen3.5-0.8b-4bit | Qwen3.5-0.8B-4bit | ⚠️ | 70 | 23.21 | 182.11 | VLM-mode result |

> The VLM sweep treats every model directory as a candidate, so non-VLM models that load the image preprocessor without error appear here. Pure text models that warmup-fail in VLM mode (most of the suite) are recorded as `FAIL:warmup` in the CSV and omitted from this table.

---

## Summary

**Test date**: 2026-05-19 | **Hardware**: NVIDIA GB10 (DIGITS) | **mlxcel**: 0.0.27 | **MLX**: 0.31.2

| Metric | Count |
|--------|-------|
| **Total text models attempted** | 111 |
| **Pass (✅)** | 41 |
| **Partial (⚠️)** | 56 |
| **Fail (❌)** | 14 |
| **VLM models attempted (image)** | 111 (full sweep) |
| **VLM Pass (✅)** | 13 |
| **VLM Partial (⚠️)** | 19 |
| **VLM Fail (❌, image path)** | 3 (qwen2-vl 2B fp16/4bit, qwen3-vl-30b-a3b) |

The remaining ~76 rows in the VLM CSV are text-only models that fail warmup in VLM mode. Their text-suite results are above in the per-family tables.

### Current-state observations

- GB10 text coverage is broad but many rows are partial because the model exits before the requested 100 tokens.
- Small dense models and several VLM-capable text paths produce the highest decode rates, with `ernie-4.5-0.3b-4bit`, `smollm-135m-4bit`, and Qwen2.5 0.5B variants leading the table.
- Large MoE and hybrid models run, but decode throughput is much lower than on Apple Silicon for this campaign.
- The VLM image sweep has 13 clean passes, 19 partial rows, and 3 image-path failures; text-only warmup failures from the full VLM candidate sweep are omitted from the table.

### Failing models (14 total)

Real CUDA-specific or model-specific failures:

- `deepseek-v3-4bit`: missing weight file
- `gemma-4-26b-a4b-it-4bit`: warmup failure
- `gemma-4-31B-it-assistant-bf16`: warmup failure
- `GLM-5.1-4bit`: warmup failure
- `Qwen3.5-4B-DFlash`: warmup failure
- `internvl3-1b`, `molmo-7b`: warmup, model-specific issues
- `paligemma2-3b-6bit`: loads but 0 tokens generated

Skipped (OOM estimate, not real failure):

- `Qwen3.5-397B-A17B-4bit`
- `qwen3-next-480b-4bit`
