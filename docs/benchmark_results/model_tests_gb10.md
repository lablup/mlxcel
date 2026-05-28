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
| **Previous Benchmark** | 2026-05-19 (mlxcel 0.0.27) |
| **CSV** | `benchmarks/cuda_gb10_2026-05-28.csv`, `benchmarks/cuda_gb10_vlm_2026-05-28.csv` |

> **Prefill is not comparable to 2026-05-19.** Commit `c9a77f2` ("align benchmark warmup with measured prefill") landed after the 2026-05-19 sweep, so that run's prefill column still included cold-start / CUDA-JIT overhead (e.g. 76 prompt tokens measured at 1052 ms). The 2026-05-28 column reports warm steady-state prefill (the same 76 tokens at 16 ms), which is why prefill jumps 10-60x across the board. This is a measurement correction, not a kernel speedup, and it brings GB10 in line with the M1 Ultra / M5 Max refreshes. Decode is measured identically across both runs and is the metric used for all "vs 0519" comparisons below.

## Legend

- ✅ Pass: generates 100 tokens cleanly
- ⚠️ Partial: loads but generates fewer tokens than `max_tokens`, or output quality is suspect
- ❌ Fail: warmup/bench failure, OOM skip, or 0 tokens generated

## Basic Transformers

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| llama-3.2-1b-4bit | Llama-3.2-1B-4bit | ⚠️ | 6858.95 | 253.63 | 31 tok; +12% decode vs 0519 (226.87) |
| llama-3.1-8b-4bit | Llama-3.1-8B-Instruct-4bit | ✅ | 1361.89 | 49.15 | -1% decode vs 0519 (49.46) |
| llama-3.1-8b-bf16 | Llama-3.1-8B-Instruct (bf16) | ⚠️ | 1208.56 | 14.81 | 87 tok; -1% decode vs 0519 (15.02) |
| phi-2-4bit | phi-2-hf-4bit | ⚠️ | 135.15 | 36.49 | 1 tok; +902% decode vs 0519 (3.64) |
| phi-3-mini-4bit | Phi-3-mini-4k-instruct-4bit | ⚠️ | 280.73 | 93.19 | 25 tok; +6% decode vs 0519 (87.70) |
| phi-3.5-mini-4bit | Phi-3.5-mini-instruct-4bit | ⚠️ | 290.58 | 92.50 | 40 tok; +66% decode vs 0519 (55.82) |
| phi-4-4bit | Phi-4-4bit | ✅ | 161.06 | 27.52 | -0% decode vs 0519 (27.61) |
| qwen2-0.5b | Qwen2.5-0.5B (bf16) | ✅ | 3870.91 | 496.44 | +8% decode vs 0519 (459.78) |
| qwen2.5-0.5b-4bit | Qwen2.5-0.5B-Instruct-4bit | ✅ | 3705.50 | 502.51 | +8% decode vs 0519 (463.31) |
| qwen2.5-0.5b-bf16 | Qwen2.5-0.5B (bf16) | ✅ | 3136.60 | 202.87 | +1% decode vs 0519 (200.52) |
| qwen2.5-7b | Qwen2.5-7B (bf16) | ✅ | 634.86 | 54.07 | +0% decode vs 0519 (53.93) |
| qwen2.5-7b-4bit | Qwen2.5-7B-Instruct-4bit | ✅ | 617.25 | 53.73 | -1% decode vs 0519 (54.18) |
| qwen2.5-7b-8bit | Qwen2.5-7B-8bit | ✅ | 684.10 | 29.98 | -2% decode vs 0519 (30.71) |
| qwen3-0.6b | Qwen3-0.6B (bf16) | ⚠️ | 2021.77 | 317.75 | 9 tok; +56% decode vs 0519 (203.04) |
| qwen3-0.6b-4bit | Qwen3-0.6B-4bit | ⚠️ | 1956.00 | 314.62 | 9 tok; +52% decode vs 0519 (206.59) |
| qwen3-1.7b-4bit | Qwen3-1.7B-4bit | ⚠️ | 1134.38 | 167.90 | 14 tok; +21% decode vs 0519 (139.13) |
| qwen3-4b-4bit | Qwen3-4B-4bit | ⚠️ | 488.30 | 81.37 | 36 tok; +3% decode vs 0519 (78.79) |
| qwen3-8b-4bit | Qwen3-8B-4bit | ⚠️ | 252.73 | 48.71 | 33 tok; +1% decode vs 0519 (48.01) |
| smollm-135m-4bit | SmolLM-135M-Instruct-4bit | ✅ | 3001.35 | 643.04 | +13% decode vs 0519 (567.57) |
| smollm3-3b-4bit | SmolLM3-3B-4bit | ⚠️ | 1628.18 | 100.66 | 18 tok; -1% decode vs 0519 (101.88) |
| stablelm-1.6b-4bit | stablelm-2-1_6b-chat-4bit | ✅ | 1546.33 | 197.05 | +6% decode vs 0519 (186.64) |
| starcoder2-3b-4bit | starcoder2-3b-4bit | ✅ | 220.65 | 102.42 | -0% decode vs 0519 (102.47) |
| olmo-1b-4bit | OLMo-1B-hf-4bit | ✅ | 262.62 | 98.26 | +0% decode vs 0519 (97.95) |
| olmo2-7b-4bit | OLMo2-7B-4bit | ⚠️ | 316.99 | 53.17 | 27 tok; +3% decode vs 0519 (51.63) |
| olmo3-32b-4bit | OLMo3-32B-4bit | ✅ | 309.94 | 11.70 | +1% decode vs 0519 (11.63) |
| minicpm-2b-4bit | MiniCPM-2B-sft-bf16-4bit | ✅ | 434.56 | 122.27 | +1% decode vs 0519 (120.84) |
| mimo-7b-4bit | MiMo-7B-RL-4bit | ✅ | 358.62 | 53.33 | +0% decode vs 0519 (53.19) |

## Gemma Family

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| gemma-2b-4bit | gemma-2b-it-4bit | ⚠️ | 601.35 | 100.06 | 41 tok; +1% decode vs 0519 (99.45) |
| gemma2-2b-4bit | gemma-2-2b-it-4bit | ⚠️ | 665.31 | 117.38 | 27 tok; +60% decode vs 0519 (73.14) |
| gemma3-1b-4bit | gemma-3-1b-it-4bit | ⚠️ | 977.29 | 256.48 | 34 tok; +40% decode vs 0519 (182.97) |
| gemma3-4b-4bit | gemma-3-4b-it-4bit | ⚠️ | 392.79 | 80.17 | 72 tok; +0% decode vs 0519 (80.03) |
| gemma3n-e2b-4bit | gemma-3n-E2B-it-4bit | ⚠️ | 415.19 | 81.83 | 68 tok; +9% decode vs 0519 (75.41) |
| gemma3n-e4b-4bit | gemma-3n-E4B-it-4bit | ⚠️ | 260.59 | 53.53 | 74 tok; +2% decode vs 0519 (52.59) |
| gemma3n-e4b-bf16 | gemma-3n-E4B-it (bf16) | ⚠️ | 273.15 | 21.61 | 69 tok; -0% decode vs 0519 (21.64) |

### Gemma 4

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| gemma-4-e2b-it-4bit | Gemma-4-E2B-it-4bit | ⚠️ | 707.60 | 98.70 | 28 tok; -1% decode vs 0519 (99.24) |
| gemma-4-e2b-it-8bit | Gemma-4-E2B-it-8bit | ✅ | 522.63 | 58.24 | +2% decode vs 0519 (57.10) |
| gemma-4-e4b-it-4bit | Gemma-4-E4B-it-4bit | ⚠️ | 325.35 | 47.58 | 33 tok; +3% decode vs 0519 (46.37) |
| gemma-4-e4b-it-8bit | Gemma-4-E4B-it-8bit | ⚠️ | 272.71 | 27.22 | 33 tok; +2% decode vs 0519 (26.79) |
| gemma-4-26b-a4b-it-4bit | Gemma-4-26B-A4B-it-4bit | ❌ | - | FAIL | warmup failure |
| gemma-4-31b-4bit | Gemma-4-31B-4bit | ✅ | 23.32 | 8.79 | -1% decode vs 0519 (8.92) |
| gemma-4-31b-it-4bit | Gemma-4-31B-it-4bit | ⚠️ | 48.97 | 8.06 | 26 tok; -5% decode vs 0519 (8.47) |
| Gemma-4-31b-it-nvfp4 | Gemma-4-31B-it (NVFP4) | ⚠️ | 16.52 | 0.90 | 26 tok; -3% decode vs 0519 (0.93) |

## EXAONE

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| exaone-3.5-2.4b-4bit | EXAONE-3.5-2.4B-Instruct-4bit | ✅ | 1391.24 | 146.48 | +41% decode vs 0519 (104.06) |
| exaone4-1.2b-4bit | exaone-4.0-1.2b-4bit | ⚠️ | 1136.29 | 225.62 | 53 tok; +28% decode vs 0519 (176.69) |

## Cohere / Command R

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| command-r7b-4bit | c4ai-command-r7b-4bit | ✅ | 124.23 | 52.12 | -0% decode vs 0519 (52.38) |
| aya-expanse-8b-4bit | aya-expanse-8b-4bit | ✅ | 159.55 | 52.89 | +0% decode vs 0519 (52.84) |

## MoE (Mixture of Experts)

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| mixtral-8x7b-4bit | Mixtral-8x7B-Instruct-v0.1-4bit | ⚠️ | 12.60 | 28.00 | 73 tok; -0% decode vs 0519 (28.05) |
| qwen1.5-moe-a2.7b-4bit | Qwen1.5-MoE-A2.7B-Chat-4bit | ✅ | 248.96 | 112.09 | +5% decode vs 0519 (106.99) |
| qwen3-moe-4bit | Qwen3-30B-A3B-4bit | ⚠️ | 133.23 | 57.49 | 33 tok; +1% decode vs 0519 (56.65) |
| qwen3-30b-a3b-4bit | Qwen3-30B-A3B-4bit | ⚠️ | 134.11 | 53.55 | 34 tok; -6% decode vs 0519 (56.81) |
| phi-3.5-moe-4bit | Phi-3.5-MoE-instruct-4bit | ✅ | 28.99 | 51.35 | +1% decode vs 0519 (50.71) |
| minimax-m2-3bit | MiniMax-M2-3bit | ✅ | 26.72 | 21.85 | -1% decode vs 0519 (22.14) |
| gpt-oss-20b-mxfp4 | gpt-oss-20b-MXFP4 | ✅ | 126.41 | 77.94 | +2% decode vs 0519 (76.30) |
| gpt-oss-120b-4bit | gpt-oss-120b-4bit | ⚠️ | 54.57 | 50.63 | 73 tok; +4% decode vs 0519 (48.70) |
| deepseek-v2-lite-4bit | DeepSeek-V2-Lite-Chat-4bit | ✅ | 156.31 | 99.07 | +4% decode vs 0519 (95.27) |
| deepseek-v3-4bit | DeepSeek-V3-0324-4bit | ❌ | - | FAIL | warmup failure |
| llama-4-scout-17b-4bit | Llama-4-Scout-17B-16E-4bit | ✅ | 27.67 | 20.88 | -0% decode vs 0519 (20.93) |

## MLA (Multi-head Latent Attention)

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| minicpm3-4b-4bit | MiniCPM3-4B-4bit | ✅ | 282.58 | 57.55 | +15% decode vs 0519 (50.09) |

## DeepSeek Family

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| deepseek-coder-1.3b-4bit | deepseek-coder-1.3b-4bit | ✅ | 4655.97 | 92.61 | -0% decode vs 0519 (92.93) |
| deepseek-r1-distill-7b-4bit | DeepSeek-R1-Distill-Qwen-7B-4bit | ✅ | 210.07 | 58.55 | -0% decode vs 0519 (58.60) |

## Nemotron Family

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| nemotron-h-30b-4bit | Nemotron-H-30B-4bit | ⚠️ | 108.15 | 32.92 | 46 tok; +28% decode vs 0519 (25.75) |
| nemotron-nas-30b-4bit | Nemotron-NAS-30B-A3B-4bit | ⚠️ | 105.00 | 32.98 | 46 tok; +16% decode vs 0519 (28.35) |

## SSM / Mamba / Hybrid Models

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| falcon-mamba-7b-4bit | Falcon-Mamba-7B-4bit | ⚠️ | 83.89 | 22.09 | 2 tok; +3% decode vs 0519 (21.45) |
| mamba2-1.3b-4bit | mamba2-1.3b-4bit | ✅ | 277.75 | 80.50 | -1% decode vs 0519 (81.02) |
| jamba-v0.1-4bit | Jamba-v0.1-4bit | ✅ | 529.88 | 85.42 | +6% decode vs 0519 (80.91) |

## Chinese / Asian Language Models

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| baichuan-m1-14b-4bit | Baichuan-M1-14B-Instruct-4bit | ⚠️ | 75.66 | 24.06 | 7 tok; -10% decode vs 0519 (26.74) |
| glm4-flash-4bit | GLM-4-Flash-4bit | ✅ | 102.70 | 55.04 | +7% decode vs 0519 (51.52) |
| GLM-5.1-4bit | GLM-5.1-4bit | ❌ | - | FAIL | warmup failure |
| internlm2-7b-4bit | InternLM2-7B-4bit | ✅ | 387.86 | 50.26 | -1% decode vs 0519 (50.65) |
| internlm3-8b-4bit | internlm3-8b-instruct-4bit | ✅ | 530.75 | 43.89 | -1% decode vs 0519 (44.14) |
| ernie-4.5-0.3b-4bit | ERNIE-4.5-0.3B-Instruct-4bit | ✅ | 5403.22 | 682.24 | +14% decode vs 0519 (600.45) |
| hunyuan-13b | Hunyuan-Large (bf16, 13B) | ✅ | 19.31 | 15.15 | +3% decode vs 0519 (14.78) |
| hunyuan-4bit | Hunyuan-Large-Instruct-4bit | ✅ | 19.87 | 14.86 | +1% decode vs 0519 (14.70) |
| hunyuan-dense-4bit | Hunyuan-1.8B-Instruct-4bit | ⚠️ | 668.84 | 158.90 | 41 tok; +5% decode vs 0519 (150.94) |
| hunyuan-1.8b-4bit | Hunyuan-1.8B-Instruct-4bit | ⚠️ | 732.00 | 157.78 | 41 tok; +6% decode vs 0519 (149.45) |
| hunyuan-large-4bit | Hunyuan-Large-Instruct-4bit | ✅ | 20.04 | 15.10 | +5% decode vs 0519 (14.35) |
| hunyuan-moe-a13b-bf16 | Hunyuan-MoE-A13B (bf16) | ✅ | 19.00 | 14.79 | -0% decode vs 0519 (14.84) |

## Mistral Family

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| ministral-3b-4bit | Ministral-3B-Instruct-4bit | ⚠️ | 6316.20 | 101.17 | 34 tok; +11% decode vs 0519 (91.05) |
| mistral-small-3.1-24b-4bit | mistral-small-3.1-24b-4bit | ✅ | 65.18 | 16.08 | -1% decode vs 0519 (16.28) |
| pixtral-12b | pixtral-12b (bf16) | ✅ | 36.69 | 33.09 | -1% decode vs 0519 (33.26) |
| pixtral-12b-4bit | pixtral-12b-4bit | ✅ | 36.00 | 33.06 | -1% decode vs 0519 (33.45) |

## VLM-capable Models (text-only pass)

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| aya-vision-8b | aya-vision-8b | ✅ | 117.37 | 52.16 | +1% decode vs 0519 (51.44) |
| bunny-llama3-8b-4bit | Bunny-Llama-3-8B-V-4bit | ⚠️ | 380.57 | 52.89 | 40 tok; +1% decode vs 0519 (52.40) |
| llava-1.5-7b-4bit | llava-1.5-7b-4bit | ✅ | 207.24 | 56.02 | +0% decode vs 0519 (55.83) |
| llava-next-mistral-7b-4bit | llava-v1.6-mistral-7b-4bit | ✅ | 371.54 | 52.64 | +0% decode vs 0519 (52.48) |
| llava-interleave-qwen-0.5b-bf16 | llava-interleave-qwen-0.5b-bf16 | ⚠️ | 2448.92 | 212.61 | 49 tok; +4% decode vs 0519 (203.89) |
| molmo2-4b | molmo2-4b | ⚠️ | 188.38 | 26.76 | 33 tok; +0% decode vs 0519 (26.63) |
| molmo-7b | Molmo-7B | ⚠️ | 212.01 | 33.62 | 24 tok; recovered |
| paligemma2-3b-6bit | paligemma2-3b | ❌ | 164.80 | 0.00 | 0 tokens generated |
| phi-3.5-vision-4bit | Phi-3.5-vision-instruct-4bit | ⚠️ | 426.67 | 91.41 | 43 tok; +62% decode vs 0519 (56.30) |
| internvl3-1b | InternVL3-1B | ⚠️ | 4041.24 | 479.58 | 37 tok; recovered |
| qwen2-vl-2b | Qwen2-VL-2B (bf16) | ⚠️ | 670.87 | 101.92 | 35 tok; +5% decode vs 0519 (97.37) |
| qwen2-vl-2b-4bit | Qwen2-VL-2B-Instruct-4bit | ⚠️ | 683.57 | 101.28 | 35 tok; +5% decode vs 0519 (96.64) |
| qwen2.5-vl-3b | Qwen2.5-VL-3B (bf16) | ❌ | - | FAIL | warmup failure |
| qwen2.5-vl-3b-4bit | Qwen2.5-VL-3B-Instruct-4bit | ❌ | - | FAIL | warmup failure |
| qwen3-vl-2b | Qwen3-VL-2B (bf16) | ⚠️ | 848.87 | 166.97 | 61 tok; +3% decode vs 0519 (161.91) |
| qwen3-vl-2b-4bit | Qwen3-VL-2B-Instruct-4bit | ⚠️ | 754.03 | 165.01 | 33 tok; +1% decode vs 0519 (162.74) |
| qwen3-vl-30b-a3b-4bit | Qwen3-VL-30B-A3B-4bit | ⚠️ | 126.26 | 56.10 | 34 tok; +1% decode vs 0519 (55.81) |
| qwen3-vl-32b-4bit | Qwen3-VL-32B-4bit | ⚠️ | 86.60 | 10.92 | 37 tok; -1% decode vs 0519 (11.00) |

## Qwen3.5 / Qwen3-next (new architectures)

| Model | Test Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|-----------------|----------------|-------|
| qwen3.5-0.8b-4bit | Qwen3.5-0.8B-4bit | ⚠️ | 632.45 | 172.27 | 18 tok; +16% decode vs 0519 (147.92) |
| qwen3.5-2b-4bit | Qwen3.5-2B-4bit | ⚠️ | 512.33 | 127.49 | 31 tok; +14% decode vs 0519 (111.43) |
| qwen3.5-4b-4bit | Qwen3.5-4B-4bit | ⚠️ | 250.83 | 63.05 | 31 tok; +6% decode vs 0519 (59.28) |
| qwen3.5-9b-4bit | Qwen3.5-9B-4bit | ⚠️ | 164.08 | 39.19 | 31 tok; +3% decode vs 0519 (37.93) |
| qwen3.5-9b-bf16 | Qwen3.5-9B (bf16) | ⚠️ | 163.30 | 13.03 | 31 tok; +0% decode vs 0519 (13.02) |
| qwen3.5-27b-4bit | Qwen3.5-27B-4bit | ⚠️ | 59.42 | 12.36 | 30 tok; +2% decode vs 0519 (12.17) |
| qwen3.5-35b-a3b-4bit | Qwen3.5-35B-A3B-4bit | ⚠️ | 144.11 | 48.71 | 31 tok; +3% decode vs 0519 (47.35) |
| Qwen3.5-397B-A17B-4bit | Qwen3.5-397B-A17B-4bit | ❌ | - | SKIP | OOM skip (capacity) |
| qwen3-next-480b-4bit | Qwen3-next-480B-4bit | ❌ | - | SKIP | OOM skip (capacity) |
| solar-open-100b-4bit | Solar-Open-100B-4bit | ✅ | 49.53 | 18.52 | -2% decode vs 0519 (18.88) |

## VLM Benchmark (image input)

| Model | Test Model | Status | Generated Tokens | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|------------|--------|------------------|-----------------|----------------|-------|
| aya-vision-8b | aya-vision-8b | ✅ | 100 | 673.71 | 38.94 | -2% decode vs 0519 (39.64) |
| bunny-llama3-8b-4bit | Bunny-Llama-3-8B-V-4bit | ⚠️ | 37 | 1680.92 | 38.30 | 37 tok; +6% decode vs 0519 (36.15) |
| gemma3-4b-4bit | gemma-3-4b-it-4bit | ⚠️ | 14 | 1012.14 | 69.83 | 14 tok; +8% decode vs 0519 (64.42) |
| gemma3n-e2b-4bit | gemma-3n-E2B-it-4bit | ⚠️ | 29 | 2223.72 | 74.87 | 29 tok; +11% decode vs 0519 (67.64) |
| gemma3n-e4b-4bit | gemma-3n-E4B-it-4bit | ⚠️ | 33 | 1422.82 | 51.44 | 33 tok; +14% decode vs 0519 (45.18) |
| gemma3n-e4b-bf16 | gemma-3n-E4B-it (bf16) | ⚠️ | 24 | 2014.06 | 20.96 | 24 tok; +3% decode vs 0519 (20.38) |
| gemma-4-31b-4bit | Gemma-4-31B-4bit | ⚠️ | 8 | 322.22 | 7.82 | 8 tok; +3% decode vs 0519 (7.58) |
| gemma-4-31b-it-4bit | Gemma-4-31B-it-4bit | ⚠️ | 27 | 331.15 | 8.46 | 27 tok; +1% decode vs 0519 (8.36) |
| gemma-4-e2b-it-4bit | Gemma-4-E2B-it-4bit | ⚠️ | 20 | 2510.79 | 95.14 | 20 tok; +16% decode vs 0519 (82.05) |
| gemma-4-e2b-it-8bit | Gemma-4-E2B-it-8bit | ⚠️ | 7 | 2135.14 | 48.47 | 7 tok; +8% decode vs 0519 (44.77) |
| gemma-4-e4b-it-4bit | Gemma-4-E4B-it-4bit | ⚠️ | 47 | 1321.27 | 47.15 | 47 tok; +1% decode vs 0519 (46.52) |
| gemma-4-e4b-it-8bit | Gemma-4-E4B-it-8bit | ⚠️ | 11 | 1117.64 | 25.43 | 11 tok; +6% decode vs 0519 (23.92) |
| llama-4-scout-17b-4bit | Llama-4-Scout-17B-16E-4bit | ✅ | 100 | 28.39 | 20.65 | +5% decode vs 0519 (19.66) |
| llava-1.5-7b-4bit | llava-1.5-7b-4bit | ✅ | 100 | 1704.28 | 53.27 | +2% decode vs 0519 (52.33) |
| llava-interleave-qwen-0.5b-bf16 | llava-interleave-qwen-0.5b-bf16 | ⚠️ | 32 | 19737.74 | 191.14 | 32 tok; +30% decode vs 0519 (147.49) |
| llava-next-mistral-7b-4bit | llava-v1.6-mistral-7b-4bit | ✅ | 100 | 2170.28 | 52.26 | +1% decode vs 0519 (51.66) |
| ministral-3b-4bit | Ministral-3B-Instruct-4bit | ✅ | 100 | 2529.43 | 90.70 | +5% decode vs 0519 (86.69) |
| mistral-small-3.1-24b-4bit | mistral-small-3.1-24b-4bit | ✅ | 100 | 473.60 | 15.59 | +0% decode vs 0519 (15.52) |
| molmo2-4b | molmo2-4b | ⚠️ | 46 | 762.92 | 26.60 | 46 tok; +0% decode vs 0519 (26.54) |
| paligemma2-3b-6bit | paligemma2-3b | ⚠️ | 2 | 1846.20 | 42.61 | 2 tok; +96% decode vs 0519 (21.69) |
| phi-3.5-vision-4bit | Phi-3.5-vision-instruct-4bit | ⚠️ | 19 | 1383.21 | 80.90 | 19 tok; +78% decode vs 0519 (45.33) |
| pixtral-12b | pixtral-12b (bf16) | ✅ | 100 | 562.66 | 29.86 | -1% decode vs 0519 (30.29) |
| pixtral-12b-4bit | pixtral-12b-4bit | ✅ | 100 | 749.72 | 30.63 | +1% decode vs 0519 (30.38) |
| qwen2-vl-2b | Qwen2-VL-2B (bf16) | ⚠️ | 12 | 677.83 | 90.87 | 12 tok; recovered |
| qwen2-vl-2b-4bit | Qwen2-VL-2B-Instruct-4bit | ⚠️ | 12 | 696.50 | 92.67 | 12 tok; recovered |
| qwen3-vl-2b | Qwen3-VL-2B (bf16) | ⚠️ | 84 | 2192.64 | 131.85 | 84 tok; +0% decode vs 0519 (131.32) |
| qwen3-vl-2b-4bit | Qwen3-VL-2B-Instruct-4bit | ⚠️ | 80 | 2260.71 | 132.73 | 80 tok; +5% decode vs 0519 (126.26) |
| qwen3-vl-30b-a3b-4bit | Qwen3-VL-30B-A3B-4bit | ⚠️ | 72 | 184.85 | 45.15 | 72 tok; recovered |
| qwen3-vl-32b-4bit | Qwen3-VL-32B-4bit | ⚠️ | 55 | 158.46 | 10.40 | 55 tok; -3% decode vs 0519 (10.76) |
| qwen3.5-27b-4bit | Qwen3.5-27B-4bit | ✅ | 100 | 103.03 | 12.75 | +0% decode vs 0519 (12.72) |
| qwen3.5-35b-a3b-4bit | Qwen3.5-35B-A3B-4bit | ✅ | 100 | 179.45 | 45.23 | -6% decode vs 0519 (48.15) |
| qwen3.5-9b-bf16 | Qwen3.5-9B (bf16) | ✅ | 100 | 347.94 | 13.58 | +2% decode vs 0519 (13.28) |
| qwen3.5-2b-4bit | Qwen3.5-2B-4bit | ⚠️ | 47 | 719.30 | 123.77 | 47 tok; -1% decode vs 0519 (125.51) |
| qwen3.5-4b-4bit | Qwen3.5-4B-4bit | ⚠️ | 49 | 427.82 | 59.77 | 49 tok; -5% decode vs 0519 (63.07) |
| qwen3.5-9b-4bit | Qwen3.5-9B-4bit | ✅ | 100 | 306.12 | 39.09 | +2% decode vs 0519 (38.30) |
| qwen3.5-0.8b-4bit | Qwen3.5-0.8B-4bit | ✅ | 100 | 1013.86 | 206.67 | +13% decode vs 0519 (182.11) |
| internvl3-1b | InternVL3-1B | ⚠️ | 8 | 1898.25 | 406.33 | 8 tok; recovered |
| molmo-7b | Molmo-7B | ⚠️ | 2 | 1121.48 | 23.66 | 2 tok; recovered |

---

## Summary

**Test date**: 2026-05-28 | **Hardware**: NVIDIA GB10 (DGX Spark) | **mlxcel**: 0.1.0 | **MLX**: pin `84961223`

| Metric | Count |
|--------|-------|
| **Total text models attempted** | 109 |
| **Pass (✅)** | 46 |
| **Partial (⚠️)** | 55 |
| **Fail / skip / 0-token (❌)** | 8 (5 fail, 2 OOM skip, 1 zero-token) |
| **VLM models measured (image)** | 38 |
| **VLM Pass (✅)** | 13 |
| **VLM Partial (⚠️)** | 25 |
| **VLM image-path failures (❌, 0 tokens)** | 0 |

The remaining VLM-CSV rows are text-only models that fail warmup under an image prompt; their text-suite results are in the per-family tables above.

### Recovered models

Models that ran on 2026-05-28 after failing on 2026-05-19:

- **Text:** `internvl3-1b`, `molmo-7b`
- **VLM (image):** `internvl3-1b`, `molmo-7b`, `qwen2-vl-2b` (was 0-token), `qwen2-vl-2b-4bit` (was 0-token), `qwen3-vl-30b-a3b-4bit` (was 0-token)

### Notable decode improvements vs 2026-05-19

| Model | 2026-05-19 | 2026-05-28 | Change | Note |
|-------|-----------|-----------|--------|------|
| phi-2-4bit | 3.64 | 36.49 | +902% | 1-token run, noisy |
| phi-3.5-mini-4bit | 55.82 | 92.50 | +66% | 40 tok |
| phi-3.5-vision-4bit | 56.30 | 91.41 | +62% | 43 tok |
| gemma2-2b-4bit | 73.14 | 117.38 | +60% | 27 tok |
| qwen3-0.6b | 203.04 | 317.75 | +56% | 9 tok |
| qwen3-0.6b-4bit | 206.59 | 314.62 | +52% | 9 tok |
| exaone-3.5-2.4b-4bit | 104.06 | 146.48 | +41% |  |
| gemma3-1b-4bit | 182.97 | 256.48 | +40% | 34 tok |
| nemotron-h-30b-4bit | 25.75 | 32.92 | +28% | 46 tok |
| exaone4-1.2b-4bit | 176.69 | 225.62 | +28% | 53 tok |
| qwen3-1.7b-4bit | 139.13 | 167.90 | +21% | 14 tok |

### Notable decode regressions vs 2026-05-19

| Model | 2026-05-19 | 2026-05-28 | Change | Note |
|-------|-----------|-----------|--------|------|
| baichuan-m1-14b-4bit | 26.74 | 24.06 | -10% | few-token run, noisy |

### Failing / skipped models

- **Warmup/bench failures:** `deepseek-v3-4bit`, `gemma-4-26b-a4b-it-4bit`, `GLM-5.1-4bit`, `qwen2.5-vl-3b`, `qwen2.5-vl-3b-4bit`
- **Zero tokens generated:** `paligemma2-3b-6bit`
- **OOM-skipped (capacity, not a real failure):** `Qwen3.5-397B-A17B-4bit`, `qwen3-next-480b-4bit`

Two models present on 2026-05-19 (`gemma-4-31B-it-assistant-bf16`, `Qwen3.5-4B-DFlash`) were removed from `models/` and are no longer in the suite.
