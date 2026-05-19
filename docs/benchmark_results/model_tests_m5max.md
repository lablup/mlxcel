# Model Compatibility & Performance Tests (M5 Max)

Compatibility and performance testing for mlxcel models on **MacBook Pro M5 Max 128GB**, with same-host mlx-lm / mlx-vlm reference measurements and current cross-hardware ratios where available.

## Test Environment

| Item | Value |
|------|-------|
| **Hardware** | MacBook Pro M5 Max, 128GB RAM |
| **OS** | macOS 26.4 (Tahoe) |
| **mlxcel version** | 0.0.28 |
| **MLX version** | 0.31.2 (via mlxcel-core; pinned commit `84961223`) |
| **mlx-lm baseline** | 0.31.3 (dev checkout `references/mlx-lm`, commit `ed1fca4`) |
| **mlx-vlm baseline** | 0.4.4 |
| **Test Prompt** | "Hello, how are you today?" (text) / "What is in this image?" (VLM) |
| **Max Tokens** | 100 |
| **Test Date** | 2026-05-19 full sweep |
| **Benchmark Status** | Full text + VLM sweep on mlxcel (98 text + VLM passes) with `--cooldown 15 --big-cooldown 15`; mlx-lm / mlx-vlm baseline sub-sweeps collected in the same 2026-05-19 benchmark campaign, with CSV date fields crossing midnight |

## Legend

- ✅ Pass: Model works correctly
- ⚠️ Partial: Loads but output quality problems or low token count
- ❌ Fail: Does not work

## Basic Transformers

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| llama3 | Llama-3.2-1B-Instruct-4bit | ✅ | 2537.22 | 539.67 | **1.49x** | 31 tokens |
| llama3.1 | Llama-3.1-8B-Instruct-4bit | ✅ | 791.11 | 116.85 | 1.07x | 100 tokens |
| llama3 (8B bf16) | Llama-3.1-8B-Instruct (bf16) | ⚠️ | 216.42 | 33.14 | 0.93x | 87 tokens; bf16; slow decode |
| llama4 | Llama-4-Scout-17B-16E-4bit | ✅ | 33.22 | 49.08 | **1.34x** | 100 tokens |
| command-r7b | c4ai-command-r7b-4bit | ✅ | 54.23 | 114.02 | 1.03x | 100 tokens |
| aya-expanse-8b | aya-expanse-8b-4bit | ✅ | 61.67 | 113.47 | 1.05x | 100 tokens |
| aya-vision-8b | aya-vision-8b (text-only) | ✅ | 62.64 | 112.55 | 1.05x | 87 tokens; text-only |
| deepseek-r1 | DeepSeek-R1-Distill-Qwen-7B-4bit | ✅ | 97.01 | 126.33 | **1.13x** | 100 tokens |
| internlm2 | InternLM2-7B-4bit | ✅ | 132.26 | 116.61 | 1.05x | 100 tokens |
| internlm3 | internlm3-8b-instruct-4bit | ✅ | 193.15 | 101.67 | **1.18x** | 100 tokens |
| mimo | MiMo-7B-RL-4bit | ✅ | 212.00 | 120.09 | **1.40x** | 100 tokens |
| minicpm | MiniCPM-2B-sft-bf16-4bit | ✅ | 186.60 | 233.32 | **1.44x** | 100 tokens |
| bunny-llama3-8b | Bunny-Llama-3-8B-V-4bit (text) | ✅ | 142.00 | 114.79 | **1.13x** | 40 tokens; text-only |
| llava-1.5-7b | llava-1.5-7b-4bit (text) | ✅ | 72.62 | 124.69 | 1.06x | 100 tokens; text-only |
| llava-next | llava-v1.6-mistral-7b-4bit (text) | ✅ | 132.77 | 123.31 | 1.08x | 100 tokens; text-only |
| llava-interleave | llava-interleave-qwen-0.5b-bf16 (text) | ✅ | 1382.24 | 398.56 | **1.26x** | 49 tokens |

## Gemma Family

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| gemma | gemma-2b-it-4bit | ✅ | 185.18 | 215.48 | **2.64x** | 49 tokens |
| gemma2 | gemma-2-2b-it-4bit | ✅ | 259.94 | 194.65 | **1.45x** | 18 tokens |
| gemma3 | gemma-3-1b-it-4bit | ✅ | 393.21 | 381.28 | **1.94x** | 30 tokens |
| gemma3 (4B) | gemma-3-4b-it-4bit | ✅ | 172.83 | 150.64 | **1.45x** | 79 tokens |
| gemma3n (E2B) | gemma-3n-E2B-it-4bit | ✅ | 145.84 | 157.76 | **2.32x** | 71 tokens |
| gemma3n (E4B) | gemma-3n-E4B-it-4bit | ✅ | 113.08 | 109.61 | **2.05x** | 71 tokens |
| gemma3n (E4B bf16) | gemma-3n-E4B-it (bf16) | ✅ | 244.12 | 38.23 | **2.98x** | 72 tokens; Gemma3n language MLP bf16 preserved, other bf16 materialized as f16; 78% of mlx-lm decode |
| gemma4 (26B MoE) | gemma-4-26b-a4b-it-4bit | ✅ | 363.66 | 138.24 | **1.92x** | 26 tokens |
| gemma4 (31B) | gemma-4-31b-4bit | ✅ | 32.42 | 28.85 | **1.46x** | 100 tokens |
| gemma4 (31B IT) | gemma-4-31b-it-4bit | ✅ | 123.44 | 27.63 | **1.43x** | 26 tokens |
| gemma4 (31B nvfp4) | Gemma-4-31b-it-nvfp4 | ⚠️ | 82.98 | 7.18 | - | 26 tokens; nvfp4 has no fast Metal kernel |
| gemma4 (E2B 4bit) | gemma-4-e2b-it-4bit | ✅ | 469.95 | 223.42 | **1.81x** | 28 tokens |
| gemma4 (E2B 8bit) | gemma-4-e2b-it-8bit | ✅ | 466.79 | 136.83 | **1.57x** | 33 tokens |
| gemma4 (E4B 4bit) | gemma-4-e4b-it-4bit | ✅ | 355.58 | 137.25 | **1.62x** | 33 tokens |
| gemma4 (E4B 8bit) | gemma-4-e4b-it-8bit | ✅ | 346.58 | 80.59 | **1.33x** | 33 tokens |

## EXAONE

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| exaone | EXAONE-3.5-2.4B-Instruct-4bit | ✅ | 640.53 | 287.68 | **1.49x** | 100 tokens |
| exaone4 | exaone-4.0-1.2b-4bit | ✅ | 397.30 | 416.96 | **2.10x** | 10 tokens |

## Qwen Family

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| qwen2.5 (0.5B) | Qwen2.5-0.5B-Instruct-4bit | ✅ | 1553.86 | 678.07 | **2.03x** | 100 tokens |
| qwen2.5 (0.5B bf16) | Qwen2.5-0.5B-Instruct (bf16) | ✅ | 1852.06 | 402.30 | - | 100 tokens |
| qwen2.5 (7B) | Qwen2.5-7B-Instruct-4bit | ✅ | 299.05 | 126.63 | **1.12x** | 100 tokens |
| qwen2.5 (7B 8bit) | Qwen2.5-7B-Instruct-8bit | ✅ | 165.65 | 69.09 | 0.98x | 100 tokens |
| qwen2.5-vl (3B) | Qwen2.5-VL-3B-Instruct-4bit | ❌ | - | FAIL | - | FAIL:warmup |
| qwen2-vl (2B) | Qwen2-VL-2B-Instruct-4bit | ✅ | 558.13 | 271.60 | **1.89x** | 35 tokens |
| qwen1.5-moe | Qwen1.5-MoE-A2.7B-Chat-4bit | ✅ | 74.50 | 239.93 | **1.77x** | 100 tokens |
| qwen3 (0.6B) | Qwen3-0.6B-4bit | ✅ | 754.40 | 510.83 | **2.35x** | 9 tokens |
| qwen3 (1.7B) | Qwen3-1.7B-4bit | ✅ | 509.32 | 362.95 | **1.84x** | 14 tokens |
| qwen3 (4B) | Qwen3-4B-4bit | ✅ | 269.77 | 190.99 | **1.59x** | 41 tokens |
| qwen3 (8B) | Qwen3-8B-4bit | ✅ | 146.10 | 111.86 | **1.38x** | 33 tokens |
| qwen3-30b-a3b | Qwen3-30B-A3B-4bit | ✅ | 41.82 | 151.57 | **2.09x** | 34 tokens |
| qwen3-moe | Qwen3-MoE-30B-4bit | ✅ | 42.77 | 151.40 | **2.11x** | 34 tokens |
| qwen3-vl (2B) | Qwen3-VL-2B-Instruct-4bit | ✅ | 370.07 | 368.28 | **2.35x** | 58 tokens; text-only |
| qwen3-vl (30B MoE) | Qwen3-VL-30B-A3B-Instruct-4bit | ✅ | 34.29 | 144.25 | **3.48x** | 35 tokens; text-only |
| qwen3-vl (32B) | Qwen3-VL-32B-Instruct-4bit | ✅ | 35.45 | 27.49 | **1.59x** | 30 tokens; text-only |
| qwen3-next (480B) | Qwen3-Next-480B-4bit | ❌ | - | FAIL | - | SKIP:oom_estimate |
| qwen3.5 (0.8B) | Qwen3.5-0.8B-4bit | ✅ | 534.84 | 504.32 | **2.75x** | 29 tokens |
| qwen3.5 (2B) | Qwen3.5-2B-4bit | ✅ | 385.95 | 325.94 | **2.09x** | 28 tokens |
| qwen3.5 (4B) | Qwen3.5-4B-4bit | ✅ | 235.87 | 167.43 | **1.87x** | 31 tokens |
| qwen3.5 (9B) | Qwen3.5-9B-4bit | ✅ | 127.47 | 101.89 | **1.56x** | 31 tokens |
| qwen3.5 (9B bf16) | Qwen3.5-9B (bf16) | ✅ | 244.35 | 30.48 | 1.02x | 31 tokens |
| qwen3.5 (27B) | Qwen3.5-27B-4bit | ✅ | 47.00 | 32.67 | **1.46x** | 32 tokens |
| qwen3.5-35b-a3b | Qwen3.5-35B-A3B-4bit | ✅ | 36.32 | 145.49 | **2.33x** | 31 tokens |
| qwen3.6-35b-a3b | Qwen3.6-35B-A3B-4bit | ✅ | 35.39 | 140.99 | **2.37x** | 27 tokens |

## Phi Family

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| phi-2 | phi-2-hf-4bit-mlx | ⚠️ | 101.35 | 71.95 | **1.29x** | 1 tokens (likely EOS) |
| phi-3-mini | Phi-3-mini-4k-instruct-4bit | ✅ | 153.75 | 205.16 | **1.42x** | 25 tokens |
| phi-3.5-mini | Phi-3.5-mini-instruct-4bit | ✅ | 136.54 | 163.64 | **1.36x** | 40 tokens |
| phi-3.5-moe | Phi-3.5-MoE-instruct-4bit | ✅ | 16.69 | 114.78 | **1.46x** | 100 tokens |
| phi-3.5-vision | Phi-3.5-vision-instruct-4bit | ✅ | 196.46 | 163.69 | **1.37x** | 43 tokens; text-only |
| phi-4 | Phi-4-4bit | ✅ | 69.26 | 63.66 | 1.07x | 100 tokens |

## OLMo Family

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| olmo-1b | OLMo-1B-hf-4bit | ✅ | 135.52 | 245.18 | **1.16x** | 100 tokens |
| olmo2-7b | OLMo2-7B-4bit | ✅ | 181.48 | 117.64 | **1.13x** | 27 tokens |
| olmo3-32b | OLMo3.1-32B-4bit | ✅ | 138.49 | 29.02 | **1.32x** | 100 tokens |

## MoE (Mixture of Experts)

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| minimax | MiniMax-M2-3bit | ✅ | 3.96 | 72.71 | **2.20x** | 100 tokens |
| mixtral | Mixtral-8x7B-Instruct-v0.1-4bit | ✅ | 21.35 | 65.07 | **1.19x** | 73 tokens |
| gpt_oss (20B) | gpt-oss-20b-MXFP4-Q4 | ✅ | 245.45 | 171.75 | **2.11x** | 100 tokens |
| gpt_oss (120B) | gpt-oss-120b-4bit | ✅ | 23.71 | 113.34 | **5.75x** | 74 tokens |
| solar-open-100b | Solar-Open-100B-4bit | ✅ | 200.85 | 65.59 | **4.74x** | 100 tokens |

## DeepSeek Family

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| deepseek | deepseek-coder-1.3b-instruct-4bit | ✅ | 895.50 | 182.64 | **1.11x** | 100 tokens |
| deepseek_v2 | DeepSeek-V2-Lite-Chat-4bit | ✅ | 61.91 | 205.65 | **2.06x** | 44 tokens |
| deepseek_v3 | - | ❌ | - | FAIL | - | FAIL:warmup |

## MLA (Multi-head Latent Attention)

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| minicpm3 | MiniCPM3-4B-4bit | ✅ | 189.71 | 133.89 | **1.68x** | 100 tokens |

## Nemotron Family

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| nemotron_h | Nemotron-H-30B-4bit | ✅ | 31.22 | 171.76 | **1.91x** | 46 tokens |
| nemotron_nas | Nemotron-NAS-30B-A3B-4bit | ✅ | 30.91 | 170.87 | **1.91x** | 46 tokens |

## SSM / Mamba Models

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| mamba | Falcon-Mamba-7B-4bit | ⚠️ | 65.75 | 61.01 | **2.06x** | 2 tokens; chat template EOS |
| mamba2 | mamba2-1.3b-4bit | ✅ | 193.29 | 181.66 | **1.74x** | 100 tokens |
| jamba | Jamba-v0.1-4bit | ✅ | 400.29 | 182.82 | **1.91x** | 100 tokens |

## Chinese / Asian Language Models

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| baichuan | Baichuan-M1-14B-Instruct-4bit | ✅ | 42.90 | 63.12 | **1.33x** | 41 tokens |
| glm4_moe_lite | GLM-4.7-Flash-4bit | ✅ | 27.47 | 97.85 | **2.75x** | 18 tokens |
| ernie4_5 | ERNIE-4.5-0.3B-Instruct-4bit | ✅ | 1041.17 | 1035.74 | **2.43x** | 100 tokens |
| hunyuan_moe | Hunyuan-Large-Instruct-4bit | ✅ | 21.19 | 63.77 | **1.43x** | 36 tokens |
| hunyuan_moe_13b | HunYuan-MoE-A13B-Instruct (bf16) | ✅ | 12.36 | 64.08 | - | 36 tokens |
| hunyuan_v1_dense | Hunyuan-1.8B-Instruct-4bit | ✅ | 288.72 | 326.83 | **1.93x** | 42 tokens |

## Other Models

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| ministral3 | Ministral-3B-Instruct-4bit | ✅ | 4535.65 | 223.09 | **1.58x** | 34 tokens; VLM wrapper |
| mistral-small | mistral-small-3.1-24b-4bit | ✅ | 25.68 | 41.64 | **1.30x** | 100 tokens |
| molmo2 | molmo2-4b | ✅ | 78.79 | 63.86 | 1.06x | 33 tokens |
| molmo-7b | molmo-7b | ❌ | - | FAIL | - | FAIL:warmup |
| internvl3 | internvl3-1b | ❌ | - | FAIL | - | FAIL:warmup |
| smollm-135m | SmolLM-135M-Instruct-4bit | ✅ | 900.78 | 883.99 | **2.42x** | 100 tokens |
| smollm3-3b | SmolLM3-3B-4bit | ✅ | 939.19 | 232.92 | **1.74x** | 44 tokens |
| stablelm-1.6b | stablelm-2-1_6b-chat-4bit | ✅ | 462.00 | 424.32 | **1.64x** | 59 tokens |
| starcoder2-3b | starcoder2-3b-4bit | ✅ | 108.25 | 216.96 | **2.06x** | 100 tokens |
| pixtral-12b | pixtral-12b-4bit | ✅ | 38.11 | 76.56 | 1.08x | 100 tokens; text-only |
| paligemma2-3b | paligemma2-3b (6-bit) | ✅ | 90.23 | 150.39 | - | 100 tokens; text-only |

## VLM (image input) — full sweep

Below table reports the per-VLM-prompt run from `bench_decode.sh all --vlm`.
All entries use the VLM prompt 'What is in this image?' (no image attached;
result reflects text-only response throughput on the VLM code path).

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| aya-vision-8b | aya-vision-8b | ✅ | 957.33 | 111.35 | 1.00x | 84 tokens |
| bunny-llama3-8b | bunny-llama3-8b-4bit | ✅ | 2511.48 | 112.15 | **1.18x** | 37 tokens |
| gemma3 (4B) | gemma3-4b-4bit | ✅ | 534.35 | 127.87 | **1.65x** | 16 tokens |
| gemma4 (26B MoE) | gemma-4-26b-a4b-it-4bit | ✅ | 833.33 | 134.17 | **2.02x** | 30 tokens |
| gemma4 (31B) | gemma-4-31b-4bit | ✅ | 427.10 | 23.28 | **1.55x** | 5 tokens |
| gemma4 (31B IT) | gemma-4-31b-it-4bit | ✅ | 443.69 | 27.23 | **1.46x** | 28 tokens |
| gemma4 (E2B 4bit) | gemma-4-e2b-it-4bit | ✅ | 2534.10 | 215.86 | **2.04x** | 46 tokens |
| gemma4 (E2B 8bit) | gemma-4-e2b-it-8bit | ✅ | 2437.00 | 133.67 | **1.60x** | 43 tokens |
| gemma4 (E4B 4bit) | gemma-4-e4b-it-4bit | ✅ | 1918.33 | 134.24 | **1.78x** | 29 tokens |
| gemma4 (E4B 8bit) | gemma-4-e4b-it-8bit | ✅ | 1553.54 | 76.17 | **1.36x** | 12 tokens |
| internvl3 (1B) | internvl3-1b | ❌ | - | FAIL | - | FAIL:warmup |
| llama4 (Scout) | llama-4-scout-17b-4bit | ✅ | 99.00 | 48.18 | **1.37x** | 100 tokens |
| llava-1.5-7b | llava-1.5-7b-4bit | ✅ | 2641.79 | 118.30 | **1.15x** | 100 tokens |
| llava-interleave | llava-interleave-qwen-0.5b-bf16 | ✅ | 11604.57 | 335.07 | **1.28x** | 36 tokens |
| llava-next | llava-next-mistral-7b-4bit | ✅ | 2544.77 | 121.81 | **1.14x** | 100 tokens |
| ministral3 | ministral-3b-4bit | ✅ | 2759.83 | 195.92 | **1.57x** | 100 tokens |
| mistral-small (3.1 24B) | mistral-small-3.1-24b-4bit | ✅ | 707.88 | 39.56 | **1.33x** | 100 tokens |
| molmo-7b | molmo-7b | ❌ | - | FAIL | - | FAIL:warmup |
| molmo2 (4B) | molmo2-4b | ✅ | 1605.72 | 64.35 | 1.08x | 46 tokens |
| paligemma2 (3B 6-bit) | paligemma2-3b-6bit | ✅ | 3842.86 | 73.82 | **1.90x** | 2 tokens |
| phi-3.5-vision | phi-3.5-vision-4bit | ✅ | 2610.88 | 123.34 | **1.39x** | 19 tokens |
| pixtral (12B) | pixtral-12b-4bit | ✅ | 1984.38 | 69.66 | **1.16x** | 100 tokens |
| qwen2-vl (2B) | qwen2-vl-2b-4bit | ⚠️ | 420.59 | 0 | - | loaded; 0 generated tokens |
| qwen2.5-vl (3B) | qwen2.5-vl-3b-4bit | ❌ | - | FAIL | - | FAIL:warmup |
| qwen3-vl (2B) | qwen3-vl-2b-4bit | ✅ | 347.36 | 280.57 | **1.77x** | 100 tokens |
| qwen3-vl (30B MoE) | qwen3-vl-30b-a3b-4bit | ✅ | 37.76 | 34.85 | **1.56x** | 2 tokens |
| qwen3-vl (32B) | qwen3-vl-32b-4bit | ✅ | 37.19 | 19.68 | 1.07x | 100 tokens |

## Summary Statistics

| Status | Count |
|--------|-------|
| ✅ Pass | 89 |
| ⚠️ Partial | 4 (llama-3.1-8b-bf16, Gemma-4-31b-it-nvfp4, phi-2-4bit, falcon-mamba-7b-4bit) |
| ❌ Fail | 5 (qwen2.5-vl-3b-4bit, qwen3-next-480b-4bit, deepseek-v3-4bit, molmo-7b, internvl3-1b) |

98 models tested in total.

## Performance vs mlx-lm / mlx-vlm baseline (2026-05-19 benchmark campaign)

Source CSVs (same M5 Max host, mlxcel 0.0.28 with `--cooldown 15 --big-cooldown 15`):

- mlxcel: `benchmarks/metal_m5max_2026-05-19.csv`
- mlx-lm: `benchmarks/pylm_m5max_2026-05-18.csv` (mlx-lm 0.31.3 dev checkout in `references/mlx-lm`)
- mlxcel VLM: `benchmarks/metal_m5max_vlm_2026-05-19.csv`
- mlx-vlm: `benchmarks/pylm_m5max_vlm_2026-05-18.csv` (mlx-vlm 0.4.4)

The M5 Max baseline sub-sweeps ran as part of the same continuous benchmark
campaign and crossed calendar midnight. For public reporting, this campaign is
grouped under 2026-05-19 even though the Python baseline CSV filenames carry
2026-05-18 dates. Numbers are decode tok/s.
`mlxcel vs mlx-lm` is `mlxcel / mlx-lm` as a percentage; **bold** =
mlxcel >= mlx-lm. `FAIL` cells are real load/runtime errors on that
backend with this configuration. The mlx-lm checkout pulled in for this
run is `ed1fca4` ("Thread local generation stream"); some text models
that work on other mlx-lm releases fail on this snapshot.

### Aggregate (text)

- **Comparable text pairs**: 67
- **mlxcel >= mlx-lm**: 23 / 67 (34%)
- **mlxcel >= 90% parity**: 59 / 67 (88%)
- **Average mlxcel/mlx-lm**: 96% (median 98%, range 44%-124%)

### Aggregate (VLM, models with >=5 generated tokens both sides)

- **Comparable VLM pairs**: 19
- **mlxcel >= mlx-vlm**: 7 / 19 (36%)
- **mlxcel >= 90% parity**: 14 / 19 (73%)
- **Average mlxcel/mlx-vlm**: 95% (median 99%, range 58%-117%)

### Text decode (tok/s)

| Model | mlxcel | mlx-lm | mlxcel vs mlx-lm |
|-------|--------|--------|------------------|
| Gemma-4-31b-it-nvfp4 | 7.18 | FAIL | - |
| aya-expanse-8b-4bit | 113.47 | 113.87 | 100% |
| aya-vision-8b | 112.55 | FAIL | - |
| baichuan-m1-14b-4bit | 63.12 | 64.86 | 97% |
| bunny-llama3-8b-4bit | 114.79 | FAIL | - |
| command-r7b-4bit | 114.02 | 110.67 | **103%** |
| deepseek-coder-1.3b-4bit | 182.64 | FAIL | - |
| deepseek-r1-distill-7b-4bit | 126.33 | 125.63 | **101%** |
| deepseek-v2-lite-4bit | 205.65 | 215.00 | 96% |
| ernie-4.5-0.3b-4bit | 1035.74 | FAIL | - |
| exaone-3.5-2.4b-4bit | 287.68 | 289.01 | 100% |
| exaone4-1.2b-4bit | 416.96 | FAIL | - |
| falcon-mamba-7b-4bit | 61.01 | 140.10 | 44% |
| gemma-2b-4bit | 215.48 | 223.27 | 97% |
| gemma-4-26b-a4b-it-4bit | 138.24 | 141.08 | 98% |
| gemma-4-31b-4bit | 28.85 | 28.79 | **100%** |
| gemma-4-31b-it-4bit | 27.63 | 28.74 | 96% |
| gemma-4-e2b-it-4bit | 223.42 | FAIL | - |
| gemma-4-e2b-it-8bit | 136.83 | FAIL | - |
| gemma-4-e4b-it-4bit | 137.25 | FAIL | - |
| gemma-4-e4b-it-8bit | 80.59 | FAIL | - |
| gemma2-2b-4bit | 194.65 | 241.76 | 81% |
| gemma3-1b-4bit | 381.28 | 388.52 | 98% |
| gemma3-4b-4bit | 150.64 | 181.66 | 83% |
| gemma3n-e2b-4bit | 157.76 | FAIL | - |
| gemma3n-e4b-4bit | 109.61 | FAIL | - |
| gemma3n-e4b-bf16 | 38.23 | 48.72 | 78% |
| glm4-flash-4bit | 97.85 | 104.03 | 94% |
| gpt-oss-120b-4bit | 113.34 | 110.35 | **103%** |
| gpt-oss-20b-mxfp4 | 171.75 | 168.33 | **102%** |
| hunyuan-1.8b-4bit | 326.83 | 349.93 | 93% |
| hunyuan-large-4bit | 63.77 | FAIL | - |
| hunyuan-moe-a13b-bf16 | 64.08 | FAIL | - |
| internlm2-7b-4bit | 116.61 | 117.98 | 99% |
| internlm3-8b-4bit | 101.67 | FAIL | - |
| jamba-v0.1-4bit | 182.82 | 219.38 | 83% |
| llama-3.1-8b-4bit | 116.85 | 117.43 | 100% |
| llama-3.1-8b-bf16 | 33.14 | 34.29 | 97% |
| llama-3.2-1b-4bit | 539.67 | 578.64 | 93% |
| llama-4-scout-17b-4bit | 49.08 | FAIL | - |
| llava-1.5-7b-4bit | 124.69 | FAIL | - |
| llava-interleave-qwen-0.5b-bf16 | 398.56 | FAIL | - |
| llava-next-mistral-7b-4bit | 123.31 | FAIL | - |
| mamba2-1.3b-4bit | 181.66 | FAIL | - |
| mimo-7b-4bit | 120.09 | 118.85 | **101%** |
| minicpm-2b-4bit | 233.32 | 228.46 | **102%** |
| minicpm3-4b-4bit | 133.89 | FAIL | - |
| minimax-m2-3bit | 72.71 | 68.94 | **105%** |
| ministral-3b-4bit | 223.09 | 231.92 | 96% |
| mistral-small-3.1-24b-4bit | 41.64 | 41.49 | **100%** |
| mixtral-8x7b-4bit | 65.07 | 66.08 | 98% |
| molmo2-4b | 63.86 | FAIL | - |
| nemotron-h-30b-4bit | 171.76 | 178.80 | 96% |
| nemotron-nas-30b-4bit | 170.87 | 178.39 | 96% |
| olmo-1b-4bit | 245.18 | FAIL | - |
| olmo2-7b-4bit | 117.64 | 120.79 | 97% |
| olmo3-32b-4bit | 29.02 | 28.99 | **100%** |
| paligemma2-3b-6bit | 150.39 | FAIL | - |
| phi-2-4bit | 71.95 | FAIL | - |
| phi-3-mini-4bit | 205.16 | 212.74 | 96% |
| phi-3.5-mini-4bit | 163.64 | 207.79 | 79% |
| phi-3.5-moe-4bit | 114.78 | 107.56 | **107%** |
| phi-3.5-vision-4bit | 163.69 | FAIL | - |
| phi-4-4bit | 63.66 | 62.28 | **102%** |
| pixtral-12b-4bit | 76.56 | 74.95 | **102%** |
| qwen1.5-moe-a2.7b-4bit | 239.93 | 237.50 | **101%** |
| qwen2-vl-2b-4bit | 271.60 | 381.98 | 71% |
| qwen2.5-0.5b-4bit | 678.07 | 637.17 | **106%** |
| qwen2.5-0.5b-bf16 | 402.30 | 402.73 | 100% |
| qwen2.5-7b-4bit | 126.63 | 123.59 | **102%** |
| qwen2.5-7b-8bit | 69.09 | 67.44 | **102%** |
| qwen3-0.6b-4bit | 510.83 | 651.14 | 78% |
| qwen3-1.7b-4bit | 362.95 | 384.84 | 94% |
| qwen3-30b-a3b-4bit | 151.57 | 147.22 | **103%** |
| qwen3-4b-4bit | 190.99 | 190.94 | **100%** |
| qwen3-8b-4bit | 111.86 | 113.40 | 99% |
| qwen3-moe-4bit | 151.40 | 146.51 | **103%** |
| qwen3-vl-2b-4bit | 368.28 | 382.50 | 96% |
| qwen3-vl-30b-a3b-4bit | 144.25 | 146.87 | 98% |
| qwen3-vl-32b-4bit | 27.49 | 28.51 | 96% |
| qwen3.5-0.8b-4bit | 504.32 | 545.45 | 92% |
| qwen3.5-27b-4bit | 32.67 | 34.05 | 96% |
| qwen3.5-2b-4bit | 325.94 | 345.59 | 94% |
| qwen3.5-35b-a3b-4bit | 145.49 | 152.96 | 95% |
| qwen3.5-4b-4bit | 167.43 | 174.45 | 96% |
| qwen3.5-9b-4bit | 101.89 | 108.27 | 94% |
| qwen3.5-9b-bf16 | 30.48 | 32.09 | 95% |
| qwen3.6-35b-a3b-4bit | 140.99 | 146.93 | 96% |
| smollm-135m-4bit | 883.99 | 711.54 | **124%** |
| smollm3-3b-4bit | 232.92 | 239.14 | 97% |
| solar-open-100b-4bit | 65.59 | 66.30 | 99% |
| stablelm-1.6b-4bit | 424.32 | 423.68 | **100%** |
| starcoder2-3b-4bit | 216.96 | 214.76 | **101%** |

### VLM decode (tok/s)

| Model | mlxcel | mlx-vlm | mlxcel vs mlx-vlm |
|-------|--------|---------|-------------------|
| aya-vision-8b | 111.35 | FAIL | - |
| bunny-llama3-8b-4bit | 112.15 | FAIL | - |
| gemma-4-26b-a4b-it-4bit | 134.17 | 136.57 | 98% |
| gemma-4-31b-4bit | 23.28 | 39.85 | 58% |
| gemma-4-31b-it-4bit | 27.23 | 30.20 | 90% |
| gemma-4-e2b-it-4bit | 215.86 | 201.70 | **107%** |
| gemma-4-e2b-it-8bit | 133.67 | 150.51 | 89% |
| gemma-4-e4b-it-4bit | 134.24 | 131.24 | **102%** |
| gemma-4-e4b-it-8bit | 76.17 | 90.00 | 85% |
| gemma3-4b-4bit | 127.87 | FAIL | - |
| gemma3n-e2b-4bit | FAIL | 124.63 | - |
| gemma3n-e4b-4bit | FAIL | 93.55 | - |
| gemma3n-e4b-bf16 | FAIL | 49.88 | - |
| internvl3-1b | FAIL | 529.33 | - |
| llama-4-scout-17b-4bit | 48.31 | FAIL | - |
| llava-1.5-7b-4bit | 118.30 | FAIL | - |
| llava-interleave-qwen-0.5b-bf16 | 335.07 | 345.08 | 97% |
| llava-next-mistral-7b-4bit | 121.81 | FAIL | - |
| ministral-3b-4bit | 195.92 | FAIL | - |
| mistral-small-3.1-24b-4bit | 39.56 | FAIL | - |
| molmo-7b | FAIL | 56471.65 (anomalous) | - |
| molmo2-4b | 64.35 | 66.80 | 96% |
| paligemma2-3b-6bit | 73.82 | 124.55 | 59% |
| phi-3.5-vision-4bit | 123.34 | 159.63 | 77% |
| pixtral-12b-4bit | 69.66 | FAIL | - |
| qwen2-vl-2b-4bit | 0.00 | 279.55 | - |
| qwen3-vl-2b-4bit | 280.57 | FAIL | - |
| qwen3-vl-30b-a3b-4bit | 34.85 | FAIL | - |
| qwen3-vl-32b-4bit | 19.68 | FAIL | - |
| qwen3.5-0.8b-4bit | 477.51 | 410.96 | **116%** |
| qwen3.5-27b-4bit | 33.01 | 33.44 | 99% |
| qwen3.5-2b-4bit | 321.54 | 318.14 | **101%** |
| qwen3.5-35b-a3b-4bit | 149.57 | 128.80 | **116%** |
| qwen3.5-4b-4bit | 169.57 | 166.46 | **102%** |
| qwen3.5-9b-4bit | 102.39 | 102.48 | 100% |
| qwen3.5-9b-bf16 | 31.00 | 31.45 | 99% |
| qwen3.6-35b-a3b-4bit | 144.82 | 123.70 | **117%** |

### mlx-lm fail categories (text)

The mlx-lm-side FAILs in this campaign are:
unsupported architectures (`deepseek-v3-4bit`, `internvl3-1b`,
`molmo-7b`, `qwen2.5-vl-3b-4bit`), `transformers` config schema drift
(`exaone4-1.2b-4bit`, the `gemma-4-e{2,4}b-it-{4,8}bit` and
`gemma3n-e{2,4}b-4bit` family), tokenizer wrapper bugs
(`internlm3-8b-4bit`), `ModelArgs` mismatch (`mamba2-1.3b-4bit`,
`phi-2-4bit`, `minicpm3-4b-4bit`, `Gemma-4-31b-it-nvfp4`), VLM-only
loaders routed through the text path (`aya-vision-8b`,
`bunny-llama3-8b-4bit`, `llava-*`, `paligemma2-3b-6bit`,
`phi-3.5-vision-4bit`), custom remote code refused (`hunyuan-*`,
`deepseek-coder-1.3b-4bit`, `ernie-4.5-0.3b-4bit`), and one runtime
crash (`olmo-1b-4bit`). These are backend-specific compatibility outcomes for
the referenced mlx-lm/mlx-vlm checkouts; they should not be counted as silent
mlxcel performance wins.

## Known Issues

| Model | Issue | Priority |
|-------|-------|----------|
| deepseek-v3-4bit | MoE + MLA; still fails warmup | Medium |
| qwen3-next-480b-4bit | OOM-skip on 128 GB; weights exceed 85% memory budget | Medium |
| qwen2.5-vl-3b-4bit | FAIL:warmup | Medium |
| Gemma-4-31b-it-nvfp4 | nvfp4 quantization runs at 7.18 tok/s; no fast Metal kernel for nvfp4 | Medium |
| falcon-mamba-7b-4bit | Generic chat prompt exits after `<|im_end|>`; use a non-chat code prompt for perf checks | Low |
| phi-2-4bit | Generates only 1 token — likely EOS handling | Low |
| llama-3.1-8b-bf16 | bf16 → f16 conversion path is functional but slow | Low |
| internvl3-1b | unsupported architecture | Low |
| molmo-7b | unsupported architecture | Low |

## Notes

- All tests use 4-bit quantized models unless noted.
- Performance measured with `--profile` flag (separate prefill/decode timing).
- `vs M1 Ultra` ratios are retained as cross-hardware context and should be
  refreshed together with the M1 Ultra table when public cross-hardware claims
  depend on them.
- Prefill and decode tok/s reported separately.
- Full text + VLM sweep on 2026-05-19: 98 text models (`bench_decode.sh all`) and a matching `bench_decode.sh all --vlm` pass, both at `--cooldown 15 --big-cooldown 15`.
- Cooldown discipline on M5 Max: 15s general / 15s big-model cooldowns were sufficient for this back-to-back run on a freshly-built binary; longer cooldowns (`--cooldown 30 --big-cooldown 30`) remain the safer default for marathon sessions or when thermal headroom is uncertain.
- Measurement noise on very fast small models remains high (qwen3.5-0.8b-4bit and
  similar can span ±15% across back-to-back runs because 100 tokens generate in
  under 300 ms).

## TurboQuant KV cache — M5 Max latest readings

Latest available TurboQuant readings for the M5 Max benchmark page. These measurements are separate from the standard model sweep above because they exercise KV-cache storage modes rather than model architecture coverage.

| Item | Value |
|------|-------|
| Hardware | Apple M5 Max, 128 GB unified memory |
| Model | `models/llama-3.1-8b-4bit` |
| Decode CSV | `benchmarks/turbo_kv/2026-05-03_Apple_M5_Max_llama-3.1-8b-4bit.csv` |
| FP16 fast-path CSVs | 2026-05-07 M5 Max lazy-sidecar rows under `benchmarks/turbo_kv/` |

CSV rows where `stage=prefill` include a single-token follow-up only to populate the KV cache. Use `prefill_tok_s` for those rows and ignore their sub-millisecond `decode_tok_s` values.

### PPL evaluation throughput

The quality gate measures wikitext-2 PPL evaluation throughput over a 4K-token window. It is distinct from decode tok/s.

| Model | KV mode | PPL eval tok/s | Wall clock ms | Gate |
|---|---|---:|---:|---|
| Meta-Llama-3.1-8B-Instruct-4bit | fp16 | 733.76 | 111,617 | baseline |
| Meta-Llama-3.1-8B-Instruct-4bit | turbo4asym | 490.32 | 167,034 | pass |

For the full config guide, tuning knobs, and architectural description, see [`docs/turbo-kv-cache.md`](../turbo-kv-cache.md).

### Decode @ 4K context

| Mode | Decode tok/s | x FP16 | Generated | Gate | Verdict |
|------|-------------:|-------:|----------:|------|---------|
| `fp16` | 101.29 | 1.000x | 80 | baseline | baseline |
| `int8` | 72.79 | 0.719x | 80 | tracking | tracking |
| `turbo4-asym` | 9.15 | 0.090x | 80 | >=0.97x | fail |
| `turbo4` | 20.76 | 0.205x | 80 | >=0.93x | fail |
| `turbo4-delegated` | 27.28 | 0.269x | 80 | >=0.97x | fail |
| `turbo3-asym` | 6.36 | 0.063x | 80 | tracking | tracking |
| `turbo4-delegated` FP16 fast path + lazy sidecars | 102.15 | 0.977x | 100 | >=0.97x | pass |

### Decode @ 16K context

| Mode | Decode tok/s | x FP16 | Generated | Gate | Verdict |
|------|-------------:|-------:|----------:|------|---------|
| `fp16` | 63.58 | 1.000x | 19 | baseline | baseline |
| `int8` | 36.35 | 0.572x | 80 | tracking | tracking |
| `turbo4-asym` | 3.87 | 0.061x | 26 | >=0.95x | fail |
| `turbo4` | 6.76 | 0.106x | 80 | >=0.90x | fail |
| `turbo4-delegated` | 3.41 | 0.054x | 21 | >=0.95x | fail |
| `turbo3-asym` | 1.85 | 0.029x | 54 | tracking | tracking |
| `turbo4-delegated` FP16 fast path + lazy sidecars | 66.63 | 1.039x | 19 | >=0.95x | pass |

The repeated-paragraph 16K prompt exits early on several modes; per-token rates use the actually generated token count.

### Prefill @ 8K context

| Mode | Prefill tok/s | x FP16 | Gate | Verdict |
|------|--------------:|-------:|------|---------|
| `fp16` | 2444.34 | 1.000x | baseline | baseline |
| `int8` | 2664.41 | 1.090x | tracking | tracking |
| `turbo4-asym` | 1680.45 | 0.687x | >=1.00x | fail |
| `turbo4` | 1157.40 | 0.474x | >=1.00x | fail |
| `turbo4-delegated` | 2942.94 | 1.204x | best-effort | pass |
| `turbo3-asym` | 1579.36 | 0.646x | tracking | tracking |

### M5 Max reading

- `int8` is the practical drop-in KV mode for memory-constrained long-context workloads on M5 Max: decode is slower than fp16, but prefill is faster and the KV cache is half-sized.
- `turbo4-delegated` is the only compressed Turbo mode with a passing prefill reading in this matrix.
- Compressed-only Turbo decode modes remain far below FP16 decode throughput on these readings.
- The optional `turbo4-delegated` FP16 fast path reaches FP16-class decode at 4K and 16K, but it keeps a full FP16 V working set while enabled, so it is a speed path rather than a compressed-memory result.
