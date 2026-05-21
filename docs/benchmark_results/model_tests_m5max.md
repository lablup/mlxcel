# Model Compatibility & Performance Tests (M5 Max)

Compatibility and performance testing for mlxcel models on **MacBook Pro M5 Max 128GB**, with same-host mlx-lm / mlx-vlm reference measurements and M1 Ultra ratios where available.

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
| **Test Date** | 2026-05-19 full sweep; 2026-05-20 Molmo v1 spot-check; 2026-05-20 Phi-3.5 spot-check; 2026-05-20 Gemma2/Gemma3 dense spot-check; 2026-05-20 Jamba spot-check; 2026-05-21 near-parity remeasure |
| **Benchmark Status** | Full text + VLM sweep on mlxcel using `mlxcel-bench-decode`, 98 text + 98 VLM-mode passes with `--cooldown 15 --big-cooldown 15`. mlx-lm / mlx-vlm baseline sub-sweeps are from the same benchmark campaign; their CSV filenames carry 2026-05-18 because the run crossed calendar midnight. |

## Legend

- ✅ Pass: Model works correctly
- ⚠️ Partial: Loads but output quality problems or low token count
- ❌ Fail: Does not work

## Basic Transformers

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| llama3 | Llama-3.2-1B-Instruct-4bit | ✅ | 8088.46 | 546.81 | **1.45x** | 31 tokens |
| llama3.1 | Llama-3.1-8B-Instruct-4bit | ✅ | 2132.28 | 116.65 | 1.09x | 100 tokens |
| llama3 (8B bf16) | Llama-3.1-8B-Instruct (bf16) | ⚠️ | 1611.27 | 33.93 | 0.97x | 87 tokens; bf16; slow decode |
| llama4 | Llama-4-Scout-17B-16E-4bit | ✅ | 131.82 | 48.59 | **1.38x** | 100 tokens |
| command-r7b | c4ai-command-r7b-4bit | ✅ | 249.42 | 110.91 | 1.01x | 100 tokens |
| aya-expanse-8b | aya-expanse-8b-4bit | ✅ | 236.59 | 110.55 | 1.03x | 100 tokens |
| aya-vision-8b | aya-vision-8b (text-only) | ✅ | 249.99 | 109.24 | 1.00x | 87 tokens; text-only |
| deepseek-r1 | DeepSeek-R1-Distill-Qwen-7B-4bit | ✅ | 447.12 | 123.76 | **1.12x** | 100 tokens |
| internlm2 | InternLM2-7B-4bit | ✅ | 548.94 | 117.25 | 1.09x | 100 tokens |
| internlm3 | internlm3-8b-instruct-4bit | ✅ | 731.94 | 101.23 | **1.17x** | 100 tokens |
| mimo | MiMo-7B-RL-4bit | ✅ | 776.96 | 119.66 | **1.40x** | 100 tokens |
| minicpm | MiniCPM-2B-sft-bf16-4bit | ✅ | 859.39 | 233.46 | **1.43x** | 100 tokens |
| bunny-llama3-8b | Bunny-Llama-3-8B-V-4bit (text) | ✅ | 569.50 | 111.08 | 1.09x | 40 tokens; text-only |
| llava-1.5-7b | llava-1.5-7b-4bit (text) | ✅ | 302.87 | 124.52 | 1.07x | 100 tokens; text-only |
| llava-next | llava-v1.6-mistral-7b-4bit (text) | ✅ | 497.19 | 122.79 | 1.08x | 100 tokens; text-only |
| llava-interleave | llava-interleave-qwen-0.5b-bf16 (text) | ✅ | 5003.93 | 403.59 | **1.28x** | 49 tokens |

## Gemma Family

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| gemma | gemma-2b-it-4bit | ✅ | 1288.22 | 217.38 | 1.12x | 49 tokens |
| gemma2 | gemma-2-2b-it-4bit | ✅ | 1266.11 | 241.96 | **1.87x** | 18 tokens; full-budget raw prompt 245.83 tok/s |
| gemma3 | gemma-3-1b-it-4bit | ✅ | 2072.39 | 399.65 | **2.04x** | 30 tokens |
| gemma3 (4B) | gemma-3-4b-it-4bit | ✅ | 819.97 | 182.16 | **1.83x** | 81 tokens; full-budget raw prompt 183.77 tok/s |
| gemma3n (E2B) | gemma-3n-E2B-it-4bit | ✅ | 812.25 | 158.71 | **2.06x** | 71 tokens |
| gemma3n (E4B) | gemma-3n-E4B-it-4bit | ✅ | 601.09 | 110.24 | **1.83x** | 71 tokens |
| gemma3n (E4B bf16) | gemma-3n-E4B-it (bf16) | ✅ | 348.30 | 39.05 | 1.10x | Gemma3n language MLP bf16 preserved, other bf16 materialized as f16; M5 (Neural Accelerator) uses the split decode path while other Apple Silicon uses the fused path; ~80% of mlx-lm decode |
| gemma4 (26B MoE) | gemma-4-26b-a4b-it-4bit | ✅ | 539.57 | 137.12 | **1.87x** | 37 tokens |
| gemma4 (31B) | gemma-4-31b-4bit | ✅ | 71.51 | 28.59 | **1.42x** | 100 tokens |
| gemma4 (31B IT) | gemma-4-31b-it-4bit | ✅ | 144.09 | 27.34 | **1.43x** | 25 tokens |
| gemma4 (31B nvfp4) | Gemma-4-31b-it-nvfp4 | ⚠️ | 91.05 | 7.17 | - | 40 tokens; nvfp4 has no fast Metal kernel |
| gemma4 (E2B 4bit) | gemma-4-e2b-it-4bit | ✅ | 1123.75 | 201.90 | **1.74x** | 17 tokens |
| gemma4 (E2B 8bit) | gemma-4-e2b-it-8bit | ✅ | 986.46 | 136.69 | **1.56x** | 32 tokens |
| gemma4 (E4B 4bit) | gemma-4-e4b-it-4bit | ✅ | 655.63 | 136.68 | **1.67x** | 33 tokens |
| gemma4 (E4B 8bit) | gemma-4-e4b-it-8bit | ✅ | 571.48 | 80.88 | **1.36x** | 43 tokens |

## EXAONE

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| exaone | EXAONE-3.5-2.4B-Instruct-4bit | ✅ | 2402.06 | 282.35 | **1.42x** | 100 tokens |
| exaone4 | exaone-4.0-1.2b-4bit | ✅ | 2069.92 | 424.44 | **1.68x** | 10 tokens |

## Qwen Family

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| qwen2.5 (0.5B) | Qwen2.5-0.5B-Instruct-4bit | ✅ | 8746.62 | 682.41 | **1.92x** | 100 tokens |
| qwen2.5 (0.5B bf16) | Qwen2.5-0.5B-Instruct (bf16) | ✅ | 5793.10 | 404.68 | - | 100 tokens |
| qwen2.5 (7B) | Qwen2.5-7B-Instruct-4bit | ✅ | 917.38 | 126.36 | **1.15x** | 100 tokens |
| qwen2.5 (7B 8bit) | Qwen2.5-7B-Instruct-8bit | ✅ | 845.08 | 68.98 | 1.00x | 100 tokens |
| qwen2.5-vl (3B) | Qwen2.5-VL-3B-Instruct-4bit | ✅ | 1529.43 | 165.11 | - | 39 tokens; EOS-terminate; fixed by #34 |
| qwen2-vl (2B) | Qwen2-VL-2B-Instruct-4bit | ✅ | 2642.68 | 273.84 | **1.83x** | 35 tokens |
| qwen1.5-moe | Qwen1.5-MoE-A2.7B-Chat-4bit | ✅ | 924.51 | 237.73 | **1.65x** | 100 tokens |
| qwen3 (0.6B) | Qwen3-0.6B-4bit | ✅ | 3683.87 | 566.50 | **1.92x** | 9 tokens |
| qwen3 (1.7B) | Qwen3-1.7B-4bit | ✅ | 1764.58 | 368.50 | **1.89x** | 14 tokens |
| qwen3 (4B) | Qwen3-4B-4bit | ✅ | 981.59 | 191.04 | **1.58x** | 41 tokens |
| qwen3 (8B) | Qwen3-8B-4bit | ✅ | 558.00 | 112.38 | **1.41x** | 33 tokens |
| qwen3-30b-a3b | Qwen3-30B-A3B-4bit | ✅ | 414.20 | 156.15 | **2.20x** | 34 tokens |
| qwen3-moe | Qwen3-MoE-30B-4bit | ✅ | 414.14 | 157.16 | **2.20x** | 34 tokens |
| qwen3-vl (2B) | Qwen3-VL-2B-Instruct-4bit | ✅ | 1367.27 | 365.31 | **1.69x** | 58 tokens; text-only |
| qwen3-vl (30B MoE) | Qwen3-VL-30B-A3B-Instruct-4bit | ✅ | 322.02 | 151.16 | **2.17x** | 35 tokens; text-only; #719 |
| qwen3-vl (32B) | Qwen3-VL-32B-Instruct-4bit | ✅ | 119.55 | 27.51 | **1.33x** | 30 tokens; text-only; #719 |
| qwen3-next (480B) | Qwen3-Next-480B-4bit | ❌ | - | FAIL | - | SKIP:oom_estimate |
| qwen3.5 (0.8B) | Qwen3.5-0.8B-4bit | ✅ | 2346.25 | 517.47 | **2.13x** | 29 tokens |
| qwen3.5 (2B) | Qwen3.5-2B-4bit | ✅ | 1266.52 | 320.84 | **1.86x** | 19 tokens |
| qwen3.5 (4B) | Qwen3.5-4B-4bit | ✅ | 731.69 | 166.56 | **1.74x** | 26 tokens |
| qwen3.5 (9B) | Qwen3.5-9B-4bit | ✅ | 453.09 | 98.50 | **1.36x** | 19 tokens |
| qwen3.5 (9B bf16) | Qwen3.5-9B (bf16) | ✅ | 310.33 | 29.98 | 0.96x | 19 tokens |
| qwen3.5 (27B) | Qwen3.5-27B-4bit | ✅ | 171.00 | 32.51 | **1.34x** | 30 tokens |
| qwen3.5-35b-a3b | Qwen3.5-35B-A3B-4bit | ✅ | 480.89 | 151.63 | **2.10x** | 31 tokens |
| qwen3.6-35b-a3b | Qwen3.6-35B-A3B-4bit | ✅ | 487.21 | 147.56 | **2.15x** | 28 tokens; NEW (5-18) |

## Phi Family

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| phi-2 | phi-2-hf-4bit-mlx | ⚠️ | 391.64 | 79.60 | **1.34x** | 1 tokens; (likely EOS) |
| phi-3-mini | Phi-3-mini-4k-instruct-4bit | ✅ | 586.96 | 207.89 | **1.23x** | 25 tokens |
| phi-3.5-mini | Phi-3.5-mini-instruct-4bit | ✅ | 578.85 | 204.63 | **1.70x** | 40 tokens |
| phi-3.5-moe | Phi-3.5-MoE-instruct-4bit | ✅ | 99.27 | 115.20 | **1.51x** | 100 tokens |
| phi-3.5-vision | Phi-3.5-vision-instruct-4bit | ✅ | 868.19 | 203.14 | **1.66x** | 43 tokens; text-only |
| phi-4 | Phi-4-4bit | ✅ | 250.99 | 63.86 | **1.11x** | 100 tokens |

## OLMo Family

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| olmo-1b | OLMo-1B-hf-4bit | ✅ | 815.02 | 243.15 | **1.11x** | 100 tokens |
| olmo2-7b | OLMo2-7B-4bit | ✅ | 658.49 | 116.88 | **1.13x** | 27 tokens |
| olmo3-32b | OLMo3.1-32B-4bit | ✅ | 459.94 | 29.11 | **1.34x** | 100 tokens |

## MoE (Mixture of Experts)

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| minimax | MiniMax-M2-3bit | ✅ | 185.84 | 73.76 | - | 100 tokens |
| mixtral | Mixtral-8x7B-Instruct-v0.1-4bit | ✅ | 83.19 | 65.20 | **1.22x** | 73 tokens |
| gpt_oss (20B) | gpt-oss-20b-MXFP4-Q4 | ✅ | 960.63 | 172.33 | **1.94x** | 100 tokens |
| gpt_oss (120B) | gpt-oss-120b-4bit | ✅ | 334.68 | 114.03 | **1.94x** | 78 tokens; issue #715 fix |
| solar-open-100b | Solar-Open-100B-4bit | ✅ | 210.91 | 65.36 | **1.82x** | 100 tokens; issue #717 fix |

## DeepSeek Family

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| deepseek | deepseek-coder-1.3b-instruct-4bit | ✅ | 5971.60 | 178.03 | 1.08x | 100 tokens |
| deepseek_v2 | DeepSeek-V2-Lite-Chat-4bit | ✅ | 389.55 | 202.25 | **1.81x** | 44 tokens |
| deepseek_v3 | - | ❌ | - | FAIL | - | FAIL:bench |

## MLA (Multi-head Latent Attention)

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| minicpm3 | MiniCPM3-4B-4bit | ✅ | 587.76 | 131.00 | **1.63x** | 100 tokens |

## Nemotron Family

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| nemotron_h | Nemotron-H-30B-4bit | ✅ | 414.31 | 177.18 | **1.96x** | 46 tokens |
| nemotron_nas | Nemotron-NAS-30B-A3B-4bit | ✅ | 422.05 | 176.38 | **1.95x** | 46 tokens |

## SSM / Mamba Models

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| mamba | Falcon-Mamba-7B-4bit | ⚠️ | 235.98 | 63.19 | **1.47x** | 2 tokens; chat template EOS |
| mamba2 | mamba2-1.3b-4bit | ✅ | 877.90 | 184.69 | **1.80x** | 100 tokens |
| jamba | Jamba-v0.1-4bit | ✅ | 591.61 | 215.84 | **1.93x** | 100 tokens; raw prompt 215.74 tok/s |

## Chinese / Asian Language Models

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| baichuan | Baichuan-M1-14B-Instruct-4bit | ✅ | 152.48 | 55.89 | **1.39x** | 7 tokens |
| glm4_moe_lite | GLM-4.7-Flash-4bit | ✅ | 249.63 | 104.30 | **2.20x** | 18 tokens |
| ernie4_5 | ERNIE-4.5-0.3B-Instruct-4bit | ✅ | 7720.63 | 1053.87 | **2.00x** | 100 tokens |
| hunyuan_moe | Hunyuan-Large-Instruct-4bit | ✅ | 109.32 | 64.43 | **1.46x** | 36 tokens |
| hunyuan_moe_13b | HunYuan-MoE-A13B-Instruct (bf16) | ✅ | 65.17 | 64.09 | - | 36 tokens |
| hunyuan_v1_dense | Hunyuan-1.8B-Instruct-4bit | ✅ | 1145.26 | 329.29 | **1.80x** | 42 tokens |

## Other Models

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| ministral3 | Ministral-3B-Instruct-4bit | ✅ | 6580.56 | 223.03 | **1.54x** | 34 tokens; VLM wrapper |
| mistral-small | mistral-small-3.1-24b-4bit | ✅ | 90.62 | 41.41 | **1.31x** | 100 tokens |
| molmo2 | molmo2-4b | ✅ | 540.64 | 64.09 | 1.07x | 33 tokens |
| molmo-7b | molmo-7b | ✅ | 338.65 | 78.74 | - | 24 tokens; text spot-check |
| internvl3 | internvl3-1b | ❌ | - | FAIL | - | FAIL:bench |
| smollm-135m | SmolLM-135M-Instruct-4bit | ✅ | 6058.41 | 905.24 | **2.22x** | 100 tokens |
| smollm3-3b | SmolLM3-3B-4bit | ✅ | 2242.59 | 232.79 | **1.71x** | 46 tokens |
| stablelm-1.6b | stablelm-2-1_6b-chat-4bit | ✅ | 2887.08 | 425.14 | **1.52x** | 59 tokens |
| starcoder2-3b | starcoder2-3b-4bit | ✅ | 455.12 | 216.48 | **1.26x** | 100 tokens |
| pixtral-12b | pixtral-12b-4bit | ✅ | 149.44 | 76.56 | **1.11x** | 100 tokens; text-only |
| paligemma2-3b | paligemma2-3b (6-bit) | ✅ | 448.78 | 149.98 | - | 100 tokens; text-only |

## VLM (image input) — full sweep

Below table reports the per-VLM-prompt run from `bench_decode.sh all --vlm`.
All entries use the VLM prompt 'What is in this image?' with
`tests/fixtures/test_image.png`.

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| aya-vision-8b | aya-vision-8b | ✅ | 1616.46 | 112.09 | 1.02x | 84 tokens |
| bunny-llama3-8b | bunny-llama3-8b-4bit | ✅ | 2851.65 | 112.24 | **1.17x** | 37 tokens |
| gemma3 (4B) | gemma3-4b-4bit | ✅ | 593.32 | 159.58 | **2.11x** | 16 tokens |
| gemma3n (E2B 4bit) | gemma3n-e2b-4bit | ✅ | 2893.48 | 151.36 | **2.08x** | 29 tokens |
| gemma3n (E4B 4bit) | gemma3n-e4b-4bit | ✅ | 2186.03 | 106.01 | **1.87x** | 33 tokens |
| gemma3n (E4B bf16) | gemma3n-e4b-bf16 | ✅ | 2025.46 | 36.95 | **1.16x** | 24 tokens; bf16→f16 conversion path |
| gemma4 (26B MoE) | gemma-4-26b-a4b-it-4bit | ✅ | 864.73 | 134.38 | **2.13x** | 30 tokens |
| gemma4 (31B) | gemma-4-31b-4bit | ✅ | 430.80 | 23.41 | **1.51x** | 5 tokens |
| gemma4 (31B IT) | gemma-4-31b-it-4bit | ✅ | 445.15 | 27.21 | **1.49x** | 28 tokens |
| gemma4 (E2B 4bit) | gemma-4-e2b-it-4bit | ✅ | 2787.47 | 217.32 | **2.03x** | 46 tokens |
| gemma4 (E2B 8bit) | gemma-4-e2b-it-8bit | ✅ | 2674.11 | 133.74 | **1.66x** | 43 tokens |
| gemma4 (E4B 4bit) | gemma-4-e4b-it-4bit | ✅ | 2064.85 | 134.10 | **1.84x** | 29 tokens |
| gemma4 (E4B 8bit) | gemma-4-e4b-it-8bit | ✅ | 1911.38 | 76.28 | **1.39x** | 12 tokens |
| internvl3 (1B) | internvl3-1b | ❌ | - | FAIL | - | FAIL:bench (unsupported architecture) |
| llama4 (Scout) | llama-4-scout-17b-4bit | ✅ | 398.13 | 48.33 | **1.52x** | 100 tokens |
| llava-1.5-7b | llava-1.5-7b-4bit | ✅ | 3141.70 | 117.70 | **1.16x** | 100 tokens |
| llava-interleave | llava-interleave-qwen-0.5b-bf16 | ✅ | 13171.77 | 343.53 | **1.27x** | 36 tokens |
| llava-next | llava-next-mistral-7b-4bit | ✅ | 2969.90 | 120.38 | **1.14x** | 100 tokens |
| ministral3 | ministral-3b-4bit | ✅ | 2784.82 | 195.22 | **1.58x** | 100 tokens |
| mistral-small (3.1 24B) | mistral-small-3.1-24b-4bit | ✅ | 676.13 | 39.62 | **1.33x** | 100 tokens |
| molmo-7b | molmo-7b | ✅ | 2287.29 | 84.99 | - | 100 tokens; mlx-vlm baseline is a 1-token anomaly |
| molmo2 (4B) | molmo2-4b | ✅ | 2512.31 | 64.01 | 1.08x | 46 tokens |
| paligemma2 (3B 6-bit) | paligemma2-3b-6bit | ✅ | 4294.39 | 80.09 | **1.78x** | 2 tokens |
| phi-3.5-vision | phi-3.5-vision-4bit | ✅ | 3582.68 | 168.77 | **1.79x** | 19 tokens |
| pixtral (12B) | pixtral-12b-4bit | ✅ | 1998.87 | 69.71 | **1.18x** | 100 tokens |
| qwen2-vl (2B) | qwen2-vl-2b-4bit | ✅ | 2485.82 | 247.21 | - | 12 tokens; EOS-terminate |
| qwen2.5-vl (3B) | qwen2.5-vl-3b-4bit | ✅ | 1696.53 | 156.83 | **1.61x** | 22 tokens; EOS-terminate; fixed by #34 |
| qwen3-vl (2B) | qwen3-vl-2b-4bit | ✅ | 934.98 | 281.37 | **1.65x** | 100 tokens |
| qwen3-vl (30B MoE) | qwen3-vl-30b-a3b-4bit | ✅ | 260.23 | 36.23 | **1.37x** | 2 tokens; #719 |
| qwen3-vl (32B) | qwen3-vl-32b-4bit | ✅ | 117.14 | 19.65 | 1.05x | 100 tokens; #719 |

## Summary Statistics

| Status | Count |
|--------|-------|
| ✅ Pass | 93 |
| ⚠️ Partial | 2 (falcon-mamba-7b-4bit, phi-2-4bit) |
| ❌ Fail | 3 (deepseek-v3-4bit, internvl3-1b, qwen3-next-480b-4bit) |

98 models tested in total. `qwen2-vl-2b-4bit` was already counted under ✅ (its text mode passed); the VLM image-mode fix flipped its VLM-table row from ⚠️ to ✅ without changing the per-model total. Adding Molmo v1 support flips `molmo-7b` from FAIL to ✅ in both the text spot-check row and the VLM table.

## Performance vs mlx-lm / mlx-vlm baseline (2026-05-19 benchmark campaign)

Source CSVs (same M5 Max host, mlxcel 0.0.28 with `--cooldown 15 --big-cooldown 15`):

- mlxcel: `benchmarks/metal_m5max_2026-05-19.csv`
- mlx-lm: `benchmarks/pylm_m5max_2026-05-18.csv` (mlx-lm 0.31.3 dev checkout in `references/mlx-lm`)
- mlxcel VLM: `benchmarks/metal_m5max_vlm_2026-05-19.csv`, `benchmarks/metal_m5max_vlm_2026-05-20.csv` (Gemma3n VLM entries)
- mlx-vlm: `benchmarks/pylm_m5max_vlm_2026-05-18.csv` (mlx-vlm 0.4.4)

The M5 Max baseline sub-sweeps ran as part of the same continuous benchmark
campaign and crossed calendar midnight. For public reporting, this campaign is
grouped under 2026-05-19 even though the Python baseline CSV filenames carry
2026-05-18 dates. Numbers are decode tok/s.
`mlxcel vs mlx-lm` is `mlxcel / mlx-lm` as a percentage; **bold** =
mlxcel >= mlx-lm. `FAIL` cells are real load/runtime errors on that
backend with this configuration. The mlx-lm checkout used for this run is
`ed1fca4` ("Thread local generation stream"); some text models fail on this
snapshot.

### Aggregate (text)

- **Comparable text pairs**: 66 (models with >=5 generated tokens both sides)
- **mlxcel >= mlx-lm**: 27 / 66 (41%)
- **mlxcel >= 90% parity**: 62 / 66 (94%, the Phi-3.5, Gemma dense, and Jamba fixes raise four models past 90%)
- **Average mlxcel/mlx-lm**: 98% (median 99%, range 72%-127%)

### Aggregate (VLM, models with >=5 generated tokens both sides)

- **Comparable VLM pairs**: 20
- **mlxcel >= mlx-vlm**: 10 / 20 (50%, the Phi-3.5 vision fix raises it from 77% to 106%)
- **mlxcel >= 90% parity**: 17 / 20 (85%)
- **Average mlxcel/mlx-vlm**: 101% (median 100%, range 74%-123%)

### Text decode (tok/s)

| Model | mlxcel | mlx-lm | mlxcel vs mlx-lm |
|-------|--------|--------|------------------|
| Gemma-4-31b-it-nvfp4 | 7.17 | FAIL | - |
| aya-expanse-8b-4bit | 110.55 | 113.87 | 97% |
| aya-vision-8b | 109.24 | FAIL | - |
| baichuan-m1-14b-4bit | 64.73 | 64.68 | **100%** |
| bunny-llama3-8b-4bit | 111.08 | FAIL | - |
| command-r7b-4bit | 110.91 | 110.67 | **100%** |
| deepseek-coder-1.3b-4bit | 178.03 | FAIL | - |
| deepseek-r1-distill-7b-4bit | 123.76 | 125.63 | 99% |
| deepseek-v2-lite-4bit | 202.25 | 215.00 | 94% |
| deepseek-v3-4bit | - | FAIL | - |
| ernie-4.5-0.3b-4bit | 1053.87 | FAIL | - |
| exaone-3.5-2.4b-4bit | 282.35 | 289.01 | 98% |
| exaone4-1.2b-4bit | 424.44 | FAIL | - |
| falcon-mamba-7b-4bit | 63.19 | 140.10 | 45% |
| gemma-2b-4bit | 217.38 | 223.27 | 97% |
| gemma-4-26b-a4b-it-4bit | 137.12 | 141.08 | 97% |
| gemma-4-31b-4bit | 28.59 | 28.79 | 99% |
| gemma-4-31b-it-4bit | 27.34 | 28.74 | 95% |
| gemma-4-e2b-it-4bit | 201.90 | FAIL | - |
| gemma-4-e2b-it-8bit | 136.69 | FAIL | - |
| gemma-4-e4b-it-4bit | 136.68 | FAIL | - |
| gemma-4-e4b-it-8bit | 80.88 | FAIL | - |
| gemma2-2b-4bit | 241.96 | 241.76 | **100%** |
| gemma3-1b-4bit | 399.65 | 388.52 | **103%** |
| gemma3-4b-4bit | 182.16 | 181.66 | **100%** |
| gemma3n-e2b-4bit | 158.71 | FAIL | - |
| gemma3n-e4b-4bit | 110.24 | FAIL | - |
| gemma3n-e4b-bf16 | 39.05 | 48.72 | 80% |
| glm4-flash-4bit | 104.30 | 104.03 | **100%** |
| gpt-oss-120b-4bit | 114.03 | 110.35 | **103%** |
| gpt-oss-20b-mxfp4 | 172.33 | 168.33 | **102%** |
| hunyuan-1.8b-4bit | 329.29 | 349.93 | 94% |
| hunyuan-large-4bit | 64.43 | FAIL | - |
| hunyuan-moe-a13b-bf16 | 64.09 | FAIL | - |
| internlm2-7b-4bit | 117.25 | 117.98 | 99% |
| internlm3-8b-4bit | 101.23 | FAIL | - |
| internvl3-1b | - | FAIL | - |
| jamba-v0.1-4bit | 215.84 | 219.38 | 98% |
| llama-3.1-8b-4bit | 116.65 | 117.43 | 99% |
| llama-3.1-8b-bf16 | 33.93 | 34.29 | 99% |
| llama-3.2-1b-4bit | 546.81 | 578.64 | 94% |
| llama-4-scout-17b-4bit | 48.59 | FAIL | - |
| llava-1.5-7b-4bit | 124.52 | FAIL | - |
| llava-interleave-qwen-0.5b-bf16 | 403.59 | FAIL | - |
| llava-next-mistral-7b-4bit | 122.79 | FAIL | - |
| mamba2-1.3b-4bit | 184.69 | FAIL | - |
| mimo-7b-4bit | 119.66 | 118.85 | **101%** |
| minicpm-2b-4bit | 233.46 | 228.46 | **102%** |
| minicpm3-4b-4bit | 131.00 | FAIL | - |
| minimax-m2-3bit | 73.76 | 68.94 | **107%** |
| ministral-3b-4bit | 223.03 | 231.92 | 96% |
| mistral-small-3.1-24b-4bit | 41.41 | 41.49 | 100% |
| mixtral-8x7b-4bit | 65.20 | 66.08 | 99% |
| molmo-7b | 78.74 | FAIL | - |
| molmo2-4b | 64.09 | FAIL | - |
| nemotron-h-30b-4bit | 177.18 | 178.80 | 99% |
| nemotron-nas-30b-4bit | 176.38 | 178.39 | 99% |
| olmo-1b-4bit | 243.15 | FAIL | - |
| olmo2-7b-4bit | 116.88 | 120.79 | 97% |
| olmo3-32b-4bit | 29.11 | 28.99 | **100%** |
| paligemma2-3b-6bit | 149.98 | FAIL | - |
| phi-2-4bit | 79.60 | FAIL | - |
| phi-3-mini-4bit | 207.89 | 212.74 | 98% |
| phi-3.5-mini-4bit | 204.63 | 207.79 | 98% |
| phi-3.5-moe-4bit | 115.20 | 107.56 | **107%** |
| phi-3.5-vision-4bit | 163.61 | FAIL | - |
| phi-4-4bit | 63.86 | 62.28 | **103%** |
| pixtral-12b-4bit | 76.56 | 74.95 | **102%** |
| qwen1.5-moe-a2.7b-4bit | 237.73 | 237.50 | **100%** |
| qwen2-vl-2b-4bit | 273.84 | 381.98 | 72% |
| qwen2.5-0.5b-4bit | 682.41 | 637.17 | **107%** |
| qwen2.5-0.5b-bf16 | 404.68 | 402.73 | **100%** |
| qwen2.5-7b-4bit | 126.36 | 123.59 | **102%** |
| qwen2.5-7b-8bit | 68.98 | 67.44 | **102%** |
| qwen2.5-vl-3b-4bit | 156.83 | 98.53 | **159%** |
| qwen3-0.6b-4bit | 566.50 | 651.14 | 87% |
| qwen3-1.7b-4bit | 368.50 | 384.84 | 96% |
| qwen3-30b-a3b-4bit | 156.15 | 147.22 | **106%** |
| qwen3-4b-4bit | 191.04 | 190.94 | **100%** |
| qwen3-8b-4bit | 112.38 | 113.40 | 99% |
| qwen3-moe-4bit | 157.16 | 146.51 | **107%** |
| qwen3-next-480b-4bit | - | FAIL | - |
| qwen3-vl-2b-4bit | 365.31 | 382.50 | 96% |
| qwen3-vl-30b-a3b-4bit | 151.16 | 146.87 | **103%** |
| qwen3-vl-32b-4bit | 27.51 | 28.51 | 96% |
| qwen3.5-0.8b-4bit | 517.47 | 545.45 | 95% |
| qwen3.5-27b-4bit | 32.51 | 34.05 | 95% |
| qwen3.5-2b-4bit | 320.84 | 345.59 | 93% |
| qwen3.5-35b-a3b-4bit | 151.63 | 152.96 | 99% |
| qwen3.5-4b-4bit | 166.56 | 174.45 | 95% |
| qwen3.5-9b-4bit | 98.50 | 108.27 | 91% |
| qwen3.5-9b-bf16 | 29.98 | 32.09 | 93% |
| qwen3.6-35b-a3b-4bit | 147.56 | 146.93 | **100%** |
| smollm-135m-4bit | 905.24 | 711.54 | **127%** |
| smollm3-3b-4bit | 232.79 | 239.14 | 97% |
| solar-open-100b-4bit | 65.36 | 66.30 | 99% |
| stablelm-1.6b-4bit | 425.14 | 423.68 | **100%** |
| starcoder2-3b-4bit | 216.48 | 214.76 | **101%** |

### VLM decode (tok/s)

| Model | mlxcel | mlx-vlm | mlxcel vs mlx-vlm |
|-------|--------|---------|-------------------|
| aya-vision-8b | 112.09 | FAIL | - |
| bunny-llama3-8b-4bit | 112.24 | FAIL | - |
| gemma-4-26b-a4b-it-4bit | 134.38 | 136.57 | 98% |
| gemma-4-31b-4bit | 23.41 | 39.85 | - |
| gemma-4-31b-it-4bit | 27.21 | 30.20 | 90% |
| gemma-4-e2b-it-4bit | 217.32 | 201.70 | **108%** |
| gemma-4-e2b-it-8bit | 133.74 | 150.51 | 89% |
| gemma-4-e4b-it-4bit | 134.10 | 131.24 | **102%** |
| gemma-4-e4b-it-8bit | 76.28 | 90.00 | 85% |
| gemma3-4b-4bit | 159.58 | FAIL | - |
| gemma3n-e2b-4bit | 151.36 | 124.63 | **121%** |
| gemma3n-e4b-4bit | 106.01 | 93.55 | **113%** |
| gemma3n-e4b-bf16 | 36.95 | 49.88 | 74% |
| internvl3-1b | FAIL | 529.33 | - |
| llama-4-scout-17b-4bit | 48.33 | FAIL | - |
| llava-1.5-7b-4bit | 117.70 | FAIL | - |
| llava-interleave-qwen-0.5b-bf16 | 343.53 | 345.08 | **100%** |
| llava-next-mistral-7b-4bit | 120.38 | FAIL | - |
| ministral-3b-4bit | 195.22 | FAIL | - |
| mistral-small-3.1-24b-4bit | 39.62 | FAIL | - |
| molmo-7b | 84.99 | 56471.65 (anomalous, 1 token) | - |
| molmo2-4b | 64.01 | 66.80 | 96% |
| paligemma2-3b-6bit | 80.09 | 124.55 | - |
| phi-3.5-vision-4bit | 168.77 | 159.63 | **106%** |
| pixtral-12b-4bit | 69.71 | FAIL | - |
| qwen2-vl-2b-4bit | 247.21 | 279.55 | 88% |
| qwen2.5-vl-3b-4bit | 156.83 | FAIL | - |
| qwen3-vl-2b-4bit | 281.37 | FAIL | - |
| qwen3-vl-30b-a3b-4bit | 36.23 | FAIL | - |
| qwen3-vl-32b-4bit | 19.65 | FAIL | - |
| qwen3.5-0.8b-4bit | 505.94 | 410.96 | **123%** |
| qwen3.5-27b-4bit | 32.84 | 33.44 | 98% |
| qwen3.5-2b-4bit | 323.00 | 318.14 | **102%** |
| qwen3.5-35b-a3b-4bit | 151.34 | 128.80 | **117%** |
| qwen3.5-4b-4bit | 170.78 | 166.46 | **103%** |
| qwen3.5-9b-4bit | 102.39 | 102.48 | **100%** |
| qwen3.5-9b-bf16 | 30.92 | 31.45 | 98% |
| qwen3.6-35b-a3b-4bit | 147.38 | 123.70 | **119%** |

### mlx-lm fail categories (text)

The mlx-lm-side FAILs are unchanged from the 2026-05-18 baseline:
unsupported architectures (`deepseek-v3-4bit`, `internvl3-1b`,
`molmo-7b`), `transformers` config schema drift
(`exaone4-1.2b-4bit`, the `gemma-4-e{2,4}b-it-{4,8}bit` and
`gemma3n-e{2,4}b-4bit` family), tokenizer wrapper bugs
(`internlm3-8b-4bit`), `ModelArgs` mismatch (`mamba2-1.3b-4bit`,
`phi-2-4bit`, `minicpm3-4b-4bit`, `Gemma-4-31b-it-nvfp4`), VLM-only
loaders routed through the text path (`aya-vision-8b`,
`bunny-llama3-8b-4bit`, `llava-*`, `paligemma2-3b-6bit`,
`phi-3.5-vision-4bit`), custom remote code refused (`hunyuan-*`,
`deepseek-coder-1.3b-4bit`, `ernie-4.5-0.3b-4bit`), and one runtime
crash (`olmo-1b-4bit`). These are mlx-lm/mlx-vlm regressions in the
development checkout under `references/`, not silent mlxcel wins.


## Issue #722 FP32 Promotion Audit

Short prompt A/B runs on 2026-05-18 used `origin/main` at `5ebc074` as the
baseline and the issue #722 branch as the candidate. Each row used:

```text
mlxcel generate -m models/<model> -p "Hello, how are you today?" -n 20 --profile --no-chat-template
```

The intent is a hot-path regression/impact check, not a replacement for the
100-token full sweep above. The clearest gains are the MoE rows that still used
the `nkh,nk->nh` expert combine contraction.

| Model | main prefill tok/s | #722 prefill tok/s | main decode tok/s | #722 decode tok/s | Decode change |
|---|---:|---:|---:|---:|---:|
| `glm4-flash-4bit` | 5.54 | 15.82 | 54.85 | 108.23 | **+97.3%** |
| `solar-open-100b-4bit` | 9.51 | 7.98 | 17.04 | 42.72 | **+150.7%** |
| `qwen3-vl-30b-a3b-4bit` | 5.47 | 5.46 | 60.24 | 58.79 | -2.4% |
| `gpt-oss-120b-4bit` | 1.47 | 1.47 | 114.33 | 115.16 | +0.7% |
| `qwen3-30b-a3b-4bit` | 5.52 | 5.52 | 159.80 | 156.18 | -2.3% |
| `qwen3.5-35b-a3b-4bit` | 4.82 | 4.77 | 136.19 | 135.88 | -0.2% |
| `qwen3.6-35b-a3b-4bit` | 4.79 | 4.79 | 133.75 | 133.02 | -0.5% |
| `mixtral-8x7b-4bit` | 4.10 | 4.13 | 69.36 | 67.33 | -2.9% |
| `phi-3.5-mini-4bit` | 41.88 | 85.28 | 165.43 | 167.31 | +1.1% |
| `gemma3n-e4b-bf16` | 6.78 | 6.87 | 11.29 | 11.25 | -0.4% |
| `qwen3.5-0.8b-4bit` | 135.49 | 179.83 | 402.99 | 405.63 | +0.7% |
| `jamba-v0.1-4bit` | 55.65 | 51.80 | 176.60 | 174.16 | -1.4% |
| `stablelm-1.6b-4bit` | 63.66 | 115.82 | 394.16 | 427.62 | +8.5% |

Reading:

- `glm4-flash-4bit` and `solar-open-100b-4bit` confirm the same FP32-promotion
  class as #715 in remaining MoE expert-weight combines.
- Qwen3/Qwen3.5/Qwen3.6 A3B, Qwen3-VL A3B, Mixtral, and gpt-oss are effectively
  guardrail-neutral in this short run. Their MoE combines now share the same
  dtype-preserving helper, but the previous contraction was not the dominant
  measured bottleneck for these rows.
- Non-MoE guardrails (`phi-3.5-mini`, `gemma3n-e4b-bf16`,
  `qwen3.5-0.8b`, `jamba`, `stablelm`) did not show a decode regression from
  the compiled activation, softcap, scalar-helper, or intentional-FP32 comments
  and tests added for this audit.

## Issue #717 SolarOpen Decode Sync Audit

Issue #722 removed the accidental FP32 expert-weight combine and raised
`solar-open-100b-4bit` decode into the low 40 tok/s range, but #717 still missed
the >=85% mlx-lm decode gate. The remaining SolarOpen-specific difference was
the Rust implementation forcing `eval_all()` after every decoder layer. That is
useful for multi-token prefill graph size control, but in single-token decode it
adds 48 GPU synchronizations per generated token. mlx-lm does not synchronize at
each layer in the decode path.

The #717 branch keeps per-layer eval for prefill and skips it only when the input
sequence length is one token. Validation used the same direct real-model command:

```text
mlxcel generate -m models/solar-open-100b-4bit -p "Hello, how are you today?" -n 100 --profile --no-chat-template
```

| Build | Prefill tok/s | Decode tok/s | vs mlx-lm 66.30 tok/s |
|---|---:|---:|---:|
| `origin/main` after #723 (`616c470`) | 9.19 | 41.35 | 62% |
| #717 branch | 34.02 | 65.66 | **99%** |

This is +58.8% decode over current main and +298.4% over the original 16.48
tok/s issue baseline. The issue acceptance gate (>=56 tok/s) is met.

## Issue #720 Moderate Gap Triage

The four #720 rows were rechecked on the M5 Max after #722, #717, #718, and
#719 had landed on `main`. The original `falcon-mamba` row used a generic chat
prompt that exits after `<|im_end|>` in both mlxcel and mlx-lm, so the useful
comparison uses a raw code prompt that generates the full 100-token budget.

| Model | Triage | Refreshed mlxcel decode | mlx-lm decode | Result |
|---|---|---:|---:|---:|
| `glm4-flash-4bit` | Real regression fixed by the already-merged #722 MoE combine change | 111.09 | 108.32 | **103%** |
| `falcon-mamba-7b-4bit` | Measurement artifact from early EOS on the generic chat prompt | 94.49 | 94.32 | **100%** |
| `starcoder2-3b-4bit` | Already fixed on current `main`; dense transformer row now matches mlx-lm | 214.85 | 213.95 | **100%** |
| `qwen3.5-0.8b-4bit` | Real GatedDeltaNet decode overhead fixed here via fast RMSNorm q/k and gated norm paths | 535.43 | 555.43 | 96% |

For the only row changed by this issue, `qwen3.5-0.8b-4bit`, the before/after
measurement used:

```text
mlxcel generate -m models/qwen3.5-0.8b-4bit -p "def fibonacci(n):\n    " -n 100 --profile --no-chat-template
```

Five back-to-back runs on current `main` before the patch decoded at
`427.17`, `428.86`, `423.32`, `423.65`, and `422.78` tok/s (mean 425.16). The
same five-run sequence after the patch decoded at `535.44`, `535.26`,
`535.06`, `536.12`, and `535.25` tok/s (mean 535.43), a **+25.9%** decode
increase and 96% of mlx-lm's 555.43 tok/s on the same prompt.

## Known Issues

| Model | Issue | Priority |
|-------|-------|----------|
| deepseek-v3-4bit | MoE + MLA; still fails warmup | Medium |
| qwen3-next-480b-4bit | OOM-skip on 128 GB; weights exceed 85% memory budget | Medium |
| qwen3-0.6b-4bit | Full-budget raw prompt stays at ~93% of mlx-lm; sub-95% decode gap | Medium |
| Gemma-4-31b-it-nvfp4 | nvfp4 quantization runs at ~1.5 tok/s; no fast Metal kernel for nvfp4 | Medium |
| falcon-mamba-7b-4bit | Generic chat prompt exits after `<|im_end|>`; use a non-chat code prompt for perf checks | Low |
| phi-2-4bit | Generates only 1 token — likely EOS handling | Low |
| llama-3.1-8b-bf16 | bf16 → f16 conversion path is functional but slow | Low |
| internvl3-1b | unsupported architecture | Low |

## Notes

- All tests use 4-bit quantized models unless noted.
- Performance measured with `mlxcel-bench-decode` (model load, warmup, and
  measured pass in one process).
- vs M1 Ultra ratio uses M1 Ultra values from `benchmarks/metal_m1ultra_2026-05-19.csv`
  (mlxcel 0.0.28, same-day MLX pin `84961223`).
- Prefill and decode tok/s reported separately.
- Full text + VLM sweep on 2026-05-19: 98 text models (`bench_decode.sh all`) and a matching `bench_decode.sh all --vlm` pass, both at `--cooldown 15 --big-cooldown 15`. Failures match the 2026-05-18 baseline (same 5 text-side and 26 VLM-side fails — all pre-existing).
- Cooldown discipline on M5 Max: 15s general / 15s big-model cooldowns were sufficient for this back-to-back run on a freshly-built binary; longer cooldowns (`--cooldown 30 --big-cooldown 30`) remain the safer default for marathon sessions or when thermal headroom is uncertain.
- Measurement noise on very fast small models remains high (qwen3.5-0.8b-4bit and
  similar can span ±15% across back-to-back runs because 100 tokens generate in
  under 300 ms).

## TurboQuant KV cache — M5 Max results (epic #458)

> Note: The 2026-04-26 benchmark run (`benchmarks/turbo_kv/2026-04-26_Mac.localdomain.csv`)
> was performed on a development machine (`Mac.localdomain`), not on the
> reference M5 Max MacBook Pro. The hardware identity is unconfirmed;
> results may not be directly comparable to the M5 Max decode/prefill numbers
> above. A dedicated M5 Max run should be appended once available.

### PPL evaluation throughput — 2026-04-26 run

The quality gate measures wikitext-2 PPL evaluation throughput (tok/s over
a 4K-token window), which is distinct from the decode tok/s reported in the
standard model tables above. These numbers characterize TurboQuant overhead
on the MLX graph execution path, not peak generation throughput.

| Model | KV mode | PPL eval tok/s | Wall clock ms | Gate |
|---|---|---|---|---|
| Meta-Llama-3.1-8B-Instruct-4bit | fp16 | 733.76 | 111,617 | baseline |
| Meta-Llama-3.1-8B-Instruct-4bit | turbo4asym | 490.32 | 167,034 | **pass** |
| Qwen2.5-1.5B-Instruct-4bit (superseded) | fp16 | 3205.54 | 25,550 | superseded — see #506 |
| Qwen2.5-1.5B-Instruct-4bit (superseded) | turbo4asym | 2227.09 | 36,775 | superseded — see #506 |

The Qwen2.5-1.5B-Instruct-4bit rows above are retained for historical reference. Issue #506 found that fixture collapses on raw wikitext without a chat template; the B3 gate now uses the base variant `Qwen2.5-1.5B-4bit`. Re-run pending.

For the full interpretation and per-model recommendations see
[`docs/turbo-kv-cache.md`](turbo-kv-cache.md).

## TurboQuant KV cache — M5 Max speed gate readings (issue #509)

First dedicated M5 Max reading of the epic-#458 KV speed gate matrix.
Hardware: Apple M5 Max, 128 GB unified memory, macOS 26.4.1 (build 25E253).
Model: `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (local dir
`models/llama-3.1-8b-4bit`). Date: 2026-05-03. Binary: mlxcel 0.0.25
post-#511 (fused Sparse-V Metal kernel landed). Reproducer:

```bash
./scripts/bench_kv_cache.sh \
  --modes fp16,int8,turbo4-asym,turbo4,turbo4-delegated,turbo3-asym \
  --contexts 4096,16384 \
  --prefill-contexts 8192 \
  --decode-tokens 80 --warmup-tokens 16 \
  --run-cooldown 15 --mode-cooldown 30 \
  models/llama-3.1-8b-4bit
```

Full CSV at `benchmarks/turbo_kv/2026-05-03_Apple_M5_Max_llama-3.1-8b-4bit.csv`.

> **CSV schema note:** Rows where `stage=prefill` record a single-token follow-up to force the KV
> cache to be populated. The resulting `decode_tok_s` value (e.g. 1200480 tok/s) reflects a
> sub-millisecond single-token step and is not a meaningful decode throughput figure; ignore it
> for prefill rows. Use `prefill_tok_s` from those rows and `decode_tok_s` from `stage=decode` rows.

### Decode @ 4K context (80 generated tokens)

| Mode | Decode tok/s | × FP16 | M5 Max gate | Verdict |
|------|--------------|--------|------|------|
| `fp16`             | 101.29 | 1.000× | baseline | baseline |
| `int8`             |  72.79 | 0.719× | (no gate; tracking) | tracking |
| `turbo4-asym`      |   9.15 | 0.090× | ≥0.97× | **fail** |
| `turbo4`           |  20.76 | 0.205× | ≥0.93× | **fail** |
| `turbo4-delegated` |  27.28 | 0.269× | ≥0.97× | **fail** (issue #521; partial fix landed) |
| `turbo3-asym`      |   6.36 | 0.063× | (tracking only) | tracking |

Issue #521 (PR #525) caches the cold-V dequant graph across decode steps;
informal in-tree A/B (100-token decode at the same 4K prompt) measures
`turbo4-delegated` at ~41 tok/s post-fix vs ~27 tok/s on v0.0.25, a ~1.5×
decode speedup that scales sharply at longer contexts.

**PR #529 Phase-1b (K-side unification, issue #527):** Removes `cold_keys` and
the per-step `concat(cold_k, hot_k)` graph node. Informal A/B on M5 Max
(3 warm runs each, `llama-3.1-8b-4bit`, 4109-token prompt, 100 generated
tokens): fp16 baseline 101.5–102.7 tok/s; turbo4-delegated post-PR #529
43.0–43.7 tok/s (~0.43× FP16, up from ~0.41× pre-fix). The modest speedup is
explained by `SliceUpdate::eval_gpu` semantics: MLX copies the full source
buffer before writing the update region, so per-step K-side memory traffic is
approximately conserved between the old concat layout and the new slice-update
layout. The remaining cost is the V-side `concat(cold_v_dequant, hot_v)`
graph node (Phase 2, issue #528).

**Issue #528 Phase-2 (fused dequant + SDPA kernel):** Adds a Metal kernel
that reads the packed cold V indices directly inside the kernel, removing
the PR-#525 `cold_v_dequant_cache` memo and the per-step
`concat(cold_v_dequant, hot_v)` graph node. The dequantised cold V never
materialises in global memory — V-memory budget stays at 4-bit packed.

Measured on `llama-3.1-8b-4bit`, 4109-token prompt, 100 generated tokens
(`benchmarks/turbo_kv/2026-05-04_Apple_M5_Max_issue_528_fused_delegated_sdpa.csv`):

| Path | tok/s | × FP16 | gate |
|---|---:|---:|---:|
| `fp16` | 101.76 | 1.000× | baseline |
| `turbo4-delegated` default (no memo) | 29.60 | 0.291× | ≥0.97× — **fail** |
| `turbo4-delegated` fused kernel (`MLXCEL_TURBO4_DELEGATED_FUSED=1`) | 18.90 | 0.186× | ≥0.97× — **fail** |

Removing the PR-#525 memo (per the issue body's "creates dead state, must
not remain" requirement) drops the legacy `update_and_fetch` route from
~0.43× to 0.29×. The fused kernel runs slower than the no-memo legacy
route because the host pipeline now composes Q·K + softmax + cold-kernel +
hot-matmul + sum out of many small MLX graph ops; the memo path could feed
the dequantised cold V into a single steel-attention SDPA call. The
0.97× M5 Max decode gate is **not cleared** by issue #528. Bringing the
kernel inside the steel-attention envelope is left to follow-up work.

**Issue #531 Phase-3 (steel-attention-envelope kernel, M5 Max measurement).**
PR #532 lands `turbo4_delegated_steel_sdpa` — a JIT-compiled Metal kernel
that runs the entire post-Q·K SDPA inline (per-Q numerically stable
softmax, cold-V dequant + weighted sum, hot-V FP16 weighted sum, all
normalised against the same softmax denominator). Bit-parity is gate-1
and was validated at PR-landing time (RMS < 5e-3 over 200 decode steps,
two new parity tests in `cache::turbo_tests`). M5 Max throughput was
deferred to a follow-up bench run because the kernel-author agent had no
M5 Max access from PR #532's run.

Measured on `llama-3.1-8b-4bit`, 4109-token prompt, 100 generated tokens
(`benchmarks/turbo_kv/2026-05-06_Apple_M5_Max_issue_531_steel_envelope.csv`):

| Path | tok/s | × FP16 | gate |
|---|---:|---:|---:|
| `fp16` | 102.97 | 1.000× | baseline |
| `turbo4-delegated` legacy `update_and_fetch + attention()` (env unset) | 29.60* | 0.291× | ≥0.97× — **fail** (issue #528 reading) |
| `turbo4-delegated` cold-only fused kernel (`MLXCEL_TURBO4_DELEGATED_FUSED=1`, pre-#532) | 18.90* | 0.186× | ≥0.97× — **fail** (issue #528 reading) |
| `turbo4-delegated` steel envelope (`MLXCEL_TURBO4_DELEGATED_FUSED=1`, post-#532) | 16.23 | 0.158× | ≥0.97× — **fail** |

`*` cross-referenced from the 2026-05-04 issue #528 CSV.

The steel envelope runs slower than both the cold-only fused kernel and
the legacy fetch route. The likely cause is the kernel's single-thread-per-Q
softmax + V accumulation pass — at decode time (`Tq=1`, `B=1`, `Hkv=8` on
llama-3.1-8b) only 8 threads are dispatched per kernel call, each scanning
the full T_total range serially. The PR #532 implementation note
acknowledges this design ("single thread; T_total reads << kernel launch
overhead, avoids threadgroup tree-reduction barriers") — the assumption
held on M1 Ultra at parity contexts but breaks on M5 Max where the
threadgroup tree-reduction would actually be faster than the serial scan.

**Issue #534 readings (Pass 1 parallelization + cold-loop sparse cutoff on top
of #532).** Issue #534 splits the kernel's Pass 1 (per-Q max + sum_exp) across
all D threads of each threadgroup with a tree reduction. The follow-up in this
PR also precomputes a score-space sparse-V cutoff
`max + log(threshold * sum_exp)`, letting the cold loop reject fully-dead tokens
before paying the exp + dequant cost. Pass 2's weighted-sum remains
D-parallelized exactly as in #532. Measured on `llama-3.1-8b-4bit`,
4109-token prompt, 100 generated tokens
(`benchmarks/turbo_kv/2026-05-06_Apple_M5_Max_issue_534_post_fix.csv`):

| Path | tok/s | × FP16 | gate |
|---|---:|---:|---:|
| `fp16` | 103.28 | 1.000× | baseline |
| `turbo4-delegated` steel envelope **pre-#534** (PR #533 reading) | 16.23 | 0.158× | ≥0.97× — **fail** |
| `turbo4-delegated` steel envelope **post-#534** (this fix)        | 19.21 | 0.186× | ≥0.97× — **still fail** |

Pass 1 parallelization plus the score-space cutoff nudges the steel envelope
slightly past the issue #528 cold-only fused kernel at 4K (19.21 vs 18.90)
but does not move the needle far enough to clear the 0.97× gate. The residual
cost is in Pass 2's per-token T_total scan. A simdgroup broadcast experiment
was also measured during #534 and regressed 4K decode, so it was not retained.
The broader simdgroup-hybrid pattern from MLX upstream's
`metal::steel::SDPA` (per-simdgroup `simd_max` / `simd_sum` plus
per-simdgroup partial-sum accumulators in Pass 2) is the proposed next
iteration; it would change the per-token per-thread T_total scan into a
per-token per-simdgroup scan (4–8× fewer scans on M5 Max for D=128 / D=256).

#### Post-#534 TurboQuant+ delegated FP16 working-set experiment (4K)

Follow-up on 2026-05-07 after comparing `references/turboquant_plus`: the MLX
delegated KVCache keeps FP16 K/V in an internal native cache and routes decode
through native SDPA, while packed storage is compacted outside the hot path.
mlxcel now has an opt-in analogue via
`MLXCEL_TURBO4_DELEGATED_FP16_FAST_PATH=1`. The follow-up handoff compacts the
initial packed-V sidecars after prefill and before the first decode forward
for `max_tokens > 1`, matching TurboQuant+'s `compact_turbo_cache(...)` shape
without putting that cost in decode timing. Measured on `llama-3.1-8b-4bit`,
4109-token prompt, 100 generated tokens
(`benchmarks/turbo_kv/2026-05-07_Apple_M5_Max_issue_534_fp16_fast_path_predecode_compact_4k.csv`):

| Path | tok/s | x FP16 | gate |
|---|---:|---:|---:|
| `fp16` | 105.23 | 1.000x | baseline |
| `turbo4-delegated` steel envelope **post-#534** | 19.21 | 0.183x | >=0.97x — **fail** |
| `turbo4-delegated` FP16 fast path + pre-decode compact | 104.09 | 0.989x | >=0.97x — **pass** |

The fast path is 5.4x faster than the post-#534 steel envelope at 4K and 3.5x
faster than the legacy `update_and_fetch + attention()` reading from issue #528
(29.60 tok/s). It clears the 0.97x gate because the one-time sidecar
compaction is no longer charged to the first decode forward. This remains a
speed-path experiment, not the compressed-only memory target, because the full
FP16 V working set is retained while the env var is enabled. The handoff is
still visible in prefill timing for the decode-stage row: 2462.77 ms vs
1271.07 ms for FP16 at 4K.

#### Post-#536 lazy sidecar policy experiment (4K)

Follow-up on the pre-decode handoff: `MLXCEL_TURBO4_DELEGATED_FP16_SIDECARS=lazy`
skips foreground packed sidecar folds during generation and compacts missing
sidecars only on preservation paths such as detach / prompt-cache donation.
Measured on the same `llama-3.1-8b-4bit`, 4109-token prompt, 100 generated
tokens
(`benchmarks/turbo_kv/2026-05-07_Apple_M5_Max_issue_534_fp16_fast_path_lazy_sidecars_4k.csv`):

| Path | decode tok/s | x FP16 | prefill_ms | gate |
|---|---:|---:|---:|---:|
| `fp16` | 104.51 | 1.000x | 1268.97 | baseline |
| `turbo4-delegated` FP16 fast path + pre-decode compact | 104.09 | 0.996x vs this FP16 run | 2462.77 | >=0.97x — **pass** |
| `turbo4-delegated` FP16 fast path + lazy sidecars | 102.15 | 0.977x | 1480.04 | >=0.97x — **pass** |

Lazy sidecars keep decode within the gate while dropping most of the handoff
cost introduced by pre-decode compaction. The remaining prefill delta versus
FP16 is now ~211 ms at 4K instead of ~1194 ms.

### Decode @ 16K context (80 generated tokens, fewer if early EOS)

| Mode | Decode tok/s | × FP16 | Generated | M5 Max gate | Verdict |
|------|--------------|--------|-----------|------|------|
| `fp16`             | 63.58 | 1.000× | 19 | baseline | baseline |
| `int8`             | 36.35 | 0.572× | 80 | (no gate; tracking) | tracking |
| `turbo4-asym`      |  3.87 | 0.061× | 26 | ≥0.95× | **fail** |
| `turbo4`           |  6.76 | 0.106× | 80 | ≥0.90× | **fail** |
| `turbo4-delegated` |  3.41 | 0.054× | 21 | ≥0.95× | **fail** (issue #528 — see below) |
| `turbo3-asym`      |  1.85 | 0.029× | 54 | (tracking only) | tracking |

The repeated-paragraph prompt hits an EOS early on `fp16`, `turbo4-asym`,
`turbo4-delegated`, and `turbo3-asym` at 16K; the per-token rate is
computed over the actually generated tokens. `int8` and symmetric `turbo4`
ran the full 80 tokens.

#### Issue #528 16K reading (50-token decode, no EOS early-exit)

Measured on `llama-3.1-8b-4bit`, ~16065-token prompt
(`benchmarks/turbo_kv/2026-05-04_Apple_M5_Max_issue_528_fused_delegated_sdpa.csv`):

| Path | tok/s | × FP16 | gate |
|---|---:|---:|---:|
| `fp16` | 74.25 | 1.000× | baseline |
| `turbo4-delegated` default (no memo) | 6.03 | 0.081× | ≥0.95× — **fail** |
| `turbo4-delegated` fused kernel | 5.12 | 0.069× | ≥0.95× — **fail** |

Same shape as the 4K reading: removing the PR-#525 memo (issue #528
requirement) regressed the legacy fetch path; the fused kernel is slower
still. The gate is wider here because at 16K the dequant cost dominates;
the per-step memo materialised ~52 MB / layer of FP16 cold V, which is
gone, but the kernel cannot replace the steel-attention SDPA pipeline
the memo enabled.

#### Issue #531 16K reading (early-EOS at 19 generated tokens)

Measured on `llama-3.1-8b-4bit`, 16163-token prompt, 100 requested decode
tokens (early EOS at 19 on both modes — same prompt shape early-exits FP16
and steel envelope at the same point, so the per-token ratio remains
fair). Same CSV as the 4K reading
(`benchmarks/turbo_kv/2026-05-06_Apple_M5_Max_issue_531_steel_envelope.csv`):

| Path | tok/s | Generated | × FP16 | gate |
|---|---:|---:|---:|---:|
| `fp16` | 63.94 | 19 | 1.000× | baseline |
| `turbo4-delegated` steel envelope (post-#532) | 2.39 | 19 | 0.037× | ≥0.95× — **fail** |

The 16K decode ratio (3.7% of FP16) is the gate's worst-case shortfall in
the epic-#458 matrix to date, ~1.4× worse than the cold-only kernel
reading from issue #528 (5.12 tok/s, 0.069× FP16). At 16K the per-token
serial scan over T_total is ~16K reads × 8 threads, completely dwarfing
the tens of milliseconds the FP16 attention path needs for the same step.

#### Issue #534 16K reading (Pass 1 parallelization, early-EOS at 19–21 tokens)

Same prompt shape as the issue #531 reading; FP16 early-exits at 19 and
the post-#534 turbo4-delegated path at 21 (one extra token before EOS).
CSV: `benchmarks/turbo_kv/2026-05-06_Apple_M5_Max_issue_534_post_fix.csv`.

| Path | tok/s | Generated | × FP16 | gate |
|---|---:|---:|---:|---:|
| `fp16` | 64.78 | 19 | 1.000× | baseline |
| `turbo4-delegated` steel envelope **pre-#534** (PR #533) | 2.39 | 19 | 0.037× | ≥0.95× — **fail** |
| `turbo4-delegated` steel envelope **post-#534** (this fix) | 2.99 | 21 | 0.046× | ≥0.95× — **still fail** |

The #534 fixes move the 16K ratio from 3.7% to 4.6% of FP16 (a 25% relative
improvement) but do not clear the gate. The residual gap is in Pass 2; see the
simdgroup-hybrid follow-up note in the 4K subsection above.

#### Post-#534 TurboQuant+ delegated FP16 working-set experiment (16K)

Same fast-path experiment as the 4K subsection, measured on the 16163-token
prompt. Both modes early-exited at 19 generated tokens
(`benchmarks/turbo_kv/2026-05-07_Apple_M5_Max_issue_534_fp16_fast_path_predecode_compact_16k.csv`):

| Path | tok/s | Generated | x FP16 | gate |
|---|---:|---:|---:|---:|
| `fp16` | 65.55 | 19 | 1.000x | baseline |
| `turbo4-delegated` steel envelope **post-#534** | 2.99 | 21 | 0.046x | >=0.95x — **fail** |
| `turbo4-delegated` FP16 fast path + pre-decode compact | 70.37 | 19 | 1.074x | >=0.95x — **pass** |

This confirms the earlier fast-path bottleneck was the first-decode sidecar
compaction placement. Once the initial sidecar fold runs during the handoff,
decode uses the same unified FP16 K/V native-SDPA hot path as FP16 mode. The
16K run is short because of early EOS, so the >1.0x ratio should be read as
FP16-class rather than a stable speedup claim. The handoff cost moved into the
decode-stage prefill timing: 11070.22 ms vs 7952.33 ms for FP16 at 16K.

#### Post-#536 lazy sidecar policy experiment (16K)

Same lazy-sidecar experiment as the 4K subsection, measured on the 16163-token
prompt with early EOS at 19 generated tokens
(`benchmarks/turbo_kv/2026-05-07_Apple_M5_Max_issue_534_fp16_fast_path_lazy_sidecars_16k.csv`):

| Path | decode tok/s | Generated | x FP16 | prefill_ms | gate |
|---|---:|---:|---:|---:|---:|
| `fp16` | 64.15 | 19 | 1.000x | 7852.26 | baseline |
| `turbo4-delegated` FP16 fast path + pre-decode compact | 70.37 | 19 | 1.097x vs this FP16 run | 11070.22 | >=0.95x — **pass** |
| `turbo4-delegated` FP16 fast path + lazy sidecars | 66.63 | 19 | 1.039x | 8155.47 | >=0.95x — **pass** |

The 16K lazy policy removes nearly all visible sidecar handoff overhead from
the decode-stage prefill timing: 11070.22 ms with pre-decode compaction drops
to 8155.47 ms, close to the 7852.26 ms FP16 baseline.

### Prefill @ 8K context (single-token decode follow-up)

| Mode | Prefill tok/s | × FP16 | M5 Max gate | Verdict |
|------|---------------|--------|------|------|
| `fp16`             | 2444.34 | 1.000× | baseline | baseline |
| `int8`             | 2664.41 | 1.090× | (no gate; tracking) | tracking (faster than FP16) |
| `turbo4-asym`      | 1680.45 | 0.687× | ≥1.00× | **fail** |
| `turbo4`           | 1157.40 | 0.474× | ≥1.00× | **fail** |
| `turbo4-delegated` | 2942.94 | 1.204× | best-effort | **pass** |
| `turbo3-asym`      | 1579.36 | 0.646× | (tracking only) | tracking |

`int8` prefill on M5 Max is 9% faster than `fp16` — consistent with the
M5 Neural Accelerator's INT8 matmul path. `turbo4-delegated` keeps the
prefill stage at FP16 by design and lands 20% above the FP16 baseline,
again likely thanks to the INT8 KV write-back happening only after
prefill commits.

### M5 Max reading

The Turbo decode gates from epic #458 do **not** pass on the
v0.0.25 binary as of 2026-05-03, on any of the three Turbo modes. The
shortfall is largest on `turbo4-asym` (~10× off the 4K gate) and smallest
on `turbo4-delegated` (~3.6× off). Cross-checking against the 2026-04-29
M1 Ultra reading: M5 Max is roughly 1.4–2.4× faster than M1 Ultra on the
same modes, but the headroom from M1's L2-bound regime is not large
enough to recover the gates on its own.

A targeted A/B at 4K decode with `MLXCEL_SPARSE_V_KERNEL=0` against the
default kernel-on path:

| Mode | Kernel ON tok/s | Graph fallback tok/s | Δ |
|------|------|------|------|
| `turbo4-asym`      |  9.23 | 18.51 | **graph is 2.0× faster** |
| `turbo4`           | 20.76 | 20.68 | parity |
| `turbo4-delegated` | 27.28 | 27.09 | parity |

The fused Sparse-V Metal kernel from #511 is a measured regression vs.
the graph reference for `turbo4-asym` on M5 Max — likely the per-thread
skip path is paying more in kernel-launch and codebook-load overhead than
it recovers from skipping below the `1e-6` threshold for an 8B-model
decode workload at 4K. `turbo4` and `turbo4-delegated` are at parity
because both modes do an FP16 V-side write at decode time anyway, so the
sparse-V path is largely inert. Even the faster graph fallback for
`turbo4-asym` (0.183× FP16) is far below the 0.97× gate, so disabling the
kernel is not a fix on its own.

### M5 Max hardware considerations

- **`turbo4-delegated` is the only Turbo mode that meets any gate on M5
  Max today** — the 8K prefill reading (1.20× FP16). Use it when prefill
  latency matters and the cold-tail compression ratio is acceptable.
- **`int8` is the recommended drop-in baseline for memory-constrained
  long-context workloads on M5 Max.** It loses ~28% of decode throughput
  at 4K and ~43% at 16K against FP16, but prefill is 9% faster and the
  KV cache halves. No correctness regression has been observed on the
  Llama-3.1 family.
- **The Turbo decode shortfall is not the L2 wall observed on M1 Ultra.**
  M5 Max has the headroom (M1U `turbo4-asym` 4K = 3.92 vs M5 = 9.15 tok/s
  with the kernel on; the graph path on M5 reaches 18.51) but the
  graph-level dequant cost still dominates. Closing the gates needs
  either a faster fused kernel or a structural change to fold the V-side
  dequant into the SDPA inner loop without per-token launch overhead.
- **Avoid `turbo3-asym` for decode-bound M5 Max workloads.** The 3-bit
  unpack saturates the Metal command-queue overhead; the wall-clock
  decode rate is only 0.063× of FP16 at 4K and degrades further at 16K.

### Acceptance criteria status (issue #509)

| Criterion | Status |
|---|---|
| Decode + prefill numbers measured for 5 KVCacheModes (fp16, int8, turbo4-asym, turbo4, turbo4-delegated) at 4K decode + 8K prefill on M5 Max | done |
| 16K decode reading on M5 Max (primary M5 Max gate cell) | done |
| `turbo3-asym` reading on M5 Max (tracking only per epic) | done |
| 32K decode reading | deferred — best-effort per epic; useful only after the kernel regression for `turbo4-asym` is investigated |
| Cross-hardware consistency check vs. M1 Ultra (PR #515) | done — M5 Max numbers are 1.4–2.4× M1 Ultra on Turbo decode |
| CSV committed under `benchmarks/turbo_kv/` | done |
| Docs summary in `docs/model_tests_m5max.md` | done (this section) |
| Failed-gate perf bug filed | follow-up — file an issue tracking (a) the `turbo4-asym` fused-kernel regression vs. graph fallback on M5 Max, and (b) the 0.27× ceiling on `turbo4-delegated` 4K decode |

### Deferred

- 32K decode reading on M5 Max. Best-effort per epic; the present 4K/16K
  gap means a 32K reading would just deepen an already-failed gate
  without informing kernel work.
- Multi-model expansion (Qwen 2.5, Gemma 3) — the gate matrix is keyed
  off Llama-3.1-8B per the epic, but per-family validation is open work.
- A re-run after the `turbo4-asym` kernel regression is fixed; the gate
  matrix should be expected to pass at that point.
