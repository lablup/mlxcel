# Model Compatibility & Performance Tests (M1 Ultra)

Compatibility and performance testing for mlxcel models on **Mac Studio M1 Ultra 128GB**, with comparison against Python mlx-lm / mlx-vlm.

## Test Environment

| Item | Value |
|------|-------|
| **Hardware** | Mac Studio M1 Ultra, 128GB RAM |
| **OS** | macOS 26.4 (Tahoe) |
| **mlxcel version** | 0.0.28 |
| **MLX version** | post-0.32.0 pin (commit 84961223, via mlxcel-core) |
| **mlx-lm baseline** | 0.31.3 (dev checkout `references/mlx-lm` @ `df1d3f3`) |
| **mlx-vlm baseline** | dev checkout `references/mlx-vlm` @ `d85ca4d` |
| **Test Prompt** | "Hello, how are you today?" (text) / "What is in this image?" (VLM) |
| **Max Tokens** | 100 |
| **Test Date** | 2026-05-19 full sweep (mlxcel + mlx-lm + mlx-vlm baselines all from this date) |
| **Baseline CSVs** | `benchmarks/pylm_m1ultra_2026-05-19.csv` (mlx-lm, 75 ran / 33 FAIL / 3 oversize-skip / 1 exit-fail) + `benchmarks/pylm_m1ultra_vlm_2026-05-19.csv` (mlx-vlm, 20 working VLM models) |

## Legend

- ✅ Pass: Model works correctly
- ⚠️ Partial: Loads but output quality problems
- ❌ Fail: Does not work
- 📦 Model Issue: Model file incomplete or wrong format
- ⏳ Pending: Not yet tested

## Basic Transformers

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| llama3 | Llama-3.2-1B-Instruct-4bit | ✅ | 868.15 | 332.54 | **123%** | mlx-lm: 269.91; only 48 tokens |
| llama3 (8B bf16) | Llama-3.1-8B-Instruct (bf16) | ✅ | 134.07 | 35.43 | **118%** | mlx-lm: 29.94; non-quantized |
| llama3.1 | Llama-3.1-8B-Instruct-4bit | ✅ | 333.55 | 108.45 | **122%** | mlx-lm: 89.06; only 54 tokens |
| llama4 | Llama-4-Scout-17B-16E-Instruct-4bit | ⚠️ | 3.89 | 36.36 | **108%** | mlx-lm: 33.68; long outputs repetitive |
| qwen2 | Qwen2.5-0.5B-Instruct-4bit | ✅ | 452.68 | 329.45 | **146%** | mlx-lm: 225.37 |
| qwen2 (7B 4bit) | Qwen2.5-7B-Instruct-4bit | ✅ | 163.03 | 111.29 | **125%** | mlx-lm: 89.23 |
| qwen2 (7B 8bit) | Qwen2.5-7B-Instruct-8bit | ✅ | 85.63 | 69.61 | - | 8-bit quantized |
| qwen3 | Qwen3-0.6B-4bit | ✅ | 224.19 | 198.08 | 91% | mlx-lm: 217.22 |
| qwen3 (1.7B) | Qwen3-1.7B-4bit | ✅ | 161.66 | 193.24 | - |  |
| qwen3 (4B) | Qwen3-4B-4bit | ✅ | 124.36 | 118.85 | - |  |
| qwen3 (8B) | Qwen3-8B-4bit | ✅ | 77.00 | 80.64 | - |  |
| qwen3_5 (0.8B) | Qwen3.5-0.8B-4bit | ✅ | 173.90 | 205.52 | - | Hybrid GatedDeltaNet |
| qwen3_5 (2B) | Qwen3.5-2B-4bit | ✅ | 168.61 | 175.73 | - | Hybrid GatedDeltaNet; only 36 tokens |
| qwen3_5 (4B) | Qwen3.5-4B-4bit | ✅ | 107.33 | 101.21 | - | Hybrid GatedDeltaNet; only 36 tokens |
| qwen3_5 (9B 4bit) | qwen3.5-9B-4bit | ✅ | 67.38 | 72.02 | - | Hybrid GatedDeltaNet; only 29 tokens |
| qwen3_5 (9B bf16) | qwen3.5-9B (bf16) | ✅ | 126.90 | 31.11 | - | bf16, not quantized; Hybrid GatedDeltaNet (compiled fused kernel) |
| qwen3_5 (27B) | qwen3.5-27B-4bit | ✅ | 28.07 | 24.11 | **111%** | mlx-lm: 21.76; Hybrid Transformer+GatedDeltaNet; VLM wrapper format |
| qwen3_6 | qwen3.6-35B-A3B-4bit | ✅ | 21.51 | 67.07 | - | MoE architecture; 100 tokens |
| qwen3_next | qwen3-next-480B-4bit | ⏳ | - | SKIP | - | Qwen3Next 480B architecture; >65GB skipped on 128GB host |
| qwen2 (1.5B) | Qwen2.5-1.5B-Instruct-4bit | ✅ | 299.16 | 238.66 | - | 100 tokens |
| qwen2 (1.5B base) | Qwen2.5-1.5B-4bit | ✅ | 231.58 | 236.02 | - | base variant; 100 tokens |
| phi | phi-2-hf-4bit-mlx | ✅ | 46.33 | 52.59 | - | mlx-lm fails to load; only 1 token (likely EOS) |
| phi3 | Phi-3-mini-4k-instruct-4bit | ✅ | 64.97 | 143.69 | 100% | mlx-lm: 144.17; only 25 tokens |
| phi3small | Phi-3.5-mini-instruct-4bit | ✅ | 58.62 | 116.86 | 82% | mlx-lm: 141.80; only 40 tokens |
| phi4 | Phi-4-4bit | ✅ | 34.61 | 58.03 | **116%** | mlx-lm: 49.92 |
| smollm3 | SmolLM-135M-Instruct-4bit | ✅ | 244.49 | 352.11 | **118%** | mlx-lm: 297.15 |
| smollm3 (3B) | SmolLM3-3B-4bit | ✅ | 327.29 | 136.43 | **129%** | mlx-lm: 105.56 |
| stablelm | stablelm-2-1_6b-chat-4bit | ✅ | 244.27 | 270.47 | **131%** | mlx-lm: 206.29; only 59 tokens |
| starcoder2 | starcoder2-3b-4bit | ✅ | 55.90 | 167.20 | **119%** | mlx-lm: 140.89 |
| olmo | OLMo-1B-hf-4bit | ✅ | 49.05 | 209.70 | - | mlx-lm: requires ai2-olmo |
| olmo2 | OLMo2-7B-4bit | ✅ | 88.94 | 103.37 | **113%** | mlx-lm: 91.21; only 27 tokens |
| olmo3 | OLMo3.1-32B-4bit | ✅ | 68.86 | 21.95 | **116%** | mlx-lm: 18.92 |
| minicpm | MiniCPM-2B-sft-bf16-4bit | ✅ | 94.14 | 162.79 | **130%** | mlx-lm: 124.90 |
| mimo | MiMo-7B-RL-4bit | ✅ | 110.54 | 85.14 | **121%** | mlx-lm: 70.49 |

## Gemma Family

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| gemma | gemma-2b-it-4bit | ✅ | 84.91 | 192.52 | - | mlx-lm: 69.49; only 49 tokens |
| gemma2 | gemma-2-2b-it-4bit | ✅ | 98.07 | 135.27 | 86% | mlx-lm: 157.00; only 18 tokens |
| gemma3 | gemma-3-1b-it-4bit | ✅ | 129.49 | 196.54 | **175%** | mlx-lm: 112.37; only 34 tokens |
| gemma3 (4B) | gemma-3-4b-it-4bit | ✅ | 74.48 | 103.61 | - | only 86 tokens |
| gemma4 (31B) | gemma-4-31b-4bit | ✅ | 12.70 | 20.31 | - |  |
| gemma4 (31B-it) | gemma-4-31b-it-4bit | ✅ | 27.73 | 19.39 | - | instruction-tuned variant |
| gemma4 (26B A4B) | gemma-4-26b-a4b-it-4bit | ✅ | 137.85 | 70.27 | - | only 26 tokens |
| gemma4 (E2B 4bit) | gemma-4-e2b-it-4bit | ✅ | 170.04 | 118.94 | - | only 34 tokens |
| gemma4 (E2B 8bit) | gemma-4-e2b-it-8bit | ✅ | 91.80 | 87.56 | - | only 38 tokens |
| gemma4 (E4B 4bit) | gemma-4-e4b-it-4bit | ✅ | 129.20 | 82.77 | - | only 25 tokens |
| gemma4 (E4B 8bit) | gemma-4-e4b-it-8bit | ✅ | 72.12 | 59.06 | - | only 39 tokens |
| gemma3n | gemma-3n-E2B-it-4bit | ✅ | 69.38 | 76.18 | - | only 69 tokens |
| gemma3n (E4B) | gemma-3n-E4B-it-4bit | ✅ | 52.01 | 58.96 | - | only 74 tokens |
| gemma3n (E4B bf16) | gemma-3n-E4B-it (bf16) | ✅ | 92.67 | 34.36 | - | bf16; only 72 tokens |
| recurrent_gemma | - | ⏳ | - | - | - | Griffin SSM+attention hybrid |

## EXAONE

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| exaone | EXAONE-3.5-2.4B-Instruct-4bit | ✅ | 259.44 | 190.43 | **126%** | mlx-lm: 150.80 |
| exaone4 | exaone-4.0-1.2b-4bit | ✅ | 153.22 | 213.60 | - | mlx-lm: fails to load; only 18 tokens |
| exaone_moe | - | ⏳ | - | - | - | |

## Cohere Command R

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| cohere | c4ai-command-r7b-12-2024-4bit | ✅ | 29.10 | 111.42 | **142%** | mlx-lm: 78.67 |
| cohere2 | aya-expanse-8b-4bit | ✅ | 29.80 | 107.92 | **130%** | mlx-lm: 83.08 |

## MoE (Mixture of Experts)

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| minimax | MiniMax-M2-3bit | ⏳ | - | SKIP | - | mlx-lm: 18.2; >65GB skipped on 128GB host (93GB) |
| mixtral | Mixtral-8x7B-Instruct-v0.1-4bit | ✅ | 11.57 | 53.49 | **114%** | mlx-lm: 47.12; only 73 tokens |
| qwen2_moe | Qwen1.5-MoE-A2.7B-Chat-4bit | ✅ | 34.93 | 142.23 | **127%** | mlx-lm: 111.82; only 43 tokens |
| qwen3_moe | Qwen3-30B-A3B-4bit | ✅ | 24.08 | 70.56 | **124%** | mlx-lm: 56.91 |
| qwen3_5_moe | qwen3.5-35B-A3B-4bit | ✅ | 20.92 | 70.21 | **176%** | mlx-lm: 39.95; Hybrid GatedDeltaNet + MoE (256 experts); only 34 tokens |
| phimoe | Phi-3.5-MoE-instruct-4bit | ✅ | 9.16 | 76.13 | **124%** | mlx-lm: 61.16 |
| solar_open | Solar-Open-100B-4bit | ✅ | 11.19 | 36.20 | **120%** | mlx-lm: 30.26; 128 experts, top-8; 54GB |
| solar_open (int4) | Solar-Open-100B-int4 | ✅ | - | 11.55 | - | mlx-lm: fails to load; 128 experts, top-8; int4 quantization; 54GB |
| olmoe | - | ⏳ | - | - | - | |
| gpt_oss (20B) | gpt-oss-20b-MXFP4-Q4 | ✅ | 133.13 | 92.08 | **130%** | mlx-lm: 71.06; MXFP4 quantization; 32 experts |
| gpt_oss (120B) | gpt-oss-120b-4bit | ✅ | 3.79 | 59.62 | **127%** | mlx-lm: 47.12; 128 experts, top-4; 61GB model |

## DeepSeek Family

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| deepseek | deepseek-coder-1.3b-instruct-4bit | ✅ | 445.51 | 162.80 | - | mlx-lm: fails to load |
| deepseek_v2 | DeepSeek-V2-Lite-Chat-4bit | ✅ | 31.17 | 94.74 | **402%** | mlx-lm: 23.55; only 18 tokens |
| deepseek_r1 | DeepSeek-R1-Distill-Qwen-7B-4bit | ✅ | 51.47 | 110.52 | **126%** | mlx-lm: 87.53 |
| deepseek_v3 | deepseek-v3-4bit | ⏳ | - | SKIP | - | MoE + MLA; >65GB skipped on 128GB host (99GB) |
| deepseek_v32 | - | ⏳ | - | - | - | |

## MLA (Multi-head Latent Attention)

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| minicpm3 | MiniCPM3-4B-4bit | ✅ | 87.60 | 79.08 | **125%** | mlx-lm: 63.04 |

## Nemotron Family

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| nemotron_h | Nemotron-H-30B-4bit | ✅ | 17.03 | 89.96 | **119%** | mlx-lm: 75.78; Hybrid Mamba2+Transformer+MoE; SSM Metal kernel |
| nemotron_nas | Nemotron-NAS-30B-A3B-4bit | ✅ | 17.85 | 89.10 | **117%** | mlx-lm: 76.81; Hybrid Mamba2+Transformer+MoE |
| nemotron_h_nano_omni | Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | ✅ | 16.22 | 80.67 | - | Mamba2+Transformer+MoE+Parakeet audio; 100 tokens |

## SSM / Mamba Models

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| mamba | Falcon-Mamba-7B-4bit | ⚠️ | 31.10 | 30.78 | - | mlx-lm: fails to load; only 2 tokens due to chat template EOS |
| mamba2 | mamba2-1.3b-4bit | ✅ | 83.50 | 107.35 | - | mlx-lm: fails (ModelArgs error) |
| jamba | Jamba-v0.1-4bit | ✅ | 193.67 | 92.02 | 92% | mlx-lm: 100.13; only 76 tokens |
| rwkv7 | - | ⏳ | - | - | - | RWKV v7 linear attention |

## Chinese / Asian Language Models

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| baichuan | Baichuan-M1-14B-Instruct-4bit | ✅ | 21.74 | 46.86 | **114%** | mlx-lm: 40.95; only 39 tokens |
| glm4 | GLM-4-Flash-4bit | ✅ | 13.81 | 45.37 | - | Only 18 tokens |
| glm4_moe | - | ⏳ | - | - | - | |
| glm4_moe_lite | GLM-4.7-Flash-4bit | ✅ | - | 31.54 | 76% | mlx-lm: 41.55; only 18 tokens |
| glm5 | GLM-5-4bit | ❌ | - | FAIL | - | warmup failure (persistent) |
| internlm2 | InternLM2-7B-4bit | ✅ | 81.86 | 109.04 | **120%** | mlx-lm: 90.69 |
| internlm3 | internlm3-8b-instruct-4bit | ✅ | 109.49 | 86.13 | - | mlx-lm: fails to load |
| ernie4_5 | ERNIE-4.5-0.3B-Instruct-4bit | ✅ | 474.07 | 413.23 | - | mlx-lm: fails to load |
| ernie4_5_moe | - | ⏳ | - | - | - | |
| hunyuan_moe | Hunyuan-Large-Instruct-4bit | ✅ | 5.72 | 44.51 | - | mlx-lm: fails to load |
| hunyuan_moe_13b | HunYuan-MoE-A13B-Instruct (bf16) | ❌ | - | FAIL | - | mlx-lm: fails to load; Tiktoken tokenizer; bf16; warmup failure |
| hunyuan_v1_dense | Hunyuan-1.8B-Instruct-4bit | ✅ | 114.33 | 176.58 | **1214%** | mlx-lm: 14.54; only 41 tokens |
| kimi_linear | - | ⏳ | - | - | - | Kimi linear attention (Moonshot) |
| step3p5 | - | ⏳ | - | - | - | Step 3.5 (StepFun) |

## Other Models

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| ministral3 | Ministral-3B-Instruct-4bit | ✅ | 803.53 | 140.72 | 91% | mlx-lm: 154.36; VLM wrapper; text-only mode; only 34 tokens |
| mistral4 | - | ⏳ | - | - | - | MLA + MoE; implemented but no MLX model available |
| moondream3 | moondream3-preview-4bit | ⚠️ | - | 8.45 | - | mlx-lm: fails to load; text-only test; SigLIP + MLP; image output garbled; only 14 tokens |
| longcat_flash | - | ⏳ | - | - | - | |
| longcat_flash_ngram | - | ⏳ | - | - | - | |
| mistral_small | mistral-small-3.1-24b-4bit | ✅ | 12.88 | 31.82 | - | text-only mode |

## Vision-Language Models (VLM)

| Model | Test Model | Status | Prefill | Decode | vs mlx-vlm | Notes |
|-------|------------|--------|---------|--------|------------|-------|
| gemma3 | gemma-3-4b-it-4bit | ✅ | 249.32 | 80.13 | - | SigLIP + AvgPool; 275 prompt, 16 gen |
| gemma3n (E2B) | gemma-3n-E2B-it-4bit | ✅ | 509.31 | 71.05 | - | MobileNetV5 + MSFA; 273 prompt, 49 gen |
| gemma3n (E4B bf16) | gemma-3n-E4B-it (bf16) | ✅ | 582.41 | 31.57 | - | MobileNetV5 + MSFA; bf16; 273 prompt, 24 gen |
| gemma3n (E4B 4bit) | gemma-3n-E4B-it-4bit | ✅ | 367.98 | 55.82 | - | 273 prompt, 33 gen |
| gemma4 (E2B 4bit) | gemma-4-e2b-it-4bit | ✅ | 579.44 | 104.47 | - | 274 prompt, 100 gen |
| gemma4 (E2B 8bit) | gemma-4-e2b-it-8bit | ✅ | 480.19 | 80.92 | - | 274 prompt, 100 gen |
| gemma4 (E4B 4bit) | gemma-4-e4b-it-4bit | ✅ | 393.36 | 74.13 | - | 274 prompt, 54 gen |
| gemma4 (E4B 8bit) | gemma-4-e4b-it-8bit | ✅ | 343.13 | 54.78 | - | 274 prompt, 35 gen |
| gemma4 (31B 4bit) | gemma-4-31b-4bit | ✅ | 75.67 | 15.54 | - | 274 prompt, 100 gen |
| gemma4 (31B-it 4bit) | gemma-4-31b-it-4bit | ✅ | 78.04 | 18.57 | - | 274 prompt, 100 gen |
| gemma4 (26B A4B) | gemma-4-26b-a4b-it-4bit | ✅ | 262.22 | 65.65 | - | 277 prompt, 28 gen |
| llava 1.5 | llava-1.5-7b-4bit | ✅ | 675.20 | 103.82 | - | CLIP + MLP; Vicuna-7b; 583 prompt, 100 gen; mlx-vlm requires PyTorch |
| llava-interleave | llava-interleave-qwen-0.5b-bf16 | ✅ | 3334.88 | 263.41 | **122%** | mlx-vlm: 215.71; SigLIP + MLP; Qwen2-0.5b; 754 prompt, 36 gen |
| llava-next | llava-v1.6-mistral-7b-4bit | ✅ | 642.25 | 106.65 | - | CLIP + MLP; Mistral; 590 prompt, 100 gen; mlx-vlm template error |
| llava-bunny | Bunny-Llama-3-8B-V-4bit | ✅ | 619.07 | 95.70 | **150%** | mlx-vlm: 63.66; SigLIP + MLP; Llama3; 746 prompt, 37 gen |
| llama4 | Llama-4-Scout-17B-16E-Instruct-4bit | ✅ | 13.86 | 35.66 | - | 162 prompt, 100 gen |
| aya-vision | aya-vision-8b | ✅ | 349.35 | 111.11 | **126%** | mlx-vlm: 87.87; SigLIP + SwiGLU; Cohere2; 176 prompt, 100 gen |
| paligemma | paligemma2-3b (6-bit) | ⚠️ | 1195.96 | 38.37 | - | SigLIP + Linear; Gemma2; 1032 prompt, only 2 gen tokens |
| pixtral | pixtral-12b-4bit | ✅ | 442.19 | 60.29 | **106%** | mlx-vlm: 57.15; Pixtral ViT; Mistral; 4102 prompt, 100 gen |
| mistral3 | mistral-small-3.1-24b-4bit | ✅ | 127.52 | 29.78 | - | Pixtral ViT + PatchMerger; Mistral; 3032 prompt, 100 gen; mlx-vlm error |
| ministral3 | Ministral-3B-Instruct-4bit | ✅ | 524.83 | 124.42 | - | Pixtral ViT; 3566 prompt, 100 gen |
| phi3.5-vision | Phi-3.5-vision-instruct-4bit | ✅ | 793.61 | 92.64 | 89% | mlx-vlm: 103.52; CLIP + HD tiling; Phi3; 773 prompt, 19 gen |
| phi4mm | phi-4-multimodal-instruct (bf16) | ✅ | 571.90 | 25.42 | - | SigLIP + HD transform + AvgPool2d; Phi3; SuScaledRoPE + runtime LoRA; 2635 tokens; 12GB bf16 |
| moondream3 | moondream3-preview-4bit | ⚠️ | 1.36 | 10.05 | - | SigLIP + MLP; image output garbled; only 63 tokens |
| minicpm-o | MiniCPM-o-2_6-4bit | ✅ | 33.67 | 70.80 | - | SigLIP + Resampler; Qwen3; 80 tokens |
| molmo | Molmo-7B | ❌ | - | FAIL | - | warmup failure; unsupported architecture (only molmo2 supported) |
| molmo2 | molmo2-4b | ✅ | 576.46 | 59.54 | **102%** | mlx-vlm: 58.65; fast SDPA vision encoder; 430 prompt, 100 gen |
| internvl3 | InternVL3-1B | ❌ | - | FAIL | - | warmup failure; unsupported architecture |
| nemotron-omni | Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | ✅ | 134.62 | 68.89 | - | Mamba2+Transformer+MoE+Parakeet audio; 100 gen |
| youtu-vl | youtu-vl-4b-instruct | ⚠️ | 343.06 | 20.70 | - | only 1 gen token |
| qwen2-vl | Qwen2-VL-2B-Instruct-4bit | ⚠️ | 122.33 | 0.00 | - | Custom ViT + MRoPE; text-only pass; VLM warmup failure |
| qwen2.5-vl | Qwen2.5-VL-3B-Instruct-4bit | ✅ | 306.99 | 97.66 | - | Windowed ViT + MRoPE; 91 prompt, 46 gen; mlx-vlm requires PyTorch |
| qwen3-vl | Qwen3-VL-2B-Instruct-4bit | ✅ | 128.47 | 167.93 | - | DeepStack + vectorized MRoPE; 100 gen |
| qwen3-vl (4B) | Qwen3-VL-4B-Instruct-4bit | ✅ | 92.27 | 94.79 | - | DeepStack + vectorized MRoPE; 100 gen |
| qwen3-vl (8B) | Qwen3-VL-8B-Instruct-4bit | ✅ | 60.24 | 66.35 | - | DeepStack + vectorized MRoPE; 100 gen |
| qwen3-vl (32B) | Qwen3-VL-32B-Instruct-4bit | ✅ | 20.99 | 18.47 | - | DeepStack + vectorized MRoPE; 100 gen |
| qwen3-vl-moe | Qwen3-VL-30B-A3B-Instruct-4bit | ✅ | 20.40 | 22.41 | - | MoE (128 experts) + DeepStack; 100 gen |
| qwen3.5-vl (0.8B) | qwen3.5-0.8B-4bit | ✅ | 145.40 | 212.96 | - | Hybrid GatedDeltaNet VLM; 57 prompt, 53 gen |
| qwen3.5-vl (2B) | qwen3.5-2B-4bit | ✅ | 118.75 | 178.66 | - | Hybrid GatedDeltaNet VLM; 57 prompt, 58 gen |
| qwen3.5-vl (4B) | qwen3.5-4B-4bit | ✅ | 80.81 | 101.63 | - | Hybrid GatedDeltaNet VLM; 57 prompt, 30 gen |
| qwen3.5-vl (9B 4bit) | qwen3.5-9B-4bit | ✅ | 31.13 | 73.21 | - | Hybrid GatedDeltaNet VLM; 57 prompt, 62 gen |
| qwen3.5-vl (9B bf16) | qwen3.5-9B (bf16) | ✅ | 105.93 | 31.76 | - | Hybrid GatedDeltaNet VLM; 57 prompt, 78 gen; bf16 |
| qwen3.5-vl (27B) | qwen3.5-27B-4bit | ✅ | 26.50 | 24.96 | - | Hybrid GatedDeltaNet VLM; 57 prompt, 42 gen |
| qwen3.5-vl-moe | qwen3.5-35B-A3B-4bit | ✅ | 20.21 | 68.55 | - | Hybrid GatedDeltaNet + MoE VLM; 57 prompt, 47 gen |
| qwen3.6-vl-moe | qwen3.6-35B-A3B-4bit | ✅ | 20.94 | 68.62 | - | Hybrid GatedDeltaNet + MoE VLM; 100 gen |
| molmo-point | - | ⏳ | - | - | - | Molmo-Point (point detection); implemented but no MLX model available |

**VLM test conditions**: Image: 224x224 PNG (test_image.png) unless noted. Prompt: "What is in this image?" Max tokens: 100. Prefill includes vision encoder + projector overhead. mlx-vlm v0.4.1. Decode speed measured separately from prefill using `--profile` mode. Many models require PyTorch for mlx-vlm's HuggingFace processor: marked with "-" in vs mlx-vlm column. Three text-only models (`deepseek-v3-4bit` 99GB, `minimax-m2-3bit` 93GB, `qwen3-next-480b-4bit` 251GB) skipped on this 128GB host per >65GB threshold.

## Summary Statistics

| Status | Count |
|--------|-------|
| ✅ Pass | 115 (82 text + 33 VLM) |
| ⚠️ Partial | 6 (2 text + 4 VLM) |
| ❌ Fail | 8 (6 text + 2 VLM) |
| ⏳ Pending / Skipped (>65GB) | 16 (13 text pending + 3 oversize skip) |

## Performance Comparison

### Outperforming mlx-lm (>100%)

| Model | mlxcel | mlx-lm | vs mlx-lm |
|-------|--------|--------|-----------|
| hunyuan_v1_dense | 176.58 | 14.54 | **1214%** |
| deepseek_v2 | 94.74 | 23.55 | **402%** |
| qwen3_5_moe | 70.21 | 39.95 | **176%** |
| gemma3 (1B) | 196.54 | 112.37 | **175%** |
| qwen2 | 329.45 | 225.37 | **146%** |
| cohere | 111.42 | 78.67 | **142%** |
| stablelm | 270.47 | 206.29 | **131%** |
| cohere2 | 107.92 | 83.08 | **130%** |
| minicpm | 162.79 | 124.90 | **130%** |
| gpt_oss (20B) | 92.08 | 71.06 | **130%** |
| smollm3 (3B) | 136.43 | 105.56 | **129%** |
| qwen2_moe | 142.23 | 111.82 | **127%** |
| gpt_oss (120B) | 59.62 | 47.12 | **127%** |
| exaone | 190.43 | 150.80 | **126%** |
| deepseek_r1 | 110.52 | 87.53 | **126%** |
| qwen2 (7B 4bit) | 111.29 | 89.23 | **125%** |
| minicpm3 | 79.08 | 63.04 | **125%** |
| qwen3_moe | 70.56 | 56.91 | **124%** |
| phimoe | 76.13 | 61.16 | **124%** |
| llama3 (1B) | 332.54 | 269.91 | **123%** |
| llama3.1 | 108.45 | 89.06 | **122%** |
| mimo | 85.14 | 70.49 | **121%** |
| internlm2 | 109.04 | 90.69 | **120%** |
| solar_open | 36.20 | 30.26 | **120%** |
| nemotron_h | 89.96 | 75.78 | **119%** |
| starcoder2 | 167.20 | 140.89 | **119%** |
| smollm3 | 352.11 | 297.15 | **118%** |
| llama3 (8B bf16) | 35.43 | 29.94 | **118%** |
| nemotron_nas | 89.10 | 76.81 | **117%** |
| phi4 | 58.03 | 49.92 | **116%** |
| olmo3 | 21.95 | 18.92 | **116%** |
| baichuan | 46.86 | 40.95 | **114%** |
| mixtral | 53.49 | 47.12 | **114%** |
| olmo2 | 103.37 | 91.21 | **113%** |
| qwen3_5 (27B) | 24.11 | 21.76 | **111%** |
| llama4 | 36.36 | 33.68 | **108%** |

### Near parity (90-100%)

| Model | mlxcel | mlx-lm | vs mlx-lm |
|-------|--------|--------|-----------|
| phi3 | 143.69 | 144.17 | 100% |
| jamba | 92.02 | 100.13 | 92% |
| qwen3 (0.6B) | 198.08 | 217.22 | 91% |
| ministral3 | 140.72 | 154.36 | 91% |

### Needs optimization (<90%)

| Model | mlxcel | mlx-lm | vs mlx-lm | Notes |
|-------|--------|--------|-----------|-------|
| phi3.5-vision (VLM) | 92.64 | 103.52 | 89% | Only 19 gen tokens |
| gemma2 | 135.27 | 157.00 | 86% | Only 18 tokens |
| phi3small | 116.86 | 141.80 | 82% | Only 40 tokens |
| glm4_moe_lite | 31.54 | 41.55 | 76% | Only 18 tokens |

### No mlx-lm comparison available

| Model | mlxcel | Reason |
|-------|--------|--------|
| ernie4_5 | 413.23 | mlx-lm: fails to load |
| qwen3_5 (0.8B) | 205.52 | Not benchmarked in mlx-lm |
| exaone4 | 213.60 | mlx-lm: fails to load; only 18 tokens |
| olmo | 209.70 | mlx-lm: requires ai2-olmo |
| qwen3 (1.7B) | 193.24 | Not benchmarked in mlx-lm |
| gemma (2B) | 192.52 | only 49 tokens |
| deepseek | 162.80 | mlx-lm: fails to load |
| qwen3_5 (2B) | 175.73 | Not benchmarked in mlx-lm |
| qwen3 (4B) | 118.85 | Not benchmarked in mlx-lm |
| mamba2 | 107.35 | mlx-lm: ModelArgs error |
| gemma3 (4B) | 103.61 | Not benchmarked in mlx-lm |
| qwen3_5 (4B) | 101.21 | Not benchmarked in mlx-lm |
| internlm3 | 86.13 | mlx-lm: fails to load |
| qwen3 (8B) | 80.64 | Not benchmarked in mlx-lm |
| qwen2 (7B 8bit) | 69.61 | 8-bit quantized; no mlx-lm comparison |
| gemma3n (E2B) | 76.18 | only 69 tokens |
| qwen3_5 (9B 4bit) | 72.02 | Not benchmarked in mlx-lm |
| gemma4 (E2B 4bit) | 118.94 | only 34 tokens |
| gemma4 (E2B 8bit) | 87.56 | only 38 tokens |
| gemma4 (E4B 4bit) | 82.77 | only 25 tokens |
| gemma3n (E4B) | 58.96 | only 74 tokens |
| gemma4 (E4B 8bit) | 59.06 | only 39 tokens |
| phi | 52.59 | mlx-lm: fails to load; only 1 token (EOS) |
| gemma4 (26B A4B) | 70.27 | only 26 tokens |
| hunyuan_moe | 44.51 | mlx-lm: fails to load |
| glm4 | 45.37 | Only 18 tokens |
| mamba | 30.78 | mlx-lm: fails to load; only 2 tokens (chat template EOS) |
| qwen3_5 (9B bf16) | 31.11 | bf16, not quantized |
| gemma4 (31B) | 20.31 | no mlx-lm comparison |
| gemma4 (31B-it) | 19.39 | no mlx-lm comparison |
| gemma3n (E4B bf16) | 34.36 | bf16; only 72 tokens |
| nemotron_h_nano_omni | 80.67 | Mamba2+Transformer+MoE+Parakeet audio |
| solar_open (int4) | 11.55 | mlx-lm: fails to load; int4 quantization; 54GB |
| moondream3 | 8.45 | mlx-lm: fails to load; text-only test; only 14 tokens |
| mistral4 | - | No MLX model available; implemented |
| molmo-point | - | No MLX model available; implemented |

### VLM Performance Comparison (vs mlx-vlm, decode-only)

| Model | mlxcel | mlx-vlm | vs mlx-vlm | Notes |
|-------|--------|---------|------------|-------|
| llava-bunny | 95.70 | 63.66 | **150%** | |
| aya-vision | 111.11 | 87.87 | **126%** | |
| llava-interleave | 263.41 | 215.71 | **122%** | |
| pixtral | 60.29 | 57.15 | **106%** | |
| molmo2 | 59.54 | 58.65 | **102%** | fast SDPA vision encoder |
| phi3.5-vision | 92.64 | 103.52 | 89% | Only 19 gen tokens |

### Overall

- **Text models with comparison (40):** all of `Outperforming mlx-lm`, `Near parity`, and `Needs optimization` tables combined
- **Beating mlx-lm (>100%):** 36/40 (90%)
- **At 90%+ parity:** 40/40 (100%) of the comparable set
- **VLM models with comparison (6):** 5/6 at or above parity (83%)
- **No comparison available:** 40+ (mlx-lm / mlx-vlm fails to load or no benchmark)

## Performance vs mlx-lm / mlx-vlm baseline (2026-05-19, same-day sweep)

Source CSVs (same M1 Ultra host, mlxcel 0.0.28 with `--cooldown 0`; mlx-lm/mlx-vlm baselines from same 2026-05-19 sweep with `PYLM_BENCH_MAX_GB=65`):

- mlxcel: `benchmarks/metal_m1ultra_2026-05-19.csv`
- mlx-lm: `benchmarks/pylm_m1ultra_2026-05-19.csv` (mlx-lm 0.31.3 dev checkout in `references/mlx-lm` @ `df1d3f3`)
- mlxcel VLM: `benchmarks/metal_m1ultra_vlm_2026-05-19.csv`
- mlx-vlm: `benchmarks/pylm_m1ultra_vlm_2026-05-19.csv` (mlx-vlm dev checkout in `references/mlx-vlm` @ `d85ca4d`)

Both sides were re-run on 2026-05-19 (unlike the M5 Max page, which reused 5-18 baselines). All 4 sweeps use the same `--max-tokens 100`, same `Hello, how are you today?` / `What is in this image?` prompts, and the same >65GB skip threshold (deepseek-v3-4bit, minimax-m2-3bit, qwen3-next-480b-4bit excluded from both sides).

Numbers are decode tok/s. `mlxcel vs mlx-lm` is `mlxcel / mlx-lm` as a percentage; **bold** = mlxcel >= mlx-lm. `FAIL` cells are real load/runtime errors on that backend with this configuration. The mlx-lm checkout used here (`df1d3f3`) differs from the M5 Max page's `ed1fca4`, so some FAIL categories differ.

### Aggregate (text)

- **Comparable text pairs**: 74
- **mlxcel >= mlx-lm**: 14 / 74 (19%)
- **mlxcel >= 90% parity**: 56 / 74 (76%)
- **Average mlxcel/mlx-lm**: 93% (median 96%, range 34%-110%)

### Aggregate (VLM, models with >=5 generated tokens both sides)

- **Comparable VLM pairs**: 17
- **mlxcel >= mlx-vlm**: 8 / 17 (47%)
- **mlxcel >= 90% parity**: 11 / 17 (65%)
- **Average mlxcel/mlx-vlm**: 98% (median 98%, range 77%-119%)

### Text decode (tok/s)

| Model | mlxcel | mlx-lm | mlxcel vs mlx-lm |
|-------|--------|--------|------------------|
| GLM-5-4bit | FAIL | FAIL | - |
| Meta-Llama-3.1-8B-Instruct-4bit | 107.55 | 109.84 | 98% |
| Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | 80.67 | FAIL | - |
| Qwen2.5-1.5B-4bit | 236.02 | 241.41 | 98% |
| Qwen2.5-1.5B-Instruct-4bit | 238.66 | 239.20 | 100% |
| Qwen2.5-7B-Instruct-4bit | 111.29 | 110.90 | **100%** |
| Qwen3.5-0.8B-OptiQ-4bit | FAIL | 265.86 | - |
| Qwen3.5-27B-DFlash | FAIL | FAIL | - |
| Qwen3.5-4B-DFlash | FAIL | FAIL | - |
| aya-expanse-8b-4bit | 107.92 | 112.74 | 96% |
| aya-vision-8b | 110.88 | FAIL | - |
| baichuan-m1-14b-4bit | 46.86 | 49.11 | 95% |
| bunny-llama3-8b-4bit | 101.44 | FAIL | - |
| command-r7b-4bit | 111.42 | 107.75 | **103%** |
| deepseek-coder-1.3b-4bit | 162.80 | FAIL | - |
| deepseek-r1-distill-7b-4bit | 110.52 | 111.34 | 99% |
| deepseek-v2-lite-4bit | 94.74 | 117.06 | 81% |
| deepseek-v3-4bit | FAIL | FAIL | - |
| ernie-4.5-0.3b-4bit | 413.23 | FAIL | - |
| exaone-3.5-2.4b-4bit | 190.43 | 194.65 | 98% |
| exaone4-1.2b-4bit | 213.60 | FAIL | - |
| falcon-mamba-7b-4bit | 30.78 | 91.04 | 34% |
| gemma-2b-4bit | 192.52 | 207.78 | 93% |
| gemma-3-4b-it-4bit | 103.12 | 109.72 | 94% |
| gemma-4-26b-a4b-it-4bit | 70.27 | 72.52 | 97% |
| gemma-4-31B-it-assistant-bf16 | FAIL | FAIL | - |
| gemma-4-31b-4bit | 20.31 | 20.36 | 100% |
| gemma-4-31b-it-4bit | 19.39 | 20.23 | 96% |
| gemma-4-e2b-it-4bit | 118.94 | FAIL | - |
| gemma-4-e2b-it-8bit | 87.56 | FAIL | - |
| gemma-4-e4b-it-4bit | 82.77 | FAIL | - |
| gemma-4-e4b-it-8bit | 59.06 | FAIL | - |
| gemma2-2b-4bit | 135.27 | 153.50 | 88% |
| gemma3-1b-4bit | 196.54 | 211.50 | 93% |
| gemma3-4b-4bit | 103.61 | 109.48 | 95% |
| gemma3n-e2b-4bit | 76.18 | FAIL | - |
| gemma3n-e4b-4bit | 58.96 | FAIL | - |
| gemma3n-e4b-bf16 | 34.36 | 39.02 | 88% |
| glm4-flash-4bit | 45.37 | 49.47 | 92% |
| gpt-oss-120b-4bit | 59.62 | 57.58 | **104%** |
| gpt-oss-20b-mxfp4 | 92.08 | 89.51 | **103%** |
| hunyuan-1.8b-4bit | 176.58 | 200.59 | 88% |
| hunyuan-large-4bit | 44.51 | FAIL | - |
| hunyuan-moe-a13b-bf16 | FAIL | FAIL | - |
| internlm2-7b-4bit | 109.04 | 111.92 | 97% |
| internlm3-8b-4bit | 86.13 | FAIL | - |
| internvl3-1b | FAIL | FAIL | - |
| jamba-v0.1-4bit | 92.02 | 131.04 | 70% |
| llama-3.1-8b-4bit | 108.45 | 110.66 | 98% |
| llama-3.1-8b-bf16 | 35.43 | 35.32 | **100%** |
| llama-3.2-1b-4bit | 332.54 | 418.25 | 80% |
| llama-4-scout-17b-4bit | 36.36 | FAIL | - |
| llava-1.5-7b-4bit | 115.81 | FAIL | - |
| llava-interleave-qwen-0.5b-bf16 | 304.49 | FAIL | - |
| llava-next-mistral-7b-4bit | 112.56 | FAIL | - |
| mamba2-1.3b-4bit | 107.35 | FAIL | - |
| mimo-7b-4bit | 85.14 | 86.17 | 99% |
| minicpm-2b-4bit | 162.79 | 156.47 | **104%** |
| minicpm3-4b-4bit | 79.08 | 73.26 | **108%** |
| minimax-m2-3bit | FAIL | FAIL | - |
| ministral-3b-4bit | 140.72 | 159.34 | 88% |
| mistral-small-3.1-24b-4bit | 31.82 | 31.97 | 100% |
| mixtral-8x7b-4bit | 53.49 | 54.91 | 97% |
| molmo-7b | FAIL | FAIL | - |
| molmo2-4b | 59.66 | FAIL | - |
| nemotron-h-30b-4bit | 89.96 | 93.34 | 96% |
| nemotron-nas-30b-4bit | 89.10 | 92.93 | 96% |
| olmo-1b-4bit | 209.70 | FAIL | - |
| olmo2-7b-4bit | 103.37 | 110.88 | 93% |
| olmo3-32b-4bit | 21.95 | 21.57 | **102%** |
| paligemma2-3b-6bit | 0.00 | FAIL | - |
| phi-2-4bit | 52.59 | FAIL | - |
| phi-3-mini-4bit | 143.69 | 171.36 | 84% |
| phi-3.5-mini-4bit | 116.86 | 166.30 | 70% |
| phi-3.5-moe-4bit | 76.13 | 69.28 | **110%** |
| phi-3.5-vision-4bit | 117.59 | FAIL | - |
| phi-4-4bit | 58.03 | 58.68 | 99% |
| pixtral-12b-4bit | 69.10 | 69.49 | 99% |
| qwen1.5-moe-a2.7b-4bit | 142.23 | 144.98 | 98% |
| qwen2-vl-2b-4bit | 143.25 | 236.86 | 60% |
| qwen2.5-0.5b-4bit | 329.45 | 315.48 | **104%** |
| qwen2.5-0.5b-bf16 | FAIL | FAIL | - |
| qwen2.5-7b-4bit | 111.18 | 111.38 | 100% |
| qwen2.5-7b-8bit | 69.61 | 70.46 | 99% |
| qwen2.5-vl-3b-4bit | 98.81 | 160.42 | 62% |
| qwen3-0.6b-4bit | 198.08 | 299.61 | 66% |
| qwen3-1.7b-4bit | 193.24 | 221.37 | 87% |
| qwen3-30b-a3b-4bit | 70.56 | 70.18 | **101%** |
| qwen3-4b-4bit | 118.85 | 123.92 | 96% |
| qwen3-8b-4bit | 80.64 | 84.54 | 95% |
| qwen3-moe-4bit | 70.34 | 69.67 | **101%** |
| qwen3-next-480b-4bit | FAIL | FAIL | - |
| qwen3-vl-2b-4bit | 211.50 | 222.67 | 95% |
| qwen3-vl-30b-a3b-4bit | 67.65 | 70.04 | 97% |
| qwen3-vl-32b-4bit | 21.07 | 21.99 | 96% |
| qwen3-vl-4b-4bit | 117.49 | 124.02 | 95% |
| qwen3-vl-8b-4bit | 81.16 | 84.46 | 96% |
| qwen3.5-0.8b-4bit | 205.52 | 269.52 | 76% |
| qwen3.5-27b-4bit | 24.11 | 25.93 | 93% |
| qwen3.5-2b-4bit | 175.73 | 211.68 | 83% |
| qwen3.5-35b-a3b-4bit | 70.21 | 76.44 | 92% |
| qwen3.5-4b-4bit | 101.21 | 115.60 | 88% |
| qwen3.5-9b-4bit | 72.02 | 81.27 | 89% |
| qwen3.5-9b-bf16 | 31.11 | 34.22 | 91% |
| qwen3.6-35b-a3b-4bit | 67.07 | 73.18 | 92% |
| smollm-135m-4bit | 352.11 | 375.91 | 94% |
| smollm3-3b-4bit | 136.43 | 141.66 | 96% |
| solar-open-100b-4bit | 36.20 | 35.69 | **101%** |
| stablelm-1.6b-4bit | 270.47 | 280.65 | 96% |
| starcoder2-3b-4bit | 167.20 | 166.17 | **101%** |
| youtu-vl-4b-instruct | 0.00 | FAIL | - |

### VLM decode (tok/s)

| Model | mlxcel | mlx-vlm | mlxcel vs mlx-vlm |
|-------|--------|--------|------------------|
| Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | 68.89 | FAIL | - |
| aya-vision-8b | 111.11 | 103.74 | **107%** |
| bunny-llama3-8b-4bit | 95.70 | FAIL | - |
| gemma-3-4b-it-4bit | 77.84 | 97.36 | 80% |
| gemma-4-26b-a4b-it-4bit | 65.65 | 61.07 | **107%** |
| gemma-4-31b-4bit | 15.54 | 20.30 | 77% |
| gemma-4-31b-it-4bit | 18.57 | 19.78 | 94% |
| gemma-4-e2b-it-4bit | 104.47 | 97.19 | **107%** |
| gemma-4-e2b-it-8bit | 80.92 | 91.06 | 89% |
| gemma-4-e4b-it-4bit | 74.13 | 70.34 | **105%** |
| gemma-4-e4b-it-8bit | 54.78 | 63.25 | 87% |
| gemma3-4b-4bit | 80.13 | 93.79 | 85% |
| gemma3n-e2b-4bit | 71.05 | 59.57 | **119%** |
| gemma3n-e4b-4bit | 55.82 | 50.00 | **112%** |
| gemma3n-e4b-bf16 | 31.57 | 36.18 | 87% |
| internvl3-1b | FAIL | 264.40 | - |
| llama-4-scout-17b-4bit | 35.66 | FAIL | - |
| llava-1.5-7b-4bit | 103.82 | FAIL | - |
| llava-interleave-qwen-0.5b-bf16 | 263.41 | 225.15 | **117%** |
| llava-next-mistral-7b-4bit | 106.65 | 109.51 | 97% |
| ministral-3b-4bit | 124.42 | FAIL | - |
| mistral-small-3.1-24b-4bit | 29.78 | FAIL | - |
| molmo-7b | FAIL | 38399.52 (anomalous) | - |
| molmo2-4b | 59.54 | 60.87 | 98% |
| paligemma2-3b-6bit | 38.37 | 70.45 | 54% |
| phi-3.5-vision-4bit | 92.64 | 92.53 | **100%** |
| pixtral-12b-4bit | 60.29 | FAIL | - |
| qwen2-vl-2b-4bit | 0.00 | FAIL | - |
| qwen2.5-vl-3b-4bit | 97.66 | FAIL | - |
| qwen3-vl-2b-4bit | 167.93 | FAIL | - |
| qwen3-vl-30b-a3b-4bit | 22.41 | FAIL | - |
| qwen3-vl-32b-4bit | 18.47 | FAIL | - |
| qwen3-vl-4b-4bit | 94.79 | FAIL | - |
| qwen3-vl-8b-4bit | 66.35 | FAIL | - |
| qwen3.5-0.8b-4bit | 212.96 | FAIL | - |
| qwen3.5-27b-4bit | 24.96 | FAIL | - |
| qwen3.5-2b-4bit | 178.66 | FAIL | - |
| qwen3.5-35b-a3b-4bit | 68.55 | FAIL | - |
| qwen3.5-4b-4bit | 101.63 | FAIL | - |
| qwen3.5-9b-4bit | 73.21 | FAIL | - |
| qwen3.5-9b-bf16 | 31.76 | FAIL | - |
| qwen3.6-35b-a3b-4bit | 68.62 | FAIL | - |
| youtu-vl-4b-instruct | 20.70 | FAIL | - |

### mlx-lm / mlx-vlm fail categories (text + VLM)

mlx-lm side text FAILs are mostly the same architecture support gaps as on M5 Max: VLM-only loaders routed through the text path (`aya-vision-8b`, `bunny-llama3-8b-4bit`, `llava-*`, `paligemma2-3b-6bit`, `phi-3.5-vision-4bit`, `gemma-4-e{2,4}b-it-{4,8}bit`, `gemma3n-e{2,4}b-4bit`), `transformers` config schema drift (`exaone4-1.2b-4bit`), custom remote code refused (`hunyuan-*`, `deepseek-coder-1.3b-4bit`, `ernie-4.5-0.3b-4bit`), unsupported architectures (`internvl3-1b`, `molmo-7b`, `internlm3-8b-4bit`, `nemotron-*`-omni and 4 drafter checkpoints), `ModelArgs` mismatch (`mamba2-1.3b-4bit`, `phi-2-4bit`, `gemma-4-31B-it-assistant-bf16`), and 4 oversize SKIPs (deepseek-v3-4bit, minimax-m2-3bit, qwen3-next-480b-4bit applied symmetrically). These are backend-specific compatibility outcomes for the `df1d3f3` checkout and should not be counted as silent mlxcel performance wins.

mlx-vlm side VLM FAILs: most text models trivially fail when routed through `mlx-vlm` (no image processor for text-only checkpoints), so the VLM table effectively compares only the ~20 actual VLM models. Two genuine VLM cells where mlx-vlm works but mlxcel does not (`internvl3-1b`, `molmo-7b`) are architectures mlxcel does not yet implement — the molmo-7b 38k tok/s reading is a metric artifact (1 generated token) and should not be read as real throughput.

The four mlxcel rows where the gap is widest on text — `falcon-mamba-7b-4bit` (34%), `qwen2-vl-2b-4bit` (60%), `qwen2.5-vl-3b-4bit` (62%), `qwen3-0.6b-4bit` (66%) — are the same model classes flagged on M5 Max as having the most headroom: mamba SSM kernel, MRoPE VLM text-only fast path, and the small-Qwen3 decode hot path.

## Known Issues

| Model | Issue | Priority |
|-------|-------|----------|
| hunyuan-moe-a13b-bf16 | Warmup failure; Tiktoken tokenizer; bf16 | High |
| qwen2.5-0.5b-bf16 | Warmup failure; bf16 non-quantized | Medium |
| Qwen3.5-4B-DFlash / Qwen3.5-27B-DFlash | Drafter checkpoint — not a standalone inference model | Low |
| Qwen3.5-0.8B-OptiQ-4bit | Warmup failure on new OptiQ quant variant | Medium |
| gemma-4-31B-it-assistant-bf16 | Drafter checkpoint — not a standalone inference model | Low |
| falcon-mamba | Chat template causes early EOS (only 2 tokens); decode now 30.78 tok/s | Medium |
| paligemma | Only 2 VLM gen tokens | High |
| youtu-vl-4b-instruct | VLM produces only 1 token; text-only produces 0 tokens | Medium |
| llama4 | Repetitive output on long generations | Low |
| moondream3 | Image output garbled; text-only works; needs reconstruct_from_crops | Medium |
| internvl3 | Warmup failure; unsupported architecture | Medium |
| molmo-7b | Warmup failure; unsupported architecture (only molmo2 supported) | Low |
| qwen-vl family | Broadcast shape errors on M5 Max with 224x224 images | Medium |
| GLM-5-4bit | Persistent warmup failure | Medium |

## Notes

- All tests use 4-bit quantized models unless noted (Nemotron uses 8-bit)
- Performance measured with `--profile` flag (separate prefill/decode timing)
- vs mlx-lm percentage is based on **decode speed only**
- Prefill tok/s and decode tok/s are reported separately; units omitted from table headers for brevity
- Prefill shown as "-" for models not measured in this run

## Tokenizer Support

| Format | File | Models | Crate |
|--------|------|--------|-------|
| HuggingFace | `tokenizer.json` | Most models | `tokenizers` |
| SentencePiece | `tokenizer.model` | Gemma, Llama 1/2, older models | `sentencepiece` |
| Tiktoken | `*.tiktoken` | HunYuan MoE (13B) | Custom BPE (fancy-regex) |

## TurboQuant KV cache — M1 Ultra latest readings

Latest available TurboQuant readings for the M1 Ultra benchmark page. These measurements are separate from the standard model sweep above because they exercise KV-cache storage modes rather than model architecture coverage.

| Item | Value |
|------|-------|
| Hardware | Apple M1 Ultra, 128 GB unified memory |
| Model | `Meta-Llama-3.1-8B-Instruct-4bit` |
| CSV | `benchmarks/turbo_kv/2026-04-29_Apple_M1_Ultra_Meta-Llama-3.1-8B-Instruct-4bit.csv` |

CSV rows where `stage=prefill` include a single-token follow-up only to populate the KV cache. Use `prefill_tok_s` for those rows and ignore their sub-millisecond `decode_tok_s` values.

### Decode @ 4K context

| Mode | Decode tok/s | x FP16 | Gate context |
|------|-------------:|-------:|--------------|
| `fp16` | 90.36 | 1.000x | baseline |
| `int8` | 61.24 | 0.678x | tracking only |
| `turbo4-asym` | 3.92 | 0.043x | below M5 Max gate |
| `turbo4` | 16.34 | 0.181x | below M5 Max gate |
| `turbo4-delegated` | 18.22 | 0.202x | below M5 Max gate |

`turbo4-asym` produced 51 tokens before early EOS; the other Turbo modes ran the full 80-token decode budget.

### Prefill @ 8K context

| Mode | Prefill tok/s | x FP16 | Gate context |
|------|--------------:|-------:|--------------|
| `fp16` | 678.28 | 1.000x | baseline |
| `int8` | 676.82 | 0.998x | tracking only |
| `turbo4-asym` | 471.12 | 0.694x | below M5 Max gate |
| `turbo4` | 365.07 | 0.538x | below M5 Max gate |
| `turbo4-delegated` | 678.96 | 1.001x | meets best-effort target |

### M1 Ultra reading

- `turbo4-delegated` is the practical Turbo mode on this M1 Ultra reading when prefill latency matters.
- Turbo decode modes are far below FP16 decode throughput on this hardware generation.
- `int8` has near-FP16 prefill and lower decode throughput, with a half-sized KV cache.
- Use compressed Turbo modes on M1 Ultra only when the memory savings outweigh the decode cost for the workload.
