# Model Compatibility & Performance Tests (M1 Ultra)

Compatibility and performance testing for mlxcel models on **Mac Studio M1 Ultra 128GB**, with comparison against Python mlx-lm / mlx-vlm.

## Test Environment

| Item | Value |
|------|-------|
| **Hardware** | Mac Studio M1 Ultra, 128GB RAM |
| **OS** | macOS 26.4 (Tahoe) |
| **mlxcel version** | 0.1.0 |
| **MLX version** | post-0.32.0 pin (commit 84961223, via mlxcel-core) |
| **Bench harness** | `mlxcel-bench-decode` (model load, warmup, and measured pass in one process) |
| **mlx-lm baseline** | 0.31.3 (dev checkout `references/mlx-lm` @ `df1d3f3` — "Fix Gemma 4 sanitize() not stripping KV projections for shared layers" ml-explore/mlx-lm#1240) |
| **mlx-vlm baseline** | dev checkout `references/mlx-vlm` @ `d85ca4d` — "Compatibility bridge for non-VL models" Blaizzy/mlx-vlm#1181 |
| **Test Prompt** | "Hello, how are you today?" (text) / "What is in this image?" (VLM) |
| **Max Tokens** | 100 (measured pass); 20 (warmup pass, same process) |
| **Test Date** | 2026-05-19 full sweep (baseline); 2026-05-28 full text + VLM re-benchmark on mlxcel 0.1.0 (`--cooldown 0`) |
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
| llama3 | Llama-3.2-1B-Instruct-4bit | ✅ | 1951.01 | 373.43 | 89% | mlx-lm: 418.25; only 48 tokens |
| llama3 (8B bf16) | Llama-3.1-8B-Instruct (bf16) | ✅ | 427.93 | 35.77 | **101%** | mlx-lm: 35.32; non-quantized |
| llama3.1 | Llama-3.1-8B-Instruct-4bit | ✅ | 486.38 | 109.49 | 99% | mlx-lm: 110.66; only 54 tokens |
| llama4 | Llama-4-Scout-17B-16E-Instruct-4bit | ⚠️ | 120.77 | 36.66 | - | mlx-lm: FAIL; long outputs repetitive |
| qwen2 | Qwen2.5-0.5B-Instruct-4bit | ✅ | 1243.98 | 349.52 | **111%** | mlx-lm: 315.48 |
| qwen2 (7B 4bit) | Qwen2.5-7B-Instruct-4bit | ✅ | 318.26 | 112.68 | **102%** | mlx-lm: 110.90 |
| qwen2 (7B 8bit) | Qwen2.5-7B-Instruct-8bit | ✅ | 314.56 | 71.15 | **101%** | mlx-lm: 70.46; 8-bit quantized |
| qwen3 | Qwen3-0.6B-4bit | ✅ | 572.83 | 284.03 | 95% | mlx-lm: 299.61 |
| qwen3 (1.7B) | Qwen3-1.7B-4bit | ✅ | 416.92 | 197.55 | 89% | mlx-lm: 221.37 |
| qwen3 (4B) | Qwen3-4B-4bit | ✅ | 250.77 | 121.42 | 98% | mlx-lm: 123.92 |
| qwen3 (8B) | Qwen3-8B-4bit | ✅ | 166.07 | 81.38 | 96% | mlx-lm: 84.54 |
| qwen3_5 (0.8B) | Qwen3.5-0.8B-4bit | ✅ | 541.70 | 244.18 | 91% | mlx-lm: 269.52; Hybrid GatedDeltaNet |
| qwen3_5 (2B) | Qwen3.5-2B-4bit | ✅ | 423.60 | 179.54 | 85% | mlx-lm: 211.68; Hybrid GatedDeltaNet; only 36 tokens |
| qwen3_5 (4B) | Qwen3.5-4B-4bit | ✅ | 252.63 | 102.31 | 89% | mlx-lm: 115.60; Hybrid GatedDeltaNet; only 36 tokens |
| qwen3_5 (9B 4bit) | qwen3.5-9B-4bit | ✅ | 158.73 | 72.75 | 90% | mlx-lm: 81.27; Hybrid GatedDeltaNet; only 29 tokens |
| qwen3_5 (9B bf16) | qwen3.5-9B (bf16) | ✅ | 152.39 | 31.85 | 93% | mlx-lm: 34.22; bf16, not quantized; Hybrid GatedDeltaNet (compiled fused kernel) |
| qwen3_5 (27B) | qwen3.5-27B-4bit | ✅ | 53.45 | 24.52 | 95% | mlx-lm: 25.93; Hybrid Transformer+GatedDeltaNet; VLM wrapper format |
| qwen3_6 | qwen3.6-35B-A3B-4bit | ✅ | 244.19 | 70.36 | 96% | mlx-lm: 73.18; MoE architecture; 100 tokens |
| qwen3_next | qwen3-next-480B-4bit | ⏳ | - | SKIP | - | Qwen3Next 480B architecture; >65GB skipped on 128GB host |
| qwen2 (1.5B) | Qwen2.5-1.5B-Instruct-4bit | ✅ | 874.09 | 243.11 | **102%** | mlx-lm: 239.20; 100 tokens |
| qwen2 (1.5B base) | Qwen2.5-1.5B-4bit | ✅ | 756.00 | 244.93 | **101%** | mlx-lm: 241.41; base variant; 100 tokens |
| phi | phi-2-hf-4bit-mlx | ✅ | 146.30 | 62.61 | - | mlx-lm fails to load; only 1 token (likely EOS) |
| phi3 | Phi-3-mini-4k-instruct-4bit | ✅ | 198.89 | 172.17 | **100%** | mlx-lm: 171.36; only 25 tokens |
| phi3small | Phi-3.5-mini-instruct-4bit | ✅ | 231.09 | 167.07 | **100%** | mlx-lm: 166.30; only 40 tokens |
| phi4 | Phi-4-4bit | ✅ | 112.86 | 58.83 | **100%** | mlx-lm: 58.68 |
| smollm3 | SmolLM-135M-Instruct-4bit | ✅ | 607.34 | 383.55 | **102%** | mlx-lm: 375.91 |
| smollm3 (3B) | SmolLM3-3B-4bit | ✅ | 582.75 | 137.92 | 97% | mlx-lm: 141.66 |
| stablelm | stablelm-2-1_6b-chat-4bit | ✅ | 667.72 | 285.79 | **102%** | mlx-lm: 280.65; only 59 tokens |
| starcoder2 | starcoder2-3b-4bit | ✅ | 177.03 | 172.82 | **104%** | mlx-lm: 166.17 |
| olmo | OLMo-1B-hf-4bit | ✅ | 188.64 | 212.53 | - | mlx-lm: FAIL |
| olmo2 | OLMo2-7B-4bit | ✅ | 278.00 | 104.17 | 94% | mlx-lm: 110.88; only 27 tokens |
| olmo3 | OLMo3.1-32B-4bit | ✅ | 81.70 | 22.14 | **103%** | mlx-lm: 21.57 |
| minicpm | MiniCPM-2B-sft-bf16-4bit | ✅ | 298.90 | 165.32 | **106%** | mlx-lm: 156.47 |
| mimo | MiMo-7B-RL-4bit | ✅ | 232.90 | 86.26 | **100%** | mlx-lm: 86.17 |

## Gemma Family

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| gemma | gemma-2b-it-4bit | ✅ | 332.44 | 194.67 | 94% | mlx-lm: 207.78; only 49 tokens |
| gemma2 | gemma-2-2b-it-4bit | ✅ | 340.38 | 169.77 | **111%** | mlx-lm: 153.50; only 18 tokens |
| gemma3 | gemma-3-1b-it-4bit | ✅ | 412.77 | 232.91 | **110%** | mlx-lm: 211.50; only 34 tokens |
| gemma3 (4B) | gemma-3-4b-it-4bit | ✅ | 204.48 | 117.12 | **107%** | mlx-lm: 109.48; only 86 tokens |
| gemma4 (31B) | gemma-4-31b-4bit | ✅ | 16.81 | 20.48 | **101%** | mlx-lm: 20.36 |
| gemma4 (31B-it) | gemma-4-31b-it-4bit | ✅ | 47.28 | 19.61 | 97% | mlx-lm: 20.23; instruction-tuned variant |
| gemma4 (26B A4B) | gemma-4-26b-a4b-it-4bit | ✅ | 166.86 | 71.76 | 99% | mlx-lm: 72.52; only 26 tokens |
| gemma4 (E2B 4bit) | gemma-4-e2b-it-4bit | ✅ | 254.45 | 119.67 | - | mlx-lm: FAIL; only 34 tokens |
| gemma4 (E2B 8bit) | gemma-4-e2b-it-8bit | ✅ | 230.17 | 89.23 | - | mlx-lm: FAIL; only 38 tokens |
| gemma4 (E4B 4bit) | gemma-4-e4b-it-4bit | ✅ | 182.08 | 84.43 | - | mlx-lm: FAIL; only 25 tokens |
| gemma4 (E4B 8bit) | gemma-4-e4b-it-8bit | ✅ | 163.78 | 60.61 | - | mlx-lm: FAIL; only 39 tokens |
| gemma3n | gemma-3n-E2B-it-4bit | ✅ | 227.00 | 78.75 | - | mlx-lm: FAIL; only 69 tokens |
| gemma3n (E4B) | gemma-3n-E4B-it-4bit | ✅ | 171.02 | 61.51 | - | mlx-lm: FAIL; only 74 tokens |
| gemma3n (E4B bf16) | gemma-3n-E4B-it (bf16) | ✅ | 174.94 | 34.96 | 90% | mlx-lm: 39.02; bf16; AltUp/MLP decode graph scheduling |
| recurrent_gemma | - | ⏳ | - | - | - | Griffin SSM+attention hybrid |

## EXAONE

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| exaone | EXAONE-3.5-2.4B-Instruct-4bit | ✅ | 687.90 | 200.53 | **103%** | mlx-lm: 194.65 |
| exaone4 | exaone-4.0-1.2b-4bit | ✅ | 409.39 | 247.55 | - | mlx-lm: FAIL; only 18 tokens |
| exaone_moe | - | ⏳ | - | - | - | |

## Cohere Command R

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| cohere | c4ai-command-r7b-12-2024-4bit | ✅ | 81.17 | 114.34 | **106%** | mlx-lm: 107.75 |
| cohere2 | aya-expanse-8b-4bit | ✅ | 97.98 | 110.58 | 98% | mlx-lm: 112.74 |

## MoE (Mixture of Experts)

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| minimax | MiniMax-M2-3bit | ⏳ | - | SKIP | - | mlx-lm: 18.2; >65GB skipped on 128GB host (93GB) |
| mixtral | Mixtral-8x7B-Instruct-v0.1-4bit | ✅ | 81.80 | 54.66 | 100% | mlx-lm: 54.91; only 73 tokens |
| qwen2_moe | Qwen1.5-MoE-A2.7B-Chat-4bit | ✅ | 390.03 | 147.24 | **102%** | mlx-lm: 144.98; only 43 tokens |
| qwen3_moe | Qwen3-30B-A3B-4bit | ✅ | 191.96 | 72.17 | **103%** | mlx-lm: 70.18 |
| qwen3_5_moe | qwen3.5-35B-A3B-4bit | ✅ | 239.05 | 71.38 | 93% | mlx-lm: 76.44; Hybrid GatedDeltaNet + MoE (256 experts); only 34 tokens |
| phimoe | Phi-3.5-MoE-instruct-4bit | ✅ | 112.10 | 77.71 | **112%** | mlx-lm: 69.28 |
| solar_open | Solar-Open-100B-4bit | ✅ | 75.37 | 36.26 | **102%** | mlx-lm: 35.69; 128 experts, top-8; layer-eval skip; 54GB |
| solar_open (int4) | Solar-Open-100B-int4 | ✅ | - | 11.55 | - | mlx-lm: fails to load; 128 experts, top-8; int4 quantization; 54GB |
| olmoe | - | ⏳ | - | - | - | |
| gpt_oss (20B) | gpt-oss-20b-MXFP4-Q4 | ✅ | 286.09 | 93.46 | **104%** | mlx-lm: 89.51; MXFP4 quantization; 32 experts; bf16 decode fix |
| gpt_oss (120B) | gpt-oss-120b-4bit | ✅ | 114.12 | 61.19 | **106%** | mlx-lm: 57.58; 128 experts, top-4; 61GB model; bf16 decode fix |

## DeepSeek Family

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| deepseek | deepseek-coder-1.3b-instruct-4bit | ✅ | 1306.40 | 165.72 | - | mlx-lm: FAIL |
| deepseek_v2 | DeepSeek-V2-Lite-Chat-4bit | ✅ | 207.71 | 112.45 | 96% | mlx-lm: 117.06; only 18 tokens |
| deepseek_r1 | DeepSeek-R1-Distill-Qwen-7B-4bit | ✅ | 156.36 | 113.52 | **102%** | mlx-lm: 111.34 |
| deepseek_v3 | deepseek-v3-4bit | ⏳ | - | SKIP | - | MoE + MLA; >65GB skipped on 128GB host (99GB) |
| deepseek_v32 | - | ⏳ | - | - | - | |

## MLA (Multi-head Latent Attention)

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| minicpm3 | MiniCPM3-4B-4bit | ✅ | 241.44 | 80.78 | **110%** | mlx-lm: 73.26 |

## Nemotron Family

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| nemotron_h | Nemotron-H-30B-4bit | ✅ | 168.46 | 91.68 | 98% | mlx-lm: 93.34; Hybrid Mamba2+Transformer+MoE; SSM Metal kernel |
| nemotron_nas | Nemotron-NAS-30B-A3B-4bit | ✅ | 169.94 | 92.26 | 99% | mlx-lm: 92.93; Hybrid Mamba2+Transformer+MoE |
| nemotron_h_nano_omni | Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | ✅ | 171.39 | 87.56 | - | mlx-lm: FAIL; NEW (5-19); Mamba2+Transformer+MoE+Parakeet audio; 100 tokens |

## SSM / Mamba Models

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| mamba | Falcon-Mamba-7B-4bit | ⚠️ | 91.13 | 42.83 | 47% | mlx-lm: 91.04; only 2 tokens due to chat template EOS |
| mamba2 | mamba2-1.3b-4bit | ✅ | 172.43 | 101.13 | - | mlx-lm: FAIL |
| jamba | Jamba-v0.1-4bit | ✅ | 336.05 | 123.65 | 94% | mlx-lm: 131.04; only 76 tokens |
| rwkv7 | - | ⏳ | - | - | - | RWKV v7 linear attention |

## Chinese / Asian Language Models

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| baichuan | Baichuan-M1-14B-Instruct-4bit | ✅ | 53.98 | 40.13 | 82% | mlx-lm: 49.11; only 39 tokens |
| glm4 | GLM-4-Flash-4bit | ✅ | 132.26 | 47.92 | 97% | mlx-lm: 49.47; Only 18 tokens |
| glm4_moe | - | ⏳ | - | - | - | |
| glm4_moe_lite | GLM-4.7-Flash-4bit | ✅ | - | 31.54 | 76% | mlx-lm: 41.55; only 18 tokens |
| glm5 | GLM-5-4bit | ❌ | - | FAIL | - | warmup failure (persistent) |
| internlm2 | InternLM2-7B-4bit | ✅ | 215.62 | 110.98 | 99% | mlx-lm: 111.92 |
| internlm3 | internlm3-8b-instruct-4bit | ✅ | 310.99 | 87.98 | - | mlx-lm: FAIL |
| ernie4_5 | ERNIE-4.5-0.3B-Instruct-4bit | ✅ | 1009.89 | 510.17 | - | mlx-lm: FAIL |
| ernie4_5_moe | - | ⏳ | - | - | - | |
| hunyuan_moe | Hunyuan-Large-Instruct-4bit | ✅ | 65.24 | 45.22 | - | mlx-lm: FAIL |
| hunyuan_moe_13b | HunYuan-MoE-A13B-Instruct (bf16) | ❌ | - | FAIL | - | mlx-lm: fails to load; Tiktoken tokenizer; bf16; warmup failure |
| hunyuan_v1_dense | Hunyuan-1.8B-Instruct-4bit | ✅ | 277.87 | 188.41 | 94% | mlx-lm: 200.59; only 41 tokens |
| kimi_linear | - | ⏳ | - | - | - | Kimi linear attention (Moonshot) |
| step3p5 | - | ⏳ | - | - | - | Step 3.5 (StepFun) |

## Other Models

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| ministral3 | Ministral-3B-Instruct-4bit | ✅ | 904.59 | 144.52 | 91% | mlx-lm: 159.34; VLM wrapper; text-only mode; only 34 tokens |
| mistral4 | - | ⏳ | - | - | - | MLA + MoE; implemented but no MLX model available |
| moondream3 | moondream3-preview-4bit | ⚠️ | - | 8.45 | - | mlx-lm: fails to load; text-only test; SigLIP + MLP; image output garbled; only 14 tokens |
| longcat_flash | - | ⏳ | - | - | - | |
| longcat_flash_ngram | - | ⏳ | - | - | - | |
| mistral_small | mistral-small-3.1-24b-4bit | ✅ | 33.87 | 32.07 | **100%** | mlx-lm: 31.97; text-only mode |

## Vision-Language Models (VLM)

| Model | Test Model | Status | Prefill | Decode | vs mlx-vlm | Notes |
|-------|------------|--------|---------|--------|------------|-------|
| gemma3 | gemma-3-4b-it-4bit | ✅ | 241.77 | 87.01 | 93% | mlx-vlm: 93.79; SigLIP + AvgPool; 275 prompt, 16 gen |
| gemma3n (E2B) | gemma-3n-E2B-it-4bit | ✅ | 771.21 | 72.83 | **122%** | mlx-vlm: 59.57; MobileNetV5 + MSFA; 273 prompt, 29 gen |
| gemma3n (E4B bf16) | gemma-3n-E4B-it (bf16) | ✅ | 659.78 | 32.46 | 90% | mlx-vlm: 36.18; MobileNetV5 + MSFA; bf16 prefill path retune; bf16; 273 prompt, 24 gen |
| gemma3n (E4B 4bit) | gemma-3n-E4B-it-4bit | ✅ | 492.52 | 57.50 | **115%** | mlx-vlm: 50.00; 273 prompt, 33 gen |
| gemma4 (E2B 4bit) | gemma-4-e2b-it-4bit | ✅ | 732.53 | 106.92 | **110%** | mlx-vlm: 97.19; 274 prompt, 100 gen |
| gemma4 (E2B 8bit) | gemma-4-e2b-it-8bit | ✅ | 724.08 | 81.98 | 90% | mlx-vlm: 91.06; 274 prompt, 100 gen |
| gemma4 (E4B 4bit) | gemma-4-e4b-it-4bit | ✅ | 460.74 | 75.15 | **107%** | mlx-vlm: 70.34; 274 prompt, 54 gen |
| gemma4 (E4B 8bit) | gemma-4-e4b-it-8bit | ✅ | 455.71 | 56.16 | 89% | mlx-vlm: 63.25; 274 prompt, 35 gen |
| gemma4 (31B 4bit) | gemma-4-31b-4bit | ✅ | 84.50 | 15.82 | 78% | mlx-vlm: 20.30; 274 prompt, 100 gen |
| gemma4 (31B-it 4bit) | gemma-4-31b-it-4bit | ✅ | 88.04 | 18.86 | 95% | mlx-vlm: 19.78; 274 prompt, 100 gen |
| gemma4 (26B A4B) | gemma-4-26b-a4b-it-4bit | ✅ | 288.28 | 66.40 | **109%** | mlx-vlm: 61.07; 277 prompt, 28 gen |
| llava 1.5 | llava-1.5-7b-4bit | ✅ | 754.41 | 104.03 | - | CLIP + MLP; Vicuna-7b; 583 prompt, 100 gen; mlx-vlm requires PyTorch |
| llava-interleave | llava-interleave-qwen-0.5b-bf16 | ✅ | 3961.62 | 265.57 | **118%** | mlx-vlm: 225.15; SigLIP + MLP; Qwen2-0.5b; 754 prompt, 36 gen |
| llava-next | llava-v1.6-mistral-7b-4bit | ✅ | 715.00 | 107.17 | 98% | mlx-vlm: 109.51; CLIP + MLP; Mistral; 590 prompt, 100 gen; mlx-vlm template error |
| llava-bunny | Bunny-Llama-3-8B-V-4bit | ✅ | 673.16 | 95.36 | - | mlx-vlm: FAIL; SigLIP + MLP; Llama3; 746 prompt, 37 gen |
| llama4 | Llama-4-Scout-17B-16E-Instruct-4bit | ✅ | 180.73 | 35.88 | - | mlx-vlm: FAIL; 162 prompt, 100 gen |
| aya-vision | aya-vision-8b | ✅ | 444.01 | 113.59 | **109%** | mlx-vlm: 103.74; SigLIP + SwiGLU; Cohere2; 176 prompt, 100 gen |
| paligemma | paligemma2-3b (6-bit) | ⚠️ | 1477.45 | 50.25 | 71% | mlx-vlm: 70.45; SigLIP + Linear; Gemma2; 1032 prompt, only 2 gen tokens |
| pixtral | pixtral-12b-4bit | ✅ | 447.97 | 60.25 | - | mlx-vlm: FAIL; Pixtral ViT; Mistral; 4102 prompt, 100 gen |
| mistral3 | mistral-small-3.1-24b-4bit | ✅ | 128.84 | 29.72 | - | mlx-vlm: FAIL; Pixtral ViT + PatchMerger; Mistral; 3032 prompt, 100 gen; mlx-vlm error |
| ministral3 | Ministral-3B-Instruct-4bit | ✅ | 526.95 | 125.75 | - | mlx-vlm: FAIL; Pixtral ViT; 3566 prompt, 100 gen |
| phi3.5-vision | Phi-3.5-vision-instruct-4bit | ✅ | 991.67 | 122.63 | **133%** | mlx-vlm: 92.53; CLIP + HD tiling; Phi3; 773 prompt, 19 gen |
| phi4mm | phi-4-multimodal-instruct (bf16) | ✅ | 571.90 | 25.42 | - | SigLIP + HD transform + AvgPool2d; Phi3; SuScaledRoPE + runtime LoRA; 2635 tokens; 12GB bf16 |
| moondream3 | moondream3-preview-4bit | ⚠️ | 1.36 | 10.05 | - | SigLIP + MLP; image output garbled; only 63 tokens |
| minicpm-o | MiniCPM-o-2_6-4bit | ✅ | 33.67 | 70.80 | - | SigLIP + Resampler; Qwen3; 80 tokens |
| molmo | Molmo-7B | ✅ | 579.49 | 81.61 | - | CLIP ViT + attention pooling + OLMo text; mlx-vlm baseline is a 1-token anomaly; 327 prompt, 100 gen |
| molmo2 | molmo2-4b | ✅ | 727.26 | 60.31 | 99% | mlx-vlm: 60.87; fast SDPA vision encoder; 430 prompt, 100 gen |
| internvl3 | InternVL3-1B | ✅ | 1902.58 | 228.83 | 87% | mlx-vlm: 264.40; InternViT + pixel-shuffle + Qwen2; 293 prompt, 8 gen |
| nemotron-omni | Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | ✅ | 263.15 | 69.37 | - | mlx-vlm: FAIL; NEW (5-19); Mamba2+Transformer+MoE+Parakeet audio; 100 gen |
| youtu-vl | youtu-vl-4b-instruct | ⚠️ | 408.21 | 24.47 | - | mlx-vlm: FAIL; NEW (5-19); only 1 gen token |
| qwen2-vl | Qwen2-VL-2B-Instruct-4bit | ✅ | 788.00 | 124.99 | - | Custom ViT + MRoPE; VLM image mode fixed; 100 gen |
| qwen2.5-vl | Qwen2.5-VL-3B-Instruct-4bit | ✅ | 599.48 | 97.81 | - | Windowed ViT + MRoPE; 91 prompt, 46 gen; mlx-vlm requires PyTorch |
| qwen3-vl | Qwen3-VL-2B-Instruct-4bit | ✅ | 766.49 | 162.21 | - | mlx-vlm: FAIL; DeepStack + vectorized MRoPE; 100 gen |
| qwen3-vl (4B) | Qwen3-VL-4B-Instruct-4bit | ✅ | 490.85 | 90.14 | - | mlx-vlm: FAIL; DeepStack + vectorized MRoPE; 100 gen |
| qwen3-vl (8B) | Qwen3-VL-8B-Instruct-4bit | ✅ | 313.03 | 62.62 | - | mlx-vlm: FAIL; DeepStack + vectorized MRoPE; 100 gen |
| qwen3-vl (32B) | Qwen3-VL-32B-Instruct-4bit | ✅ | 92.07 | 17.92 | - | mlx-vlm: FAIL; DeepStack + vectorized MRoPE; 100 gen |
| qwen3-vl-moe | Qwen3-VL-30B-A3B-Instruct-4bit | ✅ | 296.81 | 40.88 | - | mlx-vlm: FAIL; MoE (128 experts) + DeepStack; 100 gen |
| qwen3.5-vl (0.8B) | qwen3.5-0.8B-4bit | ✅ | 946.46 | 233.81 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet VLM; 57 prompt, 53 gen |
| qwen3.5-vl (2B) | qwen3.5-2B-4bit | ✅ | 546.64 | 172.92 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet VLM; 57 prompt, 58 gen |
| qwen3.5-vl (4B) | qwen3.5-4B-4bit | ✅ | 332.57 | 100.46 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet VLM; 57 prompt, 30 gen |
| qwen3.5-vl (9B 4bit) | qwen3.5-9B-4bit | ✅ | 196.03 | 74.05 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet VLM; 57 prompt, 62 gen |
| qwen3.5-vl (9B bf16) | qwen3.5-9B (bf16) | ✅ | 279.39 | 32.74 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet VLM; 57 prompt, 78 gen; bf16 |
| qwen3.5-vl (27B) | qwen3.5-27B-4bit | ✅ | 74.86 | 25.24 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet VLM; 57 prompt, 42 gen |
| qwen3.5-vl-moe | qwen3.5-35B-A3B-4bit | ✅ | 274.14 | 70.22 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet + MoE VLM; 57 prompt, 47 gen; gated delta decode RMSNorm fix |
| qwen3.6-vl-moe | qwen3.6-35B-A3B-4bit | ✅ | 274.13 | 66.80 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet + MoE VLM; 100 gen |
| molmo-point | - | ⏳ | - | - | - | Molmo-Point (point detection); implemented but no MLX model available |

**VLM test conditions**: Image: 224x224 PNG (test_image.png) unless noted. Prompt: "What is in this image?" Max tokens: 100. Prefill includes vision encoder + projector overhead. mlx-vlm baseline uses the `d85ca4d` dev checkout. mlxcel decode speed was measured with `mlxcel-bench-decode` (model load, warmup, and measured pass in one process). Models with unavailable or failed mlx-vlm runs are marked with "-" in the vs mlx-vlm column. Three text-only models (`deepseek-v3-4bit` 99GB, `minimax-m2-3bit` 93GB, `qwen3-next-480b-4bit` 251GB) skipped on this 128GB host per >65GB threshold. Two Gemma 3 VLM rows (gemma-3-4b-it-4bit, gemma3-4b-4bit) were measured with `--warmup-tokens 0` because the prepared 4D attention mask shape is single-use against the first prefill's KV cache offset. The three Gemma 3n VLM rows use the default warmup=20 path.

## Summary Statistics

| Status | Count |
|--------|-------|
| ✅ Pass | 120 (78 text + 42 VLM) |
| ⚠️ Partial | 6 (3 text + 3 VLM) |
| ❌ Fail | 2 (2 text + 0 VLM) |
| ⏳ Pending / Skipped (>65GB) | 16 (12 text pending + 1 VLM pending + 3 oversize skip) |

## Performance Comparison

The detailed same-day decode comparison tables below are the authoritative
source for baseline comparisons. Decode remains the primary apples-to-apples
runtime comparison.

### Aggregate (decode, same-day baseline)

| Mode | Comparable pairs | Median mlxcel/baseline | >=90% parity | >= baseline | Range |
|------|-----------------:|-----------------------:|-------------:|------------:|------:|
| Text vs mlx-lm | 74 | 99% | 64/74 (86%) | 35/74 (47%) | 47%-112% |
| VLM vs mlx-vlm | 18 | 98% | 12/18 (67%) | 8/18 (44%) | 78%-133% |

### Representative decode wins

| Model | mlxcel | Baseline | vs baseline |
|-------|-------:|---------:|------------:|
| qwen2.5-0.5b-4bit | 349.52 | 315.48 | **111%** |
| phi-3.5-moe-4bit | 77.71 | 69.28 | **112%** |
| minicpm3-4b-4bit | 80.78 | 73.26 | **110%** |
| smollm-135m-4bit | 383.55 | 375.91 | **102%** |
| llava-interleave-qwen-0.5b-bf16 (VLM) | 265.57 | 225.15 | **118%** |
| gemma3n-e2b-4bit (VLM) | 72.83 | 59.57 | **122%** |
| gemma-4-e2b-it-4bit (VLM) | 106.92 | 97.19 | **110%** |
| phi-3.5-vision-4bit (VLM) | 122.63 | 92.53 | **133%** |

### Main optimization gaps

| Model | mlxcel | Baseline | vs baseline | Notes |
|-------|-------:|---------:|------------:|-------|
| falcon-mamba-7b-4bit | 42.83 | 91.04 | 47% | Chat template causes early EOS; only 2 generated tokens |
| qwen2.5-vl-3b-4bit (text path) | 100.54 | 160.42 | 63% | VLM wrapper text-only comparison |
| qwen2-vl-2b-4bit (text path) | 152.33 | 236.86 | 64% | VLM wrapper text-only comparison |
| gemma-4-31b-4bit (VLM) | 15.82 | 20.30 | 78% | large VLM path |
| gemma-3-4b-it-4bit (VLM) | 87.01 | 97.36 | 89% | measured with warmup=0; see VLM test conditions |

## Performance vs mlx-lm / mlx-vlm baseline (mlxcel 2026-05-28 vs pinned 2026-05-19 reference)

Source CSVs (same M1 Ultra host; mlxcel 0.1.0 measured 2026-05-28 with `--cooldown 0`; mlx-lm / mlx-vlm baselines from the pinned 2026-05-19 reference checkout with `PYLM_BENCH_MAX_GB=65`):

- mlxcel: `benchmarks/metal_m1ultra_2026-05-28.csv`
- mlxcel VLM: `benchmarks/metal_m1ultra_vlm_2026-05-28.csv`
- mlx-lm: `benchmarks/pylm_m1ultra_2026-05-19.csv` (mlx-lm 0.31.3 dev checkout in `references/mlx-lm` @ `df1d3f3`)
- mlx-vlm: `benchmarks/pylm_m1ultra_vlm_2026-05-19.csv` (mlx-vlm dev checkout in `references/mlx-vlm` @ `d85ca4d`)

The mlx-lm / mlx-vlm baselines are the pinned 2026-05-19 reference checkout (the reference is fixed, so its decode on this host is stable); the mlxcel side is the 2026-05-28 full sweep. All sweeps use `--max-tokens 100` and the same `Hello, how are you today?` / `What is in this image?` prompts. `deepseek-v3-4bit` and `qwen3-next-480b-4bit` exceed the 128GB host on both sides; `minimax-m2-3bit` now fits and runs on the mlxcel side (32.62 tok/s) but mlx-lm still fails it, so it stays outside the comparable set.

Numbers are decode tok/s. `mlxcel vs mlx-lm` is `mlxcel / mlx-lm` as a percentage; **bold** = mlxcel >= mlx-lm. `FAIL` cells are real load/runtime errors on that backend with this configuration. The mlx-lm checkout used here (`df1d3f3` — "Fix Gemma 4 sanitize() not stripping KV projections for shared layers" ml-explore/mlx-lm#1240) is newer than the M5 Max page's `ed1fca4`, so some FAIL categories differ.

### Aggregate (text)

- **Comparable text pairs**: 74
- **mlxcel >= mlx-lm**: 35 / 74 (47%)
- **mlxcel >= 90% parity**: 64 / 74 (86%)
- **Average mlxcel/mlx-lm**: 97% (median 99%, range 47%-112%)

### Aggregate (VLM, models with >=5 generated tokens both sides)

- **Comparable VLM pairs**: 18
- **mlxcel >= mlx-vlm**: 8 / 18 (44%)
- **mlxcel >= 90% parity**: 12 / 18 (67%)
- **Average mlxcel/mlx-vlm**: 101% (median 98%, range 78%-133%)

### Text decode (tok/s)

| Model | mlxcel | mlx-lm | mlxcel vs mlx-lm |
|-------|--------|--------|------------------|
| Meta-Llama-3.1-8B-Instruct-4bit | 109.58 | 109.84 | 100% |
| Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | 87.56 | FAIL | - |
| Qwen2.5-1.5B-4bit | 244.93 | 241.41 | **101%** |
| Qwen2.5-1.5B-Instruct-4bit | 243.11 | 239.20 | **102%** |
| Qwen2.5-7B-Instruct-4bit | 112.68 | 110.90 | **102%** |
| Qwen3.5-0.8B-OptiQ-4bit | FAIL | 265.86 | - |
| aya-expanse-8b-4bit | 110.58 | 112.74 | 98% |
| aya-vision-8b | 112.84 | FAIL | - |
| baichuan-m1-14b-4bit | 40.13 | 49.11 | 82% |
| bunny-llama3-8b-4bit | 105.55 | FAIL | - |
| command-r7b-4bit | 114.34 | 107.75 | **106%** |
| deepseek-coder-1.3b-4bit | 165.72 | FAIL | - |
| deepseek-r1-distill-7b-4bit | 113.52 | 111.34 | **102%** |
| deepseek-v2-lite-4bit | 112.45 | 117.06 | 96% |
| deepseek-v3-4bit | - | FAIL | - |
| ernie-4.5-0.3b-4bit | 510.17 | FAIL | - |
| exaone-3.5-2.4b-4bit | 200.53 | 194.65 | **103%** |
| exaone4-1.2b-4bit | 247.55 | FAIL | - |
| falcon-mamba-7b-4bit | 42.83 | 91.04 | 47% |
| gemma-2b-4bit | 194.67 | 207.78 | 94% |
| gemma-3-4b-it-4bit | 117.12 | 109.72 | **107%** |
| gemma-4-26b-a4b-it-4bit | 71.76 | 72.52 | 99% |
| gemma-4-31b-4bit | 20.48 | 20.36 | **101%** |
| gemma-4-31b-it-4bit | 19.61 | 20.23 | 97% |
| gemma-4-e2b-it-4bit | 119.67 | FAIL | - |
| gemma-4-e2b-it-8bit | 89.23 | FAIL | - |
| gemma-4-e4b-it-4bit | 84.43 | FAIL | - |
| gemma-4-e4b-it-8bit | 60.61 | FAIL | - |
| gemma2-2b-4bit | 169.77 | 153.50 | **111%** |
| gemma3-1b-4bit | 232.91 | 211.50 | **110%** |
| gemma3-4b-4bit | 117.66 | 109.48 | **107%** |
| gemma3n-e2b-4bit | 78.75 | FAIL | - |
| gemma3n-e4b-4bit | 61.51 | FAIL | - |
| gemma3n-e4b-bf16 | 34.96 | 39.02 | 90% |
| glm4-flash-4bit | 47.92 | 49.47 | 97% |
| gpt-oss-120b-4bit | 61.19 | 57.58 | **106%** |
| gpt-oss-20b-mxfp4 | 93.46 | 89.51 | **104%** |
| hunyuan-1.8b-4bit | 188.41 | 200.59 | 94% |
| hunyuan-large-4bit | 45.22 | FAIL | - |
| internlm2-7b-4bit | 110.98 | 111.92 | 99% |
| internlm3-8b-4bit | 87.98 | FAIL | - |
| jamba-v0.1-4bit | 123.65 | 131.04 | 94% |
| llama-3.1-8b-4bit | 109.49 | 110.66 | 99% |
| llama-3.1-8b-bf16 | 35.77 | 35.32 | **101%** |
| llama-3.2-1b-4bit | 373.43 | 418.25 | 89% |
| llama-4-scout-17b-4bit | 36.66 | FAIL | - |
| llava-1.5-7b-4bit | 117.93 | FAIL | - |
| llava-interleave-qwen-0.5b-bf16 | 320.61 | FAIL | - |
| llava-next-mistral-7b-4bit | 116.03 | FAIL | - |
| mamba2-1.3b-4bit | 101.13 | FAIL | - |
| mimo-7b-4bit | 86.26 | 86.17 | **100%** |
| minicpm-2b-4bit | 165.32 | 156.47 | **106%** |
| minicpm3-4b-4bit | 80.78 | 73.26 | **110%** |
| minimax-m2-3bit | 32.62 | FAIL | - |
| ministral-3b-4bit | 144.52 | 159.34 | 91% |
| mistral-small-3.1-24b-4bit | 32.07 | 31.97 | **100%** |
| mixtral-8x7b-4bit | 54.66 | 54.91 | 100% |
| molmo2-4b | 60.26 | FAIL | - |
| nemotron-h-30b-4bit | 91.68 | 93.34 | 98% |
| nemotron-nas-30b-4bit | 92.26 | 92.93 | 99% |
| olmo-1b-4bit | 212.53 | FAIL | - |
| olmo2-7b-4bit | 104.17 | 110.88 | 94% |
| olmo3-32b-4bit | 22.14 | 21.57 | **103%** |
| paligemma2-3b-6bit | 0.00 | FAIL | - |
| phi-2-4bit | 62.61 | FAIL | - |
| phi-3-mini-4bit | 172.17 | 171.36 | **100%** |
| phi-3.5-mini-4bit | 167.07 | 166.30 | **100%** |
| phi-3.5-moe-4bit | 77.71 | 69.28 | **112%** |
| phi-3.5-vision-4bit | 166.72 | FAIL | - |
| phi-4-4bit | 58.83 | 58.68 | **100%** |
| pixtral-12b-4bit | 70.97 | 69.49 | **102%** |
| qwen1.5-moe-a2.7b-4bit | 147.24 | 144.98 | **102%** |
| qwen2-vl-2b-4bit | 152.33 | 236.86 | 64% |
| qwen2.5-0.5b-4bit | 349.52 | 315.48 | **111%** |
| qwen2.5-7b-4bit | 113.30 | 111.38 | **102%** |
| qwen2.5-7b-8bit | 71.15 | 70.46 | **101%** |
| qwen2.5-vl-3b-4bit | 100.54 | 160.42 | 63% |
| qwen3-0.6b-4bit | 284.03 | 299.61 | 95% |
| qwen3-1.7b-4bit | 197.55 | 221.37 | 89% |
| qwen3-30b-a3b-4bit | 70.60 | 70.18 | **101%** |
| qwen3-4b-4bit | 121.42 | 123.92 | 98% |
| qwen3-8b-4bit | 81.38 | 84.54 | 96% |
| qwen3-moe-4bit | 72.17 | 69.67 | **104%** |
| qwen3-next-480b-4bit | - | FAIL | - |
| qwen3-vl-2b-4bit | 214.75 | 222.67 | 96% |
| qwen3-vl-30b-a3b-4bit | 70.74 | 70.04 | **101%** |
| qwen3-vl-32b-4bit | 21.25 | 21.99 | 97% |
| qwen3-vl-4b-4bit | 119.23 | 124.02 | 96% |
| qwen3-vl-8b-4bit | 81.34 | 84.46 | 96% |
| qwen3.5-0.8b-4bit | 244.18 | 269.52 | 91% |
| qwen3.5-27b-4bit | 24.52 | 25.93 | 95% |
| qwen3.5-2b-4bit | 179.54 | 211.68 | 85% |
| qwen3.5-35b-a3b-4bit | 71.38 | 76.44 | 93% |
| qwen3.5-4b-4bit | 102.31 | 115.60 | 89% |
| qwen3.5-9b-4bit | 72.75 | 81.27 | 90% |
| qwen3.5-9b-bf16 | 31.85 | 34.22 | 93% |
| qwen3.6-35b-a3b-4bit | 70.36 | 73.18 | 96% |
| smollm-135m-4bit | 383.55 | 375.91 | **102%** |
| smollm3-3b-4bit | 137.92 | 141.66 | 97% |
| solar-open-100b-4bit | 36.26 | 35.69 | **102%** |
| stablelm-1.6b-4bit | 285.79 | 280.65 | **102%** |
| starcoder2-3b-4bit | 172.82 | 166.17 | **104%** |
| youtu-vl-4b-instruct | 0.00 | FAIL | - |

### VLM decode (tok/s)

| Model | mlxcel | mlx-vlm | mlxcel vs mlx-vlm |
|-------|--------|--------|------------------|
| Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | 69.37 | FAIL | - |
| aya-vision-8b | 113.59 | 103.74 | **109%** |
| bunny-llama3-8b-4bit | 95.36 | FAIL | - |
| deepseek-v3-4bit | - | FAIL | - |
| gemma-3-4b-it-4bit | 87.01 | 97.36 | 89% |
| gemma-4-26b-a4b-it-4bit | 66.40 | 61.07 | **109%** |
| gemma-4-31b-4bit | 15.82 | 20.30 | 78% |
| gemma-4-31b-it-4bit | 18.86 | 19.78 | 95% |
| gemma-4-e2b-it-4bit | 106.92 | 97.19 | **110%** |
| gemma-4-e2b-it-8bit | 81.98 | 91.06 | 90% |
| gemma-4-e4b-it-4bit | 75.15 | 70.34 | **107%** |
| gemma-4-e4b-it-8bit | 56.16 | 63.25 | 89% |
| gemma3-4b-4bit | 83.74 | 93.79 | 89% |
| gemma3n-e2b-4bit | 72.83 | 59.57 | **122%** |
| gemma3n-e4b-4bit | 57.50 | 50.00 | **115%** |
| gemma3n-e4b-bf16 | 32.46 | 36.18 | 90% |
| internvl3-1b | 228.83 | 264.40 | 87% |
| llama-4-scout-17b-4bit | 35.88 | FAIL | - |
| llava-1.5-7b-4bit | 104.03 | FAIL | - |
| llava-interleave-qwen-0.5b-bf16 | 265.57 | 225.15 | **118%** |
| llava-next-mistral-7b-4bit | 107.17 | 109.51 | 98% |
| minimax-m2-3bit | - | FAIL | - |
| ministral-3b-4bit | 125.75 | FAIL | - |
| mistral-small-3.1-24b-4bit | 29.72 | FAIL | - |
| molmo-7b | 81.61 | 38399.52 (anomalous) | - |
| molmo2-4b | 60.31 | 60.87 | 99% |
| paligemma2-3b-6bit | 50.25 | 70.45 | 71% |
| phi-3.5-vision-4bit | 122.63 | 92.53 | **133%** |
| pixtral-12b-4bit | 60.25 | FAIL | - |
| qwen2-vl-2b-4bit | 124.99 | FAIL | - |
| qwen2.5-vl-3b-4bit | 97.81 | FAIL | - |
| qwen3-next-480b-4bit | - | FAIL | - |
| qwen3-vl-2b-4bit | 162.21 | FAIL | - |
| qwen3-vl-30b-a3b-4bit | 40.88 | FAIL | - |
| qwen3-vl-32b-4bit | 17.92 | FAIL | - |
| qwen3-vl-4b-4bit | 90.14 | FAIL | - |
| qwen3-vl-8b-4bit | 62.62 | FAIL | - |
| qwen3.5-0.8b-4bit | 233.81 | FAIL | - |
| qwen3.5-27b-4bit | 25.24 | FAIL | - |
| qwen3.5-2b-4bit | 172.92 | FAIL | - |
| qwen3.5-35b-a3b-4bit | 70.22 | FAIL | - |
| qwen3.5-4b-4bit | 100.46 | FAIL | - |
| qwen3.5-9b-4bit | 74.05 | FAIL | - |
| qwen3.5-9b-bf16 | 32.74 | FAIL | - |
| qwen3.6-35b-a3b-4bit | 66.80 | FAIL | - |
| youtu-vl-4b-instruct | 24.47 | FAIL | - |

## Comprehensive Validation (2026-03-10)

Ran all 80 local models (45GB threshold) to verify text/image generation.

| Metric | Count |
|--------|-------|
| Total models | 80 |
| Tested | 74 |
| Pass | 71 (95.9%) |
| Fail | 3 |
| Skip (>45GB) | 6 |

**Failures:**
- `internvl3-1b`: Unsupported architecture (`internvl_chat`)
- `molmo-7b`: Unsupported architecture (`molmo`; only `molmo2` supported)
- `hunyuan-13b`: Fixed: added tiktoken tokenizer support

**Skipped (>45GB):** deepseek-v3, qwen3-next, minimax-m2, solar-open-100b-4bit (tested separately), solar-open-100b-int4

## Known Issues

| Model | Issue | Priority |
|-------|-------|----------|
| hunyuan-moe-a13b-bf16 | Warmup failure; Tiktoken tokenizer; bf16 | High |
| qwen2.5-0.5b-bf16 | Warmup failure; bf16 non-quantized | Medium |
| Qwen3.5-4B-DFlash / Qwen3.5-27B-DFlash | Drafter checkpoint — not a standalone inference model | Low |
| Qwen3.5-0.8B-OptiQ-4bit | Warmup failure on new OptiQ quant variant | Medium |
| gemma-4-31B-it-assistant-bf16 | Drafter checkpoint — not a standalone inference model | Low |
| falcon-mamba | Chat template causes early EOS (only 2 tokens); decode now 42.83 tok/s | Medium |
| paligemma | Only 2 VLM gen tokens; decode is not comparable despite 50.25 tok/s measured | High |
| youtu-vl-4b-instruct | NEW (5-19); VLM produces only 1 token; text-only produces 0 tokens | Medium |
| llama4 | Repetitive output on long generations | Low |
| moondream3 | Image output garbled; text-only works; needs reconstruct_from_crops | Medium |
| qwen-vl family | Broadcast shape errors on M5 Max with 224x224 images | Medium |
| GLM-5-4bit | Persistent warmup failure (since 5-08) | Medium |

## Notes

- All tests use 4-bit quantized models unless noted (Nemotron uses 8-bit)
- Performance measured with `mlxcel-bench-decode` (model load, warmup, and measured pass in one process)
- vs mlx-lm percentage is based on **decode speed only**
- Prefill tok/s and decode tok/s are reported separately; units omitted from table headers for brevity
- Prefill shown as "-" for models not measured in this run

## Tokenizer Support

| Format | File | Models | Crate |
|--------|------|--------|-------|
| HuggingFace | `tokenizer.json` | Most models | `tokenizers` |
| SentencePiece | `tokenizer.model` | Gemma, Llama 1/2, older models | `sentencepiece` |
| Tiktoken | `*.tiktoken` | HunYuan MoE (13B) | Custom BPE (fancy-regex) |

## TurboQuant KV cache — M1 Ultra speed gate readings

First dedicated M1 Ultra reading of the epic- KV speed gate matrix.
Hardware: Apple M1 Ultra, 128 GB unified memory. Model:
`Meta-Llama-3.1-8B-Instruct-4bit`. Date: 2026-04-29. Reproducer:
`./scripts/bench_kv_cache.sh --modes fp16,int8,turbo4-asym,turbo4,turbo4-delegated --contexts 4096 --prefill-contexts 8192`.

Full CSV at `benchmarks/turbo_kv/2026-04-29_Apple_M1_Ultra_Meta-Llama-3.1-8B-Instruct-4bit.csv`.

> **CSV schema note:** Rows where `stage=prefill` record a single-token follow-up to force the KV
> cache to be populated. The resulting `decode_tok_s` value (e.g. 303766 tok/s) reflects a
> sub-millisecond single-token step and is not a meaningful decode throughput figure; ignore it
> for prefill rows. Use `prefill_tok_s` from those rows and `decode_tok_s` from `stage=decode` rows.

### Decode @ 4K context (80 generated tokens)

| Mode | Decode tok/s | × FP16 | M5 Max gate (tracking on M1U) |
|------|--------------|--------|------|
| `fp16`             | 90.36 | 1.000× | baseline |
| `int8`             | 61.24 | 0.678× | (no gate; tracking only) |
| `turbo4-asym`      |  3.92 | 0.043× | ≥0.97× → off-target on M1U |
| `turbo4`           | 16.34 | 0.181× | ≥0.93× → off-target on M1U |
| `turbo4-delegated` | 18.22 | 0.202× | ≥0.97× → off-target on M1U |

`turbo4-asym` produced only 51 tokens before early EOS (vs 80 requested);
the per-token decode rate is computed over those 51 tokens. The other Turbo
modes ran the full 80 tokens.

### Prefill @ 8K context (single-token decode follow-up)

| Mode | Prefill tok/s | × FP16 | M5 Max gate (tracking on M1U) |
|------|---------------|--------|------|
| `fp16`             | 678.28 | 1.000× | baseline |
| `int8`             | 676.82 | 0.998× | (no gate) |
| `turbo4-asym`      | 471.12 | 0.694× | ≥1.00× → off-target on M1U |
| `turbo4`           | 365.07 | 0.538× | ≥1.00× → off-target on M1U |
| `turbo4-delegated` | 678.96 | 1.001× | best-effort → meets target |

### M1 Ultra reading

 §"Cross-hardware regression", the M5 Max gates are tracking
only on pre-M3 hardware. The numbers above match that guidance: `turbo4`
and `turbo4-asym` decode are well below the M5 gates on M1 Ultra, while
`turbo4-delegated` prefill is bit-identical to FP16 because the cold pages
keep their FP16 representation and only the hot tail is packed.

The decode regression is consistent with the L2-cache wall documented in
`references/turboquant_plus/`. The fused Sparse-V Metal kernel that lands
 targets the per-thread skip path inside the SDPA inner loop and is
expected to recover most of the M5 decode budget; the M1/M2 ceiling stays
limited by L2 bandwidth and is documented but not gated.

### M1 Ultra hardware considerations

- **`turbo4-delegated` is the only currently shipping mode that holds the
  prefill gate on M1 Ultra.** Use it when prefill latency matters more than
  the maximum compression ratio.
- **`turbo4-asym` on M1 Ultra needs the fused kernel** to be a viable
  decode option. The graph-level path measured here pays a per-token
  dequant cost that the fused kernel folds into the SDPA inner loop.
- **Avoid `turbo3` on M1 Ultra** for decode-bound workloads. The 3-bit
  pack/unpack loop saturates L2 bandwidth on the older GPU microarchitecture;
  on M3/M4/M5 the regression is smaller or absent.
- Memory-vs-speed trade-off on M1 Ultra: `fp16+turbo4` (alias `turbo4-asym`)
  still delivers approximately 0.39× of the FP16 baseline KV footprint at the
  default boundary-v 2 on a 32-layer model, but the decode shortfall above
  makes `turbo4-delegated` the better practical choice on this generation
  until the fused kernel lands. Pick `fp16+turbo4` only if the memory
  savings outweigh the per-token decode cost for your workload.

### Deferred

- 32K decode reading (best-effort per epic; benefits fused kernel before the gate is meaningful on any hardware).
- Per-mode 16K decode (skipped for the initial run; M5 Max is the primary
  target for the 16K gate).
- Multi-model expansion (Qwen 2.5, Gemma 3, etc.).
- M5 Max readings of the same matrix — to be filled in by a manual run on
  the M5 Max dev box; the script is hardware-agnostic and writes
  `benchmarks/turbo_kv/<date>_<hw>_<model>.csv` keyed off
  `sysctl -n machdep.cpu.brand_string`.
