# Model Compatibility & Performance Tests (M1 Ultra)

Compatibility and performance testing for mlxcel models on **Mac Studio M1 Ultra 128GB**, with comparison against Python mlx-lm / mlx-vlm.

## Test Environment

| Item | Value |
|------|-------|
| **Hardware** | Mac Studio M1 Ultra, 128GB RAM |
| **OS** | macOS 26.4 (Tahoe) |
| **mlxcel version** | 0.0.28 |
| **MLX version** | post-0.32.0 pin (commit 84961223, via mlxcel-core) |
| **Bench harness** | `mlxcel-bench-decode` (model load, warmup, and measured pass in one process) |
| **mlx-lm baseline** | 0.31.3 (dev checkout `references/mlx-lm` @ `df1d3f3` — "Fix Gemma 4 sanitize() not stripping KV projections for shared layers" #1240) |
| **mlx-vlm baseline** | dev checkout `references/mlx-vlm` @ `d85ca4d` — "Compatibility bridge for non-VL models" #1181 |
| **Test Prompt** | "Hello, how are you today?" (text) / "What is in this image?" (VLM) |
| **Max Tokens** | 100 (measured pass); 20 (warmup pass, same process) |
| **Test Date** | 2026-05-19 full sweep (mlxcel + mlx-lm + mlx-vlm baselines); 2026-05-21 mlxcel refresh of Molmo / Phi-3.5 / Gemma dense / Jamba / InternVL on current main |
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
| llama3 | Llama-3.2-1B-Instruct-4bit | ✅ | 1988.07 | 377.90 | 90% | mlx-lm: 418.25; only 48 tokens |
| llama3 (8B bf16) | Llama-3.1-8B-Instruct (bf16) | ✅ | 419.02 | 35.15 | 100% | mlx-lm: 35.32; non-quantized |
| llama3.1 | Llama-3.1-8B-Instruct-4bit | ✅ | 471.57 | 107.23 | 97% | mlx-lm: 110.66; only 54 tokens |
| llama4 | Llama-4-Scout-17B-16E-Instruct-4bit | ⚠️ | 125.84 | 35.21 | - | mlx-lm: FAIL; long outputs repetitive |
| qwen2 | Qwen2.5-0.5B-Instruct-4bit | ✅ | 1094.10 | 355.29 | **113%** | mlx-lm: 315.48 |
| qwen2 (7B 4bit) | Qwen2.5-7B-Instruct-4bit | ✅ | 310.40 | 110.67 | 100% | mlx-lm: 110.90 |
| qwen2 (7B 8bit) | Qwen2.5-7B-Instruct-8bit | ✅ | 315.31 | 69.08 | 98% | mlx-lm: 70.46; 8-bit quantized |
| qwen3 | Qwen3-0.6B-4bit | ✅ | 559.24 | 295.09 | 98% | mlx-lm: 299.61 |
| qwen3 (1.7B) | Qwen3-1.7B-4bit | ✅ | 390.56 | 195.30 | 88% | mlx-lm: 221.37 |
| qwen3 (4B) | Qwen3-4B-4bit | ✅ | 240.16 | 120.70 | 97% | mlx-lm: 123.92 |
| qwen3 (8B) | Qwen3-8B-4bit | ✅ | 165.63 | 79.85 | 94% | mlx-lm: 84.54 |
| qwen3_5 (0.8B) | Qwen3.5-0.8B-4bit | ✅ | 478.37 | 243.32 | 90% | mlx-lm: 269.52; Hybrid GatedDeltaNet |
| qwen3_5 (2B) | Qwen3.5-2B-4bit | ✅ | 387.77 | 172.05 | 81% | mlx-lm: 211.68; Hybrid GatedDeltaNet; only 36 tokens |
| qwen3_5 (4B) | Qwen3.5-4B-4bit | ✅ | 242.19 | 95.76 | 83% | mlx-lm: 115.60; Hybrid GatedDeltaNet; only 36 tokens |
| qwen3_5 (9B 4bit) | qwen3.5-9B-4bit | ✅ | 151.81 | 72.36 | 89% | mlx-lm: 81.27; Hybrid GatedDeltaNet; only 29 tokens |
| qwen3_5 (9B bf16) | qwen3.5-9B (bf16) | ✅ | 150.65 | 31.28 | 91% | mlx-lm: 34.22; bf16, not quantized; Hybrid GatedDeltaNet (compiled fused kernel) |
| qwen3_5 (27B) | qwen3.5-27B-4bit | ✅ | 50.99 | 24.25 | 94% | mlx-lm: 25.93; Hybrid Transformer+GatedDeltaNet; VLM wrapper format |
| qwen3_6 | qwen3.6-35B-A3B-4bit | ✅ | 223.86 | 68.77 | 94% | mlx-lm: 73.18; MoE architecture; 100 tokens |
| qwen3_next | qwen3-next-480B-4bit | ⏳ | - | SKIP | - | Qwen3Next 480B architecture; >65GB skipped on 128GB host |
| qwen2 (1.5B) | Qwen2.5-1.5B-Instruct-4bit | ✅ | 808.08 | 240.24 | **100%** | mlx-lm: 239.20; 100 tokens |
| qwen2 (1.5B base) | Qwen2.5-1.5B-4bit | ✅ | 701.57 | 241.44 | **100%** | mlx-lm: 241.41; base variant; 100 tokens |
| phi | phi-2-hf-4bit-mlx | ✅ | 134.96 | 59.62 | - | mlx-lm fails to load; only 1 token (likely EOS) |
| phi3 | Phi-3-mini-4k-instruct-4bit | ✅ | 173.84 | 167.27 | 98% | mlx-lm: 171.36; only 25 tokens |
| phi3small | Phi-3.5-mini-instruct-4bit | ✅ | 218.87 | 160.90 | 97% | mlx-lm: 166.30; fused SuScaledRoPE; only 40 tokens |
| phi4 | Phi-4-4bit | ✅ | 111.29 | 57.02 | 97% | mlx-lm: 58.68 |
| smollm3 | SmolLM-135M-Instruct-4bit | ✅ | 486.25 | 407.36 | **108%** | mlx-lm: 375.91 |
| smollm3 (3B) | SmolLM3-3B-4bit | ✅ | 568.57 | 136.34 | 96% | mlx-lm: 141.66 |
| stablelm | stablelm-2-1_6b-chat-4bit | ✅ | 656.55 | 280.32 | 100% | mlx-lm: 280.65; only 59 tokens |
| starcoder2 | starcoder2-3b-4bit | ✅ | 177.66 | 171.30 | **103%** | mlx-lm: 166.17; major decode improvement |
| olmo | OLMo-1B-hf-4bit | ✅ | 179.80 | 219.54 | - | mlx-lm: FAIL |
| olmo2 | OLMo2-7B-4bit | ✅ | 281.23 | 103.66 | 93% | mlx-lm: 110.88; only 27 tokens |
| olmo3 | OLMo3.1-32B-4bit | ✅ | 81.40 | 21.79 | **101%** | mlx-lm: 21.57 |
| minicpm | MiniCPM-2B-sft-bf16-4bit | ✅ | 298.42 | 163.05 | **104%** | mlx-lm: 156.47 |
| mimo | MiMo-7B-RL-4bit | ✅ | 240.27 | 85.28 | 99% | mlx-lm: 86.17 |

## Gemma Family

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| gemma | gemma-2b-it-4bit | ✅ | 378.01 | 189.91 | 91% | mlx-lm: 207.78; major decode improvement from 81.76; only 49 tokens |
| gemma2 | gemma-2-2b-it-4bit | ✅ | 334.20 | 163.85 | **107%** | mlx-lm: 153.50; tanh-approx GeGLU aligned with mlx-lm; only 18 tokens |
| gemma3 | gemma-3-1b-it-4bit | ✅ | 431.88 | 225.75 | **107%** | mlx-lm: 211.50; only 34 tokens |
| gemma3 (4B) | gemma-3-4b-it-4bit | ✅ | 203.23 | 113.61 | **104%** | mlx-lm: 109.48; tanh-approx GeGLU + fused norm-RoPE; only 86 tokens |
| gemma4 (31B) | gemma-4-31b-4bit | ✅ | 24.54 | 20.15 | 99% | mlx-lm: 20.36 |
| gemma4 (31B-it) | gemma-4-31b-it-4bit | ✅ | 52.04 | 19.06 | 94% | mlx-lm: 20.23; instruction-tuned variant |
| gemma4 (26B A4B) | gemma-4-26b-a4b-it-4bit | ✅ | 168.66 | 73.18 | **101%** | mlx-lm: 72.52; fixed SDPA threadgroup memory; only 26 tokens |
| gemma4 (E2B 4bit) | gemma-4-e2b-it-4bit | ✅ | 257.28 | 116.08 | - | mlx-lm: FAIL; fixed SDPA threadgroup memory; only 34 tokens |
| gemma4 (E2B 8bit) | gemma-4-e2b-it-8bit | ✅ | 212.56 | 87.42 | - | mlx-lm: FAIL; fixed SDPA threadgroup memory; only 38 tokens |
| gemma4 (E4B 4bit) | gemma-4-e4b-it-4bit | ✅ | 175.01 | 81.88 | - | mlx-lm: FAIL; fixed SDPA threadgroup memory; only 25 tokens |
| gemma4 (E4B 8bit) | gemma-4-e4b-it-8bit | ✅ | 165.49 | 59.35 | - | mlx-lm: FAIL; fixed SDPA threadgroup memory; only 39 tokens |
| gemma3n | gemma-3n-E2B-it-4bit | ✅ | 238.51 | 76.86 | - | mlx-lm: FAIL; fixed SDPA threadgroup memory; only 69 tokens |
| gemma3n (E4B) | gemma-3n-E4B-it-4bit | ✅ | 169.94 | 60.18 | - | mlx-lm: FAIL; fixed SDPA threadgroup memory; only 74 tokens |
| gemma3n (E4B bf16) | gemma-3n-E4B-it (bf16) | ✅ | 169.01 | 34.41 | 88% | mlx-lm: 39.02; bf16 prefill path retune (PR #727); bf16; only 72 tokens |
| recurrent_gemma | - | ⏳ | - | - | - | Griffin SSM+attention hybrid |

## EXAONE

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| exaone | EXAONE-3.5-2.4B-Instruct-4bit | ✅ | 672.62 | 199.04 | **102%** | mlx-lm: 194.65 |
| exaone4 | exaone-4.0-1.2b-4bit | ✅ | 464.33 | 252.36 | - | mlx-lm: FAIL; only 18 tokens |
| exaone_moe | - | ⏳ | - | - | - | |

## Cohere Command R

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| cohere | c4ai-command-r7b-12-2024-4bit | ✅ | 92.20 | 110.22 | **102%** | mlx-lm: 107.75 |
| cohere2 | aya-expanse-8b-4bit | ✅ | 98.87 | 107.30 | 95% | mlx-lm: 112.74 |

## MoE (Mixture of Experts)

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| minimax | MiniMax-M2-3bit | ⏳ | - | SKIP | - | mlx-lm: 18.2; >65GB skipped on 128GB host (93GB) |
| mixtral | Mixtral-8x7B-Instruct-v0.1-4bit | ✅ | 78.43 | 53.62 | 98% | mlx-lm: 54.91; only 73 tokens |
| qwen2_moe | Qwen1.5-MoE-A2.7B-Chat-4bit | ✅ | 378.81 | 144.37 | 100% | mlx-lm: 144.98; only 43 tokens |
| qwen3_moe | Qwen3-30B-A3B-4bit | ✅ | 183.00 | 71.07 | **101%** | mlx-lm: 70.18 |
| qwen3_5_moe | qwen3.5-35B-A3B-4bit | ✅ | 216.08 | 72.05 | 94% | mlx-lm: 76.44; Hybrid GatedDeltaNet + MoE (256 experts); only 34 tokens |
| phimoe | Phi-3.5-MoE-instruct-4bit | ✅ | 105.77 | 74.41 | **107%** | mlx-lm: 69.28 |
| solar_open | Solar-Open-100B-4bit | ✅ | 73.74 | 35.88 | **101%** | mlx-lm: 35.69; 128 experts, top-8; layer-eval skip (PR #724); 54GB |
| solar_open (int4) | Solar-Open-100B-int4 | ✅ | - | 11.55 | - | mlx-lm: fails to load; 128 experts, top-8; int4 quantization; 54GB |
| olmoe | - | ⏳ | - | - | - | |
| gpt_oss (20B) | gpt-oss-20b-MXFP4-Q4 | ✅ | 284.10 | 88.89 | 99% | mlx-lm: 89.51; MXFP4 quantization; 32 experts; bf16 decode fix (PR #721) |
| gpt_oss (120B) | gpt-oss-120b-4bit | ✅ | 161.69 | 58.89 | **102%** | mlx-lm: 57.58; 128 experts, top-4; 61GB model; bf16 decode fix (PR #721) |

## DeepSeek Family

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| deepseek | deepseek-coder-1.3b-instruct-4bit | ✅ | 1354.70 | 164.77 | - | mlx-lm: FAIL |
| deepseek_v2 | DeepSeek-V2-Lite-Chat-4bit | ✅ | 217.30 | 111.86 | 96% | mlx-lm: 117.06; only 18 tokens |
| deepseek_r1 | DeepSeek-R1-Distill-Qwen-7B-4bit | ✅ | 158.52 | 110.82 | 100% | mlx-lm: 111.34 |
| deepseek_v3 | deepseek-v3-4bit | ⏳ | - | SKIP | - | MoE + MLA; >65GB skipped on 128GB host (99GB) |
| deepseek_v32 | - | ⏳ | - | - | - | |

## MLA (Multi-head Latent Attention)

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| minicpm3 | MiniCPM3-4B-4bit | ✅ | 230.24 | 80.24 | **110%** | mlx-lm: 73.26 |

## Nemotron Family

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| nemotron_h | Nemotron-H-30B-4bit | ✅ | 173.69 | 90.25 | 97% | mlx-lm: 93.34; Hybrid Mamba2+Transformer+MoE; SSM Metal kernel |
| nemotron_nas | Nemotron-NAS-30B-A3B-4bit | ✅ | 165.59 | 90.67 | 98% | mlx-lm: 92.93; Hybrid Mamba2+Transformer+MoE |
| nemotron_h_nano_omni | Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | ✅ | 165.84 | 83.29 | - | mlx-lm: FAIL; NEW (5-19); Mamba2+Transformer+MoE+Parakeet audio; 100 tokens |

## SSM / Mamba Models

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| mamba | Falcon-Mamba-7B-4bit | ⚠️ | 92.67 | 42.91 | 47% | mlx-lm: 91.04; only 2 tokens due to chat template EOS |
| mamba2 | mamba2-1.3b-4bit | ✅ | 162.53 | 102.63 | - | mlx-lm: FAIL |
| jamba | Jamba-v0.1-4bit | ✅ | 325.65 | 119.33 | 91% | mlx-lm: 131.04; fused QKV + single-token Mamba fast path; only 76 tokens |
| rwkv7 | - | ⏳ | - | - | - | RWKV v7 linear attention |

## Chinese / Asian Language Models

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| baichuan | Baichuan-M1-14B-Instruct-4bit | ✅ | 57.64 | 40.32 | 82% | mlx-lm: 49.11; only 39 tokens |
| glm4 | GLM-4-Flash-4bit | ✅ | 130.01 | 47.32 | 96% | mlx-lm: 49.47; Only 18 tokens |
| glm4_moe | - | ⏳ | - | - | - | |
| glm4_moe_lite | GLM-4.7-Flash-4bit | ✅ | - | 31.54 | 76% | mlx-lm: 41.55; only 18 tokens |
| glm5 | GLM-5-4bit | ❌ | - | FAIL | - | warmup failure (persistent) |
| internlm2 | InternLM2-7B-4bit | ✅ | 211.75 | 107.52 | 96% | mlx-lm: 111.92 |
| internlm3 | internlm3-8b-instruct-4bit | ✅ | 313.73 | 86.88 | - | mlx-lm: FAIL |
| ernie4_5 | ERNIE-4.5-0.3B-Instruct-4bit | ✅ | 996.41 | 526.70 | - | mlx-lm: FAIL |
| ernie4_5_moe | - | ⏳ | - | - | - | |
| hunyuan_moe | Hunyuan-Large-Instruct-4bit | ✅ | 64.69 | 44.22 | - | mlx-lm: FAIL |
| hunyuan_moe_13b | HunYuan-MoE-A13B-Instruct (bf16) | ❌ | - | FAIL | - | mlx-lm: fails to load; Tiktoken tokenizer; bf16; warmup failure |
| hunyuan_v1_dense | Hunyuan-1.8B-Instruct-4bit | ✅ | 280.48 | 182.98 | 91% | mlx-lm: 200.59; only 41 tokens |
| kimi_linear | - | ⏳ | - | - | - | Kimi linear attention (Moonshot) |
| step3p5 | - | ⏳ | - | - | - | Step 3.5 (StepFun) |

## Other Models

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| ministral3 | Ministral-3B-Instruct-4bit | ✅ | 888.69 | 145.06 | 91% | mlx-lm: 159.34; VLM wrapper; text-only mode; only 34 tokens |
| mistral4 | - | ⏳ | - | - | - | MLA + MoE; implemented but no MLX model available |
| moondream3 | moondream3-preview-4bit | ⚠️ | - | 8.45 | - | mlx-lm: fails to load; text-only test; SigLIP + MLP; image output garbled; only 14 tokens |
| longcat_flash | - | ⏳ | - | - | - | |
| longcat_flash_ngram | - | ⏳ | - | - | - | |
| mistral_small | mistral-small-3.1-24b-4bit | ✅ | 34.88 | 31.70 | 99% | mlx-lm: 31.97; text-only mode |

## Vision-Language Models (VLM)

| Model | Test Model | Status | Prefill | Decode | vs mlx-vlm | Notes |
|-------|------------|--------|---------|--------|------------|-------|
| gemma3 | gemma-3-4b-it-4bit | ✅ | 218.56 | 85.61 | 91% | mlx-vlm: 93.79; SigLIP + AvgPool; tanh-approx GeGLU + fused norm-RoPE; 275 prompt, 16 gen |
| gemma3n (E2B) | gemma-3n-E2B-it-4bit | ✅ | 771.46 | 72.38 | **122%** | mlx-vlm: 59.57; MobileNetV5 + MSFA; fixed SDPA threadgroup memory; 273 prompt, 29 gen |
| gemma3n (E4B bf16) | gemma-3n-E4B-it (bf16) | ✅ | 644.18 | 32.12 | 89% | mlx-vlm: 36.18; MobileNetV5 + MSFA; bf16 prefill path retune (PR #727); bf16; 273 prompt, 24 gen |
| gemma3n (E4B 4bit) | gemma-3n-E4B-it-4bit | ✅ | 490.04 | 56.69 | **113%** | mlx-vlm: 50.00; fixed SDPA threadgroup memory; 273 prompt, 33 gen |
| gemma4 (E2B 4bit) | gemma-4-e2b-it-4bit | ✅ | 989.24 | 107.06 | **110%** | mlx-vlm: 97.19; fixed SDPA threadgroup memory; 274 prompt, 100 gen |
| gemma4 (E2B 8bit) | gemma-4-e2b-it-8bit | ✅ | 969.00 | 80.48 | 88% | mlx-vlm: 91.06; fixed SDPA threadgroup memory; 274 prompt, 100 gen |
| gemma4 (E4B 4bit) | gemma-4-e4b-it-4bit | ✅ | 561.68 | 72.91 | **104%** | mlx-vlm: 70.34; fixed SDPA threadgroup memory; 274 prompt, 54 gen |
| gemma4 (E4B 8bit) | gemma-4-e4b-it-8bit | ✅ | 534.96 | 54.83 | 87% | mlx-vlm: 63.25; fixed SDPA threadgroup memory; 274 prompt, 35 gen |
| gemma4 (31B 4bit) | gemma-4-31b-4bit | ✅ | 95.02 | 15.46 | 76% | mlx-vlm: 20.30; 274 prompt, 100 gen |
| gemma4 (31B-it 4bit) | gemma-4-31b-it-4bit | ✅ | 99.07 | 18.32 | 93% | mlx-vlm: 19.78; 274 prompt, 100 gen |
| gemma4 (26B A4B) | gemma-4-26b-a4b-it-4bit | ✅ | 476.14 | 63.18 | **103%** | mlx-vlm: 61.07; 277 prompt, 28 gen |
| llava 1.5 | llava-1.5-7b-4bit | ✅ | 804.92 | 101.61 | - | CLIP + MLP; Vicuna-7b; 583 prompt, 100 gen; mlx-vlm requires PyTorch |
| llava-interleave | llava-interleave-qwen-0.5b-bf16 | ✅ | 8589.48 | 269.86 | **120%** | mlx-vlm: 225.15; SigLIP + MLP; Qwen2-0.5b; 754 prompt, 36 gen |
| llava-next | llava-v1.6-mistral-7b-4bit | ✅ | 748.12 | 105.41 | 96% | mlx-vlm: 109.51; CLIP + MLP; Mistral; 590 prompt, 100 gen; mlx-vlm template error |
| llava-bunny | Bunny-Llama-3-8B-V-4bit | ✅ | 724.90 | 96.07 | - | mlx-vlm: FAIL; SigLIP + MLP; Llama3; 746 prompt, 37 gen |
| llama4 | Llama-4-Scout-17B-16E-Instruct-4bit | ✅ | 194.61 | 31.73 | - | mlx-vlm: FAIL; 162 prompt, 100 gen |
| aya-vision | aya-vision-8b | ✅ | 591.80 | 109.36 | **105%** | mlx-vlm: 103.74; SigLIP + SwiGLU; Cohere2; 176 prompt, 100 gen |
| paligemma | paligemma2-3b (6-bit) | ⚠️ | 1571.48 | 45.09 | 64% | mlx-vlm: 70.45; SigLIP + Linear; Gemma2; 1032 prompt, only 2 gen tokens |
| pixtral | pixtral-12b-4bit | ✅ | 473.22 | 59.17 | - | mlx-vlm: FAIL; Pixtral ViT; Mistral; 4102 prompt, 100 gen |
| mistral3 | mistral-small-3.1-24b-4bit | ✅ | 144.54 | 29.69 | - | mlx-vlm: FAIL; Pixtral ViT + PatchMerger; Mistral; 3032 prompt, 100 gen; mlx-vlm error |
| ministral3 | Ministral-3B-Instruct-4bit | ✅ | 891.57 | 123.91 | - | mlx-vlm: FAIL; Pixtral ViT; 3566 prompt, 100 gen |
| phi3.5-vision | Phi-3.5-vision-instruct-4bit | ✅ | 960.85 | 117.63 | **127%** | mlx-vlm: 92.53; CLIP + HD tiling; Phi3; fused SuScaledRoPE; 773 prompt, 19 gen |
| phi4mm | phi-4-multimodal-instruct (bf16) | ✅ | 571.90 | 25.42 | - | SigLIP + HD transform + AvgPool2d; Phi3; SuScaledRoPE + runtime LoRA; 2635 tokens; 12GB bf16 |
| moondream3 | moondream3-preview-4bit | ⚠️ | 1.36 | 10.05 | - | SigLIP + MLP; image output garbled; only 63 tokens |
| minicpm-o | MiniCPM-o-2_6-4bit | ✅ | 33.67 | 70.80 | - | SigLIP + Resampler; Qwen3; 80 tokens |
| molmo | Molmo-7B | ✅ | 555.19 | 77.98 | - | CLIP ViT + attention pooling + OLMo text; mlx-vlm baseline is a 1-token anomaly; 327 prompt, 100 gen |
| molmo2 | molmo2-4b | ✅ | 1011.99 | 59.36 | 98% | mlx-vlm: 60.87; fast SDPA vision encoder; 430 prompt, 100 gen |
| internvl3 | InternVL3-1B | ✅ | 1760.25 | 225.51 | 85% | mlx-vlm: 264.40; InternViT + pixel-shuffle + Qwen2; 293 prompt, 8 gen |
| nemotron-omni | Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | ✅ | 312.49 | 67.97 | - | mlx-vlm: FAIL; NEW (5-19); Mamba2+Transformer+MoE+Parakeet audio; 100 gen |
| youtu-vl | youtu-vl-4b-instruct | ⚠️ | 569.47 | 24.19 | - | mlx-vlm: FAIL; NEW (5-19); only 1 gen token |
| qwen2-vl | Qwen2-VL-2B-Instruct-4bit | ⚠️ | 527.90 | 0.00 | - | Custom ViT + MRoPE; text-only pass; VLM warmup failure |
| qwen2.5-vl | Qwen2.5-VL-3B-Instruct-4bit | ✅ | 855.62 | 97.51 | - | Windowed ViT + MRoPE; 91 prompt, 46 gen; mlx-vlm requires PyTorch |
| qwen3-vl | Qwen3-VL-2B-Instruct-4bit | ✅ | 353.61 | 170.02 | - | mlx-vlm: FAIL; DeepStack + vectorized MRoPE (PR #729); 100 gen |
| qwen3-vl (4B) | Qwen3-VL-4B-Instruct-4bit | ✅ | 235.74 | 94.30 | - | mlx-vlm: FAIL; DeepStack + vectorized MRoPE (PR #729); 100 gen |
| qwen3-vl (8B) | Qwen3-VL-8B-Instruct-4bit | ✅ | 174.56 | 66.15 | - | mlx-vlm: FAIL; DeepStack + vectorized MRoPE (PR #729); 100 gen |
| qwen3-vl (32B) | Qwen3-VL-32B-Instruct-4bit | ✅ | 52.91 | 18.69 | - | mlx-vlm: FAIL; DeepStack + vectorized MRoPE (PR #729); 100 gen |
| qwen3-vl-moe | Qwen3-VL-30B-A3B-Instruct-4bit | ✅ | 166.66 | 26.39 | - | mlx-vlm: FAIL; MoE (128 experts) + DeepStack (PR #729); 100 gen |
| qwen3.5-vl (0.8B) | qwen3.5-0.8B-4bit | ✅ | 348.40 | 202.24 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet VLM; 57 prompt, 53 gen |
| qwen3.5-vl (2B) | qwen3.5-2B-4bit | ✅ | 440.76 | 178.21 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet VLM; 57 prompt, 58 gen |
| qwen3.5-vl (4B) | qwen3.5-4B-4bit | ✅ | 259.33 | 93.67 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet VLM; 57 prompt, 30 gen |
| qwen3.5-vl (9B 4bit) | qwen3.5-9B-4bit | ✅ | 172.50 | 71.85 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet VLM; 57 prompt, 62 gen |
| qwen3.5-vl (9B bf16) | qwen3.5-9B (bf16) | ✅ | 160.80 | 30.74 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet VLM; 57 prompt, 78 gen; bf16 |
| qwen3.5-vl (27B) | qwen3.5-27B-4bit | ✅ | 57.73 | 24.35 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet VLM; 57 prompt, 42 gen |
| qwen3.5-vl-moe | qwen3.5-35B-A3B-4bit | ✅ | 233.93 | 71.20 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet + MoE VLM; 57 prompt, 47 gen; gated delta decode RMSNorm fix (PR #730) |
| qwen3.6-vl-moe | qwen3.6-35B-A3B-4bit | ✅ | 227.36 | 68.03 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet + MoE VLM; 100 gen |
| molmo-point | - | ⏳ | - | - | - | Molmo-Point (point detection); implemented but no MLX model available |

**VLM test conditions**: Image: 224x224 PNG (test_image.png) unless noted. Prompt: "What is in this image?" Max tokens: 100. Prefill includes vision encoder + projector overhead. mlx-vlm baseline uses the `d85ca4d` dev checkout. mlxcel decode speed was measured with `mlxcel-bench-decode` (model load, warmup, and measured pass in one process). Models with unavailable or failed mlx-vlm runs are marked with "-" in the vs mlx-vlm column. Three text-only models (`deepseek-v3-4bit` 99GB, `minimax-m2-3bit` 93GB, `qwen3-next-480b-4bit` 251GB) skipped on this 128GB host per >65GB threshold. Two Gemma 3 VLM rows (gemma-3-4b-it-4bit, gemma3-4b-4bit) were measured with `--warmup-tokens 0` because the prepared 4D attention mask shape is single-use against the first prefill's KV cache offset. The three Gemma 3n VLM rows use the default warmup=20 path.

## Summary Statistics

| Status | Count |
|--------|-------|
| ✅ Pass | 117 (82 text + 35 VLM) |
| ⚠️ Partial | 6 (2 text + 4 VLM) |
| ❌ Fail | 6 (6 text + 0 VLM) |
| ⏳ Pending / Skipped (>65GB) | 16 (13 text pending + 3 oversize skip) |

## Performance Comparison

The detailed same-day decode comparison tables below are the authoritative
source for baseline comparisons. Decode remains the primary apples-to-apples
runtime comparison.

### Aggregate (decode, same-day baseline)

| Mode | Comparable pairs | Median mlxcel/baseline | >=90% parity | >= baseline | Range |
|------|-----------------:|-----------------------:|-------------:|------------:|------:|
| Text vs mlx-lm | 73 | 97% | 65/73 (89%) | 20/73 (27%) | 47%-113% |
| VLM vs mlx-vlm | 18 | 98% | 12/18 (67%) | 8/18 (44%) | 76%-127% |

### Representative decode wins

| Model | mlxcel | Baseline | vs baseline |
|-------|-------:|---------:|------------:|
| qwen2.5-0.5b-4bit | 355.29 | 315.48 | **113%** |
| phi-3.5-moe-4bit | 74.41 | 69.28 | **107%** |
| minicpm3-4b-4bit | 80.24 | 73.26 | **110%** |
| smollm-135m-4bit | 407.36 | 375.91 | **108%** |
| llava-interleave-qwen-0.5b-bf16 (VLM) | 269.86 | 225.15 | **120%** |
| gemma3n-e2b-4bit (VLM) | 72.38 | 59.57 | **122%** |
| gemma-4-e2b-it-4bit (VLM) | 107.06 | 97.19 | **110%** |
| phi-3.5-vision-4bit (VLM) | 117.63 | 92.53 | **127%** |

### Main optimization gaps

| Model | mlxcel | Baseline | vs baseline | Notes |
|-------|-------:|---------:|------------:|-------|
| falcon-mamba-7b-4bit | 42.91 | 91.04 | 47% | Chat template causes early EOS; only 2 generated tokens |
| qwen2.5-vl-3b-4bit (text path) | 98.53 | 160.42 | 61% | VLM wrapper text-only comparison |
| qwen2-vl-2b-4bit (text path) | 150.02 | 236.86 | 63% | VLM wrapper text-only comparison |
| gemma-4-31b-4bit (VLM) | 15.46 | 20.30 | 76% | large VLM path |
| gemma-3-4b-it-4bit (VLM) | 85.61 | 97.36 | 88% | measured with warmup=0; see VLM test conditions |

## Performance vs mlx-lm / mlx-vlm baseline (2026-05-19, same-day sweep)

Source CSVs (same M1 Ultra host, mlxcel 0.0.28 with `--cooldown 0`; mlx-lm/mlx-vlm baselines from same 2026-05-19 sweep with `PYLM_BENCH_MAX_GB=65`):

- mlxcel: `benchmarks/metal_m1ultra_2026-05-19.csv`
- mlx-lm: `benchmarks/pylm_m1ultra_2026-05-19.csv` (mlx-lm 0.31.3 dev checkout in `references/mlx-lm` @ `df1d3f3`)
- mlxcel VLM: `benchmarks/metal_m1ultra_vlm_2026-05-19.csv`
- mlx-vlm: `benchmarks/pylm_m1ultra_vlm_2026-05-19.csv` (mlx-vlm dev checkout in `references/mlx-vlm` @ `d85ca4d`)

All 4 sweeps use the same `--max-tokens 100`, same `Hello, how are you today?` / `What is in this image?` prompts, and the same >65GB skip threshold (deepseek-v3-4bit, minimax-m2-3bit, qwen3-next-480b-4bit excluded from both sides).

Numbers are decode tok/s. `mlxcel vs mlx-lm` is `mlxcel / mlx-lm` as a percentage; **bold** = mlxcel >= mlx-lm. `FAIL` cells are real load/runtime errors on that backend with this configuration. The mlx-lm checkout used here (`df1d3f3` — "Fix Gemma 4 sanitize() not stripping KV projections for shared layers" #1240) is newer than the M5 Max page's `ed1fca4`, so some FAIL categories differ.

### Aggregate (text)

- **Comparable text pairs**: 73
- **mlxcel >= mlx-lm**: 20 / 73 (27%)
- **mlxcel >= 90% parity**: 65 / 73 (89%)
- **Average mlxcel/mlx-lm**: 96% (median 97%, range 47%-113%)

### Aggregate (VLM, models with >=5 generated tokens both sides)

- **Comparable VLM pairs**: 18
- **mlxcel >= mlx-vlm**: 8 / 18 (44%)
- **mlxcel >= 90% parity**: 12 / 18 (67%)
- **Average mlxcel/mlx-vlm**: 99% (median 98%, range 76%-127%)

### Text decode (tok/s)

| Model | mlxcel | mlx-lm | mlxcel vs mlx-lm |
|-------|--------|--------|------------------|
| Meta-Llama-3.1-8B-Instruct-4bit | 106.61 | 109.84 | 97% |
| Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | 83.29 | FAIL | - |
| Qwen2.5-1.5B-4bit | 241.44 | 241.41 | **100%** |
| Qwen2.5-1.5B-Instruct-4bit | 240.24 | 239.20 | **100%** |
| Qwen2.5-7B-Instruct-4bit | 110.67 | 110.90 | 100% |
| Qwen3.5-0.8B-OptiQ-4bit | FAIL | 265.86 | - |
| aya-expanse-8b-4bit | 107.30 | 112.74 | 95% |
| aya-vision-8b | 109.57 | FAIL | - |
| baichuan-m1-14b-4bit | 40.32 | 49.11 | 82% |
| bunny-llama3-8b-4bit | 102.29 | FAIL | - |
| command-r7b-4bit | 110.22 | 107.75 | **102%** |
| deepseek-coder-1.3b-4bit | 164.77 | FAIL | - |
| deepseek-r1-distill-7b-4bit | 110.82 | 111.34 | 100% |
| deepseek-v2-lite-4bit | 111.86 | 117.06 | 96% |
| deepseek-v3-4bit | - | FAIL | - |
| ernie-4.5-0.3b-4bit | 526.70 | FAIL | - |
| exaone-3.5-2.4b-4bit | 199.04 | 194.65 | **102%** |
| exaone4-1.2b-4bit | 252.36 | FAIL | - |
| falcon-mamba-7b-4bit | 42.91 | 91.04 | 47% |
| gemma-2b-4bit | 189.91 | 207.78 | 91% |
| gemma-3-4b-it-4bit | 113.61 | 109.72 | **104%** |
| gemma-4-26b-a4b-it-4bit | 73.18 | 72.52 | **101%** |
| gemma-4-31b-4bit | 20.15 | 20.36 | 99% |
| gemma-4-31b-it-4bit | 19.06 | 20.23 | 94% |
| gemma-4-e2b-it-4bit | 116.08 | FAIL | - |
| gemma-4-e2b-it-8bit | 87.42 | FAIL | - |
| gemma-4-e4b-it-4bit | 81.88 | FAIL | - |
| gemma-4-e4b-it-8bit | 59.35 | FAIL | - |
| gemma2-2b-4bit | 163.85 | 153.50 | **107%** |
| gemma3-1b-4bit | 225.75 | 211.50 | **107%** |
| gemma3-4b-4bit | 112.95 | 109.48 | **103%** |
| gemma3n-e2b-4bit | 76.86 | FAIL | - |
| gemma3n-e4b-4bit | 60.18 | FAIL | - |
| gemma3n-e4b-bf16 | 34.41 | 39.02 | 88% |
| glm4-flash-4bit | 47.32 | 49.47 | 96% |
| gpt-oss-120b-4bit | 58.89 | 57.58 | **102%** |
| gpt-oss-20b-mxfp4 | 88.89 | 89.51 | 99% |
| hunyuan-1.8b-4bit | 182.98 | 200.59 | 91% |
| hunyuan-large-4bit | 44.22 | FAIL | - |
| internlm2-7b-4bit | 107.52 | 111.92 | 96% |
| internlm3-8b-4bit | 86.88 | FAIL | - |
| jamba-v0.1-4bit | 119.33 | 131.04 | 91% |
| llama-3.1-8b-4bit | 107.23 | 110.66 | 97% |
| llama-3.1-8b-bf16 | 35.15 | 35.32 | 100% |
| llama-3.2-1b-4bit | 377.90 | 418.25 | 90% |
| llama-4-scout-17b-4bit | 35.21 | FAIL | - |
| llava-1.5-7b-4bit | 115.94 | FAIL | - |
| llava-interleave-qwen-0.5b-bf16 | 314.17 | FAIL | - |
| llava-next-mistral-7b-4bit | 113.47 | FAIL | - |
| mamba2-1.3b-4bit | 102.63 | FAIL | - |
| mimo-7b-4bit | 85.28 | 86.17 | 99% |
| minicpm-2b-4bit | 163.05 | 156.47 | **104%** |
| minicpm3-4b-4bit | 80.24 | 73.26 | **110%** |
| minimax-m2-3bit | - | FAIL | - |
| ministral-3b-4bit | 145.06 | 159.34 | 91% |
| mistral-small-3.1-24b-4bit | 31.70 | 31.97 | 99% |
| mixtral-8x7b-4bit | 53.62 | 54.91 | 98% |
| molmo2-4b | 59.67 | FAIL | - |
| nemotron-h-30b-4bit | 90.25 | 93.34 | 97% |
| nemotron-nas-30b-4bit | 90.67 | 92.93 | 98% |
| olmo-1b-4bit | 219.54 | FAIL | - |
| olmo2-7b-4bit | 103.66 | 110.88 | 93% |
| olmo3-32b-4bit | 21.79 | 21.57 | **101%** |
| paligemma2-3b-6bit | 0.00 | FAIL | - |
| phi-2-4bit | 59.62 | FAIL | - |
| phi-3-mini-4bit | 167.27 | 171.36 | 98% |
| phi-3.5-mini-4bit | 160.90 | 166.30 | 97% |
| phi-3.5-moe-4bit | 74.41 | 69.28 | **107%** |
| phi-3.5-vision-4bit | 160.16 | FAIL | - |
| phi-4-4bit | 57.02 | 58.68 | 97% |
| pixtral-12b-4bit | 68.90 | 69.49 | 99% |
| qwen1.5-moe-a2.7b-4bit | 144.37 | 144.98 | 100% |
| qwen2-vl-2b-4bit | 150.02 | 236.86 | 63% |
| qwen2.5-0.5b-4bit | 355.29 | 315.48 | **113%** |
| qwen2.5-7b-4bit | 109.77 | 111.38 | 99% |
| qwen2.5-7b-8bit | 69.08 | 70.46 | 98% |
| qwen2.5-vl-3b-4bit | 98.53 | 160.42 | 61% |
| qwen3-0.6b-4bit | 295.09 | 299.61 | 98% |
| qwen3-1.7b-4bit | 195.30 | 221.37 | 88% |
| qwen3-30b-a3b-4bit | 71.07 | 70.18 | **101%** |
| qwen3-4b-4bit | 120.70 | 123.92 | 97% |
| qwen3-8b-4bit | 79.85 | 84.54 | 94% |
| qwen3-moe-4bit | 71.47 | 69.67 | **103%** |
| qwen3-next-480b-4bit | - | FAIL | - |
| qwen3-vl-2b-4bit | 215.98 | 222.67 | 97% |
| qwen3-vl-30b-a3b-4bit | 69.78 | 70.04 | 100% |
| qwen3-vl-32b-4bit | 20.68 | 21.99 | 94% |
| qwen3-vl-4b-4bit | 113.84 | 124.02 | 92% |
| qwen3-vl-8b-4bit | 79.36 | 84.46 | 94% |
| qwen3.5-0.8b-4bit | 243.32 | 269.52 | 90% |
| qwen3.5-27b-4bit | 24.25 | 25.93 | 94% |
| qwen3.5-2b-4bit | 172.05 | 211.68 | 81% |
| qwen3.5-35b-a3b-4bit | 72.05 | 76.44 | 94% |
| qwen3.5-4b-4bit | 95.76 | 115.60 | 83% |
| qwen3.5-9b-4bit | 72.36 | 81.27 | 89% |
| qwen3.5-9b-bf16 | 31.28 | 34.22 | 91% |
| qwen3.6-35b-a3b-4bit | 68.77 | 73.18 | 94% |
| smollm-135m-4bit | 407.36 | 375.91 | **108%** |
| smollm3-3b-4bit | 136.34 | 141.66 | 96% |
| solar-open-100b-4bit | 35.88 | 35.69 | **101%** |
| stablelm-1.6b-4bit | 280.32 | 280.65 | 100% |
| starcoder2-3b-4bit | 171.30 | 166.17 | **103%** |
| youtu-vl-4b-instruct | 0.00 | FAIL | - |

### VLM decode (tok/s)

| Model | mlxcel | mlx-vlm | mlxcel vs mlx-vlm |
|-------|--------|--------|------------------|
| Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | 67.97 | FAIL | - |
| aya-vision-8b | 109.36 | 103.74 | **105%** |
| bunny-llama3-8b-4bit | 96.07 | FAIL | - |
| deepseek-v3-4bit | - | FAIL | - |
| gemma-3-4b-it-4bit | 85.61 | 97.36 | 88% |
| gemma-4-26b-a4b-it-4bit | 63.18 | 61.07 | **103%** |
| gemma-4-31b-4bit | 15.46 | 20.30 | 76% |
| gemma-4-31b-it-4bit | 18.32 | 19.78 | 93% |
| gemma-4-e2b-it-4bit | 107.06 | 97.19 | **110%** |
| gemma-4-e2b-it-8bit | 80.48 | 91.06 | 88% |
| gemma-4-e4b-it-4bit | 72.91 | 70.34 | **104%** |
| gemma-4-e4b-it-8bit | 54.83 | 63.25 | 87% |
| gemma3-4b-4bit | 86.97 | 93.79 | 93% |
| gemma3n-e2b-4bit | 72.38 | 59.57 | **122%** |
| gemma3n-e4b-4bit | 56.69 | 50.00 | **113%** |
| gemma3n-e4b-bf16 | 32.12 | 36.18 | 89% |
| internvl3-1b | 225.51 | 264.40 | 85% |
| llama-4-scout-17b-4bit | 31.73 | FAIL | - |
| llava-1.5-7b-4bit | 101.61 | FAIL | - |
| llava-interleave-qwen-0.5b-bf16 | 269.86 | 225.15 | **120%** |
| llava-next-mistral-7b-4bit | 105.41 | 109.51 | 96% |
| minimax-m2-3bit | - | FAIL | - |
| ministral-3b-4bit | 123.91 | FAIL | - |
| mistral-small-3.1-24b-4bit | 29.69 | FAIL | - |
| molmo-7b | 77.98 | 38399.52 (anomalous) | - |
| molmo2-4b | 59.36 | 60.87 | 98% |
| paligemma2-3b-6bit | 45.09 | 70.45 | 64% |
| phi-3.5-vision-4bit | 117.63 | 92.53 | **127%** |
| pixtral-12b-4bit | 59.17 | FAIL | - |
| qwen2-vl-2b-4bit | 0.00 | FAIL | - |
| qwen2.5-vl-3b-4bit | 97.51 | FAIL | - |
| qwen3-next-480b-4bit | - | FAIL | - |
| qwen3-vl-2b-4bit | 170.02 | FAIL | - |
| qwen3-vl-30b-a3b-4bit | 26.39 | FAIL | - |
| qwen3-vl-32b-4bit | 18.69 | FAIL | - |
| qwen3-vl-4b-4bit | 94.30 | FAIL | - |
| qwen3-vl-8b-4bit | 66.15 | FAIL | - |
| qwen3.5-0.8b-4bit | 202.24 | FAIL | - |
| qwen3.5-27b-4bit | 24.35 | FAIL | - |
| qwen3.5-2b-4bit | 178.21 | FAIL | - |
| qwen3.5-35b-a3b-4bit | 71.20 | FAIL | - |
| qwen3.5-4b-4bit | 93.67 | FAIL | - |
| qwen3.5-9b-4bit | 71.85 | FAIL | - |
| qwen3.5-9b-bf16 | 30.74 | FAIL | - |
| qwen3.6-35b-a3b-4bit | 68.03 | FAIL | - |
| youtu-vl-4b-instruct | 24.19 | FAIL | - |

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
| falcon-mamba | Chat template causes early EOS (only 2 tokens); decode now 42.91 tok/s | Medium |
| paligemma | Only 2 VLM gen tokens; decode is not comparable despite 45.09 tok/s measured | High |
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

## TurboQuant KV cache — M1 Ultra speed gate readings (issue #509)

First dedicated M1 Ultra reading of the epic-#458 KV speed gate matrix.
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

Per epic #458 §"Cross-hardware regression", the M5 Max gates are tracking
only on pre-M3 hardware. The numbers above match that guidance: `turbo4`
and `turbo4-asym` decode are well below the M5 gates on M1 Ultra, while
`turbo4-delegated` prefill is bit-identical to FP16 because the cold pages
keep their FP16 representation and only the hot tail is packed.

The decode regression is consistent with the L2-cache wall documented in
`references/turboquant_plus/`. The fused Sparse-V Metal kernel that lands
in #505 targets the per-thread skip path inside the SDPA inner loop and is
expected to recover most of the M5 decode budget; the M1/M2 ceiling stays
limited by L2 bandwidth and is documented but not gated.

### M1 Ultra hardware considerations

- **`turbo4-delegated` is the only currently shipping mode that holds the
  prefill gate on M1 Ultra.** Use it when prefill latency matters more than
  the maximum compression ratio.
- **`turbo4-asym` on M1 Ultra needs the #505 fused kernel** to be a viable
  decode option. The graph-level path measured here pays a per-token
  dequant cost that the fused kernel folds into the SDPA inner loop.
- **Avoid `turbo3` on M1 Ultra** for decode-bound workloads. The 3-bit
  pack/unpack loop saturates L2 bandwidth on the older GPU microarchitecture;
  on M3/M4/M5 the regression is smaller or absent.
- Memory-vs-speed trade-off on M1 Ultra: `fp16+turbo4` (alias `turbo4-asym`)
  still delivers approximately 0.39× of the FP16 baseline KV footprint at the
  default boundary-v 2 on a 32-layer model, but the decode shortfall above
  makes `turbo4-delegated` the better practical choice on this generation
  until the #505 fused kernel lands. Pick `fp16+turbo4` only if the memory
  savings outweigh the per-token decode cost for your workload.

### Deferred

- 32K decode reading (best-effort per epic; benefits from #505 fused kernel
  before the gate is meaningful on any hardware).
- Per-mode 16K decode (skipped for the initial run; M5 Max is the primary
  target for the 16K gate).
- Multi-model expansion (Qwen 2.5, Gemma 3, etc.).
- M5 Max readings of the same matrix — to be filled in by a manual run on
  the M5 Max dev box; the script is hardware-agnostic and writes
  `benchmarks/turbo_kv/<date>_<hw>_<model>.csv` keyed off
  `sysctl -n machdep.cpu.brand_string`.
