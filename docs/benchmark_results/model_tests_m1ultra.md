# Model Compatibility & Performance Tests (M1 Ultra)

Compatibility and performance testing for mlxcel models on **Mac Studio M1 Ultra 128GB**, with comparison against Python mlx-lm / mlx-vlm.

## Test Environment

| Item | Value |
|------|-------|
| **Hardware** | Mac Studio M1 Ultra, 128GB RAM |
| **OS** | macOS 26.4 (Tahoe) |
| **mlxcel version** | 0.1.4 |
| **MLX version** | 0.32.0-dev pin (commit a6ec712, 2026-06-11 upstream main, via mlxcel-core) |
| **Bench harness** | `mlxcel-bench-decode` (model load, warmup, and measured pass in one process) |
| **mlx-lm baseline** | 0.31.3 (dev checkout https://github.com/ml-explore/mlx-lm @ `df1d3f3` — "Fix Gemma 4 sanitize() not stripping KV projections for shared layers" ml-explore/mlx-lm#1240) |
| **mlx-vlm baseline** | dev checkout https://github.com/Blaizzy/mlx-vlm @ `d85ca4d` — "Compatibility bridge for non-VL models" Blaizzy/mlx-vlm#1181 |
| **Test Prompt** | "Hello, how are you today?" (text) / "What is in this image?" (VLM) |
| **Max Tokens** | 100 (measured pass); 20 (warmup pass, same process) |
| **Test Date** | 2026-05-19 full sweep (baseline); 2026-05-28 full text + VLM re-benchmark on mlxcel 0.1.0 (`--cooldown 0`); 2026-06-12 full text + VLM re-benchmark on mlxcel 0.1.4 (MLX pin a6ec712, post issue #222 bump) |
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
| llama3 | Llama-3.2-1B-Instruct-4bit | ✅ | 1952.18 | 365.58 | 87% | mlx-lm: 418.25; only 48 tokens |
| llama3 (8B bf16) | Llama-3.1-8B-Instruct (bf16) | ✅ | 422.24 | 35.32 | **100%** | mlx-lm: 35.32; non-quantized |
| llama3.1 | Llama-3.1-8B-Instruct-4bit | ✅ | 481.82 | 107.24 | 97% | mlx-lm: 110.66; only 54 tokens |
| llama4 | Llama-4-Scout-17B-16E-Instruct-4bit | ⚠️ | 119.41 | 36.20 | - | mlx-lm: FAIL; long outputs repetitive |
| qwen2 | Qwen2.5-0.5B-Instruct-4bit | ✅ | 1084.13 | 342.79 | **109%** | mlx-lm: 315.48 |
| qwen2 (7B 4bit) | Qwen2.5-7B-Instruct-4bit | ✅ | 310.34 | 111.46 | **101%** | mlx-lm: 110.90 |
| qwen2 (7B 8bit) | Qwen2.5-7B-Instruct-8bit | ✅ | 312.92 | 69.96 | 99% | mlx-lm: 70.46; 8-bit quantized |
| qwen3 | Qwen3-0.6B-4bit | ✅ | 561.06 | 290.40 | 97% | mlx-lm: 299.61 |
| qwen3 (1.7B) | Qwen3-1.7B-4bit | ✅ | 413.95 | 188.67 | 85% | mlx-lm: 221.37 |
| qwen3 (4B) | Qwen3-4B-4bit | ✅ | 246.14 | 119.56 | 96% | mlx-lm: 123.92 |
| qwen3 (8B) | Qwen3-8B-4bit | ✅ | 168.10 | 80.42 | 95% | mlx-lm: 84.54 |
| qwen3_5 (0.8B) | Qwen3.5-0.8B-4bit | ✅ | 460.58 | 239.31 | 89% | mlx-lm: 269.52; Hybrid GatedDeltaNet |
| qwen3_5 (2B) | Qwen3.5-2B-4bit | ✅ | 408.39 | 174.85 | 83% | mlx-lm: 211.68; Hybrid GatedDeltaNet; only 36 tokens |
| qwen3_5 (4B) | Qwen3.5-4B-4bit | ✅ | 244.40 | 99.69 | 86% | mlx-lm: 115.60; Hybrid GatedDeltaNet; only 36 tokens |
| qwen3_5 (9B 4bit) | qwen3.5-9B-4bit | ✅ | 155.57 | 71.55 | 88% | mlx-lm: 81.27; Hybrid GatedDeltaNet; only 29 tokens |
| qwen3_5 (9B bf16) | qwen3.5-9B (bf16) | ✅ | 149.19 | 31.58 | 92% | mlx-lm: 34.22; bf16, not quantized; Hybrid GatedDeltaNet (compiled fused kernel) |
| qwen3_5 (27B) | qwen3.5-27B-4bit | ✅ | 52.88 | 24.27 | 94% | mlx-lm: 25.93; Hybrid Transformer+GatedDeltaNet; VLM wrapper format |
| qwen3_6 | qwen3.6-35B-A3B-4bit | ✅ | 233.24 | 67.31 | 92% | mlx-lm: 73.18; MoE architecture; 100 tokens |
| qwen3_next | qwen3-next-480B-4bit | ⏳ | - | SKIP | - | Qwen3Next 480B architecture; >65GB skipped on 128GB host |
| qwen2 (1.5B) | Qwen2.5-1.5B-Instruct-4bit | ✅ | 837.72 | 237.54 | 99% | mlx-lm: 239.20; 100 tokens |
| qwen2 (1.5B base) | Qwen2.5-1.5B-4bit | ✅ | 697.51 | 238.28 | 99% | mlx-lm: 241.41; base variant; 100 tokens |
| phi | phi-2-hf-4bit-mlx | ✅ | 144.60 | 58.55 | - | mlx-lm fails to load; only 1 token (likely EOS) |
| phi3 | Phi-3-mini-4k-instruct-4bit | ✅ | 199.90 | 168.94 | 99% | mlx-lm: 171.36; only 25 tokens |
| phi3small | Phi-3.5-mini-instruct-4bit | ✅ | 233.84 | 164.19 | 99% | mlx-lm: 166.30; only 40 tokens |
| phi4 | Phi-4-4bit | ✅ | 115.67 | 57.98 | 99% | mlx-lm: 58.68 |
| smollm3 | SmolLM-135M-Instruct-4bit | ✅ | 571.01 | 380.11 | **101%** | mlx-lm: 375.91 |
| smollm3 (3B) | SmolLM3-3B-4bit | ✅ | 578.93 | 135.55 | 96% | mlx-lm: 141.66 |
| stablelm | stablelm-2-1_6b-chat-4bit | ✅ | 667.16 | 282.33 | **101%** | mlx-lm: 280.65; only 59 tokens |
| starcoder2 | starcoder2-3b-4bit | ✅ | 175.80 | 170.18 | **102%** | mlx-lm: 166.17 |
| olmo | OLMo-1B-hf-4bit | ✅ | 186.24 | 209.88 | - | mlx-lm: FAIL |
| olmo2 | OLMo2-7B-4bit | ✅ | 281.57 | 103.92 | 94% | mlx-lm: 110.88; only 27 tokens |
| olmo3 | OLMo3.1-32B-4bit | ✅ | 81.21 | 21.74 | **101%** | mlx-lm: 21.57 |
| minicpm | MiniCPM-2B-sft-bf16-4bit | ✅ | 300.65 | 163.58 | **105%** | mlx-lm: 156.47 |
| mimo | MiMo-7B-RL-4bit | ✅ | 230.49 | 85.85 | 100% | mlx-lm: 86.17 |

## Gemma Family

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| gemma | gemma-2b-it-4bit | ✅ | 367.53 | 191.07 | 92% | mlx-lm: 207.78; only 49 tokens |
| gemma2 | gemma-2-2b-it-4bit | ✅ | 337.08 | 165.06 | **108%** | mlx-lm: 153.50; only 18 tokens |
| gemma3 | gemma-3-1b-it-4bit | ✅ | 470.77 | 227.37 | **108%** | mlx-lm: 211.50; only 34 tokens |
| gemma3 (4B) | gemma-3-4b-it-4bit | ✅ | 207.76 | 115.98 | **106%** | mlx-lm: 109.48; only 86 tokens |
| gemma4 (12B) | gemma-4-12b-it-4bit | ✅ | 104.55 | 34.55 | - | NEW (6-12); no mlx-lm baseline |
| gemma4 (31B) | gemma-4-31b-4bit | ✅ | 24.09 | 20.25 | 99% | mlx-lm: 20.36 |
| gemma4 (31B-it) | gemma-4-31b-it-4bit | ✅ | 50.42 | 19.34 | 96% | mlx-lm: 20.23; instruction-tuned variant |
| gemma4 (31B-it QAT) | gemma-4-31B-it-qat-4bit | ✅ | 47.68 | 15.48 | - | NEW (6-12); QAT variant; no mlx-lm baseline |
| gemma4 (26B A4B) | gemma-4-26b-a4b-it-4bit | ✅ | 153.06 | 71.00 | 98% | mlx-lm: 72.52; only 26 tokens |
| gemma4 (26B A4B QAT) | gemma-4-26B-A4B-it-qat-4bit | ✅ | 173.54 | 71.65 | - | NEW (6-12); QAT variant; no mlx-lm baseline |
| gemma4 (diffusion 26B A4B) | diffusiongemma-26B-A4B-it-4bit | ✅ | 177.53 | 70.66 | - | NEW (6-12); block-diffusion; AR-equivalent decode of the same backbone; see docs/block-diffusion.md |
| gemma4 (E2B 4bit) | gemma-4-e2b-it-4bit | ✅ | 252.14 | 117.90 | - | mlx-lm: FAIL; only 34 tokens |
| gemma4 (E2B 8bit) | gemma-4-e2b-it-8bit | ✅ | 232.38 | 87.36 | - | mlx-lm: FAIL; only 38 tokens |
| gemma4 (E2B QAT) | gemma-4-e2b-it-qat-4bit | ✅ | 192.75 | 93.65 | - | NEW (6-12); QAT variant; no mlx-lm baseline |
| gemma4 (E4B 4bit) | gemma-4-e4b-it-4bit | ✅ | 172.24 | 82.09 | - | mlx-lm: FAIL; only 25 tokens |
| gemma4 (E4B 8bit) | gemma-4-e4b-it-8bit | ✅ | 150.61 | 57.80 | - | mlx-lm: FAIL; only 39 tokens |
| gemma4 (E4B QAT) | gemma-4-e4b-it-qat-4bit | ✅ | 135.27 | 62.84 | - | NEW (6-12); QAT variant; no mlx-lm baseline |
| gemma3n | gemma-3n-E2B-it-4bit | ✅ | 241.90 | 77.41 | - | mlx-lm: FAIL; only 69 tokens |
| gemma3n (E4B) | gemma-3n-E4B-it-4bit | ✅ | 176.80 | 59.91 | - | mlx-lm: FAIL; only 74 tokens |
| gemma3n (E4B bf16) | gemma-3n-E4B-it (bf16) | ✅ | 178.92 | 34.65 | 89% | mlx-lm: 39.02; bf16; AltUp/MLP decode graph scheduling |
| recurrent_gemma | - | ⏳ | - | - | - | Griffin SSM+attention hybrid |

## EXAONE

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| exaone | EXAONE-3.5-2.4B-Instruct-4bit | ✅ | 714.88 | 195.13 | **100%** | mlx-lm: 194.65 |
| exaone4 | exaone-4.0-1.2b-4bit | ✅ | 437.35 | 243.72 | - | mlx-lm: FAIL; only 18 tokens |
| exaone_moe | - | ⏳ | - | - | - | |

## Cohere Command R

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| cohere | c4ai-command-r7b-12-2024-4bit | ✅ | 99.55 | 111.27 | **103%** | mlx-lm: 107.75 |
| cohere2 | aya-expanse-8b-4bit | ✅ | 97.22 | 107.34 | 95% | mlx-lm: 112.74 |

## MoE (Mixture of Experts)

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| minimax | MiniMax-M2-3bit | ✅ | 54.31 | 33.14 | - | mlx-lm: FAIL on this host; 93GB, fits on the 128GB host since the 05-28 run |
| mixtral | Mixtral-8x7B-Instruct-v0.1-4bit | ✅ | 80.21 | 54.20 | 99% | mlx-lm: 54.91; only 73 tokens |
| qwen2_moe | Qwen1.5-MoE-A2.7B-Chat-4bit | ✅ | 367.84 | 144.05 | 99% | mlx-lm: 144.98; only 43 tokens |
| qwen3_moe | Qwen3-30B-A3B-4bit | ✅ | 193.88 | 70.27 | **100%** | mlx-lm: 70.18 |
| qwen3_5_moe | qwen3.5-35B-A3B-4bit | ✅ | 226.89 | 70.05 | 92% | mlx-lm: 76.44; Hybrid GatedDeltaNet + MoE (256 experts); only 34 tokens |
| phimoe | Phi-3.5-MoE-instruct-4bit | ✅ | 113.37 | 77.13 | **111%** | mlx-lm: 69.28 |
| solar_open | Solar-Open-100B-4bit | ✅ | 71.91 | 36.24 | **102%** | mlx-lm: 35.69; 128 experts, top-8; layer-eval skip; 54GB |
| solar_open (int4) | Solar-Open-100B-int4 | ✅ | - | 11.55 | - | mlx-lm: fails to load; 128 experts, top-8; int4 quantization; 54GB; not in the 06-12 sweep |
| olmoe | - | ⏳ | - | - | - | |
| gpt_oss (20B) | gpt-oss-20b-MXFP4-Q4 | ✅ | 283.96 | 89.48 | 100% | mlx-lm: 89.51; MXFP4 quantization; 32 experts; bf16 decode fix |
| gpt_oss (120B) | gpt-oss-120b-4bit | ✅ | 162.07 | 59.86 | **104%** | mlx-lm: 57.58; 128 experts, top-4; 61GB model; bf16 decode fix |

## DeepSeek Family

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| deepseek | deepseek-coder-1.3b-instruct-4bit | ✅ | 1402.49 | 160.53 | - | mlx-lm: FAIL |
| deepseek_v2 | DeepSeek-V2-Lite-Chat-4bit | ✅ | 203.43 | 109.70 | 94% | mlx-lm: 117.06; only 18 tokens |
| deepseek_r1 | DeepSeek-R1-Distill-Qwen-7B-4bit | ✅ | 165.96 | 110.80 | 100% | mlx-lm: 111.34 |
| deepseek_v3 | deepseek-v3-4bit | ⏳ | - | SKIP | - | MoE + MLA; >65GB skipped on 128GB host (99GB) |
| deepseek_v32 | - | ⏳ | - | - | - | |

## MLA (Multi-head Latent Attention)

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| minicpm3 | MiniCPM3-4B-4bit | ✅ | 240.07 | 79.84 | **109%** | mlx-lm: 73.26 |

## Nemotron Family

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| nemotron_h | Nemotron-H-30B-4bit | ✅ | 158.39 | 90.92 | 97% | mlx-lm: 93.34; Hybrid Mamba2+Transformer+MoE; SSM Metal kernel |
| nemotron_nas | Nemotron-NAS-30B-A3B-4bit | ✅ | 160.69 | 90.09 | 97% | mlx-lm: 92.93; Hybrid Mamba2+Transformer+MoE |
| nemotron_h_nano_omni | Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | ✅ | 160.49 | 85.02 | - | mlx-lm: FAIL; NEW (5-19); Mamba2+Transformer+MoE+Parakeet audio; 100 tokens |

## SSM / Mamba Models

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| mamba | Falcon-Mamba-7B-4bit | ⚠️ | 90.28 | 40.77 | 45% | mlx-lm: 91.04; only 2 tokens due to chat template EOS |
| mamba2 | mamba2-1.3b-4bit | ✅ | 152.04 | 92.19 | - | mlx-lm: FAIL |
| jamba | Jamba-v0.1-4bit | ✅ | 334.35 | 121.41 | 93% | mlx-lm: 131.04; only 76 tokens |
| rwkv7 | - | ⏳ | - | - | - | RWKV v7 linear attention |

## Chinese / Asian Language Models

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| baichuan | Baichuan-M1-14B-Instruct-4bit | ✅ | 58.57 | 41.09 | 84% | mlx-lm: 49.11; only 39 tokens |
| glm4 | GLM-4-Flash-4bit | ✅ | 123.11 | 46.99 | 95% | mlx-lm: 49.47; Only 18 tokens |
| glm4_moe | - | ⏳ | - | - | - | |
| glm4_moe_lite | GLM-4.7-Flash-4bit | ✅ | - | 31.54 | 76% | mlx-lm: 41.55; only 18 tokens |
| glm5 | GLM-5-4bit | ❌ | - | FAIL | - | warmup failure (persistent) |
| internlm2 | InternLM2-7B-4bit | ✅ | 214.74 | 108.59 | 97% | mlx-lm: 111.92 |
| internlm3 | internlm3-8b-instruct-4bit | ✅ | 315.53 | 86.88 | - | mlx-lm: FAIL |
| ernie4_5 | ERNIE-4.5-0.3B-Instruct-4bit | ✅ | 1044.20 | 505.06 | - | mlx-lm: FAIL |
| ernie4_5_moe | - | ⏳ | - | - | - | |
| hunyuan_moe | Hunyuan-Large-Instruct-4bit | ✅ | 63.11 | 44.30 | - | mlx-lm: FAIL |
| hunyuan_moe_13b | HunYuan-MoE-A13B-Instruct (bf16) | ❌ | - | FAIL | - | mlx-lm: fails to load; Tiktoken tokenizer; bf16; warmup failure |
| hunyuan_v1_dense | Hunyuan-1.8B-Instruct-4bit | ✅ | 276.76 | 184.13 | 92% | mlx-lm: 200.59; only 41 tokens |
| kimi_linear | - | ⏳ | - | - | - | Kimi linear attention (Moonshot) |
| step3p5 | - | ⏳ | - | - | - | Step 3.5 (StepFun) |

## Other Models

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| ministral3 | Ministral-3B-Instruct-4bit | ✅ | 900.69 | 143.36 | 90% | mlx-lm: 159.34; VLM wrapper; text-only mode; only 34 tokens |
| mistral4 | - | ⏳ | - | - | - | MLA + MoE; implemented but no MLX model available |
| moondream3 | moondream3-preview-4bit | ⚠️ | - | 8.45 | - | mlx-lm: fails to load; text-only test; SigLIP + MLP; image output garbled; only 14 tokens |
| longcat_flash | - | ⏳ | - | - | - | |
| longcat_flash_ngram | - | ⏳ | - | - | - | |
| mistral_small | mistral-small-3.1-24b-4bit | ✅ | 35.98 | 31.80 | 99% | mlx-lm: 31.97; text-only mode |

## Vision-Language Models (VLM)

| Model | Test Model | Status | Prefill | Decode | vs mlx-vlm | Notes |
|-------|------------|--------|---------|--------|------------|-------|
| gemma3 | gemma-3-4b-it-4bit | ✅ | 237.44 | 89.56 | 95% | mlx-vlm: 93.79; SigLIP + AvgPool; 275 prompt, 16 gen |
| gemma3n (E2B) | gemma-3n-E2B-it-4bit | ✅ | 741.76 | 73.35 | **123%** | mlx-vlm: 59.57; MobileNetV5 + MSFA; 273 prompt, 29 gen |
| gemma3n (E4B bf16) | gemma-3n-E4B-it (bf16) | ✅ | 644.20 | 32.07 | 89% | mlx-vlm: 36.18; MobileNetV5 + MSFA; bf16 prefill path retune; bf16; 273 prompt, 24 gen |
| gemma3n (E4B 4bit) | gemma-3n-E4B-it-4bit | ✅ | 473.58 | 57.30 | **115%** | mlx-vlm: 50.00; 273 prompt, 33 gen |
| gemma4 (12B) | gemma-4-12b-it-4bit | ✅ | 244.55 | 32.50 | - | NEW (6-12); no mlx-vlm baseline; 277 prompt, 25 gen |
| gemma4 (E2B 4bit) | gemma-4-e2b-it-4bit | ✅ | 700.49 | 105.83 | **109%** | mlx-vlm: 97.19; 274 prompt, 100 gen |
| gemma4 (E2B 8bit) | gemma-4-e2b-it-8bit | ✅ | 702.03 | 81.26 | 89% | mlx-vlm: 91.06; 274 prompt, 100 gen |
| gemma4 (E2B QAT) | gemma-4-e2b-it-qat-4bit | ✅ | 694.41 | 82.64 | - | NEW (6-12); QAT variant; 273 prompt, 42 gen |
| gemma4 (E4B 4bit) | gemma-4-e4b-it-4bit | ✅ | 443.38 | 74.67 | **106%** | mlx-vlm: 70.34; 274 prompt, 54 gen |
| gemma4 (E4B 8bit) | gemma-4-e4b-it-8bit | ✅ | 443.90 | 55.28 | 87% | mlx-vlm: 63.25; 274 prompt, 35 gen |
| gemma4 (E4B QAT) | gemma-4-e4b-it-qat-4bit | ✅ | 436.35 | 59.50 | - | NEW (6-12); QAT variant; 273 prompt, 55 gen |
| gemma4 (31B 4bit) | gemma-4-31b-4bit | ✅ | 82.56 | 15.73 | 77% | mlx-vlm: 20.30; 274 prompt, 100 gen |
| gemma4 (31B-it 4bit) | gemma-4-31b-it-4bit | ✅ | 85.71 | 18.64 | 94% | mlx-vlm: 19.78; 274 prompt, 100 gen |
| gemma4 (31B-it QAT) | gemma-4-31B-it-qat-4bit | ✅ | 85.40 | 15.07 | - | NEW (6-12); QAT variant; 277 prompt, 28 gen |
| gemma4 (26B A4B) | gemma-4-26b-a4b-it-4bit | ✅ | 273.98 | 65.84 | **108%** | mlx-vlm: 61.07; 277 prompt, 28 gen |
| gemma4 (26B A4B QAT) | gemma-4-26B-A4B-it-qat-4bit | ✅ | 276.09 | 66.58 | - | NEW (6-12); QAT variant; 277 prompt, 31 gen |
| llava 1.5 | llava-1.5-7b-4bit | ✅ | 751.21 | 103.26 | - | CLIP + MLP; Vicuna-7b; 583 prompt, 100 gen; mlx-vlm requires PyTorch |
| llava-interleave | llava-interleave-qwen-0.5b-bf16 | ✅ | 3741.25 | 265.91 | **118%** | mlx-vlm: 225.15; SigLIP + MLP; Qwen2-0.5b; 754 prompt, 36 gen |
| llava-next | llava-v1.6-mistral-7b-4bit | ✅ | 711.52 | 106.84 | 98% | mlx-vlm: 109.51; CLIP + MLP; Mistral; 590 prompt, 100 gen; mlx-vlm template error |
| llava-bunny | Bunny-Llama-3-8B-V-4bit | ✅ | 669.83 | 95.42 | - | mlx-vlm: FAIL; SigLIP + MLP; Llama3; 746 prompt, 37 gen |
| llama4 | Llama-4-Scout-17B-16E-Instruct-4bit | ✅ | 180.39 | 34.87 | - | mlx-vlm: FAIL; 230 prompt, 67 gen |
| aya-vision | aya-vision-8b | ✅ | 429.92 | 110.45 | **106%** | mlx-vlm: 103.74; SigLIP + SwiGLU; Cohere2; 176 prompt, 100 gen |
| paligemma | paligemma2-3b (6-bit) | ⚠️ | 1474.57 | 48.61 | 69% | mlx-vlm: 70.45; SigLIP + Linear; Gemma2; 1032 prompt, only 2 gen tokens |
| pixtral | pixtral-12b-4bit | ✅ | 438.27 | 59.81 | - | mlx-vlm: FAIL; Pixtral ViT; Mistral; 4102 prompt, 100 gen |
| mistral3 | mistral-small-3.1-24b-4bit | ✅ | 127.59 | 29.72 | - | mlx-vlm: FAIL; Pixtral ViT + PatchMerger; Mistral; 3032 prompt, 100 gen; mlx-vlm error |
| ministral3 | Ministral-3B-Instruct-4bit | ✅ | 523.04 | 124.01 | - | mlx-vlm: FAIL; Pixtral ViT; 3566 prompt, 100 gen |
| phi3.5-vision | Phi-3.5-vision-instruct-4bit | ✅ | 989.18 | 118.68 | **128%** | mlx-vlm: 92.53; CLIP + HD tiling; Phi3; 773 prompt, 19 gen |
| phi4mm | phi-4-multimodal-instruct (bf16) | ✅ | 571.90 | 25.42 | - | SigLIP + HD transform + AvgPool2d; Phi3; SuScaledRoPE + runtime LoRA; 2635 tokens; 12GB bf16; not in the 06-12 sweep |
| moondream3 | moondream3-preview-4bit | ⚠️ | 1.36 | 10.05 | - | SigLIP + MLP; image output garbled; only 63 tokens; not in the 06-12 sweep |
| minicpm-o | MiniCPM-o-2_6-4bit | ✅ | 33.67 | 70.80 | - | SigLIP + Resampler; Qwen3; 80 tokens; not in the 06-12 sweep |
| minicpm-v | MiniCPM-V-4.6-bf16 | ✅ | 390.14 | 176.14 | - | NEW (6-12); bf16; 32 prompt, 23 gen |
| molmo | Molmo-7B | ✅ | 572.64 | 80.44 | - | CLIP ViT + attention pooling + OLMo text; mlx-vlm baseline is a 1-token anomaly; 327 prompt, 100 gen |
| molmo2 | molmo2-4b | ✅ | 698.78 | 58.90 | 97% | mlx-vlm: 60.87; fast SDPA vision encoder; 438 prompt, 46 gen |
| internvl3 | InternVL3-1B | ✅ | 1719.55 | 224.39 | 85% | mlx-vlm: 264.40; InternViT + pixel-shuffle + Qwen2; 293 prompt, 8 gen |
| nemotron-omni | Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | ✅ | 258.59 | 68.37 | - | mlx-vlm: FAIL; NEW (5-19); Mamba2+Transformer+MoE+Parakeet audio; 6 gen |
| youtu-vl | youtu-vl-4b-instruct | ⚠️ | 395.31 | 23.84 | - | mlx-vlm: FAIL; NEW (5-19); only 1 gen token |
| qwen2-vl | Qwen2-VL-2B-Instruct-4bit | ✅ | 747.14 | 122.90 | - | Custom ViT + MRoPE; VLM image mode fixed; 12 gen |
| qwen2.5-vl | Qwen2.5-VL-3B-Instruct-4bit | ✅ | 596.70 | 96.07 | - | Windowed ViT + MRoPE; 91 prompt, 64 gen; mlx-vlm requires PyTorch |
| qwen3-vl | Qwen3-VL-2B-Instruct-4bit | ✅ | 795.43 | 158.27 | - | mlx-vlm: FAIL; DeepStack + vectorized MRoPE; 80 gen |
| qwen3-vl (4B) | Qwen3-VL-4B-Instruct-4bit | ✅ | 481.13 | 89.41 | - | mlx-vlm: FAIL; DeepStack + vectorized MRoPE; 38 gen |
| qwen3-vl (8B) | Qwen3-VL-8B-Instruct-4bit | ✅ | 299.55 | 62.07 | - | mlx-vlm: FAIL; DeepStack + vectorized MRoPE; 30 gen |
| qwen3-vl (32B) | Qwen3-VL-32B-Instruct-4bit | ✅ | 89.09 | 17.73 | - | mlx-vlm: FAIL; DeepStack + vectorized MRoPE; 59 gen |
| qwen3-vl-moe | Qwen3-VL-30B-A3B-Instruct-4bit | ✅ | 283.48 | 40.21 | - | mlx-vlm: FAIL; MoE (128 experts) + DeepStack; 34 gen |
| qwen3.5-vl (0.8B) | qwen3.5-0.8B-4bit | ✅ | 945.57 | 229.35 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet VLM; 69 prompt, 100 gen |
| qwen3.5-vl (2B) | qwen3.5-2B-4bit | ✅ | 531.13 | 171.03 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet VLM; 69 prompt, 43 gen |
| qwen3.5-vl (4B) | qwen3.5-4B-4bit | ✅ | 329.19 | 98.56 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet VLM; 69 prompt, 49 gen |
| qwen3.5-vl (9B 4bit) | qwen3.5-9B-4bit | ✅ | 198.72 | 73.27 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet VLM; 69 prompt, 100 gen |
| qwen3.5-vl (9B bf16) | qwen3.5-9B (bf16) | ✅ | 278.61 | 32.33 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet VLM; 69 prompt, 100 gen; bf16 |
| qwen3.5-vl (27B) | qwen3.5-27B-4bit | ✅ | 75.20 | 25.05 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet VLM; 69 prompt, 100 gen |
| qwen3.5-vl-moe | qwen3.5-35B-A3B-4bit | ✅ | 273.47 | 69.79 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet + MoE VLM; 69 prompt, 100 gen; gated delta decode RMSNorm fix |
| qwen3.6-vl-moe | qwen3.6-35B-A3B-4bit | ✅ | 273.49 | 65.69 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet + MoE VLM; 39 gen |
| molmo-point | - | ⏳ | - | - | - | Molmo-Point (point detection); implemented but no MLX model available |

**VLM test conditions**: Image: 224x224 PNG (test_image.png) unless noted. Prompt: "What is in this image?" Max tokens: 100. Prefill includes vision encoder + projector overhead. mlx-vlm baseline uses the `d85ca4d` dev checkout. mlxcel decode speed was measured with `mlxcel-bench-decode` (model load, warmup, and measured pass in one process). Models with unavailable or failed mlx-vlm runs are marked with "-" in the vs mlx-vlm column. Two oversize models (`deepseek-v3-4bit` 99GB, `qwen3-next-480b-4bit` 251GB) do not run on this 128GB host; `minimax-m2-3bit` (93GB) fits and runs on the text side since the 05-28 sweep. Two Gemma 3 VLM rows (gemma-3-4b-it-4bit, gemma3-4b-4bit) were measured with `--warmup-tokens 0` because the prepared 4D attention mask shape is single-use against the first prefill's KV cache offset. The three Gemma 3n VLM rows use the default warmup=20 path.

## Summary Statistics

| Status | Count |
|--------|-------|
| ✅ Pass | 133 (85 text + 48 VLM) |
| ⚠️ Partial | 6 (3 text + 3 VLM) |
| ❌ Fail | 2 (2 text + 0 VLM) |
| ⏳ Pending / Skipped (>65GB) | 15 (12 text pending + 1 VLM pending + 2 oversize skip) |

The 2026-06-12 sweep added seven newly benchmarked models (gemma-4-12b-it, four Gemma 4 QAT variants, diffusiongemma-26B-A4B-it, MiniCPM-V-4.6-bf16) and moved `minimax-m2-3bit` from oversize skip to pass.

## Performance Comparison

The detailed same-day decode comparison tables below are the authoritative
source for baseline comparisons. Decode remains the primary apples-to-apples
runtime comparison.

### Aggregate (decode, same-day baseline)

| Mode | Comparable pairs | Median mlxcel/baseline | >=90% parity | >= baseline | Range |
|------|-----------------:|-----------------------:|-------------:|------------:|------:|
| Text vs mlx-lm | 74 | 98% | 62/74 (84%) | 20/74 (27%) | 45%-111% |
| VLM vs mlx-vlm | 18 | 97% | 13/18 (72%) | 8/18 (44%) | 77%-128% |

### Run-over-run (2026-06-12 vs 2026-05-28 mlxcel)

The 2026-06-12 sweep is throughput-neutral within noise after the MLX a6ec712 bump: median -1.8% and mean -1.5% over 100 comparable text models, VLM median -1.1% over 41 models. The two apparent singletons (internvl3-1b +48%, molmo-7b -16% vs the 05-28 table) were attributed by rebuilding and re-measuring at three code states (current main, the pre-bump commit e099821, and the 05-28-era v0.1.2 with the old MLX pin): all three reproduce today's values (molmo-7b 66-69 tok/s, internvl3-1b 337-344 tok/s), so the 05-28 documented baselines for those two models were single-run measurement artifacts, not real run-over-run changes. The 06-12 values in this document are the reproducible ones.

### Representative decode wins

| Model | mlxcel | Baseline | vs baseline |
|-------|-------:|---------:|------------:|
| qwen2.5-0.5b-4bit | 342.79 | 315.48 | **109%** |
| phi-3.5-moe-4bit | 77.13 | 69.28 | **111%** |
| minicpm3-4b-4bit | 79.84 | 73.26 | **109%** |
| smollm-135m-4bit | 380.11 | 375.91 | **101%** |
| llava-interleave-qwen-0.5b-bf16 (VLM) | 265.91 | 225.15 | **118%** |
| gemma3n-e2b-4bit (VLM) | 73.35 | 59.57 | **123%** |
| gemma-4-e2b-it-4bit (VLM) | 105.83 | 97.19 | **109%** |
| phi-3.5-vision-4bit (VLM) | 118.68 | 92.53 | **128%** |

internvl3-1b has no text-path Python baseline; its 339.1 tok/s text decode is reproducible across the old and new MLX pins and the 05-28-era build, so the lower 05-28 table value was a measurement artifact rather than a since-fixed slowdown.

### Main optimization gaps

| Model | mlxcel | Baseline | vs baseline | Notes |
|-------|-------:|---------:|------------:|-------|
| falcon-mamba-7b-4bit | 40.77 | 91.04 | 45% | Chat template causes early EOS; only 2 generated tokens |
| qwen2.5-vl-3b-4bit (text path) | 98.92 | 160.42 | 62% | VLM wrapper text-only comparison |
| qwen2-vl-2b-4bit (text path) | 151.17 | 236.86 | 64% | VLM wrapper text-only comparison |
| gemma-4-31b-4bit (VLM) | 15.73 | 20.30 | 77% | large VLM path |
| gemma-3-4b-it-4bit (VLM) | 89.56 | 97.36 | 92% | measured with warmup=0; see VLM test conditions |

molmo-7b (text path, no Python baseline) reads 68.5 tok/s in the 06-12 sweep, lower than the 81.6 recorded on 05-28; re-measurement on the 05-28-era v0.1.2 build also gives 66.2 tok/s, so the old table value was a single-run artifact and there is no actual regression. Its VLM path is unchanged (80.44 tok/s).

## Performance vs mlx-lm / mlx-vlm baseline (mlxcel 2026-06-12 vs pinned 2026-05-19 reference)

Source CSVs (same M1 Ultra host; mlxcel 0.1.4 measured 2026-06-12 on the a6ec712 MLX pin; mlx-lm / mlx-vlm baselines from the pinned 2026-05-19 reference checkout with `PYLM_BENCH_MAX_GB=65`):

- mlxcel: `benchmarks/metal_m1ultra_2026-06-12.csv`
- mlxcel VLM: `benchmarks/metal_m1ultra_vlm_2026-06-12.csv`
- mlx-lm: `benchmarks/pylm_m1ultra_2026-05-19.csv` (mlx-lm 0.31.3 dev checkout in https://github.com/ml-explore/mlx-lm @ `df1d3f3`)
- mlx-vlm: `benchmarks/pylm_m1ultra_vlm_2026-05-19.csv` (mlx-vlm dev checkout in https://github.com/Blaizzy/mlx-vlm @ `d85ca4d`)

The mlx-lm / mlx-vlm baselines are the pinned 2026-05-19 reference checkout (the reference is fixed, so its decode on this host is stable); the mlxcel side is the 2026-06-12 full sweep. All sweeps use `--max-tokens 100` and the same `Hello, how are you today?` / `What is in this image?` prompts. `deepseek-v3-4bit` and `qwen3-next-480b-4bit` exceed the 128GB host on both sides; `minimax-m2-3bit` fits and runs on the mlxcel side (33.14 tok/s) but mlx-lm still fails it, so it stays outside the comparable set. Models added in the 06-12 sweep (gemma-4-12b-it, the Gemma 4 QAT variants, diffusiongemma, MiniCPM-V-4.6, plus internvl3-1b and molmo-7b on the text path) have no Python baseline and carry "-" in the comparison columns.

Numbers are decode tok/s. `mlxcel vs mlx-lm` is `mlxcel / mlx-lm` as a percentage; **bold** = mlxcel >= mlx-lm. `FAIL` cells are real load/runtime errors on that backend with this configuration. The mlx-lm checkout used here (`df1d3f3` — "Fix Gemma 4 sanitize() not stripping KV projections for shared layers" ml-explore/mlx-lm#1240) is newer than the M5 Max page's `ed1fca4`, so some FAIL categories differ.

### Aggregate (text)

- **Comparable text pairs**: 74
- **mlxcel >= mlx-lm**: 20 / 74 (27%)
- **mlxcel >= 90% parity**: 62 / 74 (84%)
- **Average mlxcel/mlx-lm**: 96% (median 98%, range 45%-111%)

### Aggregate (VLM, models with >=5 generated tokens both sides)

- **Comparable VLM pairs**: 18
- **mlxcel >= mlx-vlm**: 8 / 18 (44%)
- **mlxcel >= 90% parity**: 13 / 18 (72%)
- **Average mlxcel/mlx-vlm**: 101% (median 97%, range 77%-128%)

### Text decode (tok/s)

| Model | mlxcel | mlx-lm | mlxcel vs mlx-lm |
|-------|--------|--------|------------------|
| Meta-Llama-3.1-8B-Instruct-4bit | 107.80 | 109.84 | 98% |
| MiniCPM-V-4.6-bf16 | 195.50 | - | - |
| Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | 85.02 | FAIL | - |
| Qwen2.5-1.5B-4bit | 238.28 | 241.41 | 99% |
| Qwen2.5-1.5B-Instruct-4bit | 237.54 | 239.20 | 99% |
| Qwen2.5-7B-Instruct-4bit | 111.46 | 110.90 | **101%** |
| Qwen3.5-0.8B-OptiQ-4bit | FAIL | 265.86 | - |
| aya-expanse-8b-4bit | 107.34 | 112.74 | 95% |
| aya-vision-8b | 109.45 | FAIL | - |
| baichuan-m1-14b-4bit | 41.09 | 49.11 | 84% |
| bunny-llama3-8b-4bit | 103.60 | FAIL | - |
| command-r7b-4bit | 111.27 | 107.75 | **103%** |
| deepseek-coder-1.3b-4bit | 160.53 | FAIL | - |
| deepseek-r1-distill-7b-4bit | 110.80 | 111.34 | 100% |
| deepseek-v2-lite-4bit | 109.70 | 117.06 | 94% |
| deepseek-v3-4bit | - | FAIL | - |
| diffusiongemma-26B-A4B-it-4bit | 70.66 | - | - |
| ernie-4.5-0.3b-4bit | 505.06 | FAIL | - |
| exaone-3.5-2.4b-4bit | 195.13 | 194.65 | **100%** |
| exaone4-1.2b-4bit | 243.72 | FAIL | - |
| falcon-mamba-7b-4bit | 40.77 | 91.04 | 45% |
| gemma-2b-4bit | 191.07 | 207.78 | 92% |
| gemma-3-4b-it-4bit | 115.98 | 109.72 | **106%** |
| gemma-4-12b-it-4bit | 34.55 | - | - |
| gemma-4-26b-a4b-it-4bit | 71.00 | 72.52 | 98% |
| gemma-4-26B-A4B-it-qat-4bit | 71.65 | - | - |
| gemma-4-31b-4bit | 20.25 | 20.36 | 99% |
| gemma-4-31b-it-4bit | 19.34 | 20.23 | 96% |
| gemma-4-31B-it-qat-4bit | 15.48 | - | - |
| gemma-4-e2b-it-4bit | 117.90 | FAIL | - |
| gemma-4-e2b-it-8bit | 87.36 | FAIL | - |
| gemma-4-e2b-it-qat-4bit | 93.65 | - | - |
| gemma-4-e4b-it-4bit | 82.09 | FAIL | - |
| gemma-4-e4b-it-8bit | 57.80 | FAIL | - |
| gemma-4-e4b-it-qat-4bit | 62.84 | - | - |
| gemma2-2b-4bit | 165.06 | 153.50 | **108%** |
| gemma3-1b-4bit | 227.37 | 211.50 | **108%** |
| gemma3-4b-4bit | 115.40 | 109.48 | **105%** |
| gemma3n-e2b-4bit | 77.41 | FAIL | - |
| gemma3n-e4b-4bit | 59.91 | FAIL | - |
| gemma3n-e4b-bf16 | 34.65 | 39.02 | 89% |
| glm4-flash-4bit | 46.99 | 49.47 | 95% |
| gpt-oss-120b-4bit | 59.86 | 57.58 | **104%** |
| gpt-oss-20b-mxfp4 | 89.48 | 89.51 | 100% |
| hunyuan-1.8b-4bit | 184.13 | 200.59 | 92% |
| hunyuan-large-4bit | 44.30 | FAIL | - |
| internlm2-7b-4bit | 108.59 | 111.92 | 97% |
| internlm3-8b-4bit | 86.88 | FAIL | - |
| internvl3-1b | 339.15 | - | - |
| jamba-v0.1-4bit | 121.41 | 131.04 | 93% |
| llama-3.1-8b-4bit | 107.24 | 110.66 | 97% |
| llama-3.1-8b-bf16 | 35.32 | 35.32 | **100%** |
| llama-3.2-1b-4bit | 365.58 | 418.25 | 87% |
| llama-4-scout-17b-4bit | 36.20 | FAIL | - |
| llava-1.5-7b-4bit | 115.64 | FAIL | - |
| llava-interleave-qwen-0.5b-bf16 | 314.60 | FAIL | - |
| llava-next-mistral-7b-4bit | 113.88 | FAIL | - |
| mamba2-1.3b-4bit | 92.19 | FAIL | - |
| mimo-7b-4bit | 85.85 | 86.17 | 100% |
| minicpm-2b-4bit | 163.58 | 156.47 | **105%** |
| minicpm3-4b-4bit | 79.84 | 73.26 | **109%** |
| minimax-m2-3bit | 33.14 | FAIL | - |
| ministral-3b-4bit | 143.36 | 159.34 | 90% |
| mistral-small-3.1-24b-4bit | 31.80 | 31.97 | 99% |
| mixtral-8x7b-4bit | 54.20 | 54.91 | 99% |
| molmo-7b | 68.53 | - | - |
| molmo2-4b | 59.81 | FAIL | - |
| nemotron-h-30b-4bit | 90.92 | 93.34 | 97% |
| nemotron-nas-30b-4bit | 90.09 | 92.93 | 97% |
| olmo-1b-4bit | 209.88 | FAIL | - |
| olmo2-7b-4bit | 103.92 | 110.88 | 94% |
| olmo3-32b-4bit | 21.74 | 21.57 | **101%** |
| paligemma2-3b-6bit | 0.00 | FAIL | - |
| phi-2-4bit | 58.55 | FAIL | - |
| phi-3-mini-4bit | 168.94 | 171.36 | 99% |
| phi-3.5-mini-4bit | 164.19 | 166.30 | 99% |
| phi-3.5-moe-4bit | 77.13 | 69.28 | **111%** |
| phi-3.5-vision-4bit | 164.45 | FAIL | - |
| phi-4-4bit | 57.98 | 58.68 | 99% |
| pixtral-12b-4bit | 69.68 | 69.49 | **100%** |
| qwen1.5-moe-a2.7b-4bit | 144.05 | 144.98 | 99% |
| qwen2-vl-2b-4bit | 151.17 | 236.86 | 64% |
| qwen2.5-0.5b-4bit | 342.79 | 315.48 | **109%** |
| qwen2.5-7b-4bit | 110.41 | 111.38 | 99% |
| qwen2.5-7b-8bit | 69.96 | 70.46 | 99% |
| qwen2.5-vl-3b-4bit | 98.92 | 160.42 | 62% |
| qwen3-0.6b-4bit | 290.40 | 299.61 | 97% |
| qwen3-1.7b-4bit | 188.67 | 221.37 | 85% |
| qwen3-30b-a3b-4bit | 70.27 | 70.18 | **100%** |
| qwen3-4b-4bit | 119.56 | 123.92 | 96% |
| qwen3-8b-4bit | 80.42 | 84.54 | 95% |
| qwen3-moe-4bit | 69.55 | 69.67 | 100% |
| qwen3-next-480b-4bit | - | FAIL | - |
| qwen3-vl-2b-4bit | 211.01 | 222.67 | 95% |
| qwen3-vl-30b-a3b-4bit | 69.15 | 70.04 | 99% |
| qwen3-vl-32b-4bit | 21.11 | 21.99 | 96% |
| qwen3-vl-4b-4bit | 117.74 | 124.02 | 95% |
| qwen3-vl-8b-4bit | 80.48 | 84.46 | 95% |
| qwen3.5-0.8b-4bit | 239.31 | 269.52 | 89% |
| qwen3.5-27b-4bit | 24.27 | 25.93 | 94% |
| qwen3.5-2b-4bit | 174.85 | 211.68 | 83% |
| qwen3.5-35b-a3b-4bit | 70.05 | 76.44 | 92% |
| qwen3.5-4b-4bit | 99.69 | 115.60 | 86% |
| qwen3.5-9b-4bit | 71.55 | 81.27 | 88% |
| qwen3.5-9b-bf16 | 31.58 | 34.22 | 92% |
| qwen3.6-35b-a3b-4bit | 67.31 | 73.18 | 92% |
| smollm-135m-4bit | 380.11 | 375.91 | **101%** |
| smollm3-3b-4bit | 135.55 | 141.66 | 96% |
| solar-open-100b-4bit | 36.24 | 35.69 | **102%** |
| stablelm-1.6b-4bit | 282.33 | 280.65 | **101%** |
| starcoder2-3b-4bit | 170.18 | 166.17 | **102%** |
| youtu-vl-4b-instruct | 0.00 | FAIL | - |

### VLM decode (tok/s)

| Model | mlxcel | mlx-vlm | mlxcel vs mlx-vlm |
|-------|--------|--------|------------------|
| MiniCPM-V-4.6-bf16 | 176.14 | - | - |
| Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | 68.37 | FAIL | - |
| aya-vision-8b | 110.45 | 103.74 | **106%** |
| bunny-llama3-8b-4bit | 95.42 | FAIL | - |
| deepseek-v3-4bit | - | FAIL | - |
| gemma-3-4b-it-4bit | 89.56 | 97.36 | 92% |
| gemma-4-12b-it-4bit | 32.50 | - | - |
| gemma-4-26b-a4b-it-4bit | 65.84 | 61.07 | **108%** |
| gemma-4-26B-A4B-it-qat-4bit | 66.58 | - | - |
| gemma-4-31b-4bit | 15.73 | 20.30 | 77% |
| gemma-4-31b-it-4bit | 18.64 | 19.78 | 94% |
| gemma-4-31B-it-qat-4bit | 15.07 | - | - |
| gemma-4-e2b-it-4bit | 105.83 | 97.19 | **109%** |
| gemma-4-e2b-it-8bit | 81.26 | 91.06 | 89% |
| gemma-4-e2b-it-qat-4bit | 82.64 | - | - |
| gemma-4-e4b-it-4bit | 74.67 | 70.34 | **106%** |
| gemma-4-e4b-it-8bit | 55.28 | 63.25 | 87% |
| gemma-4-e4b-it-qat-4bit | 59.50 | - | - |
| gemma3-4b-4bit | 88.91 | 93.79 | 95% |
| gemma3n-e2b-4bit | 73.35 | 59.57 | **123%** |
| gemma3n-e4b-4bit | 57.30 | 50.00 | **115%** |
| gemma3n-e4b-bf16 | 32.07 | 36.18 | 89% |
| internvl3-1b | 224.39 | 264.40 | 85% |
| llama-4-scout-17b-4bit | 34.87 | FAIL | - |
| llava-1.5-7b-4bit | 103.26 | FAIL | - |
| llava-interleave-qwen-0.5b-bf16 | 265.91 | 225.15 | **118%** |
| llava-next-mistral-7b-4bit | 106.84 | 109.51 | 98% |
| minimax-m2-3bit | - | FAIL | - |
| ministral-3b-4bit | 124.01 | FAIL | - |
| mistral-small-3.1-24b-4bit | 29.72 | FAIL | - |
| molmo-7b | 80.44 | 38399.52 (anomalous) | - |
| molmo2-4b | 58.90 | 60.87 | 97% |
| paligemma2-3b-6bit | 48.61 | 70.45 | 69% |
| phi-3.5-vision-4bit | 118.68 | 92.53 | **128%** |
| pixtral-12b-4bit | 59.81 | FAIL | - |
| qwen2-vl-2b-4bit | 122.90 | FAIL | - |
| qwen2.5-vl-3b-4bit | 96.07 | FAIL | - |
| qwen3-next-480b-4bit | - | FAIL | - |
| qwen3-vl-2b-4bit | 158.27 | FAIL | - |
| qwen3-vl-30b-a3b-4bit | 40.21 | FAIL | - |
| qwen3-vl-32b-4bit | 17.73 | FAIL | - |
| qwen3-vl-4b-4bit | 89.41 | FAIL | - |
| qwen3-vl-8b-4bit | 62.07 | FAIL | - |
| qwen3.5-0.8b-4bit | 229.35 | FAIL | - |
| qwen3.5-27b-4bit | 25.05 | FAIL | - |
| qwen3.5-2b-4bit | 171.03 | FAIL | - |
| qwen3.5-35b-a3b-4bit | 69.79 | FAIL | - |
| qwen3.5-4b-4bit | 98.56 | FAIL | - |
| qwen3.5-9b-4bit | 73.27 | FAIL | - |
| qwen3.5-9b-bf16 | 32.33 | FAIL | - |
| qwen3.6-35b-a3b-4bit | 65.69 | FAIL | - |
| youtu-vl-4b-instruct | 23.84 | FAIL | - |

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
| gemma-4-31B-it-assistant-bf16 / gemma-4-12B-it-assistant-4bit | Drafter checkpoint, not a standalone inference model | Low |
| MiniCPM-V-4.6-mxfp4 | Warmup failure on the mxfp4 variant (bf16 variant passes) | Medium |
| docling-layout-heron-mlx-bf16 | Layout-analysis checkpoint, not a generative LM; fails bench harness | Low |
| falcon-mamba | Chat template causes early EOS (only 2 tokens); decode now 40.77 tok/s | Medium |
| paligemma | Only 2 VLM gen tokens; decode is not comparable despite 48.61 tok/s measured | High |
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
https://github.com/TheTom/turboquant_plus. The fused Sparse-V Metal kernel that lands
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
