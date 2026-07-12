# Model Compatibility & Performance Tests (M1 Ultra)

Compatibility and performance testing for mlxcel models on **Mac Studio M1 Ultra 128GB**, with comparison against Python mlx-lm / mlx-vlm.

> **2026-06-17 fused decode-MoE update.** After this 0.2.1 sweep, the fused decode-MoE kernel was wired into more MoE families (epic #307: qwen2_moe, lfm2, qwen3_vl_moe, mixtral, phi-3.5-moe, olmoe). On M1 Ultra the small-expert families gain on decode (qwen3-vl-30b-a3b text path 69 to 82 tok/s, +18.8%; lfm2-8b-a1b +3.4%; qwen1.5-moe +2.2%), while large-expert mixtral and phi-3.5-moe stay on gather_qmm via the `MLXCEL_FUSED_MOE_MAX_DFF` guard (decode unchanged) and olmoe is perf-neutral. The MoE rows in this dated sweep predate the wiring; the post-wiring decode numbers are in `benchmarks/metal_m1ultra_2026-06-17_fused_moe.csv` and [Fused decode-MoE kernel](fused-moe-decode-kernel-design.md).

## Test Environment

| Item | Value |
|------|-------|
| **Hardware** | Mac Studio M1 Ultra, 128GB RAM |
| **OS** | macOS 26.4 (Tahoe) |
| **mlxcel version** | 0.4.0-rc.1 |
| **MLX version** | pin commit 57c66cac (mlxcel-core three-place pin; superseded the a6ec712 pin) |
| **Bench harness** | `mlxcel-bench-decode` (model load, warmup, and measured pass in one process) |
| **mlx-lm baseline** | 0.31.3 (dev checkout https://github.com/ml-explore/mlx-lm @ `df1d3f3` — "Fix Gemma 4 sanitize() not stripping KV projections for shared layers" ml-explore/mlx-lm#1240) |
| **mlx-vlm baseline** | dev checkout https://github.com/Blaizzy/mlx-vlm @ `d85ca4d` — "Compatibility bridge for non-VL models" Blaizzy/mlx-vlm#1181 |
| **Test Prompt** | "Hello, how are you today?" (text) / "What is in this image?" (VLM) |
| **Max Tokens** | 100 (measured pass); 20 (warmup pass, same process) |
| **Test Date** | 2026-05-19 full sweep (baseline); 2026-05-28 full text + VLM re-benchmark on mlxcel 0.1.0 (`--cooldown 0`); 2026-06-12 full text + VLM re-benchmark on mlxcel 0.1.4 (MLX pin a6ec712, post issue #222 bump); 2026-06-15 full text + VLM re-benchmark on mlxcel 0.2.1 (post #289 fix); 2026-07-06 full text + VLM re-benchmark on mlxcel 0.3.3 (post 5-family VLM port batch #660-#664 and fix batch #666-#668, #671); 2026-07-12 full text + VLM re-benchmark on mlxcel 0.4.0-rc.1 (MLX pin 57c66cac; version-change full sweep, `--cooldown 30`; a `--cooldown 0` pass of the same build is kept as `_cooldown0` for the thermal-offset comparison) |
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
| llama3 | Llama-3.2-1B-Instruct-4bit | ✅ | 1833.15 | 421.40 | **101%** | mlx-lm: 418.25; only 48 tokens |
| llama3 (8B bf16) | Llama-3.1-8B-Instruct (bf16) | ✅ | 394.11 | 34.01 | 96% | mlx-lm: 35.32; non-quantized |
| llama3.1 | Llama-3.1-8B-Instruct-4bit | ✅ | 475.43 | 106.63 | 96% | mlx-lm: 110.66; only 54 tokens |
| llama4 | Llama-4-Scout-17B-16E-Instruct-4bit | ⚠️ | 117.78 | 35.71 | - | mlx-lm: FAIL; long outputs repetitive |
| qwen2 | Qwen2.5-0.5B-Instruct-4bit | ✅ | 1090.09 | 381.94 | **121%** | mlx-lm: 315.48 |
| qwen2 (7B 4bit) | Qwen2.5-7B-Instruct-4bit | ✅ | 296.01 | 108.86 | 98% | mlx-lm: 110.90 |
| qwen2 (7B 8bit) | Qwen2.5-7B-Instruct-8bit | ✅ | 292.95 | 67.94 | 96% | mlx-lm: 70.46; 8-bit quantized |
| qwen3 | Qwen3-0.6B-4bit | ✅ | 587.00 | 228.37 | 76% | mlx-lm: 299.61 |
| qwen3 (1.7B) | Qwen3-1.7B-4bit | ✅ | 362.89 | 211.48 | 96% | mlx-lm: 221.37 |
| qwen3 (4B) | Qwen3-4B-4bit | ✅ | 240.31 | 121.52 | 98% | mlx-lm: 123.92 |
| qwen3 (8B) | Qwen3-8B-4bit | ✅ | 158.65 | 79.64 | 94% | mlx-lm: 84.54 |
| qwen3_5 (0.8B) | Qwen3.5-0.8B-4bit | ✅ | 555.74 | 250.40 | 93% | mlx-lm: 269.52; Hybrid GatedDeltaNet |
| qwen3_5 (2B) | Qwen3.5-2B-4bit | ✅ | 381.45 | 191.85 | 91% | mlx-lm: 211.68; Hybrid GatedDeltaNet; only 36 tokens |
| qwen3_5 (4B) | Qwen3.5-4B-4bit | ✅ | 240.86 | 108.04 | 93% | mlx-lm: 115.60; Hybrid GatedDeltaNet; only 36 tokens |
| qwen3_5 (9B 4bit) | qwen3.5-9B-4bit | ✅ | 150.18 | 70.84 | 87% | mlx-lm: 81.27; Hybrid GatedDeltaNet; only 29 tokens |
| qwen3_5 (9B bf16) | qwen3.5-9B (bf16) | ✅ | 144.44 | 30.87 | 90% | mlx-lm: 34.22; bf16, not quantized; Hybrid GatedDeltaNet (compiled fused kernel) |
| qwen3_5 (27B) | qwen3.5-27B-4bit | ✅ | 51.65 | 23.91 | 92% | mlx-lm: 25.93; Hybrid Transformer+GatedDeltaNet; VLM wrapper format |
| qwen3_6 | qwen3.6-35B-A3B-4bit | ✅ | 232.36 | 79.99 | **109%** | mlx-lm: 73.18; MoE architecture; 100 tokens |
| qwen3_next | qwen3-next-480B-4bit | ⏳ | - | SKIP | - | Qwen3Next 480B architecture; >65GB skipped on 128GB host |
| qwen2 (1.5B) | Qwen2.5-1.5B-Instruct-4bit | ✅ | 774.05 | 248.14 | **104%** | mlx-lm: 239.20; 100 tokens |
| qwen2 (1.5B base) | Qwen2.5-1.5B-4bit | ✅ | 653.96 | 243.94 | **101%** | mlx-lm: 241.41; base variant; 100 tokens |
| phi | phi-2-hf-4bit-mlx | ✅ | 137.51 | 58.05 | - | mlx-lm fails to load; only 1 token (likely EOS) |
| phi3 | Phi-3-mini-4k-instruct-4bit | ✅ | 190.71 | 164.49 | 96% | mlx-lm: 171.36; only 25 tokens |
| phi3small | Phi-3.5-mini-instruct-4bit | ✅ | 191.89 | 162.64 | 98% | mlx-lm: 166.30; only 40 tokens |
| phi4 | Phi-4-4bit | ✅ | 110.32 | 57.53 | 98% | mlx-lm: 58.68 |
| smollm3 | SmolLM-135M-Instruct-4bit | ✅ | 474.49 | 418.85 | **111%** | mlx-lm: 375.91 |
| smollm3 (3B) | SmolLM3-3B-4bit | ✅ | 570.77 | 131.45 | 93% | mlx-lm: 141.66 |
| stablelm | stablelm-2-1_6b-chat-4bit | ✅ | 670.04 | 263.88 | 94% | mlx-lm: 280.65; only 59 tokens |
| starcoder2 | starcoder2-3b-4bit | ✅ | 177.97 | 163.29 | 98% | mlx-lm: 166.17 |
| olmo | OLMo-1B-hf-4bit | ✅ | 187.19 | 203.69 | - | mlx-lm: FAIL |
| olmo2 | OLMo2-7B-4bit | ✅ | 276.49 | 99.85 | 90% | mlx-lm: 110.88; only 27 tokens |
| olmo3 | OLMo3.1-32B-4bit | ✅ | 80.44 | 21.39 | 99% | mlx-lm: 21.57 |
| minicpm | MiniCPM-2B-sft-bf16-4bit | ✅ | 293.70 | 149.11 | 95% | mlx-lm: 156.47 |
| mimo | MiMo-7B-RL-4bit | ✅ | 227.13 | 84.55 | 98% | mlx-lm: 86.17 |

## Gemma Family

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| gemma | gemma-2b-it-4bit | ✅ | 326.84 | 190.96 | 92% | mlx-lm: 207.78; only 49 tokens |
| gemma2 | gemma-2-2b-it-4bit | ✅ | 334.12 | 172.42 | **112%** | mlx-lm: 153.50; only 18 tokens |
| gemma3 | gemma-3-1b-it-4bit | ✅ | 440.50 | 225.24 | **106%** | mlx-lm: 211.50; only 34 tokens |
| gemma3 (4B) | gemma-3-4b-it-4bit | ✅ | 194.64 | 117.93 | **108%** | mlx-lm: 109.48; only 86 tokens |
| gemma4 (12B) | gemma-4-12b-it-4bit | ✅ | 108.13 | 37.01 | - | NEW (6-12); no mlx-lm baseline |
| gemma4 (31B) | gemma-4-31b-4bit | ✅ | 23.28 | 19.99 | 98% | mlx-lm: 20.36 |
| gemma4 (31B-it) | gemma-4-31b-it-4bit | ✅ | 49.71 | 19.17 | 95% | mlx-lm: 20.23; instruction-tuned variant |
| gemma4 (31B-it QAT) | gemma-4-31B-it-qat-4bit | ✅ | 45.82 | 15.67 | - | NEW (6-12); QAT variant; no mlx-lm baseline |
| gemma4 (26B A4B) | gemma-4-26b-a4b-it-4bit | ✅ | 179.86 | 79.55 | **110%** | mlx-lm: 72.52; only 26 tokens |
| gemma4 (26B A4B QAT) | gemma-4-26B-A4B-it-qat-4bit | ✅ | 169.00 | 77.56 | - | NEW (6-12); QAT variant; no mlx-lm baseline |
| gemma4 (diffusion 26B A4B) | diffusiongemma-26B-A4B-it-4bit | ✅ | 176.32 | 67.87 | - | block-diffusion; AR-equivalent decode of the same backbone; current upstream export is mixed-precision (8-bit attn/mlp/embed + 4-bit default, per-layer quant), supported via per-tensor bit inference in the quantized embedding (#291); diffusion generation verified correct; see docs/block-diffusion.md |
| gemma4 (E2B 4bit) | gemma-4-e2b-it-4bit | ✅ | 245.93 | 126.43 | - | mlx-lm: FAIL; only 34 tokens |
| gemma4 (E2B 8bit) | gemma-4-e2b-it-8bit | ✅ | 234.22 | 103.77 | - | mlx-lm: FAIL; only 38 tokens |
| gemma4 (E2B QAT) | gemma-4-e2b-it-qat-4bit | ✅ | 179.70 | 113.37 | - | NEW (6-12); QAT variant; no mlx-lm baseline |
| gemma4 (E4B 4bit) | gemma-4-e4b-it-4bit | ✅ | 176.73 | 85.62 | - | mlx-lm: FAIL; only 25 tokens |
| gemma4 (E4B 8bit) | gemma-4-e4b-it-8bit | ✅ | 157.38 | 66.17 | - | mlx-lm: FAIL; only 39 tokens |
| gemma4 (E4B QAT) | gemma-4-e4b-it-qat-4bit | ✅ | 123.11 | 70.27 | - | NEW (6-12); QAT variant; no mlx-lm baseline |
| gemma3n | gemma-3n-E2B-it-4bit | ✅ | 212.37 | 88.33 | - | mlx-lm: FAIL; only 69 tokens |
| gemma3n (E4B) | gemma-3n-E4B-it-4bit | ✅ | 168.10 | 66.75 | - | mlx-lm: FAIL; only 74 tokens |
| gemma3n (E4B bf16) | gemma-3n-E4B-it (bf16) | ✅ | 169.60 | 34.77 | 89% | mlx-lm: 39.02; bf16; AltUp/MLP decode graph scheduling |
| recurrent_gemma | - | ⏳ | - | - | - | Griffin SSM+attention hybrid |

## EXAONE

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| exaone | EXAONE-3.5-2.4B-Instruct-4bit | ✅ | 706.56 | 199.11 | **102%** | mlx-lm: 194.65 |
| exaone4 | exaone-4.0-1.2b-4bit | ✅ | 436.10 | 228.05 | - | mlx-lm: FAIL; only 18 tokens |
| exaone_moe | - | ⏳ | - | - | - | |

## Cohere Command R

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| cohere | c4ai-command-r7b-12-2024-4bit | ✅ | 93.91 | 109.14 | **101%** | mlx-lm: 107.75 |
| cohere2 | aya-expanse-8b-4bit | ✅ | 98.22 | 105.68 | 94% | mlx-lm: 112.74 |

## MoE (Mixture of Experts)

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| minimax | MiniMax-M2-3bit | ✅ | 80.97 | 33.27 | - | mlx-lm: FAIL on this host; 93GB, fits on the 128GB host since the 05-28 run |
| mixtral | Mixtral-8x7B-Instruct-v0.1-4bit | ✅ | 74.78 | 51.81 | 94% | mlx-lm: 54.91; only 73 tokens |
| qwen2_moe | Qwen1.5-MoE-A2.7B-Chat-4bit | ✅ | 383.32 | 149.48 | **103%** | mlx-lm: 144.98; only 43 tokens |
| qwen3_moe | Qwen3-30B-A3B-4bit | ✅ | 186.58 | 84.46 | **120%** | mlx-lm: 70.18 |
| qwen3_5_moe | qwen3.5-35B-A3B-4bit | ✅ | 223.03 | 81.44 | **107%** | mlx-lm: 76.44; Hybrid GatedDeltaNet + MoE (256 experts); only 34 tokens |
| phimoe | Phi-3.5-MoE-instruct-4bit | ✅ | 102.95 | 75.37 | **109%** | mlx-lm: 69.28 |
| solar_open | Solar-Open-100B-4bit | ✅ | 72.65 | 35.02 | 98% | mlx-lm: 35.69; 128 experts, top-8; layer-eval skip; 54GB |
| solar_open (int4) | Solar-Open-100B-int4 | ✅ | - | 11.55 | - | mlx-lm: fails to load; 128 experts, top-8; int4 quantization; 54GB; not in the 06-12 sweep |
| olmoe | OLMoE-1B-7B-0125-Instruct-4bit | ⏳ | - | - | - | router scoring fix (#318): full softmax over all experts then gather, not top-k-only softmax; greedy temp-0 output now coherent; perf sweep pending |
| gpt_oss (20B) | gpt-oss-20b-MXFP4-Q4 | ✅ | 278.38 | 89.25 | **100%** | mlx-lm: 89.51; MXFP4 quantization; 32 experts; bf16 decode fix |
| gpt_oss (120B) | gpt-oss-120b-4bit | ✅ | 158.57 | 59.29 | **103%** | mlx-lm: 57.58; 128 experts, top-4; 61GB model; bf16 decode fix |

## DeepSeek Family

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| deepseek | deepseek-coder-1.3b-instruct-4bit | ✅ | 1405.39 | 159.69 | - | mlx-lm: FAIL |
| deepseek_v2 | DeepSeek-V2-Lite-Chat-4bit | ✅ | 249.23 | 103.34 | 88% | mlx-lm: 117.06; only 18 tokens |
| deepseek_r1 | DeepSeek-R1-Distill-Qwen-7B-4bit | ✅ | 169.73 | 109.16 | 98% | mlx-lm: 111.34 |
| deepseek_v3 | deepseek-v3-4bit | ⏳ | - | SKIP | - | MoE + MLA; >65GB skipped on 128GB host (99GB) |
| deepseek_v32 | - | ⏳ | - | - | - | |

## MLA (Multi-head Latent Attention)

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| minicpm3 | MiniCPM3-4B-4bit | ✅ | 234.98 | 87.22 | **119%** | mlx-lm: 73.26 |

## Nemotron Family

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| nemotron_h | Nemotron-H-30B-4bit | ✅ | 173.69 | 91.75 | 98% | mlx-lm: 93.34; Hybrid Mamba2+Transformer+MoE; SSM Metal kernel |
| nemotron_nas | Nemotron-NAS-30B-A3B-4bit | ✅ | 174.40 | 90.37 | 97% | mlx-lm: 92.93; Hybrid Mamba2+Transformer+MoE |
| nemotron_h_nano_omni | Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | ✅ | 166.28 | 85.39 | - | mlx-lm: FAIL; NEW (5-19); Mamba2+Transformer+MoE+Parakeet audio; 100 tokens |

## SSM / Mamba Models

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| mamba | Falcon-Mamba-7B-4bit | ⚠️ | 112.16 | 43.60 | 48% | mlx-lm: 91.04; only 2 tokens due to chat template EOS |
| mamba2 | mamba2-1.3b-4bit | ✅ | 198.48 | 115.03 | - | mlx-lm: FAIL |
| jamba | Jamba-v0.1-4bit | ✅ | 335.20 | 129.36 | 99% | mlx-lm: 131.04; only 76 tokens |
| rwkv7 | - | ⏳ | - | - | - | RWKV v7 linear attention |

## Chinese / Asian Language Models

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| baichuan | Baichuan-M1-14B-Instruct-4bit | ✅ | 56.50 | 39.43 | 80% | mlx-lm: 49.11; only 39 tokens |
| glm4 | GLM-4-Flash-4bit | ✅ | 119.68 | 48.84 | 99% | mlx-lm: 49.47; Only 18 tokens |
| glm4_moe | - | ⏳ | - | - | - | |
| glm4_moe_lite | GLM-4.7-Flash-4bit | ✅ | - | 31.54 | 76% | mlx-lm: 41.55; only 18 tokens |
| glm5 | GLM-5-4bit | ❌ | - | FAIL | - | warmup failure (persistent) |
| internlm2 | InternLM2-7B-4bit | ✅ | 211.39 | 105.12 | 94% | mlx-lm: 111.92 |
| internlm3 | internlm3-8b-instruct-4bit | ✅ | 294.30 | 83.79 | - | mlx-lm: FAIL |
| ernie4_5 | ERNIE-4.5-0.3B-Instruct-4bit | ✅ | 1091.49 | 522.31 | - | mlx-lm: FAIL |
| ernie4_5_moe | - | ⏳ | - | - | - | |
| hunyuan_moe | Hunyuan-Large-Instruct-4bit | ✅ | 77.40 | 44.07 | - | mlx-lm: FAIL |
| hunyuan_moe_13b | HunYuan-MoE-A13B-Instruct (bf16) | ❌ | - | FAIL | - | mlx-lm: fails to load; Tiktoken tokenizer; bf16; warmup failure |
| hunyuan_v1_dense | Hunyuan-1.8B-Instruct-4bit | ✅ | 285.00 | 180.89 | 90% | mlx-lm: 200.59; only 41 tokens |
| kimi_linear | - | ⏳ | - | - | - | Kimi linear attention (Moonshot) |
| step3p5 | - | ⏳ | - | - | - | Step 3.5 (StepFun) |

## Other Models

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| ministral3 | Ministral-3B-Instruct-4bit | ✅ | 864.25 | 143.47 | 90% | mlx-lm: 159.34; VLM wrapper; text-only mode; only 34 tokens |
| mistral4 | - | ⏳ | - | - | - | MLA + MoE; implemented but no MLX model available |
| moondream3 | moondream3-preview-4bit | ⚠️ | - | 8.45 | - | mlx-lm: fails to load; text-only test; SigLIP + MLP; image output garbled; only 14 tokens |
| longcat_flash | - | ⏳ | - | - | - | |
| longcat_flash_ngram | - | ⏳ | - | - | - | |
| mistral_small | mistral-small-3.1-24b-4bit | ✅ | 127.64 | 28.73 | 90% | mlx-lm: 31.97; text-only mode |

## Vision-Language Models (VLM)

| Model | Test Model | Status | Prefill | Decode | vs mlx-vlm | Notes |
|-------|------------|--------|---------|--------|------------|-------|
| gemma3 | gemma-3-4b-it-4bit | ✅ | 234.85 | 84.79 | 90% | mlx-vlm: 93.79; SigLIP + AvgPool; 275 prompt, 16 gen |
| gemma3n (E2B) | gemma-3n-E2B-it-4bit | ✅ | 693.35 | 80.49 | **135%** | mlx-vlm: 59.57; MobileNetV5 + MSFA; 273 prompt, 29 gen |
| gemma3n (E4B bf16) | gemma-3n-E4B-it (bf16) | ✅ | 629.97 | 31.92 | 88% | mlx-vlm: 36.18; MobileNetV5 + MSFA; bf16 prefill path retune; bf16; 273 prompt, 24 gen |
| gemma3n (E4B 4bit) | gemma-3n-E4B-it-4bit | ✅ | 428.35 | 60.89 | **122%** | mlx-vlm: 50.00; 273 prompt, 33 gen |
| gemma4 (12B) | gemma-4-12b-it-4bit | ✅ | 239.08 | 34.12 | - | NEW (6-12); no mlx-vlm baseline; 277 prompt, 25 gen |
| gemma4 (E2B 4bit) | gemma-4-e2b-it-4bit | ✅ | 670.29 | 108.25 | **111%** | mlx-vlm: 97.19; 277 prompt, 100 gen |
| gemma4 (E2B 8bit) | gemma-4-e2b-it-8bit | ✅ | 659.41 | 92.26 | **101%** | mlx-vlm: 91.06; 277 prompt, 100 gen |
| gemma4 (E2B QAT) | gemma-4-e2b-it-qat-4bit | ✅ | 651.71 | 94.05 | - | NEW (6-12); QAT variant; 273 prompt, 42 gen |
| gemma4 (E4B 4bit) | gemma-4-e4b-it-4bit | ✅ | 428.64 | 74.92 | **107%** | mlx-vlm: 70.34; 277 prompt, 92 gen |
| gemma4 (E4B 8bit) | gemma-4-e4b-it-8bit | ✅ | 423.25 | 59.25 | 94% | mlx-vlm: 63.25; 277 prompt, 64 gen |
| gemma4 (E4B QAT) | gemma-4-e4b-it-qat-4bit | ✅ | 418.27 | 63.73 | - | NEW (6-12); QAT variant; 273 prompt, 55 gen |
| gemma4 (31B 4bit) | gemma-4-31b-4bit | ✅ | 82.42 | 14.65 | 72% | mlx-vlm: 20.30; 265 prompt, 5 gen |
| gemma4 (31B-it 4bit) | gemma-4-31b-it-4bit | ✅ | 84.18 | 18.26 | 92% | mlx-vlm: 19.78; 277 prompt, 24 gen |
| gemma4 (31B-it QAT) | gemma-4-31B-it-qat-4bit | ✅ | 83.01 | 14.80 | - | NEW (6-12); QAT variant; 277 prompt, 28 gen |
| gemma4 (26B A4B) | gemma-4-26b-a4b-it-4bit | ✅ | 268.66 | 69.85 | **114%** | mlx-vlm: 61.07; 277 prompt, 27 gen |
| gemma4 (26B A4B QAT) | gemma-4-26B-A4B-it-qat-4bit | ✅ | 269.68 | 68.95 | - | NEW (6-12); QAT variant; 277 prompt, 31 gen |
| llava 1.5 | llava-1.5-7b-4bit | ✅ | 717.26 | 98.51 | - | CLIP + MLP; Vicuna-7b; 594 prompt, 100 gen; mlx-vlm requires PyTorch |
| llava-interleave | llava-interleave-qwen-0.5b-bf16 | ✅ | 3571.55 | 253.94 | **113%** | mlx-vlm: 225.15; SigLIP + MLP; Qwen2-0.5b; 754 prompt, 36 gen |
| llava-next | llava-v1.6-mistral-7b-4bit | ✅ | 664.64 | 101.62 | 93% | mlx-vlm: 109.51; CLIP + MLP; Mistral; 590 prompt, 100 gen; mlx-vlm template error |
| llava-bunny | Bunny-Llama-3-8B-V-4bit | ✅ | 623.91 | 92.50 | - | mlx-vlm: FAIL; SigLIP + MLP; Llama3; 746 prompt, 37 gen |
| llama4 | Llama-4-Scout-17B-16E-Instruct-4bit | ✅ | 171.40 | 33.21 | - | mlx-vlm: FAIL; 230 prompt, 67 gen |
| aya-vision | aya-vision-8b | ✅ | 625.59 | 97.16 | 94% | mlx-vlm: 103.74; SigLIP + SwiGLU; Cohere2; 735 prompt, 29 gen |
| paligemma | paligemma2-3b (6-bit) | ⚠️ | 1430.91 | 59.94 | 85% | mlx-vlm: 70.45; SigLIP + Linear; Gemma2; 1032 prompt, only 2 gen tokens |
| pixtral | pixtral-12b-4bit | ✅ | 430.62 | 58.17 | - | mlx-vlm: FAIL; Pixtral ViT; Mistral; 4099 prompt, 100 gen |
| mistral3 | mistral-small-3.1-24b-4bit | ✅ | 124.77 | 27.77 | - | mlx-vlm: FAIL; Pixtral ViT + PatchMerger; Mistral; 3206 prompt, 29 gen; mlx-vlm error |
| ministral3 | Ministral-3B-Instruct-4bit | ✅ | 508.51 | 120.50 | - | mlx-vlm: FAIL; Pixtral ViT; 3566 prompt, 100 gen |
| phi3.5-vision | Phi-3.5-vision-instruct-4bit | ✅ | 924.88 | 118.26 | **128%** | mlx-vlm: 92.53; CLIP + HD tiling; Phi3; 773 prompt, 19 gen |
| phi4mm | phi-4-multimodal-instruct (bf16) | ✅ | 571.90 | 25.42 | - | SigLIP + HD transform + AvgPool2d; Phi3; SuScaledRoPE + runtime LoRA; 2635 tokens; 12GB bf16; not in the 06-12 sweep |
| moondream3 | moondream3-preview-4bit | ⚠️ | 1.36 | 10.05 | - | SigLIP + MLP; image output garbled; only 63 tokens; not in the 06-12 sweep |
| minicpm-o | MiniCPM-o-2_6-4bit | ✅ | 33.67 | 70.80 | - | SigLIP + Resampler; Qwen3; 80 tokens; not in the 06-12 sweep |
| minicpm-v | MiniCPM-V-4.6-bf16 | ✅ | 342.81 | 176.52 | - | NEW (6-12); bf16; 32 prompt, 23 gen |
| molmo | Molmo-7B | ✅ | 570.22 | 78.81 | - | CLIP ViT + attention pooling + OLMo text; mlx-vlm baseline is a 1-token anomaly; 327 prompt, 100 gen |
| molmo2 | molmo2-4b | ✅ | 705.87 | 58.71 | 96% | mlx-vlm: 60.87; fast SDPA vision encoder; 438 prompt, 46 gen |
| internvl3 | InternVL3-1B | ✅ | 1750.54 | 216.93 | 82% | mlx-vlm: 264.40; InternViT + pixel-shuffle + Qwen2; 293 prompt, 8 gen |
| nemotron-omni | Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | ✅ | 256.66 | 69.81 | - | mlx-vlm: FAIL; NEW (5-19); Mamba2+Transformer+MoE+Parakeet audio; 6 gen |
| youtu-vl | youtu-vl-4b-instruct | ⚠️ | 213.13 | 42.98 | - | mlx-vlm: FAIL; NEW (5-19); only 1 gen token |
| qwen2-vl | Qwen2-VL-2B-Instruct-4bit | ✅ | 817.54 | 143.12 | - | Custom ViT + MRoPE; VLM image mode fixed; 12 gen |
| qwen2.5-vl | Qwen2.5-VL-3B-Instruct-4bit | ✅ | 625.02 | 106.80 | - | Windowed ViT + MRoPE; 91 prompt, 64 gen; mlx-vlm requires PyTorch |
| qwen3-vl | Qwen3-VL-2B-Instruct-4bit | ✅ | 662.94 | 176.65 | - | mlx-vlm: FAIL; DeepStack + vectorized MRoPE; 80 gen |
| qwen3-vl (4B) | Qwen3-VL-4B-Instruct-4bit | ✅ | 415.83 | 93.87 | - | mlx-vlm: FAIL; DeepStack + vectorized MRoPE; 38 gen |
| qwen3-vl (8B) | Qwen3-VL-8B-Instruct-4bit | ✅ | 284.81 | 62.61 | - | mlx-vlm: FAIL; DeepStack + vectorized MRoPE; 30 gen |
| qwen3-vl (32B) | Qwen3-VL-32B-Instruct-4bit | ✅ | 85.00 | 17.54 | - | mlx-vlm: FAIL; DeepStack + vectorized MRoPE; 59 gen |
| qwen3-vl-moe | Qwen3-VL-30B-A3B-Instruct-4bit | ✅ | 283.88 | 40.65 | - | mlx-vlm: FAIL; MoE (128 experts) + DeepStack; 34 gen |
| qwen3.5-vl (0.8B) | qwen3.5-0.8B-4bit | ✅ | 870.54 | 275.91 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet VLM; 69 prompt, 100 gen |
| qwen3.5-vl (2B) | qwen3.5-2B-4bit | ✅ | 502.83 | 190.74 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet VLM; 69 prompt, 43 gen |
| qwen3.5-vl (4B) | qwen3.5-4B-4bit | ✅ | 284.21 | 102.27 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet VLM; 69 prompt, 49 gen |
| qwen3.5-vl (9B 4bit) | qwen3.5-9B-4bit | ✅ | 195.12 | 74.41 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet VLM; 69 prompt, 100 gen |
| qwen3.5-vl (9B bf16) | qwen3.5-9B (bf16) | ✅ | 268.81 | 31.54 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet VLM; 69 prompt, 100 gen; bf16 |
| qwen3.5-vl (27B) | qwen3.5-27B-4bit | ✅ | 73.99 | 24.70 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet VLM; 69 prompt, 100 gen |
| qwen3.5-vl-moe | qwen3.5-35B-A3B-4bit | ✅ | 267.70 | 83.63 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet + MoE VLM; 69 prompt, 100 gen; gated delta decode RMSNorm fix |
| qwen3.6-vl-moe | qwen3.6-35B-A3B-4bit | ✅ | 271.38 | 79.65 | - | mlx-vlm: FAIL; Hybrid GatedDeltaNet + MoE VLM; 39 gen |
| molmo-point | - | ⏳ | - | - | - | Molmo-Point (point detection); implemented but no MLX model available |
| deepseek_vl2 | deepseek-vl2-small-4bit | ✅ | 407.92 | 95.87 | **105%** | mlx-vlm: 91.35; NEW (7-06, #660); SigLIP + 2D tile mosaic + DeepSeek-V2 MoE; 494 prompt, 24 gen |
| fastvlm | fastvlm-0.5b-bf16 | ✅ | 1566.54 | 274.15 | - | mlx-vlm: FAIL:warmup; NEW (7-06, #661); FastViTHD + Qwen2; 282 prompt, 100 gen |
| ernie4_5_moe_vl | ernie-4.5-vl-28b-a3b-thinking-4bit | ✅ | 214.45 | 89.39 | **158%** | mlx-vlm: 56.48; NEW (7-06, #662); modality-split MoE + 3D MRoPE; thinking model; 108 prompt, 100 gen |
| hunyuan_vl | hunyuanocr-mlx-4bit | ✅ | 1152.70 | 154.51 | **102%** | mlx-vlm: 151.14; NEW (7-06, #663); XD-RoPE + perceive merger; 10-token answer makes decode first-token-dominated, ratio not load-bearing; 284 prompt, 10 gen |
| qwen3_omni_moe | qwen3-omni-30b-a3b-instruct-4bit | ✅ | 288.32 | 22.63 | 56% | mlx-vlm: 40.76; NEW (7-06, #664); thinker (text+image+audio in); 2-token answer makes decode an artifact; text-mode decode is 82.81 (parity with qwen3-vl-30b's 81.80); 69 prompt, 2 gen |
| dots_ocr | dots.ocr-4bit | ⚠️ | 373.07 | 0.00 | - | NEW (7-06, #657); OCR-only model answers the generic image prompt with 0 tokens on both sides (mlx-vlm decode field also degenerate); OCR validated separately; 74 prompt, 0 gen |
| deepseekocr | deepseek-ocr-4bit | ✅ | 758.21 | 250.54 | - | mlx-vlm: FAIL:warmup; NEW (7-06, #655); SAM+CLIP dual tower; 281 prompt, 31 gen |
| deepseekocr2 | deepseek-ocr-2-4bit | ✅ | 701.55 | 148.76 | - | mlx-vlm: FAIL:warmup; NEW (7-06, #656); SAM + Qwen2-0.5B query resampler; 409 prompt, 5 gen |
| glm_ocr | glm-ocr-4bit | ⚠️ | 805.97 | 0.00 | - | mlx-vlm: 288.66; NEW (7-06, #646); OCR model, 0-token answer on the generic prompt (text-mode decode 261.93); 82 prompt, 0 gen |
| granite_vision | granite-vision-3.2-2b-4bit | ✅ | 1025.49 | 117.32 | 78% | mlx-vlm: 150.03; NEW (7-06, #647); AnyRes tiling drives the 1539-token prompt; 1539 prompt, 26 gen |
| granite4_vision | granite-4.0-3b-vision-4bit | ✅ | 619.81 | 122.79 | 97% | mlx-vlm: 126.42; NEW (7-06, #647); 337 prompt, 41 gen |
| lfm2_vl | lfm2-vl-450m-4bit | ✅ | 1416.96 | 396.14 | - | mlx-vlm: FAIL:warmup; NEW (7-06, #645); 82 prompt, 38 gen |
| idefics2 | idefics2-8b-4bit | ✅ | 273.28 | 103.34 | **102%** | mlx-vlm: 101.77; NEW (7-06, #639); SigLIP + perceiver resampler; 81 prompt, 12 gen |
| idefics3 | idefics3-8b-llama3-4bit | ✅ | 468.53 | 103.83 | **116%** | mlx-vlm: 89.83; NEW (7-06, #640); SmolVLM path; 186 prompt, 100 gen |

**VLM test conditions**: Image: 224x224 PNG (test_image.png) unless noted. Prompt: "What is in this image?" Max tokens: 100. Prefill includes vision encoder + projector overhead. mlx-vlm baseline uses the `d85ca4d` dev checkout. mlxcel decode speed was measured with `mlxcel-bench-decode` (model load, warmup, and measured pass in one process). Models with unavailable or failed mlx-vlm runs are marked with "-" in the vs mlx-vlm column. Two oversize models (`deepseek-v3-4bit` 99GB, `qwen3-next-480b-4bit` 251GB) do not run on this 128GB host; `minimax-m2-3bit` (93GB) fits and runs on the text side since the 05-28 sweep. Two Gemma 3 VLM rows (gemma-3-4b-it-4bit, gemma3-4b-4bit) were measured with `--warmup-tokens 0` because the prepared 4D attention mask shape is single-use against the first prefill's KV cache offset. The three Gemma 3n VLM rows use the default warmup=20 path.

## New Text Models (2026-07-06 sweep)

| Model | Test Model | Status | Prefill | Decode | vs mlx-lm | Notes |
|-------|------------|--------|---------|--------|-----------|-------|
| hunyuan_moe_13b (4bit) | hunyuan-a13b-instruct-4bit | ✅ | 75.79 | 44.30 | - | mlx-lm: FAIL:warmup; the 4-bit export loads where the bf16 one OOMs; 27 prompt, 100 gen |
| llada2_moe | llada2.0-mini-preview-4bit | ✅ | 421.04 | 156.64 | - | mlx-lm: FAIL:warmup (no masked-diffusion support); NEW (#659); 25 prompt, 100 gen |
| mellum2 | mellum2-12b-a2.5b-base | ✅ | 121.38 | 55.68 | - | mlx-lm: FAIL:warmup; base model, terse completions; 7 prompt, 6 gen |
| qwen3_next (80B) | qwen3-next-80b-a3b-instruct-4bit | ✅ | 144.04 | 59.35 | **115%** | mlx-lm: 51.73; hybrid gated-delta + MoE; 15 prompt, 57 gen |
| llama3.2 (1B instruct bf16) | llama-3.2-1b-instruct | ✅ | 2106.98 | 182.20 | 97% | mlx-lm: 187.88; 99 prompt, 34 gen |
| phi3_small (aq4_64) | phi-3-small-8k-instruct-aq4_64 | ✅ | 280.20 | 99.69 | **369%** | mlx-lm: 27.01 (aq4 path unoptimized in python); 22 prompt, 100 gen |
| mistral_small_4 (119B) | mistral-small-4-119b-2603-4bit | ✅ | 116.22 | 18.73 | - | mlx-lm: FAIL:warmup (load exceeds python budget); 22 prompt, 60 gen |
| qwen3.5 MoE (35B A3B) | qwen3.5-35b-a3b-4bit | ✅ | 223.03 | 81.44 | - | re-added post #670/#671 sanitize fix (load-regressed since #588); 19 prompt, 31 gen |
| qwen3.6 MoE (35B A3B) | qwen3.6-35b-a3b-4bit | ✅ | 232.36 | 79.99 | - | re-added post #670/#671 sanitize fix; 19 prompt, 27 gen |

## Summary Statistics

| Status | Count |
|--------|-------|
| ✅ Pass | ~156 text (decode > 0) + ~53 VLM |
| ⚠️ Partial | paligemma2-3b-6bit and youtu-vl-4b-instruct (0 text tokens / fewer than 5 image-mode tokens); dots.ocr-4bit and glm-ocr-4bit answer the generic image prompt with 0 tokens (OCR validated separately) |
| ❌ Fail | 2 (GLM-5, GLM-5.1) |
| ⏳ Skipped (non-standalone / off-disk) | 4 drafter/dflash checkpoints (gemma-4-12b/31b-it-assistant, qwen3.5-27b/4b-dflash), 2 non-generative checkpoints (docling-layout-heron, granite-speech-4.1), and 4 oversize models no longer on this host (deepseek-v3-4bit, qwen3-coder-480b-a35b, minimax-m2-3bit, dots.llm1-mixed) |

Counts are from the two 2026-07-12 CSVs (168 model dirs each; 166 real models plus the `models` / `large_models` container dirs), run at `--cooldown 30`. The prior four-item fail list is down to GLM-5 and GLM-5.1. `gemma-4-31b-it-nvfp4` is a real recovery on 0.4.0-rc.1 (FAIL:bench to 12.39 tok/s text). `minicpm-v-4.6-mxfp4` (216 tok/s text / 218 image) and the 4-bit `hunyuan-a13b-instruct-4bit` (42.3 tok/s) decode fine and were already passing in the 2026-07-06 CSV, so those old fail entries were stale; the HunYuan A13B bf16 export the entry named is off-disk. Since 2026-07-06 the disk set dropped four oversize checkpoints (deepseek-v3-4bit, qwen3-coder-480b-a35b, minimax-m2-3bit, dots.llm1-mixed) and added `paddleocr-vl-bfloat16` (#700) and `gemma-3-1b-it-4bit`. The VLM CSV attempts every dir in image mode; the ~53 real VLM rows are the genuine results and the rest are text models run with an image.

## Performance Comparison

The detailed same-day decode comparison tables below are the authoritative
source for baseline comparisons. Decode remains the primary apples-to-apples
runtime comparison.

### Aggregate (decode, same-day baseline)

| Mode | Comparable pairs | Median mlxcel/baseline | >=90% parity | >= baseline | Range |
|------|-----------------:|-----------------------:|-------------:|------------:|------:|
| Text vs mlx-lm | 66 | 98% | 59/66 (89%) | 21/66 (32%) | 48%-369% |
| VLM vs mlx-vlm | 26 | 99% | 20/26 (77%) | 13/26 (50%) | 56%-158% |

The text median holds at 98%, matching 2026-07-06. This sweep runs `--cooldown 30`, so it carries no thermal offset; a first `--cooldown 0` pass of the same build read about 2% lower across the board (see the run-over-run note below). The 369% high is `phi-3-small-8k-instruct-aq4_64` (the Python aq4 path is unoptimized); the 48% low is `falcon-mamba-7b-4bit`, whose chat template hits early EOS after 2 tokens.

### Run-over-run (2026-07-12 mlxcel v0.4.0-rc.1)

The 2026-07-12 sweep (`benchmarks/metal_m1ultra_2026-07-12.csv` + `_vlm_`) is a version-change full re-benchmark from 0.3.3 to 0.4.0-rc.1 on MLX pin 57c66cac, `--cooldown 30`. Decode is flat: the median row moves -2.0% against 2026-07-06 and prefill -3.0% across 153 comparable pairs, with the distribution balanced (one row past -10%, three past +10%) and no mid-size or large model regressing. 0.4.0-rc.1 decodes at 0.3.3 speed. The one row past -10% is `glm-ocr-4bit` text-mode (-10.7%), a small OCR model whose decode swings roughly +/-15% run to run (a separate re-check read it at 265 tok/s, above its 262 on 2026-07-06); the three past +10% are `qwen3-1.7b-4bit` (+11.4%), `falcon-h1-tiny-90m-instruct-4bit` (+10.9%), and `gemma-4-e2b-it-qat-4bit` (+10.3%). All are small high-throughput models near the noise floor.

The `--cooldown 30` sweep is the clean canonical run. A first `--cooldown 0` pass of this same build (kept as `benchmarks/metal_m1ultra_2026-07-12_cooldown0.csv` + `_vlm_`) read about 2% lower on decode across the board (decode median -3.3% versus -2.0% here, with 114 of 156 rows higher once cooldown was added), which is a thermal offset from running at `--cooldown 0` right after an 8.5-minute rebuild, not a code effect. A Time Machine backup that began during the first cooldown pass was stopped and automatic backups disabled before the canonical run, so the two CSVs isolate the thermal component cleanly.

The prior doc's four-item fail list is down to two (GLM-5, GLM-5.1). `gemma-4-31b-it-nvfp4` is a real recovery: FAIL:bench on 2026-07-06 to 12.4 tok/s text now. `minicpm-v-4.6-mxfp4` (216 tok/s text / 218 image) and the HunYuan A13B 4-bit path decode fine and were already passing in the 2026-07-06 CSV, so their old fail entries were stale; the HunYuan A13B bf16 export that the entry named is off-disk. The disk set dropped four oversize checkpoints (deepseek-v3-4bit, qwen3-coder-480b-a35b, minimax-m2-3bit, dots.llm1-mixed) and gained `paddleocr-vl-bfloat16` (#700) and `gemma-3-1b-it-4bit`. On the VLM side the short-answer rows, whose decode is first-token-dominated, read higher (hunyuanocr 94 to 155, idefics2 91 to 103); that tracks generation-length variance on a 10-to-30-token answer, not a kernel change.

### Run-over-run (2026-07-06 mlxcel v0.3.3)

The 2026-07-06 sweep (`benchmarks/metal_m1ultra_2026-07-06.csv` + `_vlm_`) lands after the five-family VLM port batch (#660-#664), the slice_update/causal-mask/downloader fix batch (#666-#668), and the qwen3.5-MoE sanitize fix (#671). Compared with 2026-06-15: no decode regressions; the qwen VLM family and small text models improved +10-16% (fused decode-MoE default-on and related work), youtu-vl recovered from a prior 24.2 artifact to 44.6. The sweep itself caught issue #670 (qwen3.5/3.6-35B MoE had silently stopped loading when #588 renamed the shared expert projections), which was fixed and re-benched the same day (82.25 / 78.84 tok/s, above their 2026-06-12 values). New python baselines were measured for 21 recently added models (`benchmarks/pylm_m1ultra_*2026-07-06*_single_*.csv`); mlx-lm/mlx-vlm failed to load 8 of them (fastvlm, lfm2-vl, deepseek-ocr/-2 among the VLMs; hunyuan-a13b-4bit, llada2-mini, mellum2, mistral-small-4-119b among text), so those rows carry no ratio.

### Run-over-run (2026-06-15 mlxcel v0.2.1, post #289 fix)

The 2026-06-15 sweep runs on mlxcel v0.2.1 after fixing issue #289: a bf16 to f16 quant-scale promotion added in #260 that regressed bf16-scale quantized decode by 33-41% on M1 Ultra. PR #290 keeps quantized models in bf16, restoring the prior behavior. bf16-scale models recovered to roughly 95-121% of their prior baseline (qwen3, nemotron, gpt-oss, solar, minimax, hunyuan-large, jamba, dots all return to or above their pre-regression decode; the qwen3-30B-A3B MoE rows jump to 119-120% of the 05-19 reference). f16-scale models (mixtral, llama, qwen2.5) are unchanged across the fix, as expected. The pre-fix evidence sweep is `benchmarks/metal_m1ultra_2026-06-15_pre289_regressed.csv`.

### Representative decode wins

| Model | mlxcel | Baseline | vs baseline |
|-------|-------:|---------:|------------:|
| qwen3-30b-a3b-4bit | 83.75 | 70.18 | **119%** |
| qwen2.5-0.5b-4bit | 343.91 | 315.48 | **109%** |
| phi-3.5-moe-4bit | 76.55 | 69.28 | **110%** |
| minicpm3-4b-4bit | 80.22 | 73.26 | **110%** |
| llava-interleave-qwen-0.5b-bf16 (VLM) | 264.66 | 225.15 | **118%** |
| gemma3n-e2b-4bit (VLM) | 72.95 | 59.57 | **122%** |
| gemma-4-e2b-it-4bit (VLM) | 106.05 | 97.19 | **109%** |
| phi-3.5-vision-4bit (VLM) | 122.35 | 92.53 | **132%** |

internvl3-1b has no text-path Python baseline; its 341.1 tok/s text decode in the 06-15 sweep is consistent with the reproducible 337-344 tok/s band measured across MLX pins, so its earlier low table value was a single-run measurement artifact rather than a real slowdown. The qwen3-30B-A3B MoE rows (both the `qwen3-moe-4bit` and `qwen3-30b-a3b-4bit` dirs) read 83.8 tok/s here versus the 70.2 tok/s 05-19 reference, the largest bf16-scale recovery in this sweep.

### Main optimization gaps

| Model | mlxcel | Baseline | vs baseline | Notes |
|-------|-------:|---------:|------------:|-------|
| falcon-mamba-7b-4bit | 37.54 | 91.04 | 41% | Chat template causes early EOS; only 2 generated tokens |
| qwen2.5-vl-3b-4bit (text path) | 100.09 | 160.42 | 62% | VLM wrapper text-only comparison |
| qwen2-vl-2b-4bit (text path) | 151.38 | 236.86 | 64% | VLM wrapper text-only comparison |
| gemma-4-31b-4bit (VLM) | 15.48 | 20.30 | 76% | large VLM path |
| gemma-3-4b-it-4bit (VLM) | 86.57 | 93.79 | 92% | measured with warmup=0; see VLM test conditions |

molmo-7b (text path, no Python baseline) reads 68.6 tok/s in the 06-15 sweep, in line with the reproducible 66-69 tok/s band; its VLM path reads 80.46 tok/s. The mlx-vlm baseline for molmo-7b is a 1-token anomaly, so the VLM comparison column carries "-".

## Performance vs mlx-lm / mlx-vlm baseline (mlxcel 2026-06-15 vs pinned 2026-05-19 reference)

Source CSVs (same M1 Ultra host; mlxcel 0.2.1 measured 2026-06-15 on the a6ec712 MLX pin, post #289 fix; mlx-lm / mlx-vlm baselines from the pinned 2026-05-19 reference checkout with `PYLM_BENCH_MAX_GB=65`):

- mlxcel: `benchmarks/metal_m1ultra_2026-06-15.csv`
- mlxcel VLM: `benchmarks/metal_m1ultra_vlm_2026-06-15.csv`
- mlx-lm: `benchmarks/pylm_m1ultra_2026-05-19.csv` (mlx-lm 0.31.3 dev checkout in https://github.com/ml-explore/mlx-lm @ `df1d3f3`)
- mlx-vlm: `benchmarks/pylm_m1ultra_vlm_2026-05-19.csv` (mlx-vlm dev checkout in https://github.com/Blaizzy/mlx-vlm @ `d85ca4d`)

The mlx-lm / mlx-vlm baselines are the pinned 2026-05-19 reference checkout (the reference is fixed, so its decode on this host is stable); the mlxcel side is the 2026-06-15 full sweep on v0.2.1. All sweeps use `--max-tokens 100` and the same `Hello, how are you today?` / `What is in this image?` prompts. `deepseek-v3-4bit` and `qwen3-next-480b-4bit` exceed the 128GB host on both sides; `minimax-m2-3bit` fits and runs on the mlxcel side (31.90 tok/s) but mlx-lm still fails it, so it stays outside the comparable set. New text families in this sweep (apertus-8b-2509, seed-oss-36b, dots.llm1, the granite-3.3/4.0-h/4.1 family, lfm2-350m + lfm2-8b-a1b, plamo-2-1b, falcon-h1-tiny, both BitNet b1.58 2B variants) plus gemma-4-12b-it, the Gemma 4 QAT variants, MiniCPM-V-4.6, internvl3-1b, and molmo-7b on the text path have no Python baseline and carry "-" in the comparison columns.

Numbers are decode tok/s. `mlxcel vs mlx-lm` is `mlxcel / mlx-lm` as a percentage; **bold** = mlxcel >= mlx-lm. `FAIL` cells are real load/runtime errors on that backend with this configuration. The mlx-lm checkout used here (`df1d3f3` — "Fix Gemma 4 sanitize() not stripping KV projections for shared layers" ml-explore/mlx-lm#1240) is newer than the M5 Max page's `ed1fca4`, so some FAIL categories differ.

### Aggregate (text)

- **Comparable text pairs**: 74
- **mlxcel >= mlx-lm**: 24 / 74 (32%)
- **mlxcel >= 90% parity**: 59 / 74 (80%)
- **Average mlxcel/mlx-lm**: 96% (median 98%, range 41%-120%)

### Aggregate (VLM, models with >=5 generated tokens both sides)

- **Comparable VLM pairs**: 18
- **mlxcel >= mlx-vlm**: 8 / 18 (44%)
- **mlxcel >= 90% parity**: 13 / 18 (72%)
- **Average mlxcel/mlx-vlm**: 102% (median 98%, range 76%-132%)

### Text decode (tok/s)

| Model | mlxcel | mlx-lm | mlxcel vs mlx-lm |
|-------|--------|--------|------------------|
| Meta-Llama-3.1-8B-Instruct-4bit | 108.34 | 109.84 | 99% |
| MiniCPM-V-4.6-bf16 | 198.11 | - | - |
| Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | 88.11 | FAIL | - |
| Qwen2.5-1.5B-4bit | 240.80 | 241.41 | 100% |
| Qwen2.5-1.5B-Instruct-4bit | 241.16 | 239.20 | **101%** |
| Qwen2.5-7B-Instruct-4bit | 112.58 | 110.90 | **102%** |
| Qwen3.5-0.8B-OptiQ-4bit | FAIL | 265.86 | - |
| apertus-8b-instruct-2509-4bit | 78.43 | - | - |
| aya-expanse-8b-4bit | 108.53 | 112.74 | 96% |
| aya-vision-8b | 110.54 | FAIL | - |
| baichuan-m1-14b-4bit | 40.47 | 49.11 | 82% |
| bitnet-b1.58-2b-4t | 137.01 | - | - |
| bitnet-b1.58-2b-4t-4bit | 150.50 | - | - |
| bunny-llama3-8b-4bit | 103.62 | FAIL | - |
| command-r7b-4bit | 111.72 | 107.75 | **104%** |
| deepseek-coder-1.3b-4bit | 157.78 | FAIL | - |
| deepseek-r1-distill-7b-4bit | 111.65 | 111.34 | **100%** |
| deepseek-v2-lite-4bit | 101.56 | 117.06 | 87% |
| deepseek-v3-4bit | - | FAIL | - |
| diffusiongemma-26B-A4B-it-4bit | 68.50 | - | - |
| dots.llm1.inst-mixed-4-6bit | 28.52 | - | - |
| ernie-4.5-0.3b-4bit | 495.71 | FAIL | - |
| exaone-3.5-2.4b-4bit | 197.73 | 194.65 | **102%** |
| exaone4-1.2b-4bit | 241.51 | FAIL | - |
| falcon-h1-tiny-90m-instruct-4bit | 288.14 | - | - |
| falcon-mamba-7b-4bit | 37.54 | 91.04 | 41% |
| gemma-2b-4bit | 195.69 | 207.78 | 94% |
| gemma-3-4b-it-4bit | 116.59 | 109.72 | **106%** |
| gemma-4-12b-it-4bit | 34.76 | - | - |
| gemma-4-26b-a4b-it-4bit | 79.92 | 72.52 | **110%** |
| gemma-4-26B-A4B-it-qat-4bit | 77.99 | - | - |
| gemma-4-31b-4bit | 20.25 | 20.36 | 99% |
| gemma-4-31b-it-4bit | 19.33 | 20.23 | 96% |
| gemma-4-31B-it-qat-4bit | 15.53 | - | - |
| gemma-4-e2b-it-4bit | 117.24 | FAIL | - |
| gemma-4-e2b-it-8bit | 87.87 | FAIL | - |
| gemma-4-e2b-it-qat-4bit | 95.29 | - | - |
| gemma-4-e4b-it-4bit | 82.43 | FAIL | - |
| gemma-4-e4b-it-8bit | 59.20 | FAIL | - |
| gemma-4-e4b-it-qat-4bit | 63.73 | - | - |
| gemma2-2b-4bit | 166.07 | 153.50 | **108%** |
| gemma3-1b-4bit | 229.70 | 211.50 | **109%** |
| gemma3-4b-4bit | 115.45 | 109.48 | **105%** |
| gemma3n-e2b-4bit | 75.85 | FAIL | - |
| gemma3n-e4b-4bit | 57.91 | FAIL | - |
| gemma3n-e4b-bf16 | 32.95 | 39.02 | 84% |
| glm4-flash-4bit | 45.53 | 49.47 | 92% |
| gpt-oss-120b-4bit | 58.41 | 57.58 | **101%** |
| gpt-oss-20b-mxfp4 | 89.72 | 89.51 | **100%** |
| granite-3.3-2b-instruct-4bit | 179.69 | - | - |
| granite-4.0-h-350m-4bit | 219.49 | - | - |
| granite-4.0-h-tiny-4bit | 96.28 | - | - |
| granite-4.1-3b-4bit | 120.34 | - | - |
| granite-4.1-8b-4bit | 29.89 | - | - |
| hunyuan-1.8b-4bit | 182.55 | 200.59 | 91% |
| hunyuan-large-4bit | 44.07 | FAIL | - |
| internlm2-7b-4bit | 109.69 | 111.92 | 98% |
| internlm3-8b-4bit | 87.02 | FAIL | - |
| internvl3-1b | 341.09 | - | - |
| jamba-v0.1-4bit | 122.38 | 131.04 | 93% |
| lfm2-350m-8bit | 509.90 | - | - |
| lfm2-8b-a1b-4bit | 180.31 | - | - |
| llama-3.1-8b-4bit | 107.89 | 110.66 | 97% |
| llama-3.1-8b-bf16 | 35.58 | 35.32 | **101%** |
| llama-3.2-1b-4bit | 364.36 | 418.25 | 87% |
| llama-4-scout-17b-4bit | 36.42 | FAIL | - |
| llava-1.5-7b-4bit | 116.27 | FAIL | - |
| llava-interleave-qwen-0.5b-bf16 | 317.83 | FAIL | - |
| llava-next-mistral-7b-4bit | 114.33 | FAIL | - |
| mamba2-1.3b-4bit | 79.25 | FAIL | - |
| mimo-7b-4bit | 85.30 | 86.17 | 99% |
| minicpm-2b-4bit | 164.41 | 156.47 | **105%** |
| minicpm3-4b-4bit | 80.22 | 73.26 | **110%** |
| minimax-m2-3bit | 31.90 | FAIL | - |
| ministral-3b-4bit | 142.60 | 159.34 | 89% |
| mistral-small-3.1-24b-4bit | 31.94 | 31.97 | 100% |
| mixtral-8x7b-4bit | 54.25 | 54.91 | 99% |
| molmo-7b | 68.62 | - | - |
| molmo2-4b | 60.79 | FAIL | - |
| nemotron-h-30b-4bit | 91.54 | 93.34 | 98% |
| nemotron-nas-30b-4bit | 91.56 | 92.93 | 99% |
| olmo-1b-4bit | 210.63 | FAIL | - |
| olmo2-7b-4bit | 103.48 | 110.88 | 93% |
| olmo3-32b-4bit | 21.81 | 21.57 | **101%** |
| paligemma2-3b-6bit | 0.00 | FAIL | - |
| phi-2-4bit | 65.09 | FAIL | - |
| phi-3-mini-4bit | 168.08 | 171.36 | 98% |
| phi-3.5-mini-4bit | 163.60 | 166.30 | 98% |
| phi-3.5-moe-4bit | 76.55 | 69.28 | **110%** |
| phi-3.5-vision-4bit | 163.86 | FAIL | - |
| phi-4-4bit | 58.46 | 58.68 | 100% |
| pixtral-12b-4bit | 70.04 | 69.49 | **101%** |
| plamo-2-1b | 107.13 | - | - |
| qwen1.5-moe-a2.7b-4bit | 145.88 | 144.98 | **101%** |
| qwen2-vl-2b-4bit | 151.38 | 236.86 | 64% |
| qwen2.5-0.5b-4bit | 343.91 | 315.48 | **109%** |
| qwen2.5-7b-4bit | 111.50 | 111.38 | **100%** |
| qwen2.5-7b-8bit | 70.22 | 70.46 | 100% |
| qwen2.5-vl-3b-4bit | 100.09 | 160.42 | 62% |
| qwen3-0.6b-4bit | 275.55 | 299.61 | 92% |
| qwen3-1.7b-4bit | 182.96 | 221.37 | 83% |
| qwen3-30b-a3b-4bit | 83.75 | 70.18 | **119%** |
| qwen3-4b-4bit | 120.14 | 123.92 | 97% |
| qwen3-8b-4bit | 80.29 | 84.54 | 95% |
| qwen3-moe-4bit | 83.89 | 69.67 | **120%** |
| qwen3-next-480b-4bit | - | FAIL | - |
| qwen3-vl-2b-4bit | 210.43 | 222.67 | 95% |
| qwen3-vl-30b-a3b-4bit | 69.25 | 70.04 | 99% |
| qwen3-vl-32b-4bit | 20.84 | 21.99 | 95% |
| qwen3-vl-4b-4bit | 116.50 | 124.02 | 94% |
| qwen3-vl-8b-4bit | 80.26 | 84.46 | 95% |
| qwen3.5-0.8b-4bit | 229.86 | 269.52 | 85% |
| qwen3.5-27b-4bit | 23.87 | 25.93 | 92% |
| qwen3.5-2b-4bit | 172.86 | 211.68 | 82% |
| qwen3.5-35b-a3b-4bit | 69.74 | 76.44 | 91% |
| qwen3.5-4b-4bit | 98.36 | 115.60 | 85% |
| qwen3.5-9b-4bit | 70.67 | 81.27 | 87% |
| qwen3.5-9b-bf16 | 31.35 | 34.22 | 92% |
| qwen3.6-35b-a3b-4bit | 64.61 | 73.18 | 88% |
| seed-oss-36b-instruct-4bit | 20.04 | - | - |
| smollm-135m-4bit | 374.92 | 375.91 | 100% |
| smollm3-3b-4bit | 126.29 | 141.66 | 89% |
| solar-open-100b-4bit | 32.96 | 35.69 | 92% |
| stablelm-1.6b-4bit | 270.88 | 280.65 | 97% |
| starcoder2-3b-4bit | 166.40 | 166.17 | **100%** |
| youtu-vl-4b-instruct | 0.00 | FAIL | - |

### VLM decode (tok/s)

| Model | mlxcel | mlx-vlm | mlxcel vs mlx-vlm |
|-------|--------|--------|------------------|
| MiniCPM-V-4.6-bf16 | 176.12 | - | - |
| Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | 64.72 | FAIL | - |
| aya-vision-8b | 109.57 | 103.74 | **106%** |
| bunny-llama3-8b-4bit | 94.69 | FAIL | - |
| deepseek-v3-4bit | - | FAIL | - |
| gemma-3-4b-it-4bit | 86.57 | 97.36 | 89% |
| gemma-4-12b-it-4bit | 32.07 | - | - |
| gemma-4-26b-a4b-it-4bit | 70.38 | 61.07 | **115%** |
| gemma-4-26B-A4B-it-qat-4bit | 66.77 | - | - |
| gemma-4-31b-4bit | 15.48 | 20.30 | 76% |
| gemma-4-31b-it-4bit | 18.69 | 19.78 | 94% |
| gemma-4-31B-it-qat-4bit | 15.21 | - | - |
| gemma-4-e2b-it-4bit | 106.05 | 97.19 | **109%** |
| gemma-4-e2b-it-8bit | 81.82 | 91.06 | 90% |
| gemma-4-e2b-it-qat-4bit | 84.01 | - | - |
| gemma-4-e4b-it-4bit | 75.04 | 70.34 | **107%** |
| gemma-4-e4b-it-8bit | 55.76 | 63.25 | 88% |
| gemma-4-e4b-it-qat-4bit | 60.45 | - | - |
| gemma3-4b-4bit | 88.44 | 93.79 | 94% |
| gemma3n-e2b-4bit | 72.95 | 59.57 | **122%** |
| gemma3n-e4b-4bit | 57.92 | 50.00 | **116%** |
| gemma3n-e4b-bf16 | 32.31 | 36.18 | 89% |
| internvl3-1b | 238.34 | 264.40 | 90% |
| llama-4-scout-17b-4bit | 34.90 | FAIL | - |
| llava-1.5-7b-4bit | 104.04 | FAIL | - |
| llava-interleave-qwen-0.5b-bf16 | 264.66 | 225.15 | **118%** |
| llava-next-mistral-7b-4bit | 106.80 | 109.51 | 98% |
| minimax-m2-3bit | - | FAIL | - |
| ministral-3b-4bit | 124.82 | FAIL | - |
| mistral-small-3.1-24b-4bit | 29.83 | FAIL | - |
| molmo-7b | 80.46 | 38399.52 (anomalous) | - |
| molmo2-4b | 59.87 | 60.87 | 98% |
| paligemma2-3b-6bit | 50.14 | 70.45 | 71% |
| phi-3.5-vision-4bit | 122.35 | 92.53 | **132%** |
| pixtral-12b-4bit | 59.76 | FAIL | - |
| qwen2-vl-2b-4bit | 126.81 | FAIL | - |
| qwen2.5-vl-3b-4bit | 97.65 | FAIL | - |
| qwen3-next-480b-4bit | - | FAIL | - |
| qwen3-vl-2b-4bit | 160.88 | FAIL | - |
| qwen3-vl-30b-a3b-4bit | 40.53 | FAIL | - |
| qwen3-vl-32b-4bit | 17.81 | FAIL | - |
| qwen3-vl-4b-4bit | 89.98 | FAIL | - |
| qwen3-vl-8b-4bit | 63.10 | FAIL | - |
| qwen3.5-0.8b-4bit | 232.41 | FAIL | - |
| qwen3.5-27b-4bit | 24.87 | FAIL | - |
| qwen3.5-2b-4bit | 169.42 | FAIL | - |
| qwen3.5-35b-a3b-4bit | 74.68 | FAIL | - |
| qwen3.5-4b-4bit | 99.40 | FAIL | - |
| qwen3.5-9b-4bit | 72.78 | FAIL | - |
| qwen3.5-9b-bf16 | 32.59 | FAIL | - |
| qwen3.6-35b-a3b-4bit | 70.25 | FAIL | - |
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
| qwen2.5-0.5b-bf16 | Now passes after the #289 bf16-scale fix (298.92 tok/s, 100 tokens); was a bf16 warmup failure | Resolved |
| Qwen3.5-4B-DFlash / Qwen3.5-27B-DFlash | Drafter checkpoint — not a standalone inference model | Low |
| Qwen3.5-0.8B-OptiQ-4bit | Warmup failure on new OptiQ quant variant | Medium |
| gemma-4-31B-it-assistant-bf16 / gemma-4-12B-it-assistant-4bit | Drafter checkpoint, not a standalone inference model | Low |
| MiniCPM-V-4.6-mxfp4 | Warmup failure on the mxfp4 variant (bf16 variant passes) | Medium |
| docling-layout-heron-mlx-bf16 | Layout-analysis checkpoint, not a generative LM; fails bench harness | Low |
| falcon-mamba | Chat template causes early EOS (only 2 tokens); decode now 37.54 tok/s | Medium |
| paligemma | Only 2 VLM gen tokens; decode is not comparable despite 50.14 tok/s measured | High |
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
