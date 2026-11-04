# Model Compatibility & Performance Tests (NVIDIA GB10)

Compatibility and single-stream decode/prefill performance for mlxcel models on **NVIDIA GB10 (DGX Spark)** running the CUDA backend. This is the **0.8.1** benchmark: a full sweep of all 312 model directories on 2026-11-03 (text) and 2026-11-04 (VLM), 286 of them checkpoints, with the memory gate excluding nine oversized text checkpoints and four oversized VLM checkpoints up front.

Three rows were re-measured on 2026-11-04 after the failures in the first pass were fixed: the BitNet pair (a CUDA grid-axis limit in `bitlinear_matmul`) and DiffusionGemma (the bench harness now drives the model-owned block-diffusion loop the server serves). Those rows come from `benchmarks/cuda_gb10_2026-11-04_recheck.csv` and `benchmarks/cuda_gb10_vlm_2026-11-04_recheck.csv`, at `630aaeba` plus the two uncommitted fixes, recorded as `630aaeba-dirty`.

## Test Environment

| Item | Value |
|------|-------|
| **Hardware** | NVIDIA GB10 (DGX Spark), 122 GB unified LPDDR5x |
| **OS** | Linux (aarch64), kernel 7.0.0-1019-nvidia |
| **Backend** | CUDA 13.0.88 (driver 580.178.04) |
| **mlxcel version** | 0.8.1 at `630aaeba` |
| **MLX pin** | `7ca40033` |
| **CUDA build** | `MLX_CUDA_ARCHITECTURES=121 cargo build --release --features cuda` |
| **Harness** | same-process `mlxcel-bench-decode` (engine path), warmup 20 tokens, pre-warm on, `--cooldown 15 --big-cooldown 45`, `BENCH_MEM_OVERHEAD_FACTOR=2.0` |
| **Text shape** | synthetic 512-token prompt, `--max-tokens 128 --ignore-eos` (pp512/tg128) |
| **VLM shape** | `"What is in this image?"` with `tests/fixtures/test_image.png`, `--max-tokens 128 --ignore-eos` |
| **Test Date** | 2026-11-03 (text), 2026-11-04 (VLM) |
| **CSVs** | `benchmarks/cuda_gb10_2026-11-03.csv`, `benchmarks/cuda_gb10_vlm_2026-11-04.csv`, plus `benchmarks/cuda_gb10_2026-11-04_recheck.csv` and `benchmarks/cuda_gb10_vlm_2026-11-04_recheck.csv` for the three fixed rows |

## Measurement shape and comparability

This sweep uses a different shape from the 0.4.0-rc.1 record it supersedes. The 2026-07-12 text sweep sent the short prompt `"Hello, how are you today?"` and generated at most 100 tokens, so many rows stopped early at EOS. This sweep synthesizes a deterministic 512-token prompt and suppresses EOS for the measured pass, so every engine row generates exactly 128 tokens. The one diffusion row cannot take the EOS suppression and records its own count.

Decode tok/s is therefore comparable across the two sweeps, and the deltas quoted below compare decode only. Prefill tok/s is not comparable: a 512-token prompt changes the matrix shapes and the attention mask work by an order of magnitude, so prefill columns from the two sweeps should not be subtracted. The VLM sweep keeps the fixed prompt and compares directly on prefill and decode.

## Legend

- ✅ Pass: model loads and produces tokens within the configured budget
- ❌ Fail: the checkpoint is a text generator but the measured pass fails
- ⚪ Not tested / not applicable: not a standalone text generator (embedding, reranker, classifier, speech, image-only, document layout, or an MTP/DFlash drafter), excluded by the memory gate, missing weights, or a byte-identical alias of a measured checkpoint (see Checkpoint dedup)

Prefill/Decode are the measured-pass figures from `mlxcel-bench-decode`. Models are grouped by architecture family. VLM-capable models appear once under text (text-prompt pass) and again in the image-input table at the end.

## Basic Transformers

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| apertus-8b-instruct-2509-4bit | ✅ | 2607.21 | 51.81 |  |
| iquest-coder-v1-40b-loop-instruct-4bit | ✅ | 370.27 | 4.58 |  |
| iquest-coder-v1-7b-instruct-8bit | ✅ | 3679.87 | 29.02 |  |
| laguna-xs-2.1-nvfp4 | ✅ | 1899.01 | 50.98 |  |
| llama-3.1-8b-4bit | ✅ | 4207.61 | 49.75 |  |
| llama-3.1-8b-bf16 | ✅ | 3863.27 | 15.17 |  |
| llama-3.2-1b-4bit | ✅ | 22752.38 | 273.44 |  |
| llama-3.2-1b-instruct-bf16 | ✅ | 26277.41 | 92.59 |  |
| mimo-7b-4bit | ✅ | 4037.76 | 52.69 |  |
| minicpm-2b-4bit | ✅ | 9726.37 | 113.19 |  |
| muse-glimmer-30b | ⚪ | - | - | memory-gate skip on 2026-11-03 |
| muse-glimmer-30b-4bit | ✅ | 1084.18 | 13.76 |  |
| muse-glimmer-30b-optiq-4bit | ✅ | 1045.34 | 11.29 |  |
| nanochat-d20-q8-mlx | ✅ | 39108.19 | 334.04 |  |
| olmo-1b-4bit | ✅ | 3516.58 | 109.21 |  |
| olmo-2-1124-7b-instruct-4bit | ✅ | 4271.00 | 51.54 |  |
| olmo-3.1-32b-think-4bit | ✅ | 927.64 | 11.60 |  |
| phi-2-4bit | ✅ | 4523.62 | 74.08 |  |
| phi-3-mini-4bit | ✅ | 7684.10 | 92.51 |  |
| phi-3-small-8k-instruct-aq4_64 | ✅ | 2980.18 | 49.40 |  |
| phi-3.5-mini-4bit | ✅ | 7664.29 | 90.45 |  |
| phi-4-4bit | ✅ | 2174.91 | 27.58 |  |
| qwen2-0.5b | ✅ | 43387.28 | 476.18 |  |
| qwen2.5-0.5b-bf16 | ✅ | 41824.24 | 211.57 |  |
| qwen2.5-1.5b-4bit | ✅ | 18557.23 | 210.32 |  |
| qwen2.5-1.5b-instruct-4bit | ✅ | 18020.21 | 210.22 |  |
| qwen2.5-7b | ✅ | 4220.41 | 54.27 |  |
| qwen2.5-7b-8bit | ✅ | 4227.84 | 30.39 |  |
| qwen3-0.6b | ✅ | 32915.68 | 353.87 |  |
| qwen3-1.7b-4bit | ✅ | 15672.46 | 183.69 |  |
| qwen3-4b-4bit | ✅ | 7111.37 | 85.21 |  |
| qwen3-8b-4bit | ✅ | 4096.22 | 49.28 |  |
| qwen3-reranker-0.6b-4bit | ✅ | 30261.07 | 362.57 | reranker checkpoint on a Qwen3 causal decoder; this row measures token generation, not ranking quality |
| smollm-135m-4bit | ✅ | 53385.49 | 474.99 |  |
| smollm3-3b-4bit | ✅ | 7731.59 | 111.05 |  |
| stablelm-1.6b-4bit | ✅ | 8673.14 | 123.78 |  |
| starcoder2-3b-4bit | ✅ | 6769.90 | 102.56 |  |

## Gemma Family

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| gemma-1.1-2b-it-4bit | ✅ | 9185.89 | 95.80 |  |
| gemma-2-9b-8bit | ✅ | 2376.12 | 21.42 |  |
| gemma-3n-e4b-bf16 | ✅ | 3282.52 | 22.62 |  |
| gemma2-2b-4bit | ✅ | 9820.81 | 114.87 |  |
| gemma3-1b-4bit | ✅ | 23597.91 | 247.83 |  |
| gemma3-4b-4bit | ✅ | 6900.67 | 84.33 |  |
| gemma3n-e2b-4bit | ✅ | 4891.77 | 92.95 |  |
| gemma3n-e4b-4bit | ✅ | 3207.41 | 58.40 |  |

## Gemma 4

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| diffusiongemma-26b-a4b-it-4bit | ✅ | 797.61 | 6.86 | model-owned block-diffusion loop (`decode_path=diffusion`), re-measured 2026-11-04; 128 generated tokens. Not comparable to the autoregressive rows: the rate covers canvas denoising, not one token per step |
| gemma-4-12b-it-4bit | ✅ | 1782.91 | 14.52 |  |
| gemma-4-12b-it-assistant-4bit | ⚪ | - | - | MTP/DFlash drafter (needs a target; not standalone) |
| gemma-4-26b-a4b-it-4bit | ✅ | 743.18 | 63.88 |  |
| gemma-4-26b-a4b-it-qat-4bit | ✅ | 701.32 | 57.59 |  |
| gemma-4-31b-4bit | ✅ | 577.54 | 9.12 |  |
| gemma-4-31b-it-4bit | ✅ | 894.51 | 9.13 |  |
| gemma-4-31b-it-assistant-bf16 | ⚪ | - | - | MTP/DFlash drafter (needs a target; not standalone) |
| gemma-4-31b-it-nvfp4 | ✅ | 762.59 | 5.06 |  |
| gemma-4-31b-it-qat-4bit | ✅ | 730.74 | 5.77 |  |
| gemma-4-e2b-it-4bit | ✅ | 9821.44 | 104.37 |  |
| gemma-4-e2b-it-8bit | ✅ | 7631.99 | 59.16 |  |
| gemma-4-e2b-it-qat-4bit | ✅ | 8053.26 | 69.91 |  |
| gemma-4-e4b-it-4bit | ✅ | 5315.35 | 54.23 |  |
| gemma-4-e4b-it-8bit | ✅ | 4182.95 | 30.46 |  |
| gemma-4-e4b-it-qat-4bit | ✅ | 4369.13 | 35.31 |  |

## EXAONE

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| exaone-3.5-2.4b-4bit | ✅ | 10704.05 | 131.38 |  |
| exaone-4.0-1.2b-4bit | ✅ | 16538.89 | 230.21 |  |

## Cohere / Command R

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| aya-expanse-8b-4bit | ✅ | 3704.94 | 50.78 |  |
| c4ai-command-r7b-12-2024-4bit | ✅ | 4200.25 | 50.25 |  |

## Granite (IBM)

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| granite-3.3-2b-instruct-4bit | ✅ | 9402.30 | 126.99 |  |
| granite-4.0-h-350m-4bit | ✅ | 5658.19 | 320.63 |  |
| granite-4.0-h-tiny-4bit | ✅ | 1660.87 | 138.28 |  |
| granite-4.1-3b-4bit | ✅ | 6015.59 | 88.00 |  |
| granite-4.1-8b-4bit | ✅ | 2641.70 | 39.23 |  |

## MoE (Mixture of Experts)

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| deepseek-v2-lite-4bit | ✅ | 258.48 | 101.15 | coherent output confirmed with `mlxcel generate` on this host; earlier reports of degenerate output did not reproduce |
| dots.llm1.inst-mixed-4-6bit | ⚪ | - | - | memory-gate skip on 2026-11-03 |
| gpt-oss-120b-4bit | ⚪ | - | - | memory-gate skip on 2026-11-03 |
| gpt-oss-20b-mxfp4 | ✅ | 736.22 | 89.42 |  |
| lfm2-8b-a1b-4bit | ✅ | 1073.61 | 168.16 |  |
| llada2.0-mini | ✅ | 99359.08 | 382.72 | block-diffusion MoE; the decode rate reflects parallel canvas denoising and is not comparable to autoregressive rows |
| llada2.0-mini-preview-4bit | ✅ | 69787.19 | 1147.45 | block-diffusion MoE; the decode rate reflects parallel canvas denoising and is not comparable to autoregressive rows |
| llama-4-scout-17b-16e-instruct-4bit | ⚪ | - | - | memory-gate skip on 2026-11-03 |
| mixtral-8x7b-4bit | ✅ | 352.33 | 28.34 |  |
| olmoe-1b-7b-0924-nvfp4-gs | ✅ | 1276.43 | 244.79 |  |
| phi-3.5-moe-4bit | ✅ | 403.55 | 48.76 |  |
| qwen1.5-moe-a2.7b-4bit | ✅ | 1181.74 | 119.99 |  |
| qwen3-30b-a3b-4bit | ✅ | 608.58 | 84.59 |  |
| qwen3-30b-a3b-nvfp4 | ✅ | 1968.49 | 61.49 |  |
| qwen3-coder-480b-a35b-instruct-4bit | ⚪ | - | - | memory-gate skip on 2026-11-03 |

## MLA (Multi-head Latent Attention)

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| minicpm3-4b-4bit | ✅ | 5254.39 | 58.59 |  |

## DeepSeek Family

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| deepseek-coder-1.3b-4bit | ✅ | 22956.31 | 82.15 |  |
| deepseek-r1-distill-7b-4bit | ✅ | 3751.90 | 49.58 |  |
| deepseek-v3-0324-4bit | ⚪ | - | - | memory-gate skip on 2026-11-03 |

## Nemotron Family

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| llama-3.3-nemotron-super-49b-v1-4bit | ✅ | 605.85 | 7.81 |  |
| nemotron-3-nano-omni-30b-a3b-reasoning-4bit | ✅ | 212.31 | 95.16 |  |
| nemotron-h-30b-4bit | ✅ | 206.87 | 93.93 |  |

## SSM / Mamba / Hybrid

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| ai21-jamba-reasoning-3b-4bit | ✅ | 6270.36 | 113.29 |  |
| falcon-h1-tiny-90m-instruct-4bit | ✅ | 13425.63 | 370.52 |  |
| falcon-mamba-7b-4bit | ✅ | 182.58 | 43.59 |  |
| lfm2-350m-8bit | ✅ | 49256.66 | 472.80 |  |
| mamba2-1.3b-4bit | ✅ | 3083.69 | 88.15 |  |
| mamba2-130m | ✅ | 17098.27 | 174.55 |  |
| plamo-2-1b | ✅ | 5733.75 | 44.99 |  |

## Chinese / Asian Language Models

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| baichuan-m1-14b-4bit | ✅ | 2039.70 | 28.64 |  |
| ernie-4.5-0.3b-4bit | ✅ | 51563.08 | 636.31 |  |
| glm-4.7-flash-4bit | ✅ | 527.37 | 50.22 |  |
| hunyuan-1.8b-4bit | ✅ | 12283.19 | 171.15 |  |
| hunyuan-a13b-instruct-4bit | ✅ | 201.57 | 17.54 |  |
| internlm2_5-7b-chat-4bit | ✅ | 4036.33 | 50.28 |  |
| internlm3-8b-4bit | ✅ | 3428.56 | 44.77 |  |
| seed-oss-36b-instruct-4bit | ✅ | 501.14 | 10.22 |  |
| xverse-7b-chat-4bit | ✅ | 4390.66 | 51.61 |  |
| glm-5.1-4bit | ⚪ | - | - | weights not downloaded (empty directory) |
| glm-5-4bit | ⚪ | - | - | incomplete checkpoint: weights present, tokenizer missing |

## Mistral Family

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| ministral-3-3b-instruct-2512-4bit | ✅ | 9311.25 | 104.40 |  |
| mistral-small-3.1-24b-4bit | ✅ | 1070.31 | 16.16 |  |
| mistral-small-4-119b-2603-4bit | ⚪ | - | - | memory-gate skip on 2026-11-03 |
| pixtral-12b | ✅ | 2709.86 | 33.35 |  |

## BitNet

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| bitnet-b1.58-2b-4t | ✅ | 845.05 | 140.08 | re-measured 2026-11-04 after the CUDA grid-axis fix; 128 generated tokens on a 512-token prompt |
| bitnet-b1.58-2b-4t-4bit | ✅ | 843.60 | 191.67 | re-measured 2026-11-04 after the CUDA grid-axis fix; 128 generated tokens on a 512-token prompt |

## VLM-capable Models (text-only pass)

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| aya-vision-8b | ✅ | 4240.76 | 50.24 |  |
| bunny-llama3-8b-4bit | ✅ | 4242.85 | 52.81 |  |
| deepseek-ocr-2-4bit | ✅ | 5171.74 | 450.07 |  |
| deepseek-ocr-4bit | ✅ | 5343.23 | 425.10 |  |
| deepseek-vl2-small-4bit | ✅ | 255.52 | 82.19 |  |
| docling-layout-heron-mlx-bf16 | ⚪ | - | - | document-layout checkpoint; not a text generator |
| dots.ocr-4bit | ✅ | 16241.25 | 207.28 |  |
| falcon-ocr | ✅ | 17945.24 | 188.23 |  |
| fastvlm-0.5b-bf16 | ✅ | 40495.75 | 205.70 |  |
| glm-4.1v-9b-thinking-4bit | ✅ | 3102.98 | 40.80 |  |
| glm-4.5v-4bit | ⚪ | - | - | memory-gate skip on 2026-11-03 |
| glm-ocr-4bit | ✅ | 33351.04 | 417.21 |  |
| got-ocr2_0-4bit | ✅ | 35281.63 | 444.84 |  |
| got-ocr2_0-bf16 | ✅ | 46615.90 | 212.76 |  |
| got-ocr2_0-original | ✅ | 46326.12 | 211.85 |  |
| granite-4.0-3b-vision-4bit | ✅ | 7088.08 | 99.22 |  |
| granite-vision-3.2-2b-4bit | ✅ | 9011.29 | 128.87 |  |
| idefics2-8b-4bit | ✅ | 4178.67 | 52.08 |  |
| idefics3-8b-llama3-4bit | ✅ | 3961.90 | 49.61 |  |
| internvl3-1b | ✅ | 33530.06 | 482.49 |  |
| jagle-vl-2.2b-jagle-finevision | ✅ | 16142.44 | 65.14 |  |
| jina-vlm-mlx | ✅ | 13095.81 | 175.98 |  |
| kimi-vl-a3b-thinking-4bit | ✅ | 317.49 | 84.37 |  |
| lfm2-vl-1.6b-4bit | ✅ | 19609.59 | 289.94 |  |
| lfm2-vl-450m-4bit | ✅ | 46298.60 | 671.06 |  |
| lfm2.5-vl-1.6b-4bit | ✅ | 19941.13 | 293.99 |  |
| lfm2.5-vl-3b-optiq-4bit | ✅ | 8992.37 | 106.63 |  |
| llama-3.2-11b-vision-instruct-4bit | ✅ | 3419.15 | 49.71 |  |
| llava-1.5-7b-4bit | ✅ | 4496.85 | 53.85 |  |
| llava-interleave-qwen-0.5b-bf16 | ✅ | 48456.97 | 215.76 |  |
| llava-next-mistral-7b-4bit | ✅ | 4265.03 | 52.22 |  |
| llm-jp-4-vl-9b-beta | ✅ | 4196.14 | 14.89 |  |
| locateanything-3b-4bit | ✅ | 7293.26 | 95.30 |  |
| minicpm-o-4.5-4bit | ✅ | 3832.85 | 49.08 |  |
| minicpm-v-4.6-bf16 | ✅ | 7552.61 | 128.93 |  |
| minicpm-v-4.6-mxfp4 | ✅ | 7559.85 | 160.84 |  |
| molmo-7b | ✅ | 4044.65 | 54.36 |  |
| molmo2-4b | ✅ | 1262.19 | 44.39 |  |
| moondream2 | ✅ | 12394.66 | 84.23 |  |
| moondream3-preview-mlx-4bit | ✅ | 824.59 | 104.20 |  |
| north-micro-vision-instruct-4bit | ✅ | 11864.76 | 158.17 |  |
| paddleocr-vl-bfloat16 | ✅ | 37240.11 | 219.09 |  |
| paligemma2-3b-6bit | ✅ | 5722.64 | 87.97 | image-only PaliGemma; the text pass still generated 128 tokens; see the VLM table |
| phi-3.5-vision-4bit | ✅ | 7460.36 | 91.45 |  |
| qwen2-vl-2b | ✅ | 16488.24 | 208.80 |  |
| qwen2.5-vl-3b-4bit | ✅ | 7844.54 | 111.36 |  |
| qwen2.5-vl-3b-instruct | ✅ | 9537.89 | 38.13 |  |
| qwen3-omni-30b-a3b-instruct-4bit | ✅ | 605.19 | 83.43 |  |
| qwen3-vl-2b | ✅ | 13419.08 | 182.80 |  |
| qwen3-vl-2b-instruct | ✅ | 16611.33 | 65.67 |  |
| qwen3-vl-30b-a3b-4bit | ✅ | 562.03 | 82.92 |  |
| qwen3-vl-32b-4bit | ✅ | 914.12 | 11.37 |  |
| qwen3-vl-4b-4bit | ✅ | 6583.39 | 85.09 |  |
| qwen3-vl-8b-4bit | ✅ | 3836.48 | 49.35 |  |
| qwen3-vl-reranker-2b | ✅ | 16600.66 | 65.62 |  |
| smolvlm-instruct-bf16 | ✅ | 16744.17 | 65.92 |  |
| youtu-vl-4b-instruct | ✅ | 5345.41 | 22.18 |  |

## Qwen3.5 / Qwen3.6 / Qwen3.8 / Qwen3-next

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| qwen3-next-80b-a3b-instruct-4bit | ✅ | 226.29 | 51.97 |  |
| qwen3.5-0.8b-4bit | ✅ | 7337.01 | 271.53 |  |
| qwen3.5-0.8b-optiq-4bit | ✅ | 7131.61 | 221.72 |  |
| qwen3.5-27b-4bit | ✅ | 599.42 | 13.40 |  |
| qwen3.5-27b-dflash | ⚪ | - | - | not a standalone text generator |
| qwen3.5-2b-4bit | ✅ | 6025.60 | 155.03 |  |
| qwen3.5-35b-a3b-4bit | ✅ | 464.00 | 73.11 |  |
| qwen3.5-4b-4bit | ✅ | 2547.36 | 75.21 |  |
| qwen3.5-4b-dflash | ⚪ | - | - | not a standalone text generator |
| qwen3.5-9b-4bit | ✅ | 1976.39 | 44.57 |  |
| qwen3.5-9b-bf16 | ✅ | 1787.32 | 14.13 |  |
| qwen3.6-35b-a3b-4bit | ✅ | 455.67 | 70.87 |  |
| qwen3.8-27b-4bit | ✅ | 604.09 | 13.44 |  |
| qwen3.8-27b-fp8 | ✅ | 558.56 | 7.90 |  |

## Solar

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| solar-open-100b-4bit | ⚪ | - | - | memory-gate skip on 2026-11-03 |

## Speech / Audio (not applicable)

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| kokoro-82m | ⚪ | - | - | TTS checkpoint; not a text generator |
| whisper-base | ⚪ | - | - | speech-recognition checkpoint; not a text generator |

## VLM Benchmark (image input)

Models that accept image input and generated tokens under the `"What is in this image?"` prompt with `tests/fixtures/test_image.png`. Every engine row generated 128 tokens (`--ignore-eos`); the diffusion row generated 28 before EOS.

| Model | Status | Prefill (tok/s) | Decode (tok/s) |
|-------|--------|-----------------|----------------|
| aya-vision-8b | ✅ | 3584.17 | 43.57 |
| bunny-llama3-8b-4bit | ✅ | 2619.83 | 48.87 |
| deepseek-ocr-2-4bit | ✅ | 1030.82 | 408.12 |
| deepseek-ocr-4bit | ✅ | 930.21 | 404.59 |
| deepseek-vl2-small-4bit | ✅ | 237.01 | 101.49 |
| diffusiongemma-26b-a4b-it-4bit | ✅ | 408.89 | 7.76 |
| dots.ocr-4bit | ✅ | 1775.75 | 214.16 |
| fastvlm-0.5b-bf16 | ✅ | 3199.53 | 210.33 |
| gemma-3n-e4b-bf16 | ✅ | 2039.65 | 24.17 |
| gemma-4-12b-it-4bit | ✅ | 1505.94 | 14.63 |
| gemma-4-26b-a4b-it-4bit | ✅ | 372.67 | 65.31 |
| gemma-4-26b-a4b-it-qat-4bit | ✅ | 365.72 | 59.04 |
| gemma-4-31b-4bit | ✅ | 524.68 | 9.24 |
| gemma-4-31b-it-4bit | ✅ | 541.44 | 9.23 |
| gemma-4-31b-it-nvfp4 | ✅ | 448.14 | 5.09 |
| gemma-4-31b-it-qat-4bit | ✅ | 489.33 | 5.81 |
| gemma-4-e2b-it-4bit | ✅ | 2907.46 | 105.46 |
| gemma-4-e2b-it-8bit | ✅ | 2709.85 | 59.79 |
| gemma-4-e2b-it-qat-4bit | ✅ | 2813.52 | 70.69 |
| gemma-4-e4b-it-4bit | ✅ | 2117.06 | 54.64 |
| gemma-4-e4b-it-8bit | ✅ | 1888.05 | 30.23 |
| gemma-4-e4b-it-qat-4bit | ✅ | 1968.54 | 35.28 |
| gemma3-4b-4bit | ✅ | 1253.40 | 86.99 |
| gemma3n-e2b-4bit | ✅ | 2959.35 | 99.51 |
| gemma3n-e4b-4bit | ✅ | 2225.56 | 60.73 |
| glm-4.1v-9b-thinking-4bit | ✅ | 1028.69 | 40.92 |
| glm-ocr-4bit | ✅ | 3201.13 | 414.10 |
| got-ocr2_0-4bit | ✅ | 1339.01 | 461.28 |
| granite-4.0-3b-vision-4bit | ✅ | 4388.01 | 98.37 |
| granite-vision-3.2-2b-4bit | ✅ | 7118.69 | 120.53 |
| idefics2-8b-4bit | ✅ | 454.74 | 52.81 |
| idefics3-8b-llama3-4bit | ✅ | 2063.60 | 46.07 |
| internvl3-1b | ✅ | 6264.04 | 503.18 |
| jagle-vl-2.2b-jagle-finevision | ✅ | 5105.92 | 65.22 |
| jina-vlm-mlx | ✅ | 4954.21 | 170.44 |
| kimi-vl-a3b-thinking-4bit | ✅ | 291.96 | 86.27 |
| lfm2-vl-1.6b-4bit | ✅ | 3432.08 | 289.00 |
| lfm2-vl-450m-4bit | ✅ | 8681.24 | 690.58 |
| lfm2.5-vl-1.6b-4bit | ✅ | 3570.79 | 290.05 |
| lfm2.5-vl-3b-optiq-4bit | ✅ | 2354.36 | 105.48 |
| llama-3.2-11b-vision-instruct-4bit | ✅ | 13.37 | 40.43 |
| llava-1.5-7b-4bit | ✅ | 3676.77 | 54.31 |
| llava-interleave-qwen-0.5b-bf16 | ✅ | 21109.36 | 199.11 |
| llava-next-mistral-7b-4bit | ✅ | 3586.80 | 52.34 |
| llm-jp-4-vl-9b-beta | ✅ | 2109.28 | 14.84 |
| locateanything-3b-4bit | ✅ | 2230.61 | 96.55 |
| minicpm-o-4.5-4bit | ✅ | 992.75 | 50.12 |
| minicpm-v-4.6-bf16 | ✅ | 1421.07 | 128.41 |
| minicpm-v-4.6-mxfp4 | ✅ | 1346.85 | 164.22 |
| ministral-3-3b-instruct-2512-4bit | ✅ | 7341.27 | 104.05 |
| mistral-small-3.1-24b-4bit | ✅ | 1208.73 | 16.02 |
| molmo-7b | ✅ | 2262.65 | 54.56 |
| molmo2-4b | ✅ | 762.30 | 41.75 |
| moondream2 | ✅ | 76.28 | 80.87 |
| moondream3-preview-mlx-4bit | ✅ | 4.60 | 102.07 |
| muse-glimmer-30b-4bit | ✅ | 555.91 | 13.80 |
| muse-glimmer-30b-optiq-4bit | ✅ | 525.12 | 11.43 |
| nemotron-3-nano-omni-30b-a3b-reasoning-4bit | ✅ | 239.69 | 94.37 |
| north-micro-vision-instruct-4bit | ✅ | 2950.37 | 169.46 |
| paddleocr-vl-bfloat16 | ✅ | 5175.91 | 202.44 |
| paligemma2-3b-6bit | ✅ | 4021.85 | 82.66 |
| phi-3.5-vision-4bit | ✅ | 4277.27 | 86.85 |
| pixtral-12b | ✅ | 1625.07 | 32.68 |
| qwen2-vl-2b | ✅ | 2697.23 | 183.65 |
| qwen2.5-vl-3b-4bit | ✅ | 1830.57 | 103.47 |
| qwen2.5-vl-3b-instruct | ✅ | 1577.65 | 36.80 |
| qwen3-omni-30b-a3b-instruct-4bit | ✅ | 322.07 | 72.72 |
| qwen3-vl-2b | ✅ | 3012.78 | 170.54 |
| qwen3-vl-2b-instruct | ✅ | 2428.27 | 63.13 |
| qwen3-vl-30b-a3b-4bit | ✅ | 382.18 | 73.63 |
| qwen3-vl-32b-4bit | ✅ | 324.43 | 11.19 |
| qwen3-vl-4b-4bit | ✅ | 1734.54 | 80.82 |
| qwen3-vl-8b-4bit | ✅ | 1193.32 | 47.95 |
| qwen3-vl-reranker-2b | ✅ | 2376.81 | 63.32 |
| qwen3.5-0.8b-4bit | ✅ | 1243.15 | 210.22 |
| qwen3.5-27b-4bit | ✅ | 254.75 | 13.25 |
| qwen3.5-2b-4bit | ✅ | 1113.53 | 148.53 |
| qwen3.5-35b-a3b-4bit | ✅ | 384.35 | 65.33 |
| qwen3.5-4b-4bit | ✅ | 819.16 | 72.34 |
| qwen3.5-9b-4bit | ✅ | 679.30 | 43.60 |
| qwen3.5-9b-bf16 | ✅ | 477.38 | 14.07 |
| qwen3.6-35b-a3b-4bit | ✅ | 387.34 | 67.93 |
| qwen3.8-27b-4bit | ✅ | 409.58 | 13.08 |
| qwen3.8-27b-fp8 | ✅ | 350.77 | 7.83 |
| smolvlm-instruct-bf16 | ✅ | 2512.88 | 66.99 |
| youtu-vl-4b-instruct | ✅ | 1049.27 | 22.34 |

Not measured in the image pass: ten non-generative checkpoints fail as expected (four ColPali/ColSmol retrievers, `docling-layout-heron-mlx-bf16`, `granite-speech-4.1-2b-nar-bf16`, `llama-nemotron-embed-vl-1b-v2`, `qwen3-vl-embedding-2b`, `siglip-base-patch16-224`, `whisper-base`), `gemma3-1b-4bit` has no vision tower, and `gemma-4-12b-it-assistant-4bit` is a drafter. Four checkpoints sit above the memory gate: `glm-4.5v-4bit`, `llama-4-scout-17b-16e-instruct-4bit`, `mistral-small-4-119b-2603-4bit` and `muse-glimmer-30b`. `ming-image-0.1-design` has no weights.

The `diffusiongemma-26b-a4b-it-4bit` image row was re-measured on 2026-11-04 through the model's own diffusion loop (277 prompt tokens, 28 generated before EOS, 7.76 tok/s). As in the text pass, its decode rate is not comparable to the autoregressive VLM rows.

---

## Summary

**Test date**: 2026-11-03 (text), 2026-11-04 (VLM) | **Hardware**: NVIDIA GB10 (DGX Spark) | **mlxcel**: 0.8.1 at `630aaeba` (CUDA 13.0, SM 12.1) | **MLX pin**: `7ca40033`

| Metric | Count |
|--------|-------|
| **Total model directories** | 312 |
| **Checkpoints (config.json plus readable shards)** | 286 |
| **Measured, text pass (✅)** | 172 (169 from the sweep plus three re-measured rows) |
| **Measured, VLM image pass (✅)** | 86 (85 from the sweep plus the re-measured DiffusionGemma row) |
| **Failed text pass** | 36 (all non-generative checkpoints: embedders, rerankers, classifiers, speech, layout, and MTP/DFlash drafters). The 2026-11-03 CSV records 39; the three extra rows are the fixed checkpoints re-measured on 11-04 |
| **Failed VLM pass** | 12 (all non-generative checkpoints). The 2026-11-04 VLM sweep CSV records 13; the extra row is the fixed DiffusionGemma checkpoint |
| **Memory-gate skips** | 9 text, 4 VLM |
| **Checkpoint dedup aliases** | 68 text, 26 VLM |
| **Not a checkpoint / missing weights** | 26 / 1 text, 1 VLM |

### Notable changes vs the 2026-07-12 (0.4.0-rc.1) record

- **Decode holds or improves across the sweep.** 111 models appear in both sweeps. Median decode ratio is 1.026, with the 10th percentile at 0.928 and the 90th at 1.364. The VLM pass is stronger: 52 paired rows, median 1.081, 10th percentile 1.012.
- **Large decode gains.** `lfm2-350m-8bit` reads 39.84 to 472.80 tok/s (11.87x); the old CSV carried the pre-#748 value. `granite-4.1-8b-4bit` 18.43 to 39.23 (2.13x), `phi-2-4bit` 36.56 to 74.08 (2.03x), `falcon-mamba-7b-4bit` 22.06 to 43.59 (1.98x), `qwen2-vl-2b` 107.81 to 208.80 (1.94x), `granite-4.0-h-350m-4bit` 171.22 to 320.63 (1.87x), `qwen3.5-0.8b-4bit` 178.96 to 271.53 (1.52x).
- **VLM gains.** `qwen2-vl-2b` 1.99x, `paligemma2-3b-6bit` 1.75x, `qwen2.5-vl-3b-4bit` 1.72x, `llama-3.2-11b-vision-instruct-4bit` 1.66x, `molmo2-4b` 1.53x, `molmo-7b` 1.52x. No VLM row regressed.
- **Two small models read below their old values.** `stablelm-1.6b-4bit` 203.75 to 123.78 (0.61x) and `smollm-135m-4bit` 656.73 to 474.99 (0.72x). Both are launch-bound at this size, where this host has a documented run-to-run envelope of roughly ±25% and occasionally more; a single sweep run is not evidence of a code regression. They need repeats before anyone reads the drop as real.
- **`deepseek-v2-lite-4bit` now produces coherent output.** A 32-token `mlxcel generate` run on this host returned a normal reply. The garbage-output report from 2026-07 no longer reproduces at 0.8.1.
- **57 text checkpoints and 32 VLM checkpoints were measured for the first time** (model directories added since 2026-07-12).

### Failures found by the sweep, and the fixes that closed them

Three checkpoints measured on 0.4.0-rc.1 failed at 0.8.1, for two different reasons. Both were fixed on 2026-11-04 and the rows re-measured; the numbers in the tables above are the post-fix values.

- **`bitnet-b1.58-2b-4t` and `bitnet-b1.58-2b-4t-4bit`** failed in warmup once the prefill reached 48 tokens: `cuGraphAddKernelNode ... invalid argument` with graphs enabled, `cuLaunchKernelEx ... invalid argument` with `MLX_USE_CUDA_GRAPHS=0`. 32 tokens and below passed, which matched the launch shape: `bitlinear_matmul` put one block per `(batch, out/4)` group on `grid.y`, and CUDA caps that axis at 65535. BitNet-2B's 6912-wide FFN gives 1728 groups per row, so 32 prompt tokens (55296) fit and 48 (82944) did not. The fix moves the group index to `grid.x` (cap 2^31-1) and the 32 lanes to the y axis in all three ports; the Metal source and launch tuple changed with it. A regression test (`bitlinear_matmul_launches_beyond_the_cuda_grid_y_limit`, 40000 rows x 2 groups = 80000 blocks) covers the limit on tiny tensors. Post-fix: 845.05 / 140.08 tok/s and 843.60 / 191.67 tok/s on the 512-token shape.
- **`diffusiongemma-26b-a4b-it-4bit`** panicked in the engine bench path at `src/models/diffusion_gemma/mod.rs:846` (`index out of bounds: the len is 0 but the index is 0`), for both 16-token and 512-token prompts. The engine allocates an empty cache slice for a model-owned family, and `LanguageModel::forward` was the only entry the harness used. The server does not serve this family through the engine at all: it runs the model-owned block-diffusion loop. The harness now drives that same loop when the loaded model is `DiffusionGemma` (text and image), and records `decode_path=diffusion` so the row cannot be mistaken for an engine measurement. Post-fix: 797.61 prefill / 6.86 decode tok/s for 128 text tokens, and 408.89 / 7.76 for 28 image-prompt tokens.

Both fixes were uncommitted when the rows were measured, so the CSV records `630aaeba-dirty`.

### Memory-gate exclusions (2026-11-03)

`BENCH_MEM_OVERHEAD_FACTOR=2.0` against the 85% system-memory budget (105916 MB at run time) puts the text cutoff near 51 GiB. Nine text checkpoints and four VLM checkpoints were skipped without a measurement. Runs at that scale have repeatedly destabilized the GB10 driver in earlier campaigns, so the gate is deliberate.

Text: `deepseek-v3-0324-4bit`, `dots.llm1.inst-mixed-4-6bit`, `glm-4.5v-4bit`, `gpt-oss-120b-4bit`, `llama-4-scout-17b-16e-instruct-4bit`, `mistral-small-4-119b-2603-4bit`, `muse-glimmer-30b`, `qwen3-coder-480b-a35b-instruct-4bit`, `solar-open-100b-4bit`.

VLM: `glm-4.5v-4bit`, `llama-4-scout-17b-16e-instruct-4bit`, `mistral-small-4-119b-2603-4bit`, `muse-glimmer-30b`.

### Checkpoint dedup

The sweep collapses directories that resolve to the same canonical path, or that share an identity key built from `config.json` and the sorted shard names and sizes. 68 text directories and 26 VLM directories were aliases; each group was measured once under the survivor.

- `ai21-jamba-reasoning-3b-4bit` also covers `jamba-v0.1-4bit`
- `aya-vision-8b` also covers `aya-vision-8b-4bit`
- `baichuan-m1-14b-4bit` also covers `baichuan-m1-14b-instruct-4bit`
- `bunny-llama3-8b-4bit` also covers `bunny-llama-3-8b-v-4bit`
- `c4ai-command-r7b-12-2024-4bit` also covers `command-r7b-4bit`
- `deepseek-coder-1.3b-4bit` also covers `deepseek-coder-1.3b-instruct-mlx`
- `deepseek-r1-distill-7b-4bit` also covers `deepseek-r1-distill-qwen-7b-4bit`
- `deepseek-v2-lite-4bit` also covers `deepseek-v2-lite-chat-4bit-mlx`
- `deepseek-v3-0324-4bit` also covers `deepseek-v3-4bit`
- `ernie-4.5-0.3b-4bit` also covers `ernie-4.5-0.3b-pt-4bit`
- `exaone-4.0-1.2b-4bit` also covers `exaone4-1.2b-4bit`
- `falcon-mamba-7b-4bit` also covers `falcon-mamba-7b-4bit-instruct`
- `gemma-1.1-2b-it-4bit` also covers `gemma-2b-4bit`
- `gemma-3n-e4b-bf16` also covers `gemma3n-e4b-bf16`
- `gemma2-2b-4bit` also covers `gemma-2-2b-it-4bit`
- `gemma3-1b-4bit` also covers `gemma-3-1b-it-4bit`
- `gemma3-4b-4bit` also covers `gemma-3-4b-it-4bit`
- `gemma3n-e2b-4bit` also covers `gemma-3n-e2b-it-4bit`
- `gemma3n-e4b-4bit` also covers `gemma-3n-e4b-it-4bit`
- `glm-4.7-flash-4bit` also covers `glm4-flash-4bit`
- `gpt-oss-20b-mxfp4` also covers `gpt-oss-20b-mxfp4-q4`
- `granite-speech-4.1-2b-nar-bf16` also covers `granite-speech-4.1-2b-nar-mlx`
- `internlm2_5-7b-chat-4bit` also covers `internlm2-7b-4bit`
- `internlm3-8b-4bit` also covers `internlm3-8b-instruct-4bit`
- `internvl3-1b` also covers `internvl3-1b-4bit`
- `llama-3.1-8b-4bit` also covers `meta-llama-3.1-8b-instruct-4bit`
- `llama-3.1-8b-bf16` also covers `llama-3.1-8b-instruct-bf16`
- `llama-3.2-1b-4bit` also covers `llama-3.2-1b-instruct-4bit`
- `llama-4-scout-17b-16e-instruct-4bit` also covers `llama-4-scout-17b-4bit`
- `llava-next-mistral-7b-4bit` also covers `llava-v1.6-mistral-7b-4bit`
- `mimo-7b-4bit` also covers `mimo-7b-rl-4bit`
- `minicpm-2b-4bit` also covers `minicpm-2b-sft-4bit-llama-format-mlx`
- `ministral-3-3b-instruct-2512-4bit` also covers `ministral-3b-4bit`
- `mistral-small-3.1-24b-4bit` also covers `mistral-small-3.1-24b-instruct-2503-4bit`
- `mixtral-8x7b-4bit` also covers `mixtral-8x7b-instruct-v0.1-4bit`
- `molmo-7b` also covers `molmo-7b-d-0924-4bit`
- `nemotron-h-30b-4bit` also covers `nemotron-nas-30b-4bit`, `nvidia-nemotron-3-nano-30b-a3b-4bit`
- `olmo-1b-4bit` also covers `olmo-1b-hf-4bit-mlx`
- `olmo-2-1124-7b-instruct-4bit` also covers `olmo2-7b-4bit`
- `olmo-3.1-32b-think-4bit` also covers `olmo3-32b-4bit`
- `paligemma2-3b-6bit` also covers `paligemma2-3b-ft-docci-448-6bit`
- `phi-2-4bit` also covers `phi-2-hf-4bit-mlx`
- `phi-3-mini-4bit` also covers `phi-3-mini-4k-instruct-4bit`
- `phi-3.5-mini-4bit` also covers `phi-3.5-mini-instruct-4bit`
- `phi-3.5-moe-4bit` also covers `phi-3.5-moe-instruct-4bit`
- `phi-3.5-vision-4bit` also covers `phi-3.5-vision-instruct-4bit`
- `pixtral-12b` also covers `pixtral-12b-4bit`
- `qwen1.5-moe-a2.7b-4bit` also covers `qwen1.5-moe-a2.7b-chat-4bit`
- `qwen2-0.5b` also covers `qwen2.5-0.5b-4bit`, `qwen2.5-0.5b-instruct-4bit`
- `qwen2-vl-2b` also covers `qwen2-vl-2b-4bit`, `qwen2-vl-2b-instruct-4bit`
- `qwen2.5-0.5b-bf16` also covers `qwen2.5-0.5b-instruct-bf16`
- `qwen2.5-7b` also covers `qwen2.5-7b-4bit`, `qwen2.5-7b-instruct-4bit`
- `qwen2.5-7b-8bit` also covers `qwen2.5-7b-instruct-8bit`
- `qwen2.5-vl-3b-4bit` also covers `qwen2.5-vl-3b-instruct-4bit`
- `qwen3-0.6b` also covers `qwen3-0.6b-4bit`
- `qwen3-30b-a3b-4bit` also covers `qwen3-moe-4bit`
- `qwen3-vl-2b` also covers `qwen3-vl-2b-4bit`, `qwen3-vl-2b-instruct-4bit`
- `qwen3-vl-30b-a3b-4bit` also covers `qwen3-vl-30b-a3b-instruct-4bit`
- `qwen3-vl-32b-4bit` also covers `qwen3-vl-32b-instruct-4bit`
- `qwen3-vl-4b-4bit` also covers `qwen3-vl-4b-instruct-4bit`
- `qwen3-vl-8b-4bit` also covers `qwen3-vl-8b-instruct-4bit`
- `smollm-135m-4bit` also covers `smollm-135m-instruct-4bit`
- `stablelm-1.6b-4bit` also covers `stablelm-2-1_6b-chat-4bit`

VLM aliases: `aya-vision-8b` covers `aya-vision-8b-4bit`; `bunny-llama3-8b-4bit` covers `bunny-llama-3-8b-v-4bit`; `gemma-3n-e4b-bf16` covers `gemma3n-e4b-bf16`; `gemma3-1b-4bit` covers `gemma-3-1b-it-4bit`; `gemma3-4b-4bit` covers `gemma-3-4b-it-4bit`; `gemma3n-e2b-4bit` covers `gemma-3n-e2b-it-4bit`; `gemma3n-e4b-4bit` covers `gemma-3n-e4b-it-4bit`; `granite-speech-4.1-2b-nar-bf16` covers `granite-speech-4.1-2b-nar-mlx`; `internvl3-1b` covers `internvl3-1b-4bit`; `llama-4-scout-17b-16e-instruct-4bit` covers `llama-4-scout-17b-4bit`; `llava-next-mistral-7b-4bit` covers `llava-v1.6-mistral-7b-4bit`; `ministral-3-3b-instruct-2512-4bit` covers `ministral-3b-4bit`; `mistral-small-3.1-24b-4bit` covers `mistral-small-3.1-24b-instruct-2503-4bit`; `molmo-7b` covers `molmo-7b-d-0924-4bit`; `paligemma2-3b-6bit` covers `paligemma2-3b-ft-docci-448-6bit`; `phi-3.5-vision-4bit` covers `phi-3.5-vision-instruct-4bit`; `pixtral-12b` covers `pixtral-12b-4bit`; `qwen2-vl-2b` covers `qwen2-vl-2b-4bit`, `qwen2-vl-2b-instruct-4bit`; `qwen2.5-vl-3b-4bit` covers `qwen2.5-vl-3b-instruct-4bit`; `qwen3-vl-2b` covers `qwen3-vl-2b-4bit`, `qwen3-vl-2b-instruct-4bit`; `qwen3-vl-30b-a3b-4bit` covers `qwen3-vl-30b-a3b-instruct-4bit`; `qwen3-vl-32b-4bit` covers `qwen3-vl-32b-instruct-4bit`; `qwen3-vl-4b-4bit` covers `qwen3-vl-4b-instruct-4bit`; `qwen3-vl-8b-4bit` covers `qwen3-vl-8b-instruct-4bit`.

### Newly measured checkpoints

Text pass, first measurement on this host (57): `ai21-jamba-reasoning-3b-4bit`, `c4ai-command-r7b-12-2024-4bit`, `deepseek-ocr-2-4bit`, `deepseek-ocr-4bit`, `deepseek-vl2-small-4bit`, `dots.ocr-4bit`, `exaone-4.0-1.2b-4bit`, `falcon-ocr`, `fastvlm-0.5b-bf16`, `gemma-1.1-2b-it-4bit`, `gemma-3n-e4b-bf16`, `glm-4.7-flash-4bit`, `glm-ocr-4bit`, `got-ocr2_0-4bit`, `got-ocr2_0-bf16`, `got-ocr2_0-original`, `granite-4.0-3b-vision-4bit`, `granite-vision-3.2-2b-4bit`, `idefics2-8b-4bit`, `idefics3-8b-llama3-4bit`, `internlm2_5-7b-chat-4bit`, `iquest-coder-v1-40b-loop-instruct-4bit`, `iquest-coder-v1-7b-instruct-8bit`, `jagle-vl-2.2b-jagle-finevision`, `jina-vlm-mlx`, `laguna-xs-2.1-nvfp4`, `lfm2-vl-1.6b-4bit`, `lfm2-vl-450m-4bit`, `lfm2.5-vl-1.6b-4bit`, `lfm2.5-vl-3b-optiq-4bit`, `llada2.0-mini`, `llada2.0-mini-preview-4bit`, `llama-3.2-1b-instruct-bf16`, `llama-3.3-nemotron-super-49b-v1-4bit`, `llm-jp-4-vl-9b-beta`, `locateanything-3b-4bit`, `minicpm-o-4.5-4bit`, `ministral-3-3b-instruct-2512-4bit`, `moondream3-preview-mlx-4bit`, `muse-glimmer-30b-4bit`, `muse-glimmer-30b-optiq-4bit`, `nanochat-d20-q8-mlx`, `north-micro-vision-instruct-4bit`, `olmo-2-1124-7b-instruct-4bit`, `olmo-3.1-32b-think-4bit`, `olmoe-1b-7b-0924-nvfp4-gs`, `phi-3-small-8k-instruct-aq4_64`, `qwen2.5-vl-3b-instruct`, `qwen3-30b-a3b-nvfp4`, `qwen3-next-80b-a3b-instruct-4bit`, `qwen3-omni-30b-a3b-instruct-4bit`, `qwen3-reranker-0.6b-4bit`, `qwen3-vl-2b-instruct`, `qwen3-vl-reranker-2b`, `qwen3.8-27b-4bit`, `qwen3.8-27b-fp8`, `xverse-7b-chat-4bit`.

Image pass, first measurement on this host (32): `deepseek-ocr-2-4bit`, `deepseek-ocr-4bit`, `deepseek-vl2-small-4bit`, `dots.ocr-4bit`, `fastvlm-0.5b-bf16`, `gemma-3n-e4b-bf16`, `glm-ocr-4bit`, `got-ocr2_0-4bit`, `granite-4.0-3b-vision-4bit`, `granite-vision-3.2-2b-4bit`, `idefics2-8b-4bit`, `idefics3-8b-llama3-4bit`, `jagle-vl-2.2b-jagle-finevision`, `jina-vlm-mlx`, `lfm2-vl-1.6b-4bit`, `lfm2-vl-450m-4bit`, `lfm2.5-vl-1.6b-4bit`, `lfm2.5-vl-3b-optiq-4bit`, `llm-jp-4-vl-9b-beta`, `locateanything-3b-4bit`, `minicpm-o-4.5-4bit`, `ministral-3-3b-instruct-2512-4bit`, `moondream3-preview-mlx-4bit`, `muse-glimmer-30b-4bit`, `muse-glimmer-30b-optiq-4bit`, `north-micro-vision-instruct-4bit`, `qwen2.5-vl-3b-instruct`, `qwen3-omni-30b-a3b-instruct-4bit`, `qwen3-vl-2b-instruct`, `qwen3-vl-reranker-2b`, `qwen3.8-27b-4bit`, `qwen3.8-27b-fp8`.

### Measurement notes

- **Small checkpoints are launch-bound and swing between runs.** The `#755` campaign on this host recorded identical same-boot runs of `mamba2-130m` at 147.98, 202.98 and 204.77 tok/s. Treat single-run decode deltas inside roughly ±25% on sub-1B dense, SSM and small MoE checkpoints as inside the noise floor.
- **`glm-4.7-flash-4bit` (alias `glm4-flash-4bit`) has a bimodal decode distribution** (about 41 and 52.5 tok/s modes across identical runs). The 2026-11-03 row is one draw, not a trend.
- **The engine bench path measures what the server uses.** `mlxcel-bench-decode` runs the engine's raw-completion client, so a passing row exercises the same decode path the server serves. `diffusiongemma` is the exception: the server runs it on the model-owned block-diffusion loop, so the harness drives that loop and marks the row `decode_path=diffusion`. Its tok/s counts committed tokens per second over canvas denoising and is not comparable to an autoregressive step.
- **`--ignore-eos` reaches the engine rows only.** The diffusion loop has no token-bias handle, so a diffusion row may stop at the checkpoint's EOS before the 128-token budget; the CSV's `generated_tokens` column carries the actual count (128 for the text recheck, 28 for the image recheck).
