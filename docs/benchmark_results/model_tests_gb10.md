# Model Compatibility & Performance Tests (NVIDIA GB10)

Compatibility and decode/prefill performance for mlxcel models on **NVIDIA GB10 (DGX Spark)** running the CUDA backend. This is the **0.4.0-rc.1** benchmark: a full sweep of all 159 model directories on 2026-07-12, with seven very large checkpoints excluded by an up-front memory gate (see the version note).

## Test Environment

| Item | Value |
|------|-------|
| **Hardware** | NVIDIA GB10 (DGX Spark), 122 GB unified LPDDR5x |
| **OS** | Linux (aarch64), kernel 6.17 |
| **Backend** | CUDA 13.0 (driver via `libcuda.so.1`, cuDNN 9) |
| **mlxcel version** | 0.4.0-rc.1 (full 159-directory sweep, 2026-07-12) |
| **MLX pin** | commit `57c66cac` / MLX 0.32.1 |
| **CUDA build** | `cargo build --release --features cuda` (arch auto-detect resolves GB10 = SM 12.1) |
| **Harness** | same-process `mlxcel-bench-decode`, warm prefill, pre-warm on, `--cooldown 15 --big-cooldown 45`, `BENCH_MEM_OVERHEAD_FACTOR=2.0` |
| **Test Prompt** | "Hello, how are you today?" (text) / "What is in this image?" (VLM) |
| **Max Tokens** | 100 |
| **Test Date** | 2026-07-12 |
| **CSVs** | `benchmarks/cuda_gb10_2026-07-12.csv`, `benchmarks/cuda_gb10_vlm_2026-07-12.csv` |

> Version note: every measured row in this file is from the 2026-07-12 sweep on mlxcel 0.4.0-rc.1. Seven checkpoints whose weights exceed ~51 GiB were excluded up front (`BENCH_MEM_OVERHEAD_FACTOR=2.0`, recorded as `SKIP:oom_estimate` in the CSV) because runs that push past ~70 GB have repeatedly destabilized the GB10 driver (`NV_ERR_NO_MEMORY` accumulation; see the 2026-07-06 freeze notes). The five of them that were measured on earlier sweeps keep those figures with a per-row note: `llama-4-scout-17b-4bit` (2026-07-09 / 0.4.0-rc.1), and `dots.llm1.inst-mixed-4-6bit`, `gpt-oss-120b-4bit`, `minimax-m2-3bit`, `solar-open-100b-4bit` (2026-06-17 / 0.3.1). `glm-4.5v-4bit` and `mistral-small-4-119b-2603-4bit` have never been measured on this host.
>
> Build note: a default `cargo build --release` on Linux produces a CPU-only binary (default features are `surgery` only), which silently runs MLX on the Grace CPU at ~0.4 tok/s. The GB10 benchmark must be built with `--features cuda`.
>
> Two directories present on 2026-06-17 were deleted from disk before this sweep and their rows are gone: `deepseek-v3-4bit` (incomplete 671B checkpoint, capacity-excluded anyway, tracked in #315) and `qwen3-next-480b-4bit` (capacity-excluded).

## Legend

- ✅ Pass: model loads and produces tokens within the configured budget
- ❌ Fail: warmup/bench failure or 0 tokens generated
- ⚪ Not tested / not applicable: weights not present or incomplete, the checkpoint is not a standalone text generator (image-only, speech, TTS, document-layout, or an MTP/DFlash drafter that needs a target), or the model was excluded by the memory gate without a prior measurement; not a code failure

Prefill/Decode are the measured-pass figures from `mlxcel-bench-decode`. Notes record an early-EOS token count (when a model stopped before 100) or the failure/skip cause. Models are grouped by architecture family; VLM-capable models appear once under text (text-prompt pass) and again in the image-input table at the end.

## Basic Transformers

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| apertus-8b-instruct-2509-4bit | ✅ | 1023.18 | 50.83 | 25 tok |
| llama-3.1-8b-4bit | ✅ | 1653.36 | 50.53 | 69 tok |
| llama-3.1-8b-bf16 | ✅ | 992.64 | 15.33 | 87 tok |
| llama-3.2-1b-4bit | ✅ | 7661.46 | 266.04 | 30 tok |
| meta-llama-3.1-8b-instruct-4bit | ✅ | 1680.88 | 51.38 |  |
| mimo-7b-4bit | ✅ | 354.57 | 54.01 |  |
| minicpm-2b-4bit | ✅ | 501.95 | 127.37 |  |
| olmo-1b-4bit | ✅ | 260.00 | 98.59 |  |
| olmo2-7b-4bit | ✅ | 304.55 | 52.99 | 27 tok |
| olmo3-32b-4bit | ✅ | 314.55 | 11.68 |  |
| phi-2-4bit | ✅ | 137.63 | 36.56 | 1 tok |
| phi-3.5-mini-4bit | ✅ | 274.22 | 92.93 | 40 tok |
| phi-3-mini-4bit | ✅ | 274.38 | 94.64 | 25 tok |
| phi-4-4bit | ✅ | 165.42 | 27.85 |  |
| qwen2-0.5b | ✅ | 3085.80 | 502.12 |  |
| qwen2.5-0.5b-4bit | ✅ | 3650.43 | 492.60 |  |
| qwen2.5-0.5b-bf16 | ✅ | 2491.03 | 203.24 |  |
| qwen2.5-1.5b-4bit | ✅ | 1253.70 | 207.80 |  |
| qwen2.5-1.5b-instruct-4bit | ✅ | 1513.57 | 205.96 |  |
| qwen2.5-7b | ✅ | 621.67 | 53.82 |  |
| qwen2.5-7b-4bit | ✅ | 576.27 | 54.56 |  |
| qwen2.5-7b-8bit | ✅ | 245.62 | 28.96 |  |
| qwen2.5-7b-instruct-4bit | ✅ | 608.21 | 55.65 |  |
| qwen3-0.6b | ✅ | 2510.71 | 283.90 | 9 tok |
| qwen3-0.6b-4bit | ✅ | 2722.70 | 329.14 | 9 tok |
| qwen3-1.7b-4bit | ✅ | 1328.46 | 171.81 | 14 tok |
| qwen3-4b-4bit | ✅ | 672.00 | 82.10 | 36 tok |
| qwen3-8b-4bit | ✅ | 421.01 | 48.49 | 33 tok |
| smollm-135m-4bit | ✅ | 2211.42 | 656.73 |  |
| smollm3-3b-4bit | ✅ | 2127.85 | 104.24 | 46 tok |
| stablelm-1.6b-4bit | ✅ | 1655.97 | 203.75 | 79 tok |
| starcoder2-3b-4bit | ✅ | 224.15 | 105.41 |  |

## Gemma Family

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| gemma-2-9b-8bit | ✅ | 119.74 | 22.54 |  |
| gemma-2b-4bit | ✅ | 573.76 | 100.25 | 49 tok |
| gemma-3-4b-it-4bit | ✅ | 398.22 | 90.62 | 86 tok |
| gemma2-2b-4bit | ✅ | 557.30 | 112.46 | 27 tok |
| gemma3-1b-4bit | ✅ | 1081.02 | 278.52 | 34 tok |
| gemma3-4b-4bit | ✅ | 378.18 | 90.92 | 86 tok |
| gemma3n-e2b-4bit | ✅ | 453.05 | 88.23 | 72 tok |
| gemma3n-e4b-4bit | ✅ | 290.81 | 57.51 | 71 tok |
| gemma3n-e4b-bf16 | ✅ | 275.11 | 22.49 | 69 tok |

## Gemma 4

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| diffusiongemma-26b-a4b-it-4bit | ✅ | 128.50 | 39.61 | 27 tok |
| gemma-4-12b-it-4bit | ✅ | 208.56 | 14.23 | 27 tok |
| gemma-4-12b-it-4bit-down8 | ✅ | 281.42 | 19.07 | 27 tok; local uniform-requant variant (#685 tooling) |
| gemma-4-12b-it-4bit-gs32 | ✅ | 215.87 | 21.61 | 40 tok; local uniform-requant variant (#685 tooling) |
| gemma-4-12b-it-4bit-uniform | ✅ | 192.16 | 21.15 | 27 tok; local uniform-requant variant (#685 tooling) |
| gemma-4-12b-it-assistant-4bit | ⚪ | - | - | MTP/DFlash drafter (needs a target; not standalone) |
| gemma-4-26b-a4b-it-4bit | ✅ | 134.35 | 59.88 | 26 tok; median of 3 post-reboot runs (#755): 59.57-60.07, back above the 0.3.1 record (58.59) |
| gemma-4-26b-a4b-it-qat-4bit | ✅ | 133.43 | 53.68 | 26 tok; median of 3 post-reboot runs (#755): 53.48-53.71, above the 0.3.1 record (50.33) |
| gemma-4-31b-4bit | ✅ | 23.08 | 8.84 |  |
| gemma-4-31b-it-4bit | ✅ | 52.37 | 8.33 | 26 tok |
| gemma-4-31b-it-assistant-bf16 | ⚪ | - | - | MTP/DFlash drafter (needs a target; not standalone) |
| gemma-4-31b-it-nvfp4 | ✅ | 76.38 | 4.88 | 26 tok; ModelOpt NVFP4 direct transcode (#692, #693, #697); image input works (see VLM note; fixed by #749) |
| gemma-4-31b-it-qat-4bit | ✅ | 59.15 | 5.39 | 26 tok |
| gemma-4-e2b-it-4bit | ✅ | 650.91 | 100.61 | 72 tok |
| gemma-4-e2b-it-8bit | ✅ | 654.70 | 58.55 | 70 tok |
| gemma-4-e2b-it-qat-4bit | ✅ | 530.34 | 68.31 | 39 tok |
| gemma-4-e4b-it-4bit | ✅ | 353.41 | 52.74 |  |
| gemma-4-e4b-it-8bit | ✅ | 298.76 | 29.88 | 76 tok |
| gemma-4-e4b-it-qat-4bit | ✅ | 192.20 | 30.41 | 33 tok |

## EXAONE

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| exaone-3.5-2.4b-4bit | ✅ | 1441.47 | 141.83 |  |
| exaone4-1.2b-4bit | ✅ | 1662.84 | 212.48 | 18 tok |

## Cohere / Command R

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| aya-expanse-8b-4bit | ✅ | 163.19 | 53.47 |  |
| command-r7b-4bit | ✅ | 117.25 | 52.22 |  |

## Granite (IBM)

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| granite-3.3-2b-instruct-4bit | ✅ | 1779.32 | 123.98 | 31 tok |
| granite-4.0-h-350m-4bit | ✅ | 1857.41 | 171.22 | 19 tok; post-reboot single (#755); the overnight-sweep run read 259.69 (tiny launch-bound models swing widely run-to-run on this host) |
| granite-4.0-h-tiny-4bit | ✅ | 238.51 | 101.37 | 42 tok; post-reboot single (#755) |
| granite-4.1-3b-4bit | ✅ | 291.06 | 78.85 | 7 tok |
| granite-4.1-8b-4bit | ✅ | 177.90 | 18.43 | 1 tok |
| granite-speech-4.1-2b-nar-mlx | ⚪ | - | - | not a standalone text-gen model |

## MoE (Mixture of Experts)

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| deepseek-v2-lite-4bit | ✅ | 154.68 | 94.78 | 18 tok |
| dots.llm1.inst-mixed-4-6bit | ✅ | 25.42 | 22.04 | memory-gate skip on 2026-07-12 (see version note); figures from 2026-06-17 / 0.3.1 |
| gpt-oss-120b-4bit | ✅ | 57.75 | 50.48 | memory-gate skip on 2026-07-12 (see version note); figures from 2026-06-17 / 0.3.1 |
| gpt-oss-20b-mxfp4 | ✅ | 104.58 | 79.36 |  |
| lfm2-8b-a1b-4bit | ✅ | 148.84 | 165.53 | 37 tok; post-reboot single (#755 control) |
| llama-4-scout-17b-4bit | ✅ | 27.72 | 21.46 | memory-gate skip on 2026-07-12 (see version note); figures from 2026-07-09 / 0.4.0-rc.1 |
| minimax-m2-3bit | ✅ | 26.95 | 22.03 | memory-gate skip on 2026-07-12 (see version note); figures from 2026-06-17 / 0.3.1 |
| mixtral-8x7b-4bit | ✅ | 12.63 | 28.42 | 73 tok |
| phi-3.5-moe-4bit | ✅ | 30.18 | 53.32 |  |
| qwen1.5-moe-a2.7b-4bit | ✅ | 260.27 | 122.13 |  |
| qwen3-30b-a3b-4bit | ✅ | 134.37 | 94.39 | 34 tok; post-reboot single (#755 control) |
| qwen3-moe-4bit | ✅ | 129.28 | 89.06 | 34 tok |

## MLA (Multi-head Latent Attention)

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| minicpm3-4b-4bit | ✅ | 291.27 | 59.14 |  |

## DeepSeek Family

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| deepseek-coder-1.3b-4bit | ✅ | 4728.54 | 92.96 |  |
| deepseek-r1-distill-7b-4bit | ✅ | 206.62 | 58.93 |  |

## Nemotron Family

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| nemotron-3-nano-omni-30b-a3b-reasoning-4bit | ✅ | 120.97 | 82.88 | 20 tok; post-reboot single (#755) |
| nemotron-h-30b-4bit | ✅ | 113.24 | 87.41 | 46 tok; post-reboot single (#755) |
| nemotron-nas-30b-4bit | ✅ | 117.99 | 85.91 | 46 tok; post-reboot single (#755) |

## SSM / Mamba / Hybrid

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| falcon-h1-tiny-90m-instruct-4bit | ✅ | 1193.95 | 354.43 | 30 tok; post-reboot single (#755); the overnight-sweep run read 413.00 |
| falcon-mamba-7b-4bit | ✅ | 92.54 | 22.06 | 2 tok |
| jamba-v0.1-4bit | ✅ | 523.67 | 89.63 |  |
| lfm2-350m-8bit | ✅ | 3270.66 | 393.84 | 13 tok; decode regression fixed (#748): was ~40 tok/s, restored to the 0.3.1 envelope by computing the single-step depthwise short conv as a broadcast multiply-sum instead of a tiny bf16 `conv1d` (MLX 0.32.1 routed that to cuDNN's per-channel grouped-conv engine on CUDA) |
| mamba2-130m | ✅ | 892.83 | 202.98 | median of 3 post-reboot runs (#755): decode read 147.98 / 202.98 / 204.77 across identical same-boot runs, so the earlier -10.4% sweep delta was run-to-run variance, not a regression; decode conv audited (#752): runs on the fast `conv1d_c1_k1_nhwc` engine (576 launches = 24 layers x 24 tokens, one per op), not the `convolve_common_engine` per-channel path |
| mamba2-1.3b-4bit | ✅ | 329.13 | 83.40 |  |
| plamo-2-1b | ✅ | 198.43 | 46.84 | post-reboot single (#755) |

## Chinese / Asian Language Models

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| baichuan-m1-14b-4bit | ✅ | 73.62 | 23.18 | 7 tok |
| ernie-4.5-0.3b-4bit | ✅ | 5857.00 | 625.30 |  |
| glm4-flash-4bit | ✅ | 105.46 | 47.42 | 18 tok; median of 3 post-reboot runs (#755); identical 100-token runs span 39.2-54.8 tok/s on this host, and the 0.3.1 record (53.33) sits inside that envelope |
| glm-5.1-4bit | ⚪ | - | - | not tested (weights not downloaded; empty directory) |
| glm-5-4bit | ⚪ | - | - | incomplete checkpoint: weights present but no tokenizer |
| hunyuan-13b | ✅ | 19.05 | 14.79 |  |
| hunyuan-1.8b-4bit | ✅ | 758.23 | 164.55 | 41 tok |
| hunyuan-a13b-instruct-4bit | ✅ | 19.82 | 14.69 |  |
| internlm2-7b-4bit | ✅ | 386.15 | 52.82 |  |
| internlm3-8b-4bit | ✅ | 502.05 | 46.08 |  |
| seed-oss-36b-instruct-4bit | ✅ | 77.70 | 10.46 |  |

## Mistral Family

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| ministral-3b-4bit | ✅ | 4970.50 | 104.26 | 37 tok |
| mistral-small-3.1-24b-4bit | ✅ | 891.89 | 15.31 | 20 tok |
| mistral-small-4-119b-2603-4bit | ⚪ | - | - | not measured: memory-gate skip (63 GiB weights) |
| pixtral-12b | ✅ | 120.39 | 33.57 |  |
| pixtral-12b-4bit | ✅ | 120.78 | 33.37 |  |

## BitNet

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| bitnet-b1.58-2b-4t | ✅ | 527.32 | 135.91 | 33 tok; CUDA ternary kernel (#322) |
| bitnet-b1.58-2b-4t-4bit | ✅ | 564.89 | 188.70 | 33 tok; CUDA ternary kernel (#322) |

## VLM-capable Models (text-only pass)

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| aya-vision-8b | ✅ | 1354.53 | 50.17 | 33 tok |
| bunny-llama3-8b-4bit | ✅ | 380.97 | 49.69 | 40 tok |
| docling-layout-heron-mlx-bf16 | ⚪ | - | - | not a standalone text-gen model |
| glm-4.1v-9b-thinking-4bit | ✅ | 151.01 | 36.49 | 81 tok |
| glm-4.5v-4bit | ⚪ | - | - | not measured: memory-gate skip (57 GiB weights) |
| internvl3-1b | ✅ | 3366.24 | 482.36 | 31 tok |
| kimi-vl-a3b-thinking-4bit | ✅ | 163.81 | 83.31 |  |
| llama-3.2-11b-vision-instruct-4bit | ✅ | 1729.65 | 49.47 | 23 tok |
| llava-1.5-7b-4bit | ✅ | 218.79 | 58.14 |  |
| llava-interleave-qwen-0.5b-bf16 | ✅ | 2413.68 | 217.49 | 49 tok |
| llava-next-mistral-7b-4bit | ✅ | 331.20 | 53.64 |  |
| minicpm-v-4.6-bf16 | ✅ | 168.34 | 111.35 |  |
| minicpm-v-4.6-mxfp4 | ✅ | 158.11 | 145.53 |  |
| molmo-7b | ✅ | 211.44 | 34.31 | 24 tok |
| molmo2-4b | ✅ | 239.02 | 27.07 | 33 tok |
| moondream2 | ✅ | 345.57 | 82.66 |  |
| paddleocr-vl-bfloat16 | ✅ | 729.57 | 210.76 |  |
| paligemma2-3b-6bit | ⚪ | 355.87 | 0.00 | image-only (PaliGemma): no text-gen without an image; works in the VLM table |
| phi-3.5-vision-4bit | ✅ | 420.78 | 92.80 | 43 tok |
| qwen2-vl-2b | ✅ | 843.20 | 107.81 | 35 tok |
| qwen2-vl-2b-4bit | ✅ | 838.58 | 104.68 | 35 tok |
| qwen2.5-vl-3b-4bit | ✅ | 511.79 | 60.03 | 39 tok |
| qwen3-vl-2b | ✅ | 780.04 | 168.80 | 59 tok |
| qwen3-vl-2b-4bit | ✅ | 771.29 | 170.31 | 61 tok |
| qwen3-vl-30b-a3b-4bit | ✅ | 127.31 | 86.36 | 34 tok |
| qwen3-vl-32b-4bit | ✅ | 84.07 | 11.08 | 37 tok |
| qwen3-vl-4b-4bit | ✅ | 339.73 | 77.58 | 49 tok |
| qwen3-vl-4b-instruct-4bit | ✅ | 375.29 | 78.05 | 49 tok |
| qwen3-vl-8b-4bit | ✅ | 220.94 | 48.33 | 57 tok |
| qwen3-vl-8b-instruct-4bit | ✅ | 218.10 | 48.35 | 57 tok |
| smolvlm-instruct-bf16 | ✅ | 2738.51 | 67.21 |  |
| youtu-vl-4b-instruct | ✅ | 451.42 | 22.23 | 93 tok |

## Qwen3.5 / Qwen3.6 / Qwen3-next

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| qwen3.5-0.8b-4bit | ✅ | 518.54 | 178.96 | 18 tok |
| qwen3.5-0.8b-optiq-4bit | ✅ | 493.74 | 156.08 | 19 tok |
| qwen3.5-27b-4bit | ✅ | 60.75 | 12.47 | 30 tok |
| qwen3.5-27b-dflash | ⚪ | - | - | MTP/DFlash drafter (needs a target; not standalone) |
| qwen3.5-2b-4bit | ✅ | 438.79 | 126.71 | 31 tok |
| qwen3.5-35b-a3b-4bit | ✅ | 142.32 | 63.34 | 31 tok |
| qwen3.5-4b-4bit | ✅ | 269.59 | 63.08 | 31 tok |
| qwen3.5-4b-dflash | ⚪ | - | - | MTP/DFlash drafter (needs a target; not standalone) |
| qwen3.5-9b-4bit | ✅ | 187.88 | 39.51 | 31 tok |
| qwen3.5-9b-bf16 | ✅ | 167.21 | 13.17 | 31 tok |
| qwen3.6-35b-a3b-4bit | ✅ | 149.70 | 63.20 | 27 tok |

## Solar

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| solar-open-100b-4bit | ✅ | 46.76 | 18.37 | memory-gate skip on 2026-07-12 (see version note); figures from 2026-06-17 / 0.3.1 |

## Speech / Audio (not applicable)

| Model | Status | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------|-----------------|----------------|-------|
| kokoro-82m | ⚪ | - | - | TTS checkpoint; not a text-gen model |
| whisper-base | ⚪ | - | - | speech-recognition checkpoint; not a text-gen model |

## VLM Benchmark (image input)

Models that accept image input and generated tokens under the `"What is in this image?"` prompt with `tests/fixtures/test_image.png`. All rows are from 2026-07-12 except `llama-4-scout-17b-4bit` (memory-gate skip; row carried from 2026-06-17).

| Model | Status | Generated Tokens | Prefill (tok/s) | Decode (tok/s) |
|-------|--------|------------------|-----------------|----------------|
| aya-vision-8b | ✅ | 60 | 1753.50 | 40.81 |
| bunny-llama3-8b-4bit | ✅ | 37 | 1799.17 | 45.84 |
| gemma-3-4b-it-4bit | ✅ | 16 | 1159.91 | 83.24 |
| gemma-4-12b-it-4bit | ✅ | 28 | 1201.87 | 14.12 |
| gemma-4-12b-it-4bit-down8 | ✅ | 8 | 1424.46 | 17.37 |
| gemma-4-12b-it-4bit-gs32 | ✅ | 34 | 904.71 | 21.37 |
| gemma-4-12b-it-4bit-uniform | ✅ | 14 | 1211.69 | 22.01 |
| gemma-4-26b-a4b-it-4bit | ✅ | 29 | 301.30 | 54.60 |
| gemma-4-26b-a4b-it-qat-4bit | ✅ | 31 | 330.14 | 53.01 |
| gemma-4-31b-4bit | ✅ | 5 | 395.86 | 7.56 |
| gemma-4-31b-it-4bit | ✅ | 24 | 422.29 | 8.79 |
| gemma-4-31b-it-qat-4bit | ✅ | 28 | 329.19 | 5.61 |
| gemma-4-e2b-it-4bit | ✅ | 100 | 2642.58 | 102.88 |
| gemma-4-e2b-it-8bit | ✅ | 100 | 2240.83 | 59.13 |
| gemma-4-e2b-it-qat-4bit | ✅ | 57 | 2275.22 | 67.07 |
| gemma-4-e4b-it-4bit | ✅ | 92 | 1649.55 | 50.03 |
| gemma-4-e4b-it-8bit | ✅ | 73 | 1163.17 | 29.15 |
| gemma-4-e4b-it-qat-4bit | ✅ | 48 | 1269.17 | 33.20 |
| gemma3-4b-4bit | ✅ | 16 | 1141.33 | 83.17 |
| gemma3n-e2b-4bit | ✅ | 41 | 2712.51 | 82.54 |
| gemma3n-e4b-4bit | ✅ | 33 | 1879.58 | 53.75 |
| gemma3n-e4b-bf16 | ✅ | 24 | 1887.22 | 21.11 |
| glm-4.1v-9b-thinking-4bit | ✅ | 100 | 827.97 | 36.23 |
| internvl3-1b | ✅ | 8 | 1836.68 | 392.59 |
| kimi-vl-a3b-thinking-4bit | ✅ | 100 | 161.55 | 86.14 |
| llama-3.2-11b-vision-instruct-4bit | ✅ | 37 | 12.49 | 24.38 |
| llava-1.5-7b-4bit | ✅ | 100 | 2348.32 | 54.03 |
| llava-interleave-qwen-0.5b-bf16 | ✅ | 32 | 19962.62 | 194.91 |
| llava-next-mistral-7b-4bit | ✅ | 100 | 2215.06 | 53.30 |
| minicpm-v-4.6-bf16 | ✅ | 23 | 479.11 | 102.31 |
| minicpm-v-4.6-mxfp4 | ✅ | 81 | 655.69 | 145.52 |
| ministral-3b-4bit | ✅ | 100 | 3075.47 | 87.75 |
| mistral-small-3.1-24b-4bit | ✅ | 29 | 635.16 | 15.12 |
| molmo-7b | ✅ | 100 | 1053.86 | 35.89 |
| molmo2-4b | ✅ | 46 | 803.55 | 27.35 |
| moondream2 | ✅ | 4 | 79.82 | 63.00 |
| nemotron-3-nano-omni-30b-a3b-reasoning-4bit | ✅ | 6 | 141.06 | 70.09 |
| paddleocr-vl-bfloat16 | ✅ | 12 | 4057.01 | 174.72 |
| paligemma2-3b-6bit | ✅ | 2 | 1904.98 | 47.27 |
| phi-3.5-vision-4bit | ✅ | 19 | 1245.08 | 82.87 |
| pixtral-12b | ✅ | 100 | 890.73 | 30.87 |
| pixtral-12b-4bit | ✅ | 100 | 894.23 | 30.79 |
| qwen2-vl-2b | ✅ | 12 | 606.44 | 92.14 |
| qwen2-vl-2b-4bit | ✅ | 12 | 595.12 | 96.05 |
| qwen2.5-vl-3b-4bit | ✅ | 64 | 527.31 | 60.29 |
| qwen3-vl-2b | ✅ | 58 | 2499.68 | 136.54 |
| qwen3-vl-2b-4bit | ✅ | 58 | 2402.22 | 135.58 |
| qwen3-vl-30b-a3b-4bit | ✅ | 59 | 201.17 | 65.94 |
| qwen3-vl-32b-4bit | ✅ | 49 | 245.12 | 10.52 |
| qwen3-vl-4b-4bit | ✅ | 41 | 1299.68 | 67.45 |
| qwen3-vl-4b-instruct-4bit | ✅ | 43 | 1321.10 | 67.25 |
| qwen3-vl-8b-4bit | ✅ | 39 | 963.42 | 42.08 |
| qwen3-vl-8b-instruct-4bit | ✅ | 35 | 952.98 | 41.71 |
| qwen3.5-0.8b-4bit | ✅ | 100 | 1440.60 | 207.76 |
| qwen3.5-27b-4bit | ✅ | 100 | 192.24 | 12.79 |
| qwen3.5-2b-4bit | ✅ | 47 | 1009.31 | 124.06 |
| qwen3.5-35b-a3b-4bit | ✅ | 100 | 205.84 | 63.01 |
| qwen3.5-4b-4bit | ✅ | 49 | 664.46 | 64.00 |
| qwen3.5-9b-4bit | ✅ | 100 | 515.19 | 41.34 |
| qwen3.5-9b-bf16 | ✅ | 100 | 419.14 | 13.51 |
| qwen3.6-35b-a3b-4bit | ✅ | 37 | 206.34 | 61.08 |
| smolvlm-instruct-bf16 | ✅ | 100 | 2815.60 | 67.24 |
| youtu-vl-4b-instruct | ✅ | 30 | 337.10 | 20.89 |
| llama-4-scout-17b-4bit | ✅ | 100 | 29.15 | 20.54 |
| gemma-4-31b-it-nvfp4 | ✅ | 27 | 353.20 | 4.87 |

`llama-4-scout-17b-4bit` figures are from 2026-06-17 (memory-gate skip on 2026-07-12). `gemma-4-31b-it-nvfp4` now serves image input (fixed by #749): the NVFP4 key remapper strips the `model.` prefix from all 356 `model.vision_tower.*` / `model.embed_vision.*` keys, not just the 1372 text-decoder keys, so the checkpoint routes to Gemma4VLM and the vision tower binds. The vision front-end stays dense (ModelOpt keeps `model.vision_tower*` / `model.embed_vision*` in `quantization_config.ignore`), so it needs key renaming only, no NVFP4 transcode.

---

## Summary

**Test date**: 2026-07-12 | **Hardware**: NVIDIA GB10 (DGX Spark) | **mlxcel**: 0.4.0-rc.1 (CUDA 13.0, SM 12.1) | **MLX pin**: `57c66cac` (0.32.1)

| Metric | Count |
|--------|-------|
| **Total model directories** | 159 |
| **Pass (✅, measured 2026-07-12)** | 141 |
| **Pass (✅, carried from an earlier sweep; memory-gate skip)** | 5 |
| **Fail (❌, code failure)** | 0 |
| **Not tested / N.A. (⚪)** | 13 |
| **VLM models measured (image input)** | 63 (+1 carried) |

### Notable changes vs the 2026-06-17 (0.3.1) and 2026-07-09 (rc.1 subset) records

- **SSM / hybrid / NAS decode reads 1.4-3.4x higher than the 0.3.1 record, attributed to #727 and confirmed post-reboot (#755)**: granite-4.0-h-350m 86.60 → 171.22, granite-4.0-h-tiny 33.84 → 101.37, falcon-h1-tiny 110.42 → 354.43, nemotron-h-30b 40.32 → 87.41, nemotron-nas-30b 37.33 → 85.91, nemotron-omni-30b 38.45 → 82.88, plamo-2-1b 35.14 → 46.84 (post-reboot singles; the table rows above carry these values). An earlier revision of this note called the delta environmental because "no SSM-related code has landed since" the low 2026-07-09 readings. That was wrong: the fused single-token SSM decode kernel was ported to CUDA on 2026-07-10 (#727, one launch replacing the ~55-op SSD scan graph per SSM layer for granite-4.0-h, falcon-h1, plamo-2, nemotron-h), the day after those singles ran, and its own PR measured granite-4.0-h-350m at 4.5x. The post-reboot re-measurement (#755) confirms the gains persist on a fresh host, so these are real release numbers, not an artifact. The tiny launch-bound checkpoints still swing widely between individual runs (granite-350m read 259.69 in the overnight sweep vs 171.22 post-reboot; falcon-h1-tiny 413.00 vs 354.43); see the run-to-run variance note below.
- **lfm2-350m-8bit decode regressed ~10x, now fixed (#748)**: 409.01 (0.3.1) → 39.84 at rc.1, restored to 393.84 by the fix. Root cause: MLX 0.32.1 dispatches the single-step (L=1) bf16 depthwise short conv on CUDA to cuDNN's generic `convolve_common_engine`, which launches one kernel per channel (~1024 for a 350M LFM2) and consumed 88.6% of decode GPU time; computing that decode step as a broadcast multiply-and-sum over the `L_cache` taps removes the grouped-conv dispatch. The sibling `lfm2-8b-a1b-4bit` was never affected (its conv runs on the fast `conv1d_c1_k1_nhwc` kernel; 157.73 → 161.87 → 161.39). Prefill still uses `conv1d` and was always healthy.
- **Moderate decode drops, resolved as no code regression (#755)**: the overnight sweep read gemma-4-26b-a4b-it-4bit 58.59 → 50.19 (-14%), gemma-4-26b-a4b-it-qat-4bit 50.33 → 45.29 (-10%), glm4-flash-4bit 53.33 → 45.72 (-14%), and these were never the #748 conv-dispatch pattern (neither `gemma4` MoE nor `glm4_moe_lite` calls `conv1d`, per #752). The post-reboot re-measurement settles them in two different ways. The gemma pair recovered decisively and repeatably: 59.57-60.07 (n=3) and 53.48-53.71 (n=3), at or above the 0.3.1 records with ±0.5% spread. The 26B MoE is bandwidth-bound and does not flap between runs, so its sweep-day depression was the stale ~5.5-day host/driver state and a fresh boot removes it. glm4-flash did not "recover" because there was nothing to recover from: twelve identical 100-token greedy runs on the freshly booted host span 39.2-54.8 tok/s in a bimodal pattern (a ~41 tok/s mode and a ~52.5 tok/s mode; the generated tokens are identical, so the work is constant), and the 0.3.1 record (53.33) sits inside the fast mode, meaning a single sweep run simply draws from this distribution. mamba2-130m behaves the same (147.98 then 202.98-208.25 across identical same-boot runs, vs the 181.05 record). Kill-switch A/B rules out the code suspects: `MLXCEL_QMV_MULTIROW=0` reads 49.55 (inside the envelope; the #740 multirow path keeps `M*B == 1` classic decode on the stock kernel by design), `MLXCEL_SSM_CUDA_KERNEL=0` is unchanged on mamba2 (208.25; pure mamba2 never uses the fused kernel), and #732 shipped only trace tooling plus a default-off normalization. The slow mode is not SM clock capping either (slow runs were sampled at a pinned 2411-2424 MHz SM clock), so its host-level mechanism remains unidentified, but the fast mode reproducing the 0.3.1 number rules out a code-level regression. Practical consequence: single-run decode deltas within roughly ±25% on launch-bound models (small dense/SSM checkpoints and small MoEs like glm4-flash) are below this host's run-to-run noise floor and should not be read as regressions without repeats. Separately, glm4-flash's templated greedy output shortened from 100 tokens (0.3.1) to 18 (rc.1), a different greedy continuation rather than a failure, so its sweep-to-sweep decode averages additionally stopped being length-comparable.
- **SSM / hybrid L=1 conv dispatch audited across the family (#752), no further regression found**: after #748/#751 fixed LFM2, every other decode-path depthwise-conv family was measured under `nsys` on a warm GB10 CUDA decode (`MLX_USE_CUDA_GRAPHS=0` for a complete kernel histogram). All run on the fast `conv1d_c1_k1_nhwc` cuDNN engine, not the per-channel `convolve_common_engine` that regressed LFM2: mamba2-130m (bf16), mamba2-1.3b-4bit, falcon-mamba-7b-4bit, falcon-h1-tiny-90m-4bit, granite-4.0-h-350m-4bit, jamba-v0.1-4bit, plamo-2-1b (f32 activations), qwen3.5-0.8b-4bit (covers the gated-delta conv), and nemotron-h-30b-4bit. The proof is the launch count: each shows conv instances equal to (conv layers x decode tokens), one launch per op, whereas the slow engine launches one kernel per channel (thousands per op). LFM2's slow-engine dispatch is specific to its `conv_L_cache = 3` / hidden-1024 shape and does not reproduce at the SSM `conv_kernel = 4` widths, so no family is adapted. All of this was measured on MLX pin `57c66cac` (0.32.1), the same pin whose CUDA conv dispatch heuristic produced the lfm2 slow path, so these REFUTED verdicts are point-in-time and should be re-checked against newer MLX pins. The #751 short-conv decode helper was still lifted into a shared `models::conv_decode` module so any future family that does regress can adopt it in one line.
- **Text-only prefill for several VLM-capable models is an order of magnitude higher than 0.3.1** (aya-vision-8b 124.91 → 1354.53, pixtral-12b 35.97 → 120.39, youtu-vl 134.27 → 451.42, mistral-small-3.1-24b 63.68 → 891.89), consistent with the text-prompt path no longer paying vision-tower costs rather than with a kernel speedup.
- **Decode improvements >10% on dense/MoE models**: apertus-8b +21%, gemma3-4b / gemma-3-4b-it +19%, gemma3-1b +15%, gemma3n-e2b +10%, qwen3-0.6b-4bit +13%, aya-expanse-8b +12%, plus the VLM-side gains below.
- **VLM (image-input) gains**: molmo-7b 23.81 → 35.89 (+51%), nemotron-omni-30b 32.86 → 70.09 (+113%), gemma3-4b family +13-18%.
- **gemma-4-31b-it-nvfp4 now serves image input** (fixed by #749: 27 tok, 353.20 prefill / 4.87 decode) with its text pass stable (76.38 prefill / 4.88 decode); the NVFP4 key remapper now strips the `model.` prefix from the 356 vision-tower keys, so the checkpoint routes to Gemma4VLM and the vision tower binds.

### New model directories since 2026-06-17 (14)

`gemma-2-9b-8bit` (119.74 / 22.54), the three local `gemma-4-12b-it-4bit-{uniform,gs32,down8}` requant variants from the #685 tooling, and six new VLM-capable checkpoints that all pass both passes: `glm-4.1v-9b-thinking-4bit`, `kimi-vl-a3b-thinking-4bit`, `llama-3.2-11b-vision-instruct-4bit`, `moondream2`, `paddleocr-vl-bfloat16`, `smolvlm-instruct-bf16`. `youtu-vl-4b-instruct` also gains an image-input row (it was text-only in the 0.3.1 table). The remaining four are non-runnable: `kokoro-82m` (TTS) and `whisper-base` (speech) are not text-gen models, `glm-4.5v-4bit` and `mistral-small-4-119b-2603-4bit` sit above the memory gate.

### Memory-gate exclusions (2026-07-12)

Seven checkpoints were excluded up front because their weight size exceeds ~51 GiB (`BENCH_MEM_OVERHEAD_FACTOR=2.0` against the 85% system-memory budget); runs at that scale have repeatedly destabilized the GB10 driver (`NV_ERR_NO_MEMORY` accumulation, two hard freezes on 2026-07-06): `minimax-m2-3bit` (93 GiB), `dots.llm1.inst-mixed-4-6bit` (80 GiB), `mistral-small-4-119b-2603-4bit` (63 GiB), `gpt-oss-120b-4bit` (61 GiB), `glm-4.5v-4bit` (57 GiB), `llama-4-scout-17b-4bit` (56 GiB), `solar-open-100b-4bit` (53 GiB). Their CSV rows carry `SKIP:oom_estimate`. Re-measure them individually after a fresh boot if release numbers are needed.

### Not-applicable / skipped models (by cause)

Every model that does not pass falls into one of these buckets; there are no outstanding code-level failures on the 0.4.0-rc.1 line.

- **Weights not downloaded or incomplete (⚪):** `glm-5.1-4bit` (empty directory), `glm-5-4bit` (weights present, tokenizer missing)
- **Not a standalone text generator (⚪):** `paligemma2-3b-6bit` (image-only PaliGemma; captions correctly in the VLM image table), `docling-layout-heron-mlx-bf16` (document layout), `granite-speech-4.1-2b-nar-mlx` (speech), `whisper-base` (speech recognition), `kokoro-82m` (TTS)
- **MTP/DFlash drafter checkpoints (need a target; not standalone) (⚪):** `gemma-4-12b-it-assistant-4bit`, `gemma-4-31b-it-assistant-bf16`, `qwen3.5-27b-dflash`, `qwen3.5-4b-dflash`
- **Memory-gate exclusions:** see the list above (5 keep earlier figures, 2 unmeasured)

### CUDA fused decode-MoE kernel (#319) — historical, 0.3.1

The 0.3.1 line ported the fused decode-MoE kernel to CUDA (#319). Nine MoE models that aborted at 0.3.0 on the Metal-only kernel (`[metal_kernel] No Metal back-end`) run on the CUDA fused path, faster than the `gather_qmm` fallback: qwen3-moe-4bit, qwen3-30b-a3b-4bit, qwen3.5-35b-a3b-4bit, qwen3.6-35b-a3b-4bit, qwen1.5-moe-a2.7b-4bit, gemma-4-26b-a4b-it-4bit, gemma-4-26b-a4b-it-qat-4bit, dots.llm1.inst-mixed-4-6bit, diffusiongemma-26b-a4b-it-4bit. Greedy output is byte-identical to `gather_qmm`. The CUDA fused crossover is much higher than Metal's (~13-14k vs ~4096), but the cap stays 4096 (see `fused-moe-decode-kernel-design.md`).
