# Model Compatibility & Performance Tests (M5 Max)

Compatibility and performance testing for mlxcel models on **MacBook Pro M5 Max 128GB**, with same-host mlx-lm / mlx-vlm reference measurements and M1 Ultra ratios where available.

## Test Environment

| Item | Value |
|------|-------|
| **Hardware** | MacBook Pro M5 Max, 128GB RAM |
| **OS** | macOS 26.6.2 (build 25G83) |
| **mlxcel version** | 0.7.0-beta.1 (`mlxcel_version`) |
| **Source revision** | `b2ff1eee` (`mlxcel_commit`); MLX pin `9a795735` (`mlx_commit`) |
| **MLX version** | upstream main (via mlxcel-core; pinned commit `9a795735`) |
| **mlx-lm baseline** | 0.31.3 (dev checkout https://github.com/ml-explore/mlx-lm, commit `ed1fca4`); not re-run for the 0.6.0 sweep, see note below |
| **mlx-vlm baseline** | 0.4.4; not re-run for the 0.6.0 sweep |
| **Test Prompt** | "Hello, how are you today?" (text) / "What is in this image?" (VLM) |
| **Max Tokens** | 100 |
| **Test Date** | 2026-09-03/04 full text + VLM re-benchmark (0.6.0); prior: 2026-07-11/12 full sweep (0.4.0-rc.1), 2026-06-15 full sweep (0.2.1), 2026-05-27 full sweep (0.1.0) |
| **Benchmark Status** | Full text + VLM sweep on mlxcel 0.6.0 using `mlxcel-bench-decode`: 175 text model dirs via `bench_decode.sh all` (161 with decode numbers) plus the VLM-mode pass `all --vlm` (77 with decode numbers). Both runs used `--cooldown 30 --big-cooldown 30`, which remain required on this host: without them the sweep accumulates enough heat to thermally throttle the mid-sweep Qwen cluster. Time Machine was stopped and its automatic backups disabled for the whole campaign, since a running backup starves both disk and unified-memory bandwidth; a 3.9 TB copy was in fact active when the campaign started. This round also raised the up-front memory guard to a 90 GB weight budget (`BENCH_MEM_OVERHEAD_FACTOR=1.209` against the 108.8 GB default limit), so `deepseek-v3-4bit` (99.96 GiB) now records `SKIP:oom_estimate` instead of consuming a slot and failing; `qwen3-coder-480b-a35b-instruct-4bit` (251.54 GiB) was already skipped. **Decode is comparable to the 0.4.0-rc.1 sweep; prefill is not comparable for every model.** Two correctness fixes landed after 2026-07-12 that change the measured *condition* rather than speed: #792 (aspect-ratio image processing for Pixtral/Mistral3, merged 2026-07-13) stopped upscaling every image to a fixed square, and the chat-template path now renders Llama's official template exactly. Affected rows therefore run at a much shorter prompt, and `prefill_tok_s` at the shorter length is arithmetically lower because the fixed per-call overhead is amortized over fewer tokens. See "Condition changes since 0.4.0-rc.1" below before reading any prefill delta. The `vs M1 Ultra` column is a **cross-version** ratio this round: 0.6.0 M5 Max decode over the 2026-07-12 **0.4.0-rc.1** M1 Ultra sweep (`benchmarks/metal_m1ultra_2026-07-12.csv`), because that host had not been re-measured at 0.6.0 when this campaign ran. It will be refreshed by the pending 0.6.0 M1 Ultra sweep. The `mlxcel vs mlx-lm` / `vs mlx-vlm` percentages further down still carry the 2026-05-18 Python baselines; those sweeps were not re-run this round. |

### Which version each CSV column records

One column used to carry three different meanings across the corpus, which is
why it has been split. Every CSV from this campaign records all of:

| Column | Value here | Meaning |
|--------|-----------|---------|
| `mlxcel_version` | `0.7.0-beta.1` | this repository's crate version |
| `mlxcel_commit` | `b2ff1eee` | the source revision measured, 8 characters, `-dirty` when tracked files were modified |
| `mlx_commit` | `9a795735` | the pinned MLX C++ revision the binary links |

The sweep ran at `b2ff1eee`, when `Cargo.toml` still read `0.6.0`. The rows are
labelled `0.7.0-beta.1` anyway because the tag (`64f5d9b4`) is exactly one commit
later and that commit changes no compiled source: the diff is manifests,
`Cargo.lock`, `CHANGELOG.md`, `CITATION.cff`, `README.md`,
`docs/environment-variables.md`, the recipes registry and an issue template.
Verify with:

```bash
git diff --name-only b2ff1eee v0.7.0-beta.1 | grep -E '\.(rs|cpp|metal|cu|h|hpp)$'
```

which returns nothing. `mlxcel_commit` pins the exact revision either way, so
nothing is lost by the label.

`mlx_commit` exists because an MLX pin bump changes kernels without moving
either mlxcel field, so a sweep taken across one would otherwise look identical
to a sweep taken before it.

## Legend

- ✅ Pass: Model works correctly
- ⚠️ Partial: Loads but output quality problems or low token count
- ❌ Fail: Does not work

## Basic Transformers

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| llama3 | Llama-3.2-1B-Instruct-4bit | ✅ | 5337.36 | 556.36 | **1.32x** | 48 tokens |
| llama3.1 | Llama-3.1-8B-Instruct-4bit | ✅ | 986.34 | 114.92 | 1.08x | 54 tokens |
| llama3 (8B bf16) | Llama-3.1-8B-Instruct (bf16) | ⚠️ | 800.51 | 32.81 | 0.96x | 44 tokens; bf16; slow decode |
| llama4 | Llama-4-Scout-17B-16E-4bit | ✅ | 95.48 | 48.12 | **1.35x** | 59 tokens |
| command-r7b | c4ai-command-r7b-4bit | ✅ | 240.44 | 113.27 | 1.04x | 100 tokens |
| aya-expanse-8b | aya-expanse-8b-4bit | ✅ | 228.05 | 111.57 | 1.06x | 100 tokens |
| aya-vision-8b | aya-vision-8b (text-only) | ✅ | 3245.72 | 108.66 | 1.07x | 45 tokens; text-only |
| deepseek-r1 | DeepSeek-R1-Distill-Qwen-7B-4bit | ✅ | 436.26 | 125.70 | 1.15x | 100 tokens |
| internlm2 | InternLM2-7B-4bit | ✅ | 528.62 | 118.34 | 1.13x | 100 tokens |
| internlm3 | internlm3-8b-instruct-4bit | ✅ | 733.86 | 101.66 | 1.21x | 100 tokens |
| mimo | MiMo-7B-RL-4bit | ✅ | 792.79 | 119.63 | **1.41x** | 100 tokens |
| minicpm | MiniCPM-2B-sft-bf16-4bit | ✅ | 804.30 | 229.72 | **1.54x** | 31 tokens |
| bunny-llama3-8b | Bunny-Llama-3-8B-V-4bit (text) | ✅ | 551.49 | 114.40 | 1.14x | 40 tokens; text-only |
| llava-1.5-7b | llava-1.5-7b-4bit (text) | ✅ | 332.78 | 123.31 | 1.09x | 60 tokens; text-only |
| llava-next | llava-v1.6-mistral-7b-4bit (text) | ✅ | 473.14 | 120.60 | 1.07x | 49 tokens; text-only |
| llava-interleave | llava-interleave-qwen-0.5b-bf16 (text) | ✅ | 1605.28 | 392.35 | 1.23x | 23 tokens |

## Gemma Family

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| gemma | gemma-2b-it-4bit | ✅ | 1280.60 | 213.76 | 1.12x | 49 tokens |
| gemma2 | gemma-2-2b-it-4bit | ✅ | 1234.14 | 242.32 | **1.41x** | 18 tokens; full-budget raw prompt 245.83 tok/s |
| gemma3 | gemma-3-1b-it-4bit | ✅ | 1938.99 | 391.42 | **1.72x** | 30 tokens |
| gemma3 (4B) | gemma-3-4b-it-4bit | ✅ | 850.64 | 183.69 | **1.56x** | 84 tokens; full-budget raw prompt 183.77 tok/s |
| gemma3n (E2B) | gemma-3n-E2B-it-4bit | ✅ | 782.40 | 157.92 | **1.79x** | 72 tokens |
| gemma3n (E4B) | gemma-3n-E4B-it-4bit | ✅ | 583.32 | 110.40 | **1.65x** | 74 tokens |
| gemma3n (E4B bf16) | gemma-3n-E4B-it (bf16) | ✅ | 417.46 | 39.54 | 1.14x | 69 tokens; Gemma3n language MLP bf16 preserved, other bf16 materialized as f16; M5 (Neural Accelerator) uses the split decode path while other Apple Silicon uses the fused path; ~80% of mlx-lm decode |
| gemma4 (26B MoE) | gemma-4-26b-a4b-it-4bit | ✅ | 561.93 | 149.41 | **1.88x** | 26 tokens |
| gemma4 (31B) | gemma-4-31b-4bit | ✅ | 106.44 | 28.52 | **1.43x** | 100 tokens |
| gemma4 (31B IT) | gemma-4-31b-it-4bit | ✅ | 146.30 | 27.34 | **1.43x** | 26 tokens |
| gemma4 (31B nvfp4) | Gemma-4-31b-it-nvfp4 | ⚠️ | 130.16 | 15.65 | 1.22x | 26 tokens; nvfp4 has no fast Metal kernel |
| gemma4 (E2B 4bit) | gemma-4-e2b-it-4bit | ✅ | 1466.17 | 210.27 | **1.66x** | 72 tokens |
| gemma4 (E2B 8bit) | gemma-4-e2b-it-8bit | ✅ | 1264.72 | 149.05 | **1.44x** | 79 tokens |
| gemma4 (E4B 4bit) | gemma-4-e4b-it-4bit | ✅ | 826.35 | 141.78 | **1.66x** | 100 tokens |
| gemma4 (E4B 8bit) | gemma-4-e4b-it-8bit | ✅ | 716.11 | 85.78 | 1.30x | 76 tokens |
| gemma4 (12B) | gemma-4-12b-it-4bit | ✅ | 320.22 | 43.96 | 1.19x | 27 tokens; NEW (6-13) |
| gemma4 (26B QAT) | gemma-4-26b-a4b-it-qat-4bit | ✅ | 547.50 | 139.39 | **1.80x** | 26 tokens; QAT; NEW (6-13) |
| gemma4 (31B IT QAT) | gemma-4-31b-it-qat-4bit | ✅ | 130.79 | 17.24 | 1.10x | 26 tokens; QAT; NEW (6-13) |
| gemma4 (E2B QAT) | gemma-4-e2b-it-qat-4bit | ✅ | 1109.34 | 173.58 | **1.53x** | 39 tokens; QAT; NEW (6-13) |
| gemma4 (E4B QAT) | gemma-4-e4b-it-qat-4bit | ✅ | 617.02 | 96.11 | **1.37x** | 33 tokens; QAT; NEW (6-13) |

## EXAONE

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| exaone | EXAONE-3.5-2.4B-Instruct-4bit | ✅ | 2300.77 | 284.14 | **1.43x** | 43 tokens |
| exaone4 | exaone-4.0-1.2b-4bit | ✅ | 1977.37 | 422.49 | **1.85x** | 10 tokens |

## Qwen Family

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| qwen2.5 (0.5B) | Qwen2.5-0.5B-Instruct-4bit | ✅ | 8467.27 | 660.77 | **1.73x** | 39 tokens |
| qwen2.5 (0.5B bf16) | Qwen2.5-0.5B-Instruct (bf16) | ✅ | 5814.81 | 400.41 | **1.35x** | 37 tokens |
| qwen2.5 (7B) | Qwen2.5-7B-Instruct-4bit | ✅ | 880.79 | 124.11 | 1.14x | 41 tokens |
| qwen2.5 (7B 8bit) | Qwen2.5-7B-Instruct-8bit | ✅ | 813.55 | 67.69 | 1.00x | 44 tokens |
| qwen2.5-vl (3B) | qwen2.5-vl-3b-4bit | ✅ | 1409.70 | 162.04 | **1.47x** | 39 tokens; re-downloaded (prior FAIL was a corrupt checkpoint, not a code bug) |
| qwen2-vl (2B) | Qwen2-VL-2B-Instruct-4bit | ✅ | 2506.64 | 268.31 | **1.58x** | 35 tokens |
| qwen1.5-moe | Qwen1.5-MoE-A2.7B-Chat-4bit | ✅ | 873.72 | 255.17 | **1.71x** | 36 tokens |
| qwen3 (0.6B) | Qwen3-0.6B-4bit | ✅ | 3696.62 | 601.12 | **2.63x** | 9 tokens |
| qwen3 (1.7B) | Qwen3-1.7B-4bit | ✅ | 1747.04 | 380.00 | **1.80x** | 38 tokens |
| qwen3 (4B) | Qwen3-4B-4bit | ✅ | 1004.44 | 191.19 | **1.57x** | 36 tokens |
| qwen3 (8B) | Qwen3-8B-4bit | ✅ | 571.59 | 112.24 | **1.41x** | 33 tokens |
| qwen3-30b-a3b | Qwen3-30B-A3B-4bit | ✅ | 415.14 | 174.24 | **2.06x** | 34 tokens |
| qwen3-moe | Qwen3-MoE-30B-4bit | ✅ | 415.67 | 173.61 | **2.08x** | 34 tokens |
| qwen3-vl (2B) | Qwen3-VL-2B-Instruct-4bit | ✅ | 1353.08 | 370.75 | **1.80x** | 59 tokens; text-only |
| qwen3-vl (4B) | qwen3-vl-4b-4bit | ✅ | 781.44 | 183.89 | **1.56x** | 49 tokens; text-only; NEW (6-13) |
| qwen3-vl (8B) | qwen3-vl-8b-4bit | ✅ | 444.34 | 110.70 | **1.38x** | 57 tokens; text-only; NEW (6-13) |
| qwen3-vl (30B MoE) | Qwen3-VL-30B-A3B-Instruct-4bit | ✅ | 324.18 | 167.90 | **2.04x** | 34 tokens; text-only |
| qwen3-vl (32B) | Qwen3-VL-32B-Instruct-4bit | ✅ | 122.64 | 27.46 | **1.33x** | 30 tokens; text-only |
| qwen3-next (80B MoE) | Qwen3-Next-80B-A3B-Instruct-4bit | ✅ | 328.29 | 119.37 | **2.01x** | 54 tokens; NEW (0.4.0-rc.1) |
| qwen3-omni (30B MoE) | Qwen3-Omni-30B-A3B-Instruct-4bit | ✅ | 408.75 | 167.38 | **2.02x** | 42 tokens; text path; NEW (0.4.0-rc.1) |
| qwen3-coder (480B) | Qwen3-Coder-480B-A35B-Instruct-4bit | ❌ | - | FAIL | - | SKIP:oom_estimate |
| qwen3.5 (0.8B) | Qwen3.5-0.8B-4bit | ✅ | 3275.49 | 500.09 | **2.00x** | 18 tokens |
| qwen3.5 (2B) | Qwen3.5-2B-4bit | ✅ | 1671.10 | 327.38 | **1.71x** | 31 tokens |
| qwen3.5 (4B) | Qwen3.5-4B-4bit | ✅ | 876.75 | 168.71 | **1.56x** | 31 tokens |
| qwen3.5 (9B) | Qwen3.5-9B-4bit | ✅ | 512.04 | 101.74 | **1.44x** | 31 tokens |
| qwen3.5 (9B bf16) | Qwen3.5-9B (bf16) | ✅ | 337.09 | 30.12 | 0.98x | 31 tokens |
| qwen3.5 (27B) | Qwen3.5-27B-4bit | ✅ | 168.69 | 32.75 | **1.37x** | 30 tokens |
| qwen3.5-35b-a3b | Qwen3.5-35B-A3B-4bit | ✅ | 694.44 | 159.97 | **1.96x** | 31 tokens |
| qwen3.6-35b-a3b | Qwen3.6-35B-A3B-4bit | ✅ | 696.81 | 152.72 | **1.91x** | 27 tokens; NEW (5-18) |

## Phi Family

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| phi-2 | phi-2-hf-4bit-mlx | ⚠️ | 313.07 | 106.08 | **1.83x** | 1 token; (likely EOS) |
| phi-3-mini | Phi-3-mini-4k-instruct-4bit | ✅ | 591.88 | 210.39 | 1.28x | 25 tokens |
| phi-3.5-mini | Phi-3.5-mini-instruct-4bit | ✅ | 582.33 | 203.68 | 1.25x | 40 tokens |
| phi-3.5-moe | Phi-3.5-MoE-instruct-4bit | ✅ | 126.64 | 110.02 | **1.46x** | 35 tokens |
| phi-3.5-vision | Phi-3.5-vision-instruct-4bit | ✅ | 872.57 | 203.24 | 1.25x | 36 tokens; text-only |
| phi-4 | Phi-4-4bit | ✅ | 240.72 | 59.45 | 1.03x | 16 tokens |

## OLMo Family

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| olmo-1b | OLMo-1B-hf-4bit | ✅ | 780.33 | 242.64 | 1.19x | 100 tokens |
| olmo2-7b | OLMo2-7B-4bit | ✅ | 621.99 | 117.83 | 1.18x | 27 tokens |
| olmo3-32b | OLMo3.1-32B-4bit | ✅ | 470.91 | 29.29 | **1.37x** | 100 tokens |

## MoE (Mixture of Experts)

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| minimax | MiniMax-M2-3bit | ✅ | 185.81 | 73.79 | - | 100 tokens |
| mixtral | Mixtral-8x7B-Instruct-v0.1-4bit | ✅ | 106.22 | 65.56 | 1.27x | 73 tokens |
| gpt_oss (20B) | gpt-oss-20b-MXFP4-Q4 | ✅ | 1231.56 | 173.19 | **1.94x** | 100 tokens |
| gpt_oss (120B) | gpt-oss-120b-4bit | ✅ | 332.76 | 112.83 | **1.90x** | 58 tokens |
| solar-open-100b | Solar-Open-100B-4bit | ✅ | 287.52 | 65.51 | **1.87x** | 100 tokens |
| dots.llm1 | dots.llm1.inst-mixed-4-6bit | ✅ | 102.76 | 50.88 | - | 39 tokens; mixed 4/6-bit; NEW (6-13) |
| lfm2-moe | lfm2-8b-a1b-4bit | ✅ | 1088.26 | 338.21 | **1.83x** | 37 tokens; NEW (6-13) |

## DeepSeek Family

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| deepseek | deepseek-coder-1.3b-instruct-4bit | ✅ | 6278.89 | 176.76 | 1.11x | 37 tokens |
| deepseek_v2 | DeepSeek-V2-Lite-Chat-4bit | ✅ | 389.18 | 210.84 | **2.04x** | 44 tokens |
| deepseek_v3 | - | ❌ | - | FAIL | - | SKIP:oom_estimate |

## MLA (Multi-head Latent Attention)

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| minicpm3 | MiniCPM3-4B-4bit | ✅ | 568.98 | 131.78 | **1.51x** | 39 tokens |

## Nemotron Family

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| nemotron_h | Nemotron-H-30B-4bit | ✅ | 406.57 | 176.01 | **1.92x** | 46 tokens |
| nemotron_nas | Nemotron-NAS-30B-A3B-4bit | ✅ | 406.48 | 176.61 | **1.95x** | 46 tokens |
| nemotron-omni | Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | ✅ | 404.72 | 171.42 | **2.01x** | 19 tokens; text path; NEW (6-14) |

## SSM / Mamba Models

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| mamba | Falcon-Mamba-7B-4bit | ⚠️ | 274.98 | 66.60 | **1.53x** | 2 tokens; chat template EOS |
| mamba2 | mamba2-1.3b-4bit | ✅ | 959.29 | 163.21 | **1.42x** | 100 tokens |
| mamba2 (130M) | mamba2-130m | ✅ | 2260.89 | 348.79 | **1.34x** | 100 tokens; NEW (6-14) |
| jamba | Jamba-v0.1-4bit | ✅ | 1086.94 | 216.87 | **1.68x** | 100 tokens; raw prompt 215.74 tok/s |
| falcon-h1 | falcon-h1-tiny-90m-instruct-4bit | ✅ | 1016.95 | 154.82 | 0.48x | 30 tokens; Mamba2 + attention hybrid; NEW (6-13) |
| plamo2 | plamo-2-1b | ✅ | 381.17 | 81.49 | 0.77x | 100 tokens; Mamba + attention hybrid; NEW (6-13) |

## Chinese / Asian Language Models

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| baichuan | Baichuan-M1-14B-Instruct-4bit | ✅ | 158.15 | 57.66 | **1.46x** | 7 tokens |
| glm4_moe_lite | GLM-4.7-Flash-4bit | ✅ | 242.30 | 106.11 | **2.17x** | 18 tokens |
| ernie4_5 | ERNIE-4.5-0.3B-Instruct-4bit | ✅ | 7753.12 | 1068.37 | **2.05x** | 100 tokens |
| hunyuan_moe | hunyuan-a13b-instruct-4bit | ✅ | 153.14 | 64.61 | **1.46x** | 36 tokens; A13B MoE (4-bit), canonical after checkpoint dedup |
| hunyuan_v1_dense | Hunyuan-1.8B-Instruct-4bit | ✅ | 1098.09 | 330.11 | **1.82x** | 42 tokens |

## Other Models

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| ministral3 | Ministral-3B-Instruct-4bit | ✅ | 6729.43 | 225.24 | **1.57x** | 34 tokens; VLM wrapper |
| mistral-small | mistral-small-3.1-24b-4bit | ✅ | 1000.14 | 39.65 | **1.38x** | 20 tokens |
| molmo2 | molmo2-4b | ✅ | 680.14 | 103.47 | **1.76x** | 33 tokens |
| molmo-7b | molmo-7b | ✅ | 372.30 | 123.87 | **1.84x** | 59 tokens; text spot-check |
| internvl3 | internvl3-1b | ✅ | 8271.03 | 664.13 | **1.91x** | 37 tokens |
| smollm-135m | SmolLM-135M-Instruct-4bit | ✅ | 5105.84 | 926.25 | **2.21x** | 100 tokens |
| smollm3-3b | SmolLM3-3B-4bit | ✅ | 2315.35 | 231.61 | **1.76x** | 19 tokens |
| stablelm-1.6b | stablelm-2-1_6b-chat-4bit | ✅ | 2487.59 | 428.36 | **1.62x** | 59 tokens |
| starcoder2-3b | starcoder2-3b-4bit | ✅ | 437.34 | 217.03 | **1.33x** | 100 tokens |
| pixtral-12b | pixtral-12b-4bit | ✅ | 205.42 | 72.60 | 1.07x | 28 tokens; text-only |
| paligemma2-3b | paligemma2-3b (6-bit) | ✅ | 482.68 | 195.86 | - | 100 tokens; text-only |

## Granite Family

Ported 2026-06-13 (dense + Mamba2/attention hybrid + hybrid-MoE). vs M1 Ultra
ratios are from the 2026-07-12 0.4.0-rc.1 M1 Ultra sweep.

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| granite | granite-3.3-2b-instruct-4bit | ✅ | 3384.19 | 258.60 | **1.39x** | 31 tokens; dense |
| granite4_h (350M) | granite-4.0-h-350m-4bit | ✅ | 1682.36 | 130.12 | 0.55x | 19 tokens; Mamba2 + attention hybrid |
| granite4_h (tiny) | granite-4.0-h-tiny-4bit | ✅ | 428.69 | 75.05 | 0.70x | 44 tokens; hybrid MoE |
| granite4.1 (3B) | granite-4.1-3b-4bit | ✅ | 688.29 | 177.49 | **1.37x** | 7 tokens |
| granite4.1 (8B) | granite-4.1-8b-4bit | ⚠️ | 383.80 | 53.99 | **1.90x** | 1 token; likely early EOS, re-check with a code prompt |

## Recently Ported Families (2026-06-13/14)

New architectures landed in the 06-13/06-14 wave. vs M1 Ultra ratios are from
the 2026-07-12 0.4.0-rc.1 M1 Ultra sweep.

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| apertus | apertus-8b-instruct-2509-4bit | ✅ | 1297.71 | 112.88 | **1.39x** | 40 tokens; xIELU, QK-norm, llama3 RoPE |
| bitnet (4bit pack) | bitnet-b1.58-2b-4t-4bit | ✅ | 393.03 | 327.82 | **2.21x** | 33 tokens; 1.58-bit ternary |
| bitnet | bitnet-b1.58-2b-4t | ✅ | 384.45 | 260.61 | **1.91x** | 33 tokens; 1.58-bit ternary |
| lfm2 | lfm2-350m-8bit | ✅ | 5516.27 | 858.03 | **1.53x** | 13 tokens; 8-bit |
| seed-oss | seed-oss-36b-instruct-4bit | ✅ | 92.27 | 26.57 | **1.35x** | 100 tokens |
| minicpm-v (4.6) | minicpm-v-4.6-bf16 | ✅ | 1465.65 | 276.17 | **1.31x** | 100 tokens; text path |
| youtu-vl | youtu-vl-4b-instruct | ✅ | 770.85 | 47.37 | 1.07x | 93 tokens; text path |

The following newly-added checkpoints are present in `models/` but are not
measurable by the text decode harness this round:

- `glm-5-4bit`, `glm-5.1-4bit` — `FAIL:bench`, resolved at the 0.6.0 sweep as a **local checkpoint problem, not a load-path defect**: `glm-5-4bit` still holds 21 GB of `.incomplete` download blobs with no materialized `*.safetensors` or tokenizer, and `glm-5.1-4bit` holds only `.gitattributes`. Both need a re-download before GLM-5 support can be judged.
- `minicpm-v-4.6-mxfp4` — was `FAIL:bench` at 0.4.0-rc.1; **passes at 0.6.0** (1568.60 prefill / 346.72 decode, 100 tokens). See the 0.6.0 section below.
- `diffusiongemma-26b-a4b-it-4bit` — block-diffusion generation; decode tok/s is not a meaningful metric for this harness.
- `docling-layout-heron-mlx-bf16` — layout/vision model; no text decode path.
- `granite-speech-4.1-2b-nar-mlx` — non-autoregressive speech model; no text decode path.
- `gemma-4-12b-it-assistant-4bit`, `gemma-4-31b-it-assistant-bf16` — MTP drafter checkpoints, not standalone generators.
- `qwen3.5-0.8b-optiq-4bit`, `qwen3.5-27b-dflash`, `qwen3.5-4b-dflash` — experimental quant/decode variants; `FAIL:bench` standalone.

## Recently Ported Families (0.4.0-rc.1 / 2026-07-11)

New architectures landed since the 0.2.1 sweep. Numbers are the text-decode
path (`bench_decode.sh all`); the vision-capable models also appear in the VLM
table below with their image-prompt numbers. vs M1 Ultra ratios are from the
2026-07-12 0.4.0-rc.1 M1 Ultra sweep (`-` where the M1 Ultra sweep lacks the model).

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| gemma2 (9B 8bit) | gemma-2-9b-8bit | ✅ | 172.29 | 49.84 | - | 100 tokens |
| mistral-small-4 (119B) | mistral-small-4-119b-2603-4bit | ✅ | 164.76 | 19.05 | 1.02x | 60 tokens; dense 119B |
| phi-3-small | phi-3-small-8k-instruct-aq4_64 | ✅ | 662.83 | 111.71 | 1.12x | 100 tokens; aq4_64 |
| llada2.0-mini | llada2.0-mini-preview-4bit | ✅ | 1826.58 | 336.84 | **2.15x** | 100 tokens; diffusion LM |
| deepseek-ocr | deepseek-ocr-4bit | ✅ | 979.99 | 699.31 | **2.26x** | 21 tokens; text path |
| deepseek-ocr-2 | deepseek-ocr-2-4bit | ✅ | 965.56 | 700.41 | **2.23x** | 20 tokens; text path |
| deepseek-vl2 | deepseek-vl2-small-4bit | ✅ | 749.35 | 207.18 | **1.85x** | 25 tokens; text path |
| fastvlm | fastvlm-0.5b-bf16 | ✅ | 4492.54 | 402.20 | **1.36x** | 46 tokens; text path |
| glm-4.1v | glm-4.1v-9b-thinking-4bit | ✅ | 298.77 | 66.38 | - | 81 tokens; text path |
| glm-4.5v | glm-4.5v-4bit | ✅ | 84.44 | 16.74 | - | 47 tokens; text path |
| glm-ocr | glm-ocr-4bit | ✅ | 2781.34 | 460.04 | **1.97x** | 7 tokens; text path |
| granite4-vision (3B) | granite-4.0-3b-vision-4bit | ✅ | 1504.10 | 201.95 | **1.57x** | 26 tokens; text path |
| granite-vision (2B) | granite-vision-3.2-2b-4bit | ✅ | 2473.11 | 257.63 | **1.60x** | 20 tokens; text path |
| idefics2 | idefics2-8b-4bit | ✅ | 317.03 | 121.95 | 1.11x | 100 tokens; text path |
| idefics3 | idefics3-8b-llama3-4bit | ✅ | 277.43 | 118.32 | 1.13x | 100 tokens; text path |
| kimi-vl | kimi-vl-a3b-thinking-4bit | ✅ | 596.60 | 177.21 | - | 100 tokens; A3B MoE; text path |
| lfm2-vl | lfm2-vl-450m-4bit | ✅ | 5423.66 | 927.56 | **2.02x** | 19 tokens; text path |
| llama-3.2-vision (11B) | llama-3.2-11b-vision-instruct-4bit | ✅ | 517.49 | 113.78 | - | 51 tokens; text path |
| moondream2 | moondream2 | ✅ | 183.69 | 43.10 | - | 100 tokens; text path |
| paddleocr-vl | paddleocr-vl-bfloat16 | ✅ | 931.09 | 157.76 | 1.20x | 100 tokens; text path |
| smolvlm | smolvlm-instruct-bf16 | ✅ | 716.61 | 131.02 | - | 100 tokens; text path |

Newly-added checkpoints present in `models/` but not measurable by the text
decode harness this round:

- `deepseek-v4-flash-4bit`, `qwen3-coder-480b-a35b-instruct-4bit`: `SKIP:oom_estimate` on the 128 GB budget.
- `kokoro-82m` (TTS), `whisper-base` (ASR): `FAIL:bench`; no autoregressive text-decode path.
- `dots.ocr-4bit`: loads and runs but emits 0 text tokens on the plain prompt; decode tok/s is not meaningful.

## Recently Ported Families (0.6.0 / 2026-09-03)

New checkpoints measured for the first time in the 0.6.0 sweep. Numbers are the
text-decode path (`bench_decode.sh all`). vs M1 Ultra ratios are cross-version
against the 2026-07-12 0.4.0-rc.1 M1 Ultra sweep (`-` where that sweep lacks the
model); they will be refreshed by the pending 0.6.0 M1 Ultra run.

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| qwen3.8-27b | qwen3.8-27b-4bit | ✅ | 168.08 | 32.88 | - | 37 tokens; NEW (0.6.0); qualified on the `qwen3_5` path (#1174) |
| hunyuan (13B) | hunyuan-13b | ✅ | 89.28 | 64.22 | **1.46x** | 31 tokens; NEW (0.6.0) |
| diffusiongemma (26B MoE) | diffusiongemma-26b-a4b-it-4bit | ✅ | 529.23 | 114.69 | **1.69x** | 27 tokens; NEW (0.6.0) |
| minicpm-v (4.6 mxfp4) | minicpm-v-4.6-mxfp4 | ✅ | 1568.60 | 346.72 | **1.56x** | 100 tokens; NEW (0.6.0) |
| qwen3.5 (0.8B optiq) | qwen3.5-0.8b-optiq-4bit | ✅ | 3209.41 | 429.25 | **1.84x** | 19 tokens; NEW (0.6.0) |
| qwen2.5 (1.5B) | qwen2.5-1.5b-instruct-4bit | ✅ | 3445.58 | 383.36 | **1.54x** | 31 tokens; NEW (0.6.0) |
| dots.ocr | dots.ocr-4bit | ⚠️ | 630.78 | 0.00 | - | 0 tokens; loads and prefills but emits no text on a text-only prompt |
| glm-5 | glm-5-4bit | ❌ | - | FAIL | - | FAIL:bench, but not a runtime defect: the local checkpoint is an interrupted download (21 GB still sitting as `.incomplete` blobs under `.cache/huggingface/download/`, no `*.safetensors` and no tokenizer materialized). Re-download before reading this as a GLM-5 support gap |
| glm-5.1 | glm-5.1-4bit | ❌ | - | FAIL | - | FAIL:bench, but not a runtime defect: the local directory holds only `.gitattributes` (8 KB of cache metadata, no `config.json`), i.e. the download never started. Re-download before reading this as a GLM-5.1 support gap |

### Duplicate checkpoint directories (not listed separately)

The local `models/` store keeps several checkpoints under two names. Each pair
below has byte-identical safetensors sizes **and** an identical tensor layout
(same names, shapes, dtypes and offsets, verified by hashing the safetensors
headers), so they are the same weights and are benchmarked twice per sweep.
Only the name already carried in the tables above gets a row; adding the alias
would double-count the model in the Summary Statistics. Teaching the harness to skip them is tracked as issue #1615.

| Listed as | Also on disk as |
|-----------|-----------------|
| `qwen2.5-7b-instruct-4bit` | `qwen2.5-7b`, `qwen2.5-7b-4bit` |
| `qwen2.5-1.5b-instruct-4bit` | `qwen2.5-1.5b-4bit` |
| `qwen2-vl-2b-4bit` | `qwen2-vl-2b` |
| `qwen3-vl-2b-4bit` | `qwen3-vl-2b` |
| `qwen3-vl-4b-4bit` | `qwen3-vl-4b-instruct-4bit` |
| `qwen3-vl-8b-4bit` | `qwen3-vl-8b-instruct-4bit` |
| `qwen3-0.6b-4bit` | `qwen3-0.6b` |
| `qwen2.5-0.5b-4bit` | `qwen2-0.5b` |
| `gemma-3-4b-it-4bit` | `gemma3-4b-4bit` |
| `llama-3.1-8b-4bit` | `meta-llama-3.1-8b-instruct-4bit` |
| `pixtral-12b-4bit` | `pixtral-12b` |


## VLM (image input) — full sweep

Below table reports the per-VLM-prompt run from `bench_decode.sh all --vlm`.
All entries use the VLM prompt 'What is in this image?' with
`tests/fixtures/test_image.png`.

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| aya-vision-8b | aya-vision-8b | ✅ | 2660.08 | 109.15 | 1.12x | 29 tokens |
| bunny-llama3-8b | bunny-llama3-8b-4bit | ✅ | 2844.29 | 109.72 | 1.19x | 37 tokens |
| gemma3 (4B) | gemma3-4b-4bit | ✅ | 549.56 | 134.26 | **1.51x** | 9 tokens |
| gemma3n (E2B 4bit) | gemma3n-e2b-4bit | ✅ | 2968.80 | 152.58 | **1.90x** | 28 tokens |
| gemma3n (E4B 4bit) | gemma3n-e4b-4bit | ✅ | 2240.39 | 105.43 | **1.73x** | 29 tokens |
| gemma3n (E4B bf16) | gemma3n-e4b-bf16 | ✅ | 2171.43 | 38.08 | 1.19x | 20 tokens; bf16→f16 conversion path |
| gemma4 (26B MoE) | gemma-4-26b-a4b-it-4bit | ✅ | 885.66 | 140.39 | **2.01x** | 27 tokens |
| gemma4 (31B) | gemma-4-31b-4bit | ✅ | 428.47 | 22.04 | **1.50x** | 5 tokens |
| gemma4 (31B IT) | gemma-4-31b-it-4bit | ✅ | 439.72 | 25.54 | **1.40x** | 24 tokens |
| gemma4 (E2B 4bit) | gemma-4-e2b-it-4bit | ✅ | 2833.78 | 220.59 | **2.04x** | 100 tokens |
| gemma4 (E2B 8bit) | gemma-4-e2b-it-8bit | ✅ | 2644.83 | 146.41 | **1.59x** | 100 tokens |
| gemma4 (E4B 4bit) | gemma-4-e4b-it-4bit | ✅ | 2086.99 | 131.67 | **1.76x** | 90 tokens |
| gemma4 (E4B 8bit) | gemma-4-e4b-it-8bit | ✅ | 1922.20 | 78.15 | **1.32x** | 64 tokens |
| internvl3 (1B) | internvl3-1b | ✅ | 6553.50 | 601.00 | **2.77x** | 8 tokens |
| llama4 (Scout) | llama-4-scout-17b-4bit | ✅ | 397.72 | 48.01 | **1.45x** | 100 tokens |
| llava-1.5-7b | llava-1.5-7b-4bit | ✅ | 3216.08 | 112.16 | 1.14x | 21 tokens |
| llava-interleave | llava-interleave-qwen-0.5b-bf16 | ✅ | 15979.88 | 355.35 | **1.40x** | 32 tokens |
| llava-next | llava-next-mistral-7b-4bit | ✅ | 2977.26 | 114.61 | 1.13x | 23 tokens |
| ministral3 | ministral-3b-4bit | ✅ | 5527.54 | 223.31 | **1.85x** | 56 tokens |
| mistral-small (3.1 24B) | mistral-small-3.1-24b-4bit | ✅ | 1048.45 | 40.02 | **1.44x** | 31 tokens |
| molmo-7b | molmo-7b | ✅ | 2348.36 | 122.79 | **1.56x** | 65 tokens; mlx-vlm baseline is a 1-token anomaly |
| molmo2 (4B) | molmo2-4b | ✅ | 2471.17 | 100.97 | **1.72x** | 46 tokens |
| paligemma2 (3B 6-bit) | paligemma2-3b-6bit | ✅ | 5246.06 | 96.78 | **1.61x** | 2 tokens |
| phi-3.5-vision | phi-3.5-vision-4bit | ✅ | 3754.55 | 175.75 | **1.49x** | 14 tokens |
| pixtral (12B) | pixtral-12b-4bit | ✅ | 1588.48 | 73.58 | 1.26x | 30 tokens; intermittent slow VLM decode reads (~20 tok/s) seen on M5, not consistently reproducible (see Known Issues) |
| qwen2-vl (2B) | qwen2-vl-2b-4bit | ✅ | 2452.49 | 248.76 | **1.74x** | 12 tokens; EOS-terminate |
| qwen2.5-vl (3B) | qwen2.5-vl-3b-4bit | ✅ | 1629.01 | 143.87 | **1.35x** | 8 tokens; re-downloaded (prior FAIL was a corrupt checkpoint) |
| qwen3-vl (2B) | qwen3-vl-2b-4bit | ✅ | 2159.95 | 273.88 | **1.55x** | 100 tokens |
| qwen3-vl (4B) | qwen3-vl-4b-4bit | ✅ | 1201.21 | 136.19 | **1.43x** | 41 tokens; NEW (6-13) |
| qwen3-vl (8B) | qwen3-vl-8b-4bit | ✅ | 991.47 | 81.68 | 1.30x | 38 tokens; NEW (6-13) |
| qwen3-vl (30B MoE) | qwen3-vl-30b-a3b-4bit | ✅ | 548.17 | 58.01 | **1.43x** | 63 tokens |
| qwen3-vl (32B) | qwen3-vl-32b-4bit | ✅ | 297.26 | 19.00 | 1.08x | 49 tokens |
| gemma4 (12B) | gemma-4-12b-it-4bit | ✅ | 1408.03 | 43.74 | 1.28x | 25 tokens; NEW (6-13) |
| minicpm-v (4.6) | minicpm-v-4.6-bf16 | ✅ | 967.29 | 263.50 | **1.49x** | 23 tokens; NEW (6-13) |
| nemotron-omni | nemotron-3-nano-omni-30b-a3b-reasoning-4bit | ✅ | 646.83 | 154.51 | **2.21x** | 6 tokens; NEW (6-14) |
| youtu-vl | youtu-vl-4b-instruct | ✅ | 521.88 | 45.52 | 1.06x | 30 tokens; NEW (6-13) |
| deepseek-ocr | deepseek-ocr-4bit | ✅ | 1536.22 | 614.38 | **2.45x** | 15 tokens; NEW (0.4.0-rc.1) |
| deepseek-ocr-2 | deepseek-ocr-2-4bit | ✅ | 1528.38 | 382.59 | **2.57x** | 5 tokens; NEW (0.4.0-rc.1) |
| deepseek-vl2 | deepseek-vl2-small-4bit | ✅ | 859.31 | 172.79 | **1.80x** | 8 tokens; NEW (0.4.0-rc.1) |
| fastvlm | fastvlm-0.5b-bf16 | ✅ | 2655.91 | 389.27 | **1.42x** | 100 tokens; NEW (0.4.0-rc.1) |
| glm-4.1v | glm-4.1v-9b-thinking-4bit | ✅ | 826.97 | 62.57 | - | 100 tokens; NEW (0.4.0-rc.1) |
| glm-4.5v | glm-4.5v-4bit | ✅ | 212.16 | 15.98 | - | 39 tokens; NEW (0.4.0-rc.1) |
| granite4-vision (3B) | granite-4.0-3b-vision-4bit | ✅ | 2687.46 | 201.95 | **1.64x** | 30 tokens; NEW (0.4.0-rc.1) |
| granite-vision (2B) | granite-vision-3.2-2b-4bit | ✅ | 6215.28 | 230.81 | **1.97x** | 47 tokens; NEW (0.4.0-rc.1) |
| idefics2 | idefics2-8b-4bit | ✅ | 929.92 | 114.51 | 1.11x | 12 tokens; NEW (0.4.0-rc.1) |
| idefics3 | idefics3-8b-llama3-4bit | ✅ | 2101.44 | 118.51 | 1.14x | 100 tokens; NEW (0.4.0-rc.1) |
| kimi-vl | kimi-vl-a3b-thinking-4bit | ✅ | 757.42 | 175.48 | - | 100 tokens; A3B MoE; NEW (0.4.0-rc.1) |
| lfm2-vl | lfm2-vl-450m-4bit | ✅ | 5713.97 | 1019.85 | **2.57x** | 47 tokens; NEW (0.4.0-rc.1) |
| llama-3.2-vision (11B) | llama-3.2-11b-vision-instruct-4bit | ✅ | 14.33 | 72.31 | - | 69 tokens; NEW (0.4.0-rc.1) |
| moondream2 | moondream2 | ✅ | 53.20 | 31.50 | - | 4 tokens; NEW (0.4.0-rc.1) |
| paddleocr-vl | paddleocr-vl-bfloat16 | ✅ | 3374.52 | 124.87 | 1.18x | 12 tokens; NEW (0.4.0-rc.1) |
| smolvlm | smolvlm-instruct-bf16 | ✅ | 2164.34 | 116.10 | - | 5 tokens; NEW (0.4.0-rc.1) |
| qwen3-omni (30B) | qwen3-omni-30b-a3b-instruct-4bit | ✅ | 583.49 | 37.54 | **1.66x** | 2 tokens; NEW (0.4.0-rc.1) |

## Condition changes since 0.4.0-rc.1

Two fixes landed after the 2026-07-12 sweep that change **what the harness
measures**, not how fast it runs. Both shorten the prompt, and `prefill_tok_s`
is prompt tokens over prefill milliseconds, so a shorter prompt amortizes the
fixed per-call overhead over fewer tokens and reads lower. Decode is unaffected.
Do not read either as a slowdown.

### 1. Image processing no longer upscales (#792, merged 2026-07-13)

`fix(vision): aspect-ratio image processing for Pixtral and Mistral3` replaced
the fixed-square SigLIP path, which force-resized every image to a square, with
a processor that downscales to fit `size.longest_edge` and **never upscales**.
The VLM fixture is 224x224 and Pixtral's `longest_edge` is 1024 with patch 16,
so the arithmetic is exact:

| | old (forced square) | new (aspect-preserving) |
|---|---|---|
| resized to | 1024 x 1024 | 224 x 224 (unchanged) |
| image tokens | (1024/16)^2 = 4096 | (224/16)^2 = 196 |
| measured `prompt_tokens` | 4099 | 213 |

VLM `prompt_tokens` for the affected families dropped accordingly:

| Model | 2026-07-12 | 2026-09-04 |
|-------|-----------|-----------|
| `pixtral-12b-4bit`, `pixtral-12b` | 4099 | 213 |
| `mistral-small-4-119b-2603-4bit` | 3046 | 93 |
| `mistral-small-3.1-24b-4bit` | 3206 | 253 |
| `ministral-3b-4bit` | 3566 | 613 |

Their VLM decode deltas (`mistral-small-4-119b` +20.4%, `ministral-3b` +13.1%,
pixtral +9.9%) are context-length effects, not speedups.

### 2. The chat template renders Llama's official prompt

The text harness applies the checkpoint's chat template. At 0.6.0 the Llama
family's rendering of the standard test prompt is 42 tokens; the 0.4.0-rc.1
sweep recorded 98. Tokenizing the canonical Llama 3.1 rendering of
`"Hello, how are you today?"` with the checkpoint's own `tokenizer.json` gives
**42** tokens (and 7 with `--no-chat-template`), which is exactly what the 0.6.0
binary reports, so the current value is the correct one and 98 carried roughly
56 spurious tokens.

Ten models therefore show a double-digit `prefill_tok_s` drop that is entirely a
prompt-length change, not a regression:

| Model | prefill 07-12 | prefill 09-03 | prompt tokens |
|-------|--------------|--------------|---------------|
| `smolvlm-instruct-bf16` | 4099.59 | 716.61 | 56 -> 9 |
| `llama-3.2-11b-vision-instruct-4bit` | 2132.46 | 517.49 | 98 -> 17 |
| `llava-interleave-qwen-0.5b-bf16` | 5050.14 | 1605.28 | 26 -> 8 |
| `llama-3.1-8b-4bit` | 2138.85 | 986.34 | 98 -> 42 |
| `idefics3-8b-llama3-4bit` | 580.55 | 277.43 | 18 -> 9 |
| `llama-4-scout-17b-4bit` | 195.19 | 95.48 | 69 -> 18 |
| `llama-3.1-8b-bf16` | 1611.89 | 800.51 | 99 -> 43 |
| `llama-3.2-1b-4bit` | 8068.68 | 5337.36 | 99 -> 43 |
| `granite-vision-3.2-2b-4bit` | 2827.36 | 2473.11 | 55 -> 47 |
| `meta-llama-3.1-8b-instruct-4bit` | 2142.46 | 987.11 | 98 -> 42 |

Only three checkpoints lost more than 10% of prefill at an **unchanged** prompt
length, and none is a clean signal on its own: `dots.ocr-4bit` (-24.4%, already
⚠️ because it emits no text), `smollm-135m-4bit` (-16.2%, a 135M model at 16
prompt tokens) and `stablelm-1.6b-4bit` (-13.7%, 26 prompt tokens). All three sit
at prompt lengths where per-call overhead dominates. They are the only prefill
candidates worth a targeted re-measurement, tracked as issue #1614.

## Summary Statistics

Counts reflect the 2026-09-03/04 `bench_decode.sh all --cooldown 30 --big-cooldown 30`
text sweep on mlxcel 0.6.0, with `BENCH_MEM_OVERHEAD_FACTOR=1.209` (a 90 GB weight budget).

| Status | Count |
|--------|-------|
| ✅ Pass (measured decode) | 144 |
| ⚠️ Partial (loads; early EOS, slow path, or no text output) | 6 |
| ❌ Fail / OOM-skip | 4 |

**How 154 table rows reconcile with 175 benchmarked checkpoints.** The sweep walks
every directory in `models/`, but the tables above deliberately do not carry a row
per directory:

| | Count |
|---|---|
| Text table rows | 154 |
| ...of which map to a row in this sweep's CSV | 153 |
| ...of which have no CSV row (`MiniMax-M2-3bit`, removed from disk since 0.4.0-rc.1) | 1 |
| Benchmarked checkpoints with no table row | 22 |
| ...duplicate directories of a listed checkpoint (see the alias table above) | 12 |
| ...non-text checkpoints the text harness cannot decode (`whisper-base`, `kokoro-82m`, `granite-speech-4.1-2b-nar-mlx`, `docling-layout-heron-mlx-bf16`) | 4 |
| ...speculative drafters and `dflash` variants, measured in the speculative table instead | 6 |
| **Total benchmarked checkpoints** | **175** |

161 of the 175 checkpoints produced decode numbers. The 14 non-runs are 2
`SKIP:oom_estimate` (`deepseek-v3-4bit` at 99.96 GiB and
`qwen3-coder-480b-a35b-instruct-4bit` at 251.54 GiB, both over the 90 GB weight
budget) and 12 `FAIL:bench`, none of which is a decode-path defect: the 4
non-text checkpoints above, the 6 drafter / `dflash` variants (not standalone
generative models), and the GLM-5 pair, which this round was traced to
interrupted local downloads rather than a load-path bug (see Known Issues).

The 6 ⚠️ partials are `phi-2-4bit`, `falcon-mamba-7b-4bit`,
`gemma-4-31b-it-nvfp4`, `llama-3.1-8b-bf16` and `granite-4.1-8b-4bit` (early EOS
or no fast kernel), plus `dots.ocr-4bit`, which loads and prefills but emits no
text on a text-only prompt.

**No decode regression survived verification.** 159 checkpoints are comparable
with the 0.4.0-rc.1 sweep (both runs produced decode numbers); 156 of them moved
less than 10% either way and none moved more than 10% slower.
Three moved more than 10% faster, and they are not equivalent: `molmo2-4b`
(+61.7%, 63.99 -> 103.47) generated the same 33 tokens with prefill also up 26%
and is a real gain, corroborated by the VLM pass (+56.9% at an unchanged 46
tokens); `molmo-7b` (+56.5%) changed run length from 24 to 59 tokens, so the two
numbers are not directly comparable; and `phi-2-4bit` (+37.6%) is a single-token
run, which is a latency sample rather than a throughput measurement.

## Batched serving (B = 1/2/4)

Source: `benchmarks/metal_m5max_batch_2026-09-04.csv`, produced by
`scripts/bench_serving_concurrency.py` against one fresh `mlxcel-server` per
model, pinned to `--parallel 4 --max-batch-prefill 4` at the standard condition
(`--prompt-tokens 512 --max-tokens 128`). Under continuous batching N concurrent
streaming clients occupy N decode slots, so the concurrency level is the
effective decode batch size.

**Reading the TTFT column.** Levels run in ascending order and share the
synthetic prompt, so B=2 and B=4 start with a warm prompt cache while B=1 pays
the cold prefill. That is the intended serving-side condition, and it is why the
two dense models show TTFT *falling* from B=1 to B=2. Compare TTFT across B
levels only with that in mind; the aggregate column is the headline number.

### qwen2.5-0.5b-bf16 (small dense bf16; isolates scheduler overhead)

| B | ok/fail | TTFT mean (ms) | TTFT p95 (ms) | decode tok/s per request | aggregate tok/s | scaling vs B=1 |
|---|---------|----------------|---------------|--------------------------|-----------------|----------------|
| 1 | 1 / 0 | 27.8 | 27.8 | 378.4 | 352.2 | 1.00x |
| 2 | 2 / 0 | 14.4 | 18.3 | 249.1 | 488.1 | 1.39x |
| 4 | 4 / 0 | 21.7 | 33.1 | 291.6 | 1118.1 | **3.17x** |

### llama-3.1-8b-4bit (canonical dense 4-bit)

| B | ok/fail | TTFT mean (ms) | TTFT p95 (ms) | decode tok/s per request | aggregate tok/s | scaling vs B=1 |
|---|---------|----------------|---------------|--------------------------|-----------------|----------------|
| 1 | 1 / 0 | 173.1 | 173.1 | 104.8 | 92.4 | 1.00x |
| 2 | 2 / 0 | 53.7 | 71.0 | 93.7 | 181.7 | 1.97x |
| 4 | 4 / 0 | 88.0 | 139.4 | 78.6 | 300.4 | **3.25x** |

### qwen3-30b-a3b-4bit (MoE; batched decode hits the fused-MoE path)

| B | ok/fail | TTFT mean (ms) | TTFT p95 (ms) | decode tok/s per request | aggregate tok/s | scaling vs B=1 |
|---|---------|----------------|---------------|--------------------------|-----------------|----------------|
| 1 | 1 / 0 | 149.1 | 149.1 | 169.6 | 142.5 | 1.00x |
| 2 | 2 / 0 | 192.0 | 192.1 | 118.3 | 202.3 | 1.42x |
| 4 | 4 / 0 | 379.2 | 379.5 | 65.5 | 220.9 | **1.55x** |

### Reading

The dense models scale close to linearly to B=4 (3.17x and 3.25x on aggregate
throughput) while giving up 23-25% of per-request decode. The MoE model does not:
it reaches only **1.55x** aggregate at B=4, per-request decode falls 61% (169.6
-> 65.5), and TTFT rises 2.5x (149 -> 379 ms) *despite* the warm prompt cache
that helps the dense rows.

The mechanism is in the source rather than in the measurement. The fused MoE
decode kernel is gated on a single token: `qwen3_moe.rs` takes it only when
`array_shape(&x_flat)[0] == 1`, where `x_flat` is `[batch * seq_len, hidden]`.
The scheduler switches away from the per-sequence path at exactly B=2
(`dispatch_sync_decode` uses it only when `seq_ids.len() <= 1`) and
`execute_batched_decode` builds its input as `[b, 1]`, so from B=2 the gate fails
on every layer of every tick and the model falls back to `gather_qmm`. That also
explains why the scheduler is not the suspect: `qwen2.5-0.5b-bf16` is in this
trio precisely to isolate scheduler overhead, and it scales fine. Tracked as
issue #1616.

No requests failed at any level for any model (0 fail across all 9 cells).

SSM / hybrid / mixed-cache families are intentionally absent: the server
serializes their slots, so a B-ladder over them measures nothing.

## Performance vs mlx-lm / mlx-vlm baseline (2026-05-19 benchmark campaign)

> **Stale baseline.** This section is the 2026-05-19 campaign (mlxcel 0.0.28 vs
> mlx-lm 0.31.3 / mlx-vlm 0.4.4, MLX 0.31.2). It was **not** re-run for the
> 2026-07-11/12 0.4.0-rc.1 sweep, so the mlxcel columns here predate several MLX
> bumps and kernel changes. The current 0.4.0-rc.1 mlxcel numbers are in the
> per-family tables above; treat the percentages below as the last measured parity
> snapshot, pending a fresh mlx-lm/mlx-vlm baseline run at the 0.4.0-rc.1 pin.

Source CSVs (same M5 Max host, mlxcel 0.0.28 with `--cooldown 15 --big-cooldown 15`):

- mlxcel: `benchmarks/metal_m5max_2026-05-19.csv`
- mlx-lm: `benchmarks/pylm_m5max_2026-05-18.csv` (mlx-lm 0.31.3 dev checkout in https://github.com/ml-explore/mlx-lm)
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
- **Average mlxcel/mlx-lm**: 99% (median 99%, range 72%-127%)

### Aggregate (VLM, models with >=5 generated tokens both sides)

- **Comparable VLM pairs**: 22
- **mlxcel >= mlx-vlm**: 11 / 22 (50%)
- **mlxcel >= 90% parity**: 18 / 22 (82%)
- **Average mlxcel/mlx-vlm**: 102% (median 101%, range 74%-123%)

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
| internvl3-1b | 661.48 | FAIL | - |
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
| paligemma2-3b-6bit | 168.83 | FAIL | - |
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
| internvl3-1b | 601.50 | 529.33 | **114%** |
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
| qwen3-vl-30b-a3b-4bit | 56.38 | FAIL | - |
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


## FP32 Promotion Audit

Short prompt A/B runs on 2026-05-18 used `origin/main` at `5ebc074` as the
baseline and the branch as the candidate. Each row used:

```text
mlxcel generate -m models/<model> -p "Hello, how are you today?" -n 20 --profile --no-chat-template
```

The intent is a hot-path regression/impact check, not a replacement for the
100-token full sweep above. The clearest gains are the MoE rows that still used
the `nkh,nk->nh` expert combine contraction.

| Model | main prefill tok/s | prefill tok/s | main decode tok/s | decode tok/s | Decode change |
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
  class as in remaining MoE expert-weight combines.
- Qwen3/Qwen3.5/Qwen3.6 A3B, Qwen3-VL A3B, Mixtral, and gpt-oss are effectively
  guardrail-neutral in this short run. Their MoE combines now share the same
  dtype-preserving helper, but the previous contraction was not the dominant
  measured bottleneck for these rows.
- Non-MoE guardrails (`phi-3.5-mini`, `gemma3n-e4b-bf16`,
  `qwen3.5-0.8b`, `jamba`, `stablelm`) did not show a decode regression from
  the compiled activation, softcap, scalar-helper, or intentional-FP32 comments
  and tests added for this audit.

## SolarOpen Decode Sync Audit

 removed the accidental FP32 expert-weight combine and raised
`solar-open-100b-4bit` decode into the low 40 tok/s range, but still missed
the >=85% mlx-lm decode gate. The remaining SolarOpen-specific difference was
the Rust implementation forcing `eval_all()` after every decoder layer. That is
useful for multi-token prefill graph size control, but in single-token decode it
adds 48 GPU synchronizations per generated token. mlx-lm does not synchronize at
each layer in the decode path.

The branch keeps per-layer eval for prefill and skips it only when the input
sequence length is one token. Validation used the same direct real-model command:

```text
mlxcel generate -m models/solar-open-100b-4bit -p "Hello, how are you today?" -n 100 --profile --no-chat-template
```

| Build | Prefill tok/s | Decode tok/s | vs mlx-lm 66.30 tok/s |
|---|---:|---:|---:|
| `origin/main` after (`616c470`) | 9.19 | 41.35 | 62% |
| branch | 34.02 | 65.66 | **99%** |

This is +58.8% decode over current main and +298.4% over the original 16.48
tok/s issue baseline. The issue acceptance gate (>=56 tok/s) is met.

## Moderate Gap Triage

The four rows were rechecked on the M5 Max after, and
 had landed on `main`. The original `falcon-mamba` row used a generic chat
prompt that exits after `<|im_end|>` in both mlxcel and mlx-lm, so the useful
comparison uses a raw code prompt that generates the full 100-token budget.

| Model | Triage | Refreshed mlxcel decode | mlx-lm decode | Result |
|---|---|---:|---:|---:|
| `glm4-flash-4bit` | Real regression fixed by the already-merged MoE combine change | 111.09 | 108.32 | **103%** |
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
| pixtral-12b (VLM) | Intermittent slow VLM decode on M5 Max: repeated runs read either ~68 tok/s or ~20 tok/s, roughly 50/50, while the llava VLM control stays steady. **The stated trigger no longer applies as written**: the issue was attributed to pixtral's large ~4100-token image context, but #792 (aspect-ratio image processing, merged 2026-07-13) stopped upscaling the 224x224 fixture to a 1024x1024 square, so the same run is now a 213-token context and read 73.68 tok/s at 0.6.0. Re-characterize against the new context length before keeping this open | Low |
| glm-5-4bit / glm-5.1-4bit | Not a runtime defect. Investigated at the 0.6.0 sweep: both local checkpoints are interrupted downloads (`glm-5-4bit` = 21 GB of `.incomplete` blobs, no safetensors and no tokenizer; `glm-5.1-4bit` = `.gitattributes` only). Re-download, then re-test before filing anything against the GLM-5 load path | Low (data, not code) |
| hunyuan-moe-a13b-bf16 (bf16 A13B) | Dropped: bf16 weights exceed the 128 GB budget; use `hunyuan-a13b-instruct-4bit` (4-bit, 64.92 tok/s). The size estimate passed it before it OOM'd at load, so the harness logged `FAIL:bench` instead of `SKIP:oom` | Low |
| deepseek-v3-4bit | 99.96 GiB of weights on a 128 GB host. From the 0.6.0 sweep it is classified `SKIP:oom_estimate` under the 90 GB weight budget instead of being launched and recorded as `FAIL:bench`, which is what earlier sweeps did. A capacity exclusion, not a MoE + MLA defect | Low (capacity) |
| qwen3-coder-480b-a35b-instruct-4bit | OOM-skip on 128 GB; weights exceed the memory budget (the 480B Qwen3-Next was retired) | Medium |
| qwen3-0.6b-4bit | Full-budget raw prompt stays at ~93% of mlx-lm; sub-95% decode gap | Medium |
| gemma-4-31b-it-nvfp4 | Now decodes at ~15.6 tok/s via the native NVFP4 Metal path (was ~7 tok/s at 0.2.1); still about half the 4-bit rate, so flagged ⚠️ | Low |
| falcon-mamba-7b-4bit | Generic chat prompt exits after `<\|im_end\|>`; use a non-chat code prompt for perf checks | Low |
| phi-2-4bit | Generates only 1 token — likely EOS handling | Low |
| llama-3.1-8b-bf16 | bf16 → f16 conversion path is functional but slow | Low |

## Notes

- All tests use 4-bit quantized models unless noted.
- Performance measured with `mlxcel-bench-decode` (model load, warmup, and
  measured pass in one process).
- vs M1 Ultra ratios are M5 Max decode divided by the 2026-07-12 `benchmarks/metal_m1ultra_2026-07-12.csv` decode (same mlxcel 0.4.0-rc.1 / MLX pin `57c66cac` / cooldown-30 conditions). Rows show `-` where the M1 Ultra sweep did not measure the model.
- The 2026-07-11/12 sweep used `--cooldown 30 --big-cooldown 30`. Without cooldowns, heat accumulated over the larger 0.4.0-rc.1 model set thermally throttles the mid-sweep Qwen block (see Test Environment). Re-run full sweeps on this host with cooldowns.
- Prefill and decode tok/s reported separately.
- Current per-model values are the 2026-06-15 full sweep on mlxcel 0.2.1 (MLX pin `a6ec7123`): 151 text models (`bench_decode.sh all`) + 150 VLM-mode (`all --vlm`), bare run (pre-warm on, no cooldown). Source CSVs: `benchmarks/metal_m5max_2026-06-15.csv` and `benchmarks/metal_m5max_vlm_2026-06-15.csv`.
- vs the 2026-05-27 sweep: 0 decode regressions among the 93 models measured in both; 11 improved >10% (MoE families from the fused-decode default plus broad MLX-bump gains, e.g. gemma2-2b +16.5%, gemma3-4b +12.5%, qwen3-30b-a3b +12.2%, qwen3-moe +12.1%). The two sweep FAILs first read as regressions both turned out to be environmental, not code (corrupt qwen2.5-vl checkpoint re-downloaded; oversized bf16 hunyuan dropped for the 4-bit); see Summary Statistics.
- Measurement noise on very fast small models remains high (qwen3.5-0.8b-4bit and
  similar can span ±15% across back-to-back runs because 100 tokens generate in
  under 300 ms).

## TurboQuant KV cache — M5 Max results

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
| Qwen2.5-1.5B-Instruct-4bit (superseded) | fp16 | 3205.54 | 25,550 | superseded — |
| Qwen2.5-1.5B-Instruct-4bit (superseded) | turbo4asym | 2227.09 | 36,775 | superseded — |

The Qwen2.5-1.5B-Instruct-4bit rows above are retained for historical reference. A later run found that the fixture collapses on raw wikitext without a chat template; the B3 gate now uses the base variant `Qwen2.5-1.5B-4bit`. Re-run pending.

For the full interpretation and per-model recommendations see
[`docs/turbo-kv-cache.md`](../turbo-kv-cache.md).

## TurboQuant KV cache — M5 Max speed gate readings

First dedicated M5 Max reading of the TurboQuant KV speed gate matrix.
Hardware: Apple M5 Max, 128 GB unified memory, macOS 26.4.1 (build 25E253).
Model: `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (local dir
`models/llama-3.1-8b-4bit`). Date: 2026-05-03. Binary: mlxcel 0.0.25
post- (fused Sparse-V Metal kernel landed). Reproducer:

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
| `turbo4-delegated` |  27.28 | 0.269× | ≥0.97× | **fail** (partial fix landed) |
| `turbo3-asym`      |   6.36 | 0.063× | (tracking only) | tracking |

 caches the cold-V dequant graph across decode steps;
informal in-tree A/B (100-token decode at the same 4K prompt) measures
`turbo4-delegated` at ~41 tok/s post-fix vs ~27 tok/s on v0.0.25, a ~1.5×
decode speedup that scales sharply at longer contexts.

**Phase-1b (K-side unification):** Removes `cold_keys` and
the per-step `concat(cold_k, hot_k)` graph node. Informal A/B on M5 Max
(3 warm runs each, `llama-3.1-8b-4bit`, 4109-token prompt, 100 generated
tokens): fp16 baseline 101.5–102.7 tok/s; turbo4-delegated post-
43.0–43.7 tok/s (~0.43× FP16, up from ~0.41× pre-fix). The modest speedup is
explained by `SliceUpdate::eval_gpu` semantics: MLX copies the full source
buffer before writing the update region, so per-step K-side memory traffic is
approximately conserved between the old concat layout and the new slice-update
layout. The remaining cost is the V-side `concat(cold_v_dequant, hot_v)`
graph node (Phase 2).

**Phase-2 (fused dequant + SDPA kernel):** Adds a Metal kernel
that reads the packed cold V indices directly inside the kernel, removing
the earlier `cold_v_dequant_cache` memo and the per-step
`concat(cold_v_dequant, hot_v)` graph node. The dequantised cold V never
materialises in global memory — V-memory budget stays at 4-bit packed.

Measured on `llama-3.1-8b-4bit`, 4109-token prompt, 100 generated tokens
(`benchmarks/turbo_kv/2026-05-04_Apple_M5_Max_issue_528_fused_delegated_sdpa.csv`):

| Path | tok/s | × FP16 | gate |
|---|---:|---:|---:|
| `fp16` | 101.76 | 1.000× | baseline |
| `turbo4-delegated` default (no memo) | 29.60 | 0.291× | ≥0.97× — **fail** |
| `turbo4-delegated` fused kernel (`MLXCEL_TURBO4_DELEGATED_FUSED=1`) | 18.90 | 0.186× | ≥0.97× — **fail** |

Removing the earlier memo (per the issue body's "creates dead state, must
not remain" requirement) drops the legacy `update_and_fetch` route from
~0.43× to 0.29×. The fused kernel runs slower than the no-memo legacy
route because the host pipeline now composes Q·K + softmax + cold-kernel +
hot-matmul + sum out of many small MLX graph ops; the memo path could feed
the dequantised cold V into a single steel-attention SDPA call. The
0.97× M5 Max decode gate is **not cleared**. Bringing the
kernel inside the steel-attention envelope is left to follow-up work.

**Phase-3 (steel-attention-envelope kernel, M5 Max measurement).**
Phase 3 lands `turbo4_delegated_steel_sdpa` — a JIT-compiled Metal kernel
that runs the entire post-Q·K SDPA inline (per-Q numerically stable
softmax, cold-V dequant + weighted sum, hot-V FP16 weighted sum, all
normalised against the same softmax denominator). Bit-parity is gate-1
and was validated at PR-landing time (RMS < 5e-3 over 200 decode steps,
two new parity tests in `cache::turbo_tests`). M5 Max throughput was
deferred to a follow-up bench run because the kernel-author agent had no
M5 Max access's run.

Measured on `llama-3.1-8b-4bit`, 4109-token prompt, 100 generated tokens
(`benchmarks/turbo_kv/2026-05-06_Apple_M5_Max_issue_531_steel_envelope.csv`):

| Path | tok/s | × FP16 | gate |
|---|---:|---:|---:|
| `fp16` | 102.97 | 1.000× | baseline |
| `turbo4-delegated` legacy `update_and_fetch + attention()` (env unset) | 29.60* | 0.291× | ≥0.97× — **fail** (reading) |
| `turbo4-delegated` cold-only fused kernel (`MLXCEL_TURBO4_DELEGATED_FUSED=1`, earlier) | 18.90* | 0.186× | ≥0.97× — **fail** (reading) |
| `turbo4-delegated` steel envelope (`MLXCEL_TURBO4_DELEGATED_FUSED=1`, post-Phase 3) | 16.23 | 0.158× | ≥0.97× — **fail** |

`*` cross-referenced from the 2026-05-04 CSV.

The steel envelope runs slower than both the cold-only fused kernel and
the legacy fetch route. The likely cause is the kernel's single-thread-per-Q
softmax + V accumulation pass — at decode time (`Tq=1`, `B=1`, `Hkv=8` on
llama-3.1-8b) only 8 threads are dispatched per kernel call, each scanning
the full T_total range serially. The implementation note
acknowledges this design ("single thread; T_total reads << kernel launch
overhead, avoids threadgroup tree-reduction barriers") — the assumption
held on M1 Ultra at parity contexts but breaks on M5 Max where the
threadgroup tree-reduction would actually be faster than the serial scan.

**Follow-up readings (Pass 1 parallelization + cold-loop sparse cutoff on top).** The follow-up splits the kernel's Pass 1 (per-Q max + sum_exp) across
all D threads of each threadgroup with a tree reduction. The same follow-up
also precomputes a score-space sparse-V cutoff
`max + log(threshold * sum_exp)`, letting the cold loop reject fully-dead tokens
before paying the exp + dequant cost. Pass 2's weighted-sum remains
D-parallelized exactly as in Phase 3. Measured on `llama-3.1-8b-4bit`,
4109-token prompt, 100 generated tokens
(`benchmarks/turbo_kv/2026-05-06_Apple_M5_Max_issue_534_post_fix.csv`):

| Path | tok/s | × FP16 | gate |
|---|---:|---:|---:|
| `fp16` | 103.28 | 1.000× | baseline |
| `turbo4-delegated` steel envelope **earlier** (reading) | 16.23 | 0.158× | ≥0.97× — **fail** |
| `turbo4-delegated` steel envelope **post-** (this fix)        | 19.21 | 0.186× | ≥0.97× — **still fail** |

Pass 1 parallelization plus the score-space cutoff nudges the steel envelope
slightly past the cold-only fused kernel at 4K (19.21 vs 18.90)
but does not move the needle far enough to clear the 0.97× gate. The residual
cost is in Pass 2's per-token T_total scan. A simdgroup broadcast experiment
was also measured during and regressed 4K decode, so it was not retained.
The broader simdgroup-hybrid pattern from MLX upstream's
`metal::steel::SDPA` (per-simdgroup `simd_max` / `simd_sum` plus
per-simdgroup partial-sum accumulators in Pass 2) is the proposed next
iteration; it would change the per-token per-thread T_total scan into a
per-token per-simdgroup scan (4–8× fewer scans on M5 Max for D=128 / D=256).

#### Post- TurboQuant+ delegated FP16 working-set experiment (4K)

Follow-up on 2026-05-07 after comparing `turboquant_plus`: the MLX
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
| `turbo4-delegated` steel envelope **post-** | 19.21 | 0.183x | >=0.97x — **fail** |
| `turbo4-delegated` FP16 fast path + pre-decode compact | 104.09 | 0.989x | >=0.97x — **pass** |

The fast path is 5.4x faster than the post- steel envelope at 4K and 3.5x
faster than the legacy `update_and_fetch + attention` reading
(29.60 tok/s). It clears the 0.97x gate because the one-time sidecar
compaction is no longer charged to the first decode forward. This remains a
speed-path experiment, not the compressed-only memory target, because the full
FP16 V working set is retained while the env var is enabled. The handoff is
still visible in prefill timing for the decode-stage row: 2462.77 ms vs
1271.07 ms for FP16 at 4K.

#### Post- lazy sidecar policy experiment (4K)

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
| `turbo4-delegated` |  3.41 | 0.054× | 21 | ≥0.95× | **fail** (— see below) |
| `turbo3-asym`      |  1.85 | 0.029× | 54 | (tracking only) | tracking |

The repeated-paragraph prompt hits an EOS early on `fp16`, `turbo4-asym`,
`turbo4-delegated`, and `turbo3-asym` at 16K; the per-token rate is
computed over the actually generated tokens. `int8` and symmetric `turbo4`
ran the full 80 tokens.

#### 16K reading (50-token decode, no EOS early-exit)

Measured on `llama-3.1-8b-4bit`, ~16065-token prompt
(`benchmarks/turbo_kv/2026-05-04_Apple_M5_Max_issue_528_fused_delegated_sdpa.csv`):

| Path | tok/s | × FP16 | gate |
|---|---:|---:|---:|
| `fp16` | 74.25 | 1.000× | baseline |
| `turbo4-delegated` default (no memo) | 6.03 | 0.081× | ≥0.95× — **fail** |
| `turbo4-delegated` fused kernel | 5.12 | 0.069× | ≥0.95× — **fail** |

Same shape as the 4K reading: removing the earlier memo (requirement) regressed the legacy fetch path; the fused kernel is slower
still. The gate is wider here because at 16K the dequant cost dominates;
the per-step memo materialised ~52 MB / layer of FP16 cold V, which is
gone, but the kernel cannot replace the steel-attention SDPA pipeline
the memo enabled.

#### 16K reading (early-EOS at 19 generated tokens)

Measured on `llama-3.1-8b-4bit`, 16163-token prompt, 100 requested decode
tokens (early EOS at 19 on both modes — same prompt shape early-exits FP16
and steel envelope at the same point, so the per-token ratio remains
fair). Same CSV as the 4K reading
(`benchmarks/turbo_kv/2026-05-06_Apple_M5_Max_issue_531_steel_envelope.csv`):

| Path | tok/s | Generated | × FP16 | gate |
|---|---:|---:|---:|---:|
| `fp16` | 63.94 | 19 | 1.000× | baseline |
| `turbo4-delegated` steel envelope (post-Phase 3) | 2.39 | 19 | 0.037× | ≥0.95× — **fail** |

The 16K decode ratio (3.7% of FP16) is the gate's worst-case shortfall in
the TurboQuant KV speed gate matrix to date, ~1.4× worse than the cold-only kernel
reading (5.12 tok/s, 0.069× FP16). At 16K the per-token
serial scan over T_total is ~16K reads × 8 threads, completely dwarfing
the tens of milliseconds the FP16 attention path needs for the same step.

#### 16K reading (Pass 1 parallelization, early-EOS at 19–21 tokens)

Same prompt shape as the reading; FP16 early-exits at 19 and
the post- turbo4-delegated path at 21 (one extra token before EOS).
CSV: `benchmarks/turbo_kv/2026-05-06_Apple_M5_Max_issue_534_post_fix.csv`.

| Path | tok/s | Generated | × FP16 | gate |
|---|---:|---:|---:|---:|
| `fp16` | 64.78 | 19 | 1.000× | baseline |
| `turbo4-delegated` steel envelope **earlier** | 2.39 | 19 | 0.037× | ≥0.95× — **fail** |
| `turbo4-delegated` steel envelope **post-** (this fix) | 2.99 | 21 | 0.046× | ≥0.95× — **still fail** |

The fixes move the 16K ratio from 3.7% to 4.6% of FP16 (a 25% relative
improvement) but do not clear the gate. The residual gap is in Pass 2; see the
simdgroup-hybrid follow-up note in the 4K subsection above.

#### Post- TurboQuant+ delegated FP16 working-set experiment (16K)

Same fast-path experiment as the 4K subsection, measured on the 16163-token
prompt. Both modes early-exited at 19 generated tokens
(`benchmarks/turbo_kv/2026-05-07_Apple_M5_Max_issue_534_fp16_fast_path_predecode_compact_16k.csv`):

| Path | tok/s | Generated | x FP16 | gate |
|---|---:|---:|---:|---:|
| `fp16` | 65.55 | 19 | 1.000x | baseline |
| `turbo4-delegated` steel envelope **post-** | 2.99 | 21 | 0.046x | >=0.95x — **fail** |
| `turbo4-delegated` FP16 fast path + pre-decode compact | 70.37 | 19 | 1.074x | >=0.95x — **pass** |

This confirms the earlier fast-path bottleneck was the first-decode sidecar
compaction placement. Once the initial sidecar fold runs during the handoff,
decode uses the same unified FP16 K/V native-SDPA hot path as FP16 mode. The
16K run is short because of early EOS, so the >1.0x ratio should be read as
FP16-class rather than a stable speedup claim. The handoff cost moved into the
decode-stage prefill timing: 11070.22 ms vs 7952.33 ms for FP16 at 16K.

#### Post- lazy sidecar policy experiment (16K)

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

The Turbo decode gates do **not** pass on the
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

The fused Sparse-V Metal kernel is a measured regression vs.
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

### Acceptance criteria status

| Criterion | Status |
|---|---|
| Decode + prefill numbers measured for 5 KVCacheModes (fp16, int8, turbo4-asym, turbo4, turbo4-delegated) at 4K decode + 8K prefill on M5 Max | done |
| 16K decode reading on M5 Max (primary M5 Max gate cell) | done |
| `turbo3-asym` reading on M5 Max (tracking only per epic) | done |
| 32K decode reading | deferred — best-effort per epic; useful only after the kernel regression for `turbo4-asym` is investigated |
| Cross-hardware consistency check vs. M1 Ultra | done — M5 Max numbers are 1.4–2.4× M1 Ultra on Turbo decode |
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
