# Model Compatibility & Performance Tests (M5 Max)

Compatibility and performance testing for mlxcel models on **MacBook Pro M5 Max 128GB**, with same-host mlx-lm / mlx-vlm reference measurements and M1 Ultra ratios where available.

## Test Environment

| Item | Value |
|------|-------|
| **Hardware** | MacBook Pro M5 Max, 128GB RAM |
| **OS** | macOS 26.6.2 (build 25G83) |
| **mlxcel version** | 0.7.0-beta.1 (`mlxcel_version`) |
| **Source revision** | `a50ff440` (`mlxcel_commit`); MLX pin `9a795735` (`mlx_commit`). The VLM pass and two re-checked text rows record `a50ff440-dirty`. That working tree is exactly `a50ff440` plus the diff committed as `34455e42`, so those rows are reproducible from `34455e42` and the rest from `a50ff440`. |
| **MLX version** | upstream main (via mlxcel-core; pinned commit `9a795735`) |
| **mlx-lm baseline** | 0.31.3 (dev checkout https://github.com/ml-explore/mlx-lm, commit `ed1fca4`); not re-run for the 0.6.0 sweep, see note below |
| **mlx-vlm baseline** | 0.4.4; not re-run for the 0.6.0 sweep |
| **Test Prompt** | Text: a deterministic synthetic 512-token prompt (`--prompt-tokens 512`). VLM: "What is in this image?" plus `tests/fixtures/test_image.png`, at the checkpoint's own image token count. |
| **Max Tokens** | 128, with every end-of-generation token suppressed (`--ignore-eos`), so every row spends the full budget |
| **Test Date** | 2026-09-06 full re-benchmark (text, VLM, speculative, batched serving, embeddings) on the pp512/tg128 condition; prior: 2026-09-03/04 full text + VLM sweep (0.6.0), 2026-07-11/12 (0.4.0-rc.1), 2026-06-15 (0.2.1), 2026-05-27 (0.1.0) |
| **Benchmark Status** | Full re-benchmark on mlxcel 0.7.0-beta.1. Text: 176 directories via `bench_decode.sh all`, 147 with decode numbers. VLM: `all --vlm`, 71 with decode numbers. Both used `--cooldown 30 --big-cooldown 30`, which remain required on this host, and `BENCH_MEM_OVERHEAD_FACTOR=1.209` (a 90 GB weight budget). Time Machine was confirmed idle for the whole campaign. **The measurement condition changed this round and the prefill column is not comparable to any earlier sweep**; see "Measurement condition: pp512/tg128" below before reading any delta. The `vs M1 Ultra` column is blanked for the same reason: the M1 Ultra reference is the 2026-07-12 sweep taken at the old condition, and a ratio spanning both a version and a condition change is not a hardware comparison. It returns when that host is re-swept at pp512/tg128. The `mlxcel vs mlx-lm` / `vs mlx-vlm` percentages further down still carry the 2026-05-18 Python baselines and are likewise on the old condition. |

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

## Measurement condition: pp512/tg128

This sweep is the first on a fixed measurement interval, and the change is large
enough that per-model numbers should not be compared to any earlier sweep
without reading this section.

**What it replaces.** Decode throughput used to be timed over a model-dependent
number of tokens: a 6-token chat prompt, up to 100 generated tokens, stopping at
EOS. Only a quarter of models reached the budget and a sixth stopped under 20
tokens, so each row was measured over a different interval. The bias runs both
ways, which is why it went unnoticed for so long: a very short run charges
first-token latency to steady-state throughput and reads high, while a merely
short one enjoys an almost-empty KV cache and also reads high. Fixed in
`3a746ea8` to llama-bench's pp512/tg128 default, which
`scripts/bench_serving_concurrency.py` already used, so the single-stream and
batched harnesses now agree on the condition.

**The fix is visible in its own results.** Every model that got materially
*faster* is one whose 2026-09-03 row was statistically meaningless:

| Model | 2026-09-06 | 2026-09-03 | ratio | 09-03 sample |
|-------|-----------|-----------|-------|--------------|
| `phi-2-4bit` | 181.80 | 106.08 | 1.71x | **1 token** |
| `granite-4.1-8b-4bit` | 90.32 | 53.99 | 1.67x | **1 token** |
| `falcon-mamba-7b-4bit` | 94.64 | 66.60 | 1.42x | **2 tokens** |
| `baichuan-m1-14b-4bit` | 63.51 | 57.66 | 1.10x | 7 tokens |

Nothing about those models changed. The old rows were latency samples wearing a
throughput label.

**Prefill is not comparable at all.** The prompt went from 6 tokens to 512, and
`prefill_tok_s` at the longer length amortizes fixed per-call overhead over far
more tokens, so ratios against 2026-09-03 run from 2.5x to 27x. These are not
speedups. Compare prefill only against another pp512 sweep.

**Decode carries a small systematic offset.** The median decode ratio against
2026-09-03 is 0.982 across 146 comparable models. Decoding with a 512-entry KV
cache costs slightly more than with a 6-entry one; roughly 1.8% is the price of
the honest condition, not a regression.

### The Qwen VL cluster is context sensitivity, not a regression

Seven Qwen VL checkpoints read 15-26% below their 2026-09-03 rows while every
non-VL Qwen sat at parity, which looked like an architecture-specific
regression. It is not. Re-running the **old** condition on the **current** build
reproduces the old numbers within 1%:

| Model | old condition, current build | 2026-09-03 | ratio | pp512/tg128 |
|-------|------------------------------|-----------|-------|-------------|
| `qwen3-vl-4b-4bit` | 184.46 | 183.89 | 1.00 | 135.89 |
| `qwen3-vl-2b-4bit` | 368.44 | 370.75 | 0.99 | 295.86 |
| `qwen2.5-vl-3b-4bit` | 161.92 | 162.04 | 1.00 | 139.55 |
| `qwen3-4b-4bit` (control) | 190.11 | 191.19 | 0.99 | 186.47 |
| `llama-3.1-8b-4bit` (control) | 116.32 | 114.92 | 1.01 | 114.63 |

No code changed under these models between `b2ff1eee` and `a50ff440`. What the
comparison does show is a real and previously invisible property: Qwen VL decode
loses 26% going from a 6-token to a 512-token context where a plain Qwen decoder
loses 2%. That is a standing optimization target for the M-RoPE attention path,
and the old condition was hiding it rather than the new one inventing it.

All twelve models that moved more than 10% were also re-measured individually on
an idle machine; every one reproduced its sweep value within 3%, so no row here
is a thermal or I/O artifact. Two of them (`glm-4.1v-9b-thinking-4bit`,
`glm-4.5v-4bit`) landed during a storage-clearing operation on the host and read
2-3% low; their isolated re-measurements replace the sweep rows.

### The VLM pass had to be run twice

The first `all --vlm` pass of this campaign produced a byte-for-byte duplicate of
the text sweep. `--prompt-tokens` synthesizes a text-only prompt and
`src/bin/bench_decode.rs` documents that it ignores `--image` in that mode;
`3a746ea8` made `PROMPT_TOKENS=512` the default for *every* mode, so the VLM
sweep passed both and the image was dropped without a warning. The tell was that
all 147 rows reported `prompt_tokens=512` and text-only checkpoints such as
`llama-3.1-8b-4bit` were "passing" a VLM sweep. The harness now omits
`--prompt-tokens` in VLM mode and the runner rejects the combination outright
rather than ignoring half of it (`34455e42`). The VLM table below is the
corrected re-run, taken after that fix was applied and the binary rebuilt; its
prompt token counts vary per checkpoint, as they should. The defective first
pass was discarded rather than recorded, so no committed CSV carries a row from
it.

## Legend

- ✅ Pass: Model works correctly
- ⚠️ Partial: Loads but output quality problems or low token count
- ❌ Fail: Does not work

## Basic Transformers

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| llama3 | Llama-3.2-1B-Instruct-4bit | ✅ | 19674.78 | 524.03 | n/a |  |
| llama3.1 | Llama-3.1-8B-Instruct-4bit | ✅ | 3421.12 | 114.63 | n/a |  |
| llama3 (8B bf16) | Llama-3.1-8B-Instruct (bf16) | ⚠️ | 3749.97 | 33.30 | n/a | bf16; slow decode |
| llama4 | Llama-4-Scout-17B-16E-4bit | ✅ | 1021.71 | 47.05 | n/a |  |
| command-r7b | c4ai-command-r7b-4bit | ✅ | 3256.89 | 112.63 | n/a |  |
| aya-expanse-8b | aya-expanse-8b-4bit | ✅ | 3229.75 | 112.15 | n/a |  |
| aya-vision-8b | aya-vision-8b (text-only) | ✅ | 3255.30 | 111.74 | n/a | text-only |
| deepseek-r1 | DeepSeek-R1-Distill-Qwen-7B-4bit | ✅ | 3648.50 | 124.34 | n/a |  |
| internlm2 | InternLM2-7B-4bit | ✅ | 3462.83 | 115.66 | n/a |  |
| internlm3 | internlm3-8b-instruct-4bit | ✅ | 2980.66 | 99.00 | n/a |  |
| mimo | MiMo-7B-RL-4bit | ✅ | 3618.60 | 116.71 | n/a |  |
| minicpm | MiniCPM-2B-sft-bf16-4bit | ✅ | 8800.59 | 214.17 | n/a |  |
| bunny-llama3-8b | Bunny-Llama-3-8B-V-4bit (text) | ✅ | 3422.10 | 115.74 | n/a | text-only |
| llava-1.5-7b | llava-1.5-7b-4bit (text) | ✅ | 3906.08 | 118.31 | n/a | text-only |
| llava-next | llava-v1.6-mistral-7b-4bit (text) | ✅ | 3586.26 | 120.39 | n/a | text-only |
| llava-interleave | llava-interleave-qwen-0.5b-bf16 (text) | ✅ | 49730.26 | 367.95 | n/a |  |

## Gemma Family

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| gemma | gemma-2b-it-4bit | ✅ | 9733.08 | 210.43 | n/a |  |
| gemma2 | gemma-2-2b-it-4bit | ✅ | 8685.58 | 197.96 | n/a | full-budget raw prompt 245.83 tok/s |
| gemma3 | gemma-3-1b-it-4bit | ✅ | 21786.19 | 374.16 | n/a |  |
| gemma3 (4B) | gemma-3-4b-it-4bit | ✅ | 6290.55 | 178.27 | n/a | full-budget raw prompt 183.77 tok/s |
| gemma3n (E2B) | gemma-3n-E2B-it-4bit | ✅ | 6264.62 | 155.22 | n/a |  |
| gemma3n (E4B) | gemma-3n-E4B-it-4bit | ✅ | 3922.32 | 108.50 | n/a |  |
| gemma3n (E4B bf16) | gemma-3n-E4B-it (bf16) | ✅ | 3921.86 | 39.68 | n/a | Gemma3n language MLP bf16 preserved, other bf16 materialized as f16; M5 (Neural Accelerator) uses the split decode path while other Apple Silicon uses the fused path; ~80% of mlx-lm decode |
| gemma4 (26B MoE) | gemma-4-26b-a4b-it-4bit | ✅ | 3493.90 | 146.08 | n/a |  |
| gemma4 (31B) | gemma-4-31b-4bit | ✅ | 809.75 | 27.71 | n/a |  |
| gemma4 (31B IT) | gemma-4-31b-it-4bit | ✅ | 810.52 | 27.45 | n/a |  |
| gemma4 (31B nvfp4) | Gemma-4-31b-it-nvfp4 | ⚠️ | 843.61 | 16.18 | n/a | nvfp4 has no fast Metal kernel |
| gemma4 (E2B 4bit) | gemma-4-e2b-it-4bit | ✅ | 9219.08 | 218.38 | n/a |  |
| gemma4 (E2B 8bit) | gemma-4-e2b-it-8bit | ✅ | 7779.86 | 146.65 | n/a |  |
| gemma4 (E4B 4bit) | gemma-4-e4b-it-4bit | ✅ | 4746.74 | 135.79 | n/a |  |
| gemma4 (E4B 8bit) | gemma-4-e4b-it-8bit | ✅ | 4197.58 | 84.84 | n/a |  |
| gemma4 (12B) | gemma-4-12b-it-4bit | ✅ | 1850.64 | 45.19 | n/a | NEW (6-13) |
| gemma4 (26B QAT) | gemma-4-26b-a4b-it-qat-4bit | ✅ | 3393.37 | 138.51 | n/a | QAT; NEW (6-13) |
| gemma4 (31B IT QAT) | gemma-4-31b-it-qat-4bit | ✅ | 745.20 | 17.71 | n/a | QAT; NEW (6-13) |
| gemma4 (E2B QAT) | gemma-4-e2b-it-qat-4bit | ✅ | 7859.86 | 169.19 | n/a | QAT; NEW (6-13) |
| gemma4 (E4B QAT) | gemma-4-e4b-it-qat-4bit | ✅ | 4238.43 | 97.57 | n/a | QAT; NEW (6-13) |

## EXAONE

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| exaone | EXAONE-3.5-2.4B-Instruct-4bit | ✅ | 9753.57 | 260.16 | n/a |  |
| exaone4 | exaone-4.0-1.2b-4bit | ✅ | 15536.14 | 419.31 | n/a |  |

## Qwen Family

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| qwen2.5 (0.5B) | Qwen2.5-0.5B-Instruct-4bit | ✅ | 41900.10 | 624.52 | n/a | same checkpoint as `qwen2-0.5b`, measured under that name (dedup #1615) |
| qwen2.5 (0.5B bf16) | Qwen2.5-0.5B-Instruct (bf16) | ✅ | 42838.17 | 387.22 | n/a |  |
| qwen2.5 (7B) | Qwen2.5-7B-Instruct-4bit | ✅ | 3646.36 | 124.00 | n/a | same checkpoint as `qwen2.5-7b-4bit`, measured under that name (dedup #1615) |
| qwen2.5 (7B 8bit) | Qwen2.5-7B-Instruct-8bit | ✅ | 3457.43 | 67.58 | n/a |  |
| qwen2.5-vl (3B) | qwen2.5-vl-3b-4bit | ✅ | 5575.58 | 139.55 | n/a | re-downloaded (prior FAIL was a corrupt checkpoint, not a code bug) |
| qwen2-vl (2B) | Qwen2-VL-2B-Instruct-4bit | ✅ | 10836.69 | 233.65 | n/a |  |
| qwen1.5-moe | Qwen1.5-MoE-A2.7B-Chat-4bit | ✅ | 6188.95 | 248.02 | n/a |  |
| qwen3 (0.6B) | Qwen3-0.6B-4bit | ✅ | 33266.02 | 519.05 | n/a |  |
| qwen3 (1.7B) | Qwen3-1.7B-4bit | ✅ | 13903.90 | 354.06 | n/a |  |
| qwen3 (4B) | Qwen3-4B-4bit | ✅ | 6055.01 | 186.47 | n/a |  |
| qwen3 (8B) | Qwen3-8B-4bit | ✅ | 3341.12 | 111.66 | n/a |  |
| qwen3-30b-a3b | Qwen3-30B-A3B-4bit | ✅ | 3727.82 | 170.53 | n/a |  |
| qwen3-moe | Qwen3-MoE-30B-4bit | ✅ | 3727.82 | 170.53 | n/a | same checkpoint as `qwen3-30b-a3b-4bit`, measured under that name (dedup #1615) |
| qwen3-vl (2B) | Qwen3-VL-2B-Instruct-4bit | ✅ | 13723.71 | 295.86 | n/a | text-only |
| qwen3-vl (4B) | qwen3-vl-4b-4bit | ✅ | 6005.71 | 135.89 | n/a | text-only; NEW (6-13) |
| qwen3-vl (8B) | qwen3-vl-8b-4bit | ✅ | 3326.71 | 89.32 | n/a | text-only; NEW (6-13) |
| qwen3-vl (30B MoE) | Qwen3-VL-30B-A3B-Instruct-4bit | ✅ | 3595.54 | 131.22 | n/a | text-only |
| qwen3-vl (32B) | Qwen3-VL-32B-Instruct-4bit | ✅ | 804.08 | 23.22 | n/a | text-only |
| qwen3-next (80B MoE) | Qwen3-Next-80B-A3B-Instruct-4bit | ✅ | 2229.26 | 121.49 | n/a | NEW (0.4.0-rc.1) |
| qwen3-omni (30B MoE) | Qwen3-Omni-30B-A3B-Instruct-4bit | ✅ | 3616.09 | 131.12 | n/a | text path; NEW (0.4.0-rc.1) |
| qwen3-coder (480B) | Qwen3-Coder-480B-A35B-Instruct-4bit | ❌ | - | FAIL | - | SKIP:oom_estimate |
| qwen3.5 (0.8B) | Qwen3.5-0.8B-4bit | ✅ | 20078.20 | 522.03 | n/a |  |
| qwen3.5 (2B) | Qwen3.5-2B-4bit | ✅ | 10490.49 | 332.03 | n/a |  |
| qwen3.5 (4B) | Qwen3.5-4B-4bit | ✅ | 4895.66 | 170.95 | n/a |  |
| qwen3.5 (9B) | Qwen3.5-9B-4bit | ✅ | 2894.86 | 104.18 | n/a |  |
| qwen3.5 (9B bf16) | Qwen3.5-9B (bf16) | ✅ | 3125.40 | 31.09 | n/a |  |
| qwen3.5 (27B) | Qwen3.5-27B-4bit | ✅ | 918.98 | 33.60 | n/a |  |
| qwen3.5-35b-a3b | Qwen3.5-35B-A3B-4bit | ✅ | 3123.24 | 163.75 | n/a |  |
| qwen3.6-35b-a3b | Qwen3.6-35B-A3B-4bit | ✅ | 3176.30 | 157.11 | n/a | NEW (5-18) |

## Phi Family

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| phi-2 | phi-2-hf-4bit-mlx | ⚠️ | 5975.28 | 181.80 | n/a | 1 token; (likely EOS) |
| phi-3-mini | Phi-3-mini-4k-instruct-4bit | ✅ | 6837.66 | 195.47 | n/a |  |
| phi-3.5-mini | Phi-3.5-mini-instruct-4bit | ✅ | 6829.00 | 191.42 | n/a |  |
| phi-3.5-moe | Phi-3.5-MoE-instruct-4bit | ✅ | 1955.69 | 112.54 | n/a |  |
| phi-3.5-vision | Phi-3.5-vision-instruct-4bit | ✅ | 6824.92 | 190.89 | n/a | text-only |
| phi-4 | Phi-4-4bit | ✅ | 1831.00 | 62.42 | n/a |  |

## OLMo Family

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| olmo-1b | OLMo-1B-hf-4bit | ✅ | 14713.84 | 208.80 | n/a |  |
| olmo2-7b | OLMo2-7B-4bit | ✅ | 3697.82 | 113.66 | n/a |  |
| olmo3-32b | OLMo3.1-32B-4bit | ✅ | 798.39 | 28.73 | n/a |  |

## MoE (Mixture of Experts)

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| minimax | MiniMax-M2-3bit | n/a | n/a | n/a | n/a | not measured 2026-09-06: checkpoint absent from `models/`; last value from the 0.4.0-rc.1 sweep |
| mixtral | Mixtral-8x7B-Instruct-v0.1-4bit | ✅ | 1307.11 | 65.82 | n/a |  |
| gpt_oss (20B) | gpt-oss-20b-MXFP4-Q4 | ✅ | 3736.59 | 168.92 | n/a |  |
| gpt_oss (120B) | gpt-oss-120b-4bit | ✅ | 1585.03 | 113.56 | n/a |  |
| solar-open-100b | Solar-Open-100B-4bit | ✅ | 1113.43 | 63.49 | n/a |  |
| dots.llm1 | dots.llm1.inst-mixed-4-6bit | ✅ | 818.69 | 49.96 | n/a | mixed 4/6-bit; NEW (6-13) |
| lfm2-moe | lfm2-8b-a1b-4bit | ✅ | 6321.67 | 339.90 | n/a | NEW (6-13) |

## DeepSeek Family

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| deepseek | deepseek-coder-1.3b-instruct-4bit | ✅ | 19055.95 | 185.30 | n/a |  |
| deepseek_v2 | DeepSeek-V2-Lite-Chat-4bit | ✅ | 1060.07 | 206.45 | n/a |  |
| deepseek_v3 | - | ❌ | - | FAIL | - | SKIP:oom_estimate |

## MLA (Multi-head Latent Attention)

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| minicpm3 | MiniCPM3-4B-4bit | ✅ | 4878.64 | 119.33 | n/a |  |

## Nemotron Family

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| nemotron_h | Nemotron-H-30B-4bit | ✅ | 762.37 | 178.43 | n/a |  |
| nemotron_nas | Nemotron-NAS-30B-A3B-4bit | ✅ | 762.37 | 178.43 | n/a | same checkpoint as `nemotron-h-30b-4bit`, measured under that name (dedup #1615) |
| nemotron-omni | Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | ✅ | 766.89 | 178.77 | n/a | text path; NEW (6-14) |

## SSM / Mamba Models

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| mamba | Falcon-Mamba-7B-4bit | ⚠️ | 306.17 | 94.64 | n/a | chat template EOS |
| mamba2 | mamba2-1.3b-4bit | ✅ | 6105.67 | 167.66 | n/a |  |
| mamba2 (130M) | mamba2-130m | ✅ | 38730.66 | 342.94 | n/a | NEW (6-14) |
| jamba | Jamba-v0.1-4bit | ✅ | 227.97 | 213.02 | n/a | raw prompt 215.74 tok/s |
| falcon-h1 | falcon-h1-tiny-90m-instruct-4bit | ✅ | 13954.79 | 160.70 | n/a | Mamba2 + attention hybrid; NEW (6-13) |
| plamo2 | plamo-2-1b | ✅ | 10923.87 | 86.83 | n/a | Mamba + attention hybrid; NEW (6-13) |

## Chinese / Asian Language Models

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| baichuan | Baichuan-M1-14B-Instruct-4bit | ✅ | 1871.46 | 63.51 | n/a |  |
| glm4_moe_lite | GLM-4.7-Flash-4bit | ✅ | 3071.02 | 107.13 | n/a |  |
| ernie4_5 | ERNIE-4.5-0.3B-Instruct-4bit | ✅ | 51892.77 | 949.30 | n/a |  |
| hunyuan_moe | hunyuan-a13b-instruct-4bit | ✅ | 1004.87 | 66.10 | n/a | A13B MoE (4-bit), canonical after checkpoint dedup; same checkpoint as `hunyuan-13b`, measured under that name (dedup #1615) |
| hunyuan_v1_dense | Hunyuan-1.8B-Instruct-4bit | ✅ | 11094.20 | 315.14 | n/a |  |

## Other Models

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| ministral3 | Ministral-3B-Instruct-4bit | ✅ | 7311.74 | 227.35 | n/a | VLM wrapper |
| mistral-small | mistral-small-3.1-24b-4bit | ✅ | 1142.12 | 41.18 | n/a |  |
| molmo2 | molmo2-4b | ✅ | 4496.59 | 102.93 | n/a |  |
| molmo-7b | molmo-7b | ✅ | 3581.47 | 123.21 | n/a | text spot-check |
| internvl3 | internvl3-1b | ✅ | 41046.88 | 629.06 | n/a |  |
| smollm-135m | SmolLM-135M-Instruct-4bit | ✅ | 95618.27 | 812.22 | n/a |  |
| smollm3-3b | SmolLM3-3B-4bit | ✅ | 7475.02 | 226.48 | n/a |  |
| stablelm-1.6b | stablelm-2-1_6b-chat-4bit | ✅ | 16343.71 | 394.89 | n/a |  |
| starcoder2-3b | starcoder2-3b-4bit | ✅ | 7834.05 | 208.94 | n/a |  |
| pixtral-12b | pixtral-12b-4bit | ✅ | 2241.66 | 74.66 | n/a | text-only |
| paligemma2-3b | paligemma2-3b (6-bit) | ✅ | 7499.35 | 161.72 | n/a | text-only |

## Granite Family

Ported 2026-06-13 (dense + Mamba2/attention hybrid + hybrid-MoE). vs M1 Ultra
ratios are from the 2026-07-12 0.4.0-rc.1 M1 Ultra sweep.

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| granite | granite-3.3-2b-instruct-4bit | ✅ | 8562.27 | 251.29 | n/a | dense |
| granite4_h (350M) | granite-4.0-h-350m-4bit | ✅ | 8191.91 | 132.59 | n/a | Mamba2 + attention hybrid |
| granite4_h (tiny) | granite-4.0-h-tiny-4bit | ✅ | 4118.80 | 76.54 | n/a | hybrid MoE |
| granite4.1 (3B) | granite-4.1-3b-4bit | ✅ | 6602.98 | 188.91 | n/a |  |
| granite4.1 (8B) | granite-4.1-8b-4bit | ⚠️ | 2984.26 | 90.32 | n/a | 1 token; likely early EOS, re-check with a code prompt |

## Recently Ported Families (2026-06-13/14)

New architectures landed in the 06-13/06-14 wave. vs M1 Ultra ratios are from
the 2026-07-12 0.4.0-rc.1 M1 Ultra sweep.

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| apertus | apertus-8b-instruct-2509-4bit | ✅ | 3283.86 | 113.04 | n/a | xIELU, QK-norm, llama3 RoPE |
| bitnet (4bit pack) | bitnet-b1.58-2b-4t-4bit | ✅ | 974.29 | 314.45 | n/a | 1.58-bit ternary |
| bitnet | bitnet-b1.58-2b-4t | ✅ | 974.99 | 252.11 | n/a | 1.58-bit ternary |
| lfm2 | lfm2-350m-8bit | ✅ | 54799.17 | 836.77 | n/a | 8-bit |
| seed-oss | seed-oss-36b-instruct-4bit | ✅ | 725.07 | 25.67 | n/a |  |
| minicpm-v (4.6) | minicpm-v-4.6-bf16 | ✅ | 21476.55 | 269.13 | n/a | text path |
| youtu-vl | youtu-vl-4b-instruct | ✅ | 4728.61 | 47.64 | n/a | text path |

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
| gemma2 (9B 8bit) | gemma-2-9b-8bit | ✅ | 2303.64 | 41.53 | n/a |  |
| mistral-small-4 (119B) | mistral-small-4-119b-2603-4bit | ✅ | 902.50 | 19.04 | n/a | dense 119B |
| phi-3-small | phi-3-small-8k-instruct-aq4_64 | ✅ | 3162.30 | 111.33 | n/a | aq4_64 |
| llada2.0-mini | llada2.0-mini-preview-4bit | ✅ | 8917.72 | 322.03 | n/a | diffusion LM |
| deepseek-ocr | deepseek-ocr-4bit | ✅ | 26108.30 | 624.68 | n/a | text path |
| deepseek-ocr-2 | deepseek-ocr-2-4bit | ✅ | 26197.97 | 621.93 | n/a | text path |
| deepseek-vl2 | deepseek-vl2-small-4bit | ✅ | 1058.06 | 205.11 | n/a | text path |
| fastvlm | fastvlm-0.5b-bf16 | ✅ | 42753.06 | 386.55 | n/a | text path |
| glm-4.1v | glm-4.1v-9b-thinking-4bit | ✅ | 1989.65 | 56.01 | n/a | text path |
| glm-4.5v | glm-4.5v-4bit | ✅ | 658.90 | 15.24 | n/a | text path |
| glm-ocr | glm-ocr-4bit | ✅ | 25457.07 | 438.18 | n/a | text path |
| granite4-vision (3B) | granite-4.0-3b-vision-4bit | ✅ | 6567.06 | 202.62 | n/a | text path |
| granite-vision (2B) | granite-vision-3.2-2b-4bit | ✅ | 8580.53 | 252.68 | n/a | text path |
| idefics2 | idefics2-8b-4bit | ✅ | 3549.20 | 118.97 | n/a | text path |
| idefics3 | idefics3-8b-llama3-4bit | ✅ | 3386.60 | 115.61 | n/a | text path |
| kimi-vl | kimi-vl-a3b-thinking-4bit | ✅ | 1081.23 | 165.40 | n/a | A3B MoE; text path |
| lfm2-vl | lfm2-vl-450m-4bit | ✅ | 45398.45 | 1016.41 | n/a | text path |
| llama-3.2-vision (11B) | llama-3.2-11b-vision-instruct-4bit | ✅ | 3422.87 | 114.46 | n/a | text path |
| moondream2 | moondream2 | ✅ | 8777.41 | 41.54 | n/a | text path |
| paddleocr-vl | paddleocr-vl-bfloat16 | ✅ | 29541.80 | 141.46 | n/a | text path |
| smolvlm | smolvlm-instruct-bf16 | ✅ | 15042.14 | 134.43 | n/a | text path |

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
| qwen3.8-27b | qwen3.8-27b-4bit | ✅ | 916.96 | 33.61 | n/a | NEW (0.6.0); qualified on the `qwen3_5` path (#1174) |
| hunyuan (13B) | hunyuan-13b | ✅ | 1004.87 | 66.10 | n/a | NEW (0.6.0) |
| diffusiongemma (26B MoE) | diffusiongemma-26b-a4b-it-4bit | ✅ | 3437.62 | 114.20 | n/a | NEW (0.6.0) |
| minicpm-v (4.6 mxfp4) | minicpm-v-4.6-mxfp4 | ✅ | 20231.72 | 338.46 | n/a | NEW (0.6.0) |
| qwen3.5 (0.8B optiq) | qwen3.5-0.8b-optiq-4bit | ✅ | 19553.86 | 435.58 | n/a | NEW (0.6.0) |
| qwen2.5 (1.5B) | qwen2.5-1.5b-instruct-4bit | ✅ | 15957.70 | 361.85 | n/a | NEW (0.6.0) |
| dots.ocr | dots.ocr-4bit | ⚠️ | 15942.04 | 360.87 | n/a | loads and prefills but emits no text on a text-only prompt |
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
`tests/fixtures/test_image.png`, and each generates exactly 128 tokens with EOS
suppressed. Unlike the text tables, prefill here runs at each checkpoint's own
image token count (32 to 1032 tokens across the roster), which is inherent to
VLM prefill and why no single `--prompt-tokens` value applies.

Every VLM row that moved more than 10% against 2026-09-04 moved *up*, at an
identical prompt token count, because the old rows stopped at EOS after 2 to 28
tokens where these run the full 128: `qwen3-omni-30b-a3b-instruct-4bit` +57%
(2 tokens then), `paligemma2-3b-6bit` +44% (2), `deepseek-ocr-2-4bit` +65% (5),
`moondream2` +31% (4). 81% of the 69 shared models land within 10%, median
1.035.

| Model | Test Model | Status | Prefill | Decode | vs M1 Ultra | Notes |
|-------|------------|--------|---------|--------|-------------|-------|
| aya-vision-8b | aya-vision-8b | ✅ | 2636.11 | 112.10 | n/a |  |
| bunny-llama3-8b | bunny-llama3-8b-4bit | ✅ | 2842.91 | 114.89 | n/a |  |
| gemma3 (4B) | gemma3-4b-4bit | ✅ | 564.65 | 177.96 | n/a | same checkpoint as `gemma-3-4b-it-4bit`, measured under that name (dedup #1615) |
| gemma3n (E2B 4bit) | gemma3n-e2b-4bit | ✅ | 2973.46 | 157.54 | n/a |  |
| gemma3n (E4B 4bit) | gemma3n-e4b-4bit | ✅ | 2228.35 | 109.94 | n/a |  |
| gemma3n (E4B bf16) | gemma3n-e4b-bf16 | ✅ | 2164.95 | 40.06 | n/a | bf16→f16 conversion path |
| gemma4 (26B MoE) | gemma-4-26b-a4b-it-4bit | ✅ | 902.47 | 150.31 | n/a |  |
| gemma4 (31B) | gemma-4-31b-4bit | ✅ | 432.98 | 28.13 | n/a |  |
| gemma4 (31B IT) | gemma-4-31b-it-4bit | ✅ | 442.50 | 28.14 | n/a |  |
| gemma4 (E2B 4bit) | gemma-4-e2b-it-4bit | ✅ | 2841.78 | 223.27 | n/a |  |
| gemma4 (E2B 8bit) | gemma-4-e2b-it-8bit | ✅ | 2644.07 | 148.93 | n/a |  |
| gemma4 (E4B 4bit) | gemma-4-e4b-it-4bit | ✅ | 2067.23 | 137.61 | n/a |  |
| gemma4 (E4B 8bit) | gemma-4-e4b-it-8bit | ✅ | 1927.85 | 85.88 | n/a |  |
| internvl3 (1B) | internvl3-1b | ✅ | 6451.25 | 645.33 | n/a |  |
| llama4 (Scout) | llama-4-scout-17b-4bit | ✅ | 396.92 | 48.35 | n/a |  |
| llava-1.5-7b | llava-1.5-7b-4bit | ✅ | 3188.79 | 116.95 | n/a |  |
| llava-interleave | llava-interleave-qwen-0.5b-bf16 | ✅ | 16088.92 | 351.16 | n/a |  |
| llava-next | llava-next-mistral-7b-4bit | ✅ | 2965.97 | 119.84 | n/a |  |
| ministral3 | ministral-3b-4bit | ✅ | 5446.72 | 224.49 | n/a |  |
| mistral-small (3.1 24B) | mistral-small-3.1-24b-4bit | ✅ | 1042.62 | 41.32 | n/a |  |
| molmo-7b | molmo-7b | ✅ | 2341.68 | 123.58 | n/a | mlx-vlm baseline is a 1-token anomaly |
| molmo2 (4B) | molmo2-4b | ✅ | 2465.99 | 103.02 | n/a |  |
| paligemma2 (3B 6-bit) | paligemma2-3b-6bit | ✅ | 5232.20 | 139.53 | n/a |  |
| phi-3.5-vision | phi-3.5-vision-4bit | ✅ | 3731.31 | 185.16 | n/a |  |
| pixtral (12B) | pixtral-12b-4bit | ✅ | 1595.85 | 75.68 | n/a | intermittent slow VLM decode reads (~20 tok/s) seen on M5, not consistently reproducible (see Known Issues) |
| qwen2-vl (2B) | qwen2-vl-2b-4bit | ✅ | 2488.92 | 270.25 | n/a | EOS-terminate |
| qwen2.5-vl (3B) | qwen2.5-vl-3b-4bit | ✅ | 1724.32 | 160.01 | n/a | re-downloaded (prior FAIL was a corrupt checkpoint) |
| qwen3-vl (2B) | qwen3-vl-2b-4bit | ✅ | 2115.66 | 273.10 | n/a |  |
| qwen3-vl (4B) | qwen3-vl-4b-4bit | ✅ | 1171.47 | 136.35 | n/a | NEW (6-13) |
| qwen3-vl (8B) | qwen3-vl-8b-4bit | ✅ | 986.15 | 81.70 | n/a | NEW (6-13) |
| qwen3-vl (30B MoE) | qwen3-vl-30b-a3b-4bit | ✅ | 539.25 | 59.15 | n/a |  |
| qwen3-vl (32B) | qwen3-vl-32b-4bit | ✅ | 295.95 | 19.14 | n/a |  |
| gemma4 (12B) | gemma-4-12b-it-4bit | ✅ | 1416.44 | 45.45 | n/a | NEW (6-13) |
| minicpm-v (4.6) | minicpm-v-4.6-bf16 | ✅ | 933.30 | 272.43 | n/a | NEW (6-13) |
| nemotron-omni | nemotron-3-nano-omni-30b-a3b-reasoning-4bit | ✅ | 646.27 | 181.31 | n/a | NEW (6-14) |
| youtu-vl | youtu-vl-4b-instruct | ✅ | 518.03 | 48.06 | n/a | NEW (6-13) |
| deepseek-ocr | deepseek-ocr-4bit | ✅ | 1610.78 | 659.63 | n/a | NEW (0.4.0-rc.1) |
| deepseek-ocr-2 | deepseek-ocr-2-4bit | ✅ | 1553.99 | 631.67 | n/a | NEW (0.4.0-rc.1) |
| deepseek-vl2 | deepseek-vl2-small-4bit | ✅ | 860.61 | 207.02 | n/a | NEW (0.4.0-rc.1) |
| fastvlm | fastvlm-0.5b-bf16 | ✅ | 2669.19 | 392.97 | n/a | NEW (0.4.0-rc.1) |
| glm-4.1v | glm-4.1v-9b-thinking-4bit | ✅ | 807.13 | 64.84 | n/a | NEW (0.4.0-rc.1) |
| glm-4.5v | glm-4.5v-4bit | ✅ | 210.95 | 17.02 | n/a | NEW (0.4.0-rc.1) |
| granite4-vision (3B) | granite-4.0-3b-vision-4bit | ✅ | 2635.07 | 205.31 | n/a | NEW (0.4.0-rc.1) |
| granite-vision (2B) | granite-vision-3.2-2b-4bit | ✅ | 6192.79 | 233.33 | n/a | NEW (0.4.0-rc.1) |
| idefics2 | idefics2-8b-4bit | ✅ | 913.68 | 121.44 | n/a | NEW (0.4.0-rc.1) |
| idefics3 | idefics3-8b-llama3-4bit | ✅ | 2095.94 | 117.62 | n/a | NEW (0.4.0-rc.1) |
| kimi-vl | kimi-vl-a3b-thinking-4bit | ✅ | 754.18 | 173.45 | n/a | A3B MoE; NEW (0.4.0-rc.1) |
| lfm2-vl | lfm2-vl-450m-4bit | ✅ | 5228.47 | 1026.21 | n/a | NEW (0.4.0-rc.1) |
| llama-3.2-vision (11B) | llama-3.2-11b-vision-instruct-4bit | ✅ | 14.23 | 72.57 | n/a | NEW (0.4.0-rc.1) |
| moondream2 | moondream2 | ✅ | 52.90 | 41.36 | n/a | NEW (0.4.0-rc.1) |
| paddleocr-vl | paddleocr-vl-bfloat16 | ✅ | 3359.38 | 149.77 | n/a | NEW (0.4.0-rc.1) |
| smolvlm | smolvlm-instruct-bf16 | ✅ | 2109.44 | 130.30 | n/a | NEW (0.4.0-rc.1) |
| qwen3-omni (30B) | qwen3-omni-30b-a3b-instruct-4bit | ✅ | 579.05 | 59.08 | n/a | NEW (0.4.0-rc.1) |

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

Counts reflect the 2026-09-06 `bench_decode.sh all --cooldown 30 --big-cooldown 30`
text sweep on mlxcel 0.7.0-beta.1, with `BENCH_MEM_OVERHEAD_FACTOR=1.209`
(a 90 GB weight budget), at pp512/tg128.

| Status | Count |
|--------|-------|
| ✅ Pass (measured decode) | 144 |
| ⚠️ Partial (loads; slow path or no text output) | 6 |
| ❌ Fail / OOM-skip | 4 |

The ⚠️ set carries over from 2026-09-03 and was **not** re-derived this round.
Three of the six were flagged for early EOS (`phi-2-4bit`,
`falcon-mamba-7b-4bit`, `granite-4.1-8b-4bit`), and `--ignore-eos` makes early
EOS unobservable by construction: every model now runs the full 128 tokens
whether or not it wanted to stop. The remaining three are unaffected by the
condition (`gemma-4-31b-it-nvfp4` and `llama-3.1-8b-bf16` have no fast kernel;
`dots.ocr-4bit` loads and prefills but emits no text on a text-only prompt).
Re-deriving the early-EOS flags needs a separate pass with EOS live.

**How 154 table rows reconcile with 176 enumerated directories.** The sweep walks
every directory in `models/`, but the tables above deliberately do not carry a row
per directory:

| | Count |
|---|---|
| Directories enumerated by `all` | 176 |
| ...measured (decode numbers) | 147 |
| ...collapsed as duplicate checkpoints (#1615, see the alias table above) | 14 |
| ...`FAIL:bench` | 10 |
| ...`SKIP:oom_estimate` (over the 90 GB weight budget) | 2 |
| ...`SKIP:not_a_checkpoint` (no `config.json`) | 2 |
| ...`SKIP:missing_weights` (`config.json` but no readable shard) | 1 |

Of the 147 measured checkpoints, 146 have a row in the text or VLM tables above;
`qwen2.5-1.5b-4bit` is measured but unlisted, being a second quantization of a
listed checkpoint rather than a distinct family. Of the 154 text table rows, 153
map to a measured checkpoint and one (`MiniMax-M2-3bit`) does not, because that
checkpoint has been removed from disk since 0.4.0-rc.1; its row is blanked rather
than left carrying a stale number.

**None of the 10 `FAIL:bench` is a decode-path defect.** Four are non-text
checkpoints the text harness cannot decode (`whisper-base`, `kokoro-82m`,
`granite-speech-4.1-2b-nar-mlx`, `docling-layout-heron-mlx-bf16`); six are
speculative drafters and `dflash` / `mtp` head variants, which are not standalone
generative models and are measured in the speculative table instead. The GLM-5
pair that used to sit in this bucket is now correctly classified as a local data
problem (`SKIP:missing_weights` and `SKIP:not_a_checkpoint`) rather than sharing
a token with real defects, which is one of the reclassifications `3a746ea8`
added.

**No decode regression survived verification.** 146 checkpoints are comparable
with 2026-09-03, at a median ratio of 0.982. 122 moved less than 10% either way.
All 12 that fell more than 10% were re-measured individually on an idle machine
and every one reproduced its sweep value within 3%, and the largest cluster among
them (seven Qwen VL checkpoints) was traced to context length rather than code by
re-running the old condition on the current build. The 4 that gained more than
10% all had 1 to 7 token samples in the old condition. See
"Measurement condition: pp512/tg128" above for both.

## Batched serving (B = 1/2/4)

Source: `benchmarks/metal_m5max_batch_2026-09-06.csv`, produced by
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
| 1 | 1 / 0 | 26.7 | 26.7 | 377.0 | 352.1 | 1.00x |
| 2 | 2 / 0 | 14.8 | 18.6 | 324.0 | 629.2 | 1.79x |
| 4 | 4 / 0 | 22.1 | 33.6 | 290.5 | 1113.5 | **3.16x** |

### llama-3.1-8b-4bit (canonical dense 4-bit)

| B | ok/fail | TTFT mean (ms) | TTFT p95 (ms) | decode tok/s per request | aggregate tok/s | scaling vs B=1 |
|---|---------|----------------|---------------|--------------------------|-----------------|----------------|
| 1 | 1 / 0 | 175.3 | 175.3 | 105.4 | 92.7 | 1.00x |
| 2 | 2 / 0 | 53.8 | 70.9 | 99.7 | 192.7 | **2.08x** |
| 4 | 4 / 0 | 87.7 | 139.2 | 78.7 | 300.5 | **3.24x** |

### qwen3-30b-a3b-4bit (MoE; batched decode hits the fused-MoE path)

| B | ok/fail | TTFT mean (ms) | TTFT p95 (ms) | decode tok/s per request | aggregate tok/s | scaling vs B=1 |
|---|---------|----------------|---------------|--------------------------|-----------------|----------------|
| 1 | 1 / 0 | 149.3 | 149.3 | 169.7 | 142.6 | 1.00x |
| 2 | 2 / 0 | 197.8 | 197.9 | 118.5 | 201.7 | 1.41x |
| 4 | 4 / 0 | 458.5 | 458.5 | 79.4 | 248.6 | 1.74x |

### Reading

The dense models scale close to linearly to B=4 (3.16x and 3.24x on aggregate
throughput) while giving up 23-25% of per-request decode. The MoE model does not:
it reaches only **1.74x** aggregate at B=4, per-request decode falls 53% (169.7
-> 79.4), and TTFT rises 3.1x (149 -> 459 ms) *despite* the warm prompt cache
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


**Attribution was done on M1 Ultra, and the M5 Max re-run is pending.** Issue #1616 read these rows as the MoE decode path declining the fused kernel at B>=2. Profiling on M1 Ultra found a different cause: `Qwen3MoeModel` never overrode `forward_batched`, so every batching family without that override ran the single-sequence `forward` once per row and got its aggregate only from overlapping independent graphs. The fix and the full attribution, including an op-level measurement showing the batched fused kernel the issue proposed loses to `gather_qmm` from n=4, are in [moe-batched-decode-m1ultra-2026-09-04.md](moe-batched-decode-m1ultra-2026-09-04.md) and in the M1 Ultra document. These M5 Max numbers predate that change and are left as measured; re-running this ladder on M5 Max is what would show its effect here.

## Performance vs mlx-lm / mlx-vlm baseline (2026-05-19 benchmark campaign)

> **This section is on the old measurement condition and was not re-run on
> 2026-09-06.** Both sides of every comparison below were taken with the 6-token
> prompt, 100-token budget and live EOS, so the percentages remain internally
> consistent and must not be read against the pp512/tg128 tables above. Mixing
> the two would compare a 512-token-context mlxcel number against a
> 6-token-context Python number and attribute the difference to the runtime.
> `scripts/bench_mlxlm.py` gained the matching pp512/tg128 shape in `3a746ea8`
> (deterministic synthetic prompt from a byte-identical corpus, EOS suppressed
> through `logits_processors`), so refreshing this section is now a matter of
> re-running the Python baselines rather than a harness change.

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
