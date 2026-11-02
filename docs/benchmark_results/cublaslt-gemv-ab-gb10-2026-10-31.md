# cuBLASLt version and gemv-vs-cuBLASLt decode A/B (GB10): result

**Status: measured on spark-102, 2026-11-02.** Answer to the two questions:

1. Does mlxcel get faster on CUDA if it runs the cuBLASLt that the `mlx` 0.32.4 wheel ships (13.8.1.7) instead of the one it links today (the CUDA 13.0 toolkit's 13.1.1.3)? **No resolved effect.** The only resolved point is a 2.3% prefill TTFT win for 13.8 on the 1B at prompt 4096, which the 9B does not reproduce; decode and the server ladder are unresolved.
2. Does B=1 decode of a bf16 model get faster if MLX's own gemv kernel is bypassed and those matmuls go to cuBLASLt? **No. MLX gemv is faster**, by 21.2% (1B), 8.9% (8B) and 18.9% (9B) decode throughput.

The campaign ran on spark-102 after the first attempt on spark-101 was stopped by the host gate (spark-101 is an Actions runner). Raw records are in this directory.

## Why these two questions

Issue #2732 traced a small numerical difference between mlxcel and the `mlx` wheel oracle in the image-generation conditioners to the cuBLASLt version, not to mlxcel code (`docs/image-generation.md`, "GEMM differences against the oracle"). With the wheel's libraries preloaded into mlxcel (`LD_PRELOAD`) the two outputs were bit-identical. The difference is the kernel each version picks for small-M bf16 GEMMs: under 13.1 they run `cutlass_80_wmma` kernels, under 13.8 they run `nvjet_sm121_*` kernels, which are tuned for this GPU. That raised the performance question: if 13.8 picks sm_121-specific kernels, it may be faster where cuBLASLt is on the hot path.

Where cuBLASLt is on the hot path matters for what to expect:

| Workload | Dense matmul path on CUDA | Expected effect of a newer cuBLASLt |
|---|---|---|
| B=1 decode, bf16/f16 model | MLX gemv (`cu::can_use_gemv`: `M == 1` with transposed `B`, or `N == 1`, no bias) | little; cuBLASLt is barely called |
| Prefill (large M) | cuBLASLt | possible |
| Batched decode, M = batch (2 to 7 rows on CUDA) | cuBLASLt | likeliest place for a difference; small M is exactly where the kernel choice changed |
| Any quantized model (4bit, 8bit, nvfp4) | MLX qmv/qmm kernels | none expected (control) |

Question 2 comes from the first row. B=1 decode of `llama-3.2-1b-instruct-bf16` runs at about 88 tok/s, about 218 GB/s of weight traffic. That is below the GB10's nominal memory bandwidth, and MLX's gemv is a general kernel. If cuBLASLt's sm_121 kernels read weights closer to peak, routing M=1 to them would help every unquantized decode. The `MLXCEL_CUDA_DISABLE_GEMV=1` switch added with this record does that routing (see `docs/environment-variables.md`).

## Environment where the campaign ran

| Item | Value |
|------|-------|
| **Host** | spark-102, NVIDIA GB10 (sm_121), driver 580.178.04, Linux 7.0.0-1019-nvidia |
| **Linked cuBLASLt** | 13.1.1.3 from the CUDA 13.0 toolkit, `/usr/local/cuda/targets/sbsa-linux/lib` |
| **Alternate cuBLASLt** | 13.8.1.7 from the pip `nvidia-cublas` wheel the `mlx` 0.32.4 oracle venv installs, `/tmp/cublas138/lib/python3.12/site-packages/nvidia/cu13/lib` (`libcublasLt.so.13`, `libcublas.so.13`, depending only on the system libc) |
| **apt candidates** (not installed) | `libcublas-13-2` 13.4.1.3, `libcublas-13-3` 13.6.0.2, `libcublas-13-4` 13.8.0.4 |

The same rows from the first attempt on spark-101, where setup ran and the campaign was stopped, were identical except for the host. `campaign.log` writes these rows for every run.

## Harness

Everything lives in `data/cublaslt-gemv-ab-gb10-2026-10-31/harness/`:

| File | Role |
|---|---|
| `bench_wrap.sh` | `--bin` for `scripts/engine_bench_rounds.py`. Arms pass only flags, so it turns `--cublas-alt` into an `LD_PRELOAD` of `$MLXCEL_AB_CUBLAS_DIR` and `--no-gemv` into `MLXCEL_CUDA_DISABLE_GEMV=1`, then runs `mlxcel-bench-engine` from `$MLXCEL_AB_BIN_DIR` (default `target/release`) |
| `ladder_rounds.py` | Server A/B for batched decode: arms `sys`, `alt` and a null repeat of each, rotated per round; every arm is a fresh `mlxcel-server --parallel 7 --max-batch-prefill 4` followed by `scripts/bench_serving_concurrency.py --concurrency 1,4,7`. Each record names the `libcublasLt` the server actually mapped (from `/proc/<pid>/maps`), so a failed preload is visible |
| `campaign.sh` | Runs the three stages below in order and writes JSONL, text logs and `campaign.log` to `$MLXCEL_AB_OUT` (default `./ab-results`). Uses `gpu-lock` when it is installed |

Both drivers use the interleaved-rounds protocol with a null arm per arm (`--null-every-arm`) and the #1820 host gate (`--hostgate`). A delta inside the null arms' spread is unresolved.

### Stages

| Stage | Models | Arms | Shape |
|---|---|---|---|
| `version` | `llama-3.2-1b-instruct-bf16`, `qwen3.5-9b-bf16`, `qwen3-1.7b-4bit` (control) | `sys`, `alt` | CLI path, prompts 256 and 4096, 128 decode tokens, 5 rounds |
| `gemv` | `llama-3.2-1b-instruct-bf16`, `qwen3.5-9b-bf16`, `llama-3.1-8b-instruct-bf16` | `gemv` (default), `lt-sys` (no gemv, linked cuBLASLt), `lt-alt` (no gemv, alternate cuBLASLt) | CLI path, prompt 256, 256 decode tokens, 5 rounds |
| `ladder` | `llama-3.2-1b-instruct-bf16`, `llama-3.1-8b-instruct-bf16` | `sys`, `alt` | server, concurrency 1/4/7, prompt 512, 128 decode tokens, 3 rounds |

`qwen3.5` is a hybrid model, which the server clamps to `--parallel 1`, so it is left out of the ladder. CUDA batched decode covers at most 7 rows, which is why the ladder stops at 7.

## Results

Run at `ceb76567` (main `f1839fd6` plus this PR's commit), CUDA release build (`--features cuda`, `MLX_CUDA_ARCHITECTURES=121`), 2026-11-02 17:49 to 21:30 KST. Raw JSONL, per-model text summaries and `campaign.log` sit next to this file. Every arm passed the host gate; the CLI stages recorded a zero wait on every arm but one (155 s, qwen3.5-9b in the gemv stage), so the host was quiet throughout. The ladder records confirm the preload took effect: `sys` mapped `/usr/local/cuda-13.0/.../libcublasLt.so.13.1.1.3` and `alt` mapped `/tmp/cublas138/.../libcublasLt.so.13` in every round.

### cuBLASLt 13.8 vs 13.1 (version stage)

Medians over five rounds, with the paired delta `alt` vs `sys` and the range of that paired delta. `null spread` is each arm's paired delta against its own repeat, as a percent; a delta whose range overlaps it is unresolved.

| Model | Prompt | Metric | sys | alt | alt vs sys | null spread | Verdict |
|---|---|---|---|---|---|---|---|
| llama-3.2-1b-bf16 | 256 | decode tok/s | 99.34 | 99.44 | +0.08% (-0.23..+0.56) | -0.59% / -0.04% | unresolved |
| llama-3.2-1b-bf16 | 256 | TTFT ms | 15.14 | 14.89 | -0.58% (-8.16..+3.14) | -0.47% / -2.11% | unresolved |
| llama-3.2-1b-bf16 | 4096 | decode tok/s | 93.97 | 94.12 | +0.31% (-0.13..+0.77) | +0.32% / -0.28% | unresolved |
| llama-3.2-1b-bf16 | 4096 | TTFT ms | 148.48 | 144.93 | -2.32% (-3.38..-1.50) | +0.45% / +0.69% | alt faster |
| qwen3.5-9b-bf16 | 256 | decode tok/s | 14.19 | 14.17 | -0.01% (-3.47..+0.08) | +0.04% / 0.00% | unresolved |
| qwen3.5-9b-bf16 | 256 | TTFT ms | 159.50 | 158.99 | -2.15% (-6.36..+3.24) | -2.48% / -0.83% | unresolved |
| qwen3.5-9b-bf16 | 4096 | decode tok/s | 14.04 | 14.05 | +0.03% (-2.39..+0.12) | +0.08% / +0.02% | unresolved |
| qwen3.5-9b-bf16 | 4096 | TTFT ms | 2694.33 | 2745.25 | +1.74% (-0.39..+2.46) | -0.38% / -0.75% | unresolved (trend slower) |
| qwen3-1.7b-4bit | 256 | decode tok/s | 184.83 | 185.47 | +0.21% (-0.05..+0.49) | -0.26% / +0.02% | unresolved |
| qwen3-1.7b-4bit | 256 | TTFT ms | 18.77 | 18.77 | +0.33% (-1.99..+5.32) | -0.38% / +0.21% | unresolved |
| qwen3-1.7b-4bit | 4096 | decode tok/s | 135.39 | 135.12 | -0.53% (-0.60..+0.42) | -0.36% / -0.52% | unresolved |
| qwen3-1.7b-4bit | 4096 | TTFT ms | 291.13 | 291.26 | +0.06% (-0.80..+0.90) | -0.16% / -0.12% | unresolved |

Decode is unaffected everywhere, consistent with the gemv stage: cuBLASLt is not on the B=1 hot path. The single resolved point, the 1B prefill TTFT at prompt 4096, is not reproduced on the 9B, and the 4bit control (where cuBLASLt should not be on the hot path at all) stays inside noise. One resolved point out of twelve, on the smallest model, is not a reason to change the shipped library.

### gemv vs cuBLASLt (gemv stage)

Prompt 256, 256 decode tokens, five rounds. `lt-sys` and `lt-alt` set `MLXCEL_CUDA_DISABLE_GEMV=1` and differ only in which cuBLASLt is preloaded. Numbers are decode tok/s; the deltas are paired per round against `gemv`.

| Model | gemv | lt-sys | lt-alt | lt-sys vs gemv | lt-alt vs gemv | Null spread | Verdict |
|---|---|---|---|---|---|---|---|
| llama-3.2-1b-bf16 | 90.94 | 71.58 | 71.68 | -21.21% (-21.40..-21.05) | -21.17% (-21.38..-20.87) | +0.02% / +0.01% | gemv faster |
| llama-3.1-8b-bf16 | 15.38 | 14.00 | 14.02 | -8.86% (-9.01..-7.56) | -8.86% (-8.95..-3.71) | +0.20% / +0.01% | gemv faster |
| qwen3.5-9b-bf16 | 13.43 | 11.01 | 11.43 | -18.87% (-19.08..-16.42) | -18.93% (-19.17..-14.89) | -0.16% / -0.05% | gemv faster |

All decode deltas are outside the null spread. Disabling gemv also costs prefill TTFT on the 1B (`lt-sys` +12.08%, `lt-alt` +9.37%, null spread -0.48% / +0.25%); on the 8B +1.65% is inside its null (-1.26..+1.68), and on the 9B the TTFT delta is unresolved. `lt-alt` vs `lt-sys` decode medians are +0.05% (1B), +0.02% (8B) and -0.13% (9B), so the cuBLASLt version does not matter once gemv is off either.

### Server ladder (ladder stage)

Medians over three rounds. The first group is decode tok/s, the second aggregate tok/s, and the last column the null arms' paired drift against their own source.

| Model | Conc | sys decode | alt decode | alt vs sys | sys aggregate | alt aggregate | alt vs sys | Null drift (decode) |
|---|---|---|---|---|---|---|---|---|
| llama-3.2-1b-bf16 | 1 | 86.4 | 86.2 | -0.12% | 84.1 | 84.0 | -0.12% | -0.23% / 0.00% |
| llama-3.2-1b-bf16 | 4 | 65.0 | 66.2 | +1.38% | 247.0 | 251.2 | +1.05% | +0.46% / -0.15% |
| llama-3.2-1b-bf16 | 7 | 55.8 | 56.8 | +1.79% (1.63..1.93) | 382.1 | 388.6 | +1.70% (1.56..2.08) | +2.89% (2.28..3.76) / +0.17% (-3.20..+1.06) |
| llama-3.1-8b-bf16 | 1 | 14.8 | 14.9 | +0.68% | 14.7 | 14.7 | 0.00% | 0.00% / 0.00% |
| llama-3.1-8b-bf16 | 4 | 13.3 | 13.2 | -0.75% | 52.0 | 51.8 | -0.38% | 0.00% / 0.00% |
| llama-3.1-8b-bf16 | 7 | 12.0 | 12.0 | -0.83% | 82.1 | 82.0 | -0.36% | -0.83% / -0.83% |

No ladder point is resolved. On the 1B at concurrency 7 both decode and aggregate favor `alt` by 1.7 to 1.8 percent, but the `sys-null` repeat is 2.9 percent faster than `sys`, so the null drift is larger than the effect. Concurrency 4 on the 1B contains one round with a 92 percent outlier in both `sys` and `sys-null` after a server start; the medians ignore it, and it explains the wide paired ranges there. The 8B is flat at every level.

## What to do with the result

- **cuBLASLt version: keep 13.1.1.3.** 13.8 does not make mlxcel faster on any path that is resolved, except the single 2.3% prefill TTFT win on the 1B that the 9B does not reproduce. The ladder, where a newer cuBLASLt would matter most (small-M batched GEMMs), is unresolved. Revisit only if a future workload shows cuBLASLt-bound time that moves with the version.
- **gemv: keep it as the default.** Routing M=1 dense matmuls to cuBLASLt costs 21.2% (1B), 8.9% (8B) and 18.9% (9B) decode throughput, and on the 1B another 12% TTFT. Nothing in this campaign shows cuBLASLt winning a B=1 shape.
- **The switch stays a diagnostic.** It is what made the second answer measurable, it is off by default, and `docs/environment-variables.md` documents it as such. Do not promote it to a tuning knob.
- **Numerics.** The gemv-off arms are not bit-identical to gemv (different reduction order), which is a second reason the switch is diagnostic-only. The #2732 oracle difference is unchanged because the shipped library stays 13.1.

## Status

| Item | State |
|---|---|
| `MLXCEL_CUDA_DISABLE_GEMV` switch | in this PR, exercised by the `gemv` stage |
| `version` stage | ran on spark-102, 3 models, 5 rounds, no failures |
| `gemv` stage | ran on spark-102, 3 models, 5 rounds, no failures |
| `ladder` stage | ran on spark-102, 2 models, 3 rounds, no failures |
| Raw data | this directory: `version_*`, `gemv_*`, `ladder_*` (`.jsonl` and `.txt`), `campaign.log` |
| First attempt | spark-101, stopped by the host gate after 2+ hours with 0 records |

Refs #2732.
