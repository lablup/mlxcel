# Volta (sm_70) baseline: throughput, roofline attainment, and kernel profile

This is Phase 0 of epic #1536, the Volta decode program. Every sub-issue from #1539 to #1545 states its acceptance criteria as a delta against the numbers here, so this document is the reference they are measured against. It is analogous in role to #624 for the GB10 program (#623).

Everything below was produced from a build made for this document on the machine described in the environment table, not copied from the audit in #1536. Where a reproduced number disagrees with that audit, the disagreement is reported rather than smoothed over; the "What reproduced" section lists every case.

Raw rows: `benchmarks/cuda_v100_2026-08-31.csv`, written by `scripts/bench_decode.sh` in the schema the GB10 sweeps in the same directory use.

## Environment

| Item | Value |
|------|-------|
| **GPU** | Tesla V100-PCIE-32GB, sm_70, compute capability 7.0, 32 GB HBM2 |
| **Peak bandwidth** | 900 GB/s, the roofline denominator for decode |
| **Peak FP32** | 14 TFLOPS, the roofline denominator for prefill. The 112 TFLOPS FP16 tensor path is not reached by any kernel measured here |
| **Driver / toolkit** | 575.51.03 / CUDA 12.9.41 |
| **Host** | Intel Xeon Gold 6138, 16 cores visible to the build, 755 GB RAM, Ubuntu 24.04.2, kernel 5.15.0-156, x86_64 |
| **Toolchain** | gcc 13.3.0, cmake 3.31.6, Rust 1.97.1 (pinned in `rust-toolchain.toml`) |
| **mlxcel** | 0.6.0 at `c7ef9e4e` (main, after PR #1551 for issue #1537) |
| **MLX** | pinned commit `9a795735` (`src/lib/mlx-cpp/CMakeLists.txt` `GIT_TAG`) |
| **nsys** | Nsight Systems 2025.3.1.90 |
| **Contention** | none. This machine has exactly one GPU and every measurement ran with `nvidia-smi --query-compute-apps` empty at the start |

### Checkpoints

All from `mlx-community`, all `group_size` 64, all `affine` quantization mode.

| Checkpoint | Kind | Bits | On disk | Params | Bytes read per decode step |
|---|---|---|---|---|---|
| `qwen3.8-27B-4bit` | dense hybrid, 64 layers (48 linear-attention, 16 full-attention), head_dim 256 | 4 | 16.05 GB | 27.36 B | 14.42 GB |
| `gemma-4-12B-it-4bit` | dense, 48 layers (40 sliding, 8 full) | 4 | 6.74 GB | 11.96 B | 6.17 GB |
| `gemma-4-12b-it-8bit` | dense, 48 layers (40 sliding, 8 full) | 8 | 12.72 GB | 11.96 B | 11.65 GB |
| `gemma-4-26b-a4b-it-4bit` | MoE, 30 layers, 128 experts | 4 | 15.61 GB | 26.35 B | not applicable |
| `gemma-4-26b-a4b-it-8bit` | MoE, 30 layers, 128 experts | 8 | 27.95 GB | 25.81 B | not applicable |

"Bytes read per decode step" is the checkpoint minus the token embedding table (a decode step reads one row of it) and minus the vision tower (not executed on a text-only step), counting every remaining plane together with its scales and biases once. It excludes KV-cache traffic, under one percent of the total at the context lengths used here. It is left blank for the MoE pair, where a step reads only the routed experts, so no MoE roofline is reported.

The two dense Gemma checkpoints are a **controlled pair**. A recursive diff of their `config.json` files reports exactly two differing leaves, `quantization.bits` and `quantization_config.bits`, both 4 against 8; `group_size`, `mode` and every architecture field match, and the parameter counts come out identical to two decimal places (11.96 B). That is what makes the 4-bit against 8-bit comparison below a comparison of quantization width and not of two different models.

## Build provenance

```
MLX_CUDA_ARCHITECTURES=70 make release-cuda
```

25 minutes 30 seconds, incremental against a shared target directory already holding this MLX pin. The architecture list is passed explicitly and never left to detection; see the traps section for why.

```
cuobjdump --list-elf target/release/build/mlxcel-core-*/out/build/lib/libmlx.a \
  | grep -oE 'sm_[0-9]+a?' | sort | uniq -c
     96 sm_70
```

96 cubins, every one of them sm_70, and nothing else. `libmlx.a` is 155,355,188 bytes at this single architecture. For scale, `release.yml` compiles six architectures for the x86_64 artifact and records a cold build of that matrix at about three hours.

The runtime agrees with the archive, which is the check that matters at use time rather than at build time:

```
$ MLXCEL_TRACE_ARCH=1 ./target/release/mlxcel generate -m ./models/mlx-community/gemma-4-12B-it-4bit -p "Hi." -n 8
[mlxcel arch] compute capability 7.0 (sm_70); compiled for [70]; coverage: cubin
[mlxcel arch] quantized matmul path on first call: qmm_naive (device sm_70)
```

## Method, and why each rule is load-bearing

These are not style preferences. Breaking any one of them changes the answer by more than the effects this program is trying to measure, and three of them produced a wrong result during the work that led to this document. Rule 7 was added afterwards, by #1545, on the same basis: it was breaking it that produced the wrong result.

**1. Decode rate is a slope, never `tokens / wall_time` from one run.** Report the marginal per-token cost, obtained from two runs at the same prompt and different `-n`:

```
t = (decode_ms(n_hi) - decode_ms(n_lo)) / (n_hi - n_lo)
C = decode_ms(n_lo) - n_lo * t
```

This record uses `n_lo = 40`, `n_hi = 120`, three repetitions for the dense models and five for the noisier MoE pair, and reports the mean of the per-repetition slopes together with the spread. `C` is not discarded: it is reported in its own right, and it is what #1545 is about.

**2. Both runs must reach the token budget.** A prompt like `"Hi."` makes every instruct checkpoint here emit EOS after roughly ten tokens, so `-n 40` and `-n 120` return the same short generation and the slope becomes a difference of two nearly equal numbers over nearly zero. The first attempt at this record did exactly that and produced slopes ranging from a `ZeroDivisionError` to 193 ms/token on a model whose real figure is 220. Every run below used a prompt that generates past 120 tokens, and the harness asserts `generated_tokens == n` before it computes a slope. The CLI has no `--ignore-eos`; that flag exists on `mlxcel-server` only.

**3. State the PTX cache state.** mlxcel points `MLX_PTX_CACHE_DIR` at `~/.cache/mlxcel/cuda-ptx/<mlx-commit>` (`ensure_persistent_ptx_cache` in `src/lib/mlxcel-core/src/lib.rs`). The cache is keyed on the pinned MLX commit, so it survives rebuilds and every process, and the first run after a fresh machine or a pin bump costs several times what the second one does. The measured gap is in its own section below. Everything else in this document is a warm-cache number.

**4. nsys requires `--cuda-graph-trace=node`.** MLX captures decode into CUDA graphs. Without that flag nsys attributes the graph's work to the launch, roughly all GPU time lands on `event_signal_kernel`, and every real kernel shows near zero.

**5. nsys absolute times are trustworthy only when they reconcile against an unprofiled wall clock.** Graph-node tracing adds per-node instrumentation and does not add it evenly. Compare the profiled decode time against the same run unprofiled before drawing any kernel-level conclusion, and report the ratio. Here the dense pair reconciles at 102.4% and 101.6% and carries the kernel-level conclusion; the MoE 4-bit arm inflates to 144.0% and is reported as a wall-clock observation only.

**6. Percentage-of-total shares do not compare across runs.** Two runs with different totals can move a kernel's share opposite to its absolute time. Compare absolute kernel time and instance counts, and only across runs whose instance counts match.

**7. State the file page-cache state too, and for a first-token number state it first.** Added by #1545, which measured it. MLX loads safetensors lazily, so the first token reads the whole language model off disk; on `qwen3.8-27B-4bit` a 2-token first token costs 15.54 s with the checkpoint in the page cache and 79.22 s without, a 5.1x penalty against the PTX cache's 1.7x. This host's cgroup caps memory at 64 GB while the five checkpoints total 79 GB, so a sweep across models evicts whatever it is not reading and the next model's first run lands on a cold file. Warm the checkpoint with `cat model*.safetensors > /dev/null` immediately before any first-token measurement and say so in the record. Decode numbers are insensitive to this; first-token numbers are not.

## Decode throughput

Slope over `-n 40` to `-n 120` at a fixed 46-token prompt, warm PTX cache.

| Checkpoint | Per-rep ms/token | Mean ms/token | tok/s | Spread | Fixed decode cost `C` |
|---|---|---|---|---|---|
| `qwen3.8-27B-4bit` | 220.25 / 220.29 / 220.44 | **220.33** | **4.54** | 0.09% | -0.10 s |
| `gemma-4-12B-it-4bit` | 124.76 / 124.71 / 123.76 | **124.41** | **8.04** | 0.80% | -0.04 s |
| `gemma-4-12b-it-8bit` | 66.70 / 64.54 / 66.62 | **65.96** | **15.16** | 3.28% | +0.01 s |
| `gemma-4-26b-a4b-it-4bit` | 40.39 / 37.99 / 37.44 / 39.12 / 38.13 | **38.61** | **25.90** | 7.64% | +3.91 s |
| `gemma-4-26b-a4b-it-8bit` | 33.40 / 32.88 / 34.65 / 33.02 / 37.51 | **34.29** | **29.16** | 13.48% | +4.04 s |

The dense models are extremely repeatable, `qwen3.8-27B-4bit` to within 0.09% across three repetitions. The MoE pair is not, which is why it got five repetitions; its 8-bit arm still spans 32.88 to 37.51 ms/token, and a single repetition of it can land slower than the 4-bit mean. Any MoE conclusion drawn from one run of each arm is unsafe.

`C`, the fixed cost inside the decode loop, is zero within measurement error for the dense models on a warm cache and about 4 s for both MoE arms.

### Roofline attainment, decode

| Checkpoint | Bytes/step | ms/token | Achieved | Of 900 GB/s |
|---|---|---|---|---|
| `qwen3.8-27B-4bit` | 14.42 GB | 220.33 | 65.4 GB/s | **7.27%** |
| `gemma-4-12B-it-4bit` | 6.17 GB | 124.41 | 49.6 GB/s | **5.51%** |
| `gemma-4-12b-it-8bit` | 11.65 GB | 65.96 | 176.6 GB/s | **19.63%** |

This is the number the program is steering. Decode on this part is nowhere near bandwidth bound: the best case reaches a fifth of the roofline and the 4-bit cases reach a twentieth. The 8-bit member of the controlled pair attains 3.6 times the bandwidth of the 4-bit member while running the same architecture, which is the whole finding of the next section stated as a roofline number.

## Prefill and time to first token

Prefill is measured as the marginal cost of added prompt tokens, at `-n 1`, so the fixed per-process cost is separated rather than folded into a rate. Ladder on `qwen3.8-27B-4bit`, two repetitions per rung, warm cache:

| Prompt tokens | Prefill (s) | Segment rate |
|---|---|---|
| 54 | 24.94 | |
| 161 | 41.69 | 6.39 tok/s |
| 365 | 61.00 | 10.56 tok/s |
| 671 | 101.60 | 7.54 tok/s |
| 1283 | 179.73 | 7.83 tok/s |

Least squares over the five rungs: **125.07 ms per prompt token, that is 8.00 tok/s marginal**, with a **fixed intercept of 18.41 s**. The segment rates bracket that figure and are not monotonic, so the marginal rate is quoted from the fit rather than from any single pair.

**Time to first token is 24.94 s** for this model at a 54-token prompt, after the model is loaded, and 18.41 s of that is the fixed cost rather than the prompt. A single-run prefill rate over the 671-token prompt gives 6.61 tok/s, which is 17% low precisely because it charges the fixed cost to the prompt.

### Roofline attainment, prefill

FLOPs are counted as `2 * non_embedding_params * prompt_tokens`, that is 24.81 B parameters for `qwen3.8-27B-4bit`, and exclude the attention score and value products, so this is a lower bound on the work and therefore on the attained rate.

| Basis | ms per prompt token | Achieved | Of 14 TFLOPS |
|---|---|---|---|
| Marginal (the fit above) | 125.07 | 0.397 TFLOPS | **2.83%** |
| Single 671-token run | 151.36 | 0.328 TFLOPS | 2.34% |

Prefill reaches under 3% of FP32 peak, and the FP16 tensor path is not reached at all.

## The cold PTX cache

Identical command on `gemma-4-12B-it-4bit`, `-n 40` at a 46-token prompt, with the persistent PTX cache moved aside and then restored:

| Phase | Model load | Prefill (46 tok) | Decode (40 tok) | Process wall |
|---|---|---|---|---|
| Cold cache | 45.79 s | 76.24 s | 9.95 s | **135.13 s** |
| Warm, run 1 | 9.35 s | 11.13 s | 4.93 s | 28.13 s |
| Warm, run 2 | 8.87 s | 11.24 s | 4.93 s | 27.80 s |

The first run after a fresh machine or an MLX pin bump costs 4.8 times the wall clock of the second, prefill 6.8 times, and the decode loop 2.0 times.

This is why rule 3 exists. A measurement that does not state its cache state is not reproducible, and a first-token cost quoted without it can be wrong by a factor of six. It does not by itself explain the audit's TTFT figure; see the next section.

## Kernel profile

`nsys profile -t cuda,nvtx --cuda-graph-trace=node` on `qwen3.8-27B-4bit`, prompt "Explain what a GPU tensor core does." (61 tokens), `-n 24`. Summed GPU time 20.707 s.

| Kernel | Time | Instances | Share |
|---|---|---|---|
| `qmm_naive_kernel` | 14.5741 s | 497 | 70.38% |
| `qmv_kernel` | 5.2279 s | 11,431 | 25.25% |
| `event_signal_kernel` | 0.1377 s | 1,847 | 0.67% |
| `rms_norm_small` | 0.1098 s | 7,320 | 0.53% |
| `volta_sgemm_*` | 0.0749 s | 3,536 | 0.36% |
| `naive_grouped_unfold_transpose_nd` | 0.0695 s | 1,152 | 0.34% |

`qmm_naive` is prefill and `qmv` is decode; the instance counts make that unambiguous, since 11,431 divided by 24 generated tokens is 476 launches per decode step while `qmm_naive` fires once per projection for the single prefill pass.

`cuda_api_sum` on the same run, which is where the graph-construction cost lives:

| API | Time | Calls |
|---|---|---|
| `cudaMemcpyAsync` | 3.9217 s | 1,847 |
| `cudaGraphAddKernelNode_v10000` | 1.8901 s | 64,717 |
| `cudaGraphInstantiate_v12000` | 1.2249 s | 196 |
| `cudaMallocAsync_v11020` | 0.4974 s | 2,524 |
| `cudaGraphLaunch_v10000` | 0.4648 s | 3,815 |
| `cuModuleLoadDataEx` | 0.1722 s | 13 |

Graph construction plus module load is 3.3 s of a 24-token run, against 5.3 s of actual decode kernel time. That ratio is the subject of #1545.

## The 4-bit against 8-bit inversion

The question this issue exists to settle is whether an 8-bit checkpoint decodes faster than a 4-bit one on Volta, despite reading roughly twice the bytes. **It does, on both a dense pair and a MoE pair, and on the dense pair the mechanism is isolated to one kernel.**

| Pair | 4-bit | 8-bit | 8-bit advantage |
|---|---|---|---|
| Dense `gemma-4-12B-it-{4bit,8bit}` | 124.41 ms/tok (8.04 tok/s) | **65.96 ms/tok (15.16 tok/s)** | **1.886x** |
| MoE `gemma-4-26b-a4b-it-{4bit,8bit}` | 38.61 ms/tok (25.90 tok/s) | **34.29 ms/tok (29.16 tok/s)** | 1.126x |

The 4-bit arm reads 6.17 GB per step and the 8-bit arm 11.65 GB, so the arm moving 1.9 times the bytes finishes in 53% of the time. Decode on this part is not bandwidth bound.

### Verdict on the accumulator hypothesis: confirmed

`nsys` on the dense pair, both arms at `-n 120` and the same 46-token prompt. The two arms launch **identical instance counts for every kernel**, so the comparison is of time and not of work:

| Kernel | Instances (both arms) | 4-bit | 8-bit | Delta per generated token |
|---|---|---|---|---|
| `qmv_kernel` (decode) | 39,151 | 12.8366 s | **5.9911 s** | **+57.05 ms/token** favoring 8-bit |
| `qmm_naive_kernel` (prefill) | 329 | **10.0733 s** | 11.9215 s | -15.40 ms/token favoring 4-bit |

The measured wall-clock slope delta is 124.41 minus 65.96, that is 58.45 ms/token. The `qmv` delta alone is 57.05 ms/token, so **`qmv` accounts for 97.6% of the entire gap**. Both arms' profiles reconcile against their unprofiled runs (102.4% and 101.6%), so those absolute times are usable.

`qmm_naive` moves the other way, and that is the control. The two kernels differ in exactly one relevant respect:

- `qmv` selects its accumulator on bit width: `cuda::std::conditional_t<(bits >= 8), float, T> sums[elems_per_thread]` at `src/lib/mlx-cpp/patches/mlx/backend/cuda/quantized/qmm/qmv.cu:191`. At `bits < 8` with a bf16 checkpoint, `T` is `bfloat16_t`, and sm_70 has no bf16 ALU, so the accumulation is emulated. At `bits >= 8` it accumulates in float, which sm_70 does have.
- `qmm_naive` accumulates in float regardless of bit width, through `UniversalFMA<float, Element, Element>` in MLX's `mlx/backend/cuda/device/gemm_sm70.cuh`, which is the non-SM80 arm of its MMA atom selection.

Where the accumulator is bit-width dependent, the 4-bit arm loses by 2.14x. Where it is not, the 4-bit arm wins as bandwidth predicts. That is the hypothesis in #1539 confirmed on a controlled pair, with the counterfactual measured in the same profile.

### The MoE pair confirms the direction but not the mechanism

The 12.6% MoE advantage is a black-box wall-clock observation and no kernel-level conclusion should be drawn from it, for two reasons.

First, the profile does not reconcile. The 4-bit arm's profiled decode time is 12.303 s against 8.542 s unprofiled, an inflation of 144.0%, while the 8-bit arm inflates only to 106.2%. Graph-node tracing is charging the two arms differently, so their absolute kernel times are not comparable to each other.

Second, restricted to the kernels that actually run during MoE decode, the deltas explain only part of the gap:

| Kernel | Instances (both arms) | 4-bit | 8-bit | Delta per generated token |
|---|---|---|---|---|
| `qmv_kernel` (attention projections) | 28,084 | 2.6224 s | 1.4921 s | +9.42 ms/token favoring 8-bit |
| `custom_kernel_moe_gateup` | 3,570 | 0.9375 s | 1.5728 s | -5.29 ms/token favoring 4-bit |
| `custom_kernel_moe_down` | 3,570 | 0.6207 s | 0.9038 s | -2.36 ms/token favoring 4-bit |
| `qmm_naive_kernel` (prefill) | 326 | 9.8733 s | 11.7773 s | not a decode kernel |

Net over the decode kernels is +1.77 ms/token favoring 8-bit against a measured slope delta of 4.32 ms/token, so 41% of the gap is accounted for.

This also **corrects a claim made while this issue was open**, that MoE decode routes through `GatherQMM` to `qmm_naive` because the expert count makes `M * B >= 8`. On this build it does not. The expert path in decode is mlxcel's own fused kernels, `custom_kernel_moe_gateup_kernel_cu_*` and `custom_kernel_moe_down_kernel_cu_*`, firing 3,570 times each, which is 30 layers times 119 steps. `qmm_naive` appears with a prefill-sized 326 instances in both arms. The `qmv` inversion is present in the MoE profile at the same 1.76x ratio the earlier audit found (2.6224 s against 1.4921 s over identical 28,084 instances), but the fused expert kernels move the other way and nearly cancel it, which is why the MoE advantage is 1.13x rather than the dense pair's 1.89x.

**What this profile does not show, and why** (added by #1544). No CUTLASS grouped GEMM appears anywhere in it, which reads as though MoE on this build never touches that path. It does, in prefill, above a prompt-length gate this profile sits below. #629's sorted-MoE prefill fast path (`patches/mlx/backend/cuda/quantized/quantized.cpp:359`) routes a quantized `GatherQMM` into `cutlass_grouped_gemm_unaligned` once `B >= min_rows * E`, with `min_rows` defaulting to 8; for this checkpoint `E = 128` and `top_k = 8`, so the gate is 128 prompt tokens and the 46-token prompt used here gives `B = 368`. Re-profiled at 573 prompt tokens, the same binary and checkpoint launch `cutlass::Kernel<GemmGrouped>` 180 times for 3.8% of GPU time, and prefill runs 2.22x faster than with the path disabled. The 326 `qmm_naive` instances above reproduce exactly at the short prompt, which is the cross-check that the two profiles are the same measurement at different prompt lengths. See `grouped-gemm-arch-v100-2026-08-31.md`.

### Practical guidance

Prefer an 8-bit checkpoint over a 4-bit one on Volta wherever it fits in 32 GB. This is a workaround for #1539 rather than a substitute for it: once `qmv` uses float accumulators below Ampere, the 4-bit arm should regain its bandwidth advantage and this guidance should be re-measured and probably reversed.

**Re-measured after #1539 landed, and it no longer holds.** With float accumulators below Ampere the 8-bit arm's advantage falls from 1.886x to 1.074x (66.73 against 71.66 ms/token) while it still reads 1.9x the bytes, so a 4-bit checkpoint now costs about 7% of decode rate and saves 47% of the weight footprint. Prefer 4-bit again unless the 7% matters more than the memory. It did not fully reverse: the residual 7.4% is the dequantization work the 4-bit arm does and the 8-bit arm does not, which the `qmm_naive` control above measures at 5.9% on its own. See `qmv-float-accum-v100-2026-08-31.md`.

## Three harnesses, and which number each produces

This repository has three ways to get a decode number and on this host they do not agree. The disagreement is not noise, it is whether the fixed cost is inside the measurement.

- **`make bench-model`** (`Makefile`, `run_bench`). One `mlxcel generate` process at `MAX_TOKENS=100`, no warmup, reporting the `[Generated N tokens in Ts = R tok/s]` line. That line divides generated tokens by `decode_time_ms`, which starts before the first decode step, so the fixed cost lands inside it, and the generation stops at EOS well before 100 tokens so the fixed cost is amortized over very few tokens.
- **`scripts/bench_decode.sh`** (`mlxcel-bench-decode`). Loads the model once, runs a discarded 20-token warmup, then measures in the same process. The warmup pays the fixed cost. This is also the script that writes the CSV shape `benchmarks/` uses, and it is the artifact producer for this record.
- **The slope**, defined above. The reference, because it is the only one that separates the fixed cost from the marginal cost rather than either including or hiding it.

| Checkpoint | `make bench-model` | `bench_decode.sh` | Slope | Harness error |
|---|---|---|---|---|
| `qwen3.8-27B-4bit` | 1.67 tok/s | 4.45 tok/s | **4.54 tok/s** | 2.7x low |
| `gemma-4-12B-it-4bit` | 3.78 tok/s | 8.59 tok/s | **8.04 tok/s** | 2.1x low |
| `gemma-4-12b-it-8bit` | 6.33 tok/s | 17.40 tok/s | **15.16 tok/s** | 2.4x low |
| `gemma-4-26b-a4b-it-4bit` | 3.28 tok/s | 26.52 tok/s | **25.90 tok/s** | 7.9x low |
| `gemma-4-26b-a4b-it-8bit` | 3.21 tok/s | 31.35 tok/s | **29.16 tok/s** | 9.1x low |

`bench_decode.sh` lands within 2% to 15% of the slope, which is why it stays the artifact producer. `make bench-model` is between 2.1x and 9.1x low, and on this run it is **not even sign stable**: it reports the 4-bit MoE arm as faster than the 8-bit one (3.28 against 3.21) while both the slope and `bench_decode.sh` put 8-bit ahead by 12% and 18%. A Volta record that quoted the Makefile harness alone would have published the opposite of this document's central finding.

Any future Volta number must therefore report the slope. The Makefile harness is fine as a smoke test that a model runs, and no better than that here.

## What reproduced, and what did not

Against the one-off audit recorded in #1536 and in the comments on #1538. The 10% band is the criterion this issue set for a second person on a different V100.

| Quantity | Audit | This record | Verdict |
|---|---|---|---|
| `qwen3.8-27B-4bit` decode | 239 ms/tok (4.2 tok/s) | 220.33 ms/tok (4.54 tok/s) | reproduced, 7.8% faster here |
| `qwen3.8-27B-4bit` prefill | ~7.7 tok/s | 8.00 tok/s marginal | reproduced, +3.9% |
| `qwen3.8-27B-4bit` roofline attainment | ~70 GB/s, 7.7% | 65.4 GB/s, 7.27% | reproduced |
| `qmv` instances, qwen 24-token run | 11,431 | 11,431 | exact |
| `cudaMemcpyAsync` calls, same run | 1,847 | 1,847 | exact |
| Dense 4-bit decode | 122.04 ms/tok | 124.41 ms/tok | reproduced, +1.9% |
| Dense 8-bit decode | 65.46 ms/tok | 65.96 ms/tok | reproduced, +0.8% |
| Dense 8-bit advantage | 1.86x | 1.886x | reproduced |
| Dense `qmv` time, 39,151 instances | 12.84 s / 5.99 s | 12.8366 s / 5.9911 s | exact |
| Dense `qmv` attribution of the slope gap | ~101% | 97.6% | reproduced |
| MoE `qmv` time, 28,084 instances | 2.62 s / 1.49 s | 2.6224 s / 1.4921 s | exact |
| MoE 4-bit decode | 37.63 ms/tok | 38.61 ms/tok | reproduced, +2.6% |
| MoE 8-bit decode | 31.38 ms/tok | 34.29 ms/tok | reproduced, +9.3%, at the edge of the band |
| MoE 8-bit advantage | 19.9% | 12.6% | **direction reproduced, magnitude did not** |
| `make bench-model`, qwen | 1.70 tok/s | 1.67 tok/s | reproduced |
| `make bench-model`, MoE pair | 3.31 / 3.60 tok/s | 3.28 / 3.21 tok/s | 4-bit reproduced; the 8-bit arm did not, and the sign flipped |
| TTFT, qwen | ~13 s | 24.94 s at a 54-token prompt, warm cache, after load | reproduced once both are reduced to their prompt-independent term (12.60 s against 12.66 s, 0.5%) |
| TTFT, qwen, confirming test | ~13 s predicted at a 3-token prompt | 15.54 s at a 2-token prompt (#1545, post-#1539 build) | **direction confirmed, magnitude not.** A short prompt lands nowhere near 25 s, which is what the reconciliation predicted; it lands 20% above 13 s. The 0.5% agreement below is tighter than the method supports, because prefill is not affine at the short end |
| `qmm_naive` instances, qwen 24-token run | 994 | 497 | **did not reproduce**, exactly half |
| `qmm_naive` instances, dense pair | 658 | 329 | **did not reproduce**, exactly half |
| `cuModuleLoadDataEx`, qwen | 848 ms / 13 calls | 172 ms / 13 calls | calls exact, time 4.9x lower |
| `cudaGraphInstantiate`, qwen | 2.14 s / 208 calls | 1.2249 s / 196 calls | close, not exact |
| `cudaGraphAddKernelNode`, qwen | 1.95 s / 72,539 calls | 1.8901 s / 64,717 calls | close, not exact |
| MoE nsys reconciliation | fails | fails, 144.0% on the 4-bit arm | reproduced |
| Build provenance | 96 cubins, all sm_70 | 96 cubins, all sm_70 | exact |

Four disagreements are worth naming rather than burying.

**TTFT reconciles once each figure is reduced to its prompt-independent term.** The two numbers were not measuring the same thing, which is why the direct comparison looked irreconcilable. Both are intercepts of a two-point slope fit, but each intercept carries the prefill cost of its own prompt: the audit fitted at a 3-token prompt and this record fits at a 46-token prompt. Subtracting each one's prefill at the 8.00 tok/s marginal rate measured here leaves 12.98 - 3/8.00 = 12.60 s for the audit against 18.41 - 46/8.00 = 12.66 s for this record, which agree to 0.5%. The residual after that subtraction is the genuinely prompt-independent fixed cost, and it is what #1545 is about.

This is an arithmetic reconciliation, not a controlled experiment, and it is recorded as such. The confirming test is cheap and has not been run: measure TTFT on `qwen3.8-27B-4bit` at a 3-token prompt with a warm cache and the model already loaded, and check that it lands near 13 s rather than near 25 s. Until someone runs it, treat the agreement as strong evidence that prompt length explains the gap rather than as proof. What the reconciliation does settle is that neither figure is wrong; quoting either one without its prompt length is what was wrong, which is the reason this record states prompt length everywhere.

**`qmm_naive` instance counts are exactly half the audit's, on both models.** 497 against 994 and 329 against 658. A factor of exactly two on two independent models is a counting difference, not a performance difference; the most likely cause is that the audit's report counted a graph node and its kernel separately. The per-instance times are consistent with that reading, and the conclusion drawn from `qmm_naive` here is a sign comparison within one profile, which is unaffected.

**The MoE 8-bit advantage is 12.6% here against 19.9% in the audit.** The direction is solid across five repetitions per arm, but that arm's repeat spread is 13.5%, wide enough that a three-repetition estimate can land anywhere in this range. The MoE numbers should be treated as ordinal, not as a precise ratio.

**`make bench-model` on the MoE pair reports the opposite ordering to the audit and to reality.** Covered in the harness section. It is an argument against the harness, not evidence about the models.

## Build coverage: what changed, and the GPU-runner decision

Before this record, nothing in the repository compiled MLX for anything below sm_80. `ci.yml` pinned `MLX_CUDA_ARCHITECTURES: "121"` in both CUDA jobs, `release.yml` ships `90a;100;121` on aarch64 and `80;86;89;90a;100;120` on x86_64, and `build.rs` auto-detects the host with a `90a` last resort. Volta worked only by accident of whoever last built on one, and an arch-conditional compile break could sit on `main` indefinitely.

**CUDA 13 cannot build for Volta at all, which bounds what any gate can do.** The first run of the job below failed in 54 seconds with `nvcc fatal : Unsupported gpu architecture 'compute_70'`. The CUDA runners carry CUDA 13.0.88, and CUDA 13 removed Volta support, so nvcc rejects `compute_70` before compiling anything. This is not a defect in the job; it is a toolkit limitation, and it has three consequences worth stating plainly. Building mlxcel for Volta requires a CUDA 12.x toolchain, which is what this host has (12.9.41). sm_70 can never enter the release matrix while releases are built on CUDA 13, which is a harder reason than the cost argument given below. And no gate can run on the current fleet, so the job probes `nvcc --list-gpu-arch` and skips with an explicit step summary rather than red-lighting every PR that touches these paths; it starts gating for real, with no edit, the moment a CUDA 12.x runner joins.

**Added: a compile-only sm_70 gate.** `cuda-sm70-compile` in `.github/workflows/ci.yml` runs `cargo check --features cuda --all-targets` with `MLX_CUDA_ARCHITECTURES: "70"` on the existing self-hosted CUDA runner, then asserts with `cuobjdump --list-elf` that the resulting `libmlx.a` contains sm_70 and nothing else. `cargo check` still runs the build script, so nvcc compiles MLX even though nothing links, and the cubin assertion is what turns "the environment variable was set" into "nvcc actually emitted pre-Ampere code". It is path-filtered on a new `cuda_arch` filter covering `src/lib/mlx-cpp/**`, the two build scripts and the workflow itself, so a Rust-only PR does not pay a CUDA rebuild, and it takes its own persistent target directory because the architecture list is part of the build-script fingerprint and sharing with an sm_121 job would make the two invalidate each other on every run.

**Not added: a GPU-backed Volta job. This is a decision, not an omission.** The realistic failure mode this program can introduce is a compile-time one. The remaining items in #1536 add `cc < 8` branches to the quantized-matmul overlays, and a `__CUDA_ARCH__` guard that does not cover cc 7, a CUTLASS type with no pre-Ampere instantiation, or a bf16 intrinsic with no sm_70 implementation all fail at nvcc, which the compile gate catches without a Volta card. Against that:

- There is no Volta runner in this repository's pool, and adding one means adding and maintaining a machine.
- No release artifact targets sm_70, so nothing published to a user is exercised by a Volta runtime job.
- The runtime symptom a Volta job would catch is already named rather than opaque. #1537 landed `cuda_arch_mismatch()` and `enforce_cuda_arch_compatibility()`, so a binary whose compiled architecture list does not cover the running device refuses to start with a message naming both, instead of failing at the first kernel launch with a CUDA load error that names neither.
- The one Volta machine that exists is this development host. It has a single GPU, and a CI job on it would serialize against exactly the measurements this document exists to reproduce.

The cost of that decision is explicit: a change that compiles for sm_70 but computes the wrong answer on it, or regresses its throughput, is caught by nothing automatic. What covers it instead is this document plus a re-run of the reproduce commands below, which is the arrangement #1539 through #1545 each verify against.

**Not added: sm_70 in the release matrix.** A seventh architecture in the x86_64 fat binary was considered and rejected. A cold six-architecture build is already recorded at about three hours in `release.yml`, the archive grows with each architecture, and shipping sm_70 would be a support commitment for a part that this record shows the current kernels serve badly. Revisit once #1539 through #1545 have moved these numbers.

## Known traps on this host

**`make release-cuda` on a machine without `nvidia-smi` silently targets Hopper.** `resolve_cuda_architectures()` in `src/lib/mlxcel-core/build.rs` honors an explicit `MLX_CUDA_ARCHITECTURES` verbatim, then tries `nvidia-smi`, then falls back to `90a`. In a container that cannot see the driver, that fallback produces a binary which cannot run on the machine that built it. This is covered as of #1537: the runtime compares the device against `MLXCEL_CUDA_ARCHITECTURES`, the list recorded at build time, and refuses to start with a message naming both, which `MLXCEL_DEVICE=cpu` bypasses. `MLXCEL_TRACE_ARCH` prints the capability, the compiled list and the coverage verdict once per process. Pass the architecture explicitly on this host regardless; do not rely on detection.

**The GPU is exclusive.** One V100 with 32 GB, and the 8-bit MoE checkpoint alone occupies 26 GB. Two GPU processes at once do not merely contend, they change the numbers: an accidental overlap during this work reported `gemma-4-12B-it-4bit` at 264 ms/token against its true 124, and 18 s of prefill for a 46-token prompt. Check `nvidia-smi --query-compute-apps=pid --format=csv,noheader` is empty before every run; the harness used here aborts if it is not.

**Do not read a percentage share across two nsys runs.** Rule 6. The `qmm_naive` share for `qwen3.8-27B-4bit` is 70.4% here and 82.0% in the audit while the underlying per-instance times agree, purely because the totals differ.

**A short prompt invalidates a slope.** Rule 2. Every instruct checkpoint here stops well before 100 tokens on a greeting, which is also why `make bench-model` measures what it measures.

## Post-program comparison

To be filled as epic #1536's remaining items land. Each row is a re-run of the matching reproduce command below on this host, at the same warm-cache state.

| Issue | What it changes | Baseline (this document) | After | Delta |
|---|---|---|---|---|
| #1539 | `qmv` float accumulators below Ampere at `bits < 8` | dense 4-bit 124.41 ms/tok; `qmv` 12.8366 s at 39,151 inst | dense 4-bit 71.66 ms/tok; `qmv` 6.5654 s at 39,151 inst | **1.73x** end to end, **1.96x** on `qmv`, which accounts for 99.4% of it. `qwen3.8-27B-4bit` 220.33 -> 117.83 ms/tok (1.87x), its `qmv` roofline attainment 7.27% -> 14.80%. The 8-bit control is unmoved at 66.73 ms/tok. Full record: `qmv-float-accum-v100-2026-08-31.md` |
| #1541 | `qmm_naive` tile sized from the device shared-memory budget | dense prefill `qmm_naive` 10.0733 s at 329 inst | dense prefill `qmm_naive` 10.867 s at 329 inst | **No change, by measurement.** The 128-wide tile the issue set out to unlock fits Volta's budget four times over, and loses anyway: 255 registers against 224 and a 128-byte spill, no occupancy gain (2 blocks/SM either way), and a halved CTA count on a grid already smaller than 80 SMs. `qmm_naive` measures 1.50x slower at a 106-token prompt, 1.06x at 516 and 1.02x at 4,106, so the width stays where upstream had it and the shared-memory decision becomes a real device query. The 7.9% gap between the two columns is between sessions, not between tile widths: both widths agree to 0.04% in every profile and `qmv` in the same runs reproduces #1539 to 0.04%. Full record: `qmm-naive-tile-v100-2026-08-31.md` |
| #1542 | f16 activation policy below Ampere | dense 4-bit 124.41 ms/tok; qwen 220.33 ms/tok | | |
| #1543 | `qmm_sm70`, Volta tensor-core MMA for quantized GEMM | qwen prefill 8.00 tok/s marginal, 2.83% of FP32 peak | | |
| #1544 | grouped GEMM arch tag on an sm_70 part | re-measured on this branch: MoE 4-bit 29.66 ms/tok, 8-bit 31.68 ms/tok; 4-bit prefill 12,720 ms at a 285-token prompt | MoE 4-bit 29.73 ms/tok, 8-bit 32.71 ms/tok; 4-bit prefill 12,644 ms | **No change, and byte-identical device code is why.** The pre-Ampere arm was tagged `cutlass::arch::Sm75` on a part that has no `m16n8k8` MMA. The branch is live here, not dead: #629's sorted-MoE prefill path routes a quantized `GatherQMM` into `cutlass_grouped_gemm_unaligned` once `B >= 8 * num_experts`, which for this checkpoint is a 128-token prompt, and an nsys profile at 573 prompt tokens shows 180 grouped-GEMM launches at 3.8% of GPU time. The kernel it selects is `MmaSimt` / `OpMultiplyAdd`, so the tag never reached an MMA atom and the output was never wrong; the grouped path and the legacy `qmm_naive` path give a byte-identical 64-token greedy continuation. Retagging to `Sm70` leaves the same 51 device symbols with byte-identical SASS at `compute_70`, `compute_80` and `compute_121`, so both columns above are the same machine code and every delta sits inside the before arm's own repetition spread. **The baseline column is re-measured rather than quoted**: this document's own 38.61 and 34.29 predate #1539. Full record: `grouped-gemm-arch-v100-2026-08-31.md` |
| #1545 | CUDA graph instantiation and JIT module load in TTFT | qwen fixed prefill cost 18.41 s; graph APIs 3.3 s of a 24-token run | re-measured on this branch: 2-token first token 15.54 s, of which 14.99 s is one-time; graph APIs 0.97 s at `-n 1` | **Findings only, and the issue's premise did not survive.** The fixed cost is 96.5% one-time process cost, and 12.08 s of it (77.8% of the first token) is materializing the language model's 15.13 GB of weights, which MLX loads lazily so the whole read and host-to-device copy land inside prefill: `Model loaded in 0.98s` is followed by `resident: 0.00 GB`. Graph instantiation is 0.10 s and saturates at 196 distinct graphs against a 2,000-entry cache, and `MLX_USE_CUDA_GRAPHS=0` changes the first token by less than the 6.5% repeat spread. The warm JIT cache holds sm_70 cubins rather than PTX and costs 0.06 s; cold it costs 12.0 s for 6 modules. Verdict: general, not Volta-specific, so the follow-up belongs outside #1536. Decode re-measured at 118.17 ms/token with `C` zero, so none of the fixed cost is in the decode loop. Full record: `volta-ttft-fixed-cost-2026-09-01.md` |

## Reproduce

Assumes a Volta-class part, this repository at the commit in the environment table, and `./models/mlx-community/` holding the five checkpoints. `MODELS_DIR` has to be pointed at the nested store root because the checkpoints sit under `owner/name`.

```bash
# 1. Build. Single architecture, explicit. Never rely on auto-detection here.
MLX_CUDA_ARCHITECTURES=70 make release-cuda

# 2. Build provenance: every cubin in the MLX archive must be sm_70.
cuobjdump --list-elf target/release/build/mlxcel-core-*/out/build/lib/libmlx.a \
  | grep -oE 'sm_[0-9]+a?' | sort | uniq -c

# 3. Runtime provenance, and which quantized-matmul path the dispatcher picks.
MLXCEL_TRACE_ARCH=1 ./target/release/mlxcel generate \
  -m ./models/mlx-community/gemma-4-12B-it-4bit -p "Hi." -n 8

# 4. Decode slope. Two -n values, same prompt, three repetitions, and a prompt
#    that runs past 120 tokens. Read the "Decode:" line of each --profile block,
#    never the tok/s figure, and confirm "Generated tokens" equals -n.
P="Write a detailed technical explanation of how virtual memory works in a modern operating system. Cover page tables, the TLB, page faults, and swapping. Be thorough."
for rep in 1 2 3; do for n in 40 120; do
  ./target/release/mlxcel generate -m ./models/mlx-community/qwen3.8-27B-4bit -p "$P" -n $n --profile
done; done
# slope = (decode_ms(120) - decode_ms(40)) / 80

# 5. Prefill ladder at -n 1. Build each prompt by repeating one fixed sentence to
#    the target length, record the reported "Prompt tokens", and fit
#    prefill_s = intercept + slope * prompt_tokens by least squares over the rungs.
#    The rungs used here came out at 54, 161, 365, 671 and 1283 tokens.
./target/release/mlxcel generate -m ./models/mlx-community/qwen3.8-27B-4bit -p "Hi." -n 1 --profile
./target/release/mlxcel generate -m ./models/mlx-community/qwen3.8-27B-4bit -p "$LONG_PROMPT" -n 1 --profile

# 6. Kernel profile. --cuda-graph-trace=node is mandatory.
nsys profile -t cuda,nvtx --cuda-graph-trace=node -o volta_base \
  ./target/release/mlxcel generate -m ./models/mlx-community/qwen3.8-27B-4bit \
  -p "Explain what a GPU tensor core does." -n 24
nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum --format csv volta_base.nsys-rep
# Then reconcile: profiled decode_ms against the same run unprofiled.

# 7. The CSV artifact, in the shape benchmarks/ already uses.
MODELS_DIR=./models/mlx-community ./scripts/bench_decode.sh all \
  --output benchmarks/cuda_v100_2026-08-31.csv

# 8. The Makefile harness, for the harness comparison only. Leaves the repo clean.
make bench-model MODEL=models/mlx-community/qwen3.8-27B-4bit MODELS_DIR=./models/mlx-community
make bench-clean

# 9. Cold PTX cache, if a first-run number is wanted. Move it aside, do not delete it.
mv ~/.cache/mlxcel/cuda-ptx/9a795735ad9a{,.saved}
# ... one run ...
rm -rf ~/.cache/mlxcel/cuda-ptx/9a795735ad9a && mv ~/.cache/mlxcel/cuda-ptx/9a795735ad9a{.saved,}
```

## References

- Epic #1536, the Volta decode program this baseline is Phase 0 of. Its `## GB10 (sm_121) continuation` section holds what could not be verified on this machine.
- Precedent for a Phase 0 measurement issue: #624, for the GB10 program #623.
- Compute-capability probe, the recorded architecture list, and the mismatch refusal: #1537.
- `qmv` accumulator selection: `src/lib/mlx-cpp/patches/mlx/backend/cuda/quantized/qmm/qmv.cu:191`, and issue #1539.
- `qmm_naive` accumulator: `UniversalFMA<float, Element, Element>` in MLX's `mlx/backend/cuda/device/gemm_sm70.cuh`.
- `qmv` against `qmm_naive` dispatch: the `M * B < 8` test in `src/lib/mlx-cpp/patches/mlx/backend/cuda/quantized/quantized.cpp:291`.
- Arch auto-detect and the `90a` fallback: `resolve_cuda_architectures()` and `detect_cuda_arch()` in `src/lib/mlxcel-core/build.rs`.
- Persistent PTX cache: `ensure_persistent_ptx_cache()` in `src/lib/mlxcel-core/src/lib.rs`.
- CI and release architecture pins: `.github/workflows/ci.yml`, `.github/workflows/release.yml`.
