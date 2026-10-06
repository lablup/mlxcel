# MoE prefill and the expert-batched `gather_qmm` kernel on gfx1151 (2026-10-05)

lablup/mlxcel#2066, part of #1814. The baseline ([rocm-baseline-gfx1151-2026-09-30.md](rocm-baseline-gfx1151-2026-09-30.md)) put MoE prefill on this host at about 26 tok/s for Mixtral-8x7B-4bit and under 8 tok/s for gpt-oss-20b. The overlay's expert-batched kernel, which reads each expert's weights once for all of that expert's rows, had been opt-in since `patches-rocm/LOCAL_FIXES.md` item 9 found it wrong for bf16. This page has the prefill profile, the root cause of item 9, the prefill numbers before and after, and the decision to turn the kernel on by default.

## Environment

AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`, RDNA 3.5), Debian 13, HIP 7.15.26333, AMD clang 23.0.0git, rocprofv3 1.3.5, Rust 1.97.1. mlxcel `main` at `c05d5438` plus this change, built with `cargo build --release --features rocm`; MLX pin `81ba1c6a`, ROCm overlay `75915908`. Checkpoints: the local `mlx-community` directories under `models/mlx/`.

Every measurement ran under `scripts/rocm_gpu_guard.sh --idle-secs 45` (45 s with no other GPU process and no compiler before the run, a 1 Hz monitor during it, rerun on contention); other units were building and benchmarking on the same host throughout, and the guard held each run until the GPU was idle.

## Profile

`rocprofv3 --kernel-trace --stats -f csv` around `mlxcel-bench-decode --prompt-tokens 512 --max-tokens 8 --ignore-eos` with `MLXCEL_BENCH_PHASE_MARKS=1`; the measured prefill is cut out of the trace between the bench's `measured_start` and `decode_start` marks. Shares are of the kernel time in that window.

| Model | Activations, experts, top-k | Rows per call (B) | Prefill, profiled | Dominant kernel | Share |
|---|---|---|---|---|---|
| Mixtral-8x7B-Instruct-v0.1-4bit | f16, 8, 2 | 1024 | 20.3 s | `gather_qmv_warp_shared_kernel<__half, __half, 4, 64, true, 16>` (210 ms per call) | 99.1% |
| gpt-oss-20b-MXFP4-Q4 | bf16, 32, 4 (mxfp4) | 2048 | 67.8 s | `gather_qmv_kernel<hip_bfloat16, unsigned char, 4, 32, false>` (940 ms per call) | 99.8% |
| granite-4.0-h-tiny-4bit | bf16, 64, 6 | 3072 | 0.96 s | `gather_qmv_wide_kernel<hip_bfloat16, hip_bfloat16, 64>` (4.9 ms per call) | 61.8% |

All three are per-row gather kernels that reread an expert's weights for every routed row. granite is the one checkpoint here the expert-batched gate (bf16, affine, group size 64, 4 or 8 bits, at most 64 experts, `B / E >= 4`) reached; Mixtral is f16 and gpt-oss is mxfp4, so neither could. Mixtral's kernel is the generic path the expert-batched kernel replaces, which is why f16 was added (issue step 3). gpt-oss's is the non-affine fallback from item 10, which no affine kernel can take; it is filed as #2106.

## Root cause of item 9

The kernel read `lhs_indices[b]` and `rhs_indices[b]` as flat `[B]` arrays. MLX broadcasts the two index arrays against the activation's batch shape without copying them, so an index array can have stride 0 along a broadcast axis. `x` as `[T, 1, K]` with sorted `rhs_indices` of shape `[T, 1]` (the call that found item 9) gives a `[T, T]` batch with rhs strides `(1, 0)` and implicit lhs strides `(0, 1)`, and the kernel read both arrays past their end. `SwitchGLU`'s sorted path passes flat `[B]` indices, so models never hit it: forcing the old kernel on for granite's `w256` trace gave 0 decided mismatches against Metal. The bf16-only symptom came from the gate, which sent only bf16 to the kernel.

`tests/rocm_gather_qmm_expert_batched.rs` reproduces it: with the old kernel the `BroadcastRows` case (`x` `[T, 1, 1, K]`, `rhs` `[T, top_k]`) had a relative L2 error of 1.415 against the dequantized f32 reference, where the unsorted path had 2.3e-3. The kernel now reads both arrays through the batch shape and strides, as the launched per-row gather kernels in `qmm.hip` do.

## The kernel was slower than the path it replaces

With only the index fix, the kernel took 1.33 s of granite's prefill against 0.58 s for the wide kernel it replaces (11.1 against 4.9 ms per call), so the inner loop was rewritten: each lane loads one packed word with its group's scale and bias and applies them to four rows before the next load, the lanes reduce once per row instead of once per group, all 16 lanes of a column work (only 8 did at 4 bits), and block z is the expert itself, found with two binary searches, instead of the z-th run found by walking every earlier run. In the same profile the rewritten kernel takes 0.32 s of granite's prefill (2.7 ms per call) and 4.52 s of Mixtral's (47 ms per call, against 210 ms for the warp-shared kernel). One `Exp` dispatch in the granite profile took 316 ms, an outlier the plain runs below do not show.

## Correctness

`tests/rocm_gather_qmm_expert_batched.rs` forces the kernel on and compares sorted `gather_qmm` with the unsorted path and with per-expert dense f32 matmuls of the dequantized weights: bf16 and f16, 4 and 8 bits, a Mixtral-like shape (8 experts, top 2, K 4096, N 516 so the last column block is partial) and granite's gate and down projections (64 experts, top 6), each with flat, shared-activation and both broadcast index layouts, 48 cases. In every case the expert-batched error is within 0.1% of the unsorted path's (2.8e-4 to 2.9e-4 relative L2 in f16, 2.2e-3 to 2.3e-3 in bf16; the largest ratio is 1.0007), and two sorted runs are bit-identical.

Teacher-forced logit traces with the kernel on (`examples/logit_trace`, `w8` = `8 80 8 512`, `w256` = `256 2 8 0`), compared by `scripts/compare_logit_traces.py --decided 2.0` against the Metal references the #1809 matrix uses:

| Model, window | Metal reference | Top-1 disagreement | Decided mismatches | Perplexity ref / ROCm |
|---|---|---|---|---|
| qwen3-30b-a3b, w8 | `metal_m1u_bec64748` | 17 / 640 | 0 / 319 | 10.921 / 10.907 |
| qwen3-30b-a3b, w256 | `metal_m1u_bec64748` | 23 / 512 | 0 / 248 | 13.301 / 13.237 |
| mixtral-8x7b-instruct, w8 | `metal_m1u_bec64748` | 1 / 640 | 0 / 345 | 6.383 / 6.379 |
| mixtral-8x7b-instruct, w256 | `metal_m1u_bec64748` | 2 / 512 | 0 / 245 | 7.722 / 7.726 |
| granite-4.0-h-tiny, w8 | `metal_m5_d1128266` | 20 / 640 | 0 / 260 | 14.287 / 14.378 |
| granite-4.0-h-tiny, w256 | `metal_m5_d1128266` | 23 / 512 | 0 / 208 | 18.999 / 19.107 |

Qwen3-30B-A3B has 128 experts, so the gate never sends it to this kernel; its rows confirm nothing else moved. granite against the earlier ROCm traces (`rocm_gfx1151_c5fe9a16`, per-row kernel) also has 0 decided mismatches at both widths. The traces are not committed.

## Prefill and decode

`mlxcel-bench-decode` at `scripts/bench_decode.sh`'s shape (512-token prompt, 128 generated tokens, 20-token warmup, `--ignore-eos`), one binary, with `MLX_ROCM_GATHER_QMV_EXPERT_BATCHED=0` (the per-row kernels, `main`'s behaviour) and `=1` (the default after this change) alternated run by run, three rounds.

| Model | Prefill tok/s, before | Prefill tok/s, after | Change (medians) | Decode tok/s, before / after (medians) |
|---|---|---|---|---|
| granite-4.0-h-tiny-4bit | 560.10 / 535.73 / 487.27, median 535.73 | 976.77 / 918.59 / 875.13, median 918.59 | 1.71x | 88.76 / 88.65 |
| Mixtral-8x7B-Instruct-v0.1-4bit | 25.91 / 25.95 / 26.02, median 25.95 | 125.63 / 125.56 / 125.63, median 125.63 | 4.84x | 9.40 / 10.43 |

Every after run is faster than every before run on both models. Decode does not reach the kernel (a decode step has `B = top_k`, below the gate's 64): granite's decode medians are 0.1% apart, and Mixtral's spread (8.68 to 10.71 tok/s across all six runs) is the noise of a 26 GB checkpoint on a 31 GiB host that the fused-MoE page also recorded.

### Short prompts, near the gate

Prefill time in ms for short prompts (`--prompt-tokens T --max-tokens 4`, kernel off/on alternated, guarded), to check the gate's lower edge (`B >= 64`, `B / E >= 4`):

| Model | T (B) | Off | On |
|---|---|---|---|
| granite-4.0-h-tiny-4bit | 44 (264, just above `B / E >= 4`) | 102.4 / 102.9 / 121.2 | 82.7 / 84.2 / 84.9 |
| granite-4.0-h-tiny-4bit | 64 (384) | 126.6 / 127.1 / 129.7 | 95.9 / 96.7 / 97.5 |
| granite-4.0-h-tiny-4bit | 160 (960) | 236.7 / 250.6 / 296.5 | 148.8 / 162.5 / 170.1 |
| Mixtral-8x7B-Instruct-v0.1-4bit | 32 (64, the `B >= 64` floor) | 1709.7 / 1716.2 | 616.9 / 669.3 |
| Mixtral-8x7B-Instruct-v0.1-4bit | 64 (128) | 2845.0 / 2888.2 | 836.5 / 855.2 |

The kernel is faster at every size, including at both limits of the gate.

## Decision

The issue's rule: enable by default only if every eligible dtype matches the unsorted path within that path's own error against a dequantized f32 reference, and 512-token prefill improves on every eligible model measured with no decode change. Both hold (bf16 and f16 are the eligible dtypes; granite and Mixtral the eligible models on this host), so the kernel is on by default. `MLX_ROCM_GATHER_QMV_EXPERT_BATCHED=0` turns it off.

Build cost: the f16 arm adds two instantiations (4 and 8 bits). `qmm.hip` alone, compiled with the build's own `hipcc` command and alternated with `main`'s file, took 37.5 and 37.7 s against 38.4 and 38.0 s on `main`: no measurable change, since the rewritten kernel is simpler than the one it replaces.

## Not measured

gpt-oss-20b is unchanged here (mxfp4); #2106 gave the kernel an mxfp4 arm, measured in [rocm-moe-prefill-mxfp4-gfx1151-2026-10-06.md](rocm-moe-prefill-mxfp4-gfx1151-2026-10-06.md). Models with more than 64 experts (Qwen3-30B-A3B, Nemotron-3-Nano) do not reach the kernel. Wave64 (CDNA) parts are untested; the kernel's 16-lane reduction stays inside a wave on both widths. The other MoE families the gate reaches with at most 64 experts were not measured (no such checkpoint on this host). The four-rows-per-load factor was not tuned.
