# mxfp4 MoE prefill through the expert-batched `gather_qmm` kernel on gfx1151 (2026-10-06)

lablup/mlxcel#2106, part of #1814. The #2066 page ([rocm-moe-prefill-gfx1151-2026-10-05.md](rocm-moe-prefill-gfx1151-2026-10-05.md)) profiled gpt-oss-20b's 512-token prefill at 67.8 s, 99.8% of it in the per-row `gather_qmv_kernel<hip_bfloat16, unsigned char, 4, 32, false>` (about 940 ms per call), because the expert-batched kernel took affine weights only. This page has the correctness checks, the prefill and decode numbers before and after giving that kernel an mxfp4 arm, and the default decision.

## Environment

AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`, RDNA 3.5), Debian 13, HIP 7.15.26333, AMD clang 23.0.0git, rocprofv3 1.3.5, Rust 1.97.1. mlxcel `main` at `1e561f1e` plus this change, built with `cargo build --release --features rocm`; MLX pin `81ba1c6a`, ROCm overlay `75915908`. Checkpoint: `models/mlx/gpt-oss-20b-MXFP4-Q4` (bf16 activations, 24 layers, 32 experts, top 4, hidden and expert widths 2880, mxfp4 experts at group size 32).

Every benchmark run went through `scripts/rocm_gpu_guard.sh` (rounds 1 to 3 with `--idle-secs 50`, the rest with the host-wide lock from #2161 and the default idle window); every attempt was accepted on its first try, so no run overlapped another GPU process or a compiler.

## What changed

`gather_qmv_expert_batched_kernel` in `patches-rocm/mlx/backend/rocm/quantized/qmm.hip` gains an `AFFINE=false` arm for mxfp4: a lane loads one packed word (eight e2m1 nibbles, all in one group of 32), decodes each nibble without a branch by placing its sign, exponent and mantissa bits in an fp16 (the value times 2^-14), applies the word to four rows of the expert's run, and scales each row's partial dot product back by 2^14 with the group's E8M0 scale. `GatherQMM::eval_gpu` sends mxfp4 with bf16 activations to it under the existing gate (sorted, transposed, `M == 1`, `B >= 64`, `E <= 64`, `B / E >= 4`). That is one new instantiation; f16 mxfp4 and mxfp8 keep the per-row kernel. `qmm.hip` alone, compiled with the build's `hipcc` flags and alternated with `main`'s file, took 37.9 and 37.8 s against 38.1 and 37.9 s.

A decode step has `B = top_k = 4`, below the gate, so decode does not reach the kernel. Nor does a prefill shorter than 32 tokens (`B / E < 4`).

## Correctness

`tests/rocm_gather_qmm_expert_batched.rs` (`mxfp4_expert_batched_gather_qmm_matches_unsorted_and_reference`) forces the kernel on and compares sorted mxfp4 `gather_qmm` with the unsorted path (the per-row kernel) and with per-expert dense f32 matmuls of the dequantized weights, at gpt-oss-20b's expert shape (32 experts, top 4, `K = N = 2880`) and at `K = 512`, `N = 516` (a partial last column block), each with flat (`Gathered`, `Shared`) and broadcast (`BroadcastRows`, `BroadcastIndices`) index layouts, bf16 activations, 256 rows per call:

| Shape | Gathered | Shared | BroadcastRows | BroadcastIndices |
|---|---|---|---|---|
| gpt-oss-20b experts, expert-batched / unsorted | 1.654e-3 / 1.654e-3 | 1.654e-3 / 1.654e-3 | 1.657e-3 / 1.657e-3 | 1.657e-3 / 1.657e-3 |
| `K = 512`, `N = 516`, expert-batched / unsorted | 1.653e-3 / 1.653e-3 | 1.617e-3 / 1.617e-3 | 1.673e-3 / 1.673e-3 | 1.660e-3 / 1.660e-3 |

Relative L2 error against the reference; the two paths agree to four digits in every case, and two sorted runs are bit-identical. With the sign bit dropped from the kernel's nibble decode the first case fails at 1.377 (the unsorted path stays at 1.654e-3), so the test reaches the new arm. The test also checks dispatch on its own: at `K = 2880` the sorted call with the kernel switched off (the per-row kernel) must differ in at least one byte from the sorted call with it on, which fails if the gate stops sending mxfp4 to the kernel. At `K = 512` the two round to identical bf16 outputs, so that shape skips the check. The 48 affine cases of the same file are unchanged.

Teacher-forced logit traces (`examples/logit_trace` over `tests/fixtures/wikitext2_excerpt.txt`, `w8` = `8 80 8 512`, `w256` = `256 2 8 0`). No gpt-oss trace exists in `benchmarks/logit_traces/` on either backend, so the reference is the per-row path on the same binary (`MLX_ROCM_GATHER_QMV_EXPERT_BATCHED=0`) and the candidate is the default, compared by `scripts/compare_logit_traces.py --decided 2.0`:

| Window | Top-1 disagreement | Decided mismatches | Largest gap at a disagreement | Perplexity off / on |
|---|---|---|---|---|
| `w8` (512-token prefill reaches the kernel; the 8-token chunks do not) | 29 / 640 | 0 / 88 | 0.438 | 161.96 / 159.95 |
| `w256` (two 256-token chunks, `B = 1024`) | 29 / 512 | 0 / 81 | 1.062 | 76.60 / 76.88 |

The script's verdict for both: the arms differ only where the reference was undecided (rounding, not behaviour). gpt-oss has few decided positions on this corpus (40% and 44% of positions in the two windows have a top-two gap under 0.5), which is where all but one disagreement sits; every disagreement lands at rank 2 to 8 of the candidate. The traces are not committed.

## Prefill and decode

`mlxcel-bench-decode` at `scripts/bench_decode.sh`'s shape (512-token prompt, 128 generated tokens, 20-token warmup, `--ignore-eos`), one binary, `MLX_ROCM_GATHER_QMV_EXPERT_BATCHED=0` (the per-row kernel, `main`'s dispatch for mxfp4) and `=1` (the default after this change) alternated run by run, six rounds.

| Round | Prefill tok/s, off | Prefill tok/s, on | Decode tok/s, off | Decode tok/s, on |
|---|---|---|---|---|
| 1 | 7.41 | 525.45 | 8.65 | 8.52 |
| 2 | 7.66 | 529.38 | 8.63 | 8.52 |
| 3 | 7.51 | 524.78 | 8.48 | 8.44 |
| 4 | 7.92 | 526.13 | 8.42 | 8.40 |
| 5 | 7.55 | 526.43 | 8.39 | 8.45 |
| 6 | 7.42 | 528.08 | 8.34 | 8.64 |
| Median | 7.53 | 526.28 | 8.45 | 8.49 |

Prefill: 69.9x (the first three rounds alone give 7.51 against 525.45, the same ratio). Every on run is faster than every off run by about two orders of magnitude. A prefill takes 0.97 s instead of 65 to 69 s.

Decode: the medians are 0.4% apart and the two ranges (8.34 to 8.65 off, 8.40 to 8.64 on) overlap. After three rounds the off median led by 1.3%, which is why three more were run. As a control with identical code on both arms, a 16-token prompt (`B = 64`, `B / E = 2`, so neither arm reaches the kernel) alternated three times gave 8.46 / 8.50 / 8.47 off and 8.55 / 8.47 / 8.48 on: about 1% of spread from the host alone. Decode is unchanged.

## After the change

`rocprofv3 --kernel-trace --stats` of the same bench with the kernel on (`--prompt-tokens 512 -n 8 --warmup-tokens 1`, two prefills and seven decode steps, guarded): the expert-batched kernel takes 12.2 ms per call (144 calls, 62% of kernel time), against about 940 ms for the per-row kernel before. The per-row `gather_qmv_kernel<hip_bfloat16, unsigned char, 4, 32, false>` is now the decode cost: 1.39 ms per call, 504 calls (72 per decode step, about 100 ms of a 115 ms step). That is the unsorted `B = 4` path the issue leaves alone; it is the next target for gpt-oss decode on this host.

## Decision

The #2066 rule: on by default only if every eligible dtype matches the unsorted path within that path's own error against a dequantized f32 reference, and 512-token prefill improves on every eligible model measured, with no decode change. bf16 is the one eligible mxfp4 dtype and gpt-oss-20b the one eligible mxfp4 checkpoint on this host; both conditions hold, so the mxfp4 arm is on by default with the affine one. `MLX_ROCM_GATHER_QMV_EXPERT_BATCHED=0` turns the kernel off for every scheme.

## Not measured

f16 mxfp4 and mxfp8 do not reach the kernel. Wave64 (CDNA) parts are untested; the 16-lane reduction stays inside a wave on both widths. The four-rows-per-load factor was not retuned for mxfp4. No other mxfp4 MoE checkpoint is on this host.
