# mxfp4 MoE decode through the warp-shared `gather_qmm` kernel on gfx1151 (2026-10-08)

lablup/mlxcel#2178, part of #1814. The #2106 page ([rocm-moe-prefill-mxfp4-gfx1151-2026-10-06.md](rocm-moe-prefill-mxfp4-gfx1151-2026-10-06.md)) left gpt-oss-20b decode at about 8.5 tok/s, with the per-row `gather_qmv_kernel<hip_bfloat16, unsigned char, 4, 32, false>` at 1.39 ms per call and 72 calls per step: a decode step has `B = top_k = 4`, below the expert-batched kernel's gate. This page has the correctness checks, the kernel trace, the decode and prefill numbers before and after giving the warp-shared gather kernel an mxfp4 path, and the default decision.

## Environment

AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`, RDNA 3.5), Debian 13, HIP 7.15.26333, rocprofv3 1.3.5, Rust 1.97.1. mlxcel `main` at `ad844354` plus this change, built with `cargo build --release --features rocm`; MLX pin `81ba1c6a`. Checkpoint: `models/mlx/gpt-oss-20b-MXFP4-Q4` (bf16 activations, 24 layers, 32 experts, top 4, hidden and expert widths 2880, mxfp4 experts at group size 32).

Every GPU run on this page went through `scripts/rocm_gpu_guard.sh`, nested in one hold of the host-wide guard lock; all 15 guarded attempts were accepted on their first try, so no run overlapped another GPU process or a compiler.

## What changed

`gather_qmv_warp_shared_kernel` in `patches-rocm/mlx/backend/rocm/quantized/qmm.hip` gains a path for mxfp4 at group size 32: every lane takes one packed word (eight e2m1 nibbles, all in one group of 32) per step across the whole `shared_x` chunk, reads the group's E8M0 scale, decodes each nibble without a branch (`fp4_e2m1_to_float_scaled`, the value times 2^-14) and scales the word's dot product back by 2^14. The kernel's existing per-group loop, in which only lanes 0 to 3 of 16 had work at group size 32, stays for the other non-affine cases. `GatherQMM::eval_gpu` sends bf16 mxfp4 with `K % 32 == 0` to it after the affine warp-shared arm, unsorted and sorted calls alike; `MLX_ROCM_GATHER_QMV_USE_WARP=0` (read on every call) sends it back to the per-row kernel, which is the "off" arm below. Two new instantiations (16 lanes and `WARP_SIZE` lanes per column); f16 and f32 mxfp4 and mxfp8 keep the per-row kernel. `patches-rocm/LOCAL_FIXES.md` item 41.

## Correctness

`tests/rocm_mxfp4_quant.rs` (`mxfp4_warp_shared_gather_qmv_matches_per_row_and_reference`), gpt-oss-20b's expert layer (32 experts, top 4, `K = 2880`, two `shared_x` chunks), bf16, relative L2 error against per-expert dense f32 matmuls of the CPU-dequantized weights:

| Case | `N = 2880`, default / per-row | Outputs differing | `N = 516`, default / per-row | Outputs differing |
|---|---|---|---|---|
| unsorted, 1 token (`B = 4`, a decode step) | 1.6358e-3 / 1.6358e-3 | 0 of 11520 | 1.6588e-3 / 1.6588e-3 | 0 of 2064 |
| unsorted, 8 tokens (`B = 32`) | 1.6609e-3 / 1.6609e-3 | 5 of 92160 | 1.6827e-3 / 1.6827e-3 | 1 of 16512 |
| sorted, 16 tokens (`B = 64`, misses the expert-batched gate) | 1.6569e-3 / 1.6569e-3 | 7 of 184320 | 1.6606e-3 / 1.6606e-3 | 0 of 33024 |

The two paths agree to five digits, and two default runs are bit-identical. The kernels sum in different orders, but over 2880 f32 terms the sums rarely straddle a bf16 rounding boundary, so most outputs round to the same bytes; the test requires some difference over the full-width cases (12 measured: 5 in the 8-token case and 7 in the sorted case) to prove the dispatch left the per-row kernel. With the sign bit dropped from the new path's decode the first case fails at 1.211 (the per-row path stays at 1.636e-3). The expert-batched tests (`tests/rocm_gather_qmm_expert_batched.rs`, 2 tests) and `gather_qmm_matches_per_expert_reference` pass; the expert-batched test's own dispatch check now selects the per-row kernel with `MLX_ROCM_GATHER_QMV_USE_WARP=0`, because the new path sums in the same order as the expert-batched kernel.

Teacher-forced logit traces (`examples/logit_trace` over `tests/fixtures/wikitext2_excerpt.txt`, `w8` = `8 80 8 512`: each 8-token chunk runs `B = 32` unsorted, which takes the new path), per-row against default on the same binary, `scripts/compare_logit_traces.py --decided 2.0`: top-1 disagreement 23 of 640, decided mismatches 0 of 89, largest gap at a disagreement 0.250, perplexity 159.95 per-row and 158.99 default. Every disagreement sits where the reference's top-two gap is under 0.5 and lands at rank 2 to 4 of the candidate; the script's verdict is the rounding class, not a behaviour change. The traces are not committed.

## Kernel trace

`rocprofv3 --kernel-trace --stats` of `mlxcel-bench-decode` with the default dispatch (`--prompt-tokens 512 -n 8 --warmup-tokens 1`, two prefills and seven decode steps):

| Kernel | Calls | Average per call | Share of kernel time |
|---|---|---|---|
| `gather_qmv_expert_batched_kernel<hip_bfloat16, unsigned char, 4, 32, false, 16>` (prefill) | 144 | 12.29 ms | 78.0% |
| `gather_qmv_warp_shared_kernel<hip_bfloat16, unsigned char, 4, 32, false, 16>` (decode) | 504 | 108.3 us | 2.4% |
| `gather_qmv_kernel` (any instantiation) | 0 | | |

The decode calls are the same 504 (72 per step) the #2106 trace attributed to the per-row kernel at 1.39 ms each: 12.8x per call.

## Decode and prefill

`mlxcel-bench-decode` at `scripts/bench_decode.sh`'s shape (512-token prompt, 128 generated tokens, 20-token warmup, `--ignore-eos`), one binary, `MLX_ROCM_GATHER_QMV_USE_WARP=0` (off: the per-row kernel, `main`'s dispatch for mxfp4) and the default (on) alternated run by run, six rounds:

| Round | Prefill tok/s, off | Prefill tok/s, on | Decode tok/s, off | Decode tok/s, on |
|---|---|---|---|---|
| 1 | 509.94 | 509.16 | 8.37 | 63.65 |
| 2 | 512.47 | 511.15 | 8.28 | 63.74 |
| 3 | 511.83 | 512.21 | 8.23 | 63.48 |
| 4 | 511.32 | 511.89 | 8.29 | 63.59 |
| 5 | 511.76 | 510.18 | 8.35 | 63.79 |
| 6 | 511.69 | 510.14 | 8.39 | 63.42 |
| Median | 511.73 | 510.67 | 8.32 | 63.62 |

Decode: 7.65x (medians), every on run faster than every off run, against the about 1% host spread #2106 measured with identical code on both arms; a 128-token decode takes 2.0 s instead of 15.3 s. Prefill: the medians are 0.2% apart and the ranges (509.9 to 512.5 off, 509.2 to 512.2 on) overlap, inside that spread; the 512-token prefill reaches the expert-batched kernel on both arms.

## Decision

The issue's rule: the arm is the default only if decode improves by more than the host spread over six alternated guarded rounds and 512-token prefill stays within it. Decode improved 7.65x and prefill moved 0.2%, so the arm is on by default, with no opt-in; `MLX_ROCM_GATHER_QMV_USE_WARP=0` turns it off.

## Not measured

f16 and f32 mxfp4 and mxfp8 do not reach the new path. Wave64 (CDNA) parts are untested; the 16-lane instantiation reduces inside a wave on both widths, and the `WARP_SIZE` one is selected only for `K >= 16384` with one routing entry (`B = 1`) or through `MLX_ROCM_GATHER_QMV_THREADS_PER_COL`. The dense mxfp4 `qmv_warp_shared_kernel` has the same per-group lane pattern and was not changed. No other mxfp4 MoE checkpoint is on this host.
