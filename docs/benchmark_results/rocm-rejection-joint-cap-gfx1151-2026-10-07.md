# Joint top-k + top-p rejection cap on gfx1151 (2026-10-07)

lablup/mlxcel#2157, part of #1801. The rejection sampler (#901) routes top-k together with top-p only up to a vocabulary cap, `REJECTION_JOINT_VOCAB_MAX` in `src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp`. Its value, 32768, was measured on M1 Ultra ([rejection-sampling-m1ultra-2026-08-02.md](rejection-sampling-m1ultra-2026-08-02.md)). The HIP port (#2064) inherited it unmeasured, so `--top-k 40 --top-p 0.95` on a Llama 3 checkpoint (vocab 128256) never reached the kernel ([rocm-samplers-gfx1151-2026-10-05.md](rocm-samplers-gfx1151-2026-10-05.md)). This page re-measures the crossover on gfx1151.

**Outcome: a ROCm build routes top-k with top-p up to vocab 152064; Metal and CUDA keep 32768.** The microbenchmark alone would have kept 32768: its batch 4 and 8 cells win at every vocabulary, but its synthetic batch-1 cell does not clear 1.0 above 32768. End-to-end decode, which the issue's rule names as the confirmation at 128256 and above, disagrees with that cell by a wide margin: with the joint case routed, decode is 1.15x faster on Meta-Llama-3.1-8B-Instruct-4bit (vocab 128256) and 1.12x on Qwen2.5-7B-Instruct-4bit (152064), and every kernel run was faster than every chain run; on the final build rebased onto current `main` the Llama 3.1 pairs read 1.14x. The cap is set to 152064, the largest vocabulary measured both ways. Why the synthetic batch-1 cell does not predict real decode on this GPU was not isolated; a bf16 rerun ruled out the logits dtype.

## Environment

AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`, RDNA 3.5), 96 GiB VRAM carve-out, Debian 13, ROCm 10.0.0 (HIP 7.15.26333), MLX pin `81ba1c6a`, ROCm overlay `75915908`, `cargo build --release --features rocm`. Measurement build: `main` at `84d3a7bc` plus this change with the ROCm cap raised to `INT32_MAX`, so `fused_sample` routes every joint cell; its binaries were copied out of the tree before the source moved on, so its CSVs record `mlxcel_commit` as `unknown`. Final verification: this branch rebased on `main` at `ad844354` (after #2224 and #2225 moved `mlxcel generate` onto the engine). Checkpoints `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (vocab 128256) and `mlx-community/Qwen2.5-7B-Instruct-4bit` (152064).

## Microbenchmark

`examples/rejection_sampling_microbench.rs`, 200 timed and 30 warmup iterations per cell, three runs, each under `scripts/rocm_gpu_guard.sh` (main's copy, host-wide lock from #2146). Of the f32 runs, 1 and 3 were clean on the first attempt; the guard rejected run 2's first attempt for contention and its second was clean. All three bf16 runs were clean on the first attempt. This change adds 128256 (Llama 3) and 151936 (Qwen3) to the harness's vocabularies, adds `--dtype` and `--config` options, and fixes its port gate: it refused to run on ROCm because it checked `custom_kernels_available()`, which is Metal-or-CUDA by definition; it now checks `sampling_rejection_backend_supported()`.

The pipelined arm samples through `fused_sample`, the production entry point, so on a cell production does not route it measures the stock chain against itself; that is why the measurement build lifted the cap. Every joint cell reads `routed=yes` in the raw CSVs. Read `pipe_x`, not `iso_x` (the harness header explains why). The synthetic forward measured 2110 to 2659 us per step, and the cap-overflow counter stayed at 0 in every run. Raw rows: `benchmarks/rocm_strixhalo-gfx1151_2026-10-07_rejection-microbench-run{1,2,3}.csv` (f32, every configuration) and `..._rejection-microbench-bf16-run{1,2,3}.csv` (bf16, joint only).

### top-k 40 + top-p 0.9, f32 logits

Medians of three runs (min to max in brackets). Times are microseconds, stock chain / kernel.

| vocab | batch | worst rounds | iso us | iso_x | pipe us | pipe_x |
|---|---|---|---|---|---|---|
| 32768 | 1 | 5 | 318 / 143 | 2.23x (2.19-2.23) | 2459 / 2462 | 1.00x (0.96-1.04) |
| 32768 | 4 | 5 | 702 / 174 | 4.03x (4.02-4.04) | 2858 / 2440 | 1.17x (1.10-1.22) |
| 32768 | 8 | 5 | 1254 / 207 | 6.11x (6.06-6.11) | 3453 / 2404 | 1.44x (1.29-1.44) |
| 65536 | 1 | 4 | 405 / 257 | 1.59x (1.58-1.59) | 2537 / 2591 | 0.98x (0.95-1.05) |
| 65536 | 4 | 4 | 1002 / 367 | 2.76x (2.72-2.77) | 3165 / 2675 | 1.18x (1.09-1.23) |
| 65536 | 8 | 4 | 1869 / 420 | 4.45x (4.41-4.49) | 4402 / 2698 | 1.64x (1.63-1.73) |
| 128256 | 1 | 4 | 591 / 617 | 0.96x (0.95-0.97) | 2731 / 2789 | 0.98x (0.96-0.99) |
| 128256 | 4 | 5 | 1794 / 809 | 2.23x (2.21-2.23) | 4122 / 3222 | 1.28x (1.27-1.28) |
| 128256 | 8 | 6 | 3512 / 1130 | 3.12x (3.08-3.17) | 6345 / 3518 | 1.80x (1.79-1.83) |
| 151936 | 1 | 5 | 666 / 724 | 0.92x (0.91-0.93) | 2809 / 3124 | 0.90x (0.88-0.92) |
| 151936 | 4 | 5 | 2078 / 1101 | 1.88x (1.88-1.91) | 4667 / 3428 | 1.36x (1.31-1.47) |
| 151936 | 8 | 5 | 4028 / 1349 | 2.99x (2.96-3.46) | 7083 / 3728 | 1.90x (1.87-1.99) |
| 152064 | 1 | 4 | 666 / 722 | 0.92x (0.92-0.93) | 2815 / 3024 | 0.93x (0.88-0.97) |
| 152064 | 4 | 6 | 2060 / 1136 | 1.83x (1.81-1.84) | 4452 / 3420 | 1.29x (1.28-1.38) |
| 152064 | 8 | 6 | 4351 / 1344 | 3.24x (2.95-3.45) | 7074 / 3549 | 2.02x (1.88-2.06) |

### top-k 40 + top-p 0.9, bf16 logits

Both checkpoints are bf16, so the sampler sees bf16 logits in decode. The same matrix with the synthetic logits cast to bf16 (`--dtype bf16 --config top-k+top-p`):

| vocab | batch | worst rounds | iso us | iso_x | pipe us | pipe_x |
|---|---|---|---|---|---|---|
| 32768 | 1 | 3 | 328 / 144 | 2.29x (2.27-2.30) | 2464 / 2275 | 1.08x (1.00-1.09) |
| 32768 | 4 | 3 | 686 / 179 | 3.82x (3.82-3.83) | 2841 / 2334 | 1.21x (1.17-1.22) |
| 32768 | 8 | 4 | 1203 / 208 | 5.79x (5.74-5.91) | 3376 / 2395 | 1.41x (1.40-1.45) |
| 65536 | 1 | 3 | 422 / 252 | 1.67x (1.62-1.68) | 2556 / 2513 | 1.02x (0.99-1.06) |
| 65536 | 4 | 4 | 966 / 349 | 2.81x (2.76-2.82) | 3117 / 2542 | 1.23x (1.18-1.24) |
| 65536 | 8 | 4 | 1753 / 379 | 4.67x (4.60-4.84) | 4026 / 2590 | 1.54x (1.51-1.56) |
| 128256 | 1 | 4 | 502 / 579 | 0.87x (0.86-0.89) | 2640 / 2706 | 0.98x (0.91-0.98) |
| 128256 | 4 | 5 | 1244 / 710 | 1.76x (1.73-1.76) | 3477 / 3012 | 1.14x (1.11-1.19) |
| 128256 | 8 | 5 | 2221 / 1028 | 2.17x (2.15-2.17) | 4751 / 3273 | 1.47x (1.35-1.50) |
| 151936 | 1 | 4 | 549 / 722 | 0.76x (0.76-0.78) | 2680 / 3006 | 0.89x (0.88-0.92) |
| 151936 | 4 | 4 | 1386 / 1037 | 1.34x (1.32-1.36) | 3663 / 3320 | 1.16x (1.09-1.27) |
| 151936 | 8 | 6 | 2494 / 1235 | 2.02x (2.01-2.02) | 5107 / 3671 | 1.38x (1.32-1.64) |
| 152064 | 1 | 4 | 550 / 683 | 0.81x (0.78-0.81) | 2692 / 3127 | 0.86x (0.86-0.93) |
| 152064 | 4 | 6 | 1405 / 1041 | 1.34x (1.33-1.36) | 3624 / 3463 | 1.05x (0.78-1.15) |
| 152064 | 8 | 6 | 2497 / 1248 | 2.00x (1.99-2.00) | 5209 / 3587 | 1.47x (1.41-1.48) |

The dtype does not change the picture: batch 1 still reads below 1.0 from 128256 up, so the dtype is not why the microbenchmark and end-to-end decode disagree.

How much the batch-1 pipelined cells can say is limited. Rows that production does not route (top-k alone and min-p alone in the f32 CSVs, chain against chain in the pipelined arm) scatter between 0.90x and 1.11x by median, so a batch-1 `pipe_x` between 0.95 and 1.05 is inside the noise either way. The batch-1 isolated cells are tight (spread under 3%) and show the kernel slower from 128256 up on these synthetic rows, which sample in four to six rounds.

## End-to-end decode

`scripts/bench_decode.sh` at its default pp512/tg128 shape with `--temperature 0.8 --top-k 40 --top-p 0.95` (the `--top-k` option is new in this change), the measurement build against the same binaries with `MLXCEL_SAMPLING_REJECTION=0`. Each pair ran inside one `scripts/rocm_gpu_guard.sh` attempt, alternating which arm went first, and all six pairs were clean on their first attempt. A short `mlxcel generate` run on the same build logged `sampling dispatch: rejection kernel (batch 1, vocab 128256, temperature 0.8, top_k 40, top_p 0.95, ...)`, so the kernel arm reached the kernel at the Llama 3 vocabulary. Raw rows: `benchmarks/rocm_strixhalo-gfx1151_2026-10-07_joint-cap-measurement_<model>_{kern,chain}.csv`.

| Model (vocab) | Stock chain tok/s | Rejection kernel tok/s | Medians |
|---|---|---|---|
| Meta-Llama-3.1-8B-Instruct-4bit (128256) | 33.94 / 31.64 / 32.67, median 32.67 | 37.70 / 37.37 / 37.57, median 37.57 | 1.15x |
| Qwen2.5-7B-Instruct-4bit (152064) | 42.68 / 41.91 / 40.79, median 41.91 | 46.92 / 46.40 / 47.18, median 46.92 | 1.12x |

Every kernel run is faster than every chain run on both models, by 10% to 18% within each pair. The chain arm also spreads more (4.5 to 7.0% of its median, against 0.9 to 1.7% for the kernel), as the top-p chain did on Llama 3.1 in #2064.

### Final build, rebased on current main

The same pairs on this branch's own build after rebasing onto `main` at `ad844354`, where `mlxcel-bench-decode` now decodes through the engine (#2224, #2225; the CSV's last column reads `engine`). The arms are the shipped default, which now routes the joint case at 128256, and `MLXCEL_SAMPLING_REJECTION=0`, which is the path `main` takes for this configuration (it never routed the joint case above 32768). An `mlxcel generate` run in the first pair logged `sampling dispatch: rejection kernel (batch 1, vocab 128256, temperature 0.8, top_k 40, top_p 0.95, ...)` by default and `sampling dispatch: argpartition chain: pinned by MLXCEL_SAMPLING_REJECTION` with the switch. All three pairs were clean on their first guard attempt. Raw rows: `benchmarks/rocm_strixhalo-gfx1151_2026-10-08_joint-cap-{after,before}_Meta-Llama-3.1-8B-Instruct-4bit.csv` (`mlxcel_commit` `79ea5814`).

| Model (vocab) | Before (stock chain) tok/s | After (rejection kernel) tok/s | Medians |
|---|---|---|---|
| Meta-Llama-3.1-8B-Instruct-4bit (128256) | 33.15 / 35.76 / 33.09, median 33.15 | 37.61 / 38.45 / 37.73, median 37.73 | 1.14x |

Every after run is faster than every before run, by 1.08x to 1.14x within each pair. Qwen2.5-7B was not re-run on the rebased build: the shared GPU was contended for hours at a time, and the measurement-build pairs above already separate cleanly.

## Decision

The issue's rule takes the largest measured vocabulary V where every microbenchmark batch cell at V and below has `pipe_x >= 1.0`, confirmed end to end when V is 128256 or more. Read literally, the synthetic batch-1 cells stop it at 32768. The end-to-end confirmation at 128256 and 152064 contradicts them: the configuration the issue was opened for is 15% faster on Llama 3.1 and 12% faster on Qwen 2.5 when routed, beyond the run-to-run spread, and the microbenchmark's own batch 4 and 8 cells win at every vocabulary. Keeping that configuration on the slower path in production because of a synthetic cell would be the wrong call, so the ROCm value is 152064, the largest vocabulary measured both ways.

- Vocabularies above 152064 (gpt-oss 201088, Gemma 3 262144) are unmeasured and stay on the stock chain.
- The value is a build flag, `#ifdef MLXCEL_BRIDGE_ROCM_BACKEND` around `REJECTION_JOINT_VOCAB_MAX`, the same pattern as the fused-MoE SGY default in `mlx_cxx_kernels.cpp`. A ROCm build has no Metal or CUDA backend, so there is no runtime backend comparison, and `make verify-kernel-port-dispatch` passes. Metal and CUDA builds keep 32768.
- `sampling_rejection_joint_vocab_max()` reports the active value. `the_joint_vocab_cap_is_the_measured_crossover` pins it per build (`cfg!(feature = "rocm")`) and checks that the policy turns exactly there, and the not-routed message names it. Anything that unifies the per-row sampling step (epic #2166 Phase 2) carries the policy forward by calling `sampling_rejection_routes`, or this accessor, rather than copying the number.

## Reproducing

```bash
# Measurement build only: set the ROCm REJECTION_JOINT_VOCAB_MAX to 2147483647 first.
cargo build --release --features rocm --example rejection_sampling_microbench --bin mlxcel-bench-decode
scripts/rocm_gpu_guard.sh -- ./target/release/examples/rejection_sampling_microbench --csv /tmp/rejection-gfx1151.csv
scripts/rocm_gpu_guard.sh -- ./target/release/examples/rejection_sampling_microbench --dtype bf16 --config top-k+top-p --csv /tmp/rejection-gfx1151-bf16.csv
scripts/rocm_gpu_guard.sh -- scripts/bench_decode.sh models/mlx/Meta-Llama-3.1-8B-Instruct-4bit --temperature 0.8 --top-k 40 --top-p 0.95 --output /tmp/joint-kern.csv
MLXCEL_SAMPLING_REJECTION=0 scripts/rocm_gpu_guard.sh -- scripts/bench_decode.sh models/mlx/Meta-Llama-3.1-8B-Instruct-4bit --temperature 0.8 --top-k 40 --top-p 0.95 --output /tmp/joint-chain.csv
cargo test --release --features rocm -p mlxcel-core --lib -- --test-threads=1 sampling_rejection_tests sampling_fixed_key_tests
```
