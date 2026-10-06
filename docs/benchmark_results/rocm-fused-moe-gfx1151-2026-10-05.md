# HIP ports of the fused MoE decode kernels on gfx1151 (2026-10-05)

lablup/mlxcel#2065, part of #1814. Before this change ROCm ran `gather_qmm` for every routed expert at single-token decode, because `moe_gateup_ports()` and `moe_down_ports()` had no `.rocm` entry and the SwitchGLU gate read the backend-wide `custom_kernels_available()`. The decode profile ([rocm-decode-profile-gfx1151-2026-09-30.md](rocm-decode-profile-gfx1151-2026-09-30.md)) put fused-MoE-reachable work at 46.8% of Qwen3-30B-A3B decode GPU time, but 42.1 points of that are expert GEMVs already running near the host's bandwidth, so it estimated about 5% of GPU time plus about 430 dispatches per token as the most a fused kernel could recover.

The ports (`MOE_GATEUP_HIP_SOURCE`, `MOE_DOWN_HIP_SOURCE` in `src/lib/mlxcel-core/cpp/mlx_cxx_kernels.cpp`) are the CUDA kernels with `__shfl_down(v, o, 32)` for the lane fold. `fused_moe_kernels_available()` and `moe_down_kernel_available()` read the two tables, and `switch_layers.rs` and `nemotron_h.rs` gate on them. On ROCm the rows-per-threadgroup default (`MLXCEL_FUSED_MOE_SGY`) is 2 instead of Metal's 8, for the reason in the next section.

## Environment

AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`, RDNA 3.5), 96 GiB VRAM carve-out, Debian 13, ROCm 10.0.0 (HIP 7.15.26333). Before: `origin/main` `57d8ed29`. After: the PR branch at `77afbb1d` on that main. MLX pin `81ba1c6a`, ROCm overlay `75915908`, `cargo build --release --features rocm`. Checkpoints: the same local `mlx-community` directories as the earlier correctness rows.

## Decode throughput

`scripts/bench_decode.sh` at its default pp512/tg128 shape, one model per run, before and after binaries alternated run by run (before, after, next model), three rounds. Every run went through `scripts/rocm_gpu_guard.sh` with the fix from this PR, and every attempt that overlapped another unit's GPU work or a compiler was rejected and rerun. Raw rows: `benchmarks/rocm_strixhalo-gfx1151_2026-10-05_fused-moe-before.csv` and `..._fused-moe-after.csv`.

| Model | Path change | Before tok/s | After tok/s | Change (medians) |
|---|---|---|---|---|
| Qwen3-30B-A3B-4bit | `gather_qmm` to the fused HIP pair | 61.29 / 61.05 / 61.83, median 61.29 | 62.51 / 62.68 / 62.16, median 62.51 | +2.0% |
| NVIDIA-Nemotron-3-Nano-30B-A3B-4bit | `forward_nonfused` to `fused_moe_forward` (still `gather_qmm`) | 74.30 / 74.62 / 74.37, median 74.37 | 75.11 / 75.09 / 74.88, median 75.09 | +1.0% |
| Mixtral-8x7B-Instruct-v0.1-4bit | none (Dff 14336 is above the 8192 `MLXCEL_FUSED_MOE_MAX_DFF` default) | 9.42 / 9.08 / 10.91, median 9.42 | 9.51 / 9.57 (two runs: the third never found a clean GPU window before the host's other work resumed) | +1.3% (noise; same code) |

Qwen3's gain is consistent run to run (every after run is above every before run) and matches the profile's ceiling: the fused pair removes the activation, weighted-sum and index-building dispatches but reads the same expert weights. Nemotron-H's 1% is near its run-to-run spread; its routed experts still run `gather_qmm` (the fused fc1 kernel is #2069), and only the surrounding combine changed path. Mixtral runs the same code before and after, so its spread (9.08 to 10.91 tok/s) is this host's noise for a 26 GB checkpoint on a 31 GiB host and says nothing about the change. `compare_bench_csv.py --allow-commit-change` on the per-model median rows reports a 1.016x median over the three pairs, none moved by more than 10%. Prefill is not on the fused path (a prompt is more than one token).

### Why the ROCm SGY default is 2

With Metal's default of 8 rows per threadgroup, the port made Qwen3 decode slower than the path it replaces: one guarded pair before the default changed measured 61.64 tok/s on `gather_qmm` (`c0b71344`) against 59.42 tok/s fused. A sweep of `MLXCEL_FUSED_MOE_SGY` over 1, 2, 4, 8, 16 and 32 on one binary put 2 on top in every repetition (about 62.3 tok/s, against about 61 for `MLXCEL_FUSED_MOE=0`, about 61.2 for 1, and 59.3 to 60.4 for the rest). That sweep is indicative only: each of its nine guarded attempts saw another unit's GPU process in at least one sample, so the guard rejected all of them, and the table above, measured at the new default, is the result this page stands on. SGY shapes the threadgroup only; `fused_moe_geglu_kernel_bitwise_invariant_across_sgy` pins that the output does not depend on it.

## Correctness

### Kernel against references

`fused_moe_parity_tests` runs on gfx1151 for the first time, gated on `fused_moe_kernels_available()` (a GPU backend without the ports now fails the test instead of skipping it). The fused output is within 3.0e-6 normalized RMS of an all-f32 dense reference and about 3.7e-3 from `gather_qmm`, which is itself about 3.3e-3 from that reference; bounds unchanged. A new case covers SwiGLU at 4/4, 8/8 and 4/6 bits on the Qwen3 shape, reaching the SwiGLU arm and the 6-bit down branch. Starting the down kernel's lane fold at 8 instead of 16 fails both reference tests (normalized RMS 0.63 and 0.70).

### Model logits

New traces: `benchmarks/logit_traces/rocm_gfx1151_77afbb1d/`. The fused kernels run only on a single-token forward, so `w1` (`1 128 8 0`) is the row that reaches them on Qwen3; at `w8` its `default` and `fused0` traces are identical. `python3 scripts/compare_logit_traces.py <reference> <candidate> --decided 2.0`.

| Model, window | Reference | Candidate | Top-1 disagreement | Decided mismatches | Largest gap | Perplexity ref / cand |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B, w1 | Metal fused (`metal_m1u_bec64748` default) | ROCm fused (HIP) | 1 / 128 | 0 / 74 | 0.688 | 75742.6 / 76499.3 |
| Qwen3-30B-A3B, w1 | Metal `gather_qmm` (fused0) | ROCm `gather_qmm` (fused0) | 5 / 128 | 0 / 70 | 0.750 | 76983.0 / 78852.2 |
| Qwen3-30B-A3B, w1 | ROCm `gather_qmm` (fused0) | ROCm fused | 3 / 128 | 0 / 74 | 0.062 | 78852.2 / 76499.3 |
| Qwen3-30B-A3B, w8 | Metal default | ROCm default | 17 / 640 | 0 / 319 | 0.500 | 10.921 / 10.907 |
| Mixtral-8x7B, w1 | Metal default | ROCm default (`gather_qmm`) | 0 / 128 | 0 / 17 | none | 1589.66 / 1588.28 |
| Nemotron-3-Nano, w1ctx512 | ROCm main (`rocm_gfx1151_96cbce84`) | this branch | 3 / 128 | 0 / 68 | 0.125 | 7.442 / 7.410 |
| Nemotron-3-Nano, w1ctx512 | Metal (`metal_m5_d1128266`) | this branch | 6 / 128 | 0 / 71 | 0.250 | 7.489 / 7.410 |
| Nemotron-3-Nano, w1ctx512 | Metal (`metal_m5_d1128266`) | ROCm main (control) | 3 / 128 | 0 / 71 | 0.250 | 7.489 / 7.442 |
| Nemotron-3-Nano, w8 | ROCm main | this branch | 17 / 640 | 0 / 283 | 0.500 | 10.983 / 10.980 |
| Nemotron-3-Nano, w8 | ROCm `rocm_gfx1151_c5fe9a16` (`forward_nonfused`) | this branch | 20 / 640 | 0 / 286 | 0.500 | 10.984 / 10.980 |
| Nemotron-3-Nano, w8 | Metal (`metal_m5_d1128266`) | this branch | 22 / 640 | 0 / 287 | 0.562 | 10.956 / 10.980 |

Every row has zero disagreements on a decided position. With the HIP kernels Qwen3's single-token logits are closer to Metal's fused kernel (1 top-1 disagreement) than ROCm's `gather_qmm` was to Metal's (5). Nemotron-H's change of path (`fused_moe_forward` instead of `forward_nonfused`) moves a few undecided tokens on both windows, at reference gaps of 0.5 or less. The `w1` perplexities are large because those chunks have no context; the `w1ctx512` and `w8` rows are the meaningful ones for perplexity.

The `gather_qmm` path is unchanged: the branch's Qwen3 `fused0` `w1` trace and Mixtral `w1` trace are byte-identical in every data row to `rocm_gfx1151_bec64748`'s `fused0` traces, and its Qwen3 `fused0` `w8` trace is byte-identical to one built on `main` `57d8ed29` (both differ from `bec64748` at `w8`, from changes that predate this PR).

Greedy text: Nemotron-3-Nano generates the same 40-token greedy text with and without `MLXCEL_FUSED_MOE_RELU2=1`, which now declines to `gather_qmm` on ROCm instead of refusing (its fc1 kernel has no HIP port until #2069).

## Not measured

Gemma 4 (GeGLU) and Qwen3-Next reach the same kernels through the same gate but have no checkpoint on this host. Wave64 (CDNA) parts are untested: the `#error` guard the #1814 ports carry is inert with AMD clang 23, which defines neither `__AMDGCN_WAVEFRONT_SIZE` spelling, and the explicit shuffle width of 32 is what keeps each fold inside one row there.
