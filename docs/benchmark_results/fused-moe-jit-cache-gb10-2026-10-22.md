# Fused MoE decode on GB10: a per-process JIT compile, not a slow kernel (2026-10-22)

Issue #2492 reported granite-4.0-h-tiny-4bit single-sequence decode on GB10 at about 57 tok/s against about 77 on v0.7.0, bisected to #2283 (commit a1c58265), which routed granite's single-token MoE through the fused decode kernel. The issue framed the fix as a per-device routing default, the shape of the per-device draft block width (#1937). This record is the measurement that rejected that shape: on GB10 the fused kernel is faster per token than the `gather_qmm` path for every family measured, and the whole loss came from a fixed NVRTC compile of the kernel pair that every process paid inside its first decode step. The fix caches custom CUDA kernels on disk (`src/lib/mlxcelverse/overlays/cuda/mlx/backend/cuda/jit_module.cpp`), so the routing gate stays on everywhere.

## Host

```
host: spark-101 (also the self-hosted CI runner lablup-dgxspark21)
kernel: 7.0.0-1019-nvidia
nvidia_driver: 580.178.04
gpu: NVIDIA GB10, compute capability 12.1
MLX_CUDA_ARCHITECTURES: 121 (startup line: "compiled for [121] (cubin)")
cuda_toolkit: 13.0 (V13.0.88)
mlx_pin: bff10220c742907f0c68e55ff174aaea56f74e3d
mem_total_gib: 121
base commit: b955bbec (main)
binary profile: test-fast (inherits release with opt-level 3 but no LTO and 16 codegen units; absolute tok/s are not release numbers, ratios are what this record uses)
binary_sha256 main:  7cdd15833f8c42df220a850022cef7cb308d989d76e9a51d88af6824e1788ac3
binary_sha256 fixed: 1970f926d28efa7352d27403edbecbde579203bd10063391c6f8357b2996d597
```

## Method

`mlxcel generate -m <model> -p "<fixed 439-word prompt>" -n <N> --temp 0`, one process per run, each run under `gpu-lock run`. The reported number is the CLI's `[Generated N tokens in T s = R tok/s]` line, which is `decode_tok_per_sec`: generated tokens over the time from the first sampled token to the last, so it excludes prefill but includes the first single-token forward. Arms are interleaved and the arm order rotates every round; one discarded warm-up run per arm precedes round 1. Before every round the harness waited until no CI job (`Runner.Worker` or an `_work/_temp` script) and no `cargo`, `rustc`, `nvcc` or `cc1plus` process was running; one block waited 720 s for a CI job to finish. Every row also logged `/proc/loadavg`, and no measured round started with more than four runnable tasks.

A null arm (the default arm again, same binary, same environment) runs in every block that tests a claim smaller than 10%. Its paired spread is the method's floor for that block. Paired deltas are per-round ratios against the first arm.

## 1. Main as shipped: the regression reproduces, and it is not granite-specific

Binary `main`, `-n 256`, default against `MLXCEL_FUSED_MOE=0`.

| model (experts, top-k, Dff) | arm | n | median tok/s | range | per run |
|---|---|---|---|---|---|
| granite-4.0-h-tiny-4bit (64, 6, 512) | default | 7 | 53.03 | 51.44-53.68 | 53.17 53.06 51.44 51.86 52.55 53.03 53.68 |
| | `=0` | 7 | 65.84 | 64.49-67.12 | 65.27 65.84 67.12 64.49 64.75 65.96 66.22 |
| | null | 7 | 52.77 | 51.63-53.37 | 51.94 53.34 52.77 53.37 51.63 52.09 53.37 |
| lfm2-8b-a1b-4bit (32, 4, 1792) | default | 5 | 69.93 | 69.46-70.52 | 69.93 70.12 69.47 70.52 69.46 |
| | `=0` | 5 | 101.88 | 101.74-103.11 | 101.77 101.88 103.11 102.13 101.74 |
| | null | 5 | 69.11 | 68.40-70.12 | 70.12 68.40 69.63 69.07 69.11 |
| qwen3-30b-a3b-4bit (128, 8, 768) | default | 5 | 45.99 | 44.35-46.51 | 44.35 45.64 45.99 46.48 46.51 |
| | `=0` | 5 | 47.58 | 46.23-47.87 | 47.75 47.87 46.23 46.89 47.58 |
| | null | 5 | 45.02 | 44.12-46.51 | 44.12 46.51 44.39 45.77 45.02 |

Paired `=0` against default: granite +24.1% (+22.8 to +30.5), lfm2 +45.5% (+44.8 to +48.4), qwen3-30b +2.3% (+0.5 to +7.7). Null against default: granite -0.6% (-2.3 to +2.9), lfm2 -0.5% (-2.5 to +0.3), qwen3-30b -1.5% (-3.5 to +1.9).

lfm2 loses more than granite does, so the gate's shape could not be a granite-only entry. The #626 GB10 sweep that set `FUSED_MOE_MAX_DFF_CUDA` recorded lfm2 at +12% with the kernel, but it used `mlxcel-bench-decode`, whose in-process 20-token warmup (`src/bin/bench_decode.rs`) runs the kernels before the timed pass, so that figure is not comparable with a cold-cache `mlxcel generate -n 256`.

`MLXCEL_FUSED_MOE_SGY` on granite (n = 5, same binary): default 53.59, 2 53.26, 4 53.83, 16 52.91, `=0` 67.30. Every rows-per-block value sits within -1.3% to +0.5% of the default while `=0` stays +25.6% ahead, so no scheduling config closes the gap.

## 2. The loss is a fixed cost, not a per-token one

If the kernel were slower per token, the gap would grow with the number of generated tokens. It shrinks instead.

| model | `-n` | default | `=0` |
|---|---|---|---|
| granite (n = 3) | 2 | 0.82 tok/s (2.44 s) | 1.66 tok/s (1.20 s) |
| granite (n = 3) | 32 | 11.72 (2.73 s) | 21.16 (1.51 s) |
| granite (n = 7) | 256 | 53.03 (4.83 s) | 65.84 (3.89 s) |
| lfm2 (n = 5) | 256 | 69.93 (3.66 s) | 101.88 (2.51 s) |
| lfm2 (n = 3) | 1024, 970 generated | 117.28 (8.27 s) | 128.87 (7.53 s) |

Fitting decode time = fixed + tokens x per-token on those rows: granite gather 10.6 ms/token with 1.2 s fixed, fused 9.4 ms/token with 2.4 s fixed; lfm2 gather 7.0 ms/token with 0.7 s fixed, fused 6.5 ms/token with 2.0 s fixed. The fused kernel is the faster path per token on both, and it carries about 1.2 to 1.3 s of extra fixed cost. At `-n 2` the decode time is nearly all fixed cost, and the two arms differ by 1.24 s. (A granite `-n 1024` pair was also run but is not used: the two paths hit EOS at 786 and 648 tokens, so the pair is not paired.)

The mechanism is in MLX's CUDA backend. `custom_kernel.cpp` builds every `mx.fast.cuda_kernel` through `cu::get_jit_module(..., use_disk_cache = false)`. The JIT module tries the disk cache on every build but never writes one for a custom kernel, so each process NVRTC-compiles the fused MoE gate-up and down kernels again. Their first launch is the first single-token forward, which `decode_tok_per_sec` counts. Every run of either arm also printed the cold-cache notice (`Compiling CUDA kernels (first run on this host; cached for later runs)...`), because granite's SSM step is a custom kernel too; that is part of the gather arm's own 1.2 s fixed cost.

## 3. The fix: custom kernels go to the disk cache

The `jit_module.cpp` overlay writes custom kernels to `MLX_PTX_CACHE_DIR` like the built-in ones and publishes every cache file by rename. A custom kernel's module name is `custom_kernel_<name><template args>_<hash of the full source>`, so a hit can only return code compiled from the same source.

Same session, interleaved, `-n 256`. `new*` arms run the fixed binary on a fresh cache directory populated by the warm-up run; `old` runs the main binary on its own cache directory, so it cannot read entries the fixed binary wrote (it would: the main binary already reads custom kernels from disk, it only never writes them).

| model | arm | n | median tok/s | range | per run |
|---|---|---|---|---|---|
| granite | new (default) | 7 | 89.20 | 87.76-90.69 | 89.20 89.66 88.01 90.69 87.76 90.22 88.29 |
| | new `=0` | 7 | 81.63 | 81.22-82.70 | 81.43 82.41 81.22 81.63 82.06 81.27 82.70 |
| | new null | 7 | 88.52 | 87.73-90.09 | 88.52 89.19 88.43 89.62 88.37 90.09 87.73 |
| | old (default) | 7 | 53.63 | 52.48-53.91 | 52.48 53.63 53.45 53.91 53.77 53.37 53.73 |
| lfm2 | new (default) | 5 | 109.65 | 107.77-110.89 | 107.77 109.65 109.62 110.89 110.42 |
| | new `=0` | 5 | 102.25 | 101.80-103.61 | 101.80 102.25 103.61 102.12 103.54 |
| | new null | 5 | 110.21 | 109.30-112.40 | 110.21 109.65 109.30 112.40 111.89 |
| | old (default) | 5 | 70.74 | 69.66-71.26 | 70.74 70.90 71.26 70.73 69.66 |
| qwen3-30b | new (default) | 5 | 56.20 | 53.87-59.85 | 55.64 53.87 56.20 59.85 58.95 |
| | new `=0` | 5 | 46.56 | 43.71-47.44 | 47.44 46.97 46.56 44.69 43.71 |
| | old (default) | 5 | 45.05 | 43.77-46.36 | 45.05 45.35 44.33 46.36 43.77 |

Paired against new default: granite `=0` -8.1% (-10.0 to -6.3), null -0.5% (-1.2 to +0.7), old -40.2% (-41.2 to -38.7); lfm2 `=0` -6.2% (-7.9 to -5.5), null +1.3% (-0.3 to +2.3), old -35.3% (-36.9 to -34.4); qwen3-30b `=0` -17.2% (-25.9 to -12.8), old -21.1% (-25.8 to -15.8). Granite `-n 2` on the fixed binary: default 3.78 tok/s, `=0` 3.84 (n = 3), so the extra 1.2 s intercept is gone. After the warm-ups no fixed-binary run printed the cold-cache notice, and every `old` run still did.

With the cache in place the fused kernel is the faster path on all three families on GB10, granite included, so the routing gate keeps its default (on) on every device and no per-device policy is added. The gather arm also gains (granite 65.84 to 81.63) because the SSM custom kernel is now cached as well.

## What this does not cover

- gfx1151 (ROCm) and Metal were not measured; the change is in the CUDA overlay only and does not reach either backend.
- `mlxcel-server` was not measured. A server pays the compile once per process lifetime either way, so its steady-state decode does not change; its first request after start no longer pays it from the second launch on.
- The `FUSED_MOE_MAX_DFF_CUDA` provenance (#626, #712) was measured with `mlxcel-bench-decode`, whose warmup most likely absorbed this compile, so it is probably unaffected; GB10 fused-MoE numbers taken with `mlxcel generate` before this change include the fixed cost. Neither was re-measured here; the re-measurement on the current MLX pin is tracked in #2540.
