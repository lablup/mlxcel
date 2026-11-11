# Fused decode-MoE Dff crossover on GB10, re-measured (2026-11-11)

Issue #2540 asked for a re-measurement of the CUDA fused decode-MoE bound, `FUSED_MOE_MAX_DFF_CUDA = 8192` (`src/models/switch_layers.rs`), on the current MLX pin, with a null arm and interleaved rounds. This record is that measurement, on a DGX Spark GB10 at `28d884a5` (MLX pin `9f806b36`).

## Result

- **The bound stays at 8192.** Under the issue's primary harness the fused/gather ratio crosses 1.0 at an interpolated Dff of 7318 (7148 when the point that does not run the fused kernel is left out), on the same side of 8192 as the earlier estimates (about 8000 in #626, about 7672 in #712). By the issue's decision rule that is a confirmation, not a change.
- **The 8192 point has never measured the fused kernel.** Llama 4 keeps its own `SwitchGLU` (`src/models/llama4.rs`), which calls `gather_qmm` and does not read `MLXCEL_FUSED_MOE` or `MLXCEL_FUSED_MOE_MAX_DFF`. Both arms of the llama-4-scout row run the same code; on this host neither compiled a fused MoE kernel. The file is byte-identical at the commits of the #626 sweep (`37931d7f`) and the #712 re-validation (`576bf991`) and has the same `gather_qmm`-only forward there, so "break-even at 8192" in those records compared a path with itself.
- **What the fused-path models show:** a clear win at Dff 768 (qwen3-30b, +28% to +35%), ties at 1792 (lfm2) and 6400 (phi-3.5-moe), and a clear loss at 14336 (mixtral, -1.8% to -2.3% with a null floor of 0.2% to 0.3%). The crossover lies between 6400 and 14336 and this data cannot place it more finely: the 6400 point is a tie in all three passes, and the three interpolations are 7148, 9239 and 10668.
- **The earlier lfm2 advantage does not reproduce.** +12% (#626), +14.2% (#712) and +6.6% (2026-10-22, `generate`) are all ties here, in both harnesses.
- **Neither prompt length nor decode length explains the qwen3-30b discrepancy.** All four `generate` cells and all three `bench-decode` passes agree on +23% to +35%. The outlier is the #626 figure (+0.4%, MLX pin `e9463bb`).

## Host

```
host: spark-d9e2 (DGX Spark; not a CI runner, no shared workload)
kernel: 7.0.0-1019-nvidia
nvidia_driver: 580.178.04
gpu: NVIDIA GB10, compute capability 12.1
MLX_CUDA_ARCHITECTURES: 121
cuda_toolkit: 13.0 (V13.0.88)
mlx_pin: 9f806b3606eaa9f7c807238f6bd465185cd94d6d
mem_total_gib: 121
base commit: 28d884a5f2bf1750138b1c99d99c716274aedf1d (main, clean worktree)
binary profile: release (`cargo build --release --features cuda --bin mlxcel-bench-decode --bin mlxcel`, rustc 1.97.1)
binary_sha256 mlxcel-bench-decode: 9a25f444f5947c144aed64135cf0da7b7c3ee385700cbe15bdfe7860de0c08b8
binary_sha256 mlxcel:              811c54de7fb91798fb7307bd2c66917c4d8e40b9595ba8c10af2e1c061f043ee
cudnn: 9.28.1.4 (pip wheel), openblas 0.3.30 (user-space build)
cpu: 10 cores at cpu_capacity 997-1024 and 10 at 718-731, governor performance
```

`gpu-lock`: this host has no upstream `gpu-lock`. A user-local `flock(1)` shim with the same `run --tag <tag> -- <cmd>` shape serialized every run. It is a serialization aid only.

## Checkpoints

`/home/inureyes/models` does not exist on this host, so the five checkpoints were downloaded from the public `mlx-community` repositories at pinned revisions. Whether these are the same bytes as the 2026-07 and 2026-10-22 checkpoints is not known.

| label | repository | revision | model_type | experts / top-k / Dff | quantization |
|---|---|---|---|---|---|
| qwen3-30b-a3b-4bit | `mlx-community/Qwen3-30B-A3B-4bit` | `d388dead15` | qwen3_moe | 128 / 8 / 768 | 4 bit, group 64 |
| lfm2-8b-a1b-4bit | `mlx-community/LFM2-8B-A1B-4bit` | `3266a85e9d` | lfm2_moe | 32 / 4 / 1792 | 4 bit, group 64 |
| phi-3.5-moe-4bit | `mlx-community/Phi-3.5-MoE-instruct-4bit` | `c55613aaad` | phimoe | 16 / 2 / 6400 | 4 bit, group 64 |
| llama-4-scout-17b-4bit | `mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit` | `f89d3ebf0b` | llama4 | 16 / 1 / 8192 | 4 bit, group 64 |
| mixtral-8x7b-4bit | `mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit` | `4e8f70450c` | mixtral | 8 / 2 / 14336 | 4 bit, group 64 |

## Method

Driver: [`scripts/bench_fused_moe_dff.py`](../../scripts/bench_fused_moe_dff.py); tables: [`scripts/bench_fused_moe_dff_report.py`](../../scripts/bench_fused_moe_dff_report.py); rows: [`data/fused-moe-dff-crossover-gb10-2026-11-11/runs.csv`](data/fused-moe-dff-crossover-gb10-2026-11-11/runs.csv) (450 rows: 384 measured, 66 warm-up).

- **Arms.** fused (`MLXCEL_FUSED_MOE_MAX_DFF=20000`), gather (`MLXCEL_FUSED_MOE=0`), null (fused again).
- **Rounds.** One discarded warm-up run per arm, then n rounds with the arm order rotating every round. One process per run, every run under `gpu-lock run`.
- **Primary harness.** `mlxcel-bench-decode -m <model> -p "Hello, how are you today?" -n 100 --warmup-tokens 20`, n = 5, the #712 command.
- **Full-window passes.** The same command with `--ignore-eos`, n = 7, because the plain command stopped at EOS after 33 to 76 tokens on four of the five checkpoints. One more pass pinned every run to the ten faster cores (`taskset -c 5-9,15-19`), to test whether core class explains the run-to-run spread on the fast models.
- **Secondary harness.** `mlxcel generate -m <model> -p <prompt> -n <N> --temp 0`, n = 5, for lfm2 and qwen3-30b, over two prompts (a fixed 439-word prompt and "Hello, how are you today?") and two lengths (256 and 100). The text of the 2026-10-22 record's 439-word prompt was never committed, so this uses its own: [`tests/fixtures/bench/fused-moe-dff-prompt-439-words.txt`](../../tests/fixtures/bench/fused-moe-dff-prompt-439-words.txt).
- **Host discipline.** Before every run the driver waits until no `cargo`, `rustc`, `nvcc`, `cc1plus` or `ptxas` process exists. No run had to wait. `/proc/loadavg` is logged per row and ranged from 0.77 to 2.77.
- **JIT.** A measured run that prints `Compiling CUDA kernels` is discarded and rerun. None did: the warm-up runs populated `MLX_PTX_CACHE_DIR`, and all 384 measured runs were cache hits.
- **Verdict.** Paired deltas are per-round ratios against the fused arm of the same round. A point is a win or a loss only if the paired fused-over-gather range excludes zero and its median exceeds the null floor (the largest absolute paired null delta); otherwise it is a tie.

## Arm attribution

Each arm was run once on an empty `MLX_PTX_CACHE_DIR`, and the kernels it compiled were listed.

| model | arm | modules compiled | fused MoE kernels (`moe_gateup_kernel`, `moe_down_kernel`) |
|---|---|---|---|
| lfm2-8b-a1b-4bit | fused | 30 files | 6 files |
| lfm2-8b-a1b-4bit | gather | 24 files | 0 |
| llama-4-scout-17b-4bit | fused | 21 files (7 modules) | 0 |
| llama-4-scout-17b-4bit | gather | 21 files (7 modules) | 0 |

The lfm2 arms differ as intended. The two llama-4-scout arms compiled the same seven modules and no fused MoE kernel. The shared cache of the whole session agrees: it holds `moe_gateup_kernel` entries for Dff 768, 1792, 6400 and 14336 and none for 8192.

The reason is in the source. `fused_moe_max_dff` is read by `SwitchGLU::forward_fused_kernel` in `src/models/switch_layers.rs`, which the families listed in that function's doc comment call. Llama 4 is not one of them: `src/models/llama4.rs` defines its own `SwitchLinear` and `SwitchGLU`, whose forward is `gather_qmm` for gate, up and down with a compiled SwiGLU activation between them. The llama-4-scout rows below are kept as measured, as a second null arm; they are not a fused-kernel data point.

## 1. Primary harness as specified

| model (Dff) | cell | arm | n | median tok/s | range | generated tokens | per run |
|---|---|---|---|---|---|---|---|
| qwen3-30b-a3b-4bit (768) | hello-n100 | fused | 5 | 73.42 | 71.44-80.55 | 100 | 71.44 80.55 73.42 75.74 72.20 |
| qwen3-30b-a3b-4bit (768) | hello-n100 | gather | 5 | 57.19 | 50.45-57.74 | 100 | 57.48 50.45 57.19 53.43 57.74 |
| qwen3-30b-a3b-4bit (768) | hello-n100 | null | 5 | 78.15 | 71.31-85.11 | 100 | 85.11 71.31 78.15 74.90 79.42 |
| lfm2-8b-a1b-4bit (1792) | hello-n100 | fused | 5 | 142.13 | 140.47-144.95 | 40 | 140.92 144.95 143.54 142.13 140.47 |
| lfm2-8b-a1b-4bit (1792) | hello-n100 | gather | 5 | 140.66 | 128.13-146.76 | 40 | 128.13 141.27 138.93 146.76 140.66 |
| lfm2-8b-a1b-4bit (1792) | hello-n100 | null | 5 | 143.62 | 127.23-149.27 | 40 | 138.48 143.62 127.23 149.27 145.92 |
| phi-3.5-moe-4bit (6400) | hello-n100 | fused | 5 | 43.43 | 43.16-44.54 | 33 | 43.35 43.16 44.54 43.43 43.50 |
| phi-3.5-moe-4bit (6400) | hello-n100 | gather | 5 | 43.24 | 42.19-44.36 | 33 | 43.26 43.23 43.24 42.19 44.36 |
| phi-3.5-moe-4bit (6400) | hello-n100 | null | 5 | 43.44 | 42.48-44.36 | 33 | 43.44 44.36 43.62 42.48 42.91 |
| llama-4-scout-17b-4bit (8192) | hello-n100 | fused | 5 | 20.14 | 20.04-20.22 | 54 | 20.04 20.22 20.14 20.19 20.07 |
| llama-4-scout-17b-4bit (8192) | hello-n100 | gather | 5 | 20.18 | 20.12-20.23 | 54 | 20.23 20.12 20.18 20.20 20.13 |
| llama-4-scout-17b-4bit (8192) | hello-n100 | null | 5 | 20.17 | 20.04-20.25 | 54 | 20.09 20.20 20.17 20.25 20.04 |
| mixtral-8x7b-4bit (14336) | hello-n100 | fused | 5 | 26.49 | 26.46-26.51 | 73 | 26.47 26.49 26.46 26.49 26.51 |
| mixtral-8x7b-4bit (14336) | hello-n100 | gather | 5 | 27.05 | 27.01-27.13 | 76 | 27.01 27.03 27.07 27.13 27.05 |
| mixtral-8x7b-4bit (14336) | hello-n100 | null | 5 | 26.47 | 26.45-26.50 | 73 | 26.50 26.47 26.46 26.47 26.45 |

| model (Dff) | cell | gather vs fused, median (range) | null vs fused, median (range) | fused over gather, median (range) | null floor | verdict | non-ok rows |
|---|---|---|---|---|---|---|---|
| qwen3-30b-a3b-4bit (768) | hello-n100 | -22.11% (-37.37% to -19.54%) | +6.44% (-11.47% to +19.13%) | +28.38% (+24.29% to +59.66%) | 19.13% | win | 3 warmup |
| lfm2-8b-a1b-4bit (1792) | hello-n100 | -2.54% (-9.08% to +3.26%) | -0.92% (-11.36% to +5.02%) | +2.60% (-3.15% to +9.98%) | 11.36% | tie | 3 warmup |
| phi-3.5-moe-4bit (6400) | hello-n100 | -0.21% (-2.92% to +1.98%) | -1.36% (-2.19% to +2.78%) | +0.21% (-1.94% to +3.01%) | 2.78% | tie | 3 warmup |
| llama-4-scout-17b-4bit (8192) | hello-n100 | +0.20% (-0.49% to +0.95%) | +0.15% (-0.15% to +0.30%) | -0.20% (-0.94% to +0.50%) | 0.30% | tie | 3 warmup |
| mixtral-8x7b-4bit (14336) | hello-n100 | +2.04% (+2.04% to +2.42%) | -0.08% (-0.23% to +0.11%) | -2.00% (-2.36% to -2.00%) | 0.23% | loss | 3 warmup |

Only the qwen3-30b row generated the full 100 tokens. lfm2 stopped at 40, phi-3.5-moe at 33, llama-4-scout at 54 and mixtral at 73 (fused) or 76 (gather): the two paths do not emit the same token stream on mixtral, so that row's arms are not measured over the same tokens.

| Dff | model | fused/gather (median of paired ratios) | verdict | fused kernel runs |
|---|---|---|---|---|
| 768 | qwen3-30b-a3b-4bit | 1.2838 | win | yes |
| 1792 | lfm2-8b-a1b-4bit | 1.0260 | tie | yes |
| 6400 | phi-3.5-moe-4bit | 1.0021 | tie | yes |
| 8192 | llama-4-scout-17b-4bit | 0.9980 | tie | no |
| 14336 | mixtral-8x7b-4bit | 0.9800 | loss | yes |

Interpolated crossing: 7318 between 6400 and 8192 with every point, 7148 between 6400 and 14336 with the fused-path models only.

## 2. Full 100-token windows

`--ignore-eos`, n = 7:

| model (Dff) | cell | arm | n | median tok/s | range | generated tokens | per run |
|---|---|---|---|---|---|---|---|
| qwen3-30b-a3b-4bit (768) | hello-n100-ignore-eos | fused | 7 | 75.84 | 70.77-77.34 | 100 | 75.84 77.34 70.77 76.30 76.15 71.76 71.21 |
| qwen3-30b-a3b-4bit (768) | hello-n100-ignore-eos | gather | 7 | 56.54 | 53.26-59.15 | 100 | 55.50 56.54 57.57 53.83 57.31 53.26 59.15 |
| qwen3-30b-a3b-4bit (768) | hello-n100-ignore-eos | null | 7 | 75.46 | 66.75-79.68 | 100 | 75.46 79.10 79.68 73.05 66.75 75.96 73.76 |
| lfm2-8b-a1b-4bit (1792) | hello-n100-ignore-eos | fused | 7 | 147.78 | 142.75-158.63 | 100 | 147.78 143.46 152.94 155.17 142.75 158.63 147.66 |
| lfm2-8b-a1b-4bit (1792) | hello-n100-ignore-eos | gather | 7 | 151.16 | 147.30-152.78 | 100 | 151.66 149.78 152.78 147.30 151.16 151.66 149.04 |
| lfm2-8b-a1b-4bit (1792) | hello-n100-ignore-eos | null | 7 | 153.49 | 142.41-156.45 | 100 | 155.21 156.45 153.10 155.11 153.49 142.41 148.50 |
| phi-3.5-moe-4bit (6400) | hello-n100-ignore-eos | fused | 7 | 46.59 | 45.01-48.09 | 100 | 45.01 47.28 45.29 46.59 46.45 47.63 48.09 |
| phi-3.5-moe-4bit (6400) | hello-n100-ignore-eos | gather | 7 | 46.00 | 42.11-46.75 | 100 | 45.05 42.11 46.75 46.00 46.26 45.27 46.61 |
| phi-3.5-moe-4bit (6400) | hello-n100-ignore-eos | null | 7 | 45.64 | 43.76-47.48 | 100 | 43.84 43.76 47.48 46.02 45.64 45.59 46.82 |
| llama-4-scout-17b-4bit (8192) | hello-n100-ignore-eos | fused | 7 | 20.95 | 20.88-21.00 | 100 | 21.00 20.88 20.95 20.97 20.91 20.92 20.96 |
| llama-4-scout-17b-4bit (8192) | hello-n100-ignore-eos | gather | 7 | 20.95 | 20.82-21.03 | 100 | 20.95 20.96 20.94 20.82 20.93 21.03 21.03 |
| llama-4-scout-17b-4bit (8192) | hello-n100-ignore-eos | null | 7 | 20.96 | 20.93-21.01 | 100 | 20.94 20.93 21.01 20.99 20.96 21.00 20.95 |
| mixtral-8x7b-4bit (14336) | hello-n100-ignore-eos | fused | 7 | 27.16 | 27.13-27.18 | 100 | 27.16 27.17 27.13 27.16 27.16 27.14 27.18 |
| mixtral-8x7b-4bit (14336) | hello-n100-ignore-eos | gather | 7 | 27.80 | 27.74-27.95 | 100 | 27.80 27.95 27.74 27.95 27.78 27.89 27.75 |
| mixtral-8x7b-4bit (14336) | hello-n100-ignore-eos | null | 7 | 27.15 | 27.09-27.17 | 100 | 27.17 27.16 27.16 27.14 27.15 27.15 27.09 |

| model (Dff) | cell | gather vs fused, median (range) | null vs fused, median (range) | fused over gather, median (range) | null floor | verdict | non-ok rows |
|---|---|---|---|---|---|---|---|
| qwen3-30b-a3b-4bit (768) | hello-n100-ignore-eos | -25.78% (-29.45% to -16.94%) | +2.28% (-12.34% to +12.59%) | +34.74% (+20.39% to +41.74%) | 12.59% | win | 3 warmup |
| lfm2-8b-a1b-4bit (1792) | hello-n100-ignore-eos | +0.93% (-5.07% to +5.89%) | +0.57% (-10.23% to +9.05%) | -0.93% (-5.56% to +5.34%) | 10.23% | tie | 3 warmup |
| phi-3.5-moe-4bit (6400) | hello-n100-ignore-eos | -1.27% (-10.93% to +3.22%) | -2.60% (-7.45% to +4.84%) | +1.28% (-3.12% to +12.28%) | 7.45% | tie | 3 warmup |
| llama-4-scout-17b-4bit (8192) | hello-n100-ignore-eos | +0.10% (-0.72% to +0.53%) | +0.24% (-0.29% to +0.38%) | -0.10% (-0.52% to +0.72%) | 0.38% | tie | 3 warmup |
| mixtral-8x7b-4bit (14336) | hello-n100-ignore-eos | +2.36% (+2.10% to +2.91%) | -0.04% (-0.33% to +0.11%) | -2.30% (-2.83% to -2.05%) | 0.33% | loss | 3 warmup |

| Dff | model | fused/gather (median of paired ratios) | verdict | fused kernel runs |
|---|---|---|---|---|
| 768 | qwen3-30b-a3b-4bit | 1.3474 | win | yes |
| 1792 | lfm2-8b-a1b-4bit | 0.9907 | tie | yes |
| 6400 | phi-3.5-moe-4bit | 1.0128 | tie | yes |
| 8192 | llama-4-scout-17b-4bit | 0.9990 | tie | no |
| 14336 | mixtral-8x7b-4bit | 0.9770 | loss | yes |

Pinned to the faster cores, n = 7 (llama-4-scout not repeated):

| model (Dff) | cell | arm | n | median tok/s | range | generated tokens | per run |
|---|---|---|---|---|---|---|---|
| qwen3-30b-a3b-4bit (768) | hello-n100-ignore-eos-pinned | fused | 7 | 71.93 | 70.86-78.66 | 100 | 78.66 71.65 71.93 76.92 71.62 74.39 70.86 |
| qwen3-30b-a3b-4bit (768) | hello-n100-ignore-eos-pinned | gather | 7 | 55.89 | 49.01-59.64 | 100 | 58.64 59.64 55.89 58.52 52.86 49.01 53.91 |
| qwen3-30b-a3b-4bit (768) | hello-n100-ignore-eos-pinned | null | 7 | 73.44 | 68.77-81.79 | 100 | 76.27 81.79 73.22 70.98 79.33 73.44 68.77 |
| lfm2-8b-a1b-4bit (1792) | hello-n100-ignore-eos-pinned | fused | 7 | 148.34 | 135.97-152.46 | 100 | 151.49 146.67 135.97 148.34 152.46 148.34 141.48 |
| lfm2-8b-a1b-4bit (1792) | hello-n100-ignore-eos-pinned | gather | 7 | 149.36 | 146.54-155.18 | 100 | 155.18 150.43 149.36 150.80 148.99 146.54 148.28 |
| lfm2-8b-a1b-4bit (1792) | hello-n100-ignore-eos-pinned | null | 7 | 149.98 | 137.60-156.71 | 100 | 149.98 156.71 152.19 152.69 149.61 143.79 137.60 |
| phi-3.5-moe-4bit (6400) | hello-n100-ignore-eos-pinned | fused | 7 | 44.71 | 43.69-45.38 | 100 | 45.03 44.14 43.69 45.38 43.83 44.71 45.30 |
| phi-3.5-moe-4bit (6400) | hello-n100-ignore-eos-pinned | gather | 7 | 44.24 | 39.25-45.86 | 100 | 45.86 42.58 42.43 45.47 39.25 44.49 44.24 |
| phi-3.5-moe-4bit (6400) | hello-n100-ignore-eos-pinned | null | 7 | 44.35 | 42.60-45.74 | 100 | 44.35 44.38 42.60 43.68 45.60 43.43 45.74 |
| mixtral-8x7b-4bit (14336) | hello-n100-ignore-eos-pinned | fused | 7 | 27.10 | 27.05-27.13 | 100 | 27.09 27.05 27.10 27.09 27.11 27.13 27.11 |
| mixtral-8x7b-4bit (14336) | hello-n100-ignore-eos-pinned | gather | 7 | 27.59 | 27.43-27.70 | 100 | 27.59 27.65 27.56 27.70 27.68 27.43 27.55 |
| mixtral-8x7b-4bit (14336) | hello-n100-ignore-eos-pinned | null | 7 | 27.11 | 27.07-27.14 | 100 | 27.11 27.14 27.13 27.09 27.13 27.07 27.10 |

| model (Dff) | cell | gather vs fused, median (range) | null vs fused, median (range) | fused over gather, median (range) | null floor | verdict | non-ok rows |
|---|---|---|---|---|---|---|---|
| qwen3-30b-a3b-4bit (768) | hello-n100-ignore-eos-pinned | -23.92% (-34.12% to -16.76%) | -1.28% (-7.72% to +14.15%) | +31.44% (+20.14% to +51.79%) | 14.15% | win | 3 warmup |
| lfm2-8b-a1b-4bit (1792) | hello-n100-ignore-eos-pinned | +2.44% (-2.28% to +9.85%) | -1.00% (-3.07% to +11.93%) | -2.38% (-8.96% to +2.33%) | 11.93% | tie | 3 warmup |
| phi-3.5-moe-4bit (6400) | hello-n100-ignore-eos-pinned | -2.34% (-10.45% to +1.84%) | -1.51% (-3.75% to +4.04%) | +2.40% (-1.81% to +11.67%) | 4.04% | tie | 3 warmup |
| mixtral-8x7b-4bit (14336) | hello-n100-ignore-eos-pinned | +1.85% (+1.11% to +2.25%) | +0.07% (-0.22% to +0.33%) | -1.81% (-2.20% to -1.09%) | 0.33% | loss | 3 warmup |

| Dff | model | fused/gather (median of paired ratios) | verdict | fused kernel runs |
|---|---|---|---|---|
| 768 | qwen3-30b-a3b-4bit | 1.3144 | win | yes |
| 1792 | lfm2-8b-a1b-4bit | 0.9762 | tie | yes |
| 6400 | phi-3.5-moe-4bit | 1.0240 | tie | yes |
| 14336 | mixtral-8x7b-4bit | 0.9819 | loss | yes |

The verdicts do not change with the window or the pinning: win at 768, ties at 1792 and 6400, loss at 14336. The interpolation between 6400 and 14336 moves to 9239 and 10668, because the 6400 median moves from +0.2% to +1.3% and +2.4% while staying inside its null floor each time.

The null floor is the limit of this record. It is 0.2% to 0.4% on mixtral and llama-4-scout and 2.8% to 19% on the three fast models. Pinning to the faster cores did not reduce it (lfm2 10.2% unpinned, 11.9% pinned; qwen3-30b 12.6% and 14.2%), so core class is not the cause. The cause was not found.

## 3. Decision

The issue's rule: move the bound only if the crossover moves to a different side of 8192, or past 14336; otherwise keep 8192 and record the confirmation.

- Under the primary harness as specified, the crossover is 7318 (7148 without llama-4-scout), below 8192 like the two earlier estimates. **Keep 8192.**
- The rule's fallback, "the largest measured Dff that still wins", would be 768 here, because 1792 and 6400 are ties. It does not apply, and applying it would withdraw the fused path from shapes where it is not measurably slower.
- What the confirmation rests on is weaker than the earlier records implied. No fused-path model has been measured between Dff 6400 and 14336, the 6400 point is a tie, and the only clear loss is at 14336. A fused-path checkpoint with Dff near 8192, or a synthetic expert of that width, would be needed to say more.

## 4. Secondary harness: `generate`, prompt length and decode length

| model (Dff) | cell | arm | n | median tok/s | range | generated tokens | per run |
|---|---|---|---|---|---|---|---|
| qwen3-30b-a3b-4bit (768) | hello-n100 | fused | 5 | 69.25 | 67.52-77.45 | 100 | 77.45 67.73 67.52 70.06 69.25 |
| qwen3-30b-a3b-4bit (768) | hello-n100 | gather | 5 | 54.86 | 52.88-58.53 | 100 | 53.83 54.86 55.11 58.53 52.88 |
| qwen3-30b-a3b-4bit (768) | hello-n100 | null | 5 | 70.77 | 64.51-73.48 | 100 | 72.23 73.48 69.64 64.51 70.77 |
| qwen3-30b-a3b-4bit (768) | hello-n256 | fused | 5 | 72.67 | 67.41-76.45 | 198 | 72.67 67.41 71.95 76.45 75.96 |
| qwen3-30b-a3b-4bit (768) | hello-n256 | gather | 5 | 56.66 | 52.41-58.08 | 189 | 58.08 56.66 55.41 52.41 56.67 |
| qwen3-30b-a3b-4bit (768) | hello-n256 | null | 5 | 71.82 | 69.33-77.85 | 198 | 77.85 69.33 70.35 71.82 76.69 |
| qwen3-30b-a3b-4bit (768) | long439-n100 | fused | 5 | 44.95 | 43.61-46.08 | 100 | 44.95 43.61 46.08 44.84 45.07 |
| qwen3-30b-a3b-4bit (768) | long439-n100 | gather | 5 | 36.62 | 34.75-37.32 | 100 | 34.88 36.62 37.32 34.75 36.90 |
| qwen3-30b-a3b-4bit (768) | long439-n100 | null | 5 | 43.62 | 41.04-43.86 | 100 | 43.76 43.86 42.45 41.04 43.62 |
| qwen3-30b-a3b-4bit (768) | long439-n256 | fused | 5 | 56.99 | 54.76-59.99 | 256 | 56.49 59.06 54.76 56.99 59.99 |
| qwen3-30b-a3b-4bit (768) | long439-n256 | gather | 5 | 46.27 | 43.71-47.09 | 256 | 46.44 46.26 43.71 46.27 47.09 |
| qwen3-30b-a3b-4bit (768) | long439-n256 | null | 5 | 58.68 | 58.14-58.90 | 256 | 58.90 58.57 58.68 58.70 58.14 |
| lfm2-8b-a1b-4bit (1792) | hello-n100 | fused | 5 | 92.02 | 86.91-93.43 | 40 | 93.43 86.91 92.34 92.02 90.34 |
| lfm2-8b-a1b-4bit (1792) | hello-n100 | gather | 5 | 92.34 | 91.12-92.92 | 40 | 92.52 92.05 91.12 92.92 92.34 |
| lfm2-8b-a1b-4bit (1792) | hello-n100 | null | 5 | 90.11 | 88.37-93.92 | 40 | 93.92 88.37 90.66 89.03 90.11 |
| lfm2-8b-a1b-4bit (1792) | hello-n256 | fused | 5 | 88.79 | 87.55-90.93 | 40 | 87.82 90.93 88.88 88.79 87.55 |
| lfm2-8b-a1b-4bit (1792) | hello-n256 | gather | 5 | 93.25 | 90.45-94.37 | 40 | 93.29 94.37 93.25 90.45 91.93 |
| lfm2-8b-a1b-4bit (1792) | hello-n256 | null | 5 | 91.61 | 87.98-93.97 | 40 | 91.61 93.43 89.52 87.98 93.97 |
| lfm2-8b-a1b-4bit (1792) | long439-n100 | fused | 5 | 78.45 | 75.43-79.01 | 100 | 78.68 76.66 79.01 78.45 75.43 |
| lfm2-8b-a1b-4bit (1792) | long439-n100 | gather | 5 | 78.93 | 77.29-80.57 | 100 | 77.29 80.57 78.93 80.38 78.10 |
| lfm2-8b-a1b-4bit (1792) | long439-n100 | null | 5 | 78.85 | 76.64-80.28 | 100 | 80.07 80.28 78.54 76.64 78.85 |
| lfm2-8b-a1b-4bit (1792) | long439-n256 | fused | 5 | 109.44 | 108.36-112.46 | 256 | 112.46 108.36 109.31 111.15 109.44 |
| lfm2-8b-a1b-4bit (1792) | long439-n256 | gather | 5 | 109.02 | 107.55-111.18 | 256 | 108.65 111.18 111.18 109.02 107.55 |
| lfm2-8b-a1b-4bit (1792) | long439-n256 | null | 5 | 110.99 | 107.94-115.51 | 256 | 115.51 107.94 110.99 110.85 113.11 |

| model (Dff) | cell | gather vs fused, median (range) | null vs fused, median (range) | fused over gather, median (range) | null floor | verdict | non-ok rows |
|---|---|---|---|---|---|---|---|
| qwen3-30b-a3b-4bit (768) | hello-n100 | -19.00% (-30.50% to -16.46%) | +2.19% (-7.92% to +8.49%) | +23.46% (+19.70% to +43.88%) | 8.49% | win | 3 warmup |
| qwen3-30b-a3b-4bit (768) | hello-n256 | -22.99% (-31.45% to -15.95%) | +0.96% (-6.06% to +7.13%) | +29.85% (+18.97% to +45.87%) | 7.13% | win | 3 warmup |
| qwen3-30b-a3b-4bit (768) | long439-n100 | -19.01% (-22.50% to -16.03%) | -3.22% (-8.47% to +0.57%) | +23.47% (+19.09% to +29.04%) | 8.47% | win | 3 warmup |
| qwen3-30b-a3b-4bit (768) | long439-n256 | -20.18% (-21.67% to -17.79%) | +3.00% (-3.08% to +7.16%) | +25.28% (+21.64% to +27.67%) | 7.16% | win | 3 warmup |
| lfm2-8b-a1b-4bit (1792) | hello-n100 | +0.98% (-1.32% to +5.91%) | -0.25% (-3.25% to +1.68%) | -0.97% (-5.58% to +1.34%) | 3.25% | tie | 3 warmup |
| lfm2-8b-a1b-4bit (1792) | hello-n256 | +4.92% (+1.87% to +6.23%) | +2.75% (-0.91% to +7.33%) | -4.69% (-5.86% to -1.84%) | 7.33% | tie | 3 warmup |
| lfm2-8b-a1b-4bit (1792) | long439-n100 | +2.46% (-1.77% to +5.10%) | +1.77% (-2.31% to +4.72%) | -2.40% (-4.85% to +1.80%) | 4.72% | tie | 3 warmup |
| lfm2-8b-a1b-4bit (1792) | long439-n256 | -1.73% (-3.39% to +2.60%) | +1.54% (-0.39% to +3.35%) | +1.76% (-2.54% to +3.51%) | 3.35% | tie | 3 warmup |

lfm2 answers "Hello, how are you today?" in 40 tokens, so its two `hello` cells measure the same 40-token window.

For comparison, the 2026-10-22 record (`generate`, its own 439-word prompt, `-n 256`, test-fast profile, commit `b955bbec`, MLX pin `bff10220`) reported, on the fixed binary:

| model | 2026-10-22 fused | 2026-10-22 gather | here, `long439-n256` fused | here, `long439-n256` gather |
|---|---|---|---|---|
| qwen3-30b | 56.20 | 46.56 | 56.99 | 46.27 |
| lfm2 | 109.65 | 102.25 | 109.44 | 109.02 |

Three of the four cells reproduce within 2%. The lfm2 gather arm is the one that moved, from 102.25 to 109.02 tok/s, which is what turns the earlier +6.6% into a tie. Which change made that path faster was not bisected.

On the discrepancy the issue asks about:

- **qwen3-30b, +20.8% (`generate`, 2026-10-22) against +0.4% (`bench-decode`, #626).** Not prompt length and not decode length: the four cells here give +23.5% to +29.9%, and `bench-decode` gives +28.4%, +34.7% and +31.4%. The two harnesses agree today. The #626 figure was taken on MLX pin `e9463bb`, where both arms ran at about 91 tok/s; it does not reproduce on the current pin.
- **lfm2, +6.6% (`generate`, 2026-10-22) against +14.2% (`bench-decode`, #712).** Not prompt length and not decode length: all four cells are ties (-4.7% to +1.8%), and so are the three `bench-decode` passes (+2.6%, -0.9%, -2.4%). Neither earlier figure reproduces. #626 and #712 had three runs per side and no null arm; with a null floor of 10% on this model here, a 12% to 14% difference between two three-run medians is within what the null arm alone produces.

## What this does not cover

- Dff 8192 on a fused-path model. The only 8192 checkpoint in the sweep does not use the fused kernel.
- Why the fast models have a 2.8% to 19% run-to-run spread. Core class is ruled out; GPU clocking, allocator state and the short decode window (0.3 s to 2 s) were not separated.
- Absolute tok/s against the earlier records. This is a different DGX Spark with separately downloaded checkpoints; only the ratios are compared.
- Which commit or pin change sped up the lfm2 `gather_qmm` path since 2026-10-22, and why qwen3-30b's two arms no longer coincide as they did under pin `e9463bb`.
- The Metal bound (4096), the ROCm bound, gemma4 and qwen3_next (own kernels), and batched decode (#2515).

## Reproduction

```sh
cargo build --release --features cuda --bin mlxcel-bench-decode --bin mlxcel
OUT=bench-results/fused-moe-dff
for m in lfm2-8b-a1b-4bit qwen3-30b-a3b-4bit phi-3.5-moe-4bit llama-4-scout-17b-4bit mixtral-8x7b-4bit; do
  python3 scripts/bench_fused_moe_dff.py --harness primary --bin-dir target/release \
      --model "$MODELS/$m" --name "$m" --cell hello-n100 --rounds 5 --out "$OUT"
  python3 scripts/bench_fused_moe_dff.py --harness primary --bin-dir target/release \
      --model "$MODELS/$m" --name "$m" --cell hello-n100-ignore-eos --ignore-eos --rounds 7 --out "$OUT"
done
for m in lfm2-8b-a1b-4bit qwen3-30b-a3b-4bit; do
  python3 scripts/bench_fused_moe_dff.py --harness secondary --bin-dir target/release \
      --model "$MODELS/$m" --name "$m" --cell long439-n256 --max-tokens 256 --rounds 5 --out "$OUT" \
      --prompt-file tests/fixtures/bench/fused-moe-dff-prompt-439-words.txt
done
python3 scripts/bench_fused_moe_dff_report.py "$OUT/runs.csv"
```

The other secondary cells swap `--prompt-file` for the default prompt and `--max-tokens 256` for `100`; the pinned pass adds `--cpu-list 5-9,15-19`.
