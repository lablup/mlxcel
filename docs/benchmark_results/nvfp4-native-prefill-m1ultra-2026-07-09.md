# Native NVFP4 prefill recovery and default switch on M1 Ultra

This note records the issue #705 follow-up to the issue #694 Apple Silicon A/B. The build includes the MLX 0.32.1 pin from PR #704, a shape-specific native-NVFP4 scaled MLP prefill graph, and the direct ModelOpt-triplet transcode as the default on Metal/non-CUDA. The affine fallback remains available with `MLXCEL_NVFP4_DENSE_REPACK=1`.

| Item | Value |
|------|-------|
| Date | 2026-07-09 |
| Host | M1 Ultra, 128 GB unified memory |
| Model | `models/gemma-4-31b-it-nvfp4` |
| Build | `cargo build --release --bin mlxcel-bench-decode` |
| Harness | `target/release/mlxcel-bench-decode` |
| Prompt mode | `--prompt-tokens N`, `-n 32`, `--warmup-tokens 20` |
| Native path | default direct ModelOpt triplet -> MLX native NVFP4 |
| Affine rollback | `MLXCEL_NVFP4_DENSE_REPACK=1` |

## Validation scope

Issue #705 followed issue #694 / PR #702, whose local reproducible ModelOpt NVFP4 checkpoint is `models/gemma-4-31b-it-nvfp4`. Earlier issue wording that named Llama 4 Scout was stale; the sweep below is not a reduced-size substitute, but the same local Gemma 4 ModelOpt NVFP4 validation target used for the prior native-vs-affine Metal comparison.

## What changed

Native ModelOpt NVFP4 keeps `weight_scale_2` as a per-linear `global_scale` sidecar. The previous dense MLP gate avoided the sidecar fused helper for multi-token prefill because the helper's eager fallback was slower than the op-at-a-time activation path. Issue #705 adds a shape-specific compiled scaled MLP graph for the exact native prefill layout (`group_size=16`, `bits=4`, `mode=nvfp4`). That keeps the native sidecar folds and GeGLU activation in one graph without using the decode-oriented shapeless qmm graph. `UnifiedLinear` also has a sidecar-aware C++ helper so standalone native-NVFP4 linears can apply qmm, global scale, and dense bias through one FFI boundary while preserving `apply_global_scale` semantics.

## Results

| Prompt tokens | Affine prefill tok/s | Native prefill tok/s | Native prefill gap | Affine decode tok/s | Native decode tok/s | Native decode speedup | Affine peak | Native peak | Peak delta | Affine load | Native load |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 123.38 | 118.49 | -4.0% | 4.93 | 13.15 | 2.67x | 40.44 GB | 37.17 GB | -3.27 GB | 222.3s | 82.3s |
| 1024 | 126.71 | 117.46 | -7.3% | 4.86 | 13.04 | 2.68x | 41.61 GB | 37.17 GB | -4.44 GB | 224.8s | 84.0s |
| 2048 | 126.50 | 117.12 | -7.4% | 4.93 | 13.04 | 2.65x | 44.83 GB | 37.17 GB | -7.66 GB | 224.5s | 84.0s |
| 4096 | 118.07 | 112.88 | -4.4% | 4.64 | 12.78 | 2.75x | 47.76 GB | 37.61 GB | -10.15 GB | 223.9s | 84.3s |
| 8192 | 116.88 | 111.39 | -4.7% | 4.64 | 12.27 | 2.64x | 49.11 GB | 38.50 GB | -10.61 GB | 223.9s | 82.5s |
| 16384 | 115.86 | 108.10 | -6.7% | 4.57 | 11.49 | 2.51x | 52.60 GB | 40.28 GB | -12.32 GB | 224.3s | 82.8s |

## Activation-path follow-up

Because native prefill still trails affine, the same M1 Ultra build also rechecked the issue #701 conclusion that multi-token sidecar prefill should not prefer the C++ eager-fold path over the op-at-a-time compiled GeGLU activation path. The focused A/B used the native direct-transcode load path at 2048 and 8192 prompt tokens:

| Prompt tokens | Path | Environment override | Prefill tok/s | Decode tok/s | Peak memory |
|---:|---|---|---:|---:|---:|
| 2048 | shape-specific scaled MLP (default) | unset | 116.10 | 12.99 | 37.17 GB |
| 2048 | op-at-a-time projections + compiled GeGLU activation | `MLXCEL_DISABLE_FUSED_GLOBAL_SCALE=1` | 116.11 | 13.06 | 37.17 GB |
| 2048 | C++ eager global-scale fold | `MLXCEL_COMPILED_QGELU_MLP=0` | 113.35 | 12.65 | 37.17 GB |
| 8192 | shape-specific scaled MLP (default) | unset | 110.66 | 12.29 | 38.50 GB |
| 8192 | op-at-a-time projections + compiled GeGLU activation | `MLXCEL_DISABLE_FUSED_GLOBAL_SCALE=1` | 109.99 | 12.38 | 37.91 GB |
| 8192 | C++ eager global-scale fold | `MLXCEL_COMPILED_QGELU_MLP=0` | 108.10 | 11.81 | 37.80 GB |

The result does not support switching native prefill to the compiled-activation fallback: it is effectively tied at 2048 and 0.6% slower at 8192. The issue #701 warning about the eager fold still holds on the current Metal/MLX 0.32.1 baseline: disabling the compiled qgelu graph is 2.4% slower at 2048 and 2.3% slower at 8192. The default therefore stays on the scaled compiled MLP path for native NVFP4, with `MLXCEL_DISABLE_FUSED_GLOBAL_SCALE=1` retained as the op-at-a-time A/B switch.

## Decision

The prefill gap is reduced on the mid-length cases that previously showed the largest regression, but it is not eliminated; native remains about 4-7% slower than affine for prefill on this host. The focused activation-path A/B above shows that the #701 compiled-activation fallback is not faster than the current default and that the eager fold remains slower, so the remaining gap is not closed by routing native prefill back to that fallback. The overall runtime trade-off is still strongly in favor of native: cold load is about 2.7x faster, decode is 2.5-2.75x faster, and peak memory is 3.3-12.3 GB lower across the sweep. Based on that net result, Metal/non-CUDA now follows CUDA and uses direct native NVFP4 by default, while `MLXCEL_NVFP4_DENSE_REPACK=1` keeps the dense-affine rollback for prefill-sensitive comparisons.

## Reproduction

```bash
cargo build --release --bin mlxcel-bench-decode

# Native default.
./target/release/mlxcel-bench-decode \
  -m models/gemma-4-31b-it-nvfp4 \
  -p "Hello, how are you today?" \
  --prompt-tokens 2048 \
  -n 32 \
  --warmup-tokens 20

# Affine rollback.
MLXCEL_NVFP4_DENSE_REPACK=1 ./target/release/mlxcel-bench-decode \
  -m models/gemma-4-31b-it-nvfp4 \
  -p "Hello, how are you today?" \
  --prompt-tokens 2048 \
  -n 32 \
  --warmup-tokens 20

# Native op-at-a-time sidecar path with compiled GeGLU activation.
MLXCEL_DISABLE_FUSED_GLOBAL_SCALE=1 ./target/release/mlxcel-bench-decode \
  -m models/gemma-4-31b-it-nvfp4 \
  -p "Hello, how are you today?" \
  --prompt-tokens 2048 \
  -n 32 \
  --warmup-tokens 20

# Native C++ eager-fold fallback for the scaled MLP bridge.
MLXCEL_COMPILED_QGELU_MLP=0 ./target/release/mlxcel-bench-decode \
  -m models/gemma-4-31b-it-nvfp4 \
  -p "Hello, how are you today?" \
  --prompt-tokens 2048 \
  -n 32 \
  --warmup-tokens 20
```
