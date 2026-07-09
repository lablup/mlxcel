# ModelOpt NVFP4 native Metal A/B benchmark (issue #694)

This note records the Apple Silicon A/B requested by issue #694 after the direct ModelOpt triplet to MLX native NVFP4 transcode from issue #693 landed. The test compares the current non-CUDA default, which repacks the local `gemma-4-31b-it-nvfp4` ModelOpt triplets into MLX affine 4-bit weights, against the opt-in direct native NVFP4 path selected by `MLXCEL_NVFP4_NATIVE_REPACK=1`.

## Environment

| Item | Value |
|------|-------|
| Hardware | Mac Studio M1 Ultra, 128 GB unified memory |
| OS | macOS 26.5.2 (25F84) |
| Backend | Metal |
| Model | `models/gemma-4-31b-it-nvfp4` |
| Commit | `9be591346` |
| Build | `cargo build --release --bin mlxcel --bin mlxcel-bench-decode` |
| Harness | `target/release/mlxcel-bench-decode` |
| Test date | 2026-07-09 |
| Prompt | `Hello, how are you today?` (short) and a synthesized 2048-token prompt |
| Raw CSV | `benchmarks/metal_m1ultra_issue694_nvfp4_native_vs_affine_2026-07-09.csv` |

Each row is a separate process. `mlxcel-bench-decode` resets the MLX high-water mark before model load, so `[Load] wall` and the load peak are isolated from the measured prefill/decode pass. The full-run peak includes load, prefill, and decode.

## Commands

Affine fallback, current non-CUDA default:

```bash
./target/release/mlxcel-bench-decode -m models/gemma-4-31b-it-nvfp4 -p "Hello, how are you today?" -n 100 --warmup-tokens 20
./target/release/mlxcel-bench-decode -m models/gemma-4-31b-it-nvfp4 -p "Hello, how are you today?" --prompt-tokens 2048 -n 32 --warmup-tokens 20
```

Direct native NVFP4 opt-in:

```bash
MLXCEL_NVFP4_NATIVE_REPACK=1 ./target/release/mlxcel-bench-decode -m models/gemma-4-31b-it-nvfp4 -p "Hello, how are you today?" -n 100 --warmup-tokens 20
MLXCEL_NVFP4_NATIVE_REPACK=1 ./target/release/mlxcel-bench-decode -m models/gemma-4-31b-it-nvfp4 -p "Hello, how are you today?" --prompt-tokens 2048 -n 32 --warmup-tokens 20
```

## Results

| Run | Repack | Prompt tokens | Cold-load wall (s) | Load peak (GB) | Full-run peak (GB) | Prefill tok/s | Decode tok/s |
|-----|--------|--------------:|-------------------:|---------------:|-------------------:|--------------:|-------------:|
| affine, short | MLX affine fallback | 20 | 281.78 | 37.17 | 39.01 | 30.34 | 4.76 |
| native, short | direct native NVFP4 | 20 | 83.78 | 37.17 | 37.17 | 48.20 | 12.70 |
| affine, 2048 | MLX affine fallback | 2048 | 221.06 | 37.17 | 44.83 | 124.58 | 4.92 |
| native, 2048 | direct native NVFP4 | 2048 | 82.74 | 37.17 | 37.17 | 115.61 | 12.93 |

## Decision gate

Issue #694 says to enable the native path for Metal/non-CUDA only if the 2048-token prompt improves prefill by at least 20% and decode does not regress by more than 5% versus the affine fallback on the same host.

The gate does not pass on this M1 Ultra run. The native path improves 2048-token decode from 4.92 tok/s to 12.93 tok/s and cuts cold-load wall time from 221.06 s to 82.74 s, but 2048-token prefill drops from 124.58 tok/s to 115.61 tok/s, a 7.2% regression rather than the required 20% improvement. Therefore the non-CUDA default remains the affine fallback, and `MLXCEL_NVFP4_NATIVE_REPACK=1` remains the explicit Metal opt-in for further experiments.

## Load and memory observations

The direct path is still useful as an opt-in because it avoids the expensive load-time dense f16 reconstruction and affine requantization. Short-prompt cold load drops from 281.78 s to 83.78 s; the 2048-token run drops from 221.06 s to 82.74 s. Load-time MLX peak is unchanged at 37.17 GB, but full-run peak is lower with the native path because the measured runs never exceed the loaded-model peak, while the affine fallback reaches 39.01 GB on the short prompt and 44.83 GB on the 2048-token prompt.

## Throughput observations

Short-prompt native prefill improves from 30.34 tok/s to 48.20 tok/s, and decode improves from 4.76 tok/s to 12.70 tok/s. The 2048-token path is mixed: native decode improves by 2.63x, but native prefill is 7.2% slower. That prefill miss is the decisive metric for this issue because the switch criterion intentionally requires a long-prompt prefill win before changing the Metal default.

## Greedy continuation spot-check

A 40-token greedy raw continuation spot-check used this command form for both paths:

```bash
./target/release/mlxcel generate -m models/gemma-4-31b-it-nvfp4 --no-chat-template -p "In a quiet laboratory, the engineer noticed" -n 40
```

Affine fallback output:

> In a quiet laboratory, the engineer noticed a subtle shift in the sensor's reading. A single, precise same-same same-same same-same same-same same-same same-same same-same same-same same-

Direct native NVFP4 output:

> In a quiet laboratory, the engineer noticed a subtle but significant change in the data. The signal, which had been steady for hours, suddenly shifted. The shift was not a random fluctuation; it was a a a a a a a a

The two greedy continuations share the same opening and then diverge, which is expected when comparing the faithful direct transcode against the affine fallback's re-quantized weights. The spot-check confirms the native Metal path loads and generates a 40-token greedy continuation without runtime failure; token-identical output against the affine fallback is not expected for this A/B.

## Follow-up

Do not flip the non-CUDA default from `DenseAffine` to `DirectTranscode` based on this run. A future M5 Max or newer-MLX retest can reuse the same opt-in override and gate, especially if native NVFP4 Metal kernels change their long-prompt prefill behavior.
