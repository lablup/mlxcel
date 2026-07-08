# ModelOpt NVFP4 CUDA repack benchmark (issue #630)

This note records the CUDA validation for the issue #630 NVFP4 repair path on
NVIDIA GB10. The local `gemma-4-31b-it-nvfp4` checkpoint is not an MLX affine
4-bit checkpoint: it is NVIDIA ModelOpt NVFP4 metadata with
`quant_method=modelopt`, `quant_algo=NVFP4`, and per-linear
`weight/weight_scale/weight_scale_2` triplets. The loader now accepts only that
narrow metadata shape. CUDA repacks the triplets to MLX native NVFP4 tensors at
load time so long-prompt prefill can use the faster block-float matmul path;
non-CUDA builds keep the affine fallback until Apple Silicon is re-benchmarked.

## Environment

| Item | Value |
|------|-------|
| Hardware | NVIDIA GB10 (DGX Spark), 122 GB unified LPDDR5x |
| Backend | CUDA |
| Build | `cargo build --release --features cuda --bin mlxcel --bin mlxcel-bench-decode` |
| Harness | `target/release/mlxcel-bench-decode` |
| Test date | 2026-07-08 |
| Prompt | `Hello, how are you today?` unless noted |

## Results

| Model / run | CSV | Prompt tokens | Generated | Prefill tok/s | Decode tok/s | Notes |
|-------------|-----|--------------:|----------:|--------------:|-------------:|-------|
| Gemma 4 31B NVFP4 before | `benchmarks/cuda_gb10_2026-06-17.csv` | 20 | 26 | 16.26 | 0.89 | Historical baseline |
| Gemma 4 31B NVFP4 affine stopgap | `benchmarks/cuda_gb10_issue630_nvfp4_short_gs64_2026-07-08.csv` | 20 | 42 | 51.75 | 4.48 | ModelOpt NVFP4 repacked to MLX affine gs64 |
| Gemma 4 31B NVFP4 native CUDA | `benchmarks/cuda_gb10_issue630_nvfp4_native_short_2026-07-08.csv` | 20 | 42 | 75.36 | 5.24 | ModelOpt NVFP4 repacked to MLX native NVFP4 |
| Gemma 4 31B NVFP4 affine stopgap, 2048 prompt | `benchmarks/cuda_gb10_issue630_nvfp4_prefill2048_2026-07-08.csv` | 2048 | 32 | 116.44 | 4.60 | Acceptance prefill length |
| Gemma 4 31B NVFP4 native CUDA, 2048 prompt | `benchmarks/cuda_gb10_issue630_nvfp4_native_prefill2048_2026-07-08.csv` | 2048 | 32 | 392.71 | 5.42 | Final CUDA path for this PR |
| GPT-OSS 20B MXFP4 control | `benchmarks/cuda_gb10_issue630_gptoss_mxfp4_2026-07-08.csv` | 73 | 64 | 95.86 | 70.23 | Spot-check for existing MXFP4 path |
| Gemma 4 31B IT 4-bit reference | `benchmarks/cuda_gb10_2026-07-03.csv` | 20 | 26 | 80.14 | 8.54 | Fully affine 4-bit reference checkpoint |

## Nsight Systems

After-profile command:

```bash
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt \
  -o /tmp/mlxcel_issue630_nvfp4_after \
  target/release/mlxcel-bench-decode \
  -m models/gemma-4-31b-it-nvfp4 \
  -p "Hello, how are you today?" \
  -n 8 --warmup-tokens 2
```

Profile output:

- Report: `/tmp/mlxcel_issue630_nvfp4_after.nsys-rep`
- Stats: `/tmp/mlxcel_issue630_nvfp4_after_stats_cuda_api_sum.csv`
- Short-profile result: 20 prompt tokens, 8 generated tokens, 52.58 prefill
  tok/s, 5.11 decode tok/s, 36.67 GB MLX peak memory
- CUDA API summary is dominated by `cudaEventSynchronize`,
  `cuLibraryLoadData`, `cudaMallocAsync_v11020`,
  `cudaGraphAddKernelNode_v10000`, and `cudaGraphLaunch_v10000`.

The before run is represented by the historical CSV baseline above; no matching
before-change nsys trace was captured for that baseline.

## Acceptance status

- The broken NVFP4 execution path is repaired for the local ModelOpt NVFP4
  Gemma 4 checkpoint: it loads, normalizes the NVFP4-style keys, repacks 180
  NVFP4 weight groups to MLX native NVFP4 on CUDA, and completes CUDA decode.
- Prefill at 2048 tokens is above the requested 39 tok/s target: 392.71 tok/s.
- Decode improves from 0.89 tok/s to 5.24 tok/s on the comparable short prompt,
  but it remains below the requested 9 tok/s target.
- The GPT-OSS MXFP4 control remains healthy at 70.23 decode tok/s.

The remaining decode gap is structural in this checkpoint, not just dispatch
selection. The local ModelOpt NVFP4 checkout is about 31 GB on disk and only the
MLP-style triplets are NVFP4; attention, embeddings, norms, and other tensors
remain dense. The fully affine `gemma-4-31b-it-4bit` reference checkout is about
18 GB and still measures 8.54 decode tok/s on the same GB10 benchmark line.
Reaching 9 tok/s for this ModelOpt checkpoint likely requires either native
ModelOpt/FP4 kernels or quantizing the currently dense tensors, beyond the
load-time repack stopgap.
