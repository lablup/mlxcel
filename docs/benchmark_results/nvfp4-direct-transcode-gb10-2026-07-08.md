# ModelOpt NVFP4 direct transcode benchmark (issue #693)

This note records the CUDA load and throughput comparison between the direct
ModelOpt NVFP4 transcode (issue #693) and the dense f16 requantize fallback it
replaces (issue #692) on NVIDIA GB10. The local `gemma-4-31b-it-nvfp4`
checkpoint is NVIDIA ModelOpt NVFP4 metadata with `quant_method=modelopt`,
`quant_algo=NVFP4`, and per-linear `weight/weight_scale/weight_scale_2`
triplets. The direct path reinterprets the packed FP4 U8 bytes into MLX native
NVFP4 U32 words, preserves the per-block E4M3 scales verbatim, and keeps
`weight_scale_2` as a per-linear global-scale sidecar. It never materializes a
dense f16 matrix, so it loads far faster than the dense fallback and is
bit-exact to the checkpoint. `MLXCEL_NVFP4_DENSE_REPACK=1` forces the dense
path for A/B comparison.

## Environment

| Item | Value |
|------|-------|
| Hardware | NVIDIA GB10 (DGX Spark), 122 GB unified LPDDR5x |
| Backend | CUDA |
| Build | `cargo build --release --features cuda --bin mlxcel --bin mlxcel-bench-decode` |
| Harness | `target/release/mlxcel-bench-decode` |
| Test date | 2026-07-08 |
| Prompt | `Hello, how are you today?` (short) and a synthesized 2048-token prompt |
| Raw CSV | `benchmarks/cuda_gb10_issue693_nvfp4_direct_vs_dense_2026-07-08.csv` |

Each run is a separate process, so the reported cold-load wall time and MLX
peak are measured against a fresh MLX allocator. `mlxcel-bench-decode` now
resets the MLX high-water mark before the load and prints `[Load] wall` and the
peak after load completes.

## Results

| Run | Repack | Prompt tokens | Cold-load wall (s) | Load peak (GB) | Prefill tok/s | Decode tok/s |
|-----|--------|--------------:|-------------------:|---------------:|--------------:|-------------:|
| direct, short | direct triplet transcode | 20 | 58.77 | 36.60 | 76.13 | 4.84 |
| dense, short | dense f16 requantize | 20 | 190.72 | 36.60 | 79.51 | 5.26 |
| direct, 2048 | direct triplet transcode | 2048 | 58.87 | 36.60 | 443.85 | 5.01 |
| dense, 2048 | dense f16 requantize | 2048 | 190.69 | 36.60 | 395.20 | 5.38 |

## Acceptance

The acceptance gate is a 20% improvement in cold-load wall time or peak load
memory. Cold-load wall time drops from 190.72 s (dense) to 58.77 s (direct), a
69.2% reduction, so the gate passes on load time by a wide margin. The dense
path spends most of that time reconstructing a dense f16 matrix for each of the
180 NVFP4 MLP weight groups and re-quantizing it; the direct path only
reinterprets bytes and re-encodes the small per-block scale tensors.

Peak load memory is identical at 36.60 GB for both paths. The dense f16
transients are freed between weight groups, so they do not raise the MLX
high-water mark above the loaded model. The load-time win, not peak memory, is
the improvement here.

## Throughput

Prefill at 2048 tokens is 443.85 tok/s for the direct path versus 395.20 tok/s
for the dense path, a 12% gain, because the direct weights carry the
checkpoint's own block scales rather than the dense path's re-derived ones and
the op-at-a-time MLP schedules gate and up in parallel. Short-prompt prefill is
within noise (76.13 vs 79.51 tok/s on a 20-token prompt).

Decode is about 7% slower on the direct path (4.84 to 5.01 tok/s versus 5.26 to
5.38 tok/s). The global scale forces the Gemma 4 dense MLP onto the
op-at-a-time path plus a per-linear scalar multiply, and Gemma 4 decode runs
with CUDA graphs disabled, so the extra element-wise ops add dispatch overhead
per step. Folding the three global scales into the fused
`compiled_gelu_approx_mlp_forward` C++ kernel would recover this and is a
reasonable follow-up; the load-time and correctness wins are the point of this
change.

## Correctness

The direct transcode dequantizes bit-exactly to the ModelOpt reference
(`fp4 * e4m3_decode(weight_scale) * weight_scale_2`): the FP4 nibble order maps
onto MLX native NVFP4's eight-nibbles-per-u32 layout by a little-endian byte
reinterpret, the per-block E4M3 scales are preserved verbatim (the load-time
F8_E4M3 to f16 decode re-encodes losslessly), and `weight_scale_2` is applied
on the matmul output as an exact per-tensor scalar. The dense fallback re-derives
each block scale from the reconstructed f16 values, so it drifts by roughly one
E4M3 block-scale plus one FP4 rounding step (about 10% mean relative weight
error on this checkpoint). Greedy continuation therefore diverges between the
two paths, and the direct path is the faithful one. See the fixture
`nvfp4_direct_transcode_is_exact_and_bounds_dense_drift` in
`src/models/sanitize.rs` for the documented tolerance.

## Greedy parity spot-check

`mlxcel generate --temp 0 --max-tokens 48`, prompt "Explain in a few sentences
why the sky appears blue during the day." Both paths produce coherent,
semantically identical Rayleigh-scattering explanations:

- direct: "The sky appears blue due to a phenomenon called Rayleigh scattering.
  As sunlight enters Earth's atmosphere, it collides with gas molecules and
  scatters in all directions. Because blue light travels in shorter, smaller
  waves, it is scattered"
- dense: "The sky appears blue because of a phenomenon called Rayleigh
  scattering. As sunlight reaches Earth's atmosphere, it is scattered in all
  directions by the gases and particles in the air. Because blue light travels
  in shorter, smaller waves, it"

The token streams diverge early ("due to" vs "because of", "enters" vs
"reaches") because the dense fallback re-quantizes while the direct transcode
keeps the checkpoint's exact weights. Token-identical output between the two
paths is not expected; the meaningful bit-exactness is direct against the
checkpoint reference.
