# sm_120/121 quantized GEMM tile tuning for Blackwell (issue #637, GB10)

Date: 2026-07-10. Host: NVIDIA GB10 (Grace-Blackwell, sm_121 / cc 12.1), CUDA 13.0, MLX 0.32.1 (pin `57c66cac`).
Binary: `make release-cuda`. Bench: `mlxcel-bench-decode` (`--prompt-tokens` for prefill), `mlxcel generate` (greedy parity).

## TL;DR

The CUDA quantized prefill GEMM (`qmm_sm80`) ran at ~0.55x the bf16 ceiling on GB10 because it used the Ampere CTA tile (M=64), which underutilizes Blackwell's SMs. Raising the CTA tile's M cap to 128 on sm_120/121 (a one-line, arch-gated heuristic change in the overlay `qmm_sm80.cu`) recovers **+31-38% prefill** throughput, greedy-parity identical, no decode regression. Wider-N / deeper-K tiles were also swept but break the fixed-MMA shared-memory layout (JIT failure), so `tile_m` is the one safely-tunable axis in-tree.

## Kernel dispatch map on sm_121

From `patches/mlx/backend/cuda/quantized/quantized.cpp`:

- **sm_90 (Hopper) kernel**: not available (the GB10 build passes arch `121`, not `90a`, so `supports_qmm_sm90` is false).
- **sm_80 (Ampere) kernel**: available (Ampere `mma` JIT-compiles for 121). Used for large-M prefill.
- **naive**: fallback.
- **qmv (matrix-vector)**: taken when `M==1 && B==1` (single-sequence decode) or `M*B < 8` (small batched decode).

Net: single-sequence decode uses `qmv` (GEMV-bound, correct). Prefill (large M) uses `qmm_sm80`. Batched decode with `M*B` in [2,8) uses per-row `qmv`; `>=8` uses `qmm_sm80`.

## Prefill 4bit-vs-bf16 ceiling (before the fix)

| model | prompt tokens | 4bit tok/s | bf16 tok/s | 4bit/bf16 |
|-------|--------------:|-----------:|-----------:|----------:|
| llama-3.1-8b-4bit | 2048 | 2797 | 4742 | 0.59x |
| llama-3.1-8b-4bit | 8192 | 2330 | 4261 | 0.55x |

4bit prefill should be at least as fast as bf16 (fewer weight bytes to move); 0.55x means the quantized GEMM leaves the GPU idle.

## ncu roofline (qmm_sm80, tile 64x128x64, 8192 prefill)

| metric | value |
|--------|-------|
| SM (compute) throughput | 35-47% |
| Memory throughput | 52-68% |

Neither roofline is saturated. The Ampere-tuned tile underutilizes Blackwell SMs. Headroom confirmed (ncu run via `--set roofline`; requires admin profiling access on this host).

## Tile sweep (llama-3.1-8b-4bit prefill @8192)

Swept via an env override of the CTA tiler (`MLXCEL_QMM_TILE_M/N/K`); the kernel JIT-compiles per resolved tile.

| tile (M,N,K) | prefill tok/s | note |
|--------------|--------------:|------|
| 64,128,64 (Ampere default) | 2271 | baseline |
| **128,128,64** | **3075** | **+35%, winner** |
| 64,128,128 | ERR | JIT failure (smem/MMA layout) |
| 64,128,256 | ERR | JIT failure |
| 64,256,64 | ERR | JIT failure |
| 64,256,128 | ERR | JIT failure |
| 128,128,128 | ERR | JIT failure |
| 64,64,64 | ERR | JIT failure |

Only `tile_m` is safely tunable; wider `tile_n` / deeper `tile_k` break the fixed-`make_tiled_mma` shared-memory layout.

## Fix + validation

Arch-gated `make_cta_tiler` (overlay `qmm_sm80.cu`): `tile_m` cap = 128 when `device.compute_capability_major() >= 12` (sm_120/121), else 64. Small-M (decode) is unaffected because `min(128, next_power_of_2(m))` stays small.

Numbers below are the same-session paired comparison (arch-gated default tile_m=128 vs `MLXCEL_QMM_TILE_M=64`), the most apples-to-apples measurement. The tile_m=64 baseline itself varies ~5% run-to-run (2213-2330 tok/s observed for llama @8192); every measurement clears the +25% bar.

| check | result |
|-------|--------|
| llama-3.1-8b-4bit prefill @8192 | 2213 -> 3054 (+38%) |
| qwen2.5-7b-4bit prefill @8192 | 2410 -> 3168 (+31%) |
| greedy parity (llama, 40 tok, temp 0) | generated tokens identical |
| decode (llama, m=1) | unaffected (qmv path, never reaches qmm_sm80) |

Meets #637's acceptance (>= 25% prefill @8192).

## Scope: addressed vs not

- **Addressed**: large-M prefill and batched prefill on sm_120/121 (+32%).
- **NOT addressed**: small-M batched decode (`M*B` in [2,8)) still uses per-row `qmv` (no weight amortization across the batch), and the `tile_m` cap does not raise small-M tiles. This is the batched-decode non-amortization behind #714's `--parallel 4` CUDA regression, and needs a dedicated small-M kernel (or a weight-reusing batched `qmv`) - upstream MLX territory. Until then, #714's `--parallel` default should be made backend-aware (fall back to 1 on CUDA/Blackwell).
- **Further prefill gains** toward the bf16 ceiling (0.72x -> 1.0x) need a Blackwell-tuned kernel (5th-gen tensor-core MMA, wider tiles with a compatible smem layout) - upstream MLX.

## Upstream

MLX's CUDA quantized GEMM has no sm_120/121 tile specialization: the Ampere tile underutilizes Blackwell (ncu evidence above), and there is no amortizing small-M (batched decode) quantized path on Blackwell. A tracking issue on lablup/mlxcel should carry this for upstream follow-up.
