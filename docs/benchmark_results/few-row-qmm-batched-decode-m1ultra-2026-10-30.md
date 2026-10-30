# Few-row quantized matmul for batched decode on M1 Ultra (2026-10-30)

Host: Apple M1 Ultra (Mac13,2), 128 GB, macOS 27.0.1. Backend: Metal, `--features metal,accelerate`. Base: `main` at `8385f40b`, MLX pin `7ca40033`. Model: `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (affine 4-bit, group 64, bf16 scales).

## The problem

`mlxcel-server --parallel 8` on Llama 3.1 8B 4-bit took 57 ms per 8-row decode step against 9.4 ms per 1-row step, so 8 concurrent requests reached only 1.37x the aggregate throughput of one (B=1 98 tok/s, B=8 135 tok/s, 512-token prompt, 128 tokens, prompt cache on).

The step time is not new. The 2026-07-09 serving record ([serving-throughput-defaults-m1u-2026-07-09.md](serving-throughput-defaults-m1u-2026-07-09.md)) shows 13.5 tok/s per request at 8 clients, about 74 ms per step. What moved the ratio is B=1: decode went from 86 to 105 tok/s and the warm prompt cache cut time to first token from about 780 ms to 83 ms, so the single-request baseline grew while the 8-row step did not shrink.

## Where the step goes

Neither the paged attention path nor the scheduler is the cause. `--decode-storage-backend dense` gives 62 ms at B=8, and `MLXCEL_FORCE_SYNC=1` gives 87 ms. The quantized projections account for the whole step. `src/lib/mlxcel-core/examples/qmm_batch_probe.rs` times one affine 4-bit `quantized_matmul` at each Llama 3.1 8B projection shape:

| shape (N x K) | B=1 ms | B=8 ms, upstream |
|---|---|---|
| qkv 6144 x 4096 | 0.08 | 0.29 |
| o_proj 4096 x 4096 | 0.06 | 0.15 |
| gate_up 28672 x 4096 | 0.15 | 0.83 |
| down 4096 x 14336 | 0.08 | 0.41 |
| lm_head 128256 x 4096 | 0.51 | 3.54 |

(0.29 + 0.15 + 0.83 + 0.41) x 32 layers + 3.54 = 57.3 ms, against a measured 57.2 ms step.

On GPU generation 13 (M1), MLX dispatches an affine product with 2 <= M < `get_qmv_batch_limit` (18 or 12 on M1 Ultra at these shapes) to `qmv_fast`, one threadgroup row per activation row, and pays the nibble unpack, conversion and multiply again for every row. 8-bit weights give about the same B=8 time (lm_head 4.04 ms) as 4-bit, so the cost tracks row-weight pairs, not weight bytes. The two upstream kernels that read the weight once are slower on this GPU: `qmv_wide` (gated to generation 15+ upstream) takes 4.7 ms for lm_head at B=8 and the BM=32 steel `qmm`/`qmm_splitk` 4.3 ms, three quarters of whose MMA rows are padding at M=8.

## The change

`mlxcel_affine_qmm_t_rows` (overlay `src/lib/mlxcelverse/overlays/metal/mlx/backend/metal/kernels/quantized.metal`) is `qmm_t_impl`'s steel MMA path re-tiled to one 8-row warp tile (BM = 8, WM = 1), BN = 32 with two simdgroups, BK = 64 (BK = 32 at group size 32). The `quantized.cpp` overlay sends affine 4- and 8-bit, transposed, non-batched products with 5 to 8 rows to it on generations 13 and 14, behind the existing `MLXCEL_QMV_WIDE` switch.

Tile sweep at B=8 (ms; BK, BN, WN):

| config | qkv | o_proj | gate_up | down | lm_head |
|---|---|---|---|---|---|
| 32, 32, 4 | 0.25 | 0.20 | 0.63 | 0.29 | 2.04 |
| 64, 32, 4 | 0.13 | 0.11 | 0.43 | 0.27 | 1.79 |
| 32, 64, 4 | 0.15 | 0.12 | 0.47 | 0.33 | 1.95 |
| 64, 64, 4 | 0.14 | 0.11 | 0.45 | 0.31 | 1.84 |
| 32, 32, 2 | 0.14 | 0.11 | 0.45 | 0.30 | 1.89 |
| **64, 32, 2** | **0.13** | **0.10** | **0.42** | **0.27** | **1.75** |

Crossover against per-row `qmv` (ms, upstream / new):

| B | qkv | o_proj | gate_up | down | lm_head |
|---|---|---|---|---|---|
| 4 | 0.13 / 0.16 | 0.09 / 0.14 | 0.44 / 0.41 | 0.23 / 0.27 | 1.82 / 1.95 |
| 5 | 0.15 / 0.16 | 0.11 / 0.11 | 0.53 / 0.41 | 0.27 / 0.27 | 2.24 / 1.75 |
| 6 | 0.17 / 0.17 | 0.12 / 0.11 | 0.64 / 0.41 | 0.32 / 0.27 | 2.67 / 1.75 |
| 8 | 0.21 / 0.18 | 0.15 / 0.16 | 0.84 / 0.42 | 0.42 / 0.27 | 3.54 / 1.76 |

The new kernel's time is flat in B; per-row `qmv` grows linearly. From 5 rows the sum over a layer favours the new kernel, so the dispatch starts there. Rows 1 to 4 keep upstream `qmv`.

Two designs that kept `qmv`'s per-row reduction (bitwise equal to M=1) were measured first and dropped: moving the row loop inside the K loop (lm_head B=8 4.27 ms) and widening each weight once per K block (4.16 ms with half storage, 14.5 ms with f32 storage from register spills).

## Serving result

`acceptance`-style load (`scripts/bench_serving_concurrency.py`, ~467-token prompt, 128 tokens, temperature 0, `--parallel 8 --metrics`, prompt cache on), three rounds:

| arm | B=1 step | B=8 step | B=1 tok/s | B=8 tok/s | ratio |
|---|---|---|---|---|---|
| `MLXCEL_QMV_WIDE=0` (upstream route) | 9.29 ms | 56.46 ms | 101.4 | 136.5 | 1.35 |
| new kernel, round 1 | 9.24 ms | 39.09 ms | 102.2 | 193.8 | 1.896 |
| new kernel, round 2 | 9.31 ms | 39.11 ms | 99.3 | 193.6 | 1.950 |
| new kernel, round 3 | 9.24 ms | 38.90 ms | 100.0 | 194.6 | 1.946 |

B=1 is unchanged (the 1-row path is untouched).

## Numerics

- `metal_few_row_qmm_matches_dequantized_reference` (`ffi_tests.rs`) checks rows 1 to 8 against an f32 product with the dequantized weight for group 32/64/128, 4 and 8 bits, f32/f16/bf16, K up to 4096 and an N that is not a multiple of 32. With the K loop deliberately cut one tile short it fails at the first 5-row case, which also shows the route engages from 5 rows.
- In the probe, the new kernel's largest absolute error against the f32 reference is at or below per-row `qmv`'s on every shape.
- Teacher-forced logit trace at the width the change runs at (`MLXCEL_TRACE_BATCH_ROWS=8`, Llama 3.1 8B 4-bit, 16 chunks of 128 positions after 256 context tokens, corpus `docs/architecture.md` + `docs/CONTINUOUS_BATCHING.md`, `--decided 2`):

| reference vs candidate | top-1 disagreement | on decided positions | perplexity |
|---|---|---|---|
| B=1 vs B=8 `MLXCEL_QMV_WIDE=0` | 0 / 2048 | 0 / 681 | +0.000% |
| B=8 `MLXCEL_QMV_WIDE=0` vs B=8 new | 2 / 2048 (largest gap 0.016) | 0 / 681 | -0.003% |
| B=1 vs B=8 new | 2 / 2048 (largest gap 0.016) | 0 / 681 | -0.003% |

The reverted arm is byte-identical to B=1, as the per-row `qmv` design guarantees; the new arm differs only at near-tie positions.

## Consequences and limits

- On M1 and M2 class GPUs a row of an 8-row decode batch is no longer bitwise equal to the same request decoded alone, as on generation 15+ with `qmv_wide`. Greedy output can differ at near-tie positions depending on batch composition.
- The speculative exactness gate (`models::speculative_exactness`) already retries a failing MTP probe with `MLXCEL_QMV_WIDE` off; that now also turns this kernel off on generations 13 and 14, so MTP verify blocks of 5 to 8 rows keep their byte-identity contract at the cost of the faster batched decode for that process. The live MTP probe on an M1 host was not re-run for this change.
- Measured on M1 Ultra only. The crossover at 5 rows and the BK = 64 choice are not measured on base M1, M1 Pro/Max or M2; MLX's own `get_qmv_batch_limit` table differs across those sizes.
- 9 to 11 rows (below the qmm switch-over at these shapes) still take per-row `qmv`; extending the kernel past one 8-row tile is not done here.
