# Few-row dense matmul and the gate/up kernel on M1 Ultra (2026-11-01)

Host: Apple M1 Ultra, 128 GB, Metal. Found by the #2587 acceptance run, which compared 511dce96 (before epic #2566) with db41244e. Ladder: `scripts/bench_serving_concurrency.py --prompt-tokens 512 --max-tokens 128 --concurrency 1,2,4` against a fresh `mlxcel-server --parallel 4`. Aggregate tok/s.

## Symptom

Moving a family from the per-row fallback to a native batched forward lowered B=2 throughput on some rows, reproducibly over 4 rounds, while B=1 did not move. pythia-1b went from 230 to 165 tok/s at B=2, below its 177 at B=1.

## Cause 1: dense products with 2 or more rows

Generation 15 and later send a dense float16 or bfloat16 product with `M >= 2` to `gemv_wide`. Generations 13 and 14 are excluded and fall to the steel GEMM tile, whose cost is flat in M but several times a one-row GEMV. `x @ w.T`, ms per product, bfloat16:

| shape (N, K) | 1 row | 2 rows GEMM | 2 rows GEMV | 4 rows GEMM | 4 rows GEMV | 6 rows GEMM | 6 rows GEMV | 8 rows GEMM | 8 rows GEMV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6144, 2048 | 0.055 | 0.202 | 0.071 | 0.204 | 0.112 | 0.222 | 0.145 | 0.204 | 0.183 |
| 8192, 2048 | 0.065 | 0.275 | 0.086 | 0.283 | 0.134 | 0.371 | 0.187 | 0.316 | 0.245 |
| 2048, 8192 | 0.068 | 0.197 | 0.088 | 0.201 | 0.139 | 0.229 | 0.198 | 0.237 | 0.281 |
| 50304, 2048 | 0.352 | 1.330 | 0.623 | 1.332 | 1.159 | 1.353 | 1.666 | 1.325 | 2.163 |

The matmul overlay runs up to 6 rows through the GEMV kernel as a batch: one dispatch, grid depth M. At 6 rows the projections still win while `lm_head` starts to lose; at 8 a Pythia-shaped step is a tie. `src/lib/mlxcel-core/examples/dense_matmul_batch_probe.rs` reproduces the table, and `MLXCEL_DENSE_GEMV_ROWS=0` is the upstream arm. Every batched row was bitwise equal to the same row alone. Quantized projections do not take this route, so 4-bit rows do not move: over two more rounds per arm, Qwen3-4B, Llama 3.1 8B and Command R7B stayed within 2% at every B.

| checkpoint | arm | B=1 | B=2 | B=4 |
|---|---|---:|---:|---:|
| pythia-1b | off | 189.5 | 166.5 | 270.0 |
| pythia-1b | on | 180.9 | 264.2 | 283.8 |
| ling-lite-1.5 | off | 57.4 | 69.4 | 105.9 |
| ling-lite-1.5 | on | 63.9 | 110.6 | 141.2 |
| moondream2 | off | 61.7 | 68.1 | 88.8 |
| moondream2 | on | 64.5 | 90.9 | 102.1 |
| paddleocr-vl-bfloat16 | off | 168.9 | 147.9 | 281.6 |
| paddleocr-vl-bfloat16 | on | 119.0 | 318.2 | 328.1 |

The B=1 column is the ladder's first level and carries warm-up noise; the route does not apply at one row.

## Cause 2: the gate/up kernel at B >= 2

`SwitchGLU::forward` tried the `moe_switch_gate_up` kernel (#2514) ahead of `gather_qmm` at 2 to 8 tokens. #2514 measured it on M3 Ultra only. On M1 Ultra it loses about 10%:

| checkpoint | arm | B=2 | B=4 |
|---|---|---:|---:|
| qwen3-30b-a3b-4bit | kernel | 94.2 | 115.3 |
| qwen3-30b-a3b-4bit | gather_qmm | 104.1 | 124.5 |
| qwen3-omni-30b-a3b-instruct-4bit | kernel | 93.2 | 112.4 |
| qwen3-omni-30b-a3b-instruct-4bit | gather_qmm | 102.7 | 121.2 |
| mixtral-8x7b-instruct-v0.1-4bit | kernel | 41.1 | 34.3 |
| mixtral-8x7b-instruct-v0.1-4bit | gather_qmm | 43.5 | 41.1 |

`forward` now takes it at one token, or on M3 and later. M2 and unenumerated generations were not measured and take the M1 route. `forward_activated` (Inkling) is unchanged.

## Exactness

On the fixed build, a teacher-forced trace at B=4 against B=1 shows 0 disagreements at decided positions on all 49 rows of the #2587 matrix, and `generate` output matches db41244e on every text and image row. pythia-1b's trace is byte-identical in effect.

## What stays below the fallback

A few quantized rows remain 2 to 8% under the per-row fallback at B=2, where two one-row graphs overlap well on the GPU: stablelm-2-1_6b (135 vs 146), phi-2 (88 vs 95), mixtral (45 vs 48). B=4 is within 1% of the fallback or above it on these rows; Command R7B is the one B=4 exception (98 vs 104). All stay above B=1. Neither change above affects them. Not measured: base M1, M1 Pro and Max, and M2.
