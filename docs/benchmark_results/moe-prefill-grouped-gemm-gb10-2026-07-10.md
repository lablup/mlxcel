# MoE prefill: sorted GatherQMM via dequant + CUTLASS grouped GEMM (issue #629, GB10)

Date: 2026-07-10. Host: NVIDIA GB10 (Grace-Blackwell, sm_121 / cc 12.1), CUDA 13.0, MLX 0.32.1 (pin `57c66cac`).
Binary: `cargo build --release --features cuda`. Bench: `mlxcel-bench-decode` (`--prompt-tokens 2048 --max-tokens 8`), `mlxcel generate` (greedy parity).

## TL;DR

CUDA MoE prefill collapsed to 5-10x below Metal M1 Ultra because `GatherQMM::eval_gpu` ignores the callers' sorted-indices hint: the SwitchGLU prefill contract delivers x pre-gathered to one row per (token, expert) pair (`M == 1`, `B = tokens * top_k`), and the tiled `qmm_sm80` gather path then runs B independent 1-row GEMMs, re-reading each expert's full quantized weight matrix once per assigned token (~`B * N*K*bits/8` bytes of DRAM traffic per projection). Routing the sorted `M == 1` case through a one-shot dequantization of the expert stack plus the CUTLASS grouped GEMM that the float `GatherMM` right-sorted path already uses (`cutlass_grouped_gemm_unaligned`) cuts the weight traffic to `E * N*K` per projection and lifts MoE prefill **3.6-40x across all six outlier models**; GB10 now beats the Metal M1 Ultra reference on every one of them. Greedy parity is byte-identical on mixtral / phi-3.5-moe / minimax; decode is untouched.

## Root cause

From `patches/mlx/backend/cuda/quantized/quantized.cpp` (pristine upstream dispatch before this change):

- The models' shared `SwitchGLU` (`src/models/switch_layers.rs` and per-model copies) sorts tokens by expert whenever `tokens * top_k >= 64` and calls `gather_qmm(..., sorted_indices=true)` with x flattened to `[n_sorted, 1, hidden]`.
- On Metal, sorted indices select a segmented kernel that processes each expert's contiguous token block as a real matrix-matrix GEMM.
- On CUDA, `GatherQMM::eval_gpu` never reads `right_sorted_`. The dispatch sees `M=1, B=tokens*top_k` and launches `qmm_sm80` with grid `(1, N/128, B)`: a 16-row CTA tile computing 1 real row per (token, expert) pair, with zero weight reuse across the tokens assigned to the same expert.
- Upstream #3706's JIT `gather_gemm` rework does not reach this path either (it covers the float GatherMM family; the quantized GatherQMM dispatch has no grouped path at all), which is why the #625 pin bump left the MoE prefill outliers unchanged.

nsys evidence (mixtral-8x7b-4bit, 512-token prefill, `--cuda-graph-trace=node`, 2 prefill passes):

| | top kernel | instances | total GPU time |
|---|---|---:|---:|
| before | `qmm_sm80_kernel` | 192 (32 layers x 3 projections x 2 passes) | 54.5 s = **99.0%** of GPU time, avg 284 ms/call |
| after | `GemmGrouped` + `affine_dequantize` | 192 + 192 | 1.9 s + 1.5 s = 86% of a far smaller total |

## Fix

Overlay change in `patches/mlx/backend/cuda/quantized/quantized.cpp` (`GatherQMM::eval_gpu`):

- When `right_sorted_ && transpose_ && M == 1` and the batch is large enough to amortize (`B >= min_rows * E`, default `min_rows = 8`, env `MLXCEL_GATHER_QMM_GROUPED_MIN_ROWS`), dequantize the expert weight stack `[E, N, K]` to the activation dtype (`affine_dequantize`, or `fp_dequantize` for mxfp4/mxfp8) and run `cutlass_grouped_gemm_unaligned` over the expert-contiguous row segments (one GEMM problem per expert, GPU-side pointer prep from the sorted indices).
- Kill switch: `MLXCEL_GATHER_QMM_GROUPED=0`.
- Excluded from the fast path (falls through to the legacy dispatch): nvfp4 (tensor-level global scale not plumbed), `E > 1024` (single-block histogram limit in `prepare_grouped_mm_data`), non-float32/16/bf16 activations, non-1:1 pre-gathered x.
- Decode is unaffected: single-sequence decode never sets `sorted_indices` (the `>= 64` row sort gate in the models), and small sorted batches fail the `B >= 8*E` amortization gate.

Companion fix in `patches/mlx/backend/cuda/gemms/grouped_gemm_unaligned.cu`: `prepare_grouped_mm_data` computed its per-group A/B/out byte offsets in int32 (`group * item_size * b_batch_stride`), which overflows once an expert stack exceeds ~2^31 bytes of batch stride product; minimax-m2-3bit (E=256, 1536x3072 per expert) crashed with an illegal memory access. The batch strides and offset math are now int64. This is a latent bug for the pre-existing float `gather_mm_rhs` path as well (any b with `group_index * M * N * itemsize > 2^31`).

The dequant temporaries (`E * N * K * itemsize` per projection, 0.5-2.4 GB across the six models) are stream-ordered and released at command-buffer completion; MLX's `MAX_ACTIVE_TASKS`/memory-limit backpressure bounds the in-flight footprint (mixtral peak 27.3 -> 32.0 GB at 2048-token prefill).

## Results (GB10, 2048-token prompt)

Prefill tok/s. "before" = legacy path (`MLXCEL_GATHER_QMM_GROUPED=0` on the final binary for the bottom four rows; pre-change binary for mixtral / phi-3.5-moe, which measure identically since the legacy path is untouched). "after" = final binary, fast path on:

| model | quant | E | top_k | prefill before | prefill after | speedup | Metal M1 Ultra | decode before | decode after |
|-------|-------|--:|------:|---------------:|--------------:|--------:|---------------:|--------------:|-------------:|
| mixtral-8x7b-4bit | affine 4b | 8 | 2 | 14.2 | 570.9 | 40.2x | 81.0 | 31.3 | 30.6 |
| phi-3.5-moe-4bit | affine 4b | 16 | 2 | 38.4 | 895.4 | 23.3x | 114.0 | 54.8 | 53.5 |
| llama-4-scout-17b-4bit | affine 4b | 16 | 1 | 30.1 | 398.1 | 13.2x | 119.5 | 21.0 | 20.8 |
| minimax-m2-3bit | affine 3b | 256 | 8 | 34.0 | 121.2 | 3.6x | 68.0 | 21.4 | 22.4 |
| gpt-oss-20b-mxfp4 | mxfp4 | 32 | 4 | 155.9 | 1477.6 | 9.5x | 282.4 | 79.9 | 79.6 |
| solar-open-100b-4bit | affine 4b | 128 | 8 | 56.7 | 448.5 | 7.9x | 71.0 | 19.5 | 19.5 |

Acceptance criteria from #629:

- mixtral >= 81 tok/s at a 2048-token prompt: met at 570.9 (7.0x the parity bar, 3.5x the 2x stretch goal).
- All six models improve and now beat the Metal M1 Ultra absolute numbers.
- No decode regression (all within noise).

## Parity

Greedy 40-token continuations (~110-token prompt, `--no-chat-template`, temp 0), legacy vs fast path on the same binary:

- mixtral-8x7b-4bit, phi-3.5-moe-4bit, minimax-m2-3bit: **byte-identical**.
- gpt-oss-20b-mxfp4: the fast path is self-deterministic, but greedy continuations diverge from the legacy path (on a second prompt the two paths agree for the first ~36 of 40 tokens before splitting). Both continuations are coherent and on-topic. The dequantized operand values are bit-identical between the paths (same dequant formula, same target dtype); what differs is the fp32 accumulation order inside the GEMM (CUTLASS grouped tiles vs the fused qmm mainloop), so near-tie logits can flip. This is the expected numeric-tolerance behavior, not a correctness defect; the AC's parity gate (mixtral, phi-3.5-moe) passes byte-exact.

## Reproduce

```bash
./target/release/mlxcel-bench-decode --model ./models/mixtral-8x7b-4bit --prompt x --prompt-tokens 2048 --max-tokens 8
MLXCEL_GATHER_QMM_GROUPED=0 ./target/release/mlxcel-bench-decode --model ./models/mixtral-8x7b-4bit --prompt x --prompt-tokens 2048 --max-tokens 8   # legacy path
TMPDIR=$PWD nsys profile -o moe_prefill --cuda-graph-trace=node ./target/release/mlxcel-bench-decode --model ./models/mixtral-8x7b-4bit --prompt x --prompt-tokens 512 --max-tokens 4 --warmup-tokens 1
```
