# PR #2767: Fold batched decode rows in the CUDA QuantizedMatmul

**Date**: 2026-10-31 (generated before merge)
**Status**: In review
**Languages**: C++, Rust
**Risk Level**: Medium (every CUDA device takes the fold; compute capability 12.1 also changes kernels at 5 to 8 rows, and only an RTX Spark N1X was measured)

## Executive Summary

Batched decode passes each projection `x = [B, 1, K]` against one shared 2D weight, and CUDA ran that as B independent problems, reading the whole weight once per batch entry. Metal and ROCm already fold this shape into rows. This PR adds the fold to CUDA, which raises 8-client aggregate decode on an RTX Spark N1X 1.2x to 3.2x across four models. On compute capability 12.1 a weight of at least 2^24 elements also moves to a GEMM kernel from 5 rows. Gemma 4 26B loses about 2-5% at 5 to 7 clients in exchange, for a cause this PR did not establish, and its aggregate stays flat from 4 to 8 clients because MoE gathers are untouched.

## 1. Problem Statement

### 1.1 What CUDA did

The dispatch computes `M = out.shape(-2)` and `B = out.size() / M / N`, so `[8, 1, K]` is 8 problems with `M == 1`. Every kernel then read the weight per batch entry. On the N1X an `[8, 1, 5120]` by `[17408, 5120]` 4-bit matmul took 8.7 ms against 0.9 ms for the same rows as `[8, 5120]`, both at about 430-520 MHz.

Serving showed the result: Qwen 3.8 27B fell from 26 tok/s at 4 clients to 14 at 8, and that window ran at 1570-1830 MHz, so the clock did not explain it.

### 1.2 What the other backends do

Metal's `eval_gpu` treats a 2D weight with a row-contiguous `x` as non-batched and takes `M = x.size() / K`. ROCm's `qmm.hip` multiplies `M` by the batch count under the same condition. Neither has this problem, so the fold does not carry over to them.

## 2. Technical Decisions

### 2.1 Fold in the overlay dispatch

`QuantizedMatmul::eval_gpu` folds when the weight is 2D, `B > 1`, and `x.size() == B * M * K`. Since `x` is already row-contiguous, a `[B * M, K]` view covers the same memory, and `run_row_chunked` passes that view. A 3D per-batch weight stays unfolded. `GatherQMM` is untouched: each (token, expert) pair reads a different expert weight, so its rows do not share one 2D weight.

### 2.2 The threshold depends on rows and weight size

Folding alone left 2 to 7 rows on multirow qmv. A per-width sweep sent `[B, 1, K]` to GEMM from 2 rows and kept qmv as the other arm, which gives the whole crossover from two arms:

- At 2 clients qmv wins on all four models, and at 3 it ties except on Gemma 4, where it wins.
- At 4 clients GEMM gains 18% on Qwen, 7% on Llama and 3% on gpt-oss, and loses 10-19% on Gemma 4. So 4 was rejected.
- At 5 to 7 clients GEMM gains 22-104% on Qwen and Llama and 6-10% on gpt-oss, and costs Gemma 4 5-7%.

Gemma 4's and gpt-oss's weights are mostly small (Gemma 4's `gate` is 2112 x 2816), while Qwen's and Llama's MLP weights are at least 14336 x 4096. Requiring `N * K >= 2^24` keeps the Qwen and Llama gain, cuts the Gemma 4 loss to 2-5%, and costs gpt-oss 2-4% of its gain. The value is a measured split on these four models, with nothing deriving it from hardware.

Small weights stay on qmv through 8 rows, the multirow window's limit, because that moved 8-client aggregate +0.5% to +3% without a loss. Other compute capabilities keep the threshold of 8 because they are unmeasured.

### 2.3 Only folded `M == 1` batches move

Speculative verify relies on multirow qmv being bit-identical to per-row decode, and prefill plans rely on the existing partition behavior (`docs/CONTINUOUS_BATCHING.md`). A verify block or prefill piece has `M > 1`, so it keeps the threshold of 8. Lowering the threshold globally was rejected for that reason.

## 3. Change Summary

- `quantized.cpp`: the fold, the row and size thresholds, and `MLXCEL_QMM_FOLD_BATCH`, `MLXCEL_QMM_FOLD_MIN_ROWS` and `MLXCEL_QMM_FOLD_MIN_ELEMENTS`.
- `qmm_fold_batch_tests.rs`: `[B, 1, K]` against `[B, K]` bit for bit where the kernels match, and against a dequantized reference everywhere, including an `x` view at a non-zero offset.
- `examples/qmm_fold_batch_probe.rs`: per-shape kernel timing. The name avoids #2752's `qmm_batch_probe`.
- Docs: the environment variable table, `docs/CONTINUOUS_BATCHING.md`, and the benchmark record.
- `qwen3_5_mtp/tests.rs`: a unix-only symlink is now a hard link or copy elsewhere, which let the `mlxcel-core` lib tests compile on Windows.

## 4. Results

RTX Spark N1X (sm_121), n=3, one binary, clock-gated, median aggregate tok/s. Full tables are in `docs/benchmark_results/cuda-qmm-fold-batch-n1x-2026-10-31.md`.

| Model | 5 clients, shipped / no fold | 8 clients, shipped / no fold |
|---|---:|---:|
| qwen3.8-27b | 34.9 / 23.3 | 45.0 / 14.0 |
| llama-3.1-8b | 59.9 / 47.0 | 75.6 / 33.7 |
| gpt-oss-20b | 89.6 / 86.0 | 87.9 / 72.1 |
| gemma-4-26b | 35.7 / 37.0 | 34.1 / 16.9 |

`logit_trace` at 6 rows with every weight on GEMM, 256 positions: Llama 0 top-1 disagreements, Qwen 5 with gaps of at most 0.125, and none on decided positions. Gemma 4 26B gives 29 disagreements, and 27 with qmv or with the fold off, so that divergence predates this PR.

## 5. Trade-offs and Residuals

- **Bitwise identity:** a batched row on GEMM is no longer bitwise equal to the same row decoded alone. At 8 rows it already was not, so this widens the range to 5 to 7 rows of large weights.
- **Not traced:** the trace predates the size gate. The shipped rule moves fewer weights to GEMM than the traced arm.
- **Not measured:** GB10, H100 and other compute capabilities. The fold applies to all of them. The `--parallel` advice for GB10 in `docs/CONTINUOUS_BATCHING.md` now says it predates the fold rather than repeating it.
- **Gemma 4 loss:** the probe shows GEMM equal or faster on its shapes from 5 rows except the router, which does not account for the serving loss. The cause is open, and on the same kernels 4 clients still differed about 6% between arms, so small differences on this model are near its noise.
- **MoE:** at B=8 the profile (forced evals) puts gate, up and down gathers at about 5.8 of 10.2 ms per layer. Each reads `top_k * B` expert matrices. Grouping rows by expert would need a new kernel and a measurement of how often rows share experts.
- **Test limit:** the fold test also passes with the fold off, because the unfolded dispatch is bit-identical on sm_121. It guards correctness, and nothing in CI guards the speed.

## 6. Learning Points

- A per-call probe that repeats one index set can understate bandwidth-bound work. The 64-pair gather probe read about 0.5 ms against about 2.2 ms in the live path, and that gap led to a wrong first conclusion that MoE was not the main cost at 8 clients.
- Per-call kernel timing did not predict serving on Gemma 4. It showed GEMM ahead, and serving showed a loss, so threshold decisions need the serving arm.
- A threshold sweep needs only two arms when the decision is monotonic in width: one forced each way, compared per width.
- On this laptop an accepted clock floor of 1000 MHz let a 1352 MHz run through at about a quarter of normal throughput. The active power scheme also had a 300 s display timeout despite an earlier "never" setting; a later 75-minute run needed no clock retries after resetting it, though the cause of the idle cap was not isolated.
