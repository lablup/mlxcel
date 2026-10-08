# Technical Report: PR #2253 - Run a ROCm Decode Batch's Rows as One Quantized Product

**Date**: 2026-10-09

**Status**: Merged after the full `make verify-rocm` gate. Head `474163e3` (rebased onto origin/main `7a3fcc4a`). Closes #2156.

**Languages**: C++/HIP (ROCm overlay `patches-rocm/mlx/backend/rocm/quantized/qmm.hip`), Rust (new `tests/rocm_qmm_batched_rows.rs`, new `examples/qmm_batch_rows_probe.rs`), Python (new `scripts/bench_serving_metrics.py`, `scripts/bench_serving_concurrency.py`, new `tests/test_bench_serving_concurrency.py`), Bash (`scripts/benchmark_paged_decode_production.sh`), Markdown (`LOCAL_FIXES.md` item 43, `docs/environment-variables.md`, `docs/benchmarks.md`, new results page and data directory)

**Risk Level**: Medium. The fold changes which kernel serves every batched quantized projection on ROCm, and the route change moves bf16 GEMMs of 2 to 8 rows off the fused WMMA kernel. Both are confined to the ROCm overlay, the fold only applies to an unbatched weight and a row-contiguous activation whose output bytes are the same either way, and `MLX_ROCM_WMMA_QMM=1` still forces the WMMA kernel. Single-stream decode is unchanged within noise.

## Executive Summary

Server decode with four concurrent requests on gfx1151 ran at about 6 tok/s per request against about 32 tok/s single-stream. The issue asked to separate decode from interleaved prefill, profile the decode window, check three named candidates, and fix a contained cause.

The batched decode step itself was 5.3 times the single-stream step. The batched forward feeds every projection a `[B, 1, K]` activation, and the ROCm overlay's `QuantizedMatmul::eval_gpu` read that as `B` one-row products. It launched `qmv_warp_shared_batched_kernel`, which streams the whole weight once per batch element. In the traced model-level batch-4 step, those launches took 134.8 of 148.5 ms.

PR #2253 folds the batch into the row count when the weight is unbatched and the activation is row contiguous, which is how the Metal backend reads it. The rows then take `qmv_wide_kernel` in one pass over the weight. For bf16 checkpoints, the folded rows would have reached the fused WMMA kernel, which costs 7 to 8 times the one-row GEMV at 2 to 8 rows, so `select_qmm_route` now keeps 4- and 8-bit GEMMs with biases of at most 8 rows off that kernel.

On Meta-Llama-3.1-8B-Instruct-4bit, the batch-4 ~1K server decode step went from 166.7 to 65.9 ms and per-request decode from 6.0 to 15.3 tok/s. Single-stream `mlxcel-bench-decode` went from 37.69 to 37.84 tok/s. Qwen3-0.6B-4bit (bf16) batched steps fell by 1.5 to 1.6 times at batch 2 to 8. At ~16K the per-request rate is set by other requests' chunked prefill, which is 78% flash SDPA; that is filed as #2251. The Metal reference ratio was not measured, since there is no Metal host.

## 1. Problem Statement

### 1.1 The symptom

The PR #2103 results (`docs/benchmark_results/rocm-paged-attention-gfx1151-2026-10-05.md`, `mlxcel-server --parallel 4 --ctx-size 131072`, 128 decode tokens per request, medians of 3) gave per-request decode of 5.6 to 6.1 tok/s at batch 4 ~1K, 10.2 to 10.5 at ~4K and 10.9 to 13.9 at ~16K, with either paged-attention arm. Single-stream `bench_decode` on the same model is about 35 to 37 tok/s.

The page attributed the gap to waiting while the other requests prefill, without measuring it. The issue pointed out that three 1K prefills at about 1000 tok/s account for about 3 s of a roughly 21 s decode window, so that explanation was incomplete. It also noted that the rate rising with context is not what a per-step compute cost would produce.

### 1.2 The harness could not see the step

`scripts/bench_serving_concurrency.py` computes `decode_tok_s = (completion_tokens - 1) / (total_s - ttft)`, so the window includes any scheduler time spent on other requests' prefill chunks. The server already exported `mlxcel_batch_decode_steps_total`, `mlxcel_batch_decode_tokens_total`, `mlxcel_batch_mixed_steps_total`, `mlxcel_batch_prefill_chunks_total` and `llamacpp:tokens_predicted_seconds_total`, but the harness did not read them. Step 1 of the issue was to print the per-level deltas so the batched step cost becomes visible.

### 1.3 The root cause

With the new counters, the batch-4 ~1K level showed occupancy 4.00 and 0 prefill chunks across its 127 decode steps, so the slow rate was the decode step itself: 166.7 ms against 31.3 ms at batch 1.

The model-level probe `examples/profile_batched_decode.rs` runs the same `forward_batched` the server calls, without the scheduler. Under `rocprofv3 --kernel-trace`, its traced batch-4 step was 148.5 ms, 141.3 ms (95.2%) of it GPU busy, and the quantized projections took 134.8 ms in 161 launches of `qmv_warp_shared_batched_kernel<f16>` per step. SDPA was 3.1 ms and copies 2.4 ms.

The batched forward gives each projection a `[B, 1, K]` activation against a 2-D weight. `QuantizedMatmul::eval_gpu` treated the leading `B` as a batch dimension with `M = 1` per element, so the dispatch took the batched GEMV, which reads the full weight once per batch element. A decode step is weight-bandwidth bound, so batch 4 cost about four single-stream steps plus overhead. At model level, batch-4 `forward_batched` (157.4 ms) was slower than four sequential `forward` calls (111.6 ms).

`examples/qmm_batch_rows_probe.rs` isolates the projections (Llama 3.1 8B shapes, f16, 4-bit group 64, weights rotated over enough copies to defeat the caches, microseconds per call summed over one layer's projections):

| Activation | B = 2 | B = 4 | B = 8 |
|---|---:|---:|---:|
| `[1, 1, K]` (single stream) | 562 | 554 | 562 |
| `[B, 1, K]` before | 1868 | 2747 | 4666 |
| `[B, 1, K]` after | 681 | 908 | 1368 |
| `[B, K]` (unchanged) | 676 | 901 | 1358 |

The same rows passed as `[B, K]` already ran as one product. Only the 3-D layout the batched decode uses took the per-element path.

## 2. Change Summary

| Area | Change |
|---|---|
| `patches-rocm/.../quantized/qmm.hip`, `QuantizedMatmul::eval_gpu` | When `batch_count > 1`, the weight has only singleton batch dims, `x_batch_count == batch_count` and `x` is row contiguous: `M *= batch_count` and the call proceeds as one non-batched product |
| `patches-rocm/.../quantized/qmm.hip`, `select_qmm_route` | New `wide_qmv_rows` (`M <= 8`, 4- or 8-bit, biases present); the WMMA route is skipped for those shapes unless `MLX_ROCM_WMMA_QMM=1` |
| `patches-rocm/LOCAL_FIXES.md` | Item 43 records the fix, the measurements and the fork policy (kept in mlxcelverse, not proposed upstream) |
| `tests/rocm_qmm_batched_rows.rs` (new) | `decode_batch_rows_are_one_product`: `[B, 1, K]`, `[B, K]` and `[1, B, K]` return identical bytes and match an f32 CPU reference, for f16 and bf16 at B = 2, 4, 8; a strided `[B, 1, K]` view stays on the batched path and matches the reference |
| `examples/qmm_batch_rows_probe.rs` (new) | Times the five Llama 3.1 8B projection shapes in each layout and checks the layouts agree |
| `scripts/bench_serving_metrics.py` (new) | Parses `/metrics`, computes per-level deltas, occupancy, the issue's raw ratio and the per-request decode step ms |
| `scripts/bench_serving_concurrency.py` | `--metrics` scrapes the batch counters with the path counters and prints two lines per level |
| `scripts/benchmark_paged_decode_production.sh` | Passes `--metrics` |
| `tests/test_bench_serving_concurrency.py` (new) | 12 tests of the parsing and delta arithmetic with canned `/metrics` text |
| `docs/environment-variables.md`, `docs/benchmarks.md` | The small-row exception for `MLX_ROCM_WMMA_QMM`; the `--metrics` output |
| `docs/benchmark_results/rocm-batched-decode-gfx1151-2026-10-08.md` and its data directory (new) | Results page, raw outputs, guard logs, traces and harness scripts |

Three commits: the harness extension (`3fd1654d`, 5 files, 374 insertions and 19 deletions), the fix (`895592d3`, 5 files, 436 insertions and 3 deletions) and the results page with data (`474163e3`, 159 files, 12,575 insertions). In total 169 files, 13,385 insertions and 22 deletions.

## 3. Design

### 3.1 Folding the batch into the rows

The fold sits in `QuantizedMatmul::eval_gpu` right after the singleton-batch flags are computed:

```cpp
if (batch_count > 1 && w_singleton_batch && x_batch_count == batch_count &&
    x.flags().row_contiguous) {
  M *= batch_count;
  batch_count = 1;
  x_batch_count = 1;
  x_singleton_batch = true;
}
```

Each condition guards one way the fold could change the result:

- **Unbatched weight.** Every batch element multiplies the same matrix, so the batch is just more rows. A batched weight (per-element matrices) keeps the batched path.
- **Activation batch equals output batch.** The activation carries the batch; the fold does not apply to a broadcast activation.
- **Row-contiguous activation.** `[B, M, K]` then has the same memory layout as `[B * M, K]`. A strided view, such as every other row of a `[B, 2, K]` tensor, keeps the batched path, and the test checks that case.

The output is allocated row contiguous, so `[B, M, N]` and `[B * M, N]` are the same bytes and no reshape or copy is needed after the kernel.

This is how the Metal backend reads the same call: `M = x.size() / K` when `w.ndim() == 2` and `x` is row contiguous. The ROCm overlay had diverged from that.

With the fold, the batch-4 decode rows reach the `M <= 8` dense dispatch, where `qmv_wide_kernel` serves all of them in one pass over the weight. The same fold turns the batch-4 ~1K prefill (`[4, 899, K]`) into one GEMM of 3,596 rows, which is why batch-4 TTFT also fell from 3.37 to 2.42 s.

### 3.2 Why the bf16 route had to change too

The fold alone made bf16 batched decode slower. Before the fold, a bf16 `[B, 1, K]` went to the batched qmv, about `B` times the one-row cost. After it, the folded rows met `select_qmm_route`'s WMMA branch, which takes bf16 affine GEMMs of more than one row below the 128-row ceiling. At 2 to 8 rows that kernel cost 7 to 8 times the one-row GEMV:

| bf16 probe, us per call summed over five shapes | B = 2 | B = 4 | B = 8 |
|---|---:|---:|---:|
| `[1, 1, K]` | 550 | 519 | 567 |
| `[B, 1, K]`, fold only (WMMA) | 4396 | 4218 | 3752 |
| `[B, 1, K]`, final (qmv) | 664 | 944 | 1571 |
| `[1, B, K]`, fold only (WMMA, as before the change) | 4501 | 3843 | 4111 |

At model level, the Qwen3-0.6B-4bit batch-2 step went from 10.6 to 10.7 ms before to 22.5 to 24.0 ms with the fold only (two alternated rounds, medians of 3 each).

The second change adds `wide_qmv_rows = M <= 8 && (bits == 4 || bits == 8) && biases present` and skips the WMMA branch for it unless `MLX_ROCM_WMMA_QMM=1`. Those are exactly the shapes the `qmv_wide_kernel` gate accepts on the dense path (4- or 8-bit affine with biases, up to 8 rows), so the rows take the qmv / dequantize crossover f16 already takes. 6-bit and bias-free layouts, which `qmv_wide_kernel` does not serve, keep the previous route. The rule also covers bf16 `[1, B, K]` inputs of up to 8 rows (short prompts, speculative verify steps), which took the WMMA kernel before this PR too; the last probe row shows that path was as slow.

### 3.3 Candidates checked

The issue named three candidates and asked for a verdict on each with the deciding measurement. The investigation added two more.

| Candidate | Verdict | Deciding measurement |
|---|---|---|
| Batched projections reach `DequantGemm` and rematerialize f16 weights the 256 MiB LRU cannot hold | Ruled out at batch 4, before and after | The batch-4 decode trace has no `affine_dequantize` or Tensile GEMM dispatch beyond the quantized embedding's per-step row lookup; every projection is one qmv launch (161 per step). `MLX_ROCM_QMM_DEQUANT_M_THRESHOLD=64` left batch-4 ~1K decode at 6.2, 6.1, 6.2 tok/s (6.1 without it). At batch 8, `MLX_ROCM_QMM_DEQUANT_CACHE_STATS=1` read hits=0, misses=7728 with both the default crossover and a threshold of 9, and the model-level step was 96.8 and 99.3 ms, so decode does not reach the LRU there either; the misses are the prefill passes (#2232) |
| The batched qmv route: `[B, 1, K]` read as `B` one-row products | Confirmed, fixed | 95% of the batch-4 step in `qmv_warp_shared_batched_kernel` (134.8 of 148.5 ms traced); the probe's `[B, 1, K]` against `[B, K]` columns; the server step dropped from 166.7 to 64.5 ms with the fold alone |
| Host-side scheduler work per tick | Ruled out as the cause | Server-level trace: host gaps of 3.1 ms per step at batch 1 and 6.7 ms at batch 4 ~1K (GPU busy 91.0% and 88.3%), against a 5x step gap the projection kernels account for |
| Prefill chunks interleaved into decode steps | Ruled out at ~1K; confirmed as what sets the ~16K figure | ~1K: 0 prefill chunks and 0 mixed steps across the 127 decode steps of every batch-4 level, occupancy 4.00. ~16K: 28 prefill chunks between 415 decode steps, occupancy 1.22, `mixed_steps` 0. Before the fix, each request decoded at about 2 tok/s while the next request's 16K prompt prefilled (7 chunks of 2,048 tokens, about 7 s each), and the last request, alone, at 16.4 tok/s |
| Harness window includes other requests' prefill | Confirmed at ~16K, not at ~1K | At ~1K all four requests get their first token together and decode in lockstep; at ~16K the window of the first three requests is mostly the next request's prefill |

### 3.4 Harness step accounting

`bench_serving_metrics.py` reports occupancy as decode tokens over decode steps and the decode step as `tokens_predicted_seconds` over decode tokens, which is the time between two tokens of one request. The issue's suggested ratio, `tokens_predicted_seconds` over decode steps, counts each step once per decoding request, so at batch 4 it is four times the step. The harness prints both, labels the raw one, and returns `None` instead of dividing by zero when a level has no decode steps. A missing `/metrics` endpoint prints one line saying the step cost is unknown, so the load generator still runs against older builds.

## 4. Production Impact

ROCm server decode with more than one concurrent request on 4- or 8-bit affine checkpoints, f16 or bf16. Measured on gfx1151 (Meta-Llama-3.1-8B-Instruct-4bit, f16 scales, medians of 3 under `scripts/rocm_gpu_guard.sh`):

| Case | Per-request decode tok/s, before | after | Decode step ms, before | after |
|---|---:|---:|---:|---:|
| batch 1, ~1K | 32.0 | 32.0 | 31.3 | 31.2 |
| batch 4, ~1K | 6.0 | 15.3 | 166.7 | 65.9 |
| batch 1, ~16K | 24.6 | 24.3 | 40.8 (one run) | 41.2 |
| batch 4, ~16K | 5.6 | 7.1 | 383.1 (one run) | 364.0 |

Single-stream `mlxcel-bench-decode` (pp512, tg128), before and after alternated in one guarded window: 37.51, 37.69, 37.95 tok/s before and 37.46, 37.84, 37.91 after (medians 37.69 and 37.84, +0.4%). Single-stream activations are `[1, 1, K]`, which the fold does not touch, and the bf16 route change only removes WMMA for more than one row, which a single-stream decode never had.

bf16 model-level batched step (`profile_batched_decode`, 1,024-token prompts, medians of 3, figures as LOCAL_FIXES item 43 reports them): Qwen3-0.6B-4bit went from 10.7, 18.5 and 34.7 ms to 6.9, 12.0 and 22.2 ms at batch 2, 4 and 8. gemma-3-4b-it-4bit did not move in any build at any batch size, so its batched step is not limited by these projections; it was not attributed further.

Batch-4 ~16K moved only from 5.6 to 7.1 tok/s per request. The server prefills each ~16K prompt in 2,048-token chunks and gives the decode batch one step between chunks, so a decoding request advances about one token per chunk. The traced batch-4 ~16K level (172.4 s, 98.6% GPU busy) spent 134.9 s (78%) in SDPA, 129.3 s of it in `kernel_sdpa_flash_wmma<__half, true, 128, 64, 64>` over 672 calls at about 192 ms each. That is prefill throughput, out of scope here, and filed as #2251.

Metal and CUDA are not affected: the change is in the ROCm overlay, ROCm-gated tests, and Python tooling.

## 5. Documentation

- **`docs/environment-variables.md`.** The `MLX_ROCM_WMMA_QMM` paragraph and table row now say that, unset, 4- and 8-bit GEMMs of at most 8 rows with biases skip the WMMA kernel and take the qmv / dequantize crossover, why (one weight pass for all rows; WMMA 7 to 8 times the one-row GEMV on gfx1151), that a batched decode step's `[B, 1, K]` is such a GEMM, and that `MLX_ROCM_WMMA_QMM=1` still forces the kernel.
- **`docs/benchmarks.md`.** The batched serving ladder command now passes `--metrics`, with a note on the per-level decode steps, occupancy, prefill chunks, mixed steps and decode step ms it prints.
- **`LOCAL_FIXES.md` item 43.** The fold, the route change, the measurements, the test, and the fork policy: kept in mlxcelverse, not proposed upstream.
- **Results page** `docs/benchmark_results/rocm-batched-decode-gfx1151-2026-10-08.md`: environment, decode step table, GPU-busy versus host-gap split, model-level before and after kernels, the probe tables, the bf16 section, the candidate verdicts, the ~16K analysis and the reproduce commands. The data directory holds guard logs for each session, raw outputs, kernel stats and the harness scripts.

## 6. Verification

On gfx1151 (Radeon 8060S, ROCm 7.15), every GPU run under `scripts/rocm_gpu_guard.sh`; attempts that saw a compiler or another GPU process were rejected and rerun, and only clean attempts are reported.

- **Regression test.** `cargo test --release --features rocm --test rocm_qmm_batched_rows -- --test-threads=1` passes, before and after the rebase onto `7a3fcc4a`. Before the fold, the f16 4096 x 4096 batch-4 case differed from `[B, K]` by up to 3.9e-3, so the byte comparison fails without the fix.
- **Dequant cache test.** `cargo test --release --features rocm --test rocm_qmm_dequant_cache -- --test-threads=1` passes.
- **Python.** `python3 -m unittest discover -s tests -p 'test_*.py'`: 105 pass. `pytest tests/test_bench_serving_concurrency.py`: 12 pass.
- **Lint and script gates.** `cargo clippy --features rocm --example qmm_batch_rows_probe --test rocm_qmm_batched_rows -- -D warnings` is clean. `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt verify-rocm-overlay verify-binary-assets` and `cargo test --features rocm --test dead_doc_pointers` pass.
- **Measurements.** The server matrix, single-stream decode, probes and traces in sections 1, 3 and 4; commands and scripts are in the results page's data directory.
- **Full gate.** `make verify-rocm` with `MLXCEL_ROCM_SMOKE_MODEL=models/mlx/Qwen3-0.6B-4bit` at head `474163e3` on `7a3fcc4a`: `[verify-rocm] OK`, 164 cargo test suites with 12195 passed, 0 failed and 403 ignored (8906 passed and 165 ignored in the mlxcel-core lib suite), and the ROCm smoke generated 32 tokens on the GPU.

Not verified:

- **Metal reference ratio** (issue step 3). There is no Metal host on the gfx1151 machine; the results page records it as not measured.
- **Metal and CUDA builds.** Not available on this host. No Metal or CUDA code is touched.
- **The model-level probe at ~16K.** `examples/profile_batched_decode.rs` aborts on its unchunked 16K prefill with `hipLaunchKernel ... invalid configuration argument`, in both the before and after builds on `2aa5211f`. The server chunks prefill and is not affected. #2237, merged after these runs, changed the launch grids of the strided copy kernels; whether it fixes this abort was not checked.

## 7. Technical Decisions

- **Measure before fixing.** The harness extension landed first, so the before numbers already separate the decode step from interleaved prefill. Without it, the ~1K and ~16K gaps looked like one problem with two magnitudes; with it, they are two problems with different causes.
- **Fold in `eval_gpu`, not in the model code.** Reshaping `[B, 1, K]` to `[B, K]` in each model's batched forward would fix one caller and leave the trap for the next. Folding at the backend fixes every caller, matches the Metal backend's reading of the same call, and costs no copy.
- **Keep strided activations on the batched path.** Folding a non-row-contiguous view would need a copy or a different indexing; the batched kernel already handles it correctly.
- **Gate the route change on the `qmv_wide_kernel` shapes.** Skipping WMMA exactly where the wide GEMV can take all rows in one weight pass avoids sending rows to a slower fallback. 6-bit and bias-free layouts, which the wide GEMV does not serve, keep their route.
- **Keep the override.** `MLX_ROCM_WMMA_QMM=1` still forces the WMMA kernel, so the previous behavior is one variable away if a device or shape turns out to prefer it.
- **Do not chase the ~16K figure here.** The trace showed the time is prefill attention, which the issue lists as out of scope. It is filed as #2251 with the per-kernel evidence.
- **Byte equality in the test.** Asserting `[B, 1, K]`, `[B, K]` and `[1, B, K]` return identical bytes checks that the three layouts take the same kernel, which a tolerance check against the reference would not.

## 8. Residual Risks and Follow-ups

- **#2251: flash SDPA dominates long-context chunked prefill.** The 2,048-query chunk against up to 16K keys runs at about 192 ms per call, on the order of 1.5 TFLOP/s, which suggests the kernel limits the rate (not established). Batch-4 ~16K per-request decode stays at about 7 tok/s until that is fixed.
- **The model-level probe's 16K abort.** `profile_batched_decode` fails with "invalid configuration argument" on its unchunked 16K prefill. It is outside this issue and was not rechecked after #2237.
- **Batch sizes above 8.** At batch 9 or more the folded rows leave the `qmv_wide_kernel` range and take whatever the existing route picks for that row count (for bf16 below the 128-row ceiling, the WMMA kernel). Only batch 2, 4 and 8 were measured; the issue's matrix uses `--parallel 4`.
- **The route rule does not check the wide kernel's opt-out.** `wide_qmv_rows` mirrors the bits, biases and row limits of `qmv_wide_kernel`, but not `MLX_QMV_NO_WIDE`. With that variable set, small bf16 rows skip WMMA and land on the tiled or warp-shared GEMV. That combination was not measured.
- **A stale test comment.** `decode_batch_rows_are_one_product` describes the bf16 cases as taking the fused WMMA route. After the route change they take qmv, so the test no longer exercises WMMA at those sizes; only the comment is wrong, the assertions hold.
- **Gemma 3 4B batched step.** It did not move in any build, so something else bounds it. That was not investigated.

## 9. Learning Points

- **Make the harness report the quantity in question.** The client-side tok/s mixed the decode step with other requests' prefill. Printing occupancy and step time per level from the server's own counters turned a vague gap into a 166.7 ms step at occupancy 4.00 with no prefill in it.
- **Check the layout a caller actually uses.** `[B, K]` already ran as one product; only `[B, 1, K]`, the layout the batched decode uses, took the per-element path. A probe that times the same rows in every layout found it in one run.
- **A batch that is slower than sequential calls is a dispatch problem.** Batch-4 `forward_batched` at 157.4 ms against 111.6 ms for four `forward` calls ruled out every explanation based on compute or scheduling.
- **A fix that changes rows can move them across a route threshold.** The fold was correct for f16 but sent bf16 rows into a kernel that is slow at small row counts. Measuring the second dtype before calling the change done caught a regression of about 2x at batch 2.
- **Compare with the reference backend's reading of the same call.** The Metal backend already folded the batch. The ROCm overlay's divergence was the defect, and the Metal code gave the exact conditions for the fix.
