# Batched server decode on ROCm: Radeon 8060S (gfx1151), 2026-10-08

Why batch-4 server decode ran at about 6 tok/s per request against about 32 tok/s single-stream, and what changed it (lablup/mlxcel#2156, part of #1801).

Conclusion: the batched decode step itself was 5.3 times the single-stream step. The batched forward feeds every projection a `[B, 1, K]` activation, and the ROCm overlay's `QuantizedMatmul::eval_gpu` read that as `B` one-row products and launched `qmv_warp_shared_batched_kernel`, which streams the whole weight once per batch element. Folding the batch into the row count (what the Metal backend already does) sends the same rows to `qmv_wide_kernel` in one pass over the weight. For bf16 checkpoints the folded rows would have reached the fused WMMA kernel, which is 7 to 8 times slower than the GEMV at 2 to 8 rows, so those rows now take qmv as f16 does. The batch-4 ~1K decode step went from 166.7 to 65.9 ms and per-request decode from 6.0 to 15.3 tok/s; single-stream decode is unchanged; a bf16 checkpoint (Qwen3-0.6B) gains 1.5 to 1.6x at batch 2 to 8. At ~16K, per-request decode is set by the other requests' prefill chunks, whose time is 78% flash attention (filed as #2251), not by the decode step.

## Environment

AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`, RDNA 3.5, wave32), HIP 7.15, rocprofv3 1.3.5, Debian 13. mlxcel at `2aa5211f` (before) and `2aa5211f` plus this change (after), `--features rocm`, release builds. The change landed in two steps, the batch fold and then the small-row bf16 route; f16 GEMMs never reach the WMMA route, so for this f16 checkpoint both after builds run the same kernels. Sessions, each one guarded window, by data directory: `s1/` before (server matrix, threshold arm, single stream), `s2/` before (probe, model-level profile), `s3/` fold-only build (server matrix, before arm alternated, probes, model-level profile), `s4/` bf16 on the fold-only build, `s5/` final build (~1K server rows, single stream, bf16, batch 8), `s6/` final build server trace. Model `Meta-Llama-3.1-8B-Instruct-4bit` (4-bit affine, group 64; its scales and biases are f16, so the bf16-only WMMA route is never eligible). Every run went through `scripts/rocm_gpu_guard.sh` (host-wide lock, 60 s of idle GPU and no compiler before, 1 Hz monitor during); an attempt that saw a compiler or another GPU process was rejected and rerun, and only clean attempts are reported. Guard logs and raw outputs are in [`data/rocm-batched-decode-gfx1151-2026-10-08/`](data/rocm-batched-decode-gfx1151-2026-10-08/).

Server: `mlxcel-server --parallel 4 --ctx-size 131072 --no-prompt-cache --metrics`, one fresh server per arm, a discarded warmup level, then `scripts/bench_serving_concurrency.py --max-tokens 128 --metrics` three times per case. `--no-prompt-cache` keeps the three repetitions identical (otherwise the second repetition adopts the first one's KV). Single stream: `mlxcel-bench-decode` at `bench_decode.sh`'s default shape (pp512, tg128, 20-token warmup). The Metal reference ratio (step 3 of the issue) was not measured: there is no Metal host here.

## Decode step cost, before and after

Medians of three. Decode step ms and occupancy come from the new `--metrics` lines of the harness: occupancy is `mlxcel_batch_decode_tokens_total` over `mlxcel_batch_decode_steps_total`, and decode step ms is `llamacpp:tokens_predicted_seconds_total` over decode tokens, the time between two tokens of one request. (`tokens_predicted_seconds` over decode steps, the issue's ratio, counts a step once per request, so at batch 4 it is four times the step; the harness prints it too.)

| Case | Per-request decode tok/s, before | after | Decode step ms, before | after | Occupancy | Prefill chunks between decode steps |
|---|---:|---:|---:|---:|---:|---:|
| batch 1, ~1K | 32.0 | 32.0 | 31.3 | 31.2 | 1.00 | 0 |
| batch 4, ~1K | 6.0 | 15.3 | 166.7 | 65.9 | 4.00 | 0 |
| batch 1, ~16K | 24.6 | 24.3 | 40.8 (one run) | 41.2 | 1.00 | 7 (its own prefill) |
| batch 4, ~16K | 5.6 | 7.1 | 383.1 (one run) | 364.0 | 1.22 | 28 |

Batch-4 ~1K TTFT also fell, from 3.37 to 2.42 s (the four 899-token prompts are prefilled as one `[4, 899, K]` batch, which the same fold turns into one GEMM of 3,596 rows). Single-stream `mlxcel-bench-decode`, before and after alternated in one guarded window: 37.51, 37.69, 37.95 tok/s before and 37.46, 37.84, 37.91 after (medians 37.69 and 37.84, +0.4%; the fold-only build gave 37.62 against 37.58). The ~1K before columns are the before arm of `s3/`; the ~16K before columns come from `s1/` (tok/s medians of three; step ms from its one run with `--metrics`, since its matrix servers ran without `--metrics`). The ~16K after columns are from `s3/`, the ~1K after columns from `s5/`.

## Where the step time goes

Server, final build, one fresh `mlxcel-server` under `rocprofv3 --kernel-trace` (rocprofv3's signal handler wrote the trace on SIGINT; the server then had to be killed, since it ignores SIGINT and SIGTERM under the profiler) and one run of each level. Each level's window runs from the last request's first token to the first request's end, so every request is decoding in it; it is split into GPU-busy time (union of dispatch intervals) and host gaps. The profiler adds a few ms per step at these dispatch rates, so the traced steps are slower than the unprofiled ones in the table above.

| Level | Window | Per decode step | GPU busy | Host gaps per step | Largest kernel classes in the window |
|---|---:|---:|---:|---:|---|
| batch 1, ~1K | 4.32 s, 127 steps | 34.1 ms | 91.0% | 3.1 ms | qmv 3.44 s (161 `qmv_wide_kernel` per step), paged-KV row gather 0.34 s |
| batch 4, ~1K | 7.24 s, 127 steps | 57.0 ms | 88.3% | 6.7 ms | qmv 5.61 s, SDPA 0.52 s, copies 0.13 s |
| batch 1, ~16K | 4.89 s, 127 steps | 38.5 ms | 90.3% | 3.7 ms | qmv 2.81 s, attention 1.49 s |
| batch 4, ~16K | no window where all four decode | | | | see below |

The server overlaps its host work with the GPU (the lookahead pipeline primes step n+1 while step n runs), so its host gaps are smaller than in the model-level runs below, which evaluate each step synchronously.

The model-level probe `examples/profile_batched_decode.rs` runs the same `forward_batched` the server calls, without the scheduler; under `rocprofv3 --kernel-trace` its batched decode window (after the last prefill GEMM, 23 steps) is where the before and after kernels are compared:

| Batch-4 ~1K decode step | before | after |
|---|---:|---:|
| Model-level step (no profiler, median of 3) | 157.4 ms | 59.6 ms (60.3 ms final build) |
| Server step (median of 3) | 166.7 ms | 65.9 ms (64.5 ms fold-only build) |
| Traced model-level step | 148.5 ms | 59.5 ms |
| GPU busy | 141.3 ms (95.2%) | 50.8 ms (85.4%) |
| Host gaps inside the forward | 7.2 ms | 8.7 ms |
| Quantized projections | 134.8 ms, 161 x `qmv_warp_shared_batched_kernel<f16>` | 44.1 ms, 161 x `qmv_wide_kernel<f16>` |
| SDPA (`kernel_sdpav_1pass`, 128 per step) | 3.1 ms | 3.2 ms |
| Copies (`copy_gg_byval`, 854 per step) | 2.4 ms | 2.5 ms |

At batch 1 (final build) the model step is 27.7 ms against 31.2 ms in the server; the traced model-level batch-1 step is 32.1 ms, 23.8 ms (74%) GPU-busy, 22.0 ms of it 161 `qmv_wide_kernel` launches. The one `affine_dequantize` launch per step in every window is the quantized embedding's row lookup (2,048 threads at batch 1, 8,192 at batch 4). The model-level probe was not run at ~16K: its unchunked 16K prefill aborts with `hipLaunchKernel ... invalid configuration argument` in both the before and the after build measured here (on `2aa5211f`; the server chunks prefill and is not affected). That is outside this issue; #2237, merged after these runs, changed the launch grids of the strided copy kernels and was not checked against it.

The projections alone, from `examples/qmm_batch_rows_probe.rs` (the Llama 3.1 8B shapes, f16, 4-bit g64, weights round-robin over enough copies to defeat the caches; microseconds per call, summed over q, k/v, o, gate/up and down):

| Activation | B = 2 | B = 4 | B = 8 |
|---|---:|---:|---:|
| `[1, 1, K]` (single stream) | 562 | 554 | 562 |
| `[B, 1, K]` before | 1868 | 2747 | 4666 |
| `[B, 1, K]` after | 681 | 908 | 1368 |
| `[B, K]` (unchanged) | 676 | 901 | 1358 |

Before the fix `[B, 1, K]` cost `B` times the single-row call or more, so a batch decoded slower than the same requests one after another (model level: batch-4 `forward_batched` 157.4 ms against 111.6 ms for four `forward` calls). After it, `[B, 1, K]`, `[B, K]` and `[1, B, K]` return the same bytes (the probe's last column reads 0) and cost the same.

## bf16 checkpoints

The fold alone made bf16 batched decode slower: folded bf16 rows meet the fused WMMA kernel (`qmm_wmma_dense_kernel`, the route for bf16 GEMMs of more than one row below the 128-row ceiling), and at 2 to 8 rows that kernel costs 7 to 8 times the one-row GEMV, where the batched qmv it replaced cost about `B` times. The second part of the change keeps 4- and 8-bit GEMMs with biases of at most 8 rows off the WMMA kernel (`MLX_ROCM_WMMA_QMM=1` still forces it), so they take the qmv / dequantize crossover f16 takes. The same rule covers bf16 `[1, B, K]` inputs of up to 8 rows (short prompts, verify steps), which took the WMMA kernel before this change too.

Probe, bf16, microseconds per call summed over the five shapes (`s4/` fold only, `s5/` final):

| Activation | B = 2 | B = 4 | B = 8 |
|---|---:|---:|---:|
| `[1, 1, K]` | 550 | 519 | 567 |
| `[B, 1, K]`, fold only (WMMA) | 4396 | 4218 | 3752 |
| `[B, 1, K]`, final (qmv) | 664 | 944 | 1571 |
| `[1, B, K]`, fold only (WMMA, as before the change) | 4501 | 3843 | 4111 |

Model-level batched step (`profile_batched_decode`, 1,024-token prompts, medians of 3, two alternated rounds; ms per step):

| Model | B | before | fold only | final |
|---|---:|---:|---:|---:|
| Qwen3-0.6B-4bit (bf16) | 1 | 4.29 / 4.31 | 4.30 / 4.29 | 4.32 / 4.30 |
| | 2 | 10.63 / 10.71 | 24.02 / 22.49 | 6.92 / 6.87 |
| | 4 | 18.70 / 18.53 | 25.95 / 25.88 | 12.00 / 11.96 |
| | 8 | 34.68 / 34.69 | 32.88 / 33.01 | 22.84 / 22.21 |
| gemma-3-4b-it-4bit (bf16) | 1 | 15.56 / 15.57 | 15.66 / 15.68 | 15.62 / 15.44 |
| | 2 | 28.44 / 28.49 | 28.48 / 28.60 | 28.40 / 28.64 |
| | 4 | 63.39 / 62.75 | 63.43 / 63.57 | 63.22 / 63.35 |
| | 8 | 170.95 / 170.40 | 171.10 / 172.12 | 171.35 / 169.80 |

Gemma 3 4B's batched step does not move in any build, so its batched step is not limited by these projections; it was not attributed further here.

## Candidates

| Candidate | Verdict | Deciding measurement |
|---|---|---|
| GEMM route of the batched projections reaches `DequantGemm` and rematerializes f16 weights the 256 MiB LRU cannot hold | Ruled out at batch 4, before and after | The batch-4 decode trace has no `affine_dequantize` or Tensile GEMM dispatch beyond the quantized embedding's per-step row dequantize; every projection is one qmv launch (161 per step). `MLX_ROCM_QMM_DEQUANT_M_THRESHOLD=64` left batch-4 ~1K decode at 6.2, 6.1, 6.2 tok/s (6.1 without it). At batch 8 (`--parallel 8`, not the default) the folded f16 rows are 8; `MLX_ROCM_QMM_DEQUANT_CACHE_STATS=1` read hits=0, misses=7728 both with the default crossover and with `MLX_ROCM_QMM_DEQUANT_M_THRESHOLD=9` (which keeps 8 rows on qmv), and the model-level batch-8 step was 96.8 and 99.3 ms, so decode does not reach the LRU there either; the misses are the prefill passes (#2232) |
| The batched qmv route itself: `[B, 1, K]` read as `B` one-row products | Confirmed, fixed | 95% of the batch-4 step in `qmv_warp_shared_batched_kernel`; the probe's `[B, 1, K]` against `[B, K]` columns; the step dropped from 166.7 to 64.5 ms with only this change |
| Host-side scheduler work per tick (paged metadata, masks, evals and syncs) | Ruled out as the cause | Server-level trace: host gaps are 3.1 ms per step at batch 1 and 6.7 ms at batch 4 ~1K (GPU busy 91% and 88%), against a 5x step gap that the projection kernels account for (134.8 of 148.5 ms per model-level step before the fix) |
| Prefill chunks interleaved into decode steps (`mixed_steps`) | Ruled out at ~1K; confirmed as what sets the ~16K figure (cause filed as #2251) | ~1K: 0 prefill chunks and 0 mixed steps across the 127 decode steps of every batch-4 level, occupancy 4.00. ~16K: 28 prefill chunks between 415 decode steps, occupancy 1.22, `mixed_steps` 0 (the prototype is off by default). The per-request timeline (`s1/profile_prof.json`, before the fix) shows each request decoding at about 2 tok/s while the next request's 16K prompt is prefilled (7 chunks of 2,048 tokens, about 7 s each), and the last request, alone, at 16.4 tok/s |
| Harness window includes other requests' prefill (`decode_tok_s = (completion_tokens - 1) / (total_s - ttft)`) | Confirmed at ~16K, not at ~1K | Same evidence: at ~1K all four requests get their first token together (batched prefill) and decode in lockstep; at ~16K the window of the first three requests is mostly the next request's prefill |

## ~16K

At ~16K the server prefills the four prompts one after another (each is longer than the batched-prefill budget) in 2,048-token chunks and gives the decode batch one step between chunks, so a request that is decoding while another prompt prefills advances about one token per chunk, and its 128 tokens span the next request's whole prefill. The fix does not change this (5.6 to 7.1 tok/s per request), because the decode step was never the limit there.

The server trace of the batch-4 ~16K level (first token of the first request to the end of the last, 172.4 s) is 98.6% GPU-busy, and the time is prefill attention:

| Kernel class | GPU time | Share of the 172.4 s window |
|---|---:|---:|
| SDPA (129.3 s of it `kernel_sdpa_flash_wmma<__half, true, 128, 64, 64>`, 672 calls, about 192 ms each) | 134.9 s | 78% |
| Quantized GEMMs (prefill) | 19.7 s | 11% |
| Decode qmv | 10.2 s | 6% |
| Everything else | 5.1 s | 3% |

A single 13,787-token prefill is the same: 49.7 s wall, 41.5 s of it SDPA (84%) and 6.4 s GEMM, against 0.7 s with SDPA at 22% for ~1K. One flash-attention call there is a 2,048-query chunk against up to 16K keys, roughly 3e11 FLOPs, so 192 ms per call is on the order of 1.5 TFLOP/s, which suggests the kernel, not the device, sets the rate (not established here). Prefill throughput is outside this issue; it is filed as lablup/mlxcel#2251 with this evidence.

## Reproduce

```bash
cargo build --release --features rocm --bin mlxcel-server --bin mlxcel-bench-decode \
    --example qmm_batch_rows_probe --example profile_batched_decode
scripts/rocm_gpu_guard.sh -- target/release/examples/qmm_batch_rows_probe 4
scripts/rocm_gpu_guard.sh -- target/release/examples/profile_batched_decode \
    -m models/mlx/Meta-Llama-3.1-8B-Instruct-4bit --batch-sizes 1,4 --prompt-len 1024
# server matrix: data/rocm-batched-decode-gfx1151-2026-10-08/harness/run_matrix.sh
cargo test --release --features rocm --test rocm_qmm_batched_rows -- --test-threads=1
```
