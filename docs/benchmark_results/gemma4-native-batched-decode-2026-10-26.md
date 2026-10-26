# Gemma 4 native batched decode on Metal

Issue #2546, PR #2611. Measurements use an Apple M3 Ultra with 512 GiB unified memory, release builds with `metal,accelerate`, and the `mlx-community/gemma-4-12b-it-4bit` and `mlx-community/gemma-4-26b-a4b-it-4bit` checkpoints. The baseline is `01047d75`; the final measured source is `3777cbca` (production code from `f5ea2523`). The intervening base update `afdd588d` changes Windows ROCm packaging, not Metal inference.

## Method

Each checkpoint runs alone. The server uses `--parallel 8 --ctx-size 32768 --ignore-eos --cache-ram 0`, with a one-request, 32-output-token warmup. `scripts/bench_serving_concurrency.py` then runs concurrency 1/2/4/8 with approximately 512 input tokens (445 after tokenization) and 128 output tokens per request. All requests succeeded. The metrics endpoint returned 501 in these builds; batch path attribution comes from the model's decode classification and server log, not metrics counters.

Teacher-forced traces use `docs/architecture.md`, arguments `64 4 8 512`, and `MLXCEL_TRACE_BATCH_ROWS=1` versus `=4`: 256 positions at decode width one, with independently prefixed rows. The decided-position threshold is the comparator's default top-two logit gap of 2.0. These are single sweeps, not confidence intervals.

## Serving results

Aggregate output tokens per second:

| Checkpoint | Revision | C=1 | C=2 | C=4 | C=8 |
|---|---|---:|---:|---:|---:|
| Gemma 4 12B dense / Unified | Baseline | 50.5 | 51.3 | 51.4 | 49.5 |
| Gemma 4 12B dense / Unified | Native batch | 51.5 | 80.3 | 108.0 | 120.1 |
| Gemma 4 26B A4B MoE | Baseline | 95.8 | 127.8 | 154.2 | 127.1 |
| Gemma 4 26B A4B MoE | Native batch | 96.2 | 147.4 | 198.7 | 236.7 |

At C=4, native decode reaches 2.10 times its C=1 aggregate rate for 12B and 2.07 times for 26B. Against the baseline's C=4, the gains are 2.10 times and 1.29 times respectively. C=1 differs by +2.0% and +0.4% in these sweeps; the implementation retains the single-sequence graph. The 12B baseline disables batching at the Unified wrapper and serializes requests, which also increases queueing latency.

## Exactness and numerical differences

`mlxcel generate -p Hello -n 128 --temp 0` produced byte-identical completion text before and after for both checkpoints (both stopped at EOS after nine generated tokens). Native B=1 also has a zero-tolerance synthetic comparison with the existing single-sequence forward.

| Checkpoint | Native B=4 vs B=1 top-1 disagreement | Decided-position disagreement | Largest reference gap at disagreement |
|---|---:|---:|---:|
| 12B | 19/256 (7.42%) | 2/152 (1.32%) | 5.531 |
| 26B A4B | 25/256 (9.77%) | 3/157 (1.91%) | 3.125 |

The batched and single-row numeric results are therefore not identical. These measured differences must not be described as zero-disagreement or bit-exact batched inference.

A controlled 12B rerun with `MLXCEL_QMV_WIDE=0` produces identical B=1/B=4 logits at all 256 positions, with zero top-1 or decided-position disagreements and zero reference-token logit delta. Its B=1 trace is identical to the default B=1 trace. The entire default 12B discrepancy is attributable to Metal's width-dependent quantized projection kernel, rather than cache offsets or native batching logic. The serving numbers above retain the default kernel; this diagnostic does not change the shipped default.

For 26B, disabling `QMV_WIDE` leaves 18/256 top-1 disagreements and 2/157 decided disagreements (largest reference gap 2.188; perplexity +11.9% relative to B=1). B=1 itself remains identical with or without the switch. This control does not eliminate all MoE differences: single-row fused experts and batched `gather_qmm` still use different numeric paths. This is consistent with that intentional kernel change, but is not proof that every remaining difference has been attributed to one kernel. The default-kernel numbers above remain the acceptance evidence.

The original 12B Unified wrapper has no identified batch override and reports batching unsupported. Calling its trait-default `forward_batched_with_ids` directly from the trace harness discards the sequence IDs, although prefill populated identified state. Consequently, its baseline B=1 and B=4 traces read the anonymous state and are not valid references for identified decoding: they disagree at 210/256 positions, including 87/106 decided positions. The new identified override fixes this routing. Comparing the new B=1 trace against that old trace is not a single-sequence regression measurement; the ordinary generation comparison and exact dispatch regression test are the relevant checks. The baseline 26B wrapper preserves IDs and has zero B=4/B=1 disagreement before native batching.

## Coverage and limitations

The checkpoint-free Metal regression uses three rows with distinct prompt lengths and covers sliding-window wrap, full attention with proportional RoPE, KV-shared attention, MoE, per-layer inputs, B=1, and empty batches. A separate regression preserves the historical anonymous model-owned fallback behavior. Pending image/audio per-layer inputs keep wrapper dispatch, including inputs not yet bound to a sequence ID.

ROCm MI300X and CUDA GB10 were not measured in this run; no configured remote host was available. These Metal numbers do not establish performance or numerical parity on those backends. Tensor-parallel Gemma 4 and paged decode remain outside this change.

The workspace gate reported 13,811 passed, 10 failed, and 441 ignored tests. The ten failures are unchanged BF16/CPU oracle tests in SAM3, SAM3D Body, Sapiens2, and DINOv2. A separately built clean `01047d75` baseline reproduced all ten failures in isolated processes with identical assertion locations and values, ruling out this change and test concurrency. Full Clippy and formatting passed; focused native batch, cache-pool duplicate-ID, and model-owned state regressions passed. The full suite is not reported as green.
