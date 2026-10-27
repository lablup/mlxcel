# Technical Report: PR #2670 — GPT-OSS native batched decode

**Date**: 2026-10-27
**Status**: Completed for the measured 20B checkpoint
**Risk Level**: Medium

## Executive Summary

GPT-OSS now keeps mixed full/sliding KV caches per sequence and batches transformer projections and experts during decode. On an Apple M3 Ultra with 512 GB, concurrency-four aggregate throughput increased from 110.1 to 209.0 tokens/s for `mlx-community/gpt-oss-20b-MXFP4-Q4`.

## Problem Statement

The former wrapper reset one shared cache vector at admission and correctly disabled batching to prevent requests from overwriting each other. Enabling concurrency required independent state, correct wrapped-cache positions, learned attention sinks, and server memory accounting.

## Technical Decisions

`ModelOwnedSequenceState` owns each request's caches. B=1 retains the existing forward graph and lazy state creation; B>=2 validates shape and sequence identities before any mutation. Multi-token or masked calls retain row-wise execution. Native decode batches embedding, normalization, projections, experts and output head; only ragged sink-aware attention remains per row because the generic attention interface does not carry learned sinks.

Review added atomic rejection of duplicate/missing batch identities and synchronization of actual full-cache `live_len` and rotating-cache `visible_len` into paged shadow storage. Without the latter, the default paged override could underestimate concurrent KV usage. Padded prefill remains disabled, and distributed pipeline/tensor-parallel runtimes are unchanged.

## Validation and Measurements

Focused model, shared-state, coverage, and harness tests passed, including unequal prompts, wrapped sliding windows, YaRN, sinks, admission/reset isolation, exact B=1, invalid-input atomicity and paged shadow lengths. Formatting, targeted clippy and the final 12-test batching coverage gate passed.

Release builds used `metal,accelerate`. Baseline was `a29a2692`; measured implementation was `dfd22beb`, subsequently rebased without changing the GPT-OSS implementation. Greedy `Hello`, temperature zero, 128-token-cap output was byte-identical. The 256-position B=1 trace was identical to baseline, including zero recorded logit delta. B=4 disagreed at 8/256 top-1 positions, all with reference gap at most 0.25; decided positions (gap >= 2) disagreed at 0/96. Perplexity changed from 17.5500 to 17.5108.

| Concurrency | Baseline aggregate tok/s | Native aggregate tok/s |
|---|---:|---:|
| 1 | 109.8 | 108.3 |
| 2 | 110.1 | 155.5 |
| 4 | 110.1 | 209.0 |

Five warm B=1 repetitions gave median decode throughput 136.1 versus 135.4 tok/s (-0.5%), with overlapping observed ranges. This supports no material single-request regression in this run, not an exact throughput guarantee. The server logged `native batched forward` and no clamp. The 120B checkpoint was not measured.

Reproduce with `scripts/bench_serving_concurrency.py --concurrency 1,2,4 --prompt-tokens 512 --max-tokens 128` against `--parallel 4`, and `examples/logit_trace MODEL docs/architecture.md 64 4 8 512` under `MLXCEL_TRACE_BATCH_ROWS=1` and `=4`. The compact measurements are in `benchmarks/epic-2574/gpt-oss-20b.json`.

## Change Summary

The wrapper, native sink-attention path, model-owned identity validator, cache-accounting hook, batching tests, coverage inventory and continuous-batching documentation changed. The shared harness now distinguishes scheduler-owned KV from model-owned placeholder caches. Related: #2568, #2574, #2555.
