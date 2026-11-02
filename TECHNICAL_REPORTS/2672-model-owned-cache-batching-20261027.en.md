# Technical Report: PR #2672 — model-owned cache batching

**Date**: 2026-10-27
**Status**: Completed for measured checkpoints; Step3p5/Step3p7 synthetic validation only
**Risk Level**: Medium

## Executive Summary

Ministral3, ExaOne4, Mellum, Laguna and Step3p5 now isolate request caches and batch decode projections, normalization, feed-forward/expert work and output heads. Step3p7 delegates text and embedding admission to its sequence-owned tower. Measured concurrency-four aggregate gains range from 1.56x to 1.92x.

## Problem Statement

The original single `RefCell` cache set was reset by each new request. The server therefore clamped batching to one sequence. Correct concurrency also requires ragged full/sliding attention, exact scalar behavior, invalid-input atomicity and paged-memory accounting.

## Technical Decisions

Use `ModelOwnedSequenceState` with row-local attention and shared native transformer work. Preserve the B=1 graph and lazy initialization. Keep each family's established attention details: Ministral's full-cache window hint, ExaOne local/global behavior, Laguna sinks and gates, and Step3p5 partial RoPE/head gating. Padded prefill remains disabled and Laguna DFlash remains independent.

Review added pre-route dimension/cardinality validation and atomic rejection of duplicate or missing identities for B>=2 fallback. Each wrapper synchronizes full-cache live length and rotating-cache visible length into paged shadow storage; Step3p7 delegates this hook. Without synchronization, the server could undercount concurrently retained KV.

## Validation and Measurements

Focused mixed-cache isolation, six-step interleaved/native parity, exact B=1, empty/malformed calls, admission/reset, paged storage, Step3p7 embedding/decode delegation, Laguna DFlash, coverage, formatting and targeted clippy passed. The final batching coverage gate passed all 12 tests after integration with #2670.

Measurements used release `metal,accelerate`, Apple M3 Ultra 512 GB, baseline `a29a2692`, and implementation `4cb59177` before a rebase preserving the model code. All four greedy outputs and 256-position B=1 traces match baseline exactly, with zero recorded B=1 logit delta.

| Checkpoint | Baseline aggregate 1/2/4 tok/s | Native aggregate 1/2/4 tok/s | B=4 top-1 mismatch | Decided mismatch |
|---|---|---|---:|---:|
| exaone-4.0-1.2b-4bit | 226.2 / 226.0 / 226.1 | 219.0 / 332.3 / 434.0 | 2/256 | 0/100 |
| Ministral-3-3B-Instruct-2512-4bit text view | 121.4 / 119.0 / 115.5 | 120.9 / 168.5 / 203.3 | 3/256 | 0/114 |
| Laguna-XS.2-4bit | 93.8 / 94.4 / 94.1 | 89.8 / 119.2 / 170.6 | 21/256 | 0/123 |
| Mellum2-12B-A2.5B-Base | 23.0 / 51.0 / 52.3 | 33.2 / 69.3 / 81.8 | 0/256 | 0/184 |

Decided means reference top-two gap >= 2. Five warm B=1 repetitions gave median decode 272.3→271.0 tok/s for ExaOne and 109.1→109.9 for Laguna, with overlapping ranges. Initial ladder variability should not be interpreted as an exact single-request speed guarantee. All measured servers reported native batching without a clamp.

The supplied Ministral VLM checkpoint normally selects a different text implementation. The text-only view links unchanged checkpoint files, removes `vision_config`, and changes `architectures` to `Mistral3ForCausalLM`, retaining the outer model type and nested text config. Mellum's baseline trace over `docs/architecture.md` produced a non-finite NLL; keeping BF16 passed diagnostically. Production-setting comparisons instead use the fixed Rust source corpus `git show a29a2692:src/models/mellum.rs`, which passes in both arms. The document-corpus precision problem remains pre-existing and unresolved here. Step3p5/Step3p7 had no real local checkpoint.

Use `scripts/bench_serving_concurrency.py --concurrency 1,2,4 --prompt-tokens 512 --max-tokens 128` against `--parallel 4`. Trace with `MLXCEL_TRACE_BATCH_ROWS=1` and `=4` and arguments `MODEL CORPUS 64 4 8 512`. Compact evidence is in `docs/benchmark_results/data/native-batched-decode-m3ultra-2026-10-27/mixed-cache-families.json`.

## Change Summary

Five model wrappers, Step3p7 delegation, shared validation/harness support, coverage inventory and continuous-batching documentation changed. No new dependency or checkpoint conversion is required. Related: #2569, #2574, #2555, #2670.
