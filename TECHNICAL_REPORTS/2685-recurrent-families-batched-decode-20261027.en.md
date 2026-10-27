# Technical Report: PR #2685 — recurrent-family batched decode

**Date**: 2026-10-27
**Status**: Measurements and targeted validation complete; final integrated gate pending
**Risk Level**: Medium

## Executive Summary

Mamba1, Jamba, LFM2, RecurrentGemma and RWKV7 gain native decode over independent sequence-owned recurrent state. Four real checkpoints show concurrency-four aggregate throughput gains of 2.22–3.27x over baseline. LFM2-VL's two text requests plus one image request match their solo outputs exactly when served concurrently.

## Problem Statement

Existing recurrent state was already stored per sequence, but the server restricted these families to one decode row. Correct batching requires stacking and splitting each recurrence, maintaining ragged attention histories, preserving scalar behavior, and rejecting invalid identities before state mutation.

## Technical Decisions and Review

Reuse the shared `recurrent_batch` helpers from #2571. Embedding, projections, recurrence, feed-forward/expert work and output heads execute across rows; attention keeps sequence-specific KV offsets. Mixed initialized/fresh state uses identified row fallback. Copied split state prevents one request from retaining another request's batched buffer. Paged shadow accounting counts attention storage and leaves recurrent slots at zero; B=1 preserves the existing scalar route.

Implementation and security/performance reviews passed after one MEDIUM fix: LFM2-VL's non-ID `forward_batched` must delegate to its text tower, whose multi-row path rejects missing identities. Otherwise the trait fallback could reuse shared recurrent state. Commit `a9395b7c` restores this delegation; `898108fe` strengthens the regression by comparing identified outputs after a rejected call, demonstrating atomic rejection. These changes do not alter the measured identified serving path.

## Measurements

Apple M3 Ultra, 512 GB, macOS 26.6.2; release `metal,accelerate`. Baseline model source is `a29a2692` with the scalar-reference trace-only patch; all candidate binaries were measured at `2bd2eff00fbf9dc0f8b8a1c0a5c1c310ff31ffd9`. Later changes are the non-ID wrapper correction and test-only regression above. All four greedy outputs are identical, and every data row of their 256-position B=1 traces matches baseline exactly.

| Checkpoint | Baseline aggregate C1/C2/C4 tok/s | Candidate aggregate C1/C2/C4 tok/s | B4 top-1 mismatch | Decided mismatch | PPL change |
|---|---|---|---:|---:|---:|
| AI21 Jamba Reasoning 3B 4-bit | 128.6 / 167.0 / 167.4 | 144.5 / 290.3 / 475.9 | 1/256 | 0/136 | +0.061% |
| Falcon Mamba 7B 4-bit Instruct | 38.8 / 37.5 / 78.8 | 38.8 / 45.8 / 195.5 | 2/256 | 0/105 | -0.027% |
| LFM2 350M 8-bit | 399.6 / 405.9 / 434.1 | 430.0 / 712.1 / 1417.4 | 4/256 | 0/94 | -0.457% |
| LFM2 8B A1B 4-bit | 88.3 / 109.9 / 202.7 | 111.9 / 137.5 / 450.3 | 0/256 | 0/117 | 0.000% |

Serving uses `--parallel 4`, requested nominal ~512-token prompts and a 128-token generation cap. Actual tokenizer counts vary. Traces use four 64-position chunks and maximum history 512; this is not a fixed 512-token context. Decided means reference top-two gap >=2. Zero decided disagreements does not imply all logits or free-running text are identical. All four candidate servers logged native batching without a clamp. These individual ladders do not establish statistical throughput equivalence.

LFM2-VL 450M 4-bit passes the original free-form workload 3/3 on both baseline and candidate. The baseline clamps to one row; the candidate reports native decode without a clamp. The durable runner preserves exact request settings, full messages including reasoning, the copied image and its SHA-256, server commands, timestamps, source/binary/library hashes and success/error status. Historical driver versions are retained to match recorded driver hashes. The runner now probes for a live listener rather than binding the port, avoiding false rejection of a closed connection in TIME_WAIT.

## Validation, Limits and Evidence

Targeted recurrent isolation, mixed/fresh state, scalar/native transitions, invalid-input atomicity, cache accounting and wrapper delegation tests passed, as did targeted clippy and formatting. The VLM evidence runner's eight CPU tests cover input reconstruction, reasoning-sensitive equality, errors, overwrite protection, cleanup and the port-probe regression. RecurrentGemma and RWKV7 have synthetic coverage only; no real-checkpoint performance or output claim is made for them. The final integrated workspace gate remains pending after #2572 is merged; this report does not claim the PR is merged or the full suite has passed.

Machine-readable results are in `benchmarks/epic-2574/recurrent-families.json`. Raw baseline/candidate text runs and LFM2-VL records retain original runner manifests as `run-metadata.json` where available; older text baselines explicitly lack runtime hashes. Every archived artifact is indexed by SHA-256. Reproduce the VLM workload with `scripts/bench_batched_decode_vlm.py`; keep its actual loaded Metal library with relocated binaries.

## Change Summary

Five recurrent model families, LFM2-VL delegation, batching coverage and documentation gain native decode support. Shared recurrent helpers are reused without new kernels or dependencies. Related: #2573, #2574, #2571 and #2555.
