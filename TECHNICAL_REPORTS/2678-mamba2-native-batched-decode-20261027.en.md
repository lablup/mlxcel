# Technical Report: PR #2678 — Mamba2 native batched decode

**Date**: 2026-10-27
**Status**: Validation complete; awaiting merge
**Risk Level**: Medium

## Executive Summary

Mamba2, Nemotron-H, Granite Hybrid, Falcon-H1 and PLaMo2 gain native decode with independent recurrent state and ragged attention caches. Five completed serving ladders show concurrency-four aggregate gains of 2.00–3.43x over baseline. Nemotron's admission fix passes the full serving ladder. Granite Vision retains per-row wrapper fallback to preserve exact free-form output; its final three-request concurrent test matches all solo outputs while its text tower remains native.

## Problem Statement

These families already owned state per sequence but clamped serving to one decode row. Enabling batching requires stacking each recurrence without sharing state, preserving scalar execution, accounting for retained attention memory, and keeping provider admission behavior aligned with the scheduler.

## Technical Decisions

The shared `recurrent_batch` module concatenates axis-zero rows and copies split results so a sequence does not retain a whole batched buffer. Mixed initialization falls back to identified row execution rather than inventing zero state. Embedding, projections, recurrence and feed-forward work batch together; attention retains each sequence's offsets. Invalid batch identities are rejected before state mutation. Paged shadow storage counts attention lengths and leaves recurrent slots at zero.

Mamba2's native SSD carry removes an erroneous extra broadcast axis and canonicalizes the scalar carry's optional singleton token axis only when stacking. B=1 preserves its existing graph; a native→B1→native regression covers the rank transition. Nemotron's fused projection gate requires affine-quantized input/output projections with biases, retaining eligible production dispatch while routing float and non-affine projections through the existing fallback. Nemotron Nano Omni delegates native decode to its text tower. Granite Vision uses identified per-row wrapper decode; its text tower remains native, and both wrappers synchronize model-owned storage.

Real serving exposed a Nemotron admission mismatch: the HTTP provider awaited a single-slot request-memory handshake although the multi-row scheduler uses ordinary paged admission and emits no such event. Commit `ce00333c` restricts that waiter to configured single-slot schedulers and adds a Done-only provider regression. The corrected server passed the full serving ladder. The earlier HTTP-500 run remains excluded from performance results.

## Validation and Measurements

Focused GPU tests (16), library/tests checks, and post-rebase coverage (13), wrapper tests (2), formatting and targeted strict clippy passed. Tests cover six-step isolation, exact B=1, empty/malformed batches, fresh/mixed states, native/scalar transitions, Granite positional modes, paged accounting and VLM delegation. After integration fixes, 10,603 main-library tests and integration tests passed. The seeded core device-lock regressions passed 14/14. A corrected core rerun passed 2,199 tests (4 ignored), core doctests passed 2 (6 ignored), and remaining crate tests passed 618 (19 ignored). Main doctests passed 1 test (4 ignored), with command exit 0. These results provide split coverage rather than a single uninterrupted workspace command returning success. Full-workspace all-targets clippy passed at `098b569a`; corrected core-test clippy and formatting passed after `12182cad`. Subsequent source-guard and stale Nemotron VLM assertion corrections change tests only, not runtime behavior.

Host: Apple M3 Ultra, 512 GB, macOS 26.6.2; release `metal,accelerate`. Four text families use baseline `a29a2692` and candidate `00e33351bf5bb1ac63f9d0b6ce22f3845caf272a`. Main PR #2679 subsequently changed Nemotron Metal sorted prefill, so Nemotron alone was rerun against a fair shared baseline `b4cdf023` and candidate `1249c337`. The other four implementations were unchanged by that merge. Baseline traces use the corrected explicit scalar-reference instrumentation, which preserves sequence IDs after identified prefill. All five checkpoints have byte-identical greedy text and identical recorded 256-position B=1 traces, with zero recorded logit delta.

| Checkpoint | Baseline aggregate 1/2/4 tok/s | Candidate aggregate 1/2/4 tok/s | B=4 top-1 mismatch | Decided mismatch | Trace PPL change |
|---|---|---|---:|---:|---:|
| granite-4.0-h-tiny-4bit | 97.3 / 109.1 / 129.9 | 120.0 / 173.3 / 260.2 | 10/256 | 0/105 | +0.755% |
| NVIDIA-Nemotron-3-Nano-30B-A3B-4bit | 94.0 / 71.0 / 91.8 | 88.6 / 211.4 / 313.5 | 6/256 | 0/113 | +0.361% |
| mamba2-1.3b-4bit | 85.0 / 92.3 / 100.0 | 98.2 / 157.9 / 283.4 | 1/256 | 0/109 | +0.053% |
| Falcon-H1-Tiny-90M-Instruct-4bit | 180.8 / 194.4 / 165.5 | 208.0 / 324.7 / 567.2 | 4/256 | 0/92 | -0.448% |
| plamo-2-1b | 88.3 / 95.2 / 103.8 | 99.3 / 129.2 / 273.5 | 0/256 | 0/97 | 0.000% |

Five warmed Nemotron B=1 decode samples were 141.1, 141.3, 139.8, 139.8, 140.3 tok/s on baseline and 140.0, 139.4, 138.5, 137.5, 137.2 on candidate. Medians were 140.3→138.5 tok/s (-1.3%), with overlapping ranges; these five samples do not establish statistical equivalence. Repetitions finished at 09:41:27 UTC, before focused GPU tests started at 09:42:45 UTC.

Concurrency is client concurrency; aggregate throughput includes serving overhead. Decided means reference top-two gap >= 2. Zero decided mismatches does not imply identical logits or identical free-running text. Serving uses `--parallel 4`, a nominal ~512-token textual prompt requested by `--prompt-tokens 512`, and a 128-token cap. Actual tokenized prompt lengths vary by tokenizer (Nemotron logged 448). Traces use `MODEL CORPUS 64 4 8 512` with B=1/B=4; 512 is the maximum prefill/history budget, not a fixed context at every position. Chunks begin at corpus position zero and available history grows. The baseline uses `MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1`; the native candidate does not.

## VLM Results and Final Verification

Nemotron Nano Omni's two concurrent text requests plus one image request matched their solo outputs in all three cases. Granite Vision's earlier native-wrapper free-form diagnostic matched only the image request: `[false, false, true]`, while baseline matched all three. Constrained Granite prompts matched all three, but do not replace the failing original free-form criterion. That native wrapper's image-conditioned forced trace had 4/64 top-1 mismatches, 0/11 decided mismatches, largest disagreement gap 0.125, and perplexity 122.3245→122.9666 (+0.525%).

The final Granite Vision wrapper uses `PerRowFallback` while its text tower retains native batching. Rerunning the original free-form workload produced `[true, true, true]`; the server reported per-row fallback with no clamp. Final Nemotron Nano Omni also matched all three original free-form requests and reported native batching without a clamp. Thus exact VLM output equality is verified for these workloads without relaxing the original requirement. The earlier Granite native diagnostics remain recorded to explain the fallback; they are not measurements of the final fallback path. Both VLM workloads were replayed successfully at `098b569a9aac3eac151d46f721a5b88cb1964478`, after main `cfa14e7b` changed vision loading, retaining 3/3 exact matches and the same respective fallback/native routes without a clamp. The new raw records include command/source provenance plus the observed server-binary and embedded Metal-library hashes; earlier diagnostics keep their original revisions. Validation is complete through the split verification described above. HEAD `12182cad` adds only test changes after the measured integrated runtime `098b569a`.

## Change Summary

The shared recurrent helper, five model towers, two VLM wrappers, isolation harness, coverage inventory, provider admission gate and continuous-batching documentation changed. No new SSM kernel or dependency is introduced. Related: #2571, #2574, #2555; follow-up families reuse the shared helper. Compact evidence: `docs/benchmark_results/data/native-batched-decode-m3ultra-2026-10-27/mamba2-families.json`.

## Integration fixes

Full-suite parallel execution exposed unsynchronized BitLinear kernel initialization. Metal, CUDA and ROCm holders now use `std::call_once`; five BitNet tests passed concurrently on Metal without weakening assertions. CUDA/ROCm runtime was not exercised on this host. Paged decode counter assertions use a thread-local test observer while public production metrics remain process-global. Existing strict B1 and wrapper capability assertions were corrected without changing numerical tolerances.
