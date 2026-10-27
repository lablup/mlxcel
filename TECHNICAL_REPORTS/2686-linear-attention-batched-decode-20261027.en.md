# Technical Report: PR #2686 — Linear-attention batched decode

**Date**: 2026-10-27
**Status**: Measurements complete; numerical differences documented; integrated validation pending

## Executive Summary

Issue #2572 adds native identified multi-row decode for Qwen3-Next, Qwen3.5, Bailing MoE Linear, Kimi Linear, and Kimi K3. Six measured configurations cover four checkpoints: both Qwen3.5 checkpoints have default and explicit dense storage runs. Every greedy result and all 256 B1 trace data rows exactly match baseline; all candidate C4 aggregate rates exceed candidate C1. Numerical differences below are retained rather than described as universal equivalence.

## Problem Statement

Model-owned recurrence previously serialized decode across requests; Qwen3.5 supported batching but decoded individual rows sequentially. Concurrent execution must preserve per-request recurrence, attention history, and MRoPE positioning.

## Technical Review and Decisions

Reuse shared recurrent stack/split and split-attention helpers, preserving scalar B1 routing and ID validation before mutation. Mixed initialization, row-local attention/MLA, Qwen3.5 prefill/storage policies, and per-row MRoPE retain their semantics. Qwen3.5 and Kimi K3 vision wrappers delegate to their text towers.

A HIGH review finding identified Qwen3.5 GDN recurrence incorrectly counted as paged attention history. `413e4649` records zero for recurrent slots while retaining actual attention lengths. Seven focused GPU regressions and 14 coverage checks passed, including corrected accounting; scoped clippy and formatting passed. Final integrated workspace validation remains pending after #2572 merge and #2573 rebase; this report does not claim merge or a full-suite pass.

## Change Summary and Measurements

Measured release: `413e4649ebe9451fc73a7f4d03c492807a4e9148`; baseline model source `a29a2692`. Later `9b6265f4` changes tools only. Host: Apple M3 Ultra, 512 GB, macOS 26.6.2; release `metal,accelerate`. Raw runtime manifests preserve binary and selected loaded Metal library hashes. Legacy baseline hash gaps are explicitly marked.

| Configuration | Baseline C1/C2/C4 tok/s | Candidate C1/C2/C4 tok/s | B4 top-1 | Decided | PPL change |
|---|---|---|---|---|---|
| qwen-next | 69.7/66.6/80.6 | 71.1/103.3/148.1 | 8/256 | 0/126 | +0.523% |
| ring-mini | 140.0/179.2/177.8 | 159.6/298.8/460.8 | 22/256 | 1/118 | -2.052% |
| qwen35-4b | 109.5/188.8/168.6 | 112.6/225.3/357.9 | 2/256 | 0/117 | +0.025% |
| qwen35-moe | 87.4/138.8/122.5 | 89.3/165.7/262.9 | 3/256 | 0/113 | +0.634% |
| qwen35-4b-dense | 110.9/191.8/170.3 | 112.7/225.3/361.8 | 2/256 | 0/117 | +0.025% |
| qwen35-moe-dense | 88.7/146.2/123.9 | 89.5/166.5/262.9 | 3/256 | 0/113 | +0.634% |

Ring has 22/256 top-1 disagreements, including **1/118 decided positions**, with largest disagreement gap **2.312**. This is a **nonzero decided disagreement**; the issue requires recorded magnitudes rather than a zero-disagreement acceptance threshold. The -2.052% PPL change does not erase that difference. Other measured configurations have zero decided disagreements. No full numerical or general quality equivalence is asserted. All candidate startup logs report native batching without a clamp. Single ladder samples do not establish statistical equivalence of B1 performance.

A separate `MLXCEL_FUSED_MOE=0` diagnostic for both B1/B4 retains 22/256 top-1 differences but has 0/114 decided differences (maximum disagreement gap 1.25); PPL is 60.6904 → 57.1854 (-5.775%). Review identified B1 full-fused versus B4 short gate/up and gather-QMM arithmetic. Disabling fusion removes the decided mismatch while changing B1 PPL and rounding; this is not proof of quality equivalence. Production numbers and throughput above are unchanged. The diagnostic driver and release provenance are archived separately.

Qwen3.5 original freeform VLM is `[true, false, true]` for both baseline and candidate: counting-response reasoning drifts under concurrency before and after. HTTP phases all succeed; exit 1 denotes response nonidentity. Baseline-to-candidate solo full messages are 3/3 exact; concurrent full messages are `[true, false, true]`. This is not a 3/3 VLM exactness pass. Image-conditioned teacher-forced B1 is exact across all 64 rows; B4 differs at 2/64 top-1 positions and 0/15 decided positions, with PPL 82.5008 → 81.7597 (-0.898%).

Serving prompts request approximately 512 tokens; actual tokenized lengths vary, with generation limit 128. Text traces use four 64-position chunks and maximum history 512, not fixed 512-token context. Image traces are separate 64-position runs. The affine-quantized decided-position check alone cannot establish general quality equivalence. Kimi Linear and Kimi K3 have synthetic coverage only; real-checkpoint measurements remain unavailable.

## Follow-up and Evidence

Complete the final integrated gate and review Ring/VLM differences against the issue criteria. [Structured results](../benchmarks/epic-2574/linear-attention.json) link each raw run. Raw evidence preserves phase outputs/stderr/status, original runtime manifests and their referenced logs, source and input hashes, exact VLM requests, fixture images and replay drivers. The raw root manifest covers the union with earlier epic evidence. No historical binary hash is inferred from a current build.
