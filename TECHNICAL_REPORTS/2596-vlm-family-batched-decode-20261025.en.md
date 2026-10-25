# Technical Report: PR #2596 — VLM family batched decode

**Date:** 2026-10-25

**Related issues:** #2563, #2566, #2567, #2587

## Executive Summary

Six sequence-aware VLM text towers now execute native grouped decode: GLM-4V/GLM-OCR, GLM-4V-MoE, Hunyuan-VL, ERNIE-4.5-MoE-VL, PaddleOCR-VL and Cohere Compass. Seven concrete vision wrappers delegate text-only decode to these tower entry points. Rows are grouped by plain versus multimodal-RoPE state, weight-heavy projections run once per compatible group, request-local attention caches remain isolated, and logits are scattered back to the scheduler's original row order.

The implementation extracts the proven Qwen VL position/group/cache/scatter machinery into shared helpers and preserves each tower's distinct rotary, normalization, expert and attention-window rules. Prefill, missing or mismatched sequence IDs, stored explicit positions, and unsafe model-global Compass DeepStack state retain the exact per-row fallback. Metal checkpoint parity and throughput remain explicitly deferred to #2587.

## Problem Statement

These VLM towers had enough sequence state to support batching, but their serving decode path still inherited a per-row forward. A naive concatenation would be incorrect because rows can carry different MRoPE deltas, plain text and multimodal rows can require different rotary paths, cache histories have unequal lengths, and the scheduler expects logits in its original request order rather than grouping order.

The families also differ materially. Hunyuan applies learned Q/K normalization after RoPE and uses four position axes. GLM rotates only a configured prefix and must preserve its even/odd or half-split rotary pairing. ERNIE uses interleaved 3D MRoPE and separate text and multimodal expert banks. Compass combines full and sliding-window attention, optional RoPE per layer type, and model-global visual/DeepStack state that is not sequence-isolated yet.

## Technical Decisions

### Group only compatible decode rows

The native entry point requires single-token decode width and a complete, row-aligned sequence-ID list. It reads each row's monotonic cache offset and stored rope delta, builds `[axes, B, 1]` positions, and divides rows into plain and MRoPE groups. Each group preserves ascending request order, borrows only its cache rows, runs a shared model graph, then scatters logits into slots for the original scheduler order. Family-specific diagnostics reject missing cache layers or unproduced rows.

Rows with explicit stored position tensors cannot be represented by the scalar-delta fast path. Prefill-width calls, absent or mismatched IDs, and those stored-position cases use the established per-row dispatcher instead of approximating positions. This fallback is a correctness boundary, not a batching failure: only compatible rows claim native grouping.

The grouping, row selection, cache selection, scatter and concatenation helpers were moved into the shared Qwen MRoPE state module. Existing Qwen 2 VL, Qwen 3 VL and Qwen 3 VL MoE callers now reuse the same functions, with regression tests proving that their plain/MRoPE grouping, cache updates and wrapper behavior did not drift.

### Keep rotary and model semantics inside each tower

The common mechanism supplies rows, positions and caches; each tower still owns the mathematical transform. Hunyuan keeps Q/K normalization after the four-axis rotary transform. GLM-4V and its MoE variant rotate only the configured partial dimensions and retain their respective pair layout. ERNIE applies adjacent-pair interleaved 3D MRoPE in f32, always selects the text expert bank at decode width, and leaves image-placeholder expert selection in prefill. Its existing stacked-expert quantization geometry guard remains active.

Compass initially assumed that sliding layers always used RoPE and full layers never did. Review identified that this was not a valid configuration invariant. The final implementation derives optional cosine/sine tables independently for the first sliding and first full layer types and then reuses the matching optional table for each layer. Windowing remains controlled by the attention layer itself, so sliding-NoPE, full-MRoPE and the standard variants all preserve their single-row semantics.

Compass DeepStack inputs are still stored globally on the model. When that state is present, the entire call follows the exact per-row path. This avoids misrepresenting global state as sequence isolation; converting it to request-owned state remains tracked by #2567.

## Validation

Nine tower/configuration tests cover all six towers plus Compass sliding-NoPE, full-layer-MRoPE and global-DeepStack fallback variants. Seven actual wrapper fixtures exercise each concrete wrapper. All synthetic tower fixtures use nontrivial weights and at least three unequal prompt histories, covering plain rows, different per-sequence deltas, mixed group/scatter order, stored-position fallback, exact B=1 dispatch and B=0.

The ERNIE expert-bank guard passed one focused test. Qwen regression coverage passed two shared-helper tests, one Qwen 3 VL MoE tower test and one five-wrapper integration test. The final classification ledger passed 9 tests, the logit-trace example passed 7 tests, and the library test suite compiled successfully with `--no-run`. Formatting and diff checks passed.

Repository source gates passed for workspace crate versions, documentation coverage for 320 `MLXCEL_*` names, 15 environment-documentation tests, and the Llama compatibility manifest plus all validator cases. Full-workspace, all-target Clippy with warnings denied passed in 2 minutes 5 seconds. The final architecture, correctness, security and performance review reported no findings after the Compass rotary-table correction.

No real checkpoint, Metal trace, greedy-generation comparison or server concurrency benchmark was run. Those hardware acceptance checks remain in #2587, so this report makes no measured speedup or production-parity claim.

## Learning Points

Sequence-aware batching is a partition-and-scatter problem as much as a tensor-shape problem. Rows may share weights while retaining different position state and cache ownership; grouping must therefore be explicit, stable and reversible. Falling back when stored positions cannot be losslessly reduced to scalar deltas is safer than broadening the native path with an approximate representation.

Attention type and rotary type are independent configuration axes. Compass demonstrated why “sliding implies RoPE” or “full implies no RoPE” is an unsafe shortcut: window size decides which keys are visible, while the optional rotary table decides how Q/K are encoded. Reviewable helpers should expose those choices independently.

## Change Summary

| Item | Value |
|------|-------|
| Files changed | 26 |
| Lines added | 2,957 |
| Lines deleted | 303 |
| Native text towers | 6 |
| Delegating wrappers | 7 |
| Commit | `9aae72bc` |

The PR adds grouped native decode to the six text towers, delegates seven wrappers, centralizes the Qwen-derived MRoPE grouping/cache/scatter helpers, expands runtime and coverage regressions, and documents the new native families. Metal acceptance remains under #2587, while sequence-isolating Compass DeepStack state remains under #2567.
