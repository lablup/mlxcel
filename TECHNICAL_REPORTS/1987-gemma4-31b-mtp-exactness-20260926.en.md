# Technical Report: PR #1987, Restore Gemma 4 31B MTP exactness

**Date**: 2026-09-26
**Status**: Implemented, reviewed and validated locally; pending merge.
**Languages**: Rust, Python, Markdown
**Risk Level**: High

## Executive Summary

This change restores temperature-zero exactness for Gemma 4 31B MTP verification on Apple Silicon. The previous block forward and the single-token decode chain used different full-attention widths, sliding-cache reduction order, and long-prompt prefill chunking, so the startup gate declined both QAT and non-QAT 31B checkpoints. The corrected path reproduces decode attention per query row, preserves the classic physical ring order, and matches the scheduler's prefill chunks while keeping the alternate arithmetic restricted to the measured 31B geometry.

M5 Max measurements show corrected QAT and non-QAT teacher-forced traces with zero top-1 disagreements across 256 positions, byte-identical free-running output on three 256-token prompts per target, and byte-identical 513-token long-prompt output against corrected classic decode. CUDA validation was unavailable because the GB10 host could not be accessed.

## 1. Problem Statement

Gemma 4 31B MTP verifies several candidate tokens in one forward, while classic decode advances one token at a time. The two shapes selected different reductions even with `qmv_wide` disabled. QAT first diverged at full-attention layer 41 and non-QAT at layer 5, proving the issue was not specific to the QAT checkpoint's 8-bit MLPs. Long-context probing then exposed a second mismatch in the sliding layers: speculative buffers held chronological keys while classic decode reduced over the physical rotating-ring order.

Long server prompts added a third source of drift. Classic serving prefills in configured chunks, 512 tokens in the measured server, but the MTP adapter had forwarded the full 6,449-token prompt in one call. The different prefix state caused corrected verification to disagree with default classic serving even after the attention fix.

Without the change, the exactness gate correctly fell back to classic decode and left the intended MTP path unavailable by default. Forcing `MLXCEL_MTP_ALLOW_INEXACT=1` bypassed the contract and could change greedy output at near-tie positions.

## 2. Technical Review

### 2.1 Correctness and Compatibility

The arithmetic correction is gated to Gemma 4 models with 32 query heads, 16 sliding KV heads of dimension 256, and 4 global KV heads of dimension 512. B=1 linear verification uses the correction; batched MTP declines to classic decode and nonlinear tree rounds use the supported linear dispatch path. Unified 12B does not inherit the correction merely because it also has 512-wide full heads.

The shared `RotatingKVCache` metadata path is used by Gemma 3, AFMoE, Gemma 4, and Muse Glimmer. The new reference-ring anchor survives rollback, compaction, scheduler slice reconstruction, snapshot restore, and detach/adopt. Unbuffered snapshots remain compatible. Buffered snapshots without the new anchor are rejected because their prior reduction order cannot be reconstructed safely.

Classic before/after smoke checks produced identical output for Gemma 3 4B and Qwen 2.5 7B. A Qwen 3.8 27B MTP check also produced identical output and identical acceptance counts before and after the change: 164 accepted of 274 proposals over 92 rounds. Qwen 3.6 MTP was not tested because the available local target and drafter had incompatible hidden sizes.

### 2.2 Performance

The corrected path trades some parallel attention work for exact M=1-compatible reductions. Free-running measurements were one sample per arm and prompt, in fixed order without cooldown, under thermal pressure reported as `Fair`. They establish output parity, not a stable throughput delta. Corrected MTP ranged from 0.85x to 1.38x corrected classic across the six QAT/non-QAT prompt cases; the adaptive profitability policy remains responsible for deciding when MTP should run.

The strengthened 31B startup probe took about 5.51 seconds in one QAT run, compared with about 2.26 seconds for the earlier short probe. Failure localization and the long buffered draw execute at startup, outside request forwards. Other Gemma geometries retain the original short probe.

### 2.3 Security

The change adds no network surface, authentication path, or unsafe block. Snapshot restore validates ring origins, logical bounds, K/V ranks, shapes, and dtypes before mutating cache state. Completed 31B MTP requests do not donate buffered rotating caches to the automatic prompt cache because ordinary suffix prefill cannot reproduce that layout exactly.

## 3. Technical Decisions

### 3.1 Reproduce decode attention instead of accepting numerical proximity

The implementation slices each verify query to its M=1-visible key prefix. Full-attention rows use the maskless single-query dispatch, and sliding rows reorder their visible window to the physical ring order classic decode would observe. Keeping the prior block reduction would have preserved more parallelism but failed the byte-identity contract.

### 3.2 Carry the reference ring position as cache metadata

Deriving the classic ring cursor from current speculative storage was insufficient after rollback, compaction, snapshot restore, or detach/adopt. The cache now records the logical offset and physical cursor at speculative buffering time and derives later cursors from that anchor.

### 3.3 Match prefill geometry at the adapter boundary

For the measured 31B geometry, the MTP adapter now consumes the scheduler's configured prefill chunk size and captures the assistant seed only from the final chunk. Offline generation uses the same `MLXCEL_PREFILL_CHUNK` policy as classic generation. Other Gemma variants retain their existing prefill path.

### 3.4 Localize failures only after the authoritative probe fails

The ordinary startup probe remains the verdict. On failure, an automatic diagnostic rerun captures attention, MLP, layer, norm, and head outputs and appends the first divergent layer, kind, and sub-operation to the decline reason. Request forwards never enter the capture scope.

## 4. Implementation Details

```text
Classic decode:       prefill chunks -> M=1 query -> physical rotating ring
Old MTP verification: whole prefill  -> M=K query -> chronological buffered keys
Corrected 31B MTP:    prefill chunks -> K x M=1-compatible rows -> reference ring order
```

The implementation adds query-row attention helpers, failure-only probe capture, cache anchor serialization and validation, scheduler prefill-size propagation, B=1/tree capability guards, regression tests, a Gemma-aware teacher-forced trace mode, and a reproducible measurement harness with machine-readable evidence.

## 5. Learning Points

An attention mask can preserve the visible key set while still changing arithmetic. A zero-valued mask can select another kernel, and chronological and physical-ring key orders can produce different floating-point reductions. Exact speculative decoding therefore requires matching shape, visible width, mask presence, physical order, and prefix construction.

Model-family labels are too broad for numerical corrections. Gemma 4 Unified 12B shares the 512-wide full heads but has different query/KV geometry and forward behavior. A diagnostic long draw found a separate 12B sliding-attention mismatch, tracked in issue #1986; this PR does not claim 12B long-context exactness.

## 6. Change Summary

| Item | Value |
|---|---:|
| Implementation commit files | 28 |
| Implementation lines added | 2,409 |
| Implementation lines deleted | 78 |
| Primary commit | `fb5dc631` |

| Category | Summary |
|---|---|
| Correctness | Restored exact 31B B=1 linear MTP attention and prefill behavior |
| Diagnostics | Added first-divergence localization and actionable decline reasons |
| Cache safety | Preserved and validated reference-ring metadata across cache lifecycle operations |
| Tests | Added attention, probe, cache, adapter, scheduler, and prefill regressions |
| Documentation | Added QAT/non-QAT measurements, evidence JSON, environment guidance, and model-scope clarification |

## 7. Follow-up Actions

- Investigate the Unified 12B long-context mismatch in issue #1986.
- Run the same backend-neutral checks on GB10/CUDA when the host becomes accessible.
- Re-measure corrected throughput with randomized arm order, cooldown, and multiple samples before making performance claims.

## Appendix

### A. Measurement Evidence

- QAT and non-QAT corrected teacher-forced traces: 0/256 top-1 disagreements and zero reference-choice logit delta.
- Six QAT/non-QAT free-running triples: baseline classic, corrected classic, and corrected MTP produced byte-identical 256-token output.
- Corrected long QAT run: 6,449 input tokens, 513 generated tokens, 280/698 accepted proposals, 2.1974 emitted tokens per verify, byte-identical to corrected classic.
- Qwen 3.8 MTP before/after: identical 256-token output and identical 164/274 acceptance over 92 rounds.
- The repository `make verify` gate passed, and the ignored buffered-cache donation regression passed 1/1 when run directly from the freshly built root test binary.
- The final Unified 12B startup passed its retained short gate after the automatic narrow retry with no inexact override; 14.09 seconds to health is a correctness observation, not a performance result.

### B. Limitations

CUDA was not measured. Throughput samples are diagnostic single runs under drifting thermal conditions. The original issue's exact long prompt was unavailable, so the report uses a reproducible 6,449-token reconstruction and does not present it as the original workload.

### C. References

- Issue #1983: Gemma 4 31B MTP exactness
- Issue #1279: 31B exactness policy and benchmark qualification
- Issue #1986: Unified 12B long-context exactness follow-up
- `docs/benchmark_results/gemma4-31b-mtp-exactness-2026-09-26.md`
- `docs/benchmark_results/gemma4-31b-mtp-exactness-2026-09-26.json`
