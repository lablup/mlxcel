# Technical Report: PR #2845 - perf: use absorbed MLA attention for multi-token steps below crossover

**Date**: 2026-11-03
**Status**: In review (open PR; closes #2408)
**Languages**: Rust, Markdown
**Risk Level**: Medium

---

## Executive Summary

MLA attention chose the matrix-absorbed path only when `l == 1`, so every speculative verify block, MTP verify step, and chunked-prefill continuation materialized per-head K/V across the whole cached latent even when it carried two to sixteen queries. This PR replaces that gate with a shared cost-model helper, `mlxcel_core::mla::use_absorbed_mla`, implementing the upstream bound `L < r*d / (r*d/T + 2*r - d)`, and wires one per-call boolean into seven MLA model sites. A cold prefill still materializes (`T == L` makes the bound false for every shipped geometry) and a single query is always absorbed, so existing prefill behavior is preserved; a short verify block behind a long cache now runs absorbed. Teacher-forced traces on two 4-bit checkpoints at widths 4 and 16 behind 1024 tokens report 0 decided-position disagreements against the reverted `l == 1` arm, and traced-forward timings drop between 15.8% and 53.0% for chunked-prefill continuations. MTP speculative decode did not show a clear win.

---

## 1. Problem Statement

### 1.1 Background

DeepSeek-family MLA compresses K and V into a single latent `c` of width `kv_lora_rank` plus a shared rope stream `k_pe`. The absorbed formulation folds the K up-projection into the query and the V up-projection after attention, so the model attends directly against the latent and keeps a much smaller cache. mlxcel already had both paths: an absorbed decode path and a materializing path that up-projects the live latent window through `kv_b_proj` and runs dense SDPA.

The path was selected by a single condition, `let output = if l == 1 { ... } else { ... }`. The absorbed path existed only for decode, and multi-token steps were assumed to be prefill. That assumption was already inaccurate: speculative decoding verifies several drafted tokens per step, MTP verifies a block, and chunked prefill can append a handful of tokens behind a long cached context. In all three cases the query count is small while the cached length `T` is large, which is precisely the regime where absorption is cheaper.

### 1.2 Existing Issues

- **Issue 1**: Every multi-token step materialized K and V across the whole latent window, costing `O(T)` work regardless of how few queries it carried.
- **Issue 2**: The `l == 1` gate conflated "decode" with "one query". A verify block of width 4 or 16 behind 1024 cached tokens took the same path as a cold prefill of 16 tokens.
- **Issue 3**: The absorbed path had no proof of use at multi-token widths. Issue #899 had already shipped a fused decode path that never activated, and the benchmark compared a fallback against itself, producing a clean-looking null result.
- **Issue 4**: Each MLA family reimplemented the path choice locally, so changing the policy meant editing seven files in lockstep.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| A multi-token absorbed step omits the causal mask and attends to future positions | High | Medium (mitigated: `deepseek_v2` builds the mask, `deepseek_v32` folds indexer top-k into it, other families fold causality before the branch) |
| Absorbed and materialized paths diverge numerically at multi-token widths | Medium | Medium (measured: 0 decided disagreements, but GLM shows top-1 flips at sub-2 gaps) |
| Cost model picks absorbed on a cold prefill and loses the numerically identical baseline | Medium | Low (`T == L` reduces the bound to `2*r < d`, false for shipped configs) |
| MTP speculative decode regresses | Medium | Medium (measured: inconclusive, means within noise) |

---

## 2. Technical Review

### 2.1 Security

Not applicable. The change does not touch input handling, authentication, serialization, or network boundaries.

- **Breaking Changes**: no. `use_absorbed_mla` is additive and the existing `MLXCEL_MLA_ABSORBED` gate still governs whether the absorbed machinery is built.
- **New Dependencies**: none.

### 2.2 Performance

The cost model is derived from per-head FLOP counts. With `r = kv_lora_rank`, `d = qk_nope_head_dim + v_head_dim`, `L` queries, and `T` cached latent positions after this step's update:

- Absorbed: `L*r*d + 2*L*T*r`
- Materialize: `T*r*d + L*T*d`

Absorbed wins while `L < r*d / (r*d/T + 2*r - d)`. As `T` grows the bound approaches `r*d / (2*r - d)`: 170 queries for DeepSeek-V3 geometry (512/128/128) and 398 for GLM-4.7-Flash's `glm4_moe_lite` geometry (512/192/256). On a cold cache (`T == L`) the bound reduces to `2*r < d`, which is false for every shipped configuration, so a fresh prefill keeps materializing. A single query returns true before the division, and a non-positive denominator declines rather than producing a meaningless bound.

**Performance Impact (chunked-prefill continuation, 4 and 16 tokens behind 1024 cached tokens, traced forward only):**

| Model | Width | Before | After | Change |
|------|--------|-------|-------------|-------------|
| DeepSeek-V2-Lite | 4 | 17.57 ms | 14.80 ms | -15.8% |
| DeepSeek-V2-Lite | 16 | 31.52 ms | 25.34 ms | -19.6% |
| GLM-4.7-Flash | 4 | 46.45 ms | 21.83 ms | -53.0% |
| GLM-4.7-Flash | 16 | 60.31 ms | 47.44 ms | -21.3% |

These are traced-forward wall times inside `logit_trace`, not end-to-end serving throughput.

### 2.3 Compatibility & Dependencies

All seven call sites compute the helper from their own `ModelArgs` geometry. `kimi_k3` gains a stored `v_head_dim` field because the cost model needs it and the existing projection shapes did not expose it as a struct member. No public API is removed.

### 2.4 Code Quality

- **Test Coverage**: six new tests. Five pin the helper's boundary behavior (single query, DeepSeek-V3 crossover at `T=100000`, `T == L`, verify-block widths, non-positive denominator); one runs full-model logits through both branches on one warm cache.
- **Code Complexity**: each model site now has one `absorbed` boolean and records its path, replacing an inline `l == 1` branch. `deepseek_v32` restructures its branch body to separate single-token sparse gather from multi-token absorbed attention.
- **Technical Debt**: decreased. The path policy lives in one helper with an explicit `Used by:` list of all seven call sites.

---

## 3. Technical Decisions

### 3.1 One shared cost-model helper instead of per-model thresholds

**Context:**
Seven model files carried the same `l == 1` gate, and the correct crossover depends on head geometry. Porting the upstream formula separately into each file would drift.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Keep `l == 1` and add model-specific exceptions | Minimal diff | Seven divergent policies; no principled boundary |
| Copy the upstream inequality into each model | No cross-crate dependency | Duplicate formula, easy to fix in one place only |
| Empirical per-model thresholds | Could fit measured hardware | Not portable across geometry or hardware |
| **Chosen: one helper in `mla/mod.rs`** | Single source of truth; testable in isolation; mirrors upstream | Every family depends on core MLA module |

**Rationale:**
The inequality is a closed-form FLOP comparison, not a hardware heuristic. Upstream already landed it in Blaizzy/mlx-vlm#2150 (with the mlx-lm counterpart ml-explore/mlx-lm#1817), so matching the upstream form keeps the port aligned. The helper's doc comment carries the formula, the asymptotic bounds for both shipped geometries, the cold-cache reduction, and the `Used by:` list required by the shared-function rule.

**Trade-offs:**
The helper needs `v_head_dim` at each site. Most families already had it; `kimi_k3` did not and now stores it. The helper is also geometry-only: it models FLOPs, not cache traffic, kernel launch overhead, or quantization differences, so the crossover is approximate.

### 3.2 `T` is the post-update cached length, computed once per call

**Context:**
The bound compares a step's query count against the cache the attention will actually read, which includes the rows this step appends. Reading the length at the wrong point shifts the crossover.

**Decision:**
Each site computes `absorbed` once, with `t = cache.seq_len()` after `cache.update_and_fetch`, or with `live_before + l` in `deepseek_v2`, whose `MlaLatentCache::attend` performs the update internally. No code below the boolean gates on `l` for path selection. The helper's doc states explicitly that `t` is the live window length, not the monotonic RoPE offset.

**Trade-offs:**
`deepseek_v2` is the one site whose shape differs from the rest, because it measures the cache before the attend call that appends. That asymmetry is documented in the helper and the call site.

### 3.3 Multi-token absorbed steps need an explicit causal mask

**Context:**
The materializing path obtains causality from `causal_attention`. The absorbed path passes `pe_scores` directly, and a multi-token absorbed step would otherwise see the full latent with no causal restriction.

**Decision:**
- Families that build their own `pe_scores` mask (`deepseek_v3`, `glm4_moe_lite`, `kimi_linear`, `kimi_k3`, `longcat_flash_ngram`) already fold causality before the branch.
- `deepseek_v2::forward_absorbed` constructs `create_causal_mask(l, live_before)` when `l > 1` and the caller supplied no mask; a caller-supplied padded mask takes precedence.
- `deepseek_v32` moves `apply_sparse_prefill_mask` ahead of the branch for `l > 1`, so the indexer's top-k selection becomes part of the additive mask for multi-token absorbed steps, while single-token decode keeps the cheaper `gather_topk_kv` fast path.

**Trade-offs:**
The `deepseek_v32` restructuring is the largest part of the diff (+88/-42). It duplicates the latent-attention call across the single-token and multi-token absorbed branches instead of sharing one call with different masks.

### 3.4 Make the chosen path observable

**Context:**
Issue #899's lesson was that a path can silently never activate, and the benchmark then measures the fallback in both arms.

**Decision:**
Every site records `mla::stats::MlaDecodePath` (`AbsorbedComposed` at `l == 1`, `AbsorbedPrefill` for multi-token absorbed, `Decompressed` when the model declines). `examples/logit_trace` prints those counters under `MLXCEL_TRACE_MLA_STATS=1` and reports each chunk's traced-forward wall time separately from the context pass. The trace evidence in this PR uses those counters to prove the traced widths actually took `absorbed_prefill` (216 calls for DeepSeek-V2-Lite, 376 for GLM-4.7-Flash), while the reverted arm recorded zero.

**Trade-offs:**
The counters are process-global atomics; a run must call `take()` (clearing them) to attribute counts to one arm, which is what the trace harness does. The added `eval(&logits)` in `logit_trace` forces the lazy graph early so the timing excludes host-side top-k, which changes when the graph materializes but not the computed result.

---

## 4. Implementation Details

### 4.1 Architecture Changes

```
[Before]
model forward ──> l == 1 ? absorbed : materialize   (decided independently in 7 files)

[After]
model forward ──> use_absorbed_mla(l, t, r, nope, v)
                      │
                      ├── true  ──> absorbed (single-query or multi-token)
                      └── false ──> materialize per-head K/V, dense SDPA
```

### 4.2 Key Code Changes

**File: `src/lib/mlxcel-core/src/mla/mod.rs`**

```rust
pub fn use_absorbed_mla(
    l: usize,
    t: usize,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    v_head_dim: usize,
) -> bool {
    if l == 1 {
        return true;
    }
    let r = kv_lora_rank as f64;
    let d = qk_nope_head_dim as f64 + v_head_dim as f64;
    let denom = r * d / t as f64 + 2.0 * r - d;
    if denom <= 0.0 {
        return false;
    }
    (l as f64) < r * d / denom
}
```

**Reason for change:** This is the shared policy. Evaluating in `f64` avoids integer truncation at the crossover; returning true for `l == 1` preserves the decode path without relying on the inequality; a non-positive denominator declines instead of returning a negative bound.

**File: `src/models/deepseek_v2.rs`**

```rust
let live_before = view.seq_len();
let absorbed = mla::use_absorbed_mla(
    l as usize,
    live_before as usize + l as usize,
    self.kv_lora_rank,
    self.qk_nope_head_dim,
    self.v_head_dim,
);

let output = if absorbed {
    let mask = if l == 1 {
        None
    } else if let Some(m) = mask {
        Some(m)
    } else {
        causal_storage = mlxcel_core::utils::create_causal_mask(l, live_before);
        Some(causal_storage.as_ref().unwrap())
    };
    view.attend(q_nope, &q_pe, ckv, k_pe, proj, self.scale, mask)
} else { /* materialize per-head K/V */ }
```

**Reason for change:** `attend` appends the step's rows itself, so `T` must be computed as `live_before + l`. The causal mask is built only for the multi-token absorbed case that has no caller mask.

**File: `src/models/deepseek_v32.rs`**

```rust
// Multi-token steps fold the indexer's top-k selection into the additive mask
let pe_scores = if l > 1 {
    match &topk_indices {
        Some(indices) => apply_sparse_prefill_mask(
            &pe_scores, indices, mlxcel_core::array_shape(&kv_latent)[2],
        ),
        None => pe_scores,
    }
} else {
    pe_scores
};
```

**Reason for change:** The multi-token absorbed step runs latent attention over the full latent, so the top-k restriction has to be in the mask rather than in a gathered KV subset. Single-token decode keeps `gather_topk_kv`.

**File: `src/models/glm4_moe_lite.rs` and `src/models/glm4_moe_lite_tests.rs`**

`forward_with_gate(x, cache, mask, forced_absorbed: Option<bool>)` is a test-facing entry that production `forward` calls with `None`. It lets the parity test replay one warm cache through both branches at a width the production model itself selects absorbed.

**Reason for change:** Without pinnability, the materializing side could not be exercised at the width under test.

**File: `examples/logit_trace.rs`**

```rust
let forward_started = std::time::Instant::now();
let logits = ...;
eval(&logits);
let forward_ms = forward_started.elapsed().as_secs_f64() * 1e3;
...
if std::env::var("MLXCEL_TRACE_MLA_STATS").as_deref() == Ok("1") {
    eprintln!("# mla_paths\t{}", mlxcel_core::mla::stats::take().summary());
}
```

**Reason for change:** Turns the trace harness into a measurement that can both time the multi-token continuation and prove which branch the traced width took.

### 4.3 Data Model Changes

None. Both paths write the same latent cache, so a step can switch between them without repacking.

---

## 5. Learning Points

### 5.1 Closed-form path selection beats a structural heuristic

**Concept:**
When two algorithms implement the same math with different work distributions, the cheaper one can be selected by comparing their FLOP counts symbolically rather than by a proxy such as "query count equals one".

**Application in this PR:**
Absorption folds an up-projection into the query; materializing pays an up-projection over the cached window. Comparing `L*r*d + 2*L*T*r` against `T*r*d + L*T*d` yields one inequality in five shape values. The same inequality covers decode, verify blocks, and prefill; the endpoints fall out of it (`T == L` always materializes, `T` large admits more queries).

**Common Use Cases:**
- Choosing between transposed and untransposed contractions based on matrix shape.
- Selecting a chunked or monolithic kernel from sequence length and batch.
- Deciding between fused and decomposed operators from the query-to-cache ratio.

### 5.2 Negative evidence needs a control

**Concept:**
A parity comparison that reports "zero disagreements" is only meaningful if it can detect disagreement when one exists.

**Application in this PR:**
The trace comparison was run against a control binary with its MLA rope score scaled 2x, which scored 6 of 41 decided disagreements (14.6%). The new gate scored 0. That establishes the comparator's sensitivity rather than assuming it.

**Common Use Cases:**
- Logit-trace diffing between two kernels.
- Numerical equivalence tests for quantized paths.
- A/B benchmark harnesses where both arms could accidentally run the same code.

---

## 6. Further Learning

### Key Terms

| Keyword | Description | Relevance |
|---------|-------------|-----------|
| Matrix-absorbed MLA | Folds `kv_b_proj` into the query and output contractions so attention runs on the latent | The path this PR extends to multi-token steps |
| `kv_lora_rank` (`r`) | Latent width shared by K and V | Central term in the cost model |
| Crossover bound | `L < r*d / (r*d/T + 2*r - d)` | The per-call decision rule |
| `absorbed_prefill` | Multi-token absorbed path under the cost model | Counters prove the traced width took it |
| MTP | Multi-token prediction; target verifies a drafted block | One of the shapes the change targets, with inconclusive decode results |
| Teacher-forced trace | Runs fixed token sequences and compares per-position logits | The parity evidence method |

### Related PRs/Issues

- Issue #2408: `perf: Use absorbed MLA attention for multi-token steps below crossover` — the issue this PR closes.
- PR #907: introduced matrix-absorbed MLA decode over a compressed-latent cache, including the `stats` counters.
- Issue #899: a fused decode path that never activated; the motivation for path counters.
- Issue #898: `paged_attention_merge_states`, used by Stage 2 split-KV.
- Blaizzy/mlx-vlm#2150 and ml-explore/mlx-lm#1817: upstream cost-model threshold this port mirrors.

---

## 7. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 13 |
| Lines added | +512 |
| Lines deleted | -93 |
| Tests added | 6 |

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| Performance | 8 | Shared cost model and the seven MLA call sites |
| Observability | 2 | `mla::stats` path recording and `logit_trace` counters/timing |
| Tests | 2 | Helper boundary tests and the warm-cache parity test |
| Documentation | 2 | `docs/mla-absorbed-decode.md` and `docs/benchmarks.md` |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `b2131d94` | perf | use absorbed MLA attention for multi-token steps below crossover |

---

## 8. Follow-up Actions

### Required

- [ ] Extend per-model parity tests beyond `glm4_moe_lite` to the other six MLA families, especially `deepseek_v32` whose branch restructuring is the largest change.
- [ ] Re-measure MTP speculative decode with more runs or a tighter harness; the reported sample was inconclusive.

### Monitoring Required

- With `MLXCEL_MLA_ABSORBED=1`, watch `mla::stats` in serving: a rise in `decompressed` for verify-shaped steps would mean the geometry-driven bound disagrees with the deployed shapes.
- Watch for numerical drift reports on DeepSeek-V3.2 and GLM-4.7-Flash under speculative decoding, where top-1 flips at sub-2 logit gaps were observed.

### Future Improvements

- Model cache traffic and kernel overhead in addition to FLOPs, since the current bound ignores both.
- Consolidate the duplicated latent-attention call in `deepseek_v32` once the mask variants are stable.
- Add an end-to-end serving benchmark for verify-heavy workloads.

---

## Appendix

### A. Test Results

| Test | Result |
|------|--------|
| `cargo test --profile test-fast -p mlxcel-core mla::` | Passes: absorbed at `L=169` and materialized at `L=171` with `T=100000`, never absorbed when `T == L`, always absorbed at `L == 1` |
| `models::glm4_moe_lite::tests::four_token_step_on_a_warm_cache_matches_across_both_attention_paths` | Passes: full-model logits from a 4-token step on a 64-token warm cache agree across both branches under the `2e-2` threshold |
| `cargo fmt --check` | Passes |
| `cargo clippy --lib --tests --example logit_trace --features metal,accelerate -- -D warnings` | Passes |

**Teacher-forced logit traces** (widths 4 and 16 behind 1024 tokens, new gate vs reverted `l == 1` gate, `--decided 2`):

| Model | Width | Positions | Decided | Decided disagreement | Top-1 disagreement |
|---|---|---|---|---|---|
| DeepSeek-V2-Lite-Chat-4bit-mlx | 4 | 32 | 21 | 0 | 0 |
| DeepSeek-V2-Lite-Chat-4bit-mlx | 16 | 128 | 66 | 0 | 0 |
| GLM-4.7-Flash-4bit | 4 | 32 | 14 | 0 | 1 (gap 0.5) |
| GLM-4.7-Flash-4bit | 16 | 128 | 41 | 0 | 10 (gaps below 2) |

**Branch counters:** the traced widths took `absorbed_prefill` (216 calls for DeepSeek-V2-Lite, 376 for GLM-4.7-Flash); the reverted arm recorded zero.

**Control:** the same GLM trace against a throwaway binary with its MLA rope score scaled 2x scores 6/41 decided disagreements (14.6%), so the comparator detects a genuinely different arm.

### B. Performance Benchmarks

**Chunked-prefill continuation** (4 and 16 tokens behind 1024 cached tokens, timed inside `logit_trace`, three runs per arm, first chunk dropped, 21 samples per cell):

| Model | Width | Before | After | Change |
|---|---|---|---|---|
| DeepSeek-V2-Lite | 4 | 17.57 ms | 14.80 ms | -15.8% |
| DeepSeek-V2-Lite | 16 | 31.52 ms | 25.34 ms | -19.6% |
| GLM-4.7-Flash | 4 | 46.45 ms | 21.83 ms | -53.0% |
| GLM-4.7-Flash | 16 | 60.31 ms | 47.44 ms | -21.3% |

**GLM-4.7-Flash MTP speculative decode** (4-bit target plus a drafter split from `zai-org/GLM-4.7-Flash` with `mlxcel split-mtp --q-bits 4`, block 2, 1621-token prompt, `-n 64 --profile`, three runs per arm):

| Arm | Run 1 | Run 2 | Run 3 | Median |
|---|---|---|---|---|
| Before | 93.14 tok/s | 93.81 tok/s | 92.92 tok/s | 93.14 tok/s |
| After | 80.99 tok/s | 98.42 tok/s | 98.55 tok/s | 98.42 tok/s |

The first after-run is an outlier. Medians differ by +5.7%, but the means are within noise, so block-2 MTP decode is not a clear win: verify windows are short and MoE weight traffic dominates.

### C. What Was Not Validated

The PR body does not list a full workspace run. It reports `cargo test --profile test-fast -p mlxcel-core mla::`, the single `glm4_moe_lite` parity test, `cargo fmt --check`, and a scoped clippy invocation over `--lib --tests --example logit_trace`. The issue's verification block also named `cargo test --workspace --profile test-fast --features metal,accelerate` and `cargo clippy --workspace --all-targets`, which the PR body does not report.

Only `glm4_moe_lite` has a per-model parity test. The other six sites are covered by the helper unit tests and by teacher-forced traces on two checkpoints at two widths, not by a per-family both-branch logits comparison. `deepseek_v32`'s mask restructuring in particular has no direct branch-parity test.

The MTP speculative-decode benchmark is explicitly inconclusive: the first after-run was an outlier and the means are within noise. No end-to-end serving throughput or memory measurement was reported; the timing numbers are traced-forward wall times inside `logit_trace`.

### D. References

- `docs/mla-absorbed-decode.md` — absorbed MLA design and the revised multi-token section.
- `docs/benchmarks.md` — `MLXCEL_TRACE_MLA_STATS` and chunked-prefill trace usage.
- Issue #2408 and upstream Blaizzy/mlx-vlm#2150, ml-explore/mlx-lm#1817.
