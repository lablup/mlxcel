# Technical Report: PR #2865 - perf(speculative): commit per-row accepts in batched DFlash

**Date**: 2026-11-06
**Status**: In review (open PR; closes #2442)
**Languages**: Rust, Markdown
**Risk Level**: Medium

---

## Executive Summary

PR #2865 implements item 1 of issue #2442: the batched Qwen 3.5 DFlash round loop now commits each row's own speculative-walk accept count instead of clamping the whole round to the shortest live row. It reuses the Gemma 4 `DivergentVerifyRows` attention layout, compacts each row's attention KV back to a contiguous valid prefix, masks the stale gap, rotates queries and keys at each row's logical positions, and replays each GDN layer over that row's accepted prefix. On an M3 Ultra with `qwen3.5-4b-4bit` and `qwen3.5-4b-dflash` at B = 4, aggregate decode rose from 83.06 to 116.05 tok/s (1.397x), committed accepts per row-round from 0.323 to 1.036, and rounds from 96 to 67, while the calibrated parity gate passed 2 of 2 with per-row accepts as the default. The drafter still receives a shared rectangular prefix, so the speculative walk itself falls from 2.938 to 1.036; per-row drafter context is explicitly out of scope, and no server-level concurrent-request smoke run was performed.

---

## 1. Problem Statement

### 1.1 Background

The batched DFlash round loop for Qwen 3.5 shares one rectangular target cache across every row. Issue #2353 handled divergent walks by clamping a round to the shortest live row's accept (`clamp_uniform_acceptance`), so every row committed the same number of verify positions. That was correct for a rectangular cache because a rejected tail cannot simply be zeroed: a zero key still scores `q . 0 = 0` and takes `exp(0)` weight in the softmax, which dilutes attention on every later step.

The cost was visible in the accept statistics. On the #2442 measurement the walk averaged 2.938 accepted tokens per row-round while the clamp committed only 0.323, and 93 of 96 rounds were clamped. Batching therefore bought about 1.06x over the B = 1 round loop (83.06 against 78.09 tok/s), far below the walk's potential.

Issue #2442 item 1 asked for per-row accepts. PR #2853 then established that the batched verify forward is not byte-identical to a B = 1 forward under Metal kernel dispatch, and replaced the hard byte gate with a calibrated 1.0-logit near-tie margin. That settled the prerequisite: the accept policy was not the cause of the parity divergence, so per-row accepts could proceed under the existing margin gate.

### 1.2 Existing Issues

- **The clamp discards accepted work.** Every round committed the minimum accept across live rows, so any row that walked further had its extra accepted tokens rolled back.
- **Zero-filling a rejected tail is not a valid alternative.** Stale keys still participate in the softmax with `exp(0)` weight.
- **A rectangular cache has one offset.** Rows that committed different counts need a way to keep different contiguous prefixes while sharing the same physical cache allocation.
- **Linear-attention state is per-row but its valid length is not.** A single batched GDN replay over `max(accepted)` would advance a short row through positions it rejected.
- **Batching barely paid.** At 0.323 committed accepts per row-round, B = 4 delivered only about 1.06x over B = 1.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|---|---|---|
| A ragged cache layout corrupts later attention or GDN state | High (wrong tokens) | Low: the parity gate runs with per-row accepts as the default, and the compaction is covered by unit tests |
| An unsupported cache mode reaches the per-row path | Medium (panic or corruption) | Low: `row_layout_supported` gates on dense Fp16, non-paged, non-head-trimmed caches and the compaction asserts the same preconditions |
| Per-row accepts interact badly with real server concurrency | Medium (serving correctness) | Medium: no server-level concurrent-request smoke run was performed |
| The throughput comparison is shifted by host load | Low (decision quality) | Low: the arms alternated under the same competing build load and the clamp and B = 1 cells reproduce the earlier quiet-box record |
| The lower walk is mistaken for a regression | Low (misattribution) | Certain if unrecorded: the drafter context is deliberately still rectangular, and per-row drafter context is a separate change |

---

## 2. Technical Review

### 2.1 Security

No network, authentication, input-parsing, or serialization surface changes. The only new configuration input is the `MLXCEL_DFLASH_PER_ROW_ACCEPTS` environment variable, read when the generator is constructed and accepting only an on/off value. No user data is logged; the accept statistics logged by `dflash_target.rs` already existed and only their description was updated. The rollback path validates cache shape, mode, paging, and head trim with assertions rather than proceeding on an unexpected layout.

### 2.2 Performance

Measured on an M3 Ultra, `qwen3.5-4b-4bit` target with `qwen3.5-4b-dflash`, block 16, 128 tokens per row, temperature 0, three alternating runs per arm.

| Metric | Clamp | Per-row | Change |
|---|---:|---:|---:|
| Aggregate throughput (tok/s) | 83.06 | 116.05 | 1.397x |
| B = 1 reference (tok/s) | 78.09 | 78.09 | 1.00x |
| Rounds | 96 | 67 | 30% fewer |
| Clamped rounds | 93 | 0 | eliminated |
| Committed accepts per row-round | 0.323 | 1.036 | 3.2x |
| Walk per row-round | 2.938 | 1.036 | 0.35x |

The per-row arm is 1.49x the B = 1 round loop; the clamp arm is 1.06x. Throughput is the round-loop wall clock from `DFlashBatchedRunOutput::stats` with prefill excluded, so the per-row compaction work (`copy` plus `slice_update` per row per attention layer) is inside the measured cost. The three clamp runs spanned 0.2% and the three per-row runs 0.36%.

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: none. The new `SpeculativeTarget` methods have default implementations that preserve the old behavior.
- **New Dependencies**: none.
- **Compatibility**: per-row accepts are the default only when the target reports support and the caches are dense Fp16 without paging or head trim. Quantized, paged, or head-trimmed caches select the uniform clamp automatically. `MLXCEL_DFLASH_PER_ROW_ACCEPTS=0` (also `false`, `no`, `off`) restores the clamp for A/B attribution.
- **Internal signature change**: `Qwen3NextAttention::forward_with_position_ids_verify` and its hidden-state sibling gained an `Option<&[i32]>` row-layout argument. Both are `pub(crate)` and all call sites were updated.
- **VLM**: `Qwen35VLModel` delegates the capability, offset, verify, and rollback hooks to its text model. The attention layer refuses per-row verify when MRoPE position ids are supplied; the DFlash verify path passes none, so this is a guard rather than a served path.

### 2.4 Code Quality

Three unit tests were added: two for attention compaction in `models::qwen3_5_tests` (row block moved to its own valid end; no-op when every row ends at the offset) and one for `build_divergent_row_mask` in `mlxcel-core::utils` (history attends, stale gap blocked, new block causal at each row's logical positions). The `models::qwen3_5_tests` suite now has 30 tests; the DFlash round-loop modules keep 15 (`round_loop.rs`) and 13 (`round_loop_batched.rs`) tests. The mask builder was moved out of `gemma4.rs` into `mlxcel-core::utils`, removing about 30 duplicated lines from Gemma 4 and giving both models one implementation. `cargo fmt --all -- --check` and `cargo clippy --lib --tests --features metal,accelerate -- -D warnings` are clean.

---

## 3. Technical Decisions

### 3.1 Gate per-row accepts on a target capability and an instance cache check

**Context:**
Per-row accepts require a target whose attention cache can be compacted row by row and whose linear-attention state can be replayed per row. Not every `SpeculativeTarget` can do that, and even Qwen 3.5 cannot for every cache mode.

**Alternatives Considered:**

| Option | Pros | Cons |
|---|---|---|
| Always take the per-row path for any target | Simple loop | Breaks targets that only support a uniform rollback; panics on quantized or paged caches |
| Require every target to implement the new hooks | Uniform behavior | Forces unrelated targets and test doubles to add no-op implementations |
| **Chosen: default trait methods plus a two-level gate** | Existing targets are unchanged; unsupported caches fall back safely | The served path depends on runtime cache state |

**Rationale:**
`supports_per_row_accepts()` is the family-level answer and defaults to `false`. `row_layout_supported(caches)` is the instance-level answer and checks `KVCacheMode::Fp16`, `!is_paged_backed()`, and `offset - live_len() == 0` (no head trim). The loop takes the per-row path only when both are true and the caller has not disabled it. The default `rollback_partial_batched_rows` and `verify_forward_with_row_layout` delegate to the uniform hooks, which is correct only when every row shares the offset, so a target that forgets to override them fails safe rather than silently producing a ragged layout.

**Trade-offs:**
The production behavior now depends on the cache mode chosen at runtime, so a quantized deployment silently uses the old clamp and does not get the 1.397x. The alternative was to panic or refuse the burst, which would turn a performance feature into a availability regression.

### 3.2 Reuse the Gemma 4 divergent-row layout instead of inventing a second one

**Context:**
The Gemma 4 MTP verifier already handles rows with different logical valid ends against a shared physical cache. Its `build_divergent_verify_mask` encodes exactly the three regions a ragged DFlash round needs: the row's valid history, its stale gap, and the new block.

**Alternatives Considered:**

| Option | Pros | Cons |
|---|---|---|
| Pad every row to `max(accepted)` and zero rejected positions | No compaction | Zero keys still take softmax weight; already rejected in #2353 |
| Copy the cache per row and run independent forwards | Conceptually simple | Destroys the batching win and duplicates cache memory per row |
| **Chosen: shared divergent-row mask plus per-row rotation** | One mask describes all three regions; reuses a proven layout | Mask construction is `O(B * l * (offset + l))` |
| Build a DFlash-specific mask | No cross-model coupling | A second implementation to keep correct |

**Rationale:**
`build_divergent_row_mask(l, offset, per_row_valid_end, left_padding, window)` returns `[B, 1, l, offset + l]` additive mask. Row `r` admits keys below its valid end, blocks `[valid_end, offset)`, and treats new key `t` at logical position `valid_end + t`, so query `j` sees new keys `t <= j`. `Qwen3NextAttention` detects `divergent_rows` when `target_verify`, `l > 1`, and any end differs from the shared offset, applies RoPE per row at the row's own start, and runs `attend_per_position_rows`, which slices query `j` against key width `offset + j + 1` with the matching mask slice. The Gemma 4 function now delegates to the same core utility, so the two models cannot drift.

**Trade-offs:**
The mask is built per verify forward and explicitly materialized. That cost is included in the measured throughput and is outweighed by the accept gain at the measured shape. A future fused path could avoid the materialization but is not needed for the current win.

### 3.3 Keep the drafter context rectangular at the shortest live committed prefix

**Context:**
The target can now keep ragged caches, but the drafter's input hidden tensor and its own context are still built from one rectangular slice. Making the drafter per-row would require per-row drafter cache views and per-row drafter forwards, which is a much larger change.

**Alternatives Considered:**

| Option | Pros | Cons |
|---|---|---|
| Give the drafter per-row context now | Higher walk | Requires per-row drafter cache and forward plumbing; not measured here |
| Feed the drafter the longest committed prefix | More draft material for lagging rows | The hidden slice from the verify block is only valid up to the shortest accepted block position, so this would mix rejected positions into context |
| **Chosen: shortest live committed prefix** | Correct given the captured hidden slice; reuses the existing rectangular contract | The walk falls from 2.938 to 1.036 |

**Rationale:**
`drafter_ctx` is computed as the minimum `accepted[r] + 1` over unfinished rows, and the hidden tensor is sliced to that length. Rows that committed more than the minimum still advance their target caches by their own accept, so the emitted stream stays exact; the drafter simply re-drafts from a lagging context on the next round. This keeps the change bounded to the target side and makes the throughput comparison attributable to the accept policy alone.

**Trade-offs:**
The walk drops by 65%, so the round count benefit is smaller than the accept count benefit suggests. Per-row accepts are still a clear win because the committed count, not the walk, determines emitted tokens per round. Per-row drafter context is recorded as a follow-up, not as a defect in this PR.

### 3.4 Compact the attention KV and replay GDN state per row

**Context:**
After a ragged round the shared cache holds each row's accepted block at the shared physical offset, followed by rejected positions. The next round needs every row to look like a contiguous `[0, valid_end)` prefix while the physical offset stays shared.

**Rationale:**
`rollback_speculative_cache_rows` computes `o_post = max(valid_end_pre[r] + accepted[r] + 1)`. `compact_attention_rows` moves each row's `accepted[r] + 1` block from `o_pre` to that row's logical end, zeroes the vacated gap up to `o_post`, and the caller trims the cache offset to `o_post`. The source is copied before `slice_update` so an overlapping move cannot read already-overwritten rows. GDN state is replayed per row: `replay_gdn_rows` slices each row's captured `q/k/v/a/b` and initial state to `accepted[r] + 1`, runs the chain-parity kernel at that row's own length, and stacks the per-row conv and recurrent states back into the `[B, ...]` cache tensors. This keeps the per-row numerics on the same kernel the uniform path uses.

**Trade-offs:**
Compaction adds one copy and one `slice_update` per row per attention layer per round. It is measured inside the reported throughput. The uniform path is also kept intact for fallback and A/B use, so the code carries both arms; that duplication is deliberate for attribution.

---

## 4. Implementation Details

### 4.1 Architecture Changes

```
[Before, uniform clamp]
verify [B, bs] -> walk per row -> clamp_uniform_acceptance (min over live rows)
                -> rollback_partial_batched (trim bs - (a + 1) in every row)
                -> drafter hidden [B, max_a + 1]

[After, per-row accepts]
row_valid_ends = [prefill offset; B]
verify [B, bs]
  -> per-row RoPE at row_valid_ends[r]
  -> attend with build_divergent_row_mask (history, stale gap, causal block)
  -> commit accepted[r] + 1 per row
  -> compact_attention_rows: block moves to row_valid_ends[r], gap zeroed
  -> cache offset trimmed to max(new valid ends)
  -> replay_gdn_rows: per-row conv + recurrent state over accepted prefix
  -> row_valid_ends[r] += accepted[r] + 1
  -> drafter hidden [B, min_live(accepted) + 1] (rectangular, deliberately)
```

### 4.2 Key Code Changes

**File: `src/lib/mlxcel-core/src/drafter/dflash/round_loop_batched.rs`**

The generator resolves the policy once per run and seeds the logical ends from the prefill offset:

```rust
let per_row = self.per_row_accepts
    && target.supports_per_row_accepts()
    && target.row_layout_supported(caches);
let mut row_valid_ends: Vec<i32> = if per_row {
    let offset = T::batch_cache_offset(caches);
    vec![offset; batch_size]
} else {
    Vec::new()
};
```

The verify call branches on the policy, the clamp is skipped on the per-row arm, and the rollback hands each row's own accepted prefix to the target:

```rust
if per_row {
    target.rollback_partial_batched_rows(
        caches, &verify_out, &accepted_i32, &row_valid_ends, bs as i32,
    );
    for r in 0..batch_size {
        row_valid_ends[r] += accepted_i32[r] + 1;
    }
} else {
    // uniform clamp arm: trim bs - (accepted + 1) in every row
}
```

The drafter context stays rectangular at the shortest live row:

```rust
let drafter_ctx = if per_row {
    (0..batch_size)
        .filter(|&r| !finished[r])
        .map(|r| accepted_per_row[r] + 1)
        .min()
        .unwrap_or(1)
} else {
    max_accepted + 1
};
```

**Reason for change:** the round loop owns the policy switch and the logical position bookkeeping; the target owns cache layout. Keeping the switch in one place makes the A/B measurable in a single process.

**File: `src/lib/mlxcel-core/src/drafter/dflash/round_loop.rs`**

`SpeculativeTarget` gains four defaulted hooks: `supports_per_row_accepts`, `row_layout_supported`, `batch_cache_offset`, and the static-free `verify_forward_with_row_layout`, plus `rollback_partial_batched_rows`. The existing `rollback_partial_batched` contract is narrowed to the uniform-clamp arm in its documentation.

**Reason for change:** targets that cannot represent ragged rows keep compiling and keep their old semantics.

**File: `src/models/qwen3_5.rs`**

`Qwen35Model` implements the capability gate, the shared offset, the ragged verify forward, and the ragged rollback. `forward_speculative_inner` threads `Option<&[i32]>` to each decoder layer. The non-uniform rollback is:

```rust
if !uniform {
    let o_post = ve_pre.iter().zip(accepted)
        .map(|(&ve, &a)| ve + a + 1).max().expect("non-empty accepted");
    for cache in caches.iter_mut() {
        if let Qwen3NextCache::Attention(kv) = cache {
            compact_attention_rows(kv, &ve_pre, accepted, o_pre, o_post);
            let roll_back = o_pre + block_size - o_post;
            if roll_back > 0 { kv.trim(roll_back); }
        }
    }
    self.replay_gdn_rows(caches, gdn_states, accepted, o_post);
    return accepted.iter().copied().max().unwrap_or(0);
}
```

**Reason for change:** the attention cache must become a valid contiguous prefix per row before the next round, and the GDN state must reflect only that row's accepted tokens.

**File: `src/models/qwen3_next.rs`**

`Qwen3NextAttention` gains `apply_rope_per_row` and `attend_per_position_rows`. The divergent path is selected when any row's valid end differs from the cache offset:

```rust
let divergent_rows: Option<&[i32]> = row_valid_ends.filter(|ends| {
    target_verify
        && l > 1
        && ends.len() == b as usize
        && ends.iter().any(|&end| end != offset)
});
```

Each query position slices the exact key width a B = 1 chain would see, with the divergent-row mask encoding history, gap, and block causality.

**Reason for change:** a shared rectangular attention call cannot give each row a different logical start; the per-position path can, and it matches the Gemma 4 approach.

**File: `src/lib/mlxcel-core/src/utils.rs` and `src/models/gemma4.rs`**

`build_divergent_row_mask` moves into `mlxcel-core::utils`; Gemma 4's `build_divergent_verify_mask` becomes a one-line delegation.

**Reason for change:** one tested mask implementation for both the Gemma 4 MTP verifier and the Qwen 3.5 DFlash verifier.

**File: `src/vision/qwen3_5_vl.rs`**

`Qwen35VLModel` forwards the four new hooks to its text model.

**Reason for change:** the VLM wrapper should not silently lose the capability or the fallback.

**Files: `docs/CONTINUOUS_BATCHING.md`, `docs/environment-variables.md`, `docs/benchmark_results/dflash-per-row-accepts-m3ultra-2026-11-06.md`**

The batching guide and the environment variable table record the default, the fallback, and the measured numbers; the benchmark document records the method, the per-run values, the caveat, and the correctness gate.

### 4.3 Data Model Changes

None. The cache layout stays `[B, heads, capacity, dim]`; only the valid-prefix bookkeeping and the physical offset change per round.

---

## 5. Learning Points

### 5.1 Ragged acceptance inside a shared rectangular cache

**Concept:**
A shared batched KV cache has one physical offset and one allocation. Ragged accept counts do not require per-row allocations if each row's live content is compacted to a contiguous prefix after every round. The invariant becomes "each row's cache content is a contiguous `[0, valid_end)` prefix, and the stale region up to the shared offset is masked out".

**Application in this PR:**
`compact_attention_rows` moves the accepted block from the shared offset to the row's logical end and zeroes the vacated gap. `build_divergent_row_mask` then encodes the three regions for the next verify forward. The physical offset is trimmed to the maximum new valid end, so the allocation does not grow with the raggedness.

**Common Use Cases:**
- Batched speculative decoding with per-sequence accept counts.
- Continuous batching where sequences stop, resume, or roll back at different lengths.
- Any shared-buffer scheme that needs per-row logical lengths without per-row buffers.

### 5.2 One mask layout, two models

**Concept:**
A mask that expresses "valid history, blocked gap, causal new block at logical positions" is model-agnostic. Lifting Gemma 4's implementation into a core utility removed a second copy before it could drift.

**Application in this PR:**
`build_divergent_row_mask` is now called by Gemma 4's `build_divergent_verify_mask` and by Qwen 3.5's `attend_per_position_rows`. The move also brought the mask under a dedicated unit test.

**Common Use Cases:**
- Reusing attention masks across model families with the same cache topology.
- Consolidating numerically sensitive device code behind one tested constructor.

### 5.3 Judge speculative batching by committed accepts, not by walk

**Concept:**
The scan ratio (walk) measures what the drafter proposed; the committed accept count measures what the target actually kept. A batching change can lower the walk while raising throughput if it stops discarding accepted tokens.

**Application in this PR:**
The per-row arm's walk fell from 2.938 to 1.036 because the drafter context is still the shortest live prefix, yet aggregate throughput rose 1.397x because committed accepts rose from 0.323 to 1.036 and rounds fell from 96 to 67. Reporting only the walk would have read as a regression.

**Common Use Cases:**
- Speculative decoding A/B tests.
- Any pipeline where a producer's proposal rate and a consumer's acceptance rate can move in opposite directions.

---

## 6. Further Learning

### Key Terms

| Keyword | Description | Relevance |
|---|---|---|
| DFlash | Block-draft speculative decoding with a diffusion-style drafter | The batched loop changed here |
| Uniform clamp | Committing the shortest live row's accept to every row | The previous policy, still reachable |
| Ragged accept | Different accept counts per row in one round | What per-row accepts commit |
| Valid end | Length of a row's contiguous valid cache prefix | The logical position used for RoPE and attention |
| Stale gap | `[valid_end, physical_offset)` left by a shorter accept | Masked out of attention and zeroed by compaction |
| GDN | Gated delta net linear-attention layer | Its conv and recurrent state are replayed per row |
| Walk | Tokens the drafter proposed and the target verified | Falls here because drafter context stayed rectangular |

### Related Technologies/Frameworks

- **MLX**: array framework and Metal kernels; the `copy` plus `slice_update` compaction and the per-position attention slices run on it.
- **Gemma 4 divergent MTP verify**: the source of the `DivergentVerifyRows` mask layout that this PR reuses.

### Related PRs/Issues

- Issue #2442: the parent issue; this PR implements item 1 and closes it.
- Issue #2353: introduced the uniform clamp and the original B = 4 versus B = 1 measurement.
- PR #2853 / issue #2851: calibrated the batched verify parity margin under which this PR's gate runs.
- Issue #2190: the Gemma 4 per-row verify architecture that supplied the mask layout.
- Issue #2440: the earlier per-row scoping work referenced by the benchmark document.

---

## 7. Change Summary

### Statistics

| Item | Value |
|---|---|
| Files changed | 15 |
| Lines added | +1279 |
| Lines deleted | -156 |
| Tests added | 3 unit tests, plus 1 new ignored real-model throughput harness |
| Runtime behavior changed | Yes: per-row accepts are the default when the caches support compaction |

### Changes by Category

| Category | Count | Summary |
|---|---|---|
| Performance / runtime | 7 | Round loop policy switch, target hooks, Qwen 3.5 verify and rollback, Qwen3Next per-row RoPE and attention, VL delegation, server log text |
| Shared utility | 1 | `build_divergent_row_mask` moved to `mlxcel-core::utils`; Gemma 4 delegates |
| Tests | 3 | Compaction unit tests, mask unit test, throughput comparison harness |
| Documentation | 4 | Batching guide, environment variables, benchmark record, parity-harness wiring |

### Related Commits

| Hash | Type | Message |
|---|---|---|
| `ded78dce` | perf | perf(speculative): commit per-row accepts in batched DFlash |

---

## 8. Follow-up Actions

### Required

- [ ] Run a server-level concurrent-request smoke test through the real collected-burst path at B > 1 with per-row accepts enabled, covering request completion, EOS, max-token rows, and a mixed prompt-length window. The current evidence is a harness burst on four equal-length prompts.
- [ ] Implement per-row drafter context as a separate change and re-measure. The walk is expected to recover above 1.036, but it is not part of this PR's budget.

### Unvalidated Surface

- **No server-level concurrent-request smoke run.** The measured and parity runs use the test harness (`run_burst_with_policy`), not the server's collected-burst scheduler. Row lifecycle, admission, cancellation, and logging in `src/server/batch/dflash_target.rs` are exercised only by the existing suite, not by this measurement.
- **Per-row drafter context is out of scope.** The drafter still sees the shared rectangular minimum prefix. This is the stated reason the walk falls to 1.036, and it means the measured 1.397x is a lower bound on what per-row target accepts can deliver once the drafter is addressed.
- **Measurement caveat.** Another worktree's Rust build was compiling on a few of the 80 cores during the throughput run. Both arms alternated under the same load, and the clamp and B = 1 cells reproduce the earlier quiet-box #2442 record (83.0 against 82.96 to 83.00; 78.09 against 77.70 to 77.83), so the comparison is not load-shifted, but the absolute numbers were not taken on a fully idle box.
- **Single host and checkpoint.** M3 Ultra, macOS 26.6.2, `qwen3.5-4b-4bit` plus `qwen3.5-4b-dflash`, block 16, 128 tokens per row, temperature 0, four truncated prompts. Other hardware, quantizations, block sizes, and prompt distributions are not measured.
- **Fallback paths are not re-validated here.** Quantized, paged, and head-trimmed caches take the unchanged uniform clamp; their correctness is inherited, not re-tested by the per-row gate.
- **VLM path.** `Qwen35VLModel` delegates the hooks, and the attention layer refuses divergent rows when MRoPE position ids are present. No VLM DFlash parity run is included in this PR.

### Monitoring Required

- `clamped_rounds` should stay 0 on the per-row arm and non-zero only when the clamp is explicitly selected or a cache falls back. A non-zero value on a dense Fp16 run means the capability gate regressed.
- Watch aggregate decode throughput and committed accepts per row-round after deployment, especially under mixed prompt lengths where raggedness is highest.
- Run `greedy_parity_dflash_qwen35_batched_b4_matches_b1` and `identical_prompts_dflash_qwen35_b4_matches_b1` on the CI hardware lane whenever MLX, quantization, or kernel dispatch changes.

### Future Improvements

- Extend per-row accepts to quantized and paged caches, which currently fall back to the clamp.
- Consider a fused mask-free attend path if mask construction becomes a measurable fraction of the round cost.
- Evaluate per-row drafter context together with per-row target accepts; the two interact through the walk and should be measured as a pair.

---

## Appendix

### A. Test Results

| Check | Result |
|---|---|
| `cargo test --profile test-fast --features metal,accelerate --test speculative_parity qwen35_dflash_batched -- --ignored --test-threads=1` | 2 passed with per-row accepts as the default |
| `throughput_dflash_qwen35_batched_clamp_vs_per_row` | Completed; three alternating runs per arm |
| `cargo test --profile test-fast --lib models::qwen3_5_tests` | Passed; 30 tests, including 2 new compaction tests |
| mlxcel-core round-loop unit tests | Passed; 15 in `round_loop.rs`, 13 in `round_loop_batched.rs` |
| `build_divergent_row_mask` unit test | Passed |
| `cargo clippy --lib --tests --features metal,accelerate -- -D warnings` | Pass |
| `cargo fmt --all -- --check` | Pass |

### B. Performance Benchmarks

| Run | Clamp tok/s | Per-row tok/s |
|---:|---:|---:|
| 0 | 83.04 | 115.82 |
| 1 | 83.22 | 116.24 |
| 2 | 83.06 | 116.05 |
| Median | 83.06 | 116.05 |

Accepts and rounds:

| Arm | Rounds | Clamped rounds | Walk per row-round | Committed per row-round |
|---|---:|---:|---:|---:|
| Clamp | 96 | 93 | 2.938 | 0.323 |
| Per-row | 67 | 0 | 1.036 | 1.036 |

### C. Measurement Environment

| Field | Value |
|---|---|
| Host | Apple M3 Ultra, 80-core, 512 GB unified memory, macOS 26.6.2 |
| Backend | Metal + Accelerate |
| Target | `models/qwen3.5-4b-4bit` |
| Drafter | `models/qwen3.5-4b-dflash` |
| Block size | 16 |
| Generated tokens | 128 per row |
| Sampling | Temperature 0, greedy |
| Prompts | The four prompts of the batched DFlash parity harness, truncated to the shortest tokenization |
| Method | Two discarded warm-ups, three measured runs per arm alternating clamp and per-row, under `caffeinate -i` and `scripts/with_indexers_paused.sh` |

### D. References

- `src/lib/mlxcel-core/src/drafter/dflash/round_loop.rs`
- `src/lib/mlxcel-core/src/drafter/dflash/round_loop_batched.rs`
- `src/lib/mlxcel-core/src/utils.rs`
- `src/models/qwen3_5.rs`
- `src/models/qwen3_next.rs`
- `src/models/gemma4.rs`
- `src/vision/qwen3_5_vl.rs`
- `tests/speculative_parity/qwen35_dflash_harness.rs`
- `tests/speculative_parity/qwen35_dflash_throughput.rs`
- `docs/benchmark_results/dflash-per-row-accepts-m3ultra-2026-11-06.md`
- Issue #2442, issue #2353, PR #2853, issue #2851, issue #2190, issue #2440
