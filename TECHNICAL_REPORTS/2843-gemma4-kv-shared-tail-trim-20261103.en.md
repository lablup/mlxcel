# Technical Report: PR #2843 - perf(gemma4): trim the KV-shared tail to the sampled row in prefill

**Date**: 2026-11-03
**Status**: In review (open PR; closes #2410)
**Languages**: Rust, Markdown
**Risk Level**: Low

---

## Executive Summary

Gemma 4 E-series checkpoints run their last `num_kv_shared_layers` layers on K/V borrowed from an earlier layer, so those layers write nothing to the cache. A prefill that only needs one position's logits still pushed every prompt row through that tail and discarded all but one row. This PR threads `keep_row: Option<usize>` into the text-model layer loop, slices the hidden state, the per-layer inputs, and both attention-family masks to the sampled row at the first KV-shared layer, and offsets the shared layers' RoPE base so the row keeps its absolute position. On `mlx-community/gemma-4-e2b-it-4bit` the change measured 2.79x lower TTFT at a 1k prompt and 2.83x at 4k, with 0 decided-position disagreements against the full-width trace over 200 chunks and 0 disagreements in the trim-disabled control.

---

## 1. Problem Statement

### 1.1 Background

A Gemma 4 checkpoint with `num_kv_shared_layers > 0` (the E2B / E4B class) splits its decoder stack at `first_kv_shared_layer_idx() = num_hidden_layers - num_kv_shared_layers`. Layers before that index write their K/V to the cache and, for the reference layer of each attention family, also publish a full-length slab in the in-forward `shared_kv_store`. The remaining layers are KV-shared: `is_kv_shared_layer` is true, their `k_proj` / `v_proj` / `k_norm` weights are stripped at load time, and they read the borrowed slab instead of writing a cache of their own.

Sampling a prefill needs the logits of one position only. `Gemma4Model::forward_last_with_caches_and_embeddings` already bounded the LM-head projection to that row (issue #672), but it did so by running `text_model.forward` at full prompt width and slicing the hidden state afterwards. Every prompt row therefore still flowed through the entire KV-shared tail, even though only the sampled row could affect the returned logits. The tail performs attention over borrowed K/V with no cache write, so those extra rows have no side effect that anything reads, only compute.

### 1.2 Existing Issues

- **Issue 1**: A one-row prefill paid the full tail cost. The #672 optimization reduced the LM head from `[B, L, vocab]` to `[B, 1, vocab]`, but the transformer tail still ran at `L` rows. On the E-series class this is the last `num_kv_shared_layers` layers, a measurable share of prefill time at 1k and 4k prompts.
- **Issue 2**: The structure that makes the tail trimmable was already visible (`first_kv_shared_layer_idx`, `is_kv_shared_layer`, `shared_kv_store`), but no caller could tell the forward that only one row was needed, so there was no place to make the decision.
- **Issue 3**: There was no trace arm that exercised the one-row prefill. `examples/logit_trace` always scored the full-width forward, so a trimmed branch could not be compared against the full-width branch on one binary with identical inputs.
- **Issue 4**: A naive trim is silently wrong rather than loudly wrong. Dropping rows without adjusting RoPE positions or the query-row masks produces fluent output with shifted logits, and the issue's original f16 tolerance (`2e-2`) is too loose to catch the resulting error.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Trimmed row rotates at the wrong absolute position | High | Low; the shared cache offset is advanced by `p` and a broken offset fails the parity test |
| Rebuilt mask admits or blocks the wrong keys for an interior row | High | Low; the family declines the trim when exact reconstruction is not provable |
| Capture or speculative consumers receive a narrowed hidden state | High | Low; every capture path is in the guard set |
| Right-padded batched prefill is trimmed incorrectly | Medium | Low; batched input is excluded |
| Small numerical drift versus full width | Medium | Low; 0 disagreements on decided positions, but 6/200 raw top-1 flips exist |

---

## 2. Technical Review

### 2.1 Security

The change adds no input parsing, no external data path, and no new dependency. The only new environment variable is a boolean A/B switch read by the model, and `keep_row` is an internal index bounded by `p < l` before it is used. The trim cannot be reached from a request that the caller did not already route through a `*_last_*` entry.

**Checklist:**
- [x] Input validation: `keep_row` is accepted only when `p < l`, `b == 1`, and `l > 1`
- [ ] Authentication/Authorization: not applicable
- [ ] Data encryption: not applicable
- [x] Logging (excluding sensitive data): one debug line reports layer, row, and prompt width only

**Issues Found:**

| Issue | Severity | Status |
|-------|----------|--------|
| None in this change | - | - |

### 2.2 Performance

The saved work is proportional to the shared tail depth. Layers before `first_kv_shared_layer_idx()` still run at full width because they populate the cache and the shared slab the tail reads. From the first shared layer onward, the hidden state is `[1, 1, hidden]`, the per-layer input is sliced to the same row, and each shared layer runs one query against the borrowed keys instead of `L` queries. The KV caches are updated exactly once, by the layers that own them, before the trim point.

The measurement below is the PR's evidence on an M3 Ultra. The "before" arm is not a pre-PR binary: it is the same binary with `MLXCEL_GEMMA4_NO_KV_SHARED_TAIL_TRIM=1`, which keeps the tail at full query width. Three interleaved runs per arm, `mlxcel-bench-decode --prompt-tokens N -n 1 --warmup-tokens 1`. No benchmark CSV was committed for these numbers; they live in the PR body only.

**Performance Impact:**

| Area | Before (trim disabled, same binary) | After (trim on) | Improvement |
|------|-------------------------------------|-----------------|-------------|
| TTFT, 1024-token prompt | 211.08 / 211.36 / 211.45 ms | 75.52 / 75.92 / 75.72 ms | 2.79x |
| TTFT, 4096-token prompt | 882.08 / 876.85 / 877.65 ms | 309.30 / 311.62 / 308.98 ms | 2.83x |
| Forward work in the shared tail | `L` query rows per shared layer | 1 query row per shared layer | proportional to tail depth |
| Decode, non-last-row prefill, batched prefill | unchanged | unchanged | none by design |

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: `Gemma4TextModel::forward_with_speculative_sinks` gains a twelfth argument, `keep_row: Option<usize>`. Every in-repo call site is updated in this PR, and every caller except the two `*_last_*` prefill entries passes `None`, which is documented as byte-identical to the previous behavior. Any out-of-tree caller of this public Rust method would need the extra `None`.
- **New Dependencies**: None.
- **Compatibility**: The trim applies only when the checkpoint has `num_kv_shared_layers > 0`, the input is a single sequence, more than one row is present, and no capture path is active. Text-only and VLM last-row prefills both reach it; decode, batched prefill, speculative decode, MTP verify, and capture runs do not.
- **Configuration**: `MLXCEL_GEMMA4_NO_KV_SHARED_TAIL_TRIM=1` disables the trim, documented in `docs/environment-variables.md` because it is a runtime A/B switch an operator can legitimately set, not a test-only knob.

### 2.4 Code Quality

- **Test Coverage**: 3 new lib tests in `src/models/gemma4_tests.rs` under `mod kv_shared_tail_trim` (476 lines including a 4-layer KV-shared fixture). The `gemma4_kv_shared_tail` filter passes 3/3; the full `gemma4` filter reports 275 passed, 40 ignored.
- **Code Complexity**: The decision is concentrated in one gate before the layer loop and one commit block inside it. Mask derivation is factored into `trimmed_tail_row_masks`, `trimmed_family_row_mask`, `slice_mask_query_row`, and `single_query_row_mask`, each with a narrow contract.
- **Technical Debt**: The gate is a long conjunction of exclusions. That verbosity is deliberate: each clause corresponds to a consumer that would observe the narrowed width. The alternative, deriving the guards implicitly inside the loop, would make it harder to see which paths stay byte-identical.
- **Verification**: `cargo clippy --profile test-fast -p mlxcel --lib --tests -- -D warnings` and `cargo fmt --check` pass.

---

## 3. Technical Decisions

### 3.1 Trim inside the layer loop, not after the forward

**Context:**
The caller already knows which row it needs, and the model already knows where the shared tail begins. The trim could be applied by the caller after a full forward, but by then the compute has already been spent.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Slice logits or hidden state after `text_model.forward` | No model change | Saves nothing; the tail still runs at full width |
| **Chosen: slice at the first shared layer inside the forward** | Removes the tail compute, which is the target | Adds state to the layer loop |
| Change the tail layers to a dedicated single-row forward | Clean separation | Duplicates attention, mask, and RoPE logic |

**Rationale:**
The tail is exactly the region that can be narrowed without changing the caches, because those layers write no cache entries. Committing the trim immediately before the first shared layer lets the same loop, attention code, and RoPE handling serve both widths.

**Trade-offs:**
The layer loop now carries `cur_l` and `tail_row_offset` alongside `l`. Log lines and per-layer input derivation use `cur_l`, so any future code added inside the loop must choose the right width variable.

### 3.2 Derive exact one-row masks and decline the trim otherwise

**Context:**
A trimmed query still needs the right mask over the retained keys. The text model has two attention families (`full_attention`, `sliding_attention`) and several mask shapes: a plain prefill mask may be present, or `None` when the dense path applies no mask or the native windowed route omits one.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Always rebuild the mask from the row's absolute position | Uniform | Wrong for a sliding family past the window, where retained keys are already clamped |
| Always slice the existing mask | Exact when a mask exists | Panics or falls back when the mask is `None` |
| **Chosen: slice when present, rebuild only when keys are full length, last row otherwise, decline on doubt** | Exact in every accepted case | Interior rows past the window leave the tail untrimmed |

**Rationale:**
`trimmed_family_row_mask` returns `Some(Some(sliced))` when the family has a plain mask, because row `p` of a causal or windowed mask already states exactly which keys that query may see. With no plain mask it returns `Some(None)` for the last row, where a one-row causal or windowed attention over the retained keys is the decode shape the untrimmed path would reach anyway. For an interior row it rebuilds a one-row mask only when the keys are full length (`global_live == global_offset` for the global family, `sliding_live + query_len <= sliding_window` for the sliding family), so column `k` still maps to logical key position `k`. Any other combination returns `None`, and the caller leaves the whole tail untrimmed.

**Trade-offs:**
The trim sometimes does not happen even though it safely could. That is intentional: a missed optimization costs latency, while a wrong mask costs correctness. The debug log distinguishes the two outcomes.

### 3.3 Preserve absolute position through the shared cache offset

**Context:**
The trimmed row keeps the query index `p` as its absolute position, but the shared layers do not use a normal cache offset, they use the reference layer's offset from `shared_kv_store`.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Rotate the trimmed query at offset 0 and rely on relative position | No offset change | Gemma 4 RoPE is absolute in this path; the row would attend as if it were the first token |
| Re-rotate the borrowed K/V to the new origin | Conceptually simple | Mutates or copies the shared slab every time |
| **Chosen: set the shared cache offset to `ref_offset + p`** | One addition; keeps the slab untouched | A second offset variable to track |

**Rationale:**
The shared layers derive their query positions from `cache.set_offset(...)`. Adding `tail_row_offset` makes the single retained query rotate at its original absolute position while the borrowed keys retain theirs. The tests deliberately break this offset and observe a 5.7e-4 logit shift, which fails the 1e-4 parity bound.

**Trade-offs:**
`tail_row_offset` is only nonzero after the trim commits. Layers before that point and every other caller see zero, so the change is a no-op for them.

### 3.4 Keep every non-target path byte-identical through an explicit guard set

**Context:**
The same forward serves decode, batched prefill, speculative verify, MTP, hidden capture, shared-K/V capture, and explicit per-layer capture. A narrowed hidden state would break any consumer that expects full prompt width.

**Rationale:**
`trim_row` is `Some` only when all of the following hold: the trim is not disabled by the environment, `num_kv_shared_layers > 0`, `first_kv_shared_layer_idx() > 0`, `b == 1`, `l > 1`, `p < l`, there is no speculative sink, `capture_layer_ids` is absent or empty, there is no padding, and `per_row_valid_end`, `row_offsets`, and `tree_positions` are all `None`. At the commit point the code additionally requires that the layer is KV-shared, that its `kv_shared_layer_index` is already present in `shared_kv_store`, and that both families' masks could be derived. Failing any condition leaves `cur_l == l` and the previous behavior.

**Trade-offs:**
Longer condition list and one duplicated notion of "capture" between the gate and the capture setup. The duplication is cheap compared to a silent width mismatch in a speculative path.

### 3.5 A single-binary A/B switch plus a trace arm

**Context:**
The trim is meant to be numerically neutral against the full-width path. Proving that with two binaries introduces build-to-build differences and makes a disagreement ambiguous.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Compare against a pre-PR build | Closest to the old behavior | Two builds, two environments, ambiguity when numbers differ |
| **Chosen: `MLXCEL_GEMMA4_NO_KV_SHARED_TAIL_TRIM=1` on the same binary** | One binary, identical weights and kernels; isolates the branch | Adds a runtime switch that must stay documented |
| Test-only feature flag | No production switch | The trace arm runs a normal release build |

**Rationale:**
The variable turns the new branch off without reverting the code, so the trimmed and untrimmed arms differ only in the rows fed to the tail. `examples/logit_trace` gains `MLXCEL_TRACE_LAST_ROW=1`, which scores each chunk's final position through `forward_last_logits` and emits it at the same `(chunk, pos)` the full-width arm writes, so `scripts/compare_logit_traces.py` compares directly. Setting the kill switch in that arm runs the control. A `RUST_LOG=mlxcel::models::gemma4=debug` line reports when the trimmed branch is taken, so a run that never reached it is visible.

**Trade-offs:**
The switch exists in production code for an experiment. It is documented as an A/B control rather than a serving mode.

---

## 4. Implementation Details

### 4.1 Architecture Changes

```
[Before]
forward_last_logits* ─> forward_last_with_caches_and_embeddings
                          └─> text_model.forward (full width L)
                                └─> all layers, including the KV-shared tail, run L rows
                          └─> logits_at(hidden, last_pos)

[After]
forward_last_logits* ─> forward_last_with_caches_and_embeddings(Some(last_pos))
                          └─> forward_with_speculative_sinks(keep_row = Some(p))
                                ├─> layers 0 .. first_kv_shared_layer_idx: full width L
                                │     (populate caches and shared_kv_store)
                                ├─> first KV-shared layer: slice h, per-layer inputs,
                                │     global and sliding masks to row p; cur_l = 1;
                                │     tail_row_offset = p
                                └─> shared tail: 1 query row at offset ref_offset + p
                          └─> hidden width 1 ? logits_at(hidden, 0) : logits_at(hidden, last_pos)
```

### 4.2 Key Code Changes

**File: `src/models/gemma4.rs`**

The gate that decides whether the trim is legal, evaluated once before the layer loop:

```rust
let trim_row: Option<i32> = keep_row
    .filter(|_| !kv_shared_tail_trim_disabled())
    .filter(|&p| {
        self.config.num_kv_shared_layers > 0
            && self.config.first_kv_shared_layer_idx() > 0
            && b == 1
            && l > 1
            && p < l as usize
            && sinks.is_none()
            && capture_layer_ids.is_none_or(|ids| ids.is_empty())
            && !has_padding
            && per_row_valid_end.is_none()
            && row_offsets.is_none()
            && tree_positions.is_none()
    })
    .map(|p| p as i32);
```

Cache geometry is sampled before the loop because the loop holds a mutable borrow of `caches`. The commit happens at the first shared layer, before any of that layer's inputs are derived:

```rust
if !tail_trimmed
    && i == self.config.first_kv_shared_layer_idx()
    && layer.self_attn.is_kv_shared_layer
    && layer.self_attn.kv_shared_layer_index
        .is_some_and(|ref_idx| shared_kv_store.contains_key(&ref_idx))
    && let (Some(p), Some(geometry)) = (trim_row, trim_geometry)
    && let Some((global_row, sliding_row)) = self.trimmed_tail_row_masks(...)
{
    h = slice_axis(&h, 1, p, p + 1);
    if let Some(inputs) = per_layer_inputs.as_mut() {
        *inputs = slice_axis(inputs, 1, p, p + 1);
    }
    global_mask = global_row;
    sliding_mask = sliding_row;
    cur_l = 1;
    tail_row_offset = p;
    tail_trimmed = true;
}
```

Shared layers then read at their published offset plus the retained row's position:

```rust
cache.set_offset(*ref_offset + tail_row_offset);
```

The mask contract is the part that makes the trim exact:

```rust
fn trimmed_family_row_mask(
    mask: Option<&MlxArray>,
    q_pos: i32,
    key_len: i32,
    full_length_keys: bool,
    p: i32,
    query_len: i32,
) -> Option<Option<UniquePtr<MlxArray>>> {
    if let Some(mask) = mask {
        return Some(Some(slice_mask_query_row(mask, p)));
    }
    if p + 1 == query_len {
        return Some(None);
    }
    if !full_length_keys || key_len <= 0 {
        return None;
    }
    Some(Some(single_query_row_mask(q_pos, key_len, None)))
}
```

`trimmed_tail_row_masks` calls this for each family that actually has a layer, using `global_offset + p` / `global_live + query_len` for the global family and `sliding_offset + p` / `sliding_live + query_len` with the window check for the sliding family. A `None` from either family aborts the trim for the whole tail.

**File: `src/models/gemma4.rs` (`forward_last_with_caches_and_embeddings`)**

```rust
let hidden = self.text_model.forward_with_speculative_sinks(
    input_ids, input_embeddings, caches, mask, per_layer_inputs,
    None, None, false, None, None, None,
    Some(last_pos),
);
let row = if mlxcel_core::array_shape(&hidden)[1] == 1 { 0 } else { last_pos };
self.logits_at(&hidden, row)
```

**Reason for change:** the caller no longer runs `text_model.forward` and slices afterwards. It requests the single row directly, and reads row 0 when the tail actually trimmed, falling back to `last_pos` when it did not.

**File: `examples/logit_trace.rs`**

`MLXCEL_TRACE_LAST_ROW=1` selects the one-row arm. It is restricted to the standard text arm and rejects combination with `MLXCEL_TRACE_GEMMA4`, audio, batched rows, tensor parallel, image input, or contiguous tracing. Each chunk is scored at `input_len - 1` through `forward_last_logits` and emitted at the same `(chunk, pos)` as the full-width arm.

**File: `docs/environment-variables.md`**

Documents `MLXCEL_GEMMA4_NO_KV_SHARED_TAIL_TRIM` as a Gemma 4 A/B switch, including its scope: only `*_last_*` prefill entries trim, and only when `num_kv_shared_layers > 0`.

### 4.3 Data Model Changes

None. No checkpoint format, config field, or serialized structure changes.

---

## 5. Learning Points

### 5.1 KV-shared layers make a prefill tail trimmable

**Concept:**
A KV-shared layer borrows keys and values from an earlier layer and writes no cache entry of its own. Its output is a function of the borrowed slab and its input rows, but nothing downstream reads those rows in a cache.

**Application in this PR:**
The last `num_kv_shared_layers` layers are exactly the region where dropping prompt rows is side-effect free, provided the rows being kept are rotated and masked correctly. Layers before the split still run at full width because they are the ones populating the cache and the shared slab.

**Common Use Cases:**
- Prefill that samples one position (`logits_to_keep = 1`) on architectures with cross-layer KV sharing.
- Memory-bounded LM-head projection, the existing #672 pattern this PR extends from the head to the transformer tail.

### 5.2 Exact mask reconstruction for a single query row

**Concept:**
A `[queries, keys]` additive mask can be sliced on the query axis without touching the key axis, because row `p` already encodes which keys query `p` may attend to. When no mask exists, the row has to be rebuilt from the query's logical position and the cache geometry, and that reconstruction is only valid when the retained keys map one-to-one to logical positions.

**Application in this PR:**
`slice_mask_query_row` handles the mask-present case. `single_query_row_mask` rebuilds a `0 / -inf` row with `arange(key_len) <= q_pos`, optionally bounded below by the sliding window. The `p + 1 == query_len` special case returns no mask at all, because one-row causal attention over the retained keys is already the decode shape.

**Common Use Cases:**
- Any optimization that narrows the query axis of a prefill while keeping full key history.
- Diagnosing mask bugs by checking whether the retained-key coordinate system is logical or clamped.

### 5.3 Testing numerical neutrality with a deliberately sensitive fixture

**Concept:**
A parity test only proves something if the fixture can fail. Two failure modes matter here: weights with too little entropy make attention insensitive to position, and a loose tolerance hides small but systematic errors.

**Application in this PR:**
The fixture uses deterministic pseudo-random weights from a linear congruential generator, because a linear ramp makes key vectors near-parallel inside the 2-D test head and a broken RoPE could pass unnoticed. The parity bound is 1e-4 rather than the issue's 2e-2 f16 tolerance, because a broken RoPE offset moves the logits by 5.7e-4 and a broken mask row by 9.9e-4. Both mutations were tested and both fail at 1e-4. A correct trim reproduces the reference row exactly in f32. The test also asserts the trimmed hidden width is 1, so a silent fallback cannot make the numeric comparison vacuous.

**Common Use Cases:**
- Verifying an optimization that is supposed to be exact versus a reference path.
- Building minimal architecture fixtures (4 layers, 2 KV-shared) that exercise every branch without loading a real checkpoint.

---

## 6. Further Learning

### Key Terms

| Keyword | Description | Relevance |
|---------|-------------|-----------|
| KV-shared layer | A decoder layer that reads K/V from an earlier layer instead of computing and caching its own | The last `num_kv_shared_layers` layers are the region this PR trims |
| `keep_row` | Optional sampled-row index threaded into `forward_with_speculative_sinks` | The single new parameter that expresses "only this row's logits are needed" |
| RoPE base offset | The absolute position used when rotating a query | Adding `p` keeps the trimmed row at its original position |
| Additive attention mask | A `0 / -inf` f32 mask added to attention scores | Must be sliced or rebuilt to one query row |
| `live_len` vs `offset` | `offset` is monotonic for RoPE; `live_len` is how many prior keys the cache actually returns | Mask sizing must use the live length, which is why the geometry tuple samples both |
| Decided position | A trace position whose reference top-two gap is at least the `--decided` threshold | 0/104 disagreements is the metric that gates parity |

### Related Technologies/Frameworks

- **MLX**: Apple's array framework used by the runtime.
  - Official docs: https://ml-explore.github.io/mlx/
- **mlx-vlm**: upstream reference implementation whose Gemma 4 language model trims to `logits_to_keep`.
  - https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma4/language.py
- **Gemma 4**: the model family with cross-layer KV sharing in its E-series checkpoints.

### Related PRs/Issues

- Issue #2410: the performance issue this PR closes. It proposed the `keep_row` parameter, the mask and RoPE handling, the exclusion set, and the trace and TTFT evidence required for acceptance.
- PR #672: the earlier change that bounded the LM-head projection to one position. This PR extends the same idea from the head to the transformer tail.
- PR #2842: an unrelated recent report in this repository, used only as a formatting reference.
- Upstream `Blaizzy/mlx-vlm#2157`: the corresponding upstream trim, which keeps full width when hidden-state captures are requested.

---

## 7. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 4 |
| Lines added | +810 |
| Lines deleted | -25 |
| Tests added | 3 lib tests |
| Commits | 2 |

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| Performance | 1 | Trim the KV-shared tail to the sampled row in `src/models/gemma4.rs` |
| Tests | 1 | KV-shared fixture plus parity, width, and capture-guard tests |
| Tooling | 1 | `MLXCEL_TRACE_LAST_ROW=1` one-row arm in `examples/logit_trace.rs` |
| Documentation | 1 | `MLXCEL_GEMMA4_NO_KV_SHARED_TAIL_TRIM` in `docs/environment-variables.md` |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `5105d8e4` | perf | perf(gemma4): trim the KV-shared tail to the sampled row in prefill |
| `9febc6a1` | docs | docs: document the Gemma 4 KV-shared tail trim kill switch |

---

## 8. Follow-up Actions

### Required

- [ ] None blocking merge. The PR is open and closes #2410.

### Monitoring Required

- Watch TTFT on KV-shared Gemma 4 checkpoints at 1k and 4k prompts after merge, since the measured numbers were produced on one M3 Ultra and were not committed as a benchmark CSV.
- Keep the numerical parity gate in mind for any future change to the tail: the 200-chunk trace reported 0/104 decided-position disagreements and 6/200 raw top-1 flips, all at reference top-two gaps under 0.13.
- A run that is expected to trim should log `trimmed the KV-shared tail to the sampled row layer=... keep_row=... prompt_rows=...` under `RUST_LOG=mlxcel::models::gemma4=debug`. Its absence means the guard set declined the trim.

### Future Improvements

- Right-padded batched prefill is explicitly out of scope. Supporting it would need per-row keep indices and a check that padding does not enter the shared tail.
- Interior-row trimming past the sliding window currently declines for the sliding family, leaving the whole tail untrimmed. A clamped-key reconstruction could extend the optimization to that case.
- The TTFT figures are not reproducible from a committed artifact. If this becomes a regression gate, add a committed benchmark result under the project's benchmark documentation.
- Capture paths deliberately keep full width. If a future consumer can accept a narrowed hidden state, the guard set can be relaxed for that consumer only.

---

## Appendix

### A. Test Results

| Command | Result |
|---------|--------|
| `cargo test --profile test-fast -p mlxcel --lib gemma4_kv_shared_tail` | 3 new tests pass |
| `cargo test --profile test-fast -p mlxcel --lib gemma4` | 275 passed, 40 ignored |
| `cargo clippy --profile test-fast -p mlxcel --lib --tests -- -D warnings` | pass |
| `cargo fmt --check` | pass |

The three tests:

- `gemma4_kv_shared_tail_trim_matches_full_width_rows`: a 4-layer fixture with `num_kv_shared_layers = 2`, per-layer inputs enabled, and 12 deterministic tokens. Three geometries (6-token cold prefill inside the window, 12-token cold prefill past the window, and a 4-token warm prefix plus an 8-token chunk), each at the final and an interior row. Asserts the trimmed hidden width is 1 and the logits match the full-width row within 1e-4, including an interior case where both families have no plain mask.
- `gemma4_kv_shared_tail_trim_narrows_the_hidden_state`: `keep_row = Some(5)` returns width 1; `keep_row = None` returns width 12.
- `gemma4_kv_shared_tail_trim_skipped_when_capturing_layers`: with `capture_layer_ids` set, both the returned hidden state and the captured layer output keep width 12, for the final row and an interior row.

Mutation checks reported in the PR: breaking the RoPE row offset shifts the logits by 5.7e-4 and breaking the mask row by 9.9e-4; both fail the 1e-4 parity bound. The issue's 2e-2 f16 tolerance would not have caught either.

### B. Performance Benchmarks

Environment: M3 Ultra 512 GB, macOS 26.6.2, base commit `afb208eb`, MLX `7ca40033132dae444d45f7fc3daae57e24c13f6c`, model `mlx-community/gemma-4-e2b-it-4bit`.

Trace comparison, 200 chunks x 256 tokens, `scripts/compare_logit_traces.py --decided 2`:

| arm | top-1 disagreement | on decided (gap >= 2) | max gap at a disagreement |
|---|---|---|---|
| `MLXCEL_TRACE_LAST_ROW=1` vs full width | 6 / 200 | 0 / 104 | 0.125 |
| same arm with the trim disabled | 0 / 200 | 0 / 104 | none |

All 6 disagreements had reference top-two gaps under 0.13 and landed on the runner-up. Logit delta p50 0.109, p99 0.500, the 4-bit per-row-vs-batched kernel rounding class. Perplexity moved from 88.82 to 88.42. `RUST_LOG=mlxcel::models::gemma4=debug` confirmed the trimmed branch on every chunk with `trimmed the KV-shared tail to the sampled row layer=15 keep_row=256 prompt_rows=257`.

TTFT, `mlxcel-bench-decode --prompt-tokens N -n 1 --warmup-tokens 1`, three interleaved runs per arm. The control sets `MLXCEL_GEMMA4_NO_KV_SHARED_TAIL_TRIM=1` on the same binary; there was no pre-PR binary measurement.

| prompt | trim on (ms) | trim off (ms) | speedup |
|---|---|---|---|
| 1024 | 75.52 / 75.92 / 75.72 | 211.08 / 211.36 / 211.45 | 2.79x |
| 4096 | 309.30 / 311.62 / 308.98 | 882.08 / 876.85 / 877.65 | 2.83x |

No benchmark CSV was committed. The PR changes four files, none of them a benchmark result file.

### C. References

- Issue #2410: perf: Trim the Gemma 4 KV-shared tail to the sampled row in prefill
- PR #2843: the change described here
- Upstream `Blaizzy/mlx-vlm#2157` and `mlx_vlm/models/gemma4/language.py` (`logits_to_keep`)
- `docs/environment-variables.md`: the documented A/B switch
- `scripts/compare_logit_traces.py`: the decided-position comparison used for the trace evidence
