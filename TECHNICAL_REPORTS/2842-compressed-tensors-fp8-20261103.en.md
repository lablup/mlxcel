# Technical Report: PR #2842 - feat: load compressed-tensors float-quantized FP8 checkpoints

**Date**: 2026-11-03
**Status**: In review (open PR; closes #2411)
**Languages**: Rust, Markdown
**Risk Level**: Low

---

## Executive Summary

This PR teaches the Qwen3.5 loaders to reconstruct compressed-tensors `float-quantized` FP8 checkpoints, the layout llm-compressor's `FP8-dynamic` releases use (`RedHatAI/Qwen3.5-4B-FP8-dynamic` among them). Previously only the vendor `quant_method: "fp8"` block layout was recognized, so these checkpoints reached the model as raw `uint8` bytes with their per-channel or per-block `weight_scale` unused. The change adds a detection and requantization module (`fp8_ct`), a route selector (`fp8_checkpoint`) shared by the Qwen3.5 text, VLM, and special loaders, and generalizes the existing block reconstruction from square 128x128 tiles to independent `[block_rows, block_cols]` extents with edge padding. Both FP8 layouts converge on MLX-native `mxfp8`, so the forward pass is unchanged. On M3 Ultra the converted checkpoint disagreed with `Qwen/Qwen3.5-4B` at 0 of 116 decided positions, while an arm that ignores the scale produced non-finite logits at the first position.

---

## 1. Problem Statement

### 1.1 Background

llm-compressor's `FP8-dynamic` export declares `quantization_config: {quant_method: "compressed-tensors", format: "float-quantized"}`. Every converted `Linear` is stored as raw E4M3 bytes in `<name>.weight` plus `<name>.weight_scale`: one multiplier per output channel (`strategy: "channel"`, written as `[out, 1]`) or one per `block_structure: [rows, cols]` tile. The true weight is `e4m3_decode(byte) * weight_scale[channel or tile]`. Modules listed in `quantization_config.ignore` (the vision tower, every `linear_attn` projection, `lm_head`, `re:^mtp.*`) stay bf16 and carry no scale at all.

MLX has no float8 dtype, so these bytes arrive in the loader as `uint8`. The existing detection in `src/models/fp8_block.rs` returns `Ok(None)` unless `quant_method` is exactly `fp8`, which means the compressed-tensors declaration matched no route.

### 1.2 Existing Issues

- **Issue 1**: The E4M3 bytes reached the model as `uint8` weights with `.weight_scale` never applied. Each projection was off by its own per-channel factor, silently, because nothing in the loader paired the bytes with the scale.
- **Issue 2**: `.weight_scale` and `.input_scale` surfaced as unexpected keys, so a strict weight map could fail the load outright.
- **Issue 3**: The vendor block reconstruction in `fp8_block.rs` hard-coded `SUPPORTED_FP8_BLOCK = 128` on both axes. compressed-tensors carries the tile in `block_structure`, so a non-square or differently sized tile could not reuse it.
- **Issue 4**: The same FP8 pre-pass was wired independently into three Qwen3.5 loaders (text, VLM, special). Adding a second layout naively would have duplicated the branch in two places per loader.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Wrong output accepted as correct (scale pairing mismatch) | High | Low, after this PR; refusal paths close it |
| Existing vendor FP8 and ModelOpt NVFP4 checkpoints regress | High | Low; routing is disjoint by `.weight_scale_2` and by `num_bits` |
| Peak load memory grows with the new reconstruction | Medium | Low; one tensor at a time with an `eval` per tensor |
| Another compressed-tensors family remains unsupported | Medium | Certain; other model families are out of scope |

---

## 2. Technical Review

### 2.1 Security

The change parses checkpoint configuration and tensor metadata. It does not execute code from the checkpoint, and every scale or shape mismatch is a typed load error rather than a panic or a silent reinterpretation.

**Checklist:**
- [x] Input validation: `type`, `num_bits`, `dynamic`, `strategy`, `block_structure`, and scale shapes are all validated with named errors
- [ ] Authentication/Authorization: not applicable
- [ ] Data encryption: not applicable
- [x] Logging (excluding sensitive data): logs tensor counts, layout, conversion count, and elapsed time only

**Issues Found:**

| Issue | Severity | Status |
|-------|----------|--------|
| A corrupt checkpoint could be mis-scaled silently | High | Fixed: unsupported layouts fail the load by name |
| A mismatched channel scale could broadcast over the wrong axis | High | Fixed: shape is checked against `rows` before multiplying |

### 2.2 Performance

The conversion is one tensor at a time. For each pair the pass removes the raw bytes and the scale from the map, decodes through the device E4M3 lookup table, multiplies by the scale, drops both inputs, requantizes to `mxfp8`, then forces realization with `eval` on the packed plane and its scales before moving to the next tensor. Peak memory therefore holds one f32 reconstruction plus one packed result, not four copies of the whole checkpoint.

The alternative of dequantizing the whole checkpoint to bf16 (what mlx-vlm does) would double weight memory against the checkpoint and change the forward dtype.

**Performance Impact:**

| Area | Before | After | Improvement |
|------|--------|-------|-------------|
| Supported checkpoint layout | Vendor `fp8` only | Vendor `fp8` plus compressed-tensors `float-quantized` | Not a speed change; removes a manual conversion step |
| `RedHatAI/Qwen3.5-4B-FP8-dynamic` load | Unsupported | 1.8 s load, 7.75 GB peak | New capability |
| Forward pass | unchanged (`mxfp8`) | unchanged (`mxfp8`) | none |

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: No. The vendor route and ModelOpt NVFP4 route keep their behavior. `reconstruct_block_fp8_tensor` gains two required arguments, but its only caller is in the same module and was updated.
- **New Dependencies**: None.
- **Compatibility**: Qwen3.5 text, Qwen3.5 VLM, and the special loader are wired. Other model families that might ship compressed-tensors FP8 are not covered by this PR.

### 2.4 Code Quality

- **Test Coverage**: 9 new tests in `src/models/fp8_ct_tests.rs` (785 lines), plus one call-site update in `fp8_block_tests.rs`. The `fp8` filter reports 29 passing tests in total, which includes the pre-existing block tests.
- **Code Complexity**: Two new modules with a narrow public surface. Detection and conversion are separated, and the refusals are explicit.
- **Technical Debt**: Reduced for this layout; the remaining duplication between the two reconstruction functions is limited to scale pairing, which is deliberately not shared.

---

## 3. Technical Decisions

### 3.1 Requantize to `mxfp8` instead of dequantizing to bf16

**Context:**
The checkpoint stores quantized weights. The loader could either restore them to bf16 or requantize them to a native MLX layout the model already knows how to run.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Dequantize to bf16 (mlx-vlm's approach) | Simple, exact | Doubles weight memory against the checkpoint; changes the forward dtype |
| **Chosen: requantize to `mxfp8`** | Same memory class as the checkpoint; reuses the vendor path's measured error bound; no forward change | Adds one quantization step of error |
| Keep raw bytes and dequantize per matmul | No load-time work | Requires a new forward kernel; large blast radius |

**Rationale:**
`mxfp8` (`group_size` 32, `bits` 8) is what `requantize_block_fp8_weights` already produces for the vendor layout, with an error bound the project has measured and pinned. Reusing that target means the model constructor and forward pass need no new code, and the config merge is identical for both routes.

**Trade-offs:**
The reconstruction adds the mxfp8 float-quantization error on top of the original E4M3 error. The tests bound the result at half an E4M3 step plus the existing mxfp8 bound, and the logit trace shows no disagreement at decided positions.

### 3.2 Refuse unsupported layouts by name rather than mis-scale

**Context:**
compressed-tensors can declare per-tensor, per-group, per-channel, or per-block scales, and several bit widths. Each pairs a differently shaped sidecar with the weight.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Best-effort conversion for any float-quantized config | Loads more checkpoints | Applying a channel pairing to a tensor scale yields fluent but wrong output |
| **Chosen: accept channel and block, fail the load on the rest** | Wrong output is impossible for the refused set | Some checkpoints remain unsupported |

**Rationale:**
A mis-scaled projection does not raise an error at inference time; it produces plausible text with wrong values. The failure mode is expensive to detect later. The detection therefore returns `Err` naming the exact field (`weights.strategy`, `weights.num_bits`, `weights.dynamic`, `weights.type`, `block_structure`) for anything it cannot pair correctly, including a per-group mix of channel and block layouts in one checkpoint.

**Trade-offs:**
`strategy: "tensor"` and `"group"` checkpoints, other bit widths, dynamic weights, and `mixed-precision` exports are explicitly out of scope and fail fast.

### 3.3 Route per tensor instead of by module name

**Context:**
The config's `ignore` list is expressed in HF module paths. Matching it in the loader would mean reimplementing the prefix rules and keeping them in sync with each release.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Apply the `ignore` list to decide which tensors convert | Mirrors the config | Duplicates prefix matching; brittle across releases |
| **Chosen: a tensor converts only if `.weight` is `uint8`, `.weight_scale` exists, and `.weight_scale_2` does not** | Data-driven; disjoint from ModelOpt by construction | Relies on the checkpoint actually storing bytes |

**Rationale:**
The ignored projections arrive as bf16 and have no scale, so they fail the first condition. ModelOpt NVFP4 triplets carry `.weight_scale` but always beside `.weight_scale_2`, so the third condition excludes them and leaves them to `sanitize.rs`. A bf16 tensor that happened to carry a scale is also skipped rather than decoded as E4M3.

**Trade-offs:**
The route trusts tensor dtype and key presence over configuration intent. The tests cover both the NVFP4 triplet and a pre-decoded weight to hold that line.

### 3.4 One route module for three loaders

**Context:**
The FP8 pre-pass appears in the Qwen3.5 text loader, the VLM loader, and the special loader. Each needs both a config merge and a weight pre-pass.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Branch on the layout in each loader | Local | Six branch points to keep consistent; easy to wire config and weights differently |
| **Chosen: `fp8_checkpoint_quantization` plus `requantize_fp8_weights`** | One detection and one dispatch per loader; the two layouts cannot both apply | One extra indirection |

**Rationale:**
`fp8_checkpoint_quantization` runs the vendor detection first and the compressed-tensors detection second. The two are mutually exclusive by `quant_method`, so at most one returns `Some`. The `Fp8Checkpoint` enum carries the route through to both the config merge and the weight pass, which keeps the pair from diverging.

**Trade-offs:**
`merge_quantization_block` was extracted from `merge_fp8_block_quantization` so the "an explicit `quantization` always wins" rule lives in one place.

### 3.5 Generalize block reconstruction rather than fork it

**Context:**
compressed-tensors block scales use `block_structure: [rows, cols]`, while the vendor path assumes 128x128 on both axes.

**Rationale:**
`reconstruct_block_fp8_tensor` now takes `block_rows` and `block_cols` separately, computes the per-axis block counts, pads to whole tiles, and slices the padding back off. The vendor caller passes `(block, block)`. The E4M3 decode was extracted into `decode_fp8_bytes_to_f32` and shared with the channel route, so both routes decode bytes identically and differ only in how the scale is applied.

**Trade-offs:**
`reconstruct_block_fp8_tensor` gained two arguments. It has a single caller in the same module, and the existing round-trip test was updated to pass the second extent.

---

## 4. Implementation Details

### 4.1 Architecture Changes

```
[Before]
config ──> qwen_fp8_block_quantization ──> requantize_block_fp8_weights
              (returns None for compressed-tensors)
              │
              └──> 3 loaders (text, VLM, special), each branching locally

[After]
config ──> fp8_checkpoint_quantization
              ├─> qwen_fp8_block_quantization      (quant_method: "fp8")
              └─> compressed_tensors_fp8_quantization (quant_method: "compressed-tensors")
                        │
                        ▼
                  Fp8Checkpoint
                  ├─> merge_fp8_quantization   (config)
                  └─> requantize_fp8_weights   (weights)
                            ├─> requantize_block_fp8_weights
                            └─> requantize_compressed_tensors_fp8_weights
```

### 4.2 Key Code Changes

**File: `src/models/fp8_ct.rs`**

Detection validates the whole declaration and returns the scale layout, or fails with the field name:

```rust
pub fn compressed_tensors_fp8_quantization(
    full_config: &Value,
) -> Result<Option<Fp8CtQuantization>, String> {
    let Some(quant_config) = full_config.get("quantization_config").or_else(|| {
        full_config.get("text_config").and_then(|text| text.get("quantization_config"))
    }) else {
        return Ok(None);
    };
    // quant_method must be "compressed-tensors"; format must be "float-quantized"
    // (or absent, in which case each group's format decides).
    // Each config_groups.*.weights must be type "float", num_bits 8, static,
    // with strategy "channel" or a two-dimensional block_structure.
}
```

`requantize_compressed_tensors_fp8_weights` routes per tensor and bounds peak memory:

```rust
let is_e4m3_bytes = weights
    .get(&weight_key)
    .is_some_and(|weight| mlxcel_core::array_dtype(weight) == dtype::UINT8);
if !is_e4m3_bytes { continue; }

let restored = match layout {
    Fp8CtScaleLayout::Channel => reconstruct_channel_fp8_tensor(&weight_key, &raw, &scale, &lut)?,
    Fp8CtScaleLayout::Block { rows, cols } =>
        reconstruct_block_fp8_tensor(&weight_key, &raw, &scale, rows, cols, &lut)?,
};
drop(raw);
drop(scale);

let quantized = mlxcel_core::quantize_weights_with_mode(
    &restored, MXFP8_GROUP_SIZE, MXFP8_BITS, MXFP8_MODE,
);
mlxcel_core::eval(&packed);
mlxcel_core::eval(&scales);
weights.insert(weight_key, packed);
weights.insert(format!("{prefix}.scales"), scales);
weights.remove(&format!("{prefix}{CT_FP8_INPUT_SCALE_SUFFIX}"));
```

**File: `src/models/fp8_checkpoint.rs`**

The route selector keeps the config merge and weight pass on the same branch:

```rust
pub enum Fp8Checkpoint {
    VendorBlock(Fp8BlockQuantization),
    CompressedTensors(Fp8CtQuantization),
}

pub fn fp8_checkpoint_quantization(full_config: &Value) -> Result<Option<Fp8Checkpoint>, String> {
    if let Some(detected) = qwen_fp8_block_quantization(full_config)? {
        return Ok(Some(Fp8Checkpoint::VendorBlock(detected)));
    }
    Ok(compressed_tensors_fp8_quantization(full_config)?.map(Fp8Checkpoint::CompressedTensors))
}
```

**File: `src/models/fp8_block.rs`**

Block reconstruction takes independent extents and shares the decode:

```rust
pub fn reconstruct_block_fp8_tensor(
    name: &str, raw: &MlxArray, scale: &MlxArray,
    block_rows: usize, block_cols: usize, lut: &MlxArray,
) -> Result<UniquePtr<MlxArray>, String> {
    let (decoded, rows, cols) = decode_fp8_bytes_to_f32(name, raw, lut)?;
    let row_blocks = (rows + block_rows_i32 - 1) / block_rows_i32;
    let col_blocks = (cols + block_cols_i32 - 1) / block_cols_i32;
    // pad to whole [row_blocks * block_rows, col_blocks * block_cols], reshape to
    // [row_blocks, block_rows, col_blocks, block_cols], apply the scale, slice back.
}
```

**Reason for change:** one reconstruction implementation serves both FP8 layouts, so the padding and slicing logic is tested once.

**Files: `src/models/qwen3_5.rs`, `src/loading/vlm_qwen.rs`, `src/loading/special.rs`**

Each loader replaces its vendor-only detection and its `if let` on the converted weights with the route selector and the two shared calls. The call sites shrink; behavior for vendor checkpoints is unchanged.

**File: `docs/supported-models.md`**

Documents the second FP8 layout, its scale shapes, the `.weight_scale_2` disambiguation against ModelOpt NVFP4, and the validation result below.

### 4.3 Data Model Changes

None. The weight map keys change only for converted tensors: `<name>.weight` becomes a packed `mxfp8` plane, `<name>.scales` is added, and `<name>.weight_scale` plus `<name>.input_scale` are removed. That is the same shape the vendor route already produces.

---

## 5. Learning Points

### 5.1 Sibling quantization layouts with disjoint key signatures

**Concept:**
Two quantization methods can store structurally similar data (E4M3 bytes plus a scale) and still pair the scale differently. A loader must decide per checkpoint which pairing applies, and it must be able to prove the two routes are disjoint.

**Application in this PR:**
The vendor route uses `.weight_scale_inv` per 128x128 block. The compressed-tensors route uses `.weight_scale` per channel or per `block_structure` tile. ModelOpt NVFP4 also uses `.weight_scale`, but always with `.weight_scale_2`. Requiring `.weight_scale_2` to be absent makes the compressed-tensors route disjoint from ModelOpt by construction, and `quant_method` makes it disjoint from the vendor route.

**Common Use Cases:**
- Supporting multiple vendors' quantization output without a naming registry
- Adding a new checkpoint format beside an existing one with a shared target representation

**Example Code:**
```rust
weights.keys().any(|key| {
    key.ends_with(".weight_scale") && !weights.contains_key(&format!("{key}_2"))
})
```

### 5.2 Bounded-memory multi-step tensor conversion

**Concept:**
Reconstruction, scaling, and requantization each allocate an array at least the size of the tensor. Holding all intermediates for every tensor at once multiplies peak memory by the number of stages.

**Application in this PR:**
The pass processes one tensor at a time, removes the inputs from the map before converting, and calls `eval` on each output pair before moving on. The comment in the code states the intent: the f32 reconstruction of the largest projection is the only transient, and it is released before the next is built.

**Common Use Cases:**
- Loading and converting large checkpoints on unified-memory hosts
- Any pipeline that dequantizes, transforms, and requantizes

**Example Code:**
```rust
let restored = reconstruct(...)?;
drop(raw);
drop(scale);
let quantized = quantize_weights_with_mode(&restored, ...);
mlxcel_core::eval(&packed);
mlxcel_core::eval(&scales);
```

---

## 6. Further Learning

### Key Terms

| Keyword | Description | Relevance |
|---------|-------------|-----------|
| `compressed-tensors` | Safetensors quantization metadata format used by llm-compressor | Names the layout this PR adds |
| `float-quantized` | compressed-tensors format tag for float8 weights with a separate scale | The detection gate |
| `block_structure` | Per-group `[rows, cols]` tile a single scale covers | Drives the block reconstruction path |
| `mxfp8` | MLX native 8-bit float with one shared E8M0 exponent per 32 values | The requantization target for both FP8 routes |
| E4M3 | 8-bit float format used by FP8 checkpoints | The raw byte encoding being decoded |
| `.weight_scale_2` | Second-level scale present in ModelOpt NVFP4 triplets | Disambiguates the compressed-tensors route |

### Related Technologies/Frameworks
- **MLX**: Apple's array framework; `quantize_weights_with_mode` and `take` are used for the device-side conversion. Official docs: https://ml-explore.github.io/mlx/
- **llm-compressor**: produces the `FP8-dynamic` checkpoints. https://github.com/vllm-project/llm-compressor
- **compressed-tensors**: the metadata schema. https://github.com/neuralmagic/compressed-tensors
- **mlx-vlm**: upstream reference that dequantizes to bf16, in `mlx_vlm/utils.py` (`dequantize_fp8_weight`). https://github.com/Blaizzy/mlx-vlm

### Related PRs/Issues
- Issue #2411: the feature request this PR closes; lists the acceptance criteria including the logit trace and the scale-ignored arm.
- PR #1742: added the vendor fine-grained FP8 block path that this PR extends.
- PR #1772: bumped the MLX pin so mxfp8 block scales round up, which the shared error bound depends on.
- PR #2071: proved mxfp8 end to end on ROCm, including FP8 block checkpoints.

---

## 7. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 10 |
| Lines added | +1524 |
| Lines deleted | -79 |
| New tests | 9 in `src/models/fp8_ct_tests.rs` |
| New modules | 2 (`fp8_ct`, `fp8_checkpoint`) |
| Test command | `cargo test --profile test-fast -p mlxcel --lib fp8` (29 passed) |

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| Feature | 3 | Detection and requantization, route selection, loader wiring |
| Refactor | 2 | Generalized block reconstruction and shared E4M3 decode; shared config merge |
| Tests | 2 | New compressed-tensors suite; updated vendor block call site |
| Documentation | 1 | `docs/supported-models.md` entry for the second layout |
| Module registration | 1 | `src/models/mod.rs` exports |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `68bca8f3` | feat | load compressed-tensors float-quantized FP8 checkpoints |

---

## 8. Follow-up Actions

### Required

- [ ] None blocking. The PR is open and awaiting review.
- [ ] Confirm merge CI, particularly the full workspace test suite beyond the `fp8` filter.

### Monitoring Required

- Watch load time and peak memory for compressed-tensors FP8 checkpoints other than the 4B reference. This PR measured one checkpoint on one host.
- Watch for compressed-tensors releases that use `strategy: "tensor"` or `"group"` or a different bit width: they fail deliberately and would need a new pairing.

### Future Improvements

- Extend the route to other model families. Only the Qwen3.5 text, VLM, and special loaders call `fp8_checkpoint_quantization`.
- Consider a `mixed-precision` path, which currently returns `None` and falls through to dense loading. That is correct only if the affected tensors load as bf16.
- Measure the conversion on CUDA and ROCm. The logit trace and load figures are Metal/M3 Ultra only.

---

## Appendix

### A. Test Results

| Check | Result |
|-------|--------|
| `cargo test --profile test-fast -p mlxcel --lib fp8` | 29 passed (detection, channel and block conversion, NVFP4-triplet non-interference) |
| `cargo clippy --lib --tests -- -D warnings` | Clean |
| `cargo fmt --check` | Clean |

Covered by the new suite: config detection including nested `text_config`, block and channel layouts, the `[out, 1]` scale spelling, partial-block padding on both axes, NVFP4 triplet isolation, pre-decoded bf16 weights, odd input width rejection, mismatched channel-scale rejection, and identity when no scales are present.

### B. Logit Trace

Teacher-forced over 256 positions of `docs/architecture.md` with `examples/logit_trace` (4 tokens per chunk, 64 chunks, topk 8, prefill 512), comparing `RedHatAI/Qwen3.5-4B-FP8-dynamic` against `Qwen/Qwen3.5-4B` with `scripts/compare_logit_traces.py --decided 2`:

| Metric | Value |
|--------|-------|
| Top-1 disagreement | 9/256 (3.516%) |
| Disagreement on decided positions (reference top-two gap >= 2.0) | 0/116 (0.000%) |
| Rank of the disagreements | 2 to 3, all at reference gaps below 0.5 |
| Candidate perplexity | 16.5992 |
| Reference perplexity | 16.7090 |

The scale-ignored arm (raw bytes decoded, per-channel scale not applied) does not emit a trace at all: `chunk 0 position 0 produced a non-finite NLL`. The checkpoint's `weight_scale` values span 1.4e-05 to 1.54e-03, so omitting them scales each converted row by 649x to 71392x.

End-to-end check: `mlxcel generate -m RedHatAI/Qwen3.5-4B-FP8-dynamic -p "The capital of France is" -n 300 --temp 0` answers "The capital of France is **Paris**." with 7.75 GB peak and a 1.8 s load.

### C. References

- Issue #2411, "Load compressed-tensors float-quantized FP8 checkpoints"
- `src/models/fp8_ct.rs` module documentation, which records why bf16 dequantization was rejected
- `src/models/fp8_block.rs` module documentation, which records the mxfp8 error bound and the ROCm tie-rounding caveat
- mlx-vlm `dequantize_fp8_weight`: the upstream bf16 reference implementation
