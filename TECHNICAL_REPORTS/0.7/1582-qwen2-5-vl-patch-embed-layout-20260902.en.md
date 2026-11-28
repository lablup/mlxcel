# Technical Report: PR #1582 - fix(qwen2_5_vl): normalize the raw HF patch-embed layout at load

**Date**: 2026-09-02
**Author**: mlxcel maintainers
**Reviewer**: implementation review cycle
**Status**: Completed (unit level verified locally; the shipped-binary gates are listed in the PR and remain open)
**Languages**: Rust
**Risk Level**: Medium (touches the load path of three Qwen-VL consumers; the conversion path is a proven no-op)

---

## Executive Summary

A raw HuggingFace export of Qwen2.5-VL could not be served by `mlxcel generate` or `mlxcel-server`. The vision tower is stored under `visual.*` rather than `vision_tower.*`, so the loader raised `Missing vision_tower.patch_embed.proj.weight`, and even with the names corrected the patch-embedding filter is in `Conv3d`'s native `[out, in, kT, kH, kW]` layout while the encoder accepts only the mlx-vlm conversion's channels-last `[out, kT, kH, kW, in]`. The two have the same element count, so the reshape downstream succeeds and the tower reads scrambled filters without any error. PR #1414 fixed the layout half on the ColQwen2.5 retrieval path only. This PR moves that one normalizer into the encoder module whose contract it protects, parameterizes it by weight prefix, calls it from both consumers, and adds the missing key remap to the generation loader.

---

## 1. Problem Statement

### 1.1 Background

`Qwen25VLVisionEncoder::from_weights` builds its patch embedding through `PatchEmbed::from_weights` (`src/vision/encoders/qwen2_5_vl.rs`), which accepts a 5-D weight, permutes it with `[0, 1, 4, 2, 3]`, and reshapes to `[out, in * kT * kH * kW]`. That permutation is only correct for the mlx-vlm conversion's channels-last layout. `transformers` writes the PyTorch `Conv3d` parameter directly, which puts `in_channels` on axis 1.

PR #1414 discovered this while porting ColQwen2.5, whose published checkpoint (`vidore/colqwen2.5-base`) is a raw export. It added `normalize_patch_embed_layout` to `src/models/colqwen2_5.rs`, hardcoded to that module's `VISION_PREFIX`. The measured cost of the unfixed path was a retrieval misranking: the unrelated page scored MaxSim 7.83 against the matching page's 8.04.

The generation loader `load_qwen2_5_vl` (`src/loading/vlm_qwen.rs`) never received either half of that fix. It ran `strip_language_model_prefix`, which strips only a leading `language_model.`, then handed the map straight to the encoder under the prefix `vision_tower`.

### 1.2 Existing Issues

- **A raw export could not load at all.** The `Qwen/Qwen2.5-VL-3B-Instruct` snapshot on this host carries exactly two top-level prefixes, `model` and `visual`. Nothing named `vision_tower.*` exists, so `PatchEmbed::from_weights` failed with `Missing vision_tower.patch_embed.proj.weight` before any layout question could arise.
- **Fixing only the names would have produced a silent wrong answer.** A hand-renamed export, or the remap on its own, gets past the missing-key error and then reads `[1280, 3, 2, 14, 14]` as if it were `[1280, 2, 14, 14, 3]`. The reshape succeeds because both hold 1505280 elements. Nothing throws, and the model still emits fluent text.
- **The fix already existed but was scoped to one caller.** `normalize_patch_embed_layout` sat in the ColQwen2.5 module as `pub(crate)` with the prefix baked in, so the generation loader could not reuse it without duplicating the detection rule.
- **The naming precedent was already in the same file.** `load_qwen3_vl` performs exactly this class of remap for its own family, so the Qwen2.5-VL loader was the outlier rather than the norm.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Raw HuggingFace Qwen2.5-VL checkpoints unusable for generation and serving | High | Certain before this change |
| A renamed raw checkpoint generating from scrambled vision filters with no error | High | High, and undetectable from output fluency |
| The detection rule drifting between two copies once a second caller needs it | Medium | Medium |

---

## 2. Technical Review

### 2.1 Root cause

Two independent gaps on the same load path, both invisible from the model's output.

The first is naming. `strip_language_model_prefix` handles the mlx-vlm shape (`language_model.model.*` becomes `model.*`) and nothing else. A `transformers` export uses `visual.*` with a bare `model.*` decoder (older exports) or `model.visual.*` with `model.language_model.*` (newer ones). Neither is reachable from that one strip.

The second is layout. The channel axis is what separates the two 5-D shapes: the raw layout has `in_channels` on axis 1 and a spatial extent (`patch_size`) on axis 4, and the converted layout has `in_channels` on axis 4. `in_channels` is 3 and `patch_size` is 14 for every published Qwen2.5-VL, so the test is unambiguous in practice, but it is a shape heuristic and the code should say so.

### 2.2 Compatibility & Dependencies

- **Breaking changes**: none. Both new steps are no-ops on an mlx-community conversion, whose keys are already `vision_tower.*` and `language_model.*` and whose filter is already `[1280, 2, 14, 14, 3]`.
- **New dependencies**: none.
- **Encoder contract**: unchanged. `PatchEmbed::from_weights` still accepts exactly one 5-D layout. This matters because Qwen2-VL and ColQwen2.5 consume the same code and rely on that.
- **Quantized checkpoints**: `patch_embed.proj.weight` stays float in a 4-bit conversion, so the same detection and transpose apply to it unchanged.

### 2.3 Code Quality

- The normalizer now lives beside the code whose invariant it enforces, and carries the repository's `Used by:` line naming its two callers.
- Its detection rule and its tests exist in one place instead of two.
- Test coverage grew from one test (in the ColQwen2.5 module) to four in the encoder module plus two loader-remap tests and one real-checkpoint gate.
- `src/loading/vlm_qwen.rs` gained a `#[path]` test file (`vlm_qwen_tests.rs`) rather than another inline `mod`, which keeps an already long file from growing further.

---

## 3. Technical Decisions

### 3.1 Where the layout normalization belongs

**Context:** `PatchEmbed::from_weights` could detect the layout itself and accept both, which would fix every consumer at once with no loader changes.

**Alternatives considered:**

| Option | Pros | Cons |
|--------|------|------|
| Detect inside `PatchEmbed::from_weights` | One place, fixes Qwen2-VL and ColQwen2.5 automatically | Silently accepts two layouts in a module whose three consumers all assume one; a future checkpoint with a genuinely wrong filter would be reshaped rather than rejected |
| Duplicate the ColQwen2.5 helper into the loader | Smallest diff | Two copies of a shape heuristic that must stay identical |
| **Chosen: move the helper into the encoder module, parameterized by prefix, called from both loaders** | The rule lives with the contract it guards, tested once, and the encoder's accepted layout stays a single documented shape | Every new consumer must remember to call it |

**Rationale:** the loader is the layer that has `in_channels` on hand and the only layer that knows which flavour of checkpoint it opened. Keeping the encoder strict means a genuinely malformed filter still fails loudly there.

**Trade-offs:** a future Qwen2.5-VL-family loader that forgets the call gets the pre-fix silent-corruption behaviour. The `Used by:` line and the shared location are the mitigation; a hard guard inside `PatchEmbed` would trade that for the ambiguity this decision rejects.

### 3.2 What to do when the two layouts are indistinguishable

**Context:** detection is by the channel axis. If `in_channels` equalled `patch_size`, both axis 1 and axis 4 would carry `in_channels` and shape alone could not tell the layouts apart.

The original condition (`shape.len() != 5 || shape[1] != channels || shape[4] == channels`) already fell through to "leave it alone" in that case, but only as a side effect of ordering. The rewritten form names `leading_is_channel` and `trailing_is_channel` and documents the refusal to guess as deliberate. Behaviour is identical; the intent is now readable, which is what makes the next reader trust the fall-through instead of "fixing" it.

---

## 4. Implementation Details

### 4.1 Key code changes

**File: `src/vision/encoders/qwen2_5_vl.rs`**

```rust
pub(crate) fn normalize_patch_embed_layout(
    weights: &mut WeightMap,
    prefix: &str,
    in_channels: usize,
) -> bool {
    let key = format!("{prefix}.patch_embed.proj.weight");
    let channels = in_channels as i32;
    let converted = {
        let Some(weight) = weights.get(&key) else {
            return false;
        };
        let shape = mlxcel_core::array_shape(weight);
        if shape.len() != 5 {
            return false;
        }
        let leading_is_channel = shape[1] == channels;
        let trailing_is_channel = shape[4] == channels;
        if !leading_is_channel || trailing_is_channel {
            return false;
        }
        // [out, in, kT, kH, kW] -> [out, kT, kH, kW, in].
        mlxcel_core::transpose_axes(weight, &[0, 2, 3, 4, 1])
    };
    weights.insert(key, converted);
    true
}
```

**File: `src/loading/vlm_qwen.rs`**

```rust
fn rewrite_qwen2_5_vl_native_key(key: &str) -> String {
    if let Some(rest) = key.strip_prefix("model.visual.") {
        format!("vision_tower.{rest}")
    } else if let Some(rest) = key.strip_prefix("model.language_model.") {
        format!("model.{rest}")
    } else if let Some(rest) = key.strip_prefix("visual.") {
        format!("vision_tower.{rest}")
    } else {
        key.to_string()
    }
}
```

and, inside `load_qwen2_5_vl`:

```rust
let mut weights = remap_qwen2_5_vl_native_keys(strip_language_model_prefix(
    load_vlm_weights_common(model_path, None)?,
));
models::sanitize_tied_embeddings(&mut weights, &full_config);

if normalize_patch_embed_layout(&mut weights, "vision_tower", vision_config.in_channels) {
    tracing::debug!(
        "Qwen2.5-VL: converted patch_embed.proj.weight from the PyTorch Conv3d layout"
    );
}
```

**File: `src/models/colqwen2_5.rs`**

The local copy is deleted; the call becomes `normalize_patch_embed_layout(&mut weights, VISION_PREFIX, vision_config.in_channels)`.

### 4.2 Ordering

The remap runs after `strip_language_model_prefix`, not before. That ordering matters for the mlx-vlm shape: `language_model.model.layers.*` has to become `model.layers.*` first, or the `model.language_model.` rule would never see it, and a checkpoint carrying `language_model.visual.*` (none is known) would be handled the same way. The normalizer runs after both, so it sees the final `vision_tower.` prefix regardless of which flavour the checkpoint used.

---

## 5. Validation

The permutation is checked against the real converter, not only against itself. Reading the bf16 `visual.patch_embed.proj.weight` from the local `Qwen/Qwen2.5-VL-3B-Instruct` snapshot (`[1280, 3, 2, 14, 14]`), transposing it with axes `[0, 2, 3, 4, 1]`, and comparing 20000 randomly sampled elements at f16 resolution against `mlx-community/Qwen2.5-VL-3B-Instruct-4bit`'s `vision_tower.patch_embed.proj.weight` (`[1280, 2, 14, 14, 3]`, f16) gives 20000/20000 exact matches, max absolute difference 0.0. Reading the same buffer without permuting, which is what the pre-fix loader effectively did, is off by up to 1.09e-01. That is the numerical statement the issue was making, measured rather than assumed.

Test commands run on this branch:

```
cargo fmt --all -- --check
cargo clippy --profile test-fast --lib --tests --features metal,accelerate -- -D warnings
cargo test --profile test-fast --features metal,accelerate --lib vision::encoders::qwen2_5_vl   # 4 passed
cargo test --profile test-fast --features metal,accelerate --lib loading::vlm::qwen             # 6 passed, 1 ignored
cargo test --profile test-fast --features metal,accelerate --lib models::colqwen2_5             # 6 passed
```

The `#[ignore]`d gate `qwen2_5_vl_raw_export_matches_mlx_conversion` normally looks up `Qwen/Qwen2.5-VL-3B-Instruct` and `mlx-community/Qwen2.5-VL-3B-Instruct-bf16` through the usual store lookup and soft-skips when either is absent. The bf16 conversion is not present on this host, so the gate was instead pointed at two locally present raw 3B exports through its `MLXCEL_TEST_QWEN25VL_RAW_DIR` / `MLXCEL_TEST_QWEN25VL_MLX_DIR` overrides. Both loaded through the generation path and produced identical, non-empty greedy output in 7.6 seconds. That does not test the raw-versus-converted differential, but it does prove the thing the pre-fix loader could not do at all: open a raw `transformers` export and generate from it.

### Not verified here

Three checks belong to the shipped binary and are listed as open boxes in the PR body: a coherent image description from `mlxcel generate -m <raw dir> --image ...`; byte-identical output from the 4-bit conversion before and after this change; and the ColQwen2.5 real-checkpoint gate with its checkpoint present. The differential gate against `mlx-community/Qwen2.5-VL-3B-Instruct-bf16` also remains unrun for lack of that checkpoint.

---

## 6. Change Summary

### Statistics

| Metric | Value |
|--------|-------|
| Files changed | 6 |
| Lines added | 520 |
| Lines removed | 84 |
| New test files | 2 |
| New tests | 7 (6 unit, 1 ignored real-checkpoint gate) |

### Changes by Category

| Category | Files |
|----------|-------|
| Shared normalizer relocated and parameterized | `src/vision/encoders/qwen2_5_vl.rs`, `src/models/colqwen2_5.rs` |
| Generation loader fix | `src/loading/vlm_qwen.rs` |
| Tests | `src/vision/encoders/qwen2_5_vl_tests.rs`, `src/loading/vlm_qwen_tests.rs`, `src/models/colqwen2_5_tests.rs` |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `0ea8233f6` | fix(qwen2_5_vl) | normalize the raw HF patch-embed layout at load |

### Related PRs/Issues

- Issue #1423: the issue this PR closes.
- PR #1414: added the original ColQwen2.5-scoped normalizer and measured the 7.83 against 8.04 misranking that motivated it.
- Issue #1367: vision-stripped Qwen2.5-VL checkpoints, which edits the same `load_qwen2_5_vl` body and is expected to land after this.
- Epic #1348: the parent epic this follow-up belongs to.

---

## 7. Follow-up Actions

### Broader lesson

The interesting property of this defect class is that both halves fail differently. The missing key remap fails loudly, at load, and is trivially diagnosed. The layout mismatch fails silently, at inference, and produces fluent output from scrambled filters, so no smoke test built around "does it say something sensible" catches it. What separated them was a numerical comparison against an independent reference, which is the same method PR #1414 used to find it in the first place and the same method used here to prove the permutation.

The practical rule is narrower than "always diff against a reference". It is that when a tensor's element count is invariant under the layout confusion, shape checks and reshape success carry no information at all, and only values do. `[1280, 3, 2, 14, 14]` and `[1280, 2, 14, 14, 3]` both hold 1505280 elements, which is exactly why the pre-fix path never raised anything.
