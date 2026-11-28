# Technical Report: PR #1174 - feat(models): qualify Qwen3.8-27B on the qwen3_5 path

**Date**: 2026-08-16
**Author**: AI Code Reviewer
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low

---

## Executive Summary

`mlx-community/Qwen3.8-27B-4bit` declares `model_type: "qwen3_5"` and is architecturally identical to Qwen3.5-27B down to a byte-identical weight-map key set, so it already loaded and ran on `main` with zero code changes before this PR. That was accidental rather than guaranteed: `Qwen35Config` carries no `deny_unknown_fields`, so every config key the Qwen3.8 generation added was silently dropped, three of which are load-bearing upstream (`output_gate_type`, `rope_parameters.mrope_interleaved`, and the top-level `language_model_only`). The PR converts those silent drops into explicit reads or named errors across five of the six Qwen3.5-family config parse sites, requires `vision_start_token_id` from the config instead of falling back to a stale default, and registers the checkpoint in the docs and the real-model test suite. The issue also asked to port two post-pin upstream mlx-vlm fixes; both were verified already present on `main` before any code was written, so the deliverable for that half of the issue became confirm-and-pin rather than re-implement. Finalization closed five code findings the review pass raised against the initial PR (a sixth skipped parse site, two type-confusion bypasses in the new guards, an unchecked truncating cast on vision token ids, and case-sensitive activation matching), corrected three documentation findings, and added the `CHANGELOG.md` entry with its required upgrade note.

---

## 1. Problem Statement

### 1.1 Background

Issue #1163 asked for two things: qualify `Qwen3.8-27B-4bit` on the existing `qwen3_5` path, and port two upstream mlx-vlm fixes that landed after mlxcel's MLX pin (Blaizzy/mlx-vlm#1805, a padded-vocabulary structured-output mask fix, and Blaizzy/mlx-vlm#1741, a chunked-MRoPE position-slice reuse fix). Both ports turned out to be premise corrections rather than new work: `apply_structured_mask_to_logits` in `src/server/structured.rs` already pinned the bias length to the model's logits axis rather than the matcher's vocabulary, and `Qwen35Model::forward_with_mrope_state` already sliced the stored `position_ids` tensor to the chunked-prefill window and validated the batch dimension before reuse. Per the project's performance-issue and premise-correction conventions, the deliverable for that half of the issue became confirming and pinning the existing behavior with tests, not re-implementing it.

### 1.2 Existing Issues (from the review/security pass, addressed in finalization)

- **S-M1** (Medium): `validate_supported()` was wired into five of the six `Qwen35Config` parse sites but deliberately skipped the sixth, `src/loading/vlm_special.rs`'s MiniCPM-V 4.6 text-backbone loader, on the reasoning that the invariant is scoped to the `qwen3_5` model_type string. That reasoning does not hold: the invariant (hardcoded SiLU gated-delta output, unconditional interleaved MRoPE) belongs to `Qwen35Model`, which the MiniCPM-V 4.6 loader also builds via `from_weights`, so a MiniCPM-V 4.6 checkpoint declaring `output_gate_type: "sigmoid"` reproduced the exact silently-wrong-output failure the PR set out to close.
- **S-M2** (Medium): `validate_qwen35_wrapper_config` and `Qwen35Config::mrope_interleaved()` both read their target key with `.as_bool()`, so a present-but-wrong-typed value silently read as absent instead of hitting the named error the correctly-typed value would trigger. Verified on a release binary: `language_model_only: "true"`, `1`, and `{}`, plus `mrope_interleaved: "false"` and `0`, all passed validation and loaded.
- **S-M3** (Medium): `qwen35_vl_token_ids` converted `vision_start_token_id` from `i64` to `i32` with an unchecked `as i32`, a truncating cast. Verified: `vision_start_token_id: 1099511627776` (2^40) truncated to 0 and `vision_start_token_id: -5` loaded unchanged, both with no diagnostic, reproducing exactly the silent-mis-segmentation failure the function's own error text names as the reason the key is mandatory. `image_token_id` and `video_token_id` had the same unchecked cast.
- **R-M2** (Medium): `output_gate_type` matching was exact and case-sensitive against `{"silu", "swish"}`. A checkpoint spelling it `"SiLU"` would hard-fail a load the implemented path already handles correctly.
- **S-L2** (Low): The `OutputGateType` error echoed the offending config value with no length cap. A 1,002,990-byte `output_gate_type` value reproduced as 1,000,568 bytes of process output.
- **R-M1** (doc): `docs/supported-models.md` listed Qwen3.5-VL and Qwen3.8-VL in two consecutive bullets, the family summary line and the new detail bullet, while every other family in that list appears once.
- **R-L1** (doc): The KV-footprint sentence claimed the measured 64.1 KiB/token figure was "exactly" the 65,536 B/token (64.0 KiB) architectural minimum; the two differ by about 0.16%.
- **R-L2** (doc): The weight-map key list in the same paragraph (`model.language_model.*`, `model.visual.*`, `mtp.*`, `lm_head.weight`) described the upstream `Qwen/Qwen3.8-27B` repo, not the `mlx-community/Qwen3.8-27B-4bit` conversion named as validated in the same sentence.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| MiniCPM-V 4.6 loads with the wrong gated-delta activation or MRoPE layout (S-M1) | Medium (silently wrong generation output, no error) | Real for any future checkpoint declaring these keys; zero for every checkpoint verified locally |
| A wrong-typed `language_model_only`/`mrope_interleaved` value loads instead of failing (S-M2) | Medium (defeats the guard the parent PR added) | Confirmed reproducible pre-fix on a release binary |
| An out-of-range or overflowing vision token id mis-segments MRoPE vision spans (S-M3) | Medium (silently wrong VLM output) | Confirmed reproducible pre-fix on a release binary |
| A validly-cased `output_gate_type` alias hard-fails a load the code already supports (R-M2) | Low-Medium (false-positive startup failure) | Real if a checkpoint or conversion tool ever varies casing |
| A malformed config produces megabyte-scale error output (S-L2) | Low (operational nuisance, not correctness) | Low but directly reproduced with a synthetic oversized value |
| Doc inaccuracies (R-M1, R-L1, R-L2) | Low (reader confusion, no functional impact) | N/A, corrected |

---

## 2. Technical Review

### 2.1 Security

Review and security passes were completed by the orchestrator and reviewer before finalization: no CRITICAL, no HIGH findings. This pass closed the five MEDIUM/LOW findings above.

**Issues Found:**

| Issue | Severity | Status |
|-------|----------|--------|
| Sixth `Qwen35Config` parse site (MiniCPM-V 4.6) missing `validate_supported()` | Medium | Fixed (`4e2fab1e`) |
| `language_model_only`/`mrope_interleaved` wrong-typed values silently treated as absent | Medium | Fixed (`4e2fab1e`) |
| `vision_start_token_id`/`image_token_id`/`video_token_id` unchecked truncating cast, no range check | Medium | Fixed (`4e2fab1e`) |
| `output_gate_type` matching case-sensitive | Medium | Fixed (`4e2fab1e`) |
| Unbounded config value echoed into `OutputGateType` error | Low | Fixed (`4e2fab1e`) |

Before wiring S-M1's `validate_supported()` call into the MiniCPM-V 4.6 loader, both local MiniCPM-V-4.6 checkpoints (`minicpm-v-4.6-bf16` and `minicpm-v-4.6-mxfp4`) were checked: neither's `text_config` declares `output_gate_type` or `rope_parameters.mrope_interleaved`, so the new call does not turn either working load into a startup failure. The published `text_config` shape (24 layers, `hidden_size` 1024, `vocab_size` 248094) is pinned as a regression fixture in `src/models/qwen3_5_tests.rs`.

### 2.2 Performance

None. Every fix in this pass is either a load-time config validation (runs once per process start, before any weight is read for S-M1's specific insertion point) or a documentation correction. No benchmark was required or run.

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: none beyond what the parent PR already introduced. S-M1's new `validate_supported()` call at the MiniCPM-V 4.6 site is confirmed safe against both local checkpoints (see 2.1). S-M2 and S-M3 turn previously-silent-wrong-output cases into load failures; every known checkpoint in the family supplies correctly-typed, in-range values, so no currently-working deployment is affected. R-M2 is strictly permissive (accepts more spellings than before, rejects nothing that previously loaded).
- **New Dependencies**: none.
- **Compatibility**: `qwen35_vl_token_ids`'s signature gained a `vocab_size: usize` parameter; its one production call site (`src/loading/vlm_qwen.rs`) passes `text_config.vocab_size`, which was already parsed and in scope at that point. All test call sites were updated to match.

### 2.4 Code Quality

- **Test Coverage**: 18 new tests across three files. `src/models/qwen3_5_tests.rs` gained `output_gate_type_matching_is_case_insensitive`, `output_gate_type_error_truncates_an_oversized_value`, `mrope_interleaved_wrong_type_is_a_named_error_not_absent`, `language_model_only_wrong_type_is_a_named_error_not_absent`, and three MiniCPM-V-4.6-shaped tests (`minicpmv4_6_text_config_passes_validate_supported`, `..._output_gate_type_sigmoid_is_a_named_error`, `..._mrope_interleaved_false_is_a_named_error`). `src/loading/vlm_tests.rs` gained four range/overflow tests for `qwen35_vl_token_ids`. `src/loading/vlm_special_tests.rs` gained `load_minicpmv4_6_vlm_rejects_an_unsupported_output_gate_type`, which drives the real `load_minicpmv4_6_vlm` entry point end to end (a fabricated `config.json` with no accompanying weight files is enough, since `validate_supported()` now runs before any weight is read) and fails without the S-M1 fix, since the loader would instead fail later and differently once it reaches the `vision_config` this fixture deliberately omits.
- **Code Complexity**: each fix is localized. S-M2 adds one private strict accessor (`mrope_interleaved_checked`) alongside the existing permissive one, rather than changing the public accessor's signature and breaking its other callers. S-M3 factors the range/overflow check into one small helper (`qwen35_vl_token_id_in_range`) shared by all three ids.
- **Technical Debt**: decreased. The sixth parse site no longer diverges from the other five without a documented reason, and the two type-confusion bypasses no longer make S-M2's own parent-PR guards inconsistent with the typed `output_gate_type` field that was already strict.

---

## 3. Technical Decisions

### 3.1 Keep `mrope_interleaved()` permissive, add a separate strict accessor

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Change `mrope_interleaved()`'s return type to `Result<Option<bool>, Qwen35UnsupportedConfig>` | One accessor, no duplication | Breaks its other callers and the existing `Option<bool>`-shaped test assertions; the permissive true/absent read is a legitimate use case for callers that do not need strict validation |
| **Chosen: add a private `mrope_interleaved_checked()` used only by `validate_supported`** | No signature break; the strict/permissive split is explicit in the type | Two functions reading the same JSON path |

**Rationale:** the public accessor's `Option<bool>` shape is what other code (and the existing test suite) already depends on for the common true/absent case. The strict variant is needed by exactly one caller, `validate_supported`, so scoping it there avoids a signature change with a wider blast radius than the finding requires.

### 3.2 Bound vision token ids against the checkpoint's own `vocab_size`, not a fixed constant

**Rationale:** the finding named both an overflow case (2^40 truncating to 0 under the old cast) and an in-range-but-nonsensical case (a value that fits in `i32` but exceeds the checkpoint's vocabulary). A fixed upper bound would not catch the second case for a differently-sized checkpoint in the family. Threading `text_config.vocab_size` through from the one production call site, which already has it parsed, closes both without hardcoding a family-wide constant that could go stale the way the removed 248045 default already had.

### 3.3 Truncate the echoed config value with a shared helper, not per-error-variant

**Rationale:** `S-L2` was raised against `OutputGateType` specifically, but the same unbounded-echo shape exists in the two new wrong-type errors this pass adds (S-M2). A single `truncate_for_error` helper with a shared `MAX_ERROR_VALUE_CHARS` constant closes the finding at its root instead of only at the one call site the finding happened to name.

---

## 4. Implementation Details

### 4.1 S-M1: sixth `validate_supported()` call site (`src/loading/vlm_special.rs`)

```rust
let text_config: models::qwen3_5::Qwen35Config = serde_json::from_value(text_config_value)
    .map_err(|e| anyhow::anyhow!("Failed to parse MiniCPM-V 4.6 text config: {}", e))?;
// The gated-delta / MRoPE invariants `validate_supported` enforces belong
// to `Qwen35Model`, which this loader builds via `from_weights` below,
// not to the `qwen3_5` model_type string. ...
text_config.validate_supported()?;
```

Placed immediately after `text_config` is parsed and before `vision_config` is parsed or any weight is read, so the rejection path (and its test) needs no vision config or weight files at all.

### 4.2 S-M2: strict wrong-type detection (`src/models/qwen3_5.rs`)

```rust
fn mrope_interleaved_checked(&self) -> Result<Option<bool>, Qwen35UnsupportedConfig> {
    match self.rope_parameters.as_ref().and_then(|rp| rp.get("mrope_interleaved")) {
        None => Ok(None),
        Some(value) => value.as_bool().map(Some).ok_or_else(|| {
            Qwen35UnsupportedConfig::MropeInterleavedWrongType(truncate_for_error(
                &value.to_string(),
                MAX_ERROR_VALUE_CHARS,
            ))
        }),
    }
}
```

`validate_qwen35_wrapper_config` gained the mirrored `match value.as_bool() { Some(true) => ..., Some(false) => {}, None => ... }` shape for `language_model_only`. Two new `Qwen35UnsupportedConfig` variants, `MropeInterleavedWrongType` and `LanguageModelOnlyWrongType`, carry the truncated offending value.

### 4.3 S-M3: bounds-checked vision token ids (`src/loading/vlm.rs`)

```rust
fn qwen35_vl_token_id_in_range(field: &str, raw: i64, vocab_size: usize) -> anyhow::Result<i32> {
    let id = i32::try_from(raw).map_err(|_| anyhow::anyhow!(
        "Qwen3.5-family config.json has `{field}={raw}`, which does not fit in a 32-bit token id. ..."
    ))?;
    if id < 0 || (id as usize) >= vocab_size {
        anyhow::bail!(
            "Qwen3.5-family config.json has `{field}={id}`, which is outside the checkpoint's \
             vocabulary (vocab_size={vocab_size}). ..."
        );
    }
    Ok(id)
}
```

`qwen35_vl_token_ids` gained a `vocab_size: usize` parameter and routes all three ids through this helper; its one call site in `src/loading/vlm_qwen.rs` passes `text_config.vocab_size`.

### 4.4 R-M2: case-insensitive gate matching (`src/models/qwen3_5.rs`)

```rust
if !gate.eq_ignore_ascii_case("silu") && !gate.eq_ignore_ascii_case("swish") {
    return Err(Qwen35UnsupportedConfig::OutputGateType(...));
}
```

### 4.5 S-L2: bounded error echo (`src/models/qwen3_5.rs`)

```rust
const MAX_ERROR_VALUE_CHARS: usize = 64;

fn truncate_for_error(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_string()
    } else {
        format!("{}...", text.chars().take(max_chars).collect::<String>())
    }
}
```

### 4.6 Documentation (`docs/supported-models.md`, `CHANGELOG.md`)

R-M1 removed the duplicate `Qwen3.5-VL, Qwen3.8-VL` mention from the family summary bullet. R-L1 rewrote the KV-footprint sentence to state the measured (64.1 KiB/token) and theoretical (64.0 KiB/token) numbers separately with their ~0.16% difference, instead of calling them exact. R-L2 split the weight-map key list into the upstream `Qwen/Qwen3.8-27B` shape and the validated `mlx-community/Qwen3.8-27B-4bit` shape, each verified against the corresponding local checkpoint's `model.safetensors.index.json`. `CHANGELOG.md` gained the `## [Unreleased]` entry for #1163 with the required upgrade note on `vision_start_token_id` becoming mandatory.

---

## 5. Premise Correction: the two upstream ports were already implemented

The issue's other deliverable, porting Blaizzy/mlx-vlm#1805 and Blaizzy/mlx-vlm#1741, resolved to a premise correction rather than new implementation work, verified before any code was written and re-confirmed here by mutation testing during finalization:

- **Blaizzy/mlx-vlm#1805** (padded-vocabulary structured-output mask): `apply_structured_mask_to_logits` in `src/server/structured.rs` already pins the bias length to `vocab_size_hint`, the model's logits axis, rather than the matcher's vocabulary. **Mutation-verified in this pass**: forcing `vocab_size` to read from `constraint.vocab_size()` (the matcher's 248,077-entry vocabulary) instead of `vocab_size_hint` (the checkpoint's 248,320-row logits axis) makes `apply_mask_covers_the_qwen3_8_padded_lm_head` abort the test process with `libc++abi: terminating due to uncaught exception ... [broadcast_shapes] Shapes (1,248320) and (1,248077) cannot be broadcast`, an FFI-level failure inside `mlxcel_core::add` rather than a graceful assertion failure. The mutation was reverted immediately after confirming the abort; `git diff` against the pre-mutation state is empty.
- **Blaizzy/mlx-vlm#1741** (chunked-MRoPE position-slice reuse): `mrope_position_source` in `src/models/qwen3_5.rs` already requires `shape[1] == batch` (Blaizzy/mlx-vlm#1040) and `shape[2] >= cache_offset + seq_len` (Blaizzy/mlx-vlm#1048) before reusing the stored `position_ids` tensor, falling back to delta-based recompute otherwise. **Mutation-verified in this pass**: dropping the `shape[1] == batch` conjunct makes `mrope_position_source_rejects_a_batch_mismatch_and_a_wrong_rank` fail (`left: SliceStored { start: 0, end: 8 }, right: Recompute`). The mutation was reverted immediately after confirming the failure; the post-revert diff against the parent commit is empty.

Both mutation runs used the project's release-only build constraint (`DEVELOPER_DIR=/Applications/Xcode-26.6.0.app/Contents/Developer`, `--release --features metal,accelerate`) and were reverted before any further work continued, keeping the finalization commits limited to the five code findings and three doc findings above.

---

## 6. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed (code fix commit `4e2fab1e`) | 7 |
| Files changed (docs/changelog commit `298b511e`) | 2 |
| Lines added / removed (code fix) | +519 / -45 |
| Lines added / removed (docs/changelog) | +6 / -2 |
| Tests added | 7 (`models::qwen3_5_tests`) + 4 (`loading::vlm::tests`) + 1 (`loading::vlm::special::tests`) = 12 new test functions, one covering the two config-shape regression fixtures for `output_gate_type` and `mrope_interleaved` combined with the pre-existing 6 |

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| Validation wiring | 1 | S-M1: `validate_supported()` at the MiniCPM-V 4.6 site |
| Type-safety hardening | 2 | S-M2: strict `mrope_interleaved`/`language_model_only` checks |
| Bounds checking | 1 | S-M3: `qwen35_vl_token_id_in_range` for all three vision token ids |
| Matching correctness | 1 | R-M2: case-insensitive `output_gate_type` |
| Error-output safety | 1 | S-L2: `truncate_for_error` / `MAX_ERROR_VALUE_CHARS` |
| Docs | 3 | R-M1, R-L1, R-L2 in `docs/supported-models.md` |
| Release notes | 1 | `CHANGELOG.md` `## [Unreleased]` entry with upgrade note |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `34f14df3` | feat | qualify Qwen3.8-27B on the qwen3_5 path (parent PR) |
| `4e2fab1e` | fix | harden Qwen3.5-family config validation gaps from PR #1174 review |
| `298b511e` | docs | fix Qwen3.5-VL duplicate listing, KiB precision, and weight-map attribution |

---

## 7. Follow-up Actions

### Required

- [ ] None; the review and security passes reported no CRITICAL or HIGH findings, and all five MEDIUM/LOW code findings plus all three doc findings are fixed in this pass.

### Future Improvements (recorded as known limitations, not fixed here)

- `vision_end_token_id` is still not read anywhere in the tree, unchanged from the parent PR's own documented decision; nothing consumes it, so adding a field would be dead code.
- The MTP speculative-decoding gap for this family (mlx-community conversions drop `mtp.*` and publish the drafter separately) remains tracked as #1165, video input as #1166, both out of scope for this finalization pass.
- The `check_cross_repo_refs.py` advisory flags bare `#1163`/`#1165`/`#1166` references in this tree because the script's heuristic treats any 3+-digit bare `#NNN` as likely-upstream; these are genuine same-repo `lablup/mlxcel` references and the script is advisory-only (`exit 0`), so no action was taken.

---

## Appendix

### A. Test Results

- `cargo test --release -p mlxcel --lib --features metal,accelerate qwen3_5`: 25 passed, 11 ignored (pre-existing serial-MLX gates), 0 failed.
- `cargo test --release -p mlxcel --lib --features metal,accelerate loading::vlm`: 178 passed, 0 failed, including the new `load_minicpmv4_6_vlm_rejects_an_unsupported_output_gate_type` end-to-end wiring test.
- `cargo clippy --release --lib --tests --features metal,accelerate -- -D warnings`: clean.
- `cargo fmt --check`: clean (after one `cargo fmt` pass to wrap three lines the new code pushed past 100 columns).
- `python3 scripts/ci/check_cross_repo_refs.py`: advisory findings only (see Future Improvements), exit 0.
- Local checkpoint check for S-M1: both `minicpm-v-4.6-bf16` and `minicpm-v-4.6-mxfp4`'s `text_config` were read directly and confirmed to omit `output_gate_type` and `rope_parameters.mrope_interleaved`, so the new `validate_supported()` call does not turn either into a startup failure.
- Weight-map verification for R-L2: both `qwen3.8-27b-hf-bf16/model.safetensors.index.json` (`model.language_model.*`, `model.visual.*`, 15 `mtp.*` keys, `lm_head.weight`) and `qwen3.8-27b-4bit/model.safetensors.index.json` (`language_model.*` including `language_model.lm_head.*`, `vision_tower.*`, zero `mtp.*` keys) were inspected directly.
- Mutation verification (this pass, reverted after confirming failure in each case): `apply_structured_mask_to_logits` sized from the matcher vocabulary instead of `vocab_size_hint` aborts `apply_mask_covers_the_qwen3_8_padded_lm_head` on an MLX broadcast-shape mismatch; `mrope_position_source` with the `shape[1] == batch` check dropped fails `mrope_position_source_rejects_a_batch_mismatch_and_a_wrong_rank`.

### B. References

- Issue #1163 (specification)
- Issue #1165 (MTP speculative decoding for this family, out of scope)
- Issue #1166 (video input for this family, out of scope)
- `src/models/qwen3_5.rs` (`Qwen35Config::validate_supported`, `mrope_interleaved_checked`, `truncate_for_error`), `src/loading/vlm.rs` (`qwen35_vl_token_ids`, `qwen35_vl_token_id_in_range`), `src/loading/vlm_special.rs` (`load_minicpmv4_6_vlm`)
- PR #1174 review and security comments
- Blaizzy/mlx-vlm#1805, Blaizzy/mlx-vlm#1741, Blaizzy/mlx-vlm#1040, Blaizzy/mlx-vlm#1048, Blaizzy/mlx-vlm#1812 (upstream references cited by the parent PR and re-verified here)
