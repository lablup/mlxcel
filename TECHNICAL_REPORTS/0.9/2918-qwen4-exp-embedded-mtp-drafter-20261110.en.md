# Technical Report: PR #2918 - feat(speculative): use a qwen4_exp checkpoint's own mtp.* head as the drafter

**Date**: 2026-11-10
**Reviewer**: -
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low

---

## Executive Summary

A `qwen4_exp` (Qwen3.8-Flash-Next) target checkpoint that keeps its `mtp.*` tensors can now be its own MTP drafter: the target directory is passed again as `--draft-model` or `--model-draft`. Only the head's shard is read, so the drafter adds 1.61 GiB instead of a second copy of the 113 GB checkpoint. On `mlx-community/Qwen3.8-Flash-Next-4bit-mtp` the output stays byte-identical to classic decode, and decode rises from about 46 tok/s to 54 to 68 tok/s depending on the prompt.

---

## 1. Problem Statement

### 1.1 Background

PR #2908 (issue #2880) added the native MTP drafter for `qwen4_exp`, assuming that target conversions drop `mtp.*` and that the head therefore always comes as a separate `qwen4_exp_mtp` directory. `mlx-community/Qwen3.8-Flash-Next-4bit-mtp` (revision `825df623`) keeps the head inside the target: 54 `mtp.*` tensors in a dedicated shard `model-mtp-00001-of-00001.safetensors`, with 11 quantization overrides under the `mtp.` prefix.

### 1.2 Existing Issues

- **Drafter directory required**: the loader required `model_type: qwen4_exp_mtp` and read every safetensors file in the directory, so the only way to use the embedded head was to extract it by hand and write a config.
- **Kind resolution**: `resolve_drafter_kind` resolved a `qwen4_exp` directory as a full-model drafter.
- **Quantization keys**: the target config names overrides with the `mtp.` prefix, while the drafter looks them up by stripped module names.

---

## 3. Technical Decisions

### 3.1 No new flag

| Option | Pros | Cons |
|--------|------|------|
| `--draft-kind mtp` without `--draft-model` | Shorter command line | A second convention next to the existing drafter path, more argument paths to test |
| **Chosen: pass the target directory again** | Same convention Inkling already uses, server and offline paths need no change | The path is repeated |

### 3.2 Detection by tensor names, behind the existing resolver

`resolve_drafter_kind` probes a `qwen4_exp` directory for `mtp.*`, `model.mtp.` or `language_model.mtp.` tensors (index or headers) when the kind is unset or `mtp`. An explicit `--draft-kind mtp` without the head fails with the new `DrafterError::NoEmbeddedMtpHead`. Without `--draft-kind`, a directory without the head resolves as before.

**Trade-off:** a `qwen4_exp` directory with an embedded head, passed as a drafter without `--draft-kind`, now resolves to MTP instead of a full-model drafter. `--draft-kind full-model` still selects the old behaviour.

### 3.3 Derive the drafter config from the target

`Qwen4ExpMtpConfig::from_target_json` sets `model_type: qwen4_exp_mtp` and an explicit `block_size: 4` (the `from_json` default would be `mtp_num_hidden_layers + 1 = 2`), carries over `text_config` and `tie_word_embeddings`, keeps only the default quantization pair plus the `mtp.*` overrides with the prefix stripped, then runs the existing `from_json`. Reusing `from_json` keeps the single place that turns the checkpoint's `full_attention` spelling into the one `qwen_sparse_attention` layer.

---

## 4. Implementation Details

- `src/lib/mlxcel-core/src/drafter/mod.rs`: `is_qwen4_exp_mtp_tensor_name`, `strip_qwen4_exp_mtp_prefix`, `has_qwen4_exp_mtp_tensors` (on `dir_has_tensor_name`), the resolver probe, `NoEmbeddedMtpHead`. Core `load_drafter` refuses `qwen4_exp` by name because the binary crate builds it.
- `qwen4_exp_sanitize::is_mtp_key` delegates to the core predicate, so the target and the drafter agree on what counts as the head.
- `Qwen4ExpMtpDraftModel::from_target_checkpoint` loads through `load_weights_from_dir_index_filtered(path, is_mtp_key)`, so only shards that hold head tensors are opened, and logs tensor count and resident bytes.
- `drafter_loader::load_drafter` routes `qwen4_exp` with `Mtp` to the new constructor; offline `generate` and the server reach it unchanged.
- Memory estimate: unchanged. No estimate path adds drafter bytes, and the index `total_size` already counts the head once (`mlxcel inspect` reports 113,245,755,466 bytes).

---

## 7. Change Summary

| Item | Value |
|------|-------|
| Files changed (before this report) | 10 |
| Lines added | +631 |
| Lines deleted | -25 |

| Hash | Type | Message |
|------|------|---------|
| `628e86e5` | feat | use a qwen4_exp checkpoint's own mtp.* head as the drafter |
| `b03f722f` | fix | satisfy clippy and record the embedded-head validation |

---

## 8. Follow-up Actions

- A classic `mlxcel-server` without a drafter on the 4-bit target, under `MLXCEL_QMV_WIDE=0`, diverges from offline classic decode in the first answer sentence. The server with either the embedded or the split drafter matches offline classic. The divergence is in classic server decode and predates this PR; it needs its own issue.
- A `qwen4_exp` directory without `mtp.*` passed straight to `from_target_checkpoint` gets a generic `WeightLoad` error. The factory never routes such a directory there.
- Embedded heads of other families are not covered.

---

## Appendix

### A. Test Results

- Unit: the tiny fixture written as a target checkpoint (its index points the non-mtp tensors at a shard of invalid bytes) and as a split directory gives identical draft logits for the prompt and a cached step, both matching upstream; override stripping and `block_size` 4; `NoEmbeddedMtpHead` from the factory and the constructor; the `hc_count` refusal; unchanged resolution without `mtp.*`. `mlxcel-core` drafter tests 257 passed, `qwen4_exp_mtp` 14 passed.

### B. Real model (M3 Ultra, `Qwen3.8-Flash-Next-4bit-mtp`, `--temp 0`, reasoning shown)

| Prompt | MTP vs classic (`MLXCEL_QMV_WIDE=0`) | Acceptance | tok/s classic / MTP |
|---|---|---|---|
| English | byte-identical, 256 tokens | 0.611 | 46.1 / 60.7 |
| Code | byte-identical, 214 tokens (EOS) | 0.731 | 45.2 / 67.8 |
| Korean | byte-identical, 256 tokens | 0.507 | 46.0 / 53.6 |

Single runs, not a paired benchmark. Drafter resident size 1,726,854,656 bytes, 54 tensors. `mlxcel-server --model-draft <same dir>` returned the offline output on two requests. The split drafter `Qwen3.8-Flash-Next-MTP` with the OptiQ 2-bit target stayed byte-identical to classic over 128 tokens (acceptance 0.500), and on the 4-bit server it returned the same bytes as the embedded head.
