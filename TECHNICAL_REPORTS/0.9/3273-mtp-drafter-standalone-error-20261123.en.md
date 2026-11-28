# Technical Report: PR #3273 - Name MTP drafters passed to `-m` instead of "unsupported"

**Date**: 2026-11-23
**Status**: Open (written before merge)
**Risk Level**: Low (changes only the error returned for configs that already failed detection)

## Executive Summary

`mlxcel-server -m mlx-community/gemma-4-12B-it-assistant` exited with `Unsupported model type: gemma4_unified_assistant`. The checkpoint is not an unported family: it is the MTP speculative drafter for Gemma 4 12B, already supported through `--draft-model` next to its target. The PR makes model detection recognize the MTP drafter `model_type`s registered in `mlxcel-core`'s drafter registry and return an error that names the checkpoint as a drafter and points at `--draft-model`. No model gains or loses support.

## 1. Problem Statement

An MTP drafter (`gemma4_assistant`, `gemma4_unified_assistant`, `qwen3_5_mtp`, `glm4_moe_lite_mtp`, `qwen4_exp_mtp`) predicts draft tokens from the target model's hidden states and has no standalone generation path. It is loaded by the drafter loader, which reads `model_type` through `peek_drafter_model_type`, never through `get_model_type`. Passing one to `-m` sends it through the generic detector instead, and the detector had no arm for it, so the user saw the same message as for an architecture mlxcel has never heard of. That message suggests filing a porting request, when the fix is a different flag.

DFlash-family drafters already had a dedicated rejection (`dflash_drafter_not_standalone_error`, #1168) because they declare an ordinary `model_type: qwen3` and would otherwise mis-route. MTP drafters had no equivalent.

## 2. What Changed

- `src/models/detection.rs`: a guarded arm `_ if is_mtp_drafter_model_type(&model_type)` placed immediately before the `Unsupported model type` fall-through, plus `is_mtp_drafter_model_type` (looks the lowercased `model_type` up in `mlxcel_core::drafter::drafter_kind_by_model_type()` and accepts only `DrafterKind::Mtp`) and `mtp_drafter_not_standalone_error`.
- `src/models/detection_tests.rs`: one test over `gemma4_unified_assistant`, `gemma4_assistant`, `qwen3_5_mtp` and the mixed-case `Gemma4_Unified_Assistant`, asserting the new wording, the `--draft-model` hint, and the absence of the generic message.

## 3. Technical Decisions

**Fall-through arm, not an early check.** The DFlash rejection runs before dispatch because its configs would otherwise match the `qwen3` arm. MTP `model_type`s match no arm today, so checking only where the generic error used to fire guarantees that no config that loads now is routed differently, even if a future drafter `model_type` collides with a target arm.

**Reuse the drafter registry instead of a second list.** `drafter_kind_by_model_type()` is what `--draft-model` auto-detection already uses, so a new MTP drafter registered there gets the right standalone error with no change to detection. Filtering on `DrafterKind::Mtp` leaves the Muse Glimmer assistant (registered as `Dflash`) to the existing DFlash rejection, which catches it structurally earlier.

**Error only, no auto-pairing.** Loading the target automatically from a drafter directory would need a target path the drafter config does not carry reliably; the message tells the user which flag takes which checkpoint instead.

## 4. Validation

- New detection test passes; the detection suite (278 tests after rebasing onto main), workspace clippy with `-D warnings`, and fmt pass locally. PR CI: 19 checks pass, 14 skipped.
- The release `mlxcel-server -m` against the published `mlx-community/gemma-4-12B-it-assistant-bf16` `config.json` (`model_type: gemma4_unified_assistant`, `Gemma4UnifiedAssistantForCausalLM`) now prints the new message.
- No non-test code matches on the old `Unsupported model type` string, so callers such as the WebUI catalog only see different error text for drafter directories.

## 5. Limitations

- The full `-m <gemma-4-12B-it> --draft-model <assistant>` pairing was not run because no Gemma 4 weights were on the validation host. The change does not touch that path.
- The message's example names Gemma 4 even when the drafter is a Qwen or GLM head; the wording is conditional ("for a Gemma 4 assistant"), but it is less specific for those types.
