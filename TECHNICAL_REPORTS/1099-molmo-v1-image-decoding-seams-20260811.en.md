# Technical Report: PR #1099 - fix(models): correct Molmo v1 image decoding seams

**Date**: 2026-08-10
**Author**: mlxcel maintainers
**Reviewer**: implementation, security, and finalizer review cycle
**Status**: Completed for an open PR
**Languages**: Rust, Markdown
**Risk Level**: Medium

---

## Executive Summary

PR #1099 fixes the Molmo v1 seams behind issue #1087: image-conditioned generation from `molmo-7b` was incoherent on a real COCO cats photo while `molmo2-4b` answered correctly on the same setup. The root cause was not the server path, BOS insertion, or feature scatter. It was two model-family seams in the Molmo v1 decode stack: flat configs that omit `rope_impl` were falling back to a stale local default that selected MLX traditional interleave RoPE instead of the checkpoint's effective LLaMA rotate-half layout, and the image preprocessor was deviating from reference behavior by padding in normalized space and collapsing fractional patch coverage to booleans.

The PR fixes both seams in `src/models/molmo.rs` and `src/vision/processors/molmo.rs`, adds targeted regression coverage, and was validated on the real NVIDIA GB10 CUDA path that originally reproduced the bug. The pre-fix CLI run reproduced the reported garbage output byte-for-byte at `prompt_tokens=749`; the fixed CLI and the normal BatchScheduler-backed server path both returned the same coherent description of two cats sleeping on a red couch under a pink blanket. Two independent review passes reported no further findings, and the finalizer pass judged the PR merge-ready.

---

## 1. Problem Statement

### 1.1 User-visible failure

Issue #1087 reported that `molmo-7b` generated incoherent output on image prompts while `molmo2-4b` answered correctly on the same GB10 CUDA setup. The failure was reproduced on the real 640x480 COCO cats image and was already present on `main`, so it was a pre-existing Molmo v1 defect rather than a regression from adjacent work.

### 1.2 Acceptance target

The issue required three things:

- coherent real-photo CLI output for Molmo v1,
- a fix at the actual root-cause seam rather than a prompt-side workaround,
- and verification on both `mlxcel` CLI and `mlxcel-server`.

---

## 2. Root Cause

Two independent seams were confirmed.

### 2.1 Omitted `rope_impl` selected the wrong RoPE layout

Molmo v1 flat configs can omit `rope_impl`. The released checkpoint still effectively expects the LLaMA rotate-half layout, but the local dataclass default treated the omission as MLX traditional interleave RoPE. That mismatch corrupts positional rotation before any image-conditioned reasoning happens, so decoding becomes nonsensical even when the prompt, server path, and image features are otherwise correct.

The fix makes omitted Molmo v1 `rope_impl` default to the checkpoint's effective LLaMA layout while preserving explicit `interleave` as the MLX traditional path.

### 2.2 The image preprocessor was not reference-faithful

`pad_and_partial_pad` had two reference mismatches:

- it padded after normalization, so padded pixels became zero-valued normalized-space pixels instead of normalized black,
- and it thresholded `image_masks` to booleans, losing the fractional border coverage values the reference path preserves.

Those mistakes degrade the visual tokens that Molmo v1 consumes, especially on partial border patches. The fix pads raw black before normalization and preserves float mask coverage end to end.

---

## 3. Confirmed Non-Causes

Several plausible suspects were checked and explicitly ruled out.

- The `749` prompt-token expansion itself was not the defect. The broken CLI and the fixed CLI both used the same real prompt length, so the issue was in decoding semantics, not prompt assembly length.
- The Molmo v1 BOS handling was not the differentiator. The reproduced bad output happened on the existing CLI path, and the fixed output came through the same BOS path after the model/preprocessor corrections.
- The image-feature scatter path was not the root cause. The coherent fixed result came from the same shared CLI/server multimodal runtime path after the RoPE and preprocessor fixes.
- The server transport path was not unique. The normal BatchScheduler-backed `mlxcel-server --parallel 1` path returned the same coherent answer and the same usage totals as the fixed CLI.

---

## 4. Implementation Summary

| Item | Value |
|------|-------|
| Implementation files changed | 2 |
| Implementation lines added | +158 |
| Implementation lines deleted | -16 |
| Implementation commits | 1 |
| Reviewed implementation head | `8d30a09b79138d408860eb04cf23f94d7be06897` |

The bilingual report artifacts and the report-only commit are excluded from the counts above.

- `src/models/molmo.rs`: default omitted Molmo v1 `rope_impl` to the checkpoint-effective LLaMA split-half path and add regression coverage for omitted vs explicit RoPE controls.
- `src/vision/processors/molmo.rs`: pad raw black before normalization, preserve fractional `image_masks`, and add deterministic tests for normalized padding and the known 640x480 `9/14` partial border patch case.

---

## 5. Validation

### 5.1 Local deterministic validation

- `cargo fmt --all -- --check`: passed.
- `cargo test -p mlxcel models::molmo::tests`: passed.
- `cargo test -p mlxcel vision::processors::molmo::tests`: passed.
- `cargo test -p mlxcel --test molmo_parity`: passed, but all four tests returned early because the harness searches `models/molmo-7b` rather than `/home/inureyes/models/mlx/molmo-7b`. This suite therefore did **not** provide the real-checkpoint proof for this PR.
- `cargo clippy -p mlxcel --lib --tests -- -D warnings`: passed.
- `cargo build --release --features cuda --bin mlxcel --bin mlxcel-server --locked` on NVIDIA GB10 (`sm_121`): passed.

### 5.2 Real-checkpoint A/B on the original failure path

CLI, GB10 CUDA, real 640x480 COCO cats image:

- pre-fix: reproduced the reported incoherent output byte-for-byte at `prompt_tokens=749`, `completion_tokens=40`, `9.35s`, `4.28 tok/s`,
- post-fix: generated a coherent description of two cats sleeping on a red couch under a pink blanket at `prompt_tokens=749`, `completion_tokens=40`, `4.06s`, `9.85 tok/s`.

### 5.3 Real server validation

Normal BatchScheduler server path (`mlxcel-server --parallel 1`) with the same image and prompt:

- returned the same coherent text as the fixed CLI,
- and reported `prompt_tokens=749`, `completion_tokens=40`, `total_tokens=789`.

The legacy `--no-batch` mode was also attempted and failed before generation with its pre-existing `CachePool` max-capacity-zero admission error. That failure predates this PR, is independent of the Molmo v1 decode/preprocessor seams, and does not invalidate the normal production scheduler-path verification.

### 5.4 Hosted checks observed on PR #1099

- `Detect changes`: pass
- `crate versions`: pass
- `kernel dtype keys`: pass
- `cross-repo refs`: pass
- `cargo-deny`: pass
- `cargo-fmt`: pass
- `license/cla`: pass
- `MLX pin extraction`: skipped

---

## 6. Review Outcome

The implementation went through two independent review passes after the first fix landed:

- an implementation review found no remaining correctness gaps after the real-path CLI/server validation and the added regression tests,
- a security review likewise reported no findings and no new attacker-controlled surface from the changes.

The finalizer pass confirmed that the issue acceptance criteria were met on the normal production scheduler path and that the remaining `--no-batch` `CachePool` failure is pre-existing, out of scope, and non-blocking for this PR.

---

## 7. Technical Takeaways

- Flat config omissions are dangerous when a model family's effective checkpoint behavior no longer matches an old local default. The safe boundary is the checkpoint-effective constructor behavior, not the stale dataclass assumption.
- Vision preprocessors need reference-faithful padding and mask semantics. Losing fractional border coverage is enough to poison multimodal decoding without causing a hard error.
- A passing parity harness is only useful if it actually reaches the real checkpoint. Here, `cargo test --test molmo_parity` was valuable as a code-level regression suite but not as the acceptance proof, because it returned early on this machine.

---

## 8. Related Work

- PR #1099: https://github.com/lablup/mlxcel/pull/1099
- Issue #1087: https://github.com/lablup/mlxcel/issues/1087
