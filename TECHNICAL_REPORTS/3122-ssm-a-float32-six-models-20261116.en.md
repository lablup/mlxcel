# Technical Report: PR #3122 - compute the SSM A from A_log in float32 in six more models

**Date**: 2026-11-16
**Status**: PR open (closes #3057)
**Reviewer**: independent review, no CRITICAL or HIGH findings; one MEDIUM test gap fixed in the PR, one MEDIUM CUDA performance note deferred to a follow-up
**Languages**: Rust
**Risk Level**: Low

---

## Executive Summary

The selective-state-space models compute `A = -exp(A_log)`. In float16, `exp` overflows past `A_log = 11.09`, giving `-inf`; a zero time step then computes `0 * -inf = NaN`, the logits go NaN and greedy decoding emits token 0. #3053 fixed this for Mamba. This PR applies the same rule to Jamba, PLaMo2, Falcon-H1, GraniteMoeHybrid, Mamba2 and Nemotron-H: `exp` always runs on a float32 copy of `A_log`. Float32 loads stay bit-identical; bf16 loads change only by the bf16 rounding of `exp`.

---

## 1. Problem Statement

### 1.1 Background

Checkpoints store `A_log` with values up to about 19.5. A bf16 checkpoint that the load-time dtype policy converts to f16 (or an f16 checkpoint) cannot hold `exp(A_log)` above 65504. Casting to float32 after `exp`, as `mamba2.rs` and `ssd_scan.rs` did, is too late: the overflow already happened.

### 1.2 Existing Issues

- **Six models exponentiated in the loaded dtype** on every forward pass: Jamba directly, the other five through the shared SSD scan (`chunked_ssd_scan`) or Mamba2's `ssm_attn`.
- **Failure mode is silent:** the output is a stream of EOS tokens, not an error.

---

## 2. Technical Decisions

### 2.1 Fix the shared scan instead of storing `A` in three models

**Context:** the issue asked to store `A` at load for the four models that own an `a_log` field. Since it was filed, PLaMo2, Falcon-H1 and GraniteMoeHybrid moved their prefill onto `chunked_ssd_scan`, and their decode passes `A_log` to `ssm_update_kernel`, which already computes `exp` in float32 (`mlx_cxx_kernels.cpp`).

| Option | Pros | Cons |
|---|---|---|
| Store `A` in all four models | Matches the issue text | Breaks the decode kernel, which takes `A_log` |
| **Chosen: float32 `exp` in `chunked_ssd_scan` and `ssm_attn`, store `A` only in Jamba** | One change covers five models, including the kernel-unavailable decode fallback | One per-head cast and `exp` per forward pass |

The helper `ssm_a_from_log` was split: `ssm_a_f32` builds the lazy graph (cast, `exp`, negate) and `ssm_a_from_log` calls it and evaluates. Forward passes use `ssm_a_f32`, so no sync is added; load-time code uses `ssm_a_from_log`. There is still one copy of the logic.

### 2.2 Jamba reads `A` in the activation dtype, except f16

Jamba follows `MambaBlock::scan_a`: `A` is cast to the activation dtype, except for f16 activations, where it stays float32. The graph scans then round only the decay `exp(delta * A)`, which lies in (0, 1], back to f16 (`decay_in_activation_dtype`), so the recurrent state keeps the activation dtype.

**Trade-off:** a checkpoint with a float32 `A_log` next to bf16 weights now rounds `A` and the decay to bf16, where it used to compute them in float32. This matches Mamba since #3053.

---

## 3. Review Findings

- **MEDIUM, fixed in the PR:** no test exercised Jamba with f16 activations, so a regression that cast `A` back to f16 would have passed. `jamba_f16_model_with_an_overflowing_a_log_keeps_finite_logits` now covers both the fused scan and the forced graph scan.
- **MEDIUM, deferred:** on CUDA with f16 activations, the graph-exact Mamba1 scan port rejects the float32 `A`, so f16 Jamba (and Mamba since #3053) falls back to the per-step graph scan. Metal and ROCm use the float-state port and are unaffected. Not measured; there is no CUDA host here.
- **LOW, deferred:** `ssd_scan.rs` now imports its helper from `mamba.rs`. Moving both helpers to a shared SSM module would be cleaner.

---

## 4. Change Summary

| Item | Value |
|---|---|
| Files changed | 13 (5 source, 8 test) |
| Tests added | 8 |

- `mamba.rs`: `ssm_a_f32` split out of `ssm_a_from_log`.
- `ssd_scan.rs`, `mamba2.rs`: `A` from `ssm_a_f32`.
- `jamba.rs`: stores float32 `A` at load; `scan_a` and `decay_in_activation_dtype`.
- Tests: one per model asserting finite (and, where applicable, f32-identical) logits for an f16 `A_log` ramping to 19.5 with a zero step; a bounded-tolerance test in `ssd_scan_tests.rs`; the Jamba all-f16 test; shared fixtures in `ssm_test_support.rs` (`#[cfg(test)]`).

### Validation

- Mutation: removing the float32 cast in `ssm_a_f32` fails all 8 `ssm_a_` tests; casting Jamba's f16-branch `A` to f16 or skipping the decay round-back fails the Jamba f16 test.
- `cargo test --lib models::`: 2840 passed, 0 failed, 96 ignored. Clippy and fmt clean.

---

## 5. Follow-up Actions

- Decide how CUDA's graph-exact Mamba1 port should take an f16 model's float32 `A`, and measure the fallback cost on a pre-Ampere GPU.
- Move `ssm_a_f32` and `ssm_a_from_log` into a shared SSM module.
