# Technical Report: PR #1489 - Control-vector explicit rejection

**Date**: 2026-08-28

**Status**: Completed

**Languages**: Rust

**Risk Level**: Low

## Executive Summary

Classifies the three b10621 control-vector options as explicit unsupported behavior, the alternative branch issue #1449 itself offers: `--control-vector` and `--control-vector-scaled` parse (hidden) on both server binaries and fail startup before the model load with a diagnostic naming logit-level steering as the nearest mlxcel feature, while `--control-vector-layer-range START END` is accepted as inert because without vectors it configures nothing, exactly as upstream. All three #1449-owned manifest entries leave `deferred` as `not_applicable`, with the real divergence (b10621 loads and applies the vector) recorded.

## 1. Problem Statement

Control vectors change model activations rather than loading, so accepting them as compatibility no-ops would silently change what a deployment believes it is serving. Implementing them would require per-layer forward-pass steering hooks across every model family plus a GGUF vector loader, which is exactly the half-finished cross-family breadth the project guidelines forbid landing; the issue explicitly sanctions classification as unsupported behavior instead.

## 2. Technical Decisions

### 2.1 Rejection over implementation, at parse-adjacent startup time

The flags join the existing hidden `GgmlCompatArgs` accept-and-classify surface (shared machinery rather than a three-flag module), so a configured vector fails in milliseconds, before any weight is read, with the alternative (`--lang-bias`, per-request `logit_bias`) named. A scale of zero is still a configured vector set and is rejected as a whole, never partially applied, which also satisfies the issue's atomicity criterion. Because only the no-vector configuration can ever serve, prompt/KV cache reuse across control-vector configurations is impossible by construction.

### 2.2 Honest records

The review pass verified every behavioral claim against both binaries and the pinned b10621 source and found no correctness issues; its record-keeping findings were applied: divergence entries on both rejected options (b10621 loads and applies the vector), integer parsing on the layer range exactly as upstream's `stoi`, corrected `<START> <END>` diagnostics, the `,...` repeat marker, a module-header acknowledgment of the #1449 section, and an operator-facing section in `docs/llama-server-compat.md`.

### 2.3 The scale-zero criterion

The issue's teacher-forced baseline and scale-zero reproduction criterion applies, by its own wording, to an implemented path. Under the rejection classification no sampling or forward arithmetic changed, so the trace comparison is identically zero by construction; the scale-zero case is pinned instead as a startup-rejection test, because a zero-scaled vector is a configured set, not a silent no-op.

## 3. Change Summary

| Item | Value |
|------|-------|
| Files changed | 4 (+4 in the review follow-up) |
| Lines | +157 / -19 across both commits |
| Manifest entries | 3 flip to not_applicable; zero #1449 deferred remain |

Validation: 2 new unit tests plus the 36-test ggml compat suite, compat integration 4/4 against both rebuilt binaries, manifest checker green with the shard diff confined to the three #1449 entries, clippy `-D warnings`, fmt, and live smoke (rejection pre-load with the alternative named, range-only inert, flags hidden from `--help`, non-numeric range values rejected).

## 4. Follow-up Actions

- None owned by this issue. An actual control-vector implementation, if ever wanted, starts from a fresh issue with the forward-pass hook design; this classification makes the current truth checkable instead of leaving the flags unparseable.
