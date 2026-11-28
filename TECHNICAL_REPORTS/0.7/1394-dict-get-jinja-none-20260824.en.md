# Technical Report: PR #1394 - Return None for Missing Template Map Keys

**Date**: 2026-08-24

**Author**: Jeongkyu Shin

**Status**: Completed

**Languages**: Rust, Jinja

**Risk Level**: Medium

## Executive Summary

PR #1394 aligns mlxcel's shared Python-style `dict.get()` compatibility shim with CPython Jinja2: a missing key without an explicit default now returns Minijinja `None` instead of `Undefined`. This corrects Muse Glimmer multi-turn prompts so completed assistant messages end with `<|eot|>` rather than the continuation marker `<|eom|>`, while preserving explicit defaults, falsy values, and `or`-chain behavior.

## 1. Problem Statement

The Muse Glimmer checkpoint template uses `message.get('end_turn') is none` to decide whether to infer an end-of-turn marker. mlxcel returned `Undefined` for a missing key, so the `is none` guard did not run and every plain assistant message was rendered as an unfinished turn. The resulting prompt stayed syntactically valid but diverged from the byte sequence used to train and validate the checkpoint.

## 2. Change Summary

| Area | Change |
|---|---|
| `src/server/chat_template.rs` | Return `Value::from(())` for a missing one-argument `dict.get()` call and document the Python-compatible semantics |
| Template compatibility tests | Cover missing keys as both `none` and `defined`, and preserve present JSON `null`, `false`, zero, empty strings, explicit defaults, and fallback chains |
| Muse Glimmer fixture tests | Assert the `<|eot|>` terminator by name and update the two affected render digests to the Jinja reference values |

## 3. Technical Decisions

### 3.1 Correct the shared compatibility funnel

The fix stays in `configure_environment`, the single callback installed by both typed and raw `ChatTemplateProcessor` render paths. A template-local workaround would leave every other checkpoint vulnerable to the same CPython compatibility trap and could create inconsistent behavior between chat completions, Responses, Anthropic translation, router, and offline CLI paths.

### 3.2 Preserve falsy semantics explicitly

Minijinja `None` and `Undefined` are both falsy, so existing `m.get('a') or m.get('b')` templates continue to work. The review added direct raw-JSON coverage for present `null`, `false`, `0`, and `""` values so a future refactor cannot confuse a missing key with a present falsy value.

### 3.3 Pin visible behavior as well as hashes

Digest assertions prove byte identity but can be mechanically updated after a regression. The Muse tests now also assert that completed assistant content contains `<|eot|>` and does not contain `<|eom|>`, keeping the semantic invariant readable during future fixture updates.

## 4. Verification

- `cargo fmt --all -- --check`: passed.
- `cargo test -p mlxcel --profile test-fast --lib test_dict_get_method`: 6 passed.
- `cargo test -p mlxcel --profile test-fast --lib muse_glimmer_template_`: 6 passed.
- Local Jinja2 reference render: `multi_turn` is 325 bytes with SHA-256 `433d37ff14caf2f2b177d904726b34ff09cb2aad4426a237a7b27772eab47007`; `tools_and_results` is 2340 bytes with SHA-256 `dc451d3030d24f37ecc20fc0236c0b5fa7f70032d8c5331f8f6690689620d6ae`.
- Hosted CI: formatting, Clippy, cargo-deny, OpenXLA feature compile, metadata checks, and CLA passed; change-irrelevant MLX pin extraction and OpenXLA link jobs were skipped.
- Implementation, security/performance, and finalization reviews found no unresolved correctness or security issue after the raw-value coverage addition.

## 5. Remaining Validation Boundary

A real Muse Glimmer multi-turn GPU smoke test was not run because this Linux host exposed neither a usable NVIDIA driver nor a Metal backend. The deterministic render tests and Jinja reference comparison prove the corrected prompt bytes, but they are not presented as real-checkpoint generation evidence.

## 6. Related Work

- Issue #1383: Python-style `dict.get()` returned `Undefined` for missing keys.
- PR #1394: implementation and review fixes documented here.
- PR #1382 / issue #1309: adjacent `tojson` compatibility work where the pre-existing terminator defect was discovered; not changed by this PR.
