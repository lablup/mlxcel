# Technical Report: PR #1511 - Template rejection surfaces

**Date**: 2026-08-30
**Status**: Completed
**Languages**: Rust, Markdown
**Risk Level**: Medium

## Executive Summary

PR #1511 fixes issue #1176 by keeping chat-template `raise_exception(...)` rejections visible across the Responses API, the offline CLI, and the disaggregated router front end. Deliberate template refusals now stop the request with a named client error, while malformed or unsupported template-engine failures still use the existing fallback path.

## 1. Problem Statement

The single-node `/v1/chat/completions` route already treated template-raised rejections as request errors, but adjacent surfaces could still swallow or misclassify them. `/v1/responses` needed explicit regression coverage for the current `reasoning.effort` forwarding behavior, the offline CLI fell back to the raw user prompt for every template error, and the disaggregated router converted every preparation error into an HTTP 500.

The failure mode was especially risky for reasoning-effort controls. Values are intentionally forwarded verbatim instead of translated between OpenAI and model-specific vocabularies, so a model template that rejects `high` must produce a visible client error rather than silently generating from an unframed fallback prompt.

## 2. Technical Decisions

### 2.1 Preserve the rejection sentinel through preparation

`prepare_chat_request_with_cache` now wraps template rejections with user-facing context while preserving the original render error as the source. This keeps `template_rejection_message` usable at outer route boundaries such as the disaggregated router, without changing the visible error text already used by single-node chat and Responses routes.

### 2.2 Keep CLI fallback only for engine failures

The offline CLI prompt helpers now return `Result<String>`. They return a named error when `template_rejection_message` finds the sentinel, but still return the raw prompt when minijinja cannot render the template for engine-side reasons such as malformed syntax.

### 2.3 Do not translate reasoning values

The Responses translator behavior remains unchanged: `reasoning.effort` is copied into the chat request as-is. The new tests prove that a `high` value reaches the template unchanged and remains rejectable by a model-specific template.

### 2.4 Map router template rejections to 400

The disaggregated router now checks the preserved sentinel after chat request preparation. It returns the same `400 invalid_request_error` response shape for template rejections and leaves unrelated preparation failures on the existing generic 500 path.

## 3. Change Summary

| Category | Count | Summary |
|---|---:|---|
| Error propagation | 1 | Preserved the template rejection source through preparation error wrapping. |
| CLI behavior | 1 | Made CLI prompt resolution fallible only for template rejections. |
| Router behavior | 1 | Converted disaggregated router template rejections from 500 to 400. |
| Responses coverage | 2 | Added rejection and engine-fallback regression tests. |
| CLI coverage | 3 | Updated prompt-helper tests and added rejection-path coverage. |
| Router coverage | 1 | Added HTTP 400 regression coverage for template rejection. |

## 4. Validation

- `cargo test --bin mlxcel resolve_cli_prompt`: 5 passed.
- `cargo test --bin mlxcel apply_user_chat_template_wraps_prompt_as_user_message`: 1 passed.
- `cargo test --bin mlxcel vlm_chat_template`: 4 passed.
- `cargo test --lib responses_reasoning_effort_template_rejection_survives_translation`: 1 passed.
- `cargo test --lib responses_template_engine_failure_still_uses_prompt_fallback`: 1 passed.
- `cargo test --lib router_chat_maps_template_rejection_to_bad_request`: 1 passed.
- `cargo test --lib template_rejection`: 11 passed.
- `cargo clippy --lib --tests -- -D warnings`: passed.
- `cargo clippy --bin mlxcel -- -D warnings`: passed.
- Hosted PR checks passed: cargo-clippy, cargo-deny, cargo-fmt, OpenXLA feature compile, crate versions, cross-repo refs, kernel dtype keys, llama-compat manifest, Detect changes, and license/cla. MLX pin extraction and OpenXLA feature link were skipped by change detection.

Broad workspace tests, serial all-tests, and cold release builds were not run because the implementation workflow explicitly forbids them for this issue batch.

## 5. Review Notes

- **Correctness**: Template rejections and engine failures now diverge at each affected surface. Responses forwarding remains verbatim, so unsupported value mapping is still delegated to the loaded template.
- **Security**: Rejection messages continue to use the existing 512-character `TemplateRejection` cap and control-character filtering before reaching HTTP or CLI output. No new logging of prompt contents was added.
- **Performance**: The new checks inspect an existing error chain only on render failure paths. The successful render and generation paths are unchanged.
- **Compatibility**: Existing fallback behavior remains for malformed or unsupported templates, preserving serving compatibility for models whose templates mlxcel cannot render.

## 6. Follow-up Actions

- Run a live Qwen3.8 Responses and disaggregated-router smoke test on a deployment with that checkpoint when hardware is available.
- Keep the documented Responses behavior as verbatim forwarding; if a future product decision introduces value translation, file a separate issue with explicit compatibility requirements.
