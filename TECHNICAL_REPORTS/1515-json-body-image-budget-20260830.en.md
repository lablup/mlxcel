# Technical Report: PR #1515 - JSON body image budget

**Date**: 2026-08-30
**Status**: Completed
**Languages**: Rust, Markdown
**Risk Level**: Medium

## Executive Summary

PR #1515 fixes issue #1227 by replacing Axum's implicit 2 MiB extractor limit on the main JSON route table with a bounded limit derived from the configured image payload and image count settings. Base64 image uploads above 2 MiB can now reach the real JSON handlers when they are within the configured image budget, and over-limit extractor failures return the same structured error envelope used by the rest of the OpenAI-compatible API.

## 1. Problem Statement

mlxcel advertised a default 64 MiB encoded image payload budget and up to 16 image blocks per request, but JSON requests carrying base64 `data:` images still hit Axum's default 2 MiB buffered-body limit before the handler could parse the request. Server-side HTTP image fetching honored the configured limit, while the OpenAI-standard base64 JSON path could not reach it.

The audio routes already had a larger per-route extractor limit, but their limit failures also used Axum's bare 413 response. The desired behavior was a single structured boundary: configured image limits should inform the JSON extractor budget, and extractor overflows should produce an OpenAI-shaped 413.

## 2. Technical Decisions

### 2.1 Derive the JSON body ceiling from configured image limits

The main JSON body limit is now derived from the process-wide `ImageInputLimits` that startup configures from `--max-image-payload-size` and `--max-images`. The derivation computes base64 expansion with checked ceil arithmetic, multiplies by the configured simultaneous image count, and adds fixed plus per-image JSON/data-URL overhead.

### 2.2 Preserve the existing audio override

The derived limit is applied as the default extractor limit for the route table. The audio routes still attach their explicit 25 MiB per-route `DefaultBodyLimit`, and Axum's request extension semantics let that route-specific limit override the route-table default.

### 2.3 Bound extreme configurations

Zero-image configurations keep Axum's 2 MiB default rather than shrinking ordinary JSON capacity. Arithmetic overflow and extreme operator values clamp to a 2 GiB ceiling instead of creating an effectively unbounded buffered JSON allocation path. Per-image payload, count, dimension, and decoder allocation checks still run after parsing and enforce the exact configured media limits.

### 2.4 Normalize extractor 413 responses

A lightweight middleware rewrites `413 Payload Too Large` responses into the project's `ErrorResponse` JSON envelope with `invalid_request_error`. It applies after extractor rejection and covers both the main JSON routes and audio upload routes.

## 3. Change Summary

| Category | Count | Summary |
|---|---:|---|
| Body limit derivation | 1 | Added checked base64/JSON image-budget math for the main JSON extractor. |
| DoS guard | 1 | Added a 2 GiB hard ceiling for extreme derived limits. |
| Error envelope | 1 | Normalized extractor 413 responses to OpenAI-shaped JSON. |
| Documentation | 1 | Documented the interaction between base64 JSON images and image-limit flags. |
| Tests | 6 | Covered simultaneous inputs, above-2MiB acceptance, exact/oversized boundary, audio/main parity, zero/extreme config, and error payload shape. |

## 4. Validation

- `cargo test --lib server::app::tests`: 14 passed.
- `cargo clippy --lib --tests -- -D warnings`: passed.
- Hosted checks on implementation commit `1c357ccf`: cargo-clippy, cargo-deny, cargo-fmt, OpenXLA feature compile, Detect changes, crate versions, cross-repo refs, kernel dtype keys, license/cla, and llama-compat manifest passed. MLX pin extraction and OpenXLA feature link were skipped by change detection.

Broad workspace tests, serial all-tests, and cold release builds were not run because the implementation workflow explicitly forbids them for this issue batch.

## 5. Review Notes

- **Correctness**: Tests prove a JSON request body above Axum's 2 MiB default is accepted when the derived image budget permits it, while the exact limit boundary accepts and one-byte-over rejects.
- **Security**: All arithmetic is checked or saturating and extreme inputs clamp to 2 GiB. The change does not disable extractor limits and does not bypass downstream image count, payload, dimension, or decoder allocation validation.
- **Performance**: The derivation runs at app construction. Runtime overhead is limited to the existing extractor limit and a status check in the response middleware.
- **Compatibility**: The body-limit error shape changes from Axum's bare 413 to the OpenAI-compatible `ErrorResponse` envelope; successful JSON and audio route behavior is otherwise unchanged.

## 6. Follow-up Actions

- Exercise a live VLM deployment with a real base64 image near the configured production payload ceiling when suitable hardware is available.
- Revisit the 2 GiB hard ceiling only if deployments intentionally need larger buffered JSON requests and have separate admission controls for that memory budget.
