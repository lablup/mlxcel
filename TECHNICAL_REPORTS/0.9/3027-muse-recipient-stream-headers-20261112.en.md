# Technical Report: PR #3027 - Muse recipient stream headers

**Date**: 2026-11-12
**Status**: Implementation validated locally; CI pending at report time
**Languages**: Rust
**Risk Level**: Medium

## Executive Summary

[PR #3027](https://github.com/lablup/mlxcel/pull/3027) fixes visible Muse Glimmer recipient headers in streamed replies. A response beginning with `to=user<|message|>Hello<|eot|>` now emits `Hello`, while `to=web_search<|message|>` enters tool-message suppression.

## Problem Statement

The chat template already appends the assistant role, so the first generated header can omit `<|start|>assistant`. The stream filter recognized only the primed `to=self` spelling. Direct user replies and tool calls leaked their recipient header, and a separately emitted leading space could survive reasoning into the answer. Non-streamed parsing already recognized more recipient forms.

## Technical Decisions

Boundary state limits abbreviated headers to the beginning of a turn and positions after Muse message terminators. Partial headers wait for their message delimiter; recipient and boundary-whitespace limits bound retention. Invalid `to=` prose is released without treating it as a tool call. Explicit role-bearing headers remain supported and Harmony retains its own channel isolation.

Separate Muse content, reasoning, and tool transitions share existing whitespace and separator machinery. Tool-message suppression survives an inner ATEM closing tag until a Muse boundary, preventing trailing tool-message text from becoming answer content. Non-streamed boundary parsing uses the same recipient validation and strips Muse terminators from cleaned output.

## Change Summary

The implementation commit `78da87c8` changes eight Rust files: shared envelope parsing and stream-state handling, plus filter/parity tests and Chat Completions, Responses, and Messages adapter tests. No dependencies or public API signatures change. [Issue #3025](https://github.com/lablup/mlxcel/issues/3025) records the acceptance criteria; the complementary client defense is tracked in backend.ai-go #7244.

## Validation and Limits

Local results: 502 tool-call tests, 14 Muse adapter tests, and 589 route tests passed; one route test was ignored. `cargo fmt --check` and strict clippy (`--lib --tests -- -D warnings`) passed. Route tests used `RUST_MIN_STACK=16777216`. Coverage includes Korean text, fragmented headers, multiple channels, tool extraction, ordinary `to=` prose, and overlong prefixes. Actual model inference and a desktop UI session were not rerun; these tests exercise deterministic generated-output fixtures.
