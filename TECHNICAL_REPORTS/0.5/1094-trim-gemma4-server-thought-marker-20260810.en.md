# Technical Report: PR #1094 - Trim Gemma 4 Server Thought Marker

**Date**: 2026-08-10
**Status**: Completed
**Languages**: Rust
**Risk Level**: Medium

## Executive Summary

PR #1094 removes the Gemma 4 `thought` channel argument from server-side `reasoning_content` while preserving visible answer content and streaming/non-streaming parity. Review also corrected two position-accounting and malformed-stream edge cases exposed by consuming the longer `<|channel>thought` delimiter.

## 1. Problem Statement

The server stream filter previously entered reasoning mode after consuming only `<|channel>`, so the following `thought` argument was emitted as user-visible reasoning text. Simply replacing the delimiter with `<|channel>thought` exposed two adjacent risks: a decoded token split between delimiter bytes and emitted text could be counted twice in the parallel logprob queue, and a malformed bare `<|channel>` needed to remain suppressible without preempting the valid longer opener.

## 2. Technical Decisions

### 2.1 Prefer the full opener while retaining a deferred fallback

The delimiter table keeps both `<|channel>thought` and `<|channel>`, but the shorter match is deferred while buffered bytes may still complete the longer opener. At end-of-stream the ambiguity is resolved because no additional fragment can arrive, preventing truncated channel markup from leaking through `flush()`.

### 2.2 Track whether a fragment position was already counted

Each buffered decoded fragment is represented by a span containing its remaining byte length and whether its token position still needs to be counted. When a delimiter consumes only part of a fragment, the remainder is marked as already counted; later reasoning or content emission therefore cannot drain the same logprob position twice.

## 3. Change Summary

| Item | Value |
|------|-------|
| Files changed | 2 |
| Lines added | 167 |
| Lines deleted | 33 |
| Primary modules | `server::tool_calls::stream_filter`, `server::routes::chat` |

- Server reasoning now consumes the full Gemma 4 channel opener and no longer emits `thought` in `reasoning_content`.
- Streaming and non-streaming extraction reuse the same filter behavior; final visible content is unchanged.
- Fragment-span bookkeeping preserves token/logprob alignment when a delimiter ends inside a decoded token.
- Bare-channel fallback matching remains compatible with malformed output and is resolved safely during stream flush.
- Regression coverage exercises whole and split openers, delimiter position counts, cross-family delimiter behavior, and end-of-stream fallback handling.

## 4. Review Findings

| Finding | Severity | Resolution |
|---------|----------|------------|
| A split `thought\n` fragment could count one token position twice | Medium | Fixed with counted fragment spans |
| Removing the bare channel delimiter could expose malformed channel text | Medium | Fixed with deferred shorter-delimiter matching |
| An exactly truncated bare channel could leak during `flush()` | Low | Fixed by final delimiter resolution at end-of-stream |

No Critical or High findings remained after review. Delimiter matching remains bounded by the static delimiter table and the longest delimiter tail.

## 5. Validation

- `cargo test --lib server::tool_calls::stream_filter::tests`: 78 passed, 0 failed.
- `cargo test --lib gemma4`: 225 passed, 0 failed, 30 existing hardware-dependent tests ignored.
- `cargo test --lib reasoning_split_identical_whole_vs_chunked`: passed.
- `cargo test --lib extract_reasoning_gemma4`: passed.
- `cargo fmt --check`: passed.
- Hosted PR checks passed; `MLX pin extraction` was skipped because the PR does not change the MLX pin.

## 6. Related Work

- Issue #890: server-side remainder of the Gemma 4 reasoning marker fix.
- Issue #884: earlier CLI-side fix that established consuming `<|channel>thought` as one opener.

