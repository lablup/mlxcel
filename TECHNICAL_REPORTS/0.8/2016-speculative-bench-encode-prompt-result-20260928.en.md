# Technical Report: PR #2016 - Error Instead of Panic When --prompt Fails to Tokenize

**Date**: 2026-09-28
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low

## Executive Summary

PR #2016 changes `encode_prompt` in `src/bin/speculative_bench.rs` from a panicking helper to one that returns `anyhow::Result<Vec<i32>>`, so a `--prompt` value the tokenizer rejects now produces a readable error naming `--prompt` instead of a panic backtrace. The fix mirrors the existing `tokenize_prompt` helper in the sibling `bench_decode.rs` binary and is the same class of defect as the `--target` unwrap panic fixed earlier in this binary (#1242).

## 1. Problem Statement

### 1.1 Background

`speculative_bench` is a benchmark harness comparing no-drafter baseline decode against MTP (multi-token prediction) speculative decode. Both paths tokenize the operator-supplied `--prompt` through `encode_prompt` before running inference.

### 1.2 Existing Issue

- **Panic on tokenizer rejection**: `encode_prompt` ended in `.expect("tokenizer.encode must succeed on a valid utf-8 prompt")`. Any prompt the tokenizer's `encode` call rejected aborted the whole process with a panic backtrace rather than a diagnostic naming the offending flag, even though both call sites (`run_baseline`, `run_mtp`) already returned `anyhow::Result` and could have propagated the error normally.

### 1.3 Risk Assessment

Low. `speculative_bench` is an internal benchmark binary, not part of the inference request path or a shipped CLI surface. Left unaddressed, the only cost was a confusing failure mode (panic instead of diagnostic) when an operator supplied a prompt the tokenizer could not encode.

## 2. Change Summary

| Item | Value |
|------|-------|
| Files changed | 1 |
| Lines added | +9 |
| Lines deleted | -9 |

- `encode_prompt(tokenizer: &MlxcelTokenizer, prompt: &str) -> Vec<i32>` becomes `-> Result<Vec<i32>>`, replacing `.expect(...)` with `.map_err(|err| anyhow::anyhow!("--prompt failed to tokenize: {err}"))?`.
- Both call sites, `run_baseline` (baseline path) and `run_mtp` (speculative path), add `?` to propagate the new `Result`; both functions already returned `Result`, so no further signature changes were needed.
- The doc comment above `encode_prompt` is updated to describe the new error path and to point at `tokenize_prompt` in `bench_decode.rs` (the pattern it now mirrors) instead of the previous, now-inaccurate reference to a test helper that still panics.
- The success path (BOS handling, `encode` call, `u32 -> i32` conversion) is unchanged.

## 3. Technical Decisions

### 3.1 Error message names the failing flag

**Context**: `tokenize_prompt` in `bench_decode.rs` wraps the same tokenizer error as `anyhow::anyhow!("tokenization failed: {err}")`, without naming which CLI flag supplied the rejected text.

**Rationale**: The issue's acceptance criteria explicitly required the error to name `--prompt`, since `speculative_bench` has more than one text-bearing input (draft/target model paths, prompt) and a bare "tokenization failed" message would still leave an operator guessing which argument was at fault. The message became `"--prompt failed to tokenize: {err}"`, keeping the underlying tokenizer error intact while adding the flag name.

**Trade-off**: This departs slightly from the literal wording of `bench_decode.rs`'s message but keeps the same `Result` + `map_err` + `anyhow!` shape, so the pattern is still recognizable as the same fix family.

## 4. Validation

- `cargo check --bin speculative_bench --features metal,accelerate` (capped at 2 build cores): clean, including an incremental re-run against the final file state (0.23s) to rule out a stale check from a background build that started before the last edit landed.
- `cargo clippy --bin speculative_bench --features metal,accelerate -- -D warnings`: clean (the only warning printed is a pre-existing, unrelated C++ `-Wunused-variable` from `mlxcel-core`'s build script, not gated by `-D warnings`).
- `cargo fmt --check`: clean.
- `python3 scripts/ci/check_cross_repo_refs.py`: clean.
- Independent `pr-reviewer` and `pr-security-checker` passes found zero CRITICAL/HIGH/MEDIUM findings; `pr-finalizer` confirmed no test or doc gap needed filling.
- Not run: a real benchmark against an on-disk model checkpoint. The GPU on this host is shared with concurrent work, and the change is confined to error handling in the tokenization step, not the inference path.
- No unit test added: none of the in-tree `MlxcelTokenizer` backends (HuggingFace, SentencePiece, Tiktoken) reject plain valid UTF-8 text, so there is no natural fixture to drive the new `Err` arm, and the sibling `tokenize_prompt` in `bench_decode.rs` that this PR mirrors carries no unit test either. The type change itself is verified by the compiler at both call sites.

## 5. Related Work

- Issue #1667: source issue for this change.
- Issue #1242: the prior `--target` unwrap panic fix in the same binary, the same class of defect.
- `src/bin/bench_decode.rs::tokenize_prompt`: the reference implementation this fix mirrors.
