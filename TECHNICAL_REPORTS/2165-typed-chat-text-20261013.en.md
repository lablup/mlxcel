# Technical Report: PR #2165 - Typed Chat Text

**Date**: 2026-10-13
**Status**: Ready for review
**Languages**: Rust
**Risk Level**: Medium

## Executive Summary

Checkpoint-selected chat templates that accept typed text parts but silently discard string content are now detected once during model loading. Typed `ChatMessage` renders use a borrowed text-part view only for those templates, while raw JSON rendering and string-compatible templates retain their existing contracts.

## 1. Problem Statement

The Granite Vision 3.2 checkpoint ships a standalone multimodal template that is correctly preferred for image requests, but that template reads text from typed content parts. CLI and server text-only paths supplied strings, so the question was omitted from the prompt: the real checkpoint produced a 47-token prompt and an unrelated greeting instead of answering the question.

Falling back to the tokenizer configuration's string template would split image and text behavior across templates and would make CLI, server, and prompt-cache history rendering disagree.

## 3. Technical Decisions

### Detect compatibility once at selection time

After selecting the checkpoint template, the loader renders one string probe and one typed-text probe through the unchanged raw renderer. It stores `wrap_string_content = true` only when the typed probe preserves the marker and the string probe does not; either render error leaves adaptation disabled.

### Adapt only typed messages without copying text

`apply_inner` serializes a small borrowed wrapper containing the original role and a single typed text part. The wrapper holds `&str` references and is passed directly to MiniJinja, avoiding an intermediate JSON tree and duplicate string allocation. Exhaustive `ChatMessage` destructuring makes a future field addition fail compilation until the rendering view preserves it.

### Preserve existing contracts

`apply_raw*`, explicit `with_template` processors, the default processor, string-compatible templates, typed image ordering, and probe-error behavior remain unchanged. The fix does not alter CLI or server call sites and adds no dependency.

## 7. Change Summary

| Category | Files | Summary |
|---|---:|---|
| Production | 1 | Selection-time compatibility flag and borrowed `apply_inner` rendering view |
| Regression tests | 1 | Eight tests for exact Granite prompts, roles, empty text, raw behavior, compatible templates, constructors, and probe errors |
| Pinned fixtures | 5 | Granite Vision 3.2 and Granite 4 templates with source provenance |
| Reports | 2 | English and Korean implementation records |

## Appendix

### Test Results

- `cargo test --locked -p mlxcel -p mlxcel-surgery --lib --profile test-fast --features cuda -j4 text_wrap_tests -- --test-threads=1`: 8 passed, 0 failed; 9,105 unrelated mlxcel tests and 160 surgery tests were filtered out.
- Repository contract gate passed: workspace versions, kernel dtype keys, kernel port dispatch, llama compatibility, ROCm overlay checks, 123 Python tooling tests, and 49 WebUI contract fixtures.
- On NVIDIA GB10/CUDA with `mlx-community/granite-vision-3.2-2b-4bit` revision `351831ab5218b1f66bbfdfd57bd8f20d81ce8baf`, CLI answered “The capital of France is Paris.” in 8 tokens. Server string and typed-text requests returned the same answer with 55 prompt tokens and 8 completion tokens.
- The frozen baseline returned an unrelated 20-token greeting for both server shapes with 47 prompt tokens, reproducing the defect.
- Implementation review, security rereview, and static finalization found no remaining issues.

### Provenance

- Frozen source diff SHA-256: `1d088cbdc926b50a1af5fc8810435dd2cccb3b846bf5477a00b363ea6d9ceaa4`.
- Candidate binary SHA-256: CLI `706fcf0864c6adbd251e2725db1f623044b49e6a33436617925dc9b1da4d09f8`; server `496534a54f6ea92ceaea5e3cfaec20237b11f753bd5fef114927208607a2998c`.
- Baseline binary SHA-256: CLI `592c33a141ccd7a497732e132ac6da302661e22400e27fbcdb2cb3ca858ae8ff`; server `bc076e4d0b2fde769eb15ababe241e72828c5f9c0b3ec2142137cef688c5a656`.

### Validation Limits

The issue unit ran focused CUDA tests and one real checkpoint on Linux/GB10. It did not run the full workspace test or Clippy suites, ROCm inference, macOS inference, or additional model families; the parent integration gate and hosted CI own the broad checks.
