# PR #2472: Correct the sink attention Used by comment

**Date**: 2026-10-20
**Status**: Implemented; PR open
**Risk**: Low (six comment lines in the cxx bridge; no code change)

## Summary

Closes #1786. `fast_scaled_dot_product_attention_with_sinks` is declared three times (the `cxx::bridge` block in `src/lib/mlxcel-core/src/lib.rs`, `mlx_cxx_bridge.h` and `mlx_cxx_bridge.cpp`), and each declaration carried `Used by: GptOss`. Five families call it: GPT-OSS, Laguna, DeepSeek-V4, Falcon-OCR and, since #2460, OpenAI Privacy Filter. The same comment described a sink as a "per-head attention bias for first position", which is not what a sink is. All three comments now describe the sink correctly and list the five callers.

## Problem

`docs/code-guidelines.md` ("Shared Function Comments") makes the `// Used by:` line the record of which families a change to a shared function can break. With four of five callers missing from the list, a change to the sink SDPA path would have been tested against GPT-OSS alone. The description was also wrong in a way that misleads a reader about the arithmetic: a sink is not a bias on the first position's score. It is an extra per-head logit that joins the softmax denominator and has no value row, so it absorbs probability mass without contributing a value vector; MLX's fallback concatenates it in front of the scores, applies the softmax and slices it off.

## Technical decisions

- Use the issue's wording for the description and the caller list, and add `OpenAIPrivacyFilter`, which became a caller after the issue was filed (#2460). The spelling follows the `Used by:` lines already in `src/models/gpt_oss_common.rs`, so the family is named the same way everywhere.
- Leave the three `Used by: GptOss` lines on the GPT-OSS SwiGLU activation alone, as the issue asks: that function is used by GPT-OSS only at the `mlxcel-core` level (the privacy filter shares the clipped SwiGLU through `gpt_oss_common.rs`, which has its own `Used by:` line).
- Keep the change comment-only. No build was run; the C++ edits are inside `//` comments and the Rust edit is a `///` doc comment inside the bridge block.

## Changes

- `src/lib/mlxcel-core/src/lib.rs`, `src/lib/mlxcel-core/cpp/mlx_cxx_bridge.h`, `src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp`: the two comment lines above `fast_scaled_dot_product_attention_with_sinks` replaced at each site.

## Verification

- `cargo fmt --all -- --check` passes.
- `grep -rn 'fast_scaled_dot_product_attention_with_sinks(' src/models` lists `gpt_oss.rs`, `laguna_layers.rs`, `deepseek_v4_attention.rs`, `falcon_ocr.rs`, `openai_privacy_filter_layers.rs` and `openai_privacy_filter_tests.rs`; `examples/`, `tests/` and `benches/` have no call site.
- `git diff -U0` shows exactly six changed lines, all comments.

## Follow-ups

- The caller list went stale once between the issue being filed and this fix landing (#2460 added a caller). #1141 proposes enforcing the `// Used by:` convention in CI, staleness included; this PR is the kind of drift such a check would catch.

## Statistics

| Item | Value |
|---|---|
| Files changed | 3 |
| Lines added | 6 |
| Lines deleted | 6 |
| Tests added | 0 |

## Related

- Issue #1786: chore(core): correct the sink attention Used by comment (found during review of PR #1779).
- PR #2460: OpenAI Privacy Filter, the fifth caller.
- Issue #1141: chore: enforce the `// Used by:` convention in CI, staleness included.
