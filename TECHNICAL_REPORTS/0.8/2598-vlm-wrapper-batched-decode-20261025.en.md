# Technical Report: PR #2598 — VLM wrapper batched decode

**Date:** 2026-10-25

**Related issues:** #2564, #2566, #2567, #2587

## Executive Summary

Nine VLM wrappers now reach native text decode: Molmo, Molmo2, MolmoPoint, Moondream2, Moondream3, Jina VLM, Mllama, dots.ocr and MiniCPM-o. Six own text towers use shared split attention, Mllama batches its existing self-attention blocks when image state is absent, and the two delegation-only wrappers reuse their already-native towers. Weight-heavy operations execute over the decode batch while attention retains each request's cache history.

B=1 uses the original single-sequence graph and B=0 returns the shared empty result. Mllama with model-global image state retains its historical per-row behavior; this does not provide request isolation. That correctness work remains in #2567. Metal and real-checkpoint acceptance remain in #2587.

## Problem Statement

These wrappers inherited per-row decode or delegated to towers that did. Increasing scheduler concurrency therefore repeated embedding, projection, normalization, MLP and output-head work for every row. A shared transformer adapter alone would be insufficient: these towers have distinct normalization and rotary rules, MolmoPoint consumes both pre-normalization and normalized hidden states, and Moondream3 routes tokens through experts.

## Technical Decisions

The implementation reuses `SplitAttention`, `forward_split_attention`, cache diagnostics and `route_batched_decode`. Each tower keeps its original normalization order, rotary parameters and attention scale; row-local attention is the boundary between batched weight-heavy operations. No new per-row model loop is introduced except Mllama's explicitly preserved image-state fallback.

MolmoPoint retains `forward_with_pre_ln` semantics and its `ExtendedLmHead`, including the extra-vocabulary path. Moondream3 preserves its temperature and expert computation while processing batch-shaped routed tokens. Actual expert-selection assertions distinguish same-expert and disjoint-expert fixtures before parity checks.

Mllama's no-image cross-attention layers are identities. Its batched path advances self-attention cache slots at their original layer indices and leaves cross-attention dummy slots unchanged. With image state present, the wrapper calls its historical single-row forward per row. Tests use nonzero gates and image features and assert that image state actually changes logits, preventing an inactive fixture from falsely validating the fallback.

DotsOCR and MiniCPM-o delegate to existing native Qwen2/Llama3 and Qwen3 entry points. Tests call real wrapper instances, so wrapper forwarding, decode markers and B=1 routing are covered rather than inferred from tower tests alone.

## Validation

Final integrated CPU library batch-decode tests passed: 129 passed, 1 pre-existing ignored Gemma4 test, 0 failed. Coverage classification/source guards passed 11 tests. Focused fixtures include six tower families, MolmoPoint pre/post-normalization and extended-head checks, Moondream3 same/disjoint expert cases, eight concrete non-Mllama wrappers and three Mllama tower/wrapper state tests. Shared fixtures exercise three unequal prompt histories, exact B=1 and empty B=0 behavior.

Python tooling passed 139 tests. Workspace version, environment-variable documentation, Llama compatibility, license-header, kernel dtype-key, kernel-port dispatch, ROCm overlay-record, binary-asset and Windows-path gates passed. Full-workspace, all-target Clippy with warnings denied, formatting and diff checks passed.

Final correctness, security and performance review was clean after correcting the coverage ledger to classify Mllama's wrapper rather than the non-`LanguageModel` text tower, and to follow its reused Llama block instead of requiring a direct split-attention call in the Mllama file.

No Metal checkpoint trace, greedy generation baseline comparison, serving ladder or measured speedup is claimed. All such acceptance remains in #2587; model-global image-state isolation remains in #2567.

## Learning Points

Coverage classification must follow actual trait ownership. A tower without a `LanguageModel` implementation is exercised through its wrapper, and a source guard must recognize reuse rather than force duplicate code. Runtime fixtures with active image gates and explicit expert-route assertions make otherwise vacuous parity tests meaningful.

## Change Summary

The PR adds native decode to the remaining available wrapper/tower group, registers meaningful synthetic runtime fixtures, updates the coverage ledger and documents native versus conditional fallback behavior. The implementation preserves single-row entry points and the existing image-prefill interface. Implementation commit: `788f2689`.
