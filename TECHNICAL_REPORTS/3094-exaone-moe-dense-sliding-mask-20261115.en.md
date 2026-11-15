# Technical Report: PR #3094 - Size the EXAONE-MoE sliding mask for dense caches

**Date**: 2026-11-15
**Status**: Completed
**Risk Level**: Low

## Executive Summary

`forward_standard_caches` in `src/models/exaone_moe.rs` runs on dense `KVCache`s, but it built the sliding-layer mask with the rotating-cache helper, which sizes the mask to the window. Once a chunked continuation starts at or past the sliding window, the dense cache holds more keys than the mask covers, and MLX aborts the process on `broadcast_shapes` (for example `Shapes (6,9) and (1,2,6,11)`). The path now uses `create_sliding_window_prefill_mask_dense`. A regression test compares chunked continuations at offsets 4, 5, 6 and 8 with a single-pass prefill (#3093).

## 1. Problem Statement

#3090 found an abort, not a returned error, so it takes down the whole process, which in the server means every request. With the fix reverted, the new test reproduces it as SIGABRT.

## 3. Technical Decisions

The dense helper keeps every key and applies the window, which matches the cache type this path uses. An audit of the other users of the rotating mask helper (mellum, exaone4, gpt_oss, ministral3, step3p5, afmoe, mimo_v2_flash, gemma3, deepseek_v4, and tensor-parallel gemma3) found that each pairs it with its own model-owned rotating cache, so no other model changes.

## 7. Change Summary

| Item | Value |
|------|-------|
| Files changed | 3 (`exaone_moe.rs`, `exaone_moe_tests.rs`, `CHANGELOG.md`) |
| Test | `exaone_moe_chunked_continuation_past_the_window_matches_single_pass` (window 4, 11 tokens, four splits, 1e-4 against single pass): aborts without the fix, passes with it |
| Suites | `exaone_moe` and `last_row` tests pass (168) |
