# Technical Report: PR #3090 - Project only the sampled prefill row in every model

**Date**: 2026-11-15
**Status**: Completed
**Risk Level**: High (218 files, every autoregressive model)

## Executive Summary

Most models ran the trait default for prefill: the full forward, then a slice of one row out of `[B, L, vocab]` logits. Any final softcap or scale added a second copy of the same size. A new opt-in `LanguageModel::last_row_logits` hook lets a model project only the sampled row, and the three `forward_last_logits*` defaults use it whenever a model offers it. Every autoregressive `LanguageModel` in `src/models` and `src/vision` now takes the hook on the text path, and on the embeddings path where one exists. That covers VLM wrappers and the Qwen-VL mrope/deepstack paths. On an M5 Max at 28K tokens, the single-pass peak drops from 38.17 to 23.09 GB on gemma-3n-e2b, 25.13 to 16.74 GB on qwen2.5-vl-3b, 30.61 to 22.55 GB on gpt-oss-20b and 15.70 to 9.77 GB on exaone-4.0-1.2b. Greedy output is byte-identical in all ten measured cells (#3070).

## 1. Problem Statement

The engine prefill reaches `try_forward_last_logits_with_sequence_id` (text) or `forward_last_logits_with_embeddings_and_sequence_id` (VLM). Only a dozen families overrode them. For everyone else the LM head ran over every position: about 1 GB per default 2048-token piece at a 250K-entry vocabulary, and tens of GB in a single pass. VLM wrappers also fell back to the default even when their inner text model had an override.

## 3. Technical Decisions

- **One hook.** A model implements a single `last_row_logits` instead of three overrides. The default declines, so nothing changes for a model that does not opt in. `select_row` threads an optional row through an existing forward body, and `last_row_logits_from` serves wrappers that only delegate. `&M`, `LoadedModel` and `VisionLanguageModel` forward the hook.
- **The full forward keeps its op order.** Layers, recurrent state, KV writes, mrope and deepstack still run over every row. Only the final norm, the head, and any softcap or scale run on the sampled row. Output therefore does not move, and no logit trace was needed.
- **A roster guard keeps coverage from drifting.** `last_row_logits_coverage_tests.rs` scans every `impl LanguageModel` and fails on one that neither takes the hook nor is listed. Four are listed on purpose: DiffusionGemma and Llada2Moe (their denoising loop reads every position), and Florence2 and NemotronParse (encoder-decoder runtimes).

## 7. Change Summary

| model | mode | peak, main | peak, branch | output |
|---|---|---|---|---|
| exaone-4.0-1.2b-4bit | 2048 / single | 5.98 / 15.70 GB | 5.10 / 9.77 GB | identical |
| gemma-3n-e2b-it-4bit | 2048 / single | 9.75 / 38.17 GB | 7.67 / 23.09 GB | identical |
| gpt-oss-20b-mxfp4-q4 | 2048 / single | 15.88 / 30.61 GB | 13.79 / 22.55 GB | identical |
| qwen2.5-vl-3b-instruct-4bit (text) | 2048 / single | 6.55 / 25.13 GB | 5.00 / 16.74 GB | identical |
| kimi-vl-a3b-thinking-4bit (text) | 2048 / single | 14.43 / 32.04 GB | 12.84 / 25.57 GB | identical |

Tests:
- 161 last-row tests: one tiny random-weight case per converted architecture, plus VLM wrapper and Qwen-VL / GLM-4V image-state cases, and the roster.
- The full `mlxcel` lib suite passes serially, 11,949 tests.

Details: `docs/benchmark_results/last-row-prefill-logits-m5max-2026-11-15.md`.

## 8. Follow-up Actions

- A dozen VLM wrappers have no unit case, because no tiny vision fixture exists for them. Their hooks are thin calls into tested text paths.
- An EXAONE-MoE chunked continuation that starts at or past the sliding window aborts in MLX (`broadcast_shapes`) on the dense-cache path. It was found during the work and not yet confirmed on main.
