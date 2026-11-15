# Technical Report: PR #3080 - Project only the sampled row in TP and PP prefill

**Date**: 2026-11-15
**Status**: Completed (pipeline families with their own stage logic remain, see Follow-up)
**Risk Level**: Medium

## Executive Summary

The tensor-parallel runtimes and the pipeline-parallel final stage projected every prefill position through the LM head and then sliced the last row. TP Qwen 3.5 and TP Gemma 4 therefore lost the single-process savings from #3054 and the Gemma 4 override. All seven TP runtimes now slice the hidden state at the sampled row before the final norm, the head and the softcap. The pipeline coordinator sends that row's position to the final stage, so the shared transformer stage model projects only that row.

## 1. Problem Statement

The engine prefill calls `forward_last_logits*`, and the trait default computes the full `[B, L, vocab]` logits before slicing. At a 248K vocabulary that is about 1 GB per 2048-token piece, and about 14 GB in a 28K single pass, on the rank that holds the head (#3074).

## 3. Technical Decisions

- **TP:** each runtime's forward goes through one shared inner body that takes an optional last position. The plain, embedding and last-row entry points all reach it, so they cannot diverge. The head is replicated on rank 0 in these runtimes, so no sharded-vocab gather needed changing.
- **PP:** `last_pos` is a `#[serde(default)]` field on `RunEntry` and `ActivationMessage`. An older peer ignores it and answers every position, and the coordinator still slices, so mixed-version pipelines keep working. `StageExecutor::execute_last_row` defaults to a full forward plus a slice of the logits row. `TransformerStageModel` overrides it to slice before the head, which covers the Llama, Mistral, Mixtral, Qwen3, Llama4, DeepSeekV3 and GLM4 final stages.

## 7. Change Summary

| Item | Value |
|------|-------|
| Files changed | 23 (+1380 / -370) |
| TP tests | `distributed::tensor_parallel::llama_runtime`: 43 passed, including 7 new last-row tests at TP size 1 and 2 |
| PP tests | `distributed::pipeline`: 422 passed, including new stage-executor, worker and server-model tests |
| Real-model measurement | none (one GPU, no multi-node setup) |

## 8. Follow-up Actions

- Pipeline families whose final stage runs outside `TransformerStageModel` (Gemma3, Gemma4, Qwen3.5, Qwen3-Next, GptOss, GlmMoeDsa, Jamba, NemotronH) still project every position. The default hook only shrinks the frame sent back to the coordinator.
