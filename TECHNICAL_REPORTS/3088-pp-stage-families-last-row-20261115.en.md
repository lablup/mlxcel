# Technical Report: PR #3088 - Slice the sampled row before the head in the remaining final stages

**Date**: 2026-11-15
**Status**: Completed
**Risk Level**: Low

## Executive Summary

#3080 taught the pipeline coordinator to send the sampled row's position to the final stage. It also made `TransformerStageModel` slice before the LM head. Eight families run their final stage outside that type and still projected every position before slicing: Gemma3, Gemma4, Qwen3.5, Qwen3-Next, GptOss, GlmMoeDsa (DeepSeekV32), Jamba and NemotronH. Each now overrides `execute_last_row` and slices the hidden state at the sampled row before the final norm, the head and any softcap (#3081).

## 1. Problem Statement

The `StageExecutor::execute_last_row` default runs the full forward and slices the logits row afterwards. That shrinks the frame sent back to the coordinator, but not the head's `[B, L, vocab]` transient. At a 248K to 262K vocabulary, that transient is about 1 GB per 2048-token piece.

## 3. Technical Decisions

- **Gemma3, Gemma4, Qwen3.5, Qwen3-Next, GptOss and DeepSeekV32:** each stage model takes an optional `last_pos` and slices with `logits_at_position`. Gemma4's final softcap therefore runs on one row.
- **Jamba and NemotronH:** they gain a `forward_stage_last_row`. `forward_stage` becomes a one-line wrapper around it, so the in-process forward and the `LanguageModel` hooks are untouched.
- **Non-final stages:** they ignore `last_pos`, and `None` keeps every position, so older peers and the reverse path behave as before.

## 7. Change Summary

| Item | Value |
|------|-------|
| Files changed | 27 (+852 / -75) |
| New tests | one stage-executor test per family (synthetic weights, three positions against the full forward's row, plus a non-final stage keeping its full hidden state) |
| Suites | `distributed::pipeline` and the Gemma stage-parity tests: 441 passed |
| Real-model measurement | none (no multi-node setup) |

## 8. Follow-up Actions

- The tests check the output row, not the memory path, because the override and the default produce the same row. The NemotronH fixture has no Mamba layers. The saving is unmeasured on a real checkpoint.
