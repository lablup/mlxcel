# Technical Report: PR #2590 — Dense-family batched decode

**Date:** 2026-10-25

**Related issues:** #2556, #2566, #2587

## Executive Summary

Twenty dense decoder families now share embedding, projection, normalization, feed-forward and output-head work across decode rows. Phi3-VL, Phi4-SigLIP-VL and GraniteVision delegate text decode to their converted towers. Metal performance acceptance remains tracked separately in #2587.

## Problem Statement

The previous trait default repeated each model's full forward for every concurrent sequence. A common batched shape is insufficient by itself: differences in residual order, normalization placement, rotary position handling, attention windows and logit scaling must match each single-sequence graph.

## Change Summary

Each family uses the shared split-attention and B=0/1/many routing helpers. B=1 retains the original forward. The converted families are Apertus, Cohere, Cohere2, ERNIE 4.5, ExaOne, Gemma, Gemma2, GLM4, Granite, Hunyuan v1 dense, InternLM2, InternLM3, MiMo, OLMo2, OLMo3, Phi3, Seed-OSS, SmolLM3, StarCoder2 and TeleChat3. The coverage ledger and continuous-batching documentation reflect the actual native and delegating paths.

## Technical Decisions

Phi3 partitions rows by short/long LongRoPE table, applies batched rotary operations per group and restores row order, preserving partial rotary scaling. Cohere preserves parallel attention/feed-forward residuals; Cohere2 distinguishes global NoPE attention from windowed layers. Gemma2 retains attention and output softcaps, Granite retains its multipliers, and InternLM retains sequence-dependent rotary parameters.

Review found that Hunyuan's learned Q/K normalization belongs after RoPE; the batched path was corrected and its fixture uses nonuniform norm gains to detect the ordering regression. OLMo3 preserves the code's actual legacy behavior: maskless one-token decode attends the full live cache, while an explicit shorter multi-token mask slices the window. This differs from the issue's simplified description; changing legacy semantics is outside this port.

## Validation

All twenty families have actual tiny-model runtime checks with three unequal prompts, B=1 exact equality and B=0. Three concrete vision wrapper objects execute the same delegation contracts without running vision encoders. Additional tests cover mixed LongRoPE tables, both OLMo3 mask paths and source registration. Twenty-eight focused tests, compilation and formatting passed before final review. Two additional negative tests now require family/row/layer context for missing cache entries and family/layer/root-cause context for invalid attention metadata. The full focused suite was rerun after this diagnostic change. The six-test complete-coverage gate passed on the initial implementation, and the final `d7c746df` tree passed full-workspace, all-target Clippy and the license-header check.

The owner approved local CI and review as the merge gate. Real-checkpoint output, startup capture and Metal throughput/logit measurements remain pending in #2587; native classification is not a measured speedup claim.

## Learning Points

Identity norm weights can conceal an incorrect operation order. Nonuniform learned gains and meaningful projection weights make parity fixtures sensitive to such changes. Runtime wrapper calls also provide stronger delegation evidence than source-text checks. Preserve measured or tested legacy behavior when issue prose has drifted, and record the discrepancy explicitly.
