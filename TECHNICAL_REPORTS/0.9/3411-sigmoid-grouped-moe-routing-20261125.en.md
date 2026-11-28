# PR #3411: Sigmoid grouped MoE routing

**Date**: 2026-11-25
**Status**: Validated for #3345; real 30B acceptance belongs to #3350.

## Executive Summary

Nemotron-H E blocks now execute grouped sigmoid routing and relu2 routed/shared experts through the XLA backend. Checkpoint validation and production GPU lowering were corrected after end-to-end tests exposed failures that primitive numerical tests did not cover.

## Problem Statement

The existing softmax/SwiGLU router could not represent Nemotron-H MoE. Accepting E schedules alone left the recurrent checkpoint validator requiring dense up/down weights, and tuple-reduction argmax failed compilation when fused into the complete CUDA graph.

## Technical Decisions

Selection uses sigmoid scores plus correction bias, while combination uses the uncorrected scores, optional normalization with epsilon, and routed scaling. Group selection sums its two strongest scores; exact ties prefer the lower index. Single-result reductions preserve first-NaN/lower-index behavior without the failing production tuple reduction. Packed routed/shared tensors require correctly shaped scale and bias companions; impossible group capacity, zero shared widths, and unsupported latent projections are rejected.

## Change Summary

Configuration, tensor schemas, expert loading, grouped routing, and E-bearing integration fixtures were extended together. Existing softmax MoE goldens remain byte-identical. The tokenizer fixture reserves EOS 31 and uses an untied output head; its native HF greedy reference matches all 12 emitted IDs.

## Validation and Boundaries

Runtime source `8466cabbfb596a15dfe7a2c912ed3d32670b54ae` passed 543 library tests (21 ignored), seven local-task oracle cases, CUDA hybrid 5/5, and CUDA CLI/HF 12/12. Head `bf8188d20177cde215ca1b54c6444b86396bd8bd` adds only the test-switch allowlist; its guard and 15 focused Python tests pass.

External ROCm acceptance at bf8188d2 reports seven CPU/HIP primitive cases, hybrid 5/5 on each device, CLI/HF 12/12, B4/N8, cancellation/recycling, and four concurrent HTTP text comparisons passing. Real Nemotron-H-4B oracle/CLI 40/40 and batch, plus Mamba2 synthetic/real regressions, also pass. HTTP evidence establishes rendered-text parity, not exposed HTTP token IDs. Evidence: https://github.com/lablup/mlxcel/pull/3411#issuecomment-7147301101.

Hosted CI was intentionally canceled under the batch policy; this report does not claim hosted CI success. Optimized test-fast evidence establishes correctness, not throughput. Whole-f32 MoE upload and dense all-expert computation remain constraints for #3350; no 30B capacity or active-expert performance claim is made.

## Learning Points

Primitive oracle success does not establish checkpoint loading, compiler fusion, or serving acceptance. Keep independent whole-model references and exact source hashes alongside numerical probes.
