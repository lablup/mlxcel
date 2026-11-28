# Technical Report: PR #2736 — Popular-model ROCm measurements

**Date**: 2026-10-30
**Status**: Completed
**Risk Level**: Low

## Executive Summary

Publish 45 raw text-decode measurements for 15 four-bit checkpoints on Radeon 8060S (gfx1151), measured from mlxcel commit `63b1ad9d`. Each model has three successful runs; every round passed the GPU contention guard.

## Problem Statement

The session-specific aggregate CSV added classification and round columns that differ from the repository's established raw-result schema. Publishing medians alone would discard repeat-level timing evidence.

## Change Summary

The published CSV preserves every measured field and uses the existing 20-column ROCm schema, ending in `decode_path`. Rows 1–15, 16–30 and 31–45 identify the three rounds, excluding the header. Adjacent JSON records row boundaries, rotated execution order, checkpoint revisions, binary hashes and guard results.

Measurements use 512 prompt tokens, 128 generated tokens, 20 warmup tokens, greedy decoding and batch size one. VLM checkpoints receive text-only inputs. Qwen3.6 MoE retains upstream 8-bit router exceptions; Gemma 3 indexes were repaired without changing weights.

## Validation and Limits

Verified exact equality with the original 45 rows, header compatibility, three runs per model, token counts, engine dispatch and unchanged binary hashes. Runtime code is unchanged. These measurements establish throughput for this host and workload, not generation quality, image performance or cross-runtime speedups. Original session logs remain local; the provenance JSON records hashes of the clean guard logs.
