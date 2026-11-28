# Technical Report: PR #2949 — Consolidate benchmark evidence

**Date**: 2026-11-11
**Status**: Implementation complete; merge pending
**Risk Level**: Low

## Executive Summary

Reduce `docs/benchmark_results` from 2,015 tracked files (14.4 MB) to 70 (1.04 MB). Consolidate 142 investigations into 12 version archives, preserving final results, minimal supporting data, negative decisions, and explicitly unmeasured cases.

## Problem Statement

Logs, intermediate captures, and repeated experimental reports obscured authoritative results. Deleting everything except successful final numbers would lose the evidence behind default-off policies and corrected premises. Some historical directories also contained executable tooling imported by a live benchmark runner.

## Technical Decisions

- Keep final reports and compact measurement/correctness data in the current tree; recover full historical evidence through immutable links to commit `2dc8ed47b30dc204d892328098e8cc34ec02ddc1` without rewriting history.
- Group investigation summaries by the first stable release containing their report, not by an assumed measured binary version. Preserve original provenance and later corrections.
- Move the unchanged hostgate implementation into `scripts/bench_hostgate.py` and test the runner import. Direct memory-soak output into ignored `bench-results/`.
- Convert four final JSONL datasets losslessly to CSV, retaining warmups and structured field values. Reject raw captures throughout the result tree and keep legacy exception lists empty.

## Change Summary

The result index separates final reports, unresolved validation gaps, and historical decisions. Repository references now target retained evidence, archive anchors, or frozen sources. Retired binary-asset exemptions are removed and regression-tested. Inference behavior is unchanged, and four pre-existing untracked CSVs remain outside this change.

## Validation and Limits

Before submission, 70 Python benchmark tests, raw-log and CSV-schema checks, binary-asset checks and regression tests, shell syntax validation, formatting, and Clippy passed. Preserved data was compared byte-for-byte, converted CSV fields were round-trip checked, and new links and archive anchors were validated. Real-model performance was not rerun because this change makes no inference modification. Existing native compiler warnings and pre-existing unrelated report links are outside scope. PR CI and merge state are recorded by GitHub rather than asserted here.
