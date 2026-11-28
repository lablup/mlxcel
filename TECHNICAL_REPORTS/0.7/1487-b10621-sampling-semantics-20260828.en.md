# Technical Report: PR #1487 - b10621 sampling semantics and windows

**Date**: 2026-08-28

**Status**: Completed

**Languages**: Rust

**Risk Level**: High

## Executive Summary

Delivers the b10621 sampling-semantics gaps owned by issue #1436: the `repeat_last_n` penalty window finally reaches the sampler, the DRY sentinels are reconciled with upstream, server-wide and native surfaces land for top-n-sigma, XTC, ignore-eos, and reverse-prompt, and the sampler-order flags validate against the fixed b10621 default chain. Of the 61 manifest entries the issue owned, 26 flip to `supported`, 3 to `not_applicable` with tested rejection, and 32 repoint to the new remainder issue #1485, which pin.json now lists as a shard co-owner.

## 1. Problem Statement

The `--repeat-last-n` flag was parsed and echoed by `/props` but every repetition, frequency, and presence penalty scanned the entire history; DRY read `0` as "full scan" where b10621 disables at `0`; a repeat of exactly `dry_allowed_length` was never penalized (a `>` where upstream uses `>=`); and a dozen b10621 sampling surfaces (top-n-sigma and XTC server flags, native XTC and ignore-eos fields, sampler ordering, reverse-prompt) had no mlxcel spelling at all, leaving deferred manifest entries that block #1436 from closing.

## 2. Technical Decisions

### 2.1 Windowing in Rust ahead of the fused dispatch

`SamplingConfig::penalty_last_n` (`-1` full history and the default, `0` disables the stage, `N > 0` a window) slices the history handed to the existing penalty functions. The incremental `SamplerState` serves only the full-history form, byte-identically; a positive window takes the bounded rebuild path (the b10621 default window is 64 tokens). `needs_token_history` gates on the windows, so a zero-window config stays fused-batch eligible. The server resolves the previously parse-only `repetition_context_size` (alias `repeat_last_n`) or falls back to the server default of 64, which is a deliberate, documented behavior change for penalized server requests that omit the field.

### 2.2 Upstream-verified domains, corrected by review

A review against the pinned b10621 source corrected three initial misreadings: request stop strings REPLACE the server-wide `--reverse-prompt` list when non-empty (upstream falls back to the CLI antiprompt list only when the request has no effective stops); the XTC schema limits are SOFT, so out-of-range values clamp instead of returning 400; and b10621's CLI rejects negative `--dry-penalty-last-n` outright, so both server binaries now reject at parse time while the offline generate CLI keeps `-1` as a documented mlxcel-only full-history spelling. `dry_base` is sanitized at startup and both DRY implementations mirror upstream's below-1.0 early-out and exponent cap.

### 2.3 ignore-eos as an EOG bias, honestly scoped

`--ignore-eos` / `ignore_eos` suppresses every merged end-of-generation token with a `-inf` bias through the shared token-bias map at enqueue time, exactly upstream's mechanism. The OpenXLA worker has no token-bias path and rejects such requests with a diagnostic; the disaggregated handoff drops token bias and records the gap beside the existing XTC note; both manifest entries carry the scope qualifier.

### 2.4 Fixed sampler order as validated inertness

mlxcel's chain position is pinned by #1375/#1377 to exactly b10621's default order, so `--samplers`, `--sampler-seq` / `--sampling-seq`, and the native `samplers` field accept precisely that order (an inert configuration, both upstream shapes) and reject anything else with the accepted form in the diagnostic, before the model load.

## 3. Change Summary

| Item | Value |
|------|-------|
| Files changed | 46 (+15 in the review follow-up) |
| Lines | +1335 / -311 across both commits |
| Manifest entries | 26 supported, 3 not_applicable, 32 repointed to #1485 |

Validation: 98 core sampling tests (window sentinels, covering-window byte identity, zero-window inertness plus fused eligibility, DRY sentinels and the `base^0` tier), 280 xla parity tests, compat integration against both rebuilt binaries, and real-checkpoint arms on Qwen3-4B-4bit under greedy decoding: full-history versus covering-window byte-identical, window-with-penalties-off byte-identical to baseline, DRY `0` byte-identical to DRY-off, and the window-64 treatment arm diverging from full history first at word 76 of 164, exactly where the history first exceeds the window. The traced forward path is untouched.

## 4. Follow-up Actions

- #1485 owns the remainder: Mirostat, dynatemp, adaptive-p, grammar surfaces, logit-bias, dry-sequence-breaker strings, min_keep, n_probs, post_sampling_probs, backend_sampling, the temp/top-k/top-p default-resolution divergence, and the seed below-minus-one divergence.
- The disaggregated wire frame still drops token bias (XTC, ignore-eos) and a mixed-version handoff reads an old `dry_penalty_last_n: 0` frame as disabled; both recorded.
