# Technical Report: PR #1503 - b10621 context retention, YaRN overrides, and the batching classifications

**Date**: 2026-08-30

**Status**: Completed

**Languages**: Rust

**Risk Level**: High

## Executive Summary

Closes the thirteen `deferred` entries #1450 left in `compat/llama-server/b10621/runtime-and-context.json` (#1472). The two substantive pieces are a YaRN arm on the shared RoPE path (making `--rope-scaling yarn` and the five `--yarn-*` knobs tune the rotation for real) and b10621's context-retention semantics (`--context-shift` off by default, `--keep`, per-request `n_keep` / `n_discard`), which deliberately replaces mlxcel's silent always-on KV front-trim. `--parallel` gains the `-1` auto default resolving to 4 slots as upstream's own auto does. Nine entries flip to `supported`, one to `aliased`, two to `not_applicable`; `--parallel` stays `deferred`, repointed at #1473 whose `--kv-unified` surface owns its one remaining divergence.

## 1. Problem Statement

PR #1464 landed the acceptance surface for these flags and recorded, entry by entry, that the behavior behind them did not exist: YaRN values were refused at startup, context retention was a fixed 4-token attention sink that trimmed silently and unconditionally, `--parallel -1` was rejected by a usize parser, and `--swa-full` / `--ubatch-size` / `--batch-size` had no terminal classification. Closing #1450 against that state would have left thirteen accepted-but-inert or absent surfaces claiming nothing.

## 2. Technical Decisions

### 2.1 YaRN as a frequency table plus a magnitude scalar

`RopeScalingKind::Yarn { freqs, mscale }` ports the in-tree deepseek_v2 / upstream `YarnRoPE` table math generalized with ggml's extrapolation mix, so `fast_rope_with_freqs` serves it like the llama3 table; the temperature correction pre-multiplies Q and K (skipped at 1.0, so every non-YaRN graph is byte-identical, proven by trace). b10621's own resolution order is mirrored deliberately, including its quirk that `--yarn-attn-factor` participates only at a zero extrapolation mix (`llama-context.cpp` recomputes it otherwise). Checkpoint-declared `yarn` blocks on the shared path now rotate with their declared scheme instead of warning and running unscaled, matching both b10621 and upstream mlx-lm's `initialize_rope`. Families off the shared seam keep their own YaRN and refuse a runtime override through `verify_applied` rather than ignoring it.

### 2.2 The KV bound becomes a contract instead of a silent trim

b10621's three-part contract is implemented literally: admission refuses a prompt at or over the per-slot bound (400 `exceed_context_size_error`), a bounded decode with shifting disabled stops with `truncated: true` / `stop_type: "limit"` (new `StopKind::ContextExhausted`), and `--context-shift` trims with the sequence's resolved retention (request `n_keep` over `--keep`, `-1` = whole prompt, +1 for a content-detected BOS, clamped to `bound - 4`; discard = `n_discard` or half the non-retained window, raised to the prefill-chunk overshoot). The arithmetic lives in pure helpers pinned by unit tests. The default-path change is called out as a migration note; `--context-shift --keep 4` reproduces the old rolling window. `resolve_server_max_tokens` stops clamping explicit budgets under shifting, since infinite generation is the flag's purpose.

### 2.3 Honest terminal states for the rest

`--parallel -1` resolves to 4 slots, exactly upstream's auto; the `kv_unified` half of that upstream line is recorded as the entry's one divergence and follows #1473. `--batch-size` is `aliased` onto `--prefill-chunk-size` with the default difference recorded rather than silently changing mlxcel's own 512. `--ubatch-size` is `not_applicable` (no physical micro-batch on unified memory). `--swa-full` is declared and refused at startup: the ring caches are model-owned, and the state operations the flag purchases upstream are gated on scheduler-owned caches, not ring size.

## 3. Change Summary

| Item | Value |
|------|-------|
| Files changed | 64 |
| Manifest entries | supported: --context-shift, --keep, field:n_keep, field:n_discard, the five --yarn-*; aliased: --batch-size; not_applicable: --ubatch-size, --swa-full; deferred at #1473: --parallel |

Validation: teacher-forced logit traces against `origin/main` (qwen3-0.6b-4bit no-flags and deepseek-v2-lite-4bit checkpoint-declared YaRN both byte-identical; sentinel arm byte-identical; forced yarn diverges 19.2% top-1 and `--yarn-beta-fast` 8-vs-32 separates the two yarn arms by 9.0%), plus a live server at `--ctx-size 512`: over-long prompt answers upstream's 400, default path stops at the bound with `truncated: true`, and `--context-shift --keep 32` generates 800 of 800 requested tokens through the 512-token window.

## 4. Follow-up Actions

- #1473 owns `--kv-unified` and, with it, `--parallel`'s repointed divergence.
- Context shifting stays a recorded no-op on Turbo-quantized KV layers (startup warning) and VLM sequences stay exempt, both documented on the manifest entry.
