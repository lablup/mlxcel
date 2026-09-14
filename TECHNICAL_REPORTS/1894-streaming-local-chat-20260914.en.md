# Technical report: PR #1894 — bounded local streaming chat

**Date:** 2026-09-14 · **Status:** Partial; integration and real acceptance pending · **Risk:** Medium · **Languages:** TypeScript/React, Rust

## Behavior and boundaries

Chat composes the published npmjs ui-common adapters and the approved shell. It uses the existing authenticated OpenAI streaming handler, snapshots opaque/inference identity and catalog revision per turn, retains partial cancelled/interrupted/error output, separates reasoning and non-executing tool calls, and labels server usage versus client-observed timing. No alternate generator, automatic model loading, tool execution or remote media acquisition is added.

The small escaped Markdown renderer blocks HTML and remote images, bounds displayed text/blocks and lazily highlights completed local code blocks. Image admission reads actual server limits and applies separate lower browser allocation/request ceilings. Bootstrap now projects the real mode-specific inference body budget: the main application's configured image JSON budget, limited by the router dispatch buffer in pool mode. The 2 MiB administrative budget is deliberately not mistaken for the inference limit. Complete serialized UTF-8 request bytes include transcript, base64 and JSON overhead.

History is memory-only unless explicitly enabled. The strict versioned IndexedDB adapter bounds conversations, turns, bytes and separately consented image persistence, rejects unknown fields/import versions, preserves interrupted restored turns, handles quota errors and supports Clear All. Imports revalidate image headers/dimensions before any preview or replay. Clear/import/load callbacks are fenced against unmount and replacement; mutable request work cannot resurrect cleared sessions. Immutable historical turn byte sizes are cached to avoid serializing an entire origin's history every streaming frame.

## Review corrections

Independent reviews identified and corrected late history operations, imported-image allocation bypass, pending attachment navigation, ineffective transcript memoization, missing Clipboard API handling, false imported completion, and disabled image admission accidentally blocking text-only chat. A provider-level abort test exposed a separate reader race: cancelling the reader could resolve a pending read before the abort rejection. The shared client now rechecks the abort signal after the race, preventing cancelled transport from appearing as successful EOF.

Provider selection-only cancellation and submission auth fences are owned by the earlier Settings/Activity PRs. Their duplicate development hunks were explicitly transferred, not stacked into this issue. Final rebase must wire the canonical Settings defaults and local request overrides before acceptance; this checkpoint still uses server defaults. Both independent reviewers retain the selection integration and actual-model validation gates. Manual Safari/VoiceOver/native 200% checks are user-deferred final epic gates, not a repeated per-unit approval request.

## Executed evidence

- Initial published implementation: 136 frontend unit tests, TypeScript and ESLint passed. Subsequent review-fix targeted tests passed; the final updated aggregate is recorded in the PR handoff.
- Two CPU-only configured media-limit projection tests and one mounted whole-bootstrap fixture test passed. The projection covers asymmetric image dimensions and distinct router/single body budgets.
- Scoped `cargo clippy --lib --tests --features metal,accelerate -- -D warnings` passed; the native dependency build emits its existing unused HIP source warning, not a Rust lint failure.
- Forty-six strict contract fixtures, generated DTO drift, schema negatives, compatibility, crate versions and kernel dtype-key checks passed. Deterministic bundled assets stayed within the existing size budgets and include a local lazy highlighter chunk.
- No local browser renderer, broad Rust suite or GPU checkpoint inference was executed by this unit. Hosted browser fixtures and an explicitly configured isolated production-browser harness are provided; absent real configuration fails instead of reporting a skipped pass.

## Remaining root-owned acceptance

Rebase after #1846/#1847/#1844, connect canonical request defaults and observation-only selection, and repeat final lint/unit/contract/bundle checks. Run hosted light/dark/compact accessibility and 10,000-token-sized fixture rendering; inspect any new actual screenshots without regenerating baselines blindly. Run the documented real browser harness on dense and hybrid/MoE models and a provider-confirmed VLM image, inspect response quality, and verify isolated backend request accounting returns to zero after Stop. These are pending, not replaced by fixture tests or compile success.
