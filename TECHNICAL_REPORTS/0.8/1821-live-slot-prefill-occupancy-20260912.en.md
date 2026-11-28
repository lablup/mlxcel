# Technical Report: PR #1821 - fix: track live slot occupancy during prefill

**Date**: 2026-09-12
**Author**: mlxcel maintainers
**Reviewer**: implementation review cycle (implementation review, security and performance review)
**Status**: Completed (the pinned b10621 executable was unavailable for direct comparison)
**Languages**: Rust, JSON, Markdown
**Risk Level**: Medium (cross-route observability and shared slot state; inference arithmetic is unchanged)

---

## Executive Summary

PR #1821 makes `GET /slots` reflect a request from scheduler admission through prefill, decode, and completion on every generation API. It forwards the scheduler's existing prefill progress events to route-owned slot handles, aligns live and retained counters with llama-server b10621, and reports a positive effective context window when `--ctx-size 0` delegates the limit to the checkpoint.

The review also removed repeated checkpoint configuration reads from the three metadata endpoints. The effective context window is now resolved once while constructing `AppState` and reused by `/slots`, `/props`, and `/v1/models`.

---

## Problem Statement

The slot registry originally bound requests only when a decoded text piece arrived. A long prefill therefore appeared idle, while a queued request could not be distinguished from a request already admitted by the scheduler. The registry also exposed final prompt and completion counts differently from b10621, and metadata returned `n_ctx: 0` even though generation had resolved a positive model-derived limit.

These gaps affected operators that use `/slots` for admission, saturation, or progress monitoring:

- `is_processing` stayed false during prefill.
- Prompt-cache and processed-token counters did not move while the prompt was evaluated.
- `n_prompt_tokens` did not become prompt plus accepted decode tokens during and after decode.
- `/slots`, `/props`, and `/v1/models` disagreed with the effective generation context under `--ctx-size 0`.

---

## Change Summary

- Added provider drain variants that forward every `GenerateEvent::Prefill` observation without changing token or logprob delivery.
- Wired chat completions, text completions, Responses, Anthropic messages, native completion, and the ASR streaming branch to the route-owned `SlotHandle` in both streaming and non-streaming execution.
- Bound a request on its first scheduler progress signal, leaving requests waiting in the scheduler queue unbound.
- Added `SlotHandle::on_prefill_progress` and updated live/final counters to use b10621 meanings. Request-derived values are clamped or combined with saturating arithmetic.
- Kept native `return_progress` framing independent from slot accounting: every observation updates the slot, while only requested progress frames are returned to the client.
- Resolved the effective per-slot context once at server state construction and reused it on all three metadata surfaces.
- Updated the b10621 compatibility manifests, environment-variable documentation, compatibility documentation, and focused route/state/provider tests.

The final PR contains 20 changed files, 503 additions, and 49 deletions across two commits.

---

## Technical Decisions

### Use scheduler progress as the binding boundary

Calling `SlotRegistry::begin` at HTTP admission would assign a slot to a request that might still be queued. Waiting for a decoded text fragment bound too late. `GenerateEvent::Prefill` is the first signal emitted after scheduler admission, so it provides the correct boundary without moving admission control into the observational registry.

The existing first-token update remains a fallback for a backend that produces no prefill observation. `SlotHandle` keeps RAII release semantics, so errors, cancellation, and dropped streaming tasks still return a bound slot to idle state.

### Forward progress through provider drains

The provider already owns the synchronous event-drain boundary used by streaming and non-streaming routes. Adding typed prefill observers there preserves event ordering: progress is applied before the first token and before the terminal result. It also avoids route-specific reads from scheduler internals.

### Cache effective context in `AppState`

The first implementation reused the correct checkpoint-resolution function but called it on every metadata request. Because `/slots` is commonly polled, that made an observability endpoint reread and parse `config.json` repeatedly. Startup resolution preserves the same fallback chain—explicit per-slot value, checkpoint context, then 4096—while reducing endpoint access to a field read.

---

## Technical Review

### Security and concurrency

- Prompt and generated text remain retained only under the existing debug or slot-save opt-in; the new observers store counters only by default.
- Slot updates mutate bounded server-owned state and do not accept paths, commands, or new deserialization formats.
- `SlotHandle` and `SlotRegistry` keep a consistent handle-then-registry lock order, and no new inverse acquisition was introduced.
- Cache and prompt totals are clamped to the prompt length, and additions use saturating arithmetic where externally influenced counts meet live totals.

No CRITICAL or HIGH security finding remained after review.

### Performance

Prefill observations perform constant-size counter updates under the existing slot mutex. The only actionable performance finding was repeated checkpoint configuration I/O for `--ctx-size 0`; commit `ac8162bf` resolves and stores the value once during `AppState` construction.

---

## Validation

- Full workspace gate on the primary implementation commit: `cargo test --workspace --profile test-fast --features metal,accelerate` exited successfully with zero failures.
- Final review commit:
  - `RUSTC_WRAPPER= cargo test --profile test-fast ctx_size_zero` — 3 matching route tests passed.
  - `RUSTC_WRAPPER= cargo clippy --all-targets --features metal,accelerate -- -D warnings` — passed.
  - `cargo fmt --check` and `git diff --check` — passed.
  - Final GitHub CI: crate versions, kernel dtype keys, license headers, compatibility manifest, cross-repository references, cargo-deny, cargo-fmt, cargo-clippy, and OpenXLA feature compile passed.
- Focused slot/provider/manifest suites passed, including prefill observation forwarding and queued-request isolation.
- Real checkpoint `models/gemma-3-1b-it-4bit`:
  - `/props`, `/v1/models`, and `/slots` reported `n_ctx: 1024`.
  - `/v1/completions` retained `n_prompt_tokens: 8` after a 6-token prompt and 2 decoded tokens.
  - Chat completions, completions, Responses, Anthropic messages, and native completion completed one-token streaming and non-streaming requests.

The exact pinned llama-server b10621 runtime measurement could not be repeated because that executable was not present locally; only b10883 artifacts were available. This limitation is recorded in the PR rather than substituting a different upstream version.

---

## Learning Points

- An observational slot should bind at the scheduler's first service event, not at HTTP receipt and not at the first visible text fragment.
- A single scheduler event can serve both protocol output and internal observability, but those consumers need independent gates. Native `return_progress` remains optional even though slot accounting is unconditional.
- Reusing the correct resolver is insufficient when it performs I/O. Values fixed for the lifetime of server state should be materialized at construction, especially for polled endpoints.

---

## Follow-ups

- Unified KV budgeting and auto-`--parallel` context geometry remain explicitly out of scope and are tracked by issue #1815.
- Re-run the long-prompt parallel 1/2 comparison if the pinned b10621 executable becomes available; do not use a later llama-server build as a silent substitute.

