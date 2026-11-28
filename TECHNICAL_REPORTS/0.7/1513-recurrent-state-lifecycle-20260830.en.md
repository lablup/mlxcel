# Technical Report: PR #1513 - recurrent-state-lifecycle

**Date**: 2026-08-30

**Status**: Completed with synthetic validation limits

**Languages**: Rust

**Risk Level**: High

## Executive Summary

PR #1513 fixes issue #1220 by moving RWKV7, RecurrentGemma, and KimiLinear onto the established model-owned `SequenceState` lifecycle. Before this change, each family recreated its recurrent or mixed recurrent-attention cache inside every `LanguageModel::forward()` call, so a normal greedy decode loop that feeds one token per step lost all prior hidden history after prefill.

The implementation keeps these families non-batched, uses per-`SequenceId` model-owned state for the server scheduler, preserves a single-stream fallback for CLI and benchmark paths, and resets that fallback only before fresh multi-token prefills. Scheduler calls with a `SequenceId` now require an existing prepared state slot and fail closed if the slot is missing. EOS ids are resolved from checkpoint sidecars/config through `read_eos_token_ids()` instead of hardcoded family constants.

Validation in this worker is deterministic and synthetic. It covers Rust compilation, state lifecycle behavior, EOS metadata parsing, formatting, and whitespace checks. It does not qualify real RWKV7, RecurrentGemma, or KimiLinear checkpoints because no pinned checkpoints or hardware target were provided for that run.

## 1. Problem Statement

The affected model families are recurrent or hybrid recurrent-attention architectures:

- RWKV7 stores token-shift, recurrent state, and FFN cache per layer.
- RecurrentGemma stores RGLRU recurrent state and rotating local-attention cache by layer.
- KimiLinear stores MLA attention cache and GatedDeltaNet convolution/SSM state by layer.

The generation contract is multi-call: prefill usually runs on the prompt, then each greedy decode step calls the model again with one token. If a family constructs fresh recurrent state inside each forward call, the decode step cannot see prefill history or previous generated tokens. That is a correctness bug, not only a performance issue.

## 2. Change Summary

| Area | Change |
|------|--------|
| RWKV7 | Replaced per-forward local cache allocation with `ModelOwnedSequenceState<Rwkv7Cache>`, added prepare/release/forward-by-sequence hooks, fallback reset, model-owned layout, and EOS metadata storage. |
| RecurrentGemma | Added `ModelOwnedSequenceState<GriffinLayerCache>` over mixed recurrent/attention caches, wired scheduler lifecycle hooks, fallback reset, model-owned layout, and EOS metadata storage. |
| KimiLinear | Added `ModelOwnedSequenceState<KimiLinearCache>` over mixed MLA/Delta caches, wired scheduler lifecycle hooks, fallback reset, model-owned layout, and EOS metadata storage. |
| Shared state helper | Added `with_existing_sequence_state()` so scheduler paths can reject missing prepared state instead of silently using fallback state. |
| EOS loading | Extended `read_eos_token_ids()` to prefer `generation_config.json`, then `tokenizer_config.json`, then `config.json`; tokenizer `eos_token` strings resolve through `added_tokens_decoder`. |
| Tests | Added focused deterministic coverage for EOS parsing, sequence-state continuity, per-sequence isolation, release, and missing-state rejection. |

### Statistics

| Item | Value |
|------|-------|
| Implementation files changed | 9 |
| Implementation diff | +819 / -102 |
| Primary implementation commit | `b1836a2a` `fix(models): persist recurrent decode state` |
| PR | #1513 |
| Issue | #1220 |

## 3. Technical Decisions

### 3.1 Use model-owned state instead of external KVCache state

The `LanguageModel` trait exposes homogeneous `KVCache` slices, but these families store heterogeneous recurrent state. RWKV7 has RWKV-specific layer state, RecurrentGemma mixes RGLRU and rotating KV cache, and KimiLinear mixes MLA and Delta caches. Matching the existing Mamba, Mamba2, Jamba, Plamo2, FalconH1, NemotronH, and Qwen3-Next pattern avoids forcing these shapes into incompatible external KV slots.

### 3.2 Keep fallback state for CLI paths, but reset it deliberately

Offline generation paths do not always carry a scheduler `SequenceId`. They now use the model-owned fallback slot. The fallback is reset when a fresh multi-token prefill reaches the model, and it is preserved for one-token decode steps. This keeps the existing single-stream API usable while preventing a new prompt from inheriting old hidden state.

### 3.3 Fail closed for missing scheduler state

The server scheduler allocates sequence state and calls `prepare_sequence_state()` before running prefill or decode. A `forward_with_sequence_id(Some(id), ...)` call that lacks a prepared slot indicates lifecycle corruption. PR #1513 therefore uses the new `with_existing_sequence_state()` helper on scheduler paths, producing a named failure instead of silently falling back to the legacy single-stream slot.

### 3.4 Source EOS from checkpoint metadata

RWKV7 and RecurrentGemma previously returned hardcoded EOS ids, and KimiLinear had the same default constant. The implementation now stores resolved EOS ids on the model. Direct config construction can provide `eos_token_id`; normal directory loads and special weight loads override that with `read_eos_token_ids()` when sidecar metadata is present.

## 4. Correctness Review

- `forward()` no longer creates fresh recurrent caches for the affected families.
- `make_caches()` resets only the fallback single-stream model-owned state and returns placeholder KV caches for trait compatibility.
- `prepare_sequence_state()` inserts a fresh per-sequence cache vector.
- `forward_with_sequence_id(Some(id), ...)` requires the prepared vector and updates it across calls.
- `release_sequence_state_by_id()` removes the per-sequence vector so cancellation and completion do not leak model-owned state.
- Cache cardinality is asserted before layer iteration, preventing `zip()` from silently skipping layers on malformed state.
- Scheduler inspection confirmed allocation still calls `prepare_sequence_state()` and completion/cancellation cleanup calls `release_sequence_state_by_id()`.

## 5. Security Review

No new external process execution, file deletion, credential handling, SQL, network request construction, or web rendering paths were introduced. The new sidecar reader only reads JSON files already inside the selected model directory. Invalid or missing EOS metadata returns an empty list or falls back to the next metadata source; it does not panic on malformed JSON.

The explicit missing-state failure is intentional: silently sharing fallback recurrent state across scheduler sequences would leak request-local generation history between users. Failing closed is safer than generating with corrupted hidden state.

## 6. Performance Review

The change removes repeated recurrent cache allocation from every decode step for RWKV7, RecurrentGemma, and KimiLinear. The persistent model-owned state does not add extra per-token cloning; it moves one vector out of the sequence map, mutates it, and reinserts it. The additional EOS sidecar checks run only during load/config resolution, not in the decode hot path.

The PR does not enable batching for these families. They remain `supports_batching() == false` and advertise `SequenceStateLayout::model_owned(...)`, so the scheduler should continue using single-sequence model-owned execution for them.

## 7. Validation Record

| Check | Result | Notes |
|-------|--------|-------|
| `cargo fmt --all --check` | Pass | Formatting check after implementation. |
| `cargo test --lib model_owned_sequence_state -- --nocapture` | Pass | 13 passed, 7309 filtered; includes new generic and family state-continuity filters. |
| `cargo test --lib read_eos_token_ids -- --nocapture` | Pass | 6 passed, 7316 filtered; covers generation/tokenizer/config EOS precedence and tokenizer `eos_token` resolution. |
| `cargo test --lib rwkv7_ -- --nocapture` | Pass | 4 passed, 7318 filtered; covers RWKV7 cache snapshot, EOS parsing, continuity, and missing-state rejection. |
| `cargo test --lib recurrent_gemma_ -- --nocapture` | Pass | 4 passed, 7318 filtered; covers RecurrentGemma EOS parsing, continuity, missing-state rejection, and existing windowed attention classification. |
| `cargo test --lib kimi_linear_ -- --nocapture` | Pass | 7 passed, 7315 filtered; covers existing KimiLinear guards plus new EOS parsing, continuity, and missing-state rejection. |
| `cargo test --lib parse_eos_token_ids -- --nocapture` | Pass | 3 passed, 7319 filtered; covers scalar/array/invalid EOS field parsing. |
| `git diff --check` | Pass | No whitespace errors. |
| Static old-pattern grep | Pass | No remaining fresh-per-forward cache allocation or hardcoded `[0]`, `[1]`, `[2]` EOS pattern in the three target model files. |

## 8. Validation Limits

- No real RWKV7 checkpoint generation was run.
- No real RecurrentGemma checkpoint generation was run.
- No real KimiLinear checkpoint generation was run.
- No hardware-specific MLX generation qualification was run.
- No broad workspace test, broad `cargo test --lib`, broad workspace clippy, serial all-tests, or release build was run, matching the unit constraints.

## 9. Follow-up Actions

- Run a pinned RWKV7 checkpoint through CLI greedy generation and verify multi-token output changes when prior context changes.
- Run a pinned RecurrentGemma checkpoint through CLI and server single-stream generation, including cancellation cleanup.
- Run a pinned KimiLinear checkpoint through CLI and server single-stream generation, including two concurrent queued requests to confirm state isolation.
- If these families later opt into batched decode, add explicit batched routing tests before changing `supports_batching()`.

## Appendix

- Issue: #1220
- PR: #1513
- Branch: `fix/issue-1220-recurrent-state-lifecycle`
