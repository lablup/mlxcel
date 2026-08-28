# Technical Report: PR #1492 - Props, slots, metrics, health, and slot persistence

**Date**: 2026-08-28

**Status**: Completed

**Languages**: Rust

**Risk Level**: Medium

## Executive Summary

Aligns the b10621 observability surface (#1440): a per-request slot registry backs `GET /slots` and the native `id_slot` field, `POST /slots/:id_slot` serves save/restore/erase behind `--slot-save-path`, `GET /props` reports the b10621 key set with `--props` gating `POST /props`, `GET /metrics` exports the `llamacpp:` metric families, and `/health` answers exactly b10621's two bodies. Seven manifest entries flip to `supported`; four are implemented to an honest boundary and stay `deferred` with recorded divergences; `--sleep-idle-seconds` stays untouched, so issue #1440 remains open.

## 1. Problem Statement

mlxcel's `/props`, `/slots`, `/metrics`, and `/health` each diverged from b10621 in ways a monitoring stack notices: `/props` reported an mlxcel-shaped payload behind a flag upstream leaves ungated, `/slots` had no per-request slot identity (so `id_slot` was always `-1`), `/metrics` exported only `mlxcel_`-prefixed series that no llama-server scrape config matches, `/health` answered `503 no slot available` under load (restarting busy servers behind liveness probes), and slot persistence did not exist.

## 2. Technical Decisions

### 2.1 A slot registry at the HTTP boundary, not in the scheduler

The continuous-batching scheduler has no slot concept, and threading one through it would touch every worker type. Instead a `SlotRegistry` of `--parallel` slots lives in `AppState`: every generation route acquires the lowest free slot, updates it from callbacks it already has, and releases it on drop while keeping the last task's counters visible, which is b10621's `task_prev` behavior. An oversubscribed request stays unbound (`id_slot: -1`, upstream's own sentinel) and late-binds when a slot frees. Prompt/generated text is retained only under `LLAMA_SERVER_SLOTS_DEBUG` or `--slot-save-path`, so the default `/slots` can never leak content that was not asked to be retained.

### 2.2 Token-stream persistence, recorded as a divergence

b10621's slot save serializes KV state through `llama_state_seq_save_file`. mlxcel's KV lives in scheduler-owned MLX arrays the HTTP layer must not touch, so the save file carries the slot's token stream plus model id and a tokenizer fingerprint, with b10621's `fs_validate_filename` ported rule for rule, atomic tmp+rename writes, and canonical-path confinement refusing traversal and symlink escapes in both directions. A restore rehydrates tokens and the next request re-prefills (or adopts from the prompt cache); the entry stays `deferred` with that divergence recorded rather than claiming support.

### 2.3 Health is exactly upstream's two bodies

The former rich health payload moved to where b10621 reports the same data (`/slots`, `/metrics`); `/health` is now `200 {"status": "ok"}` once ready, under load included, and the upstream `503 Loading model` envelope before that. Saturation reporting moved to `GET /slots?fail_on_no_slot=1`.

## 3. Change Summary

| Item | Value |
|------|-------|
| Files changed | 38 |
| Lines | +2905 / -722 |
| Manifest entries | 7 flip to supported; 4 implemented but deferred; id_slot divergence cleared on POST /completion and POST /completions |

Validation: 40+ new unit/route tests (registry, persistence incl. symlink/traversal/mismatch refusals, gating diagnostics, auth, label-cardinality bound), compat gates green, and a real checkpoint (`qwen2.5-0.5b-4bit`) driving concurrent slots 0/1 with live counters, `fail_on_no_slot` 503 only while saturated, save/restore/erase across a server restart, and `llamacpp:` families scraping with the `Process-Start-Time-Unix` header.

## 4. Follow-up Actions

- `--sleep-idle-seconds` (idle sleep and wake-up lifecycle) stays on #1440; a truthful implementation needs worker-level model teardown/reload and should build on the #1438 model-pool lifecycle.
- The `params` key-subset divergence on `GET /props` / `GET /slots` and the KV-persistence divergence on the slot actions remain recorded on #1440.
