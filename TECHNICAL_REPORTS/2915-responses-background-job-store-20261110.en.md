# PR #2915: Background response jobs in ResponsesStore

**Date**: 2026-11-10
**Status**: Completed (PR open as a draft; closes #2897, part of epic #2903)

## Executive Summary

PR #2915 is the storage foundation for OpenAI Responses API background mode (`background: true`). Before it, `ResponsesStore` held a response only after generation ended, and the only in-flight state was a cancel token. A background request has to be pollable from the moment it is created, so the store now keeps active jobs in a second map, `jobs`, under its own lock and outside the TTL and LRU entries. When a job ends, `finish_background` moves it into the normal entry map in one critical section. A `store: false` job leaves an *ephemeral* terminal entry that polling can read for a bounded time but `previous_response_id` cannot. Two flags bound the feature: `--responses-background-max-jobs` (default 64) and `--responses-background-ephemeral-ttl-secs` (default 600, at least 1). The PR adds no HTTP behavior; #2898 wires the routes. Review and security check found no CRITICAL or HIGH issues; both flagged one MEDIUM (double serialization under the `jobs` lock), which was fixed. 12 files, +1025/-17, of which about 450 lines are tests.

## 1. Problem Statement

OpenAI's background mode returns a `queued` response immediately and lets the client poll `GET /v1/responses/{id}`, cancel it, or wait for completion. mlxcel's store could not support this:

- There was no record of a response between creation and completion, so a poll during generation would 404.
- `ResponseStatus::Queued` existed in the type but nothing used it.
- Entries are subject to TTL and LRU eviction. A long-running job stored as an ordinary entry could be evicted while still running.
- `store: false` responses are never stored, yet a background client still needs to poll the result.

The sibling sub-issues (#2898 create-and-poll, #2899, #2900, #2901) are written against a fixed store API, so this PR had to implement the issue's signatures exactly.

## 2. Change Summary

| Area | Change |
|---|---|
| `src/server/responses_store.rs` | `jobs: Mutex<BackgroundState>` map, `BackgroundAdmitError`, `BackgroundCancel`, the eight background methods, `Entry.ephemeral_ttl`, `ResponsesStoreConfig.background_max_jobs` and `background_ephemeral_ttl`, `insert_locked` shared by foreground and background inserts |
| `src/server/responses_store_background_tests.rs` (new) | 13 tests: limit and slot release, disabled and over-budget admission, state transition, TTL/LRU survival, 8-thread finish race, ephemeral visibility and expiry, oversized terminal, byte accounting, cancel, wait |
| `src/main.rs`, `src/bin/mlx_server.rs`, `src/commands/serve.rs`, `src/server/cli_input.rs`, `src/server/startup.rs` | Two flags with env vars, forwarded into `ResponsesStoreConfig` |
| `src/main_tests.rs`, `src/bin/mlx_server.rs` tests | Defaults, flag values, env values, and TTL 0 refusal for both binaries |
| `docs/responses-api.md`, `docs/environment-variables.md` | Flag rows, env rows, and a paragraph on job retention and budgets |

## 3. Technical Decisions

### 3.1 A separate map instead of a flag on entries

Active jobs could have been ordinary entries marked "pinned". That would need every eviction path (`sweep_expired`, `evict_to_limits`) to skip them and every byte computation to decide whether they count. Keeping them in `jobs`, a different structure under a different lock, makes them exempt from TTL and LRU by construction: no eviction code can see them. The cost is a second lock, handled by a fixed order (3.2).

### 3.2 Lock order and the atomic hand-off

Every path that takes both locks takes `jobs` first, then `inner`. `finish_background` removes the job and inserts the terminal entry while holding `jobs` the whole time, so a concurrent `background_snapshot` (which also takes `jobs` first) sees either the live job or the terminal entry, never neither. The first caller to remove the job wins; later callers find no job and return the recorded entry. The 8-thread test checks that every caller gets the same object.

### 3.3 Waiting without missed wake-ups

`wait_background_terminal` uses a per-job `BackgroundSignal { Notify, OnceLock<ResponseObject> }`. `finish_background` sets the `OnceLock` before calling `notify_waiters`. The waiter creates its `Notified` future first and then checks the `OnceLock`. Tokio guarantees that a `Notified` future observes any `notify_waiters` call made after it was created, so either the check sees the value or the future is woken. Reading the result from the `OnceLock` instead of the store also means a waiter gets the terminal object even if LRU pressure evicts the stored entry immediately.

### 3.4 Byte budget without eviction pressure

Active job bytes are tracked in `active_bytes`, added into `approximate_total_bytes`, and checked at admission (`OverBudget` when active bytes plus the new job exceed `max_bytes`). They never feed `evict_to_limits`, because a burst of running jobs should not flush completed history. In the worst case the store holds about twice `max_bytes` (active jobs plus stored entries); this is the bound the issue specified.

### 3.5 Oversized terminal objects

A terminal object larger than `max_bytes` would be inserted and evicted at once, and the poll after the job finished would 404. Instead it is recorded as a `failed` object with `error.code: "server_error"`, empty output and a message naming the budget. Input items are dropped from that record, and if echoed fields (`instructions`, `tools`, `metadata`) still overflow they are dropped too.

### 3.6 Ephemeral entries for `store: false`

The terminal object of a `store: false` job is an ordinary entry with `ephemeral_ttl` set. `get()` returns `None` for it, so `previous_response_id` and the existing GET path cannot read it; `background_snapshot`, `request_background_cancel` and `wait_background_terminal` can. Expiry uses the existing lazy sweep: an entry expires at whichever of the normal TTL and the ephemeral TTL is shorter. The issue proposed an `ephemeral_until: Instant`; the implementation stores a `Duration` and compares ages, which removes any `Instant` overflow for very large TTL values. The field is private, so the sibling API is unaffected.

### 3.7 Defaults

The proposed defaults were kept. 64 jobs stays well below Tokio's 512 blocking threads, since each active job holds one generation thread. A 600-second ephemeral TTL gives a client time to poll without holding unstored output for the full hour. TTL 0 is refused by clap (`range(1..)`) for both the flag and the env var, and `startup.rs` clamps a programmatic 0 to 1 as a backstop.

## 4. Changes During Review

Both reviewers noted that `finish_background` serialized the terminal object twice (once to check the budget, once in the insert) while holding the `jobs` mutex. A multi-megabyte result would block every admit, poll and cancel for that time. The fix measures the terminal once before taking any lock, keeps the input-item size computed at admission (`BackgroundJob.input_bytes`), and passes the precomputed entry size into `insert_locked`. Foreground `insert()` now also sizes its payload before taking the store lock.

## 5. Open Points for Sibling Sub-issues

- `remove()` (DELETE) also deletes a `store: false` terminal entry. #2898 decides what DELETE returns for such ids.
- Admitting an id that is already active replaces the job and its waiters time out. Ids are server-generated UUIDs, so this is unreachable in practice.
- With a very small `max_bytes`, even the trimmed `failed` object can exceed the budget and is evicted at once. Waiters still receive it; later polls 404.
- `background_snapshot`, `request_background_cancel` and `wait_background_terminal` keep `jobs` held while sweeping `inner` on a miss. Releasing it earlier is safe but was left as is for simplicity.
- Jobs are in memory only and a restart loses them.

## 6. Verification

- `responses_store` lib tests on GB10 under `gpu-lock`: 29 passed (16 existing, unchanged assertions; 13 new).
- Flag tests in both binaries: 3 + 3 passed. `cli_input`, `server::routes::responses` and `serve` selectors pass.
- Clippy with `-D warnings` on lib, tests and both binaries; `cargo fmt --check`; env-var docs and license header checks.
- Real binary (`mlxcel-server`, test-fast profile): `--help` lists both flags; TTL 0 by flag or env exits with code 2; a bad `MLXCEL_RESPONSES_BACKGROUND_MAX_JOBS` is refused, so the env var is read; a server on `models/qwen3-0.6b-4bit` with both env vars set served a `store: true` `/v1/responses` request (`completed`, output `2 + 3 = 5.`) and returned it from `GET /v1/responses/{id}`.

Not verified here: any HTTP background behavior (it does not exist until #2898), and the full lib suite, which the orchestrator runs.

## 7. Learning Points

- When two locks guard one logical transition, a fixed acquisition order plus doing the whole transition under the first lock gives readers an all-or-nothing view without merging the locks.
- With `tokio::sync::Notify`, the safe pattern is: create `notified()`, then check the condition, then await. A shared `OnceLock` carrying the result makes the waiter independent of later eviction.
- Expensive work (serialization) that only depends on the caller's data belongs before the lock, with the computed size passed in.
