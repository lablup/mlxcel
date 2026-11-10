# PR #2928: Support background Responses across serving roles

**Date**: 2026-11-10
**Status**: Completed (PR open as draft; closes #2901, part of epic #2903)
**Languages**: Rust, Markdown
**Risk Level**: Medium

## Executive Summary

PR #2928 makes background Responses usable on the multi-model router (`--models-dir`, `--models-preset`, the model cache) and gives the disaggregated front a clear refusal. The router finds the loaded model that owns a response id, so `client.responses.retrieve(id)`, `.cancel(id)` and `.delete(id)` work without a model name; eviction leaves a model with running jobs alone; an explicit unload ends those jobs as `failed`. The disaggregated front answers `POST /v1/responses` with a 400 that says to use a serving node. Single-model and pipeline-head behavior is unchanged. 14 files, +1152/-50, about 750 lines of which are tests. The implementation review found no CRITICAL or HIGH issue.

## 1. Problem Statement

Before this PR the router's per-model sub-apps had no response store at all, so background mode was refused with the store-disabled 400 and `previous_response_id` could not chain. Even with a store, an id-addressed request (`GET`, `DELETE /v1/responses/{id}`, `POST .../cancel`) carries no model, and the router's dispatcher answered it with `model name is missing from the request`. Eviction looked only at `active_requests`, which does not count a background job once its create call has returned, so loading a third model could silently unload one in the middle of a job. The disaggregated front returned a bare 404 for `/v1/responses`.

## 2. Change Summary

| Area | Change |
|---|---|
| `src/server/startup.rs` | `build_responses_store` extracted from the single-model startup, reused per router load |
| `src/server/router_models.rs` | per-load store; `active_background_jobs`, `is_busy_for_eviction`, `holds_response`; `entry_holding_response`; `BackgroundJobsOnUnload::{Fail, Refuse}`; jobs failed at drain start and again before `shutdown_worker` |
| `src/server/router_server.rs` | `response_affinity_id` path parser and the probe in `dispatch`; `?model=` honoured on cancel |
| `src/server/responses_store.rs`, `responses_store_cancel.rs` | read-only `contains_response`; `fail_active_background` |
| `src/server/routes/responses.rs` | shared `response_not_found` 404 |
| `src/server/router_front.rs` | `POST /v1/responses` and `/responses` refused with `invalid_request_error` |
| docs | `responses-api.md` "Serving roles" section; pointers in `distributed.md`, `server-features.md` |

## 3. Technical Decisions

### 3.1 In-process probe instead of a router-level id map

The router asks each loaded entry's store `contains_response(id)` and forwards to the first match in name order (`entries` is a `BTreeMap`). A router-wide id map was rejected because it would be a second structure to bound and expire in step with each store. The probe is bounded by `--models-max`, never autoloads, does not refresh LRU, and runs behind the router-level API-key middleware like every dispatched request. It reveals nothing a caller could not learn with `?model=` on each model.

`contains_response` takes the jobs mutex, then the entries read lock, then the in-flight read lock, one at a time with no nesting. The terminal write (`record_terminal_locked`) inserts the stored entry while still holding the jobs mutex, so a probe always sees a finishing job in one of the two places and never gives a transient 404.

### 3.2 Paths the probe covers

`GET` and `DELETE /v1/responses/{id}` and `POST /v1/responses/{id}/cancel`, with or without `/v1`. The id is one non-empty segment, percent-decoded the same way the sub-app's `Path` extractor decodes it; `compact` and `input_tokens` are excluded. The query is not part of the path, so `?stream=true&starting_after=N` resumes route the same way. `GET .../input_items` is not covered: the sub-app has no such route, so it keeps the dispatcher's missing-name 400.

### 3.3 Eviction refuses, unload fails

Both eviction sites (explicit target and LRU filter) now use `is_busy_for_eviction`, and the eviction unload runs with `BackgroundJobsOnUnload::Refuse`: if a job was admitted between the busy check and the drain, the unload backs out with `mark_ready` and a capacity error instead of killing the job. An explicit unload (`POST /models/unload`, WebUI, removal, shutdown) uses `Fail`: jobs end `failed` with `server_error` / `The model was unloaded before the response finished.` at drain start, because a client streaming a job holds a request lease and the drain would otherwise wait for the job, and again just before `shutdown_worker` to catch a create that was in flight when the drain began. The router has no idle or TTL auto-unload, and `entry_idle_for_webui_removal` already requires an unloaded entry, so no other idle check needed the new helper.

### 3.4 Behavior change for existing router users

Every loaded model now gets its own store from the router's `--responses-store-*` and `--responses-background-*` flags. Foreground `/v1/responses` with the default `store: true` now persists in router mode, and the memory bound is per loaded model: the worst case is `--models-max` times `--responses-store-max-bytes` (256 MiB by default). The store is dropped with the model on unload, so every id it held returns 404 afterwards; this was confirmed on real checkpoints. The docs state the per-model scope but not the multiplied worst case (open item below).

## 4. Validation

- Lib tests on GB10 under `gpu-lock`, `--test-threads=1`: `router_` 265 passed (3 ignored), `server::routes::responses` 103, `responses_store` 37, `router_front` 56, all green, including the 12 new tests.
- `cargo clippy -p mlxcel --lib --tests --profile test-fast --features cuda -- -D warnings` and `cargo fmt --all -- --check` clean.
- Real router over `~/models-102/mlx` with `qwen3-0.6b-4bit` and `qwen2.5-0.5b-instruct-4bit` loaded (`--models-max 2`): a background job on qwen2.5 was polled and cancelled with no model and returned the full `cancelled` Response; a qwen3 job completed with coherent text and was retrieved with no model; an unknown id returned the `not_found` 404; unloading qwen2.5 during an `in_progress` job logged `unload ended background responses as failed ... jobs=1`, and the id returned 404 afterwards.

## 5. Follow-up Actions

- MEDIUM: state the worst-case memory (`--models-max` x `--responses-store-max-bytes`) in `docs/responses-api.md`, next to "the limits apply per loaded model".
- LOW: an explicit unload fails jobs at drain start, so an unload that then times out draining or hits a stale revision has already ended the jobs while the model stays loaded.
- LOW: tests inject the store through the test factory, so the production `build_model_app(..., build_responses_store(...))` wiring is covered only by the manual run.
- LOW: `fail_active_background` records `failed` even for a job whose cancel was requested but not yet recorded.
