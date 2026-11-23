# PR #3301: Bound MCP, decision and rerank server request work

**Date**: 2026-11-23
**Status**: PR open (fixes #3296). Unit suites for the touched modules pass; no real checkpoint was needed

## Executive Summary

Four server routes accepted untrusted request work without a bound or a way to stop it. A stdio MCP server launched through `npx` or `uvx` survived its client because only the launcher was killed. `/v1/decisions` accepted any number of videos, decoded an audio clip fully before applying its 300 second cap, and on a timeout answered 504 while the worker kept running and, after the handler dropped them, deleted the temp files out from under the model. `/v1/rerank` accepted any number of documents.

This PR fixes five items in one change. MCP stdio children now lead their own process group and the whole group is killed. Decisions cap videos at 8, refuse long audio from the container header before decoding, and cancel queued work on timeout while keeping temp files alive until the worker is done. Rerank caps documents at 1024 through `MLXCEL_RERANK_MAX_DOCUMENTS`. Security review found one more hole, a decision audio header declaring a huge source sample rate, and the fix holds that rate to 192 kHz. One trade-off is accepted: MCP children no longer receive a terminal SIGINT.

## 1. Problem Statement

### 1.1 Five unbounded or unstoppable paths

| Item | Route or module | Failure |
|---|---|---|
| 1 | `server/mcp/client.rs` | `kill_on_drop` and `shutdown` signal only the child. A launcher's grandchild (the real MCP server) keeps running after the request or server ends. |
| 2 | `/v1/decisions` videos | No count cap. Each video is resolved, possibly downloaded, before the model's token gate can refuse the overflow. |
| 3 | `/v1/decisions` timeout | `predict` returned `Timeout` but the queued command still ran later for a caller that had already received 504. The handler dropped the `PrivateTempDir` guards right after `spawn_blocking`, so a prediction still running could lose its files. |
| 4 | `/v1/decisions` audio | The 300 second cap ran after `decode_wav_native_compat` had decoded the whole clip under a 24 hour policy. |
| 5 | `/v1/rerank` | `validate_items` bounded nothing but emptiness, so one request could carry any number of documents and image fetches. |

### 1.2 Risk of leaving it

| Risk | Impact | Likelihood |
|---|---|---|
| Leaked MCP grandchildren accumulate across requests | Medium: process and memory exhaustion on a long-running server | High with launcher-style servers |
| One request forces unbounded fetch, decode or queue work | Medium: denial of service by an authenticated or local client | Medium |
| Temp files removed under a running decode | Low to medium: wrong or failed predictions after a 504 | Low |

## 2. Change Summary

| Area | Change |
|---|---|
| `src/server/mcp/client.rs` | On Unix, `command.process_group(0)` at spawn and the pgid stored in `process_group: Option<i32>`. `kill_process_group` sends one `killpg(SIGKILL)` and `take()`s the id. `shutdown`, `has_exited` and a new `Drop` call it. Windows is unchanged. |
| `src/server/routes/decisions.rs` | `MAX_DECISION_VIDEOS = 8`, checked in `check_video_support` before any resolve, with a 400 naming the cap. The check now runs before the ffmpeg probe. Guards move into the worker call. |
| `src/server/decision_worker.rs`, `decision_model.rs` | `DecisionCommand::Predict` carries `cancelled: Arc<AtomicBool>` and `temp_files: Vec<PrivateTempDir>`. `predict` sets the flag on timeout. `worker_loop` skips a flagged command and drops the temp files only after the prediction or the skip. The `DecisionModelProvider::predict` trait takes `temp_files`. |
| `src/audio/preprocessing.rs`, `preprocessing_policy.rs` | New `decode_native_within(bytes, max_duration_seconds)`. It runs `container::inspect` first, so the header frame count raises the `source duration samples` limit before samples decode, and holds the source rate to 192 kHz. The compatibility decoder is unchanged for other callers. |
| `src/decision/media.rs` | `decode_audio_16k` calls `decode_native_within(MAX_DECISION_AUDIO_SECONDS)`. `duration_message` maps the container limit to the same user-facing 300 second text. The post-decode check stays as a backstop. |
| `src/server/routes/rerank.rs` | `MLXCEL_RERANK_MAX_DOCUMENTS` (default 1024, read once). Unset, zero or non-numeric values keep the default, the last two with a warning. `validate_items` refuses an over-long list before any image fetch. |
| Tests | `client_tests.rs` plus `tests/fixtures/mcp/fake_mcp_server.py` (grandchild killed after shutdown and after drop), `decisions_tests.rs` (9 videos), `decision_worker_tests.rs` (skip and temp-dir lifetime), `media_tests.rs` (301 s WAV and the rate bound), `rerank_limit_tests.rs` (1025 documents and env parsing). |
| Docs | `docs/decisions.md`, `docs/embeddings.md`, `docs/environment-variables.md`. |

17 files, +763/-37. The docs have no Korean counterparts in this repository, so none were added.

## 3. Technical Decisions

### 3.1 Kill the group, once

`killpg` is sent from `shutdown`, `has_exited` (after the leader is reaped) and `Drop`. The id is `take()`n on the first send, so `Drop` after `shutdown` never signals a group id the kernel may have reused. The alternative of signalling the group only from `Drop` was rejected because `shutdown` reaps the child first and the pgid would then be stale.

### 3.2 Guards travel with the command

On a timeout the handler cannot know whether the worker is still decoding the files. Moving the `PrivateTempDir` guards into the command makes the worker thread the only place they are dropped, either after the prediction or right after it skips a cancelled command. The alternative, a separate `Arc` holder in `DecisionMedia`, would have spread file ownership across the media type and the route. Threading `temp_files` through the provider trait touched every test fake but kept ownership in one place.

Cancellation is cooperative: a prediction already running is not interrupted. Only commands still queued are skipped.

### 3.3 Reject long audio from the header

The container probe already computes a duration limit from the declared frame count, so passing `MAX_DECISION_AUDIO_SECONDS` into the policy refuses a 301 second WAV before sample decode with no new parsing. `duration_message` recovers the clip length from `actual * cap / maximum` so the 400 text is the one users already saw.

### 3.4 Hold the source rate (security review fix)

A duration cap alone does not bound work. A header declaring a multi-MHz sample rate turns a short "duration" into hundreds of millions of samples. `decode_native_within` now passes the chat policy's 192 kHz ceiling (`MAX_SOURCE_SAMPLE_RATE`, made `pub(super)`) as the source rate limit. The legacy decoder keeps `u32::MAX`, so non-decision callers are unchanged.

### 3.5 Invalid rerank cap keeps the bound

A zero or unparsable `MLXCEL_RERANK_MAX_DOCUMENTS` falls back to 1024 with a warning rather than disabling the check, so a typo cannot remove the bound.

### 3.6 Trade-off: MCP children no longer get terminal SIGINT

A process in its own group is outside the terminal's foreground group, so Ctrl-C in the server's terminal no longer reaches MCP children directly. They now stop through the server's own shutdown path (`shutdown` or `Drop`) or when their stdin closes. A hard kill of the server that skips `Drop` leaves them running, as before. Windows process-tree handling is out of scope.

## 4. Verification

Narrow selectors only, with `--profile test-fast --features metal,accelerate`, `RUST_MIN_STACK=16777216`, and `CARGO_BUILD_JOBS=6`. The exact results are in the PR hand-off and the PR comment. Selectors run: `server::mcp`, `server::routes::responses_mcp`, `server::routes::decisions`, `server::decision_worker`, `decision::media`, `server::routes::rerank`, `audio::preprocessing` and `coverage_tests`, plus `cargo clippy --lib --tests --features metal,accelerate -- -D warnings`, `cargo fmt --all -- --check`, and the CI contract scripts (`check_used_by`, `check_kernel_dtype_keys`, `check_env_var_docs`, `check_crate_versions`, `check_kernel_port_dispatch`, `check_binary_assets`, `insert_apache_header --check`).

Each of the five new tests asserts a behavior main lacks: the MCP tests check that the grandchild pid returns ESRCH within a bounded poll, the route tests check a 400 and no resolve or fetch, the worker test checks that a timed-out queued request is never predicted and that the first request's temp dir exists until its prediction returns.

Not run: the workspace-wide gate and real-checkpoint smoke runs, which are left to the orchestrator.

## 5. Unverified and Open Items

- **LOW (a): pid-reuse window in `has_exited`.** After `try_wait` reaps the leader, `killpg` runs a few microseconds later. If the kernel reused the pid as a new group in that window, the signal would hit it. The window is microseconds and needs a wrapped pid counter, so it was left.
- **LOW (b): audio duration wording.** For compressed audio (mp3, flac) with no declared frame count, the container cannot refuse from the header. The decode loop enforces the same ceiling, and the message then reports the length reached when the ceiling tripped, which can read as an approximation. The wording was not changed.
- **LOW (c): rerank text size.** Query and document text is bounded only by the JSON body size. This predates the PR and the document count cap does not address it.
- **Running predictions are not interrupted.** Only queued work is skipped on timeout.
- **Windows.** MCP grandchildren are still not killed on Windows.

## 6. Learning Points

- Killing a launcher is not killing the server it launched. Own the process group at spawn and signal it once, then forget the id.
- A caller that stops waiting must hand resource ownership to the code that is still using the resource, not release it on the way out.
- A cap that runs after the expensive step is not a bound. Move it to the earliest point the input declares its size, and bound every declared quantity (rate as well as duration).

## Related Commits

| Hash | Type | Message |
|---|---|---|
| `0e5ebd3b` | fix(security) | bound MCP, decision and rerank request work |
| `16310611` | fix(security) | bound the decision audio source sample rate |
