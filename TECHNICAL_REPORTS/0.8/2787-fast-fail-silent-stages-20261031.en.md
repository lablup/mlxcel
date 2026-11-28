# PR #2787: Fast-fail silent pipeline stages and sanitize stage logs

**Date**: 2026-10-31 (written before merge; the PR is open as a draft)
**Status**: Implementation complete at 487634ea on top of main 9f06fef2. The local gate in the PR body passed on GB10 (spark-102). At the time of writing a second local gate was running in the worktree, and the hosted `2-host logical` job had failed once at 487634ea on a timing bound in one of the new tests (see 4.3); `CPU-only link` and `WebUI installed artifact` were pending
**Languages**: Rust, Markdown
**Risk Level**: Medium (no wire change and no public API change, but the coordinator now refuses requests on its own while it believes a stage is silent, which is new serving behavior; client-facing text is unchanged)

## Executive Summary

PR #2787 (closes #2782) fixes the three residuals that the #2769 report left on the remote pipeline coordinator's failure path. The main one is availability: after #2769 one failed request against a silent stage cost about 34 s, but every following request still waited a full 30 s `stage_timeout` at prepare before failing, and the model worker is a single thread. The PR adds a runtime-wide availability gate. A full-`stage_timeout` exchange failure opens it; while it is open, prepares and forwards fail at once with a typed `PipelineUnavailable`, except that at most one request per 10 s probes every stage with the 2 s teardown wait and is served if all of them answer. The PR also escapes stage-supplied text (error replies, request ids, serde error reasons) before it reaches either side's log, and puts the "did not acknowledge" teardown warning under its own `teardown-ack` warning budget.

## 1. Problem Statement

#2782 collected three items. Priority was medium because item 1 needs no cluster secret: per-frame AEAD does not stop an on-path party from dropping frames, and a dropped frame, a partition and a stopped stage process all reach the coordinator as a stage timeout.

### 1.1 Every new request waited a full stage timeout while a stage stayed silent (availability)

`prepare_sequence_state` calls `broadcast_ack`, which calls `exchange` with the fixed 30 s `stage_timeout` per stage. Nothing remembered that a stage had just timed out, so each new request blocked the model worker for 30 s at prepare and then paid the rollback release. Sequences prepared before the stage went silent each waited 30 s at `RunEntry`. #2769 bounded the cost of one failed request; it did not bound the cost of the next hundred. #2762 had rejected a circuit breaker because it cannot restore service when every request needs every stage. That remains true, but a breaker still turns 30 s of blocked worker per request into an immediate refusal, which is what #2782 asked for.

### 1.2 Stage-supplied text reached the coordinator log raw (log forging, LOW)

The `message` of a `RemoteStageResponse::Error` was formatted verbatim at four `runtime.rs` sites, two of which reach `warn!` through `{err:#}` in `server_runtime.rs`. On the stage side, `report_activation_failure` logged the upstream-supplied `request_key` raw. An authenticated stage could put newlines or ANSI escapes in either and forge log lines. `stage_assignment.rs` already had the right helper (`loggable_abort_detail`) for `Abort { detail }`, but it was private to that module.

### 1.3 A slow but healthy stage drew unthrottled teardown warnings (log noise)

#2769 capped teardown acknowledgements at 2 s. A healthy stage still finishing a long `RunEntry` misses that wait and applies the cancel or release later, yet the coordinator logged one `warn` per command and stage with wording that read like a lost frame. Its own report (section 3.1) recorded this trade-off.

## 2. Change Summary

| Area | File | Change |
|---|---|---|
| Gate state machine | `src/distributed/pipeline/stage_availability.rs` (new) | `StageAvailability` (outage start, failed peer, last probe, generation), `GateDecision { Pass, Probe, Refuse }`, `ProbeClose`, `ProbeFailure`, the typed `PipelineUnavailable` error, `STAGE_PROBE_INTERVAL = 10 s`; 7 unit tests |
| Coordinator runtime | `src/distributed/pipeline/runtime.rs` | `availability: Mutex<StageAvailability>` and `probe_interval` fields with a `#[cfg(test)]` `with_probe_interval`; `exchange` opens the gate on `Err`; `ensure_stages_available` runs after the draining check in prepare and before `RunEntry` in forward (a refused forward goes through `fail_sequence`); `broadcast_best_effort` sends to the silent peer with a zero wait; `log_teardown_miss` under the `teardown-ack` budget; four `Error` message sites and the malformed-reply reason go through `loggable_peer_text`; a forward `Error` now reads `remote stage {i} failed the forward: ...` |
| Server model | `src/distributed/pipeline/server_runtime.rs` | `is_unavailable` walks the `anyhow` chain for `PipelineUnavailable`; such refusals log at debug, everything else at warn as before; client text unchanged |
| Peer text | `src/distributed/peer_text.rs` (new) | `MAX_PEER_TEXT_BYTES = 1024`, `truncate_peer_text`, `loggable_peer_text` (cut on a char boundary, then `escape_debug`); 3 unit tests |
| Shared helper users | `stage_assignment.rs`, `remote_service.rs` | `abort_detail` and `loggable_abort_detail` delegate to `peer_text`; the two `request_key` log sites are escaped |
| Warning budget | `src/distributed/handshake_warn.rs` | `HandshakeWarnClass::TeardownAck`, `CLASS_COUNT` 5 to 6, `Display` text `teardown-ack` |
| Test stub | `src/distributed/pipeline/stub_stage.rs` | `set_silent(bool)` over a shared `AtomicBool`; `StubRunBehavior::Error { message }`; `STUB_FORCED_ERROR`; `wait_for_commands(count, timeout)`; `Shutdown` now replies through the common path so a silent stub stays silent for it too |
| Tests | `runtime_availability_tests.rs` (new), `runtime.rs`, `handshake_warn_tests.rs`, `scheduler_pipeline_failure_tests.rs` | 4 integration-style runtime tests over real TCP, 1 serde escaping test, 1 `Display` assertion; two existing tests read the stub log through `wait_for_commands`; one scheduler command log gains `probe_state` |
| Docs | `docs/distributed.md` | The failure-path paragraph describes the gate, the probe interval, the escaping and the `teardown-ack` budget. No Korean counterpart of this document exists |

### Statistics

| Item | Value |
|---|---|
| Files changed | 14 (2 new modules, 1 new test file) |
| Lines | +1318 / -73 |
| New tests | 15 (7 gate state machine, 3 peer text, 4 runtime availability, 1 malformed-reply reason) |
| Commits | 2 |

| Commit | Purpose |
|---|---|
| 7dade070 | All three items of #2782, tests and docs |
| 487634ea | Reconcile overlapping gate probes with a generation counter; escape the serde error reason in the dropped-reply warning |

## 3. Technical Decisions

### 3.1 Probe on the first request instead of a fixed cooldown

**Context.** A timeout can mean a dead stage or one dropped reply from a healthy stage. A fixed cooldown treats both alike and refuses every request for the whole window after a single lost frame.

| Option | Assessment |
|---|---|
| Lower or configure `stage_timeout` | Also shortens the legitimate wait for a long `RunEntry`; configurability is out of scope in #2782 |
| Fixed cooldown after a timeout | One dropped reply refuses the next 10 s of traffic |
| Background health checker | Another thread and its own transport use; the model worker would still race it |
| **Chosen: open on timeout, probe on the first request after it, refuse inside the probe interval** | A single dropped reply costs the next request one probe round trip (`a_single_dropped_reply_does_not_refuse_the_next_request`); a dead stage costs at most one 2 s probe per 10 s instead of 30 s per request |

The probe is `RemoteStageCommand::ProbeState`, sent to every stage in order with `teardown_ack_timeout` (2 s, or `stage_timeout` if smaller). It stops at the first stage that does not answer `State`, which then becomes the named `failed_peer`. The probing request pays the probe and, on success, is served normally in the same call.

### 3.2 Only the full-`stage_timeout` path opens the gate

`exchange` is the only call that opens the gate; `exchange_within` with a short wait (teardown, probe) does not. A missed teardown ack says nothing new about a stage already known silent, and a slow-but-healthy stage misses it routinely (item 3). An `Error` reply is an answer and never opens it. In practice "full-`stage_timeout` failure" means any `Err` from `exchange`, so a transport send error (for example a refused connection) opens the gate too, which matches the intent of an unreachable stage.

### 3.3 A typed error inside `anyhow`, logged at debug

`PipelineUnavailable { stage, since }` is a `thiserror` type converted into `anyhow::Error`. `PipelineServerModel` finds it with `err.chain().any(|c| c.downcast_ref::<PipelineUnavailable>().is_some())` and logs it at debug, because one warn per refused request during an outage would be the very flood item 3 removes elsewhere. The operator signal is the gate's own transitions: one `warn` when it opens (stage, wait, probe interval) and one `info` when it closes (outage length). Clients still get `PIPELINE_PREPARE_REFUSED` or `PIPELINE_FORWARD_FAILED`, so no address or timing leaks to them.

### 3.4 Keep the #2753 error contract on a refused forward

A forward refused by the gate runs `fail_sequence` like every other post-mapping failure, so `Err` still means the coordinator's state for that sequence was discarded and the scheduler's later close sends nothing. The teardown goes to the silent peer with `Duration::ZERO`: `exchange_within` writes the frame, finds its deadline already passed and returns, and the miss is logged at debug. Healthy stages keep the 2 s wait. A refused prepare happens before any mapping exists, so there is nothing to tear down.

### 3.5 Never hold the gate mutex across I/O, and reconcile overlapping probes

`decide` records `last_probe` under the lock before the probe is sent, so a request arriving during the probe is refused rather than probing too. The probe exchanges then run with the lock released (they are serialized by the `io_runtime` mutex like every other exchange), and the result is applied under the lock again. Probes can still overlap when one outlasts the probe interval. The second commit adds a `generation` counter, bumped by every new timeout and every failed probe but not by a close:

| Probe result when applied | Outcome |
|---|---|
| Success, same generation, gate open | Closes the gate (`ProbeClose::Closed`) and logs the info line |
| Success, same generation, gate already closed by a sibling probe | Request served, nothing logged (`AlreadyClosed`) |
| Success, older generation (a newer timeout replaced the outage) | Request refused, gate unchanged (`Superseded`) |
| Failure, same generation, gate open | Records the stage and bumps the generation |
| Failure, same generation, gate closed by a sibling probe | Reopens as a fresh outage and logs the open line again (`reopened`), so the log never shows a closed gate that is refusing requests |
| Failure, older generation | Leaves the newer state alone; only this request is refused |

The state machine is pure (`Instant` is passed in), so these cases are unit tests without sockets or sleeps. In production the model worker is one thread and a probe is bounded by 2 s per stage, so overlap needs a slow multi-stage probe; the logic is there so the log and the gate never disagree if it happens.

### 3.6 One helper for all peer text

`loggable_peer_text` cuts to 1024 bytes on a char boundary, then applies `escape_debug`. Cutting first means the bound applies to what the peer sent; the escaped form may be longer but is one line. `stage_assignment.rs` keeps its `abort_detail` (the outgoing `Abort` still needs the raw bounded text) and `loggable_abort_detail`, both now delegating. The review round found one site the issue had not listed: serde_json echoes an unknown variant or field name from the payload verbatim, so the dropped-reply warning went through `malformed_reply_reason`, which escapes it.

### 3.7 A sixth warning class rather than reusing `stage-drop`

`stage-drop` is a stage-side budget for frames from an authenticated upstream; teardown misses are coordinator-side and expected under load. Sharing a window would let a busy coordinator push stage-drop warnings to debug. `TeardownAck` gets its own 5-per-minute window, and the reworded line says the frame was sent so an operator does not read it as a lost release.

## 4. Verification

### 4.1 Failing-first evidence

On 9f06fef2 with only the test-support plumbing (stub `set_silent`, `StubRunBehavior::Error { message }`, `with_probe_interval`): 35 passed, 4 failed.

| Test | Before | After |
|---|---|---|
| `silent_pipeline_fails_new_prepares_fast_until_a_probe_succeeds` (1 s stage timeout, 100 ms teardown, 300 ms probe interval) | prepare B: `remote stage ... timed out after 1s`, no `PipelineUnavailable` | B refused within 500 ms after one probe; C refused within 50 ms with no command sent; D served after the stage answers, log ends `probe_state, prepare` |
| `a_single_dropped_reply_does_not_refuse_the_next_request` | command-log assertion failed | `prepare, run_entry, cancel, release, probe_state, prepare, run_entry` and logits for B |
| `stage_error_text_is_escaped_before_logging` | `"boom\nINFO forged\u{1b}[31m"` reached the error raw | no `\n` or `\u{1b}` in `{err:#}`; contains `boom\\n` |
| `teardown_ack_warnings_are_rate_limited` (20 failed forwards, 10 ms teardown) | 40 of 40 misses at warn | 40 lines, at most `2 * HANDSHAKE_WARN_PER_WINDOW` at warn, each warn says `the frame was sent` |

### 4.2 Local gate at 487634ea

GB10 (spark-102) under `gpu-lock`, `--test-threads=1`, `MLXCEL_REQUIRE_MODELS=1`, `MLXCEL_SDPA_DETERMINISTIC=1`. fmt and clippy (`--lib --tests -D warnings`) clean.

| Suite | Result |
|---|---|
| `distributed::` lib tests | 1576 passed, 4 ignored |
| `scheduler_pipeline_failure_tests` | 4 passed |
| `distributed_integration` | 20 passed |
| `pipeline_e2e` | 30 passed |
| `pipeline_remote_real_models` (`--skip glm_moe_dsa`) | 10 passed |
| `pipeline_server_remote_real_models` (`--skip glm_moe_dsa`) | 7 passed |
| `pipeline_ci_multi_stage_real_models` | 4 passed |
| `disaggregated_serving_e2e` | 2 passed |
| `disaggregated_router_e2e` | 8 passed |
| `server_exit_joins_worker` | 2 passed |

### 4.3 Hosted `2-host logical` failure at 487634ea

The job (run 41161926566) failed one test of 529: `silent_pipeline_fails_new_prepares_fast_until_a_probe_succeeds` at `runtime_availability_tests.rs:139`, "prepare C took 101.71692ms; a refusal inside the probe interval sends nothing". The assertions before it (A torn down, B refused within 500 ms after one probe) held; the ones after it were not reached. The refusal path takes only the gate mutex and builds an error, so 100 ms is not the gate's own cost.

The likely cause, inferred and not verified: that workflow sets `RUST_BACKTRACE=short`, so converting `PipelineUnavailable` into `anyhow` captures a backtrace, and std's backtrace capture takes a process-wide lock. About 100 ms earlier (03:05:07.1547 versus 07.2562) a background tokio task of a concurrently running test panicked at `runtime_correlation_tests.rs:105` (a `MockRouter` send to a closed channel) and printed a resolved backtrace, which holds that lock while it symbolizes. Whatever the cause, a 50 ms wall-clock bound is tight for a parallel hosted run. #2782 requires this job to stay green, so the bound (or the backtrace setting for that test) needs attention before merge.

### 4.4 Not run

The manual `--pp-auto 2` check with `kill -STOP` on a stage was not run; the probe and refusal behavior is covered over real TCP by the stub tests, but the latencies have not been measured against a real stopped process. `glm_moe_dsa` (`glm5-4bit` does not load on this box), Thunderbolt and RDMA transports, and macOS were not run.

## 5. Remaining Gaps and Risks

- **A slow but alive stage is now treated as down.** If a `RunEntry` times out because a stage is slow (the #2769 report recorded `glm4` timing out while compiling CUDA kernels for the first time on a loaded host), the gate opens and the next probe gives that stage only 2 s. A stage still busy with the abandoned forward misses it, so requests are refused until a later probe gets through. Before this PR those requests would have waited and possibly succeeded. The 30 s fixed `stage_timeout` remains the root problem; making it configurable is out of scope.
- **Recovery latency of up to one probe interval.** After a stage comes back, requests keep being refused until the next probe is due, up to 10 s after the last one.
- **Attribution.** A `RunEntry` timeout names the entry stage until the first probe names the silent one, and a probe names only the first stage in order that fails. Per-stage attribution is out of scope.
- **Hosted CI timing bound.** See 4.3.
- **The `teardown-ack` budget is process-wide.** Several coordinators in one process share the 5-per-minute window; the rollover summary still reports the suppressed count.
- **The gate opens on any `exchange` error,** including a send failure, not only an elapsed timeout. This matches the intent but the docs phrase it as "reaches the stage timeout".

## 6. Learning Points

### 6.1 Separate the pure state machine from the I/O around it

`StageAvailability` takes `now` as an argument and returns decisions; `RemotePipelineRuntime` does the locking and the exchanges. That split made the overlapping-probe cases (3.5) cheap to enumerate as unit tests, and it is what let the second commit fix a race in the bookkeeping without touching the transport tests.

### 6.2 Generation counters for results that arrive after the state moved on

A probe result describes the outage that existed when the probe started. Tagging the decision with a generation and checking it when applying the result is the same pattern as an ABA-safe compare-and-swap: success for a stale generation must not close a newer outage, and failure for a stale generation must not overwrite a newer close. Not bumping the generation on close is the detail that lets a sibling success be served instead of refused.

### 6.3 Escape at the trust boundary, including error messages from libraries

The obvious sites were the stage's own `message` fields. The less obvious one was `serde_json::Error`'s `Display`, which quotes attacker-controlled keys from the payload. Any library error built from untrusted input is peer text too.

### 6.4 Wall-clock assertions in tests that run beside panicking siblings

A test that asserts "under 50 ms" measures the whole process, not the code under test. Backtrace capture, panic printing and scheduler load all land in that window on a shared runner. Bounds that separate "sent nothing" from "waited a timeout" can be set an order of magnitude looser and still discriminate, or the test can assert on the command log alone.

## 7. References

- Issue #2782: fast-fail silent stages and sanitize stage logs
- PR #2769 (issue #2762): bounded one failed request to one `stage_timeout` plus two 2 s teardown waits; its report section 5 listed item 1 and item 3
- PR #2753 (issue #2727): a stage failure fails one request, and `Err` from a forward means the coordinator's state was discarded
- `src/distributed/handshake_warn.rs` (#2655, #2692, #2712, #2762): the per-class warning budget
- `docs/distributed.md`, "What the secret protects", failure-path paragraph
