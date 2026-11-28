# PR #2769: Bound the remote pipeline failure path and stage state

**Date**: 2026-10-31 (written before merge; the PR is open as a draft)
**Status**: Implementation complete at 8993404e on top of base bebdd62c (behavior identical to main 5c8ef80a). The local gate in the PR body passed on GB10 (spark-102). At the time of writing a second local gate was running in the worktree and hosted CI was partly pending (`2-host logical`, `cargo-clippy`, `CPU-only link`, `OpenXLA feature compile`, `python tooling tests`, `WebUI bundle`, `WebUI installed artifact`)
**Languages**: Rust, Markdown
**Risk Level**: Low (no wire change and no public API change; the coordinator now waits at most 2 s for a teardown acknowledgement, and a stage at its prepared-sequence cap releases sequences idle for 10 minutes)

## Executive Summary

PR #2769 (closes #2762) bounds what one silent remote pipeline stage costs the coordinator. After #2753 a stage failure failed one request instead of aborting the process, but the single model worker still spent about four `stage_timeout`s (about two minutes at the fixed 30 s) on each failed request: one to detect the failure, then a full timeout each for cancel, release, and a second release from the scheduler's close. The PR caps teardown acknowledgements at 2 s, makes a failed forward discard the coordinator's sequence mapping so release goes out once, and brings the cost to one `stage_timeout` plus two short waits (about 34 s). On the stage side it lets a `PrepareSequence` that finds the 4096-sequence cap full reap sequences idle for 10 minutes, rate-limits the per-frame warnings for unowned activations under a new `stage-drop` class, and compiles the test-only service knobs out of production builds.

## 1. Problem Statement

#2762 collected the availability and hygiene residuals that the reviews of #2753 (issue #2727) and #2746 (issue #2726) left on the failure path, including section 5 of the #2753 report. Priority was medium because item 1 needs no cluster secret: per-frame AEAD does not stop an on-path party from dropping frames, and a dropped frame, a network partition and a hung stage process all look the same to the coordinator, a stage timeout.

### 1.1 One silent stage stalled the model worker for about two minutes (availability)

`RemotePipelineRuntime::exchange` waited `stage_timeout` for every command, teardown included. A failed `forward_sequence` called `cleanup_request`, which sent `CancelRequest` and `ReleaseSequence` through `broadcast_best_effort`. Against a silent stage each of those waited the full 30 s. `cleanup_request` also left the `request_ids` entry in place, so the scheduler's later close of the failed sequence called `release_sequence_state_by_id`, found the mapping and released a third time, another 30 s. The model worker is a single thread making synchronous exchanges, so every other request on the coordinator waited behind all four timeouts. `remote_runtime_teardown_reaches_every_stage_past_a_silent_one` pinned the double release as `prepare, run_entry, cancel, release, release`.

### 1.2 Prepared sequences never expired on a stage (resource leak)

A stage's `sequence_owners` map was cleared only by cancel, release, shutdown or a failed activation. A coordinator that crashed mid-request left its in-flight sequences, with their KV caches, counted against `MAX_STAGE_SEQUENCES = 4096` until the stage process restarted. At the cap every new `PrepareSequence` is refused, so enough coordinator crashes would make the stage refuse all work.

### 1.3 Unowned-activation warnings were unthrottled (log flooding)

An activation for a sequence nobody prepared on the stage fails the owner check and reaches `report_activation_failure`, which logged one `warn` per frame at two sites; the final stage's "no reply nonce" branch did the same. Only an authenticated upstream peer can send activations, so this needs the secret, but a misbehaving or buggy peer could fill the log at frame rate.

### 1.4 Test-only knobs in production and an unreachable branch

`set_max_sequences` and the `sequence_cap: Option<usize>` parameter of `spawn_service` were compiled into release builds, where production always passed `None`. In `run_activation`, the final stage's `reply_target(..) == None` branch could not be reached: the owner check runs before the executor and the service handles one frame at a time. The branch still carried its own warn-and-drop handling, which a reader had to reason about.

### 1.5 Dropped from the residual list

The #2753 report also listed "the decode lookahead fallback repeats a forward once". #2762 established that this cannot happen today: `PipelineServerModel` declares a model-owned layout and does not override `supports_decode_lookahead_rewind`, so `Engine::can_unwind_lookahead` is false and the scheduler never primes a lookahead for the only model whose forward can return `EngineError::Forward`. No change was made for it.

## 2. Change Summary

| Area | File | Change |
|---|---|---|
| Coordinator runtime | `src/distributed/pipeline/runtime.rs` | `TEARDOWN_ACK_TIMEOUT = 2 s`; a `teardown_ack_timeout = stage_timeout.min(2 s)` field with a `#[cfg(test)]` builder `with_teardown_ack_timeout`; `exchange` now delegates to `exchange_within(peer, request, timeout)`; `broadcast_best_effort` passes the teardown timeout; `cleanup_request` becomes `fail_sequence(seq_id, request_id)`, which removes the `request_ids` entry before the teardown and is called from all five failure exits of `forward_sequence` |
| Stage service | `src/distributed/pipeline/remote_service.rs` | `sequence_owners: HashMap<String, OwnedSequence { owner, last_active }>`; `touch_sequence` on prepare, re-prepare, accepted `RunEntry` and owner-checked activation; `STAGE_SEQUENCE_IDLE_TTL = 600 s` and `reap_idle_sequences`, called only when a new id finds the cap full; `log_stage_drop` for the three drop warnings; `StageOverrides` replaces `sequence_cap`; the unreachable final-stage branch becomes `ok_or_else(..)?` |
| Warning budget | `src/distributed/handshake_warn.rs` | `HandshakeWarnClass::StageDrop`, `CLASS_COUNT` 4 to 5, `Display` text `stage-drop` |
| Tests | `remote_service_tests.rs`, `runtime.rs`, `handshake_warn_tests.rs`, `scheduler_pipeline_failure_tests.rs` | 4 new tests, 1 tightened, 3 scheduler command logs updated to a single release, 1 `Display` assertion; a `tracing` capture writer and `Harness::spawn_capped_with_ttl` for the stage tests |
| Docs | `docs/distributed.md` | Single release, the 2 s teardown wait and the 34 s figure, idle reaping at the cap, the fifth `stage-drop` class. No Korean counterpart of this document exists |

### Statistics

| Item | Value |
|---|---|
| Files changed | 7 |
| Lines | +581 / -95 |
| New tests | 4 (`silent_stage_failure_is_bounded_by_one_stage_timeout`, `idle_sequences_are_reaped_when_the_cap_is_reached`, `recently_active_sequences_are_not_reaped`, `unowned_activation_warnings_are_rate_limited`) |
| Tightened tests | `remote_runtime_teardown_reaches_every_stage_past_a_silent_one` (one release), the three scheduler pipeline failure tests (command logs), `classifies_handshake_errors` (`stage-drop`) |
| Commits | 2 |

| Commit | Purpose |
|---|---|
| 968c011f | All four items of #2762, tests and docs |
| 8993404e | Review LOW items: the `fail_sequence` doc names which exits discard state; the reaping tests move from a 200 ms TTL and 300 ms wait to 1 s and 1.3 s |

## 3. Technical Decisions

### 3.1 A short acknowledgement wait for teardown only

**Context.** Cancel and release carry no result anyone acts on. The frame is written to the transport whether or not an acknowledgement arrives, a healthy but busy stage processes it in order, and a late acknowledgement is discarded by the nonce matcher of whatever exchange runs next. Waiting the full `stage_timeout` for it buys nothing except the warning that the stage did not answer.

| Option | Assessment |
|---|---|
| Lower `stage_timeout` globally | Would also cut the wait for `RunEntry`, which legitimately takes long (a large prefill, first-run CUDA kernel compilation). Making `stage_timeout` configurable is out of scope in #2762 |
| Fire-and-forget teardown (no wait at all) | Loses the per-stage log line that tells an operator which stage stopped answering |
| Per-peer circuit breaker that fails new prepares fast while a stage is known silent | Every request needs every stage, so it does not restore service; it adds recovery-probe state. Rejected in #2762 |
| **Chosen: `exchange_within` with `stage_timeout.min(2 s)` for `broadcast_best_effort`** | One parameter on an existing path; every teardown send (failed forward, prepare rollback, the scheduler's normal close) gets the bound; non-teardown commands keep `stage_timeout` |

The `min` keeps a test or a future configuration with `stage_timeout` below 2 s from waiting longer for teardown than for a forward.

**Trade-off.** A healthy stage that takes more than 2 s to acknowledge a release (for example while it is still finishing the `RunEntry` the coordinator gave up on) now produces a "did not acknowledge" warning on the coordinator even though the release will be applied. The warning is the only effect; the frame was already sent.

### 3.2 A failed forward discards the mapping before the teardown

**Context.** The `try_forward_with_sequence_id` contract from #2753 says an `Err` means the sequence's state was discarded. The runtime did not honor that: it tore down the stages but kept `request_ids`, so the scheduler's close tore them down again.

`fail_sequence` removes the `request_ids` entry first, then broadcasts cancel and release. The scheduler's close then finds no mapping and returns without sending anything. Each stage sees exactly one cancel and one release, and a silent stage costs two teardown waits instead of three full timeouts.

The two `serialize_mlx_array` exits that run before the `RunEntry` exchange deliberately do not call `fail_sequence`. Nothing reached a stage there, so no cancel is needed, and leaving the mapping lets the scheduler's close release the sequence exactly once. The finalization commit corrected the doc comment that first claimed every `Err` discards state.

In `forward_batched`, each row goes through `forward_sequence`, so a failing row discards only its own mapping and the `?` stops the batch there. The rows that had already succeeded and the rows not yet sent keep their mappings and are released once by the scheduler when it fails the batch. The mixed-shape refusal from #2753 follows the same rule: the offending row's forward succeeded, so its mapping stays for the scheduler's close.

### 3.3 Reap idle sequences only at the cap

**Context.** The leak in 1.2 matters only when it blocks admission. A stage that reaps on a timer would release sequences of a live but slow request for no gain while the cap still has room.

| Option | Assessment |
|---|---|
| Background reaper on a timer | Extra task and scheduling; reaps even when nothing is blocked |
| Reap on every `PrepareSequence` | An O(n) scan per request with no benefit below the cap |
| **Chosen: when a `PrepareSequence` for a new id finds the cap full, finish every sequence idle for at least 600 s, then refuse only if still full** | Zero cost until the cap is reached; the 4096-entry scan at the cap is microseconds; one `warn` with the count makes the event visible |

"Activity" is any message from the owner for that sequence: its prepare, its re-prepare, an accepted `RunEntry`, or an activation that passes the owner check (`touch_sequence` replaces the plain `contains_key` in those checks, so the update cannot be forgotten where ownership is checked). A sequence that another identity names is not touched. Reaping goes through `finish_request`, the same path as cancel and release, so caches are freed the same way.

**Trade-off.** A live request whose sequence sees nothing from its owner for 10 minutes and that is reaped fails its next `RunEntry` as "not prepared". Since #2753 that fails one request with the client-safe forward message, not the coordinator. In practice the scheduler issues a `RunEntry` for each active sequence every decode tick, so 10 minutes of silence means the coordinator is gone or wedged.

### 3.4 A fifth warning class instead of reusing `diagnostic`

The handshake warning budget (#2655, #2692; classes added in #2712) already splits warnings into classes so that one noisy source cannot push the others to debug. Unowned activations come from an authenticated peer, so they are neither handshake noise nor a handshake diagnostic. Putting them in `diagnostic` would let a misbehaving upstream starve the typed refusals operators rely on (probable secret mismatch, suite mismatch). `StageDrop` gets its own 5-per-minute window; the rest go to debug, and the existing rollover summary reports how many were suppressed.

### 3.5 Test knobs that do not exist in a release build

`StageOverrides` is `#[derive(Default)]` with every field under `#[cfg(test)]`, so in a release build it is an empty struct and `apply` and `log_guard` are empty `#[cfg(not(test))]` bodies. Production calls `spawn_service(.., StageOverrides::default())`; tests call `spawn_with_overrides`. This removes the `Option<usize>` parameter from the shipped signature without adding `allow(dead_code)`, and gives the idle TTL and the log capture a place to go. The gate is the non-test `cargo clippy --features cuda -p mlxcel --lib -- -D warnings`, which passed.

The `log_dispatch` field installs a `tracing` dispatcher as the service thread's default for the thread's life. That is what lets `unowned_activation_warnings_are_rate_limited` read what the stage logged at warn versus debug, since the stage runs on its own thread where a test-local subscriber would not see it.

### 3.6 Turn the unreachable branch into an ordinary error

Rather than delete the `reply_target(..) == None` branch and rely on an invariant, the PR keeps the lookup and maps `None` to `anyhow!("sequence {request_key} lost its owner during execution")`. If the invariant ever breaks (for example if the service starts handling frames concurrently), the failure takes the existing `report_activation_failure` path, which is rate-limited and cleans up, instead of a second hand-written drop path.

## 4. Verification

### 4.1 Failing-first evidence

The tests, the `#[cfg(test)]` builders they need and the item 4 `StageOverrides` refactor were run on the unchanged behavior first.

| Test | Before | After |
|---|---|---|
| `silent_stage_failure_is_bounded_by_one_stage_timeout` (1 s `stage_timeout`, 100 ms teardown wait, bound 1.7 s) | 4.006 s | within 1.7 s; stage log `prepare, run_entry, cancel, release` |
| `remote_runtime_teardown_reaches_every_stage_past_a_silent_one` | `prepare, run_entry, cancel, release, release` | `prepare, run_entry, cancel, release`; the stage after the silent one sees `prepare, cancel, release` |
| `idle_sequences_are_reaped_when_the_cap_is_reached` (cap 2) | third prepare refused: "pipeline stage already holds 2 prepared sequences" | acknowledged; both idle sequences released; a `RunEntry` for a reaped one is refused as not prepared |
| `recently_active_sequences_are_not_reaped` | same refusal | only the idle sequence is reaped |
| `unowned_activation_warnings_are_rate_limited` (20 unowned activations) | 20 of 20 at warn | 20 lines, at most `2 * HANDSHAKE_WARN_PER_WINDOW` at warn, zero executions |
| `remote_forward_error_fails_one_request_and_the_coordinator_serves_the_next` | command log with two releases; the module runs under `run_core_thread_or_abort`, so the mismatch aborted the binary (exit 134) | single release |

The rate-limit test allows two windows' worth of warns because the budget is process-wide and other stage-drop sites in the same test binary may have spent part of the current window, and one rollover can fall inside the test.

### 4.2 Local gate

GB10 (spark-102), `--profile test-fast --features cuda,test-utils`, every run under `gpu-lock run --tag issue-2762` with `--test-threads=1` and `MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1`.

| Suite | Result |
|---|---|
| The five tests above plus `prepare_beyond_the_sequence_cap_is_refused`, `final_stage_drops_result_of_unprepared_sequence`, `classifies_handshake_errors` (filter) | 23 passed |
| `server::batch::scheduler::scheduler_pipeline_failure_tests::` | 4 passed |
| `distributed::` | 1561 passed, 4 ignored |
| `distributed_integration` | 20 passed |
| `pipeline_e2e` | 30 passed |
| `server_exit_joins_worker` (real qwen3-0.6b-4bit) | 2 passed |
| `disaggregated_serving_e2e --include-ignored` | 2 passed |
| `disaggregated_router_e2e --include-ignored` | 7 passed |
| `pipeline_remote_real_models --include-ignored --skip glm_moe_dsa` | 9 passed, 1 failed in the batch run (`glm4` hit the 30 s `RunEntry` timeout while compiling CUDA kernels for the first time on a host at load average 43); 1 passed when rerun alone with the kernels cached |
| `pipeline_server_remote_real_models --include-ignored --skip glm_moe_dsa` | 7 passed |
| `pipeline_ci_multi_stage_real_models --include-ignored --skip glm_moe_dsa` | 4 passed |
| `cargo fmt --check`; clippy `-D warnings` for lib+tests (`cuda,test-utils`) and for the non-test lib (`cuda`) | clean |

After 8993404e: `distributed::pipeline::` 396 passed, the scheduler failure tests 4 passed, each reaping test 5 of 5 runs passed, fmt and clippy clean.

The `glm4` failure is on the `RunEntry` wait, which this PR does not change. It is still evidence that a 30 s fixed `stage_timeout` can be too short for a stage's first forward on a loaded host.

### 4.3 Not run

The manual check from #2762 (`--pp-auto 2` coordinator, `kill -STOP` on the last stage during a streamed request, the next request starting within about 34 s) was not run. The bounded path it would exercise is covered over real TCP by `silent_stage_failure_is_bounded_by_one_stage_timeout` and the scheduler timeout test, but the 34 s figure itself has not been measured against a real stopped process. Also not run: `glm_moe_dsa` (`glm5-4bit` does not load on this box), Thunderbolt and RDMA transports, macOS.

## 5. Remaining Gaps and Risks

- **Every new request still pays one `stage_timeout` while a stage stays silent.** The bound is per failed request. `prepare_sequence_state` uses `broadcast_ack` with the full `stage_timeout`, so each later request against the same silent stage waits 30 s at prepare (plus the 2 s rollback release per silent stage) before it fails. The coordinator serves nothing else meanwhile. This is the case the rejected circuit breaker would address; it was rejected because it cannot restore service when every request needs every stage, but it would turn 30 s per request into an immediate refusal.
- **A slow but healthy stage now draws a teardown warning.** See 3.1. An operator seeing "did not acknowledge ... release" should not read it as a lost release.
- **The 34 s figure assumes one silent stage.** Each silent stage adds two teardown waits (cancel and release) of up to 2 s, on top of the one `stage_timeout` spent detecting the failure.
- **Reaping can fail a live request.** Only one that saw nothing from its owner for 10 minutes while the stage was at its 4096-sequence cap. It fails as one request, not the coordinator.
- **The `stage-drop` budget is process-wide.** All stage services in one process share the 5-per-minute window, so a noisy stage can push another stage's drop warnings to debug. The rollover summary still reports the count.
- **Stale test comment.** The doc comment on `idle_sequences_are_reaped_when_the_cap_is_reached` (`src/distributed/pipeline/remote_service_tests.rs:1408`) still says "here 200 ms, cap 2", while 8993404e changed the TTL to 1 s. Behavior and assertions are correct; the comment should say 1 s.

## 6. Learning Points

### 6.1 Match the wait to what the caller does with the answer

A request/response transport tends to get one timeout for everything. The right timeout depends on what the caller does with the reply. For a forward, the reply is the result, so the caller must wait as long as a legitimate forward can take. For teardown, the frame's delivery is the effect and the reply is only a log line, so the caller can stop waiting early without losing anything. Splitting `exchange` into `exchange_within` made that a one-argument decision at each call site.

### 6.2 Make the error contract and the state agree

#2753 wrote down that an `Err` from a forward means the sequence's state was discarded, but the runtime kept the mapping, and the scheduler's close acted on it. Removing the mapping at the point of failure, in one helper used at every exit, makes the contract true by construction and turns the caller's close into a no-op instead of a duplicate. The exception (exits before anything was sent) is spelled out in the helper's doc so a later reader does not "fix" it.

### 6.3 Reclaim leaked state where it blocks, not on a schedule

A capacity cap plus lazy reclamation at the cap gives a bounded leak with no background machinery: nothing changes until leaked entries would refuse new work, and then exactly the entries idle past the TTL are released. The last-activity stamp lives in the same struct as the owner, updated by the same function that checks ownership, so no path can check ownership without refreshing activity.

```rust
fn touch_sequence(&mut self, request_key: &str) -> bool {
    match self.sequence_owners.get_mut(request_key) {
        Some(sequence) => {
            sequence.last_active = Instant::now();
            true
        }
        None => false,
    }
}
```

### 6.4 Compile test knobs out with an empty-in-release struct

A struct whose fields are all `#[cfg(test)]`, with `#[cfg(not(test))]` no-op methods, keeps one call site (`overrides.apply(&mut service)`) in both builds while guaranteeing the knobs do not exist in a shipped binary. The non-test clippy run with `-D warnings` is the check that nothing became dead code.

## 7. References

- Issue #2762: bound the pipeline failure path and stage state
- PR #2753 (issue #2727): a stage failure fails one request, not the coordinator; its report section 5 listed the residuals fixed here
- PR #2746 (issue #2726): the 4096 prepared-sequence cap and owner checks this PR extends with idle reaping
- `src/distributed/handshake_warn.rs` (#2655, #2692, #2712): the per-class warning budget
- `docs/distributed.md`, "What the secret protects" and the handshake warning section
