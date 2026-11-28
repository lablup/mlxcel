# PR #2720: Transport and remote pipeline robustness follow-ups

**Date**: 2026-10-28 (written before merge; the PR is open as a draft)
**Status**: Implementation complete at 89f5c196; the implementation review and the security and performance review both found no CRITICAL, HIGH or MEDIUM issues; the local gate passed on GB10 (spark-102); main merged in at 1ecb25cc; hosted CI pending at the time of writing
**Languages**: Rust, Markdown
**Risk Level**: Low to Medium (no wire format or protocol version change, but the coordinator accepts an `Error` from more stages than before, `ConnectionPool::acquire` changes its return type, and the TCP RPC path is rewritten)

## Executive Summary

PR #2720 (closes #2712) fixes six residuals that #2700, #2705 and #2708 left on the authenticated cluster transport, the remote pipeline and the disaggregated router. None of them lets a party without the cluster secret read or inject frames. They are failure latency, log visibility and API correctness problems, grouped in one PR because they share `src/distributed/`, its test harnesses and one verification pass.

The two changes with user-visible effect are on failure paths. An execution failure on a middle or final pipeline stage, or a middle stage that cannot forward downstream, now reaches the coordinator as the `Error` of the running exchange, so the request fails in well under a second instead of after the full `stage_timeout`. And TCP `rpc_call`, which before this PR timed out on every call because nothing read the response, now reads its response on the connection the request went out on.

The rest tightens the connection pool and the warning budget: capacity refusals are retried with backoff, a failed or cancelled send gives back its active slot through a drop guard, an idle connection with unread bytes is never reused, the router uses a fast-failover pool profile, router drops and capacity refusals get their own warning classes, the rate-limit summary says when the suppressing window ended, and an authenticated frame of an unknown kind is logged instead of closing the connection silently.

## 1. Problem Statement

All six items were re-checked against main at 752db4f9 when #2712 was written.

### 1.1 Non-entry stage errors ended in a coordinator timeout

Since #2694 the final stage sends logits straight to the coordinator as the `RunResult` of the `RunEntry` exchange. Errors did not follow. In `RemoteStageService::handle_activation`, an execution error on a middle or final stage sent `RemoteStageResponse::Error { nonce: 0, .. }` to the upstream neighbour, which drops anything that is not a command or an activation. A middle stage whose downstream `send_activation` failed returned `Err` to the service loop, which logged "dropping frame". In both cases nothing reached the coordinator, and `RemotePipelineRuntime::exchange` waited out `stage_timeout`. The coordinator could not have accepted such an `Error` anyway: `ExchangeMatch::reject_reason` required `Ack`, `State` and `Error` to come from the stage the command was sent to, which for `RunEntry` is stage 0.

### 1.2 `ServerAtCapacity` had no backoff and an unrated warning

`ConnectionPool::connect_with_retry` returned every handshake failure immediately, including the transient at-capacity verdict that #2705 introduced, and logged `Cluster handshake with peer {peer} failed` at warn on every attempt with no rate limit. A peer that freed a slot 100 ms later was still unreachable for that `acquire`.

### 1.3 Router drops shared the diagnostic budget, and the summary was mislabelled

`spawn_result_demux` spent `HandshakeWarnClass::Diagnostic` on every result frame it dropped, so a misconfigured peer streaming result frames could push a concurrent secret-mismatch warning to debug. The rollover summary is written by the next event of a class, which may arrive hours later, but the line said `... were logged at debug in the last 60s`.

### 1.4 An authenticated unknown kind closed the connection silently

#2708 made an authenticated frame with an unknown kind byte a distinct error, but as an untyped `anyhow!`. `handle_connection` logs only `FrameRejected` before closing, so a version-skewed member was invisible on the transport side.

### 1.5 Pool reuse and accounting

`ProtectedStream::peer_closed` reported a connection with unread bytes as alive even when a FIN followed them, and `acquire` reused it. `send`, `send_stream` and `rpc_call` returned on a write error through `?` without calling `release`, so `PeerPool::active` stayed incremented; a cancelled send leaked it the same way. The router bound its transport with `PoolConfig::default()`: 5 reconnect attempts from 100 ms doubling, about 3.1 s of sleep before a refused connect to a dead prefill node turned into failover.

### 1.6 TCP `rpc_call` never received its response

`rpc_call` wrote the request on a pooled outbound stream, released it, and waited on an `rpc_pending` oneshot for `connect_timeout`. The server wrote the `RpcResponse` back on the accepted socket, which is that same outbound stream on the client, and nothing on the client read it: responses were parsed only in `handle_connection`, which runs on accepted connections. Every TCP RPC timed out. Thunderbolt and RDMA delegate their `rpc_call` to TCP, and `bench::measure_latency` uses it.

## 2. Change Summary

| Item | Area | Change |
|---|---|---|
| 1 | `pipeline/remote_service.rs` | Both failure paths of `handle_activation` call `report_activation_failure`, which runs `finish_request` and, when the activation carries `reply_nonce`, sends the `Error` through `send_run_error` to `reply_target(request_key)` (the coordinator's authenticated identity, configured address as fallback). Without a nonce it logs one warn and sends nothing. The upstream `nonce: 0` send and `upstream_addr` are removed. The `Logits` branch also addresses its `RunResult` through `reply_target`. |
| 1 | `pipeline/runtime.rs` | `ExchangeMatch` gains `error_peers`: `&self.stage_peers[..]` for `RunEntry`, `std::slice::from_ref(peer)` otherwise. For `RunEntry` an `Error` must also carry `request_id == Some(<run id>)`. `reject_reason` is now a `match` over the response type. |
| 2 | `connection_pool.rs`, `cluster_handshake.rs` | `connect_with_retry` treats a chain containing `ServerAtCapacity` like a refused connect: it sleeps the current backoff and retries, and returns the error (type preserved) only when the budget is spent. Every other handshake failure stays terminal. The per-attempt warn goes through `handshake_warn_allowed`. |
| 3 | `handshake_warn.rs`, `router_front.rs` | New classes `RouterDrop` (`router-drop`) and `AtCapacity` (`at-capacity`), one window slot each (`CLASS_COUNT = 4`). `WarnDecision::previous_window_ended_ago` feeds the summary `N cluster warnings ({class}) were logged at debug during the 60s window that ended Ms ago`. The router demux uses `RouterDrop`. |
| 4 | `tcp_frame.rs`, `tcp_transport.rs`, `handshake_warn.rs` | `UnknownAuthenticatedKind { kind: u8 }` with the old `Display` text, returned by `read_frame`, classified `Diagnostic`, and logged by `handle_connection` as `Closing connection from {peer}: {err} (version skew?)` through the budget. |
| 5 | `tcp_frame.rs`, `connection_pool.rs`, `startup.rs` | `probe_idle() -> IdleProbe { Idle, PendingData, Closed }`, `peer_closed()` kept as `== Closed`. `acquire` reuses only `Idle` and returns a `PooledStream` guard (`Deref`/`DerefMut` to `ProtectedStream`, an `Arc<AtomicUsize>` per peer, decrement in `Drop`). `release` takes the guard. `PoolConfig::router()` (2 retries, 50 ms base, 2 s connect) is used by the router transport. |
| 6 | `tcp_transport.rs` | `rpc_call` writes the request and reads frames on the same pooled stream under one `timeout(connect_timeout, ..)` until an `RpcResponse` with the matching id; it releases the stream only after that and drops it on timeout or error. `rpc_pending` and the `RpcResponse` arm of `handle_connection` are removed; an `RpcResponse` on an accepted connection is logged at debug. |
| Docs | `docs/distributed.md`, `docs/CONTINUOUS_BATCHING.md` | Capacity backoff, the four warning classes and the new summary wording; the router pool profile next to the #2695 failover note. Module docs of the six touched modules updated. No Korean counterpart exists for these sections. |

### Statistics

| Item | Value |
|---|---|
| Files changed | 18 (9 source, 7 test, 2 docs) |
| Lines added / deleted | +1356 / -220 |
| Tests added | 19 `#[test]`/`#[tokio::test]` functions; 3 existing tests adjusted (`full_connection_cap_reports_capacity_to_the_client`, `panicking_rpc_handler_releases_its_connection_slot`, `peer_closed_probe_is_non_consuming`) |
| Commits | d7fd9803 (failing-first tests), 89f5c196 (the six items), 1ecb25cc (merge of main) |

The merge of main (1ecb25cc) brings in #2711 and #2718, which touch none of the 18 files in this PR.

## 3. Technical Decisions

### 3.1 The failing stage reports directly, and the coordinator widens only the `Error` rule

| Option | Pros | Cons |
|---|---|---|
| Relay the `Error` upstream to the entry stage, which answers the exchange | The coordinator keeps "reply comes from the addressed peer" | Needs per-request reply state on the entry stage, which #2694 just removed; one more hop per failure |
| Accept `Error` only from `result_peer` (the final stage) | Narrowest change to the matcher | A middle stage of a 3-stage pipeline can fail too, and would still time out |
| **Chosen: any configured stage may send the `Error` of a `RunEntry`** | One hop, no new stage state, covers every stage position | The coordinator accepts a reply from more identities |

The widening is no looser than the `RunResult` rule from #2694. The `Error` must still echo the per-exchange random nonce, which is carried only on that request's activations, and for `RunEntry` it must name the run's `request_id`. Every stage that can see the nonce could already stall the request into the same end state (a forward error). The `Error`'s `stage_index` and `state` are not used for matching. `Ack` and `State` keep the addressed-peer rule, and non-`RunEntry` exchanges keep `error_peers = [peer]`.

The new `request_id` check also applies to an `Error` from the entry stage. That is safe because every `RunEntry` error path in `handle_run_entry` sends `Some(request_id)`.

The stage side drops a failure that carries no `reply_nonce` with one warn. Sending it anyway would only produce a frame the coordinator discards, which is the same reasoning #2708 applied to logits without a nonce. The old upstream `nonce: 0` frame was never consumed by anything, so removing it changes no behavior beyond the log.

### 3.2 `reply_target` and `send_run_error` as the seam for #2714

Both helpers take the request key even though `reply_target` ignores it today. #2714 (bind reply targets per sequence) needs to route by request; with the key already at every call site, that change is local to `reply_target`. The `Logits` branch was moved onto the same helper in this PR so there is one addressing rule for both reply kinds.

### 3.3 At-capacity is transient, so it gets the refused-connect backoff

#2705 deliberately returned `ServerAtCapacity` without backoff so the budget would not be burned on handshakes. That kept `acquire` fast but turned a peer that is full for 100 ms into a failed send. The PR retries only the typed at-capacity verdict, with the same backoff and budget as a refused connect, and keeps every other handshake failure terminal: a wrong secret, identity or suite never becomes right on retry. When the budget is spent the error is returned with `ServerAtCapacity` still in the chain, because callers downcast it (`Transport::connect` logs and continues instead of failing). `full_connection_cap_reports_capacity_to_the_client` now uses `max_reconnect_attempts: 0` so it keeps testing the immediate verdict.

### 3.4 One budget class per warning source, and a summary that states its age

The budget is per class so that one noisy source cannot starve another. Router drops and capacity refusals are operational, not authentication diagnostics, and both can be produced at high rate by a single misconfigured but authenticated peer, so each gets its own window. `windows` is sized by `CLASS_COUNT`.

For the summary, a timer that flushes at rollover was rejected: the budget is a process-wide `static` called from synchronous code in `stage_assignment.rs`, with no runtime to own a task. The summary is still written by the next event of its class, and the line now says how long ago the suppressing window ended (`start + window`, saturating), which can be far above 60 s after a quiet period. The wording also changed from "cluster handshake warnings" to "cluster warnings", since two of the four classes are not handshake events. Any log alert that matched the old text needs updating.

### 3.5 A drop guard for the active count, and a three-state idle probe

An explicit `discard()` on each error path was rejected because it cannot cover cancellation: a `send` future dropped by a caller's timeout never reaches its error branch. `PooledStream` increments the per-peer `Arc<AtomicUsize>` at checkout and decrements in `Drop`, so the count is correct for success, error and cancellation alike. `release` moves the inner stream back to `idle` and lets the guard drop. The counter is an `Arc` so a guard outliving its `PeerPool` (after `ConnectionPool::shutdown` clears the map) decrements a counter nobody reads instead of a new pool's.

`probe_idle` splits the old "alive" answer in two. `PendingData` used to count as alive. Once item 6 consumes every RPC response on the connection that requested it, an idle outbound connection has no legitimate pending input: stray bytes would desynchronize the next read, and a FIN may sit behind them. So the pool drops such a connection and opens a fresh one. The probe is still a single non-consuming `MSG_PEEK` through `socket2`.

### 3.6 Read the RPC response on the request's connection

The protocol already sent the response back on the accepted socket, so the fix is on the client: read it there. The alternative of removing `rpc_call` from TCP was rejected because Thunderbolt and RDMA delegate to it and `bench::run_benchmark` would lose its latency figure. The read loop runs inside one `tokio::time::timeout(connect_timeout, ..)` covering write and read; a response with a different id is skipped at debug, any other frame kind is an error. The stream goes back to the pool only after the matching response; on timeout or error it is dropped, which is what lets item 5 treat pending data as unusable.

Removing `rpc_pending` also closes a hole the security review pointed out: any authenticated peer could previously resolve another call's pending entry by sending an `RpcResponse` with a guessed sequential id on an accepted connection.

### 3.7 Router pool profile values

`PoolConfig::router()` keeps the defaults except `max_reconnect_attempts: 2`, `reconnect_base_delay: 50ms` and `connect_timeout: 2s`. For a refused connect that is 50 + 100 ms of sleep before the error reaches the router's failover logic, instead of about 3.1 s. The 2 s connect timeout bounds a blackholed host; it is not the refused-connect path. The profile is applied only in the router branch of `start_server`; prefill and decode nodes keep the default.

## 4. Verification

### 4.1 Failing-first evidence

The behavioral tests were committed first (d7fd9803) and run against 752db4f9 plus only the test files. 11 of 13 failed with the expected mode:

| Test | Failure on 752db4f9 |
|---|---|
| `final_stage_execution_error_goes_to_coordinator` | the coordinator received nothing within 5 s |
| `final_stage_drops_execution_error_without_reply_nonce` | the upstream received the `nonce: 0` Error |
| `error_from_last_stage_fails_forward_promptly` | the exchange waited out the 5 s timeout |
| `server_at_capacity_is_retried_with_backoff` | handshake failed at once, no retry |
| `server_at_capacity_exhausts_retry_budget` | returned in under 60 ms, no backoff spent |
| `failed_send_releases_active_slot`, `cancelled_send_releases_active_slot` | `total_active` stayed 1 |
| `rpc_call_round_trips_over_authenticated_tcp` | `RPC call timed out` |
| `timed_out_rpc_call_drops_its_stream` | `total_idle` was 1 |
| `authenticated_unknown_kind_is_logged_before_close` | zero close lines captured |
| `acquire_drops_idle_connection_with_pending_data` | the idle connection was reused |

The two that passed on both sides (`error_from_last_stage_with_wrong_{nonce,request_id}_is_dropped`) guard the new acceptance rule rather than demonstrate the bug. Tests that use a new API (`HandshakeWarnClass::RouterDrop`, `WarnDecision::previous_window_ended_ago`, `UnknownAuthenticatedKind`, `PoolConfig::router`, `IdleProbe`) cannot compile on 752db4f9; the symbol's absence there is the evidence. `authenticated_unknown_kind_is_logged_before_close` captures at debug level with a scoped subscriber, because the process-wide budget may already be spent by earlier tests in the same binary.

### 4.2 Local gate

GB10 spark-102, `--profile test-fast --features cuda,test-utils`, built outside the GPU lock and run under `gpu-lock` with `--test-threads=1`, `MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1`, at 89f5c196:

| Target | Result |
|---|---|
| `cargo fmt --check`, `cargo clippy --lib --tests -D warnings` | clean |
| `--lib distributed::` | 1513 passed, 4 ignored |
| `--lib server::router_front` | 35 passed |
| `zero_config_cluster_init --include-ignored` | 7/7 |
| `pipeline_remote_real_models --include-ignored` | 10/10 (`glm_moe_dsa` skipped) |
| `pipeline_server_remote_real_models --include-ignored` | 7/7 (`glm_moe_dsa` skipped) |
| `pipeline_ci_multi_stage_real_models --include-ignored` | 4/4 |
| `disaggregated_serving_e2e` / `disaggregated_router_e2e` | 2/2, 7/7 |

Greedy parity on the real-model targets is unchanged against #2708. The skipped `glm_moe_dsa` tests need a 418.6 GB checkpoint that cannot load on a 128 GB GB10. No performance measurement was taken; the per-token path is unchanged (only failure paths, pool bookkeeping and the RPC path moved).

## 5. Remaining Gaps and Risks

- **Pre-forward failures on a non-entry stage still time out (LOW).** In `handle_activation`, a missing downstream peer, a `sequence_length` or `RequestId` parse failure, a `serialize_mlx_array` failure, and a `deserialize_wire_tensor` failure on the incoming activation still propagate with `?`. They end in the coordinator's stage timeout and, for the ones after `mark_request_started`, leave the request's caches until the coordinator's cleanup. #2712 named only the execution error and the failed send. Routing them through `report_activation_failure` is a small follow-up.
- **The router's 500 ms failover criterion is checked at unit level only.** `router_pool_config_fails_fast` asserts an `acquire` of a refused port fails in under 500 ms with the router profile. `disaggregated_router_balances_and_survives_node_failure` passes but asserts no timing, so end-to-end failover latency is not measured.
- **Non-entry stage errors are covered by harness tests, not by a real-model run.** No real-model target injects a stage failure; the stage side is tested with `StubOutput::Error` and the coordinator side with the correlation harness.
- **Capacity retries amplify handshakes (LOW).** At-capacity retries have no jitter and each is a full connect plus handshake, so N concurrent `acquire`s against a full peer make up to 6N handshakes (3N with the router profile), each producing an unrated "max connections reached" warn on the server. Bounded and authenticated-only; rate-limiting that warn under `AtCapacity` and adding jitter would fix it.
- **`Transport::connect` logs "Failed to pre-connect" without the rate limiter (LOW).** Once per `connect` call after the budget is spent, not per attempt.
- **`PooledStream` does not record its peer (LOW).** `release(other_peer, guard)` would put the connection in the wrong idle list. The active count stays correct and every current caller releases under the key it acquired with.
- **`rpc_call` reads with the 256 MiB `max_frame_size` (LOW).** The length header is unauthenticated until the body is opened, so an on-path party can force one large allocation per in-flight call, the same exposure accepted connections already have.
- **The no-nonce failure warn on a stage is not rate limited (LOW),** like the logits-drop warn next to it. Only the authenticated upstream can trigger it.
- **`rpc_call`'s deadline is `connect_timeout`.** This predates the PR (the old oneshot wait used it too), but it means the router profile's 2 s also bounds any RPC the router makes, and `acquire`'s own retry budget runs before that deadline starts.

## 6. Learning Points

### 6.1 Make slot accounting a value, not a call

A counter that every exit path must remember to decrement fails on the path nobody wrote: cancellation. Tying the decrement to `Drop` of the checked-out value makes success, `?` and a dropped future the same case:

```rust
pub struct PooledStream { stream: Option<ProtectedStream>, active: Arc<AtomicUsize> }
impl Drop for PooledStream {
    fn drop(&mut self) { self.active.fetch_sub(1, Ordering::SeqCst); }
}
```

`cancelled_send_releases_active_slot` drops a 32 MiB send through a 200 ms timeout to pin the cancellation case, which an explicit `discard()` would not have covered.

### 6.2 A request/response protocol reads the response where it wrote the request

The original TCP RPC had a correct server and a pending map waiting for a response that could only arrive on a socket the client never read. When the server answers on the accepted socket, the client must read the same connection, hold it until the matching response, and drop it on timeout so a late reply cannot corrupt the next user of a pooled connection. The pooled connection then becomes a precise idle signal: any pending byte means something went wrong.

### 6.3 Route errors the same way as results

When #2694 moved `RunResult` to "the stage that has the answer sends it straight to the coordinator", errors stayed on the old path and silently became timeouts. Whenever a protocol changes who replies, the failure reply has to move with the success reply, and the matcher's per-type source rules should be written as one `match` so that each reply kind's rule is visible next to the others.

### 6.4 Rate-limit summaries written lazily must say when

A summary written by the next event, not by a timer, describes the past at an unknown distance. Reporting the age of the window keeps the line accurate without introducing a background task into a synchronous, process-wide static.

## 7. References

- Issue #2712: transport and remote pipeline robustness follow-ups
- PR #2700 (closes #2695): disaggregated serving and router e2e fixes, `ProtectedStream::peer_closed`
- PR #2705 (closes #2692): pre-auth admission hardening, `ServerAtCapacity`, the handshake warning budget
- PR #2708 (closes #2694): final stage sends `RunResult` to the coordinator, `reply_nonce`, `coordinator_identity`, authenticated unknown kind error
- Issue #2637: authenticated cluster transports and response correlation (the `ExchangeMatch` rules)
- Issue #2714: authenticate handshake verdicts and bind reply targets (consumer of `reply_target`)
- Issues #2715, #2716: router parity residuals and tensor decode limits, whose regions this PR leaves untouched
