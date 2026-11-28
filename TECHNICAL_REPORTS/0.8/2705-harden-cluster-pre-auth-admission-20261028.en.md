# PR #2705: Harden cluster pre-auth admission and rejections

**Date**: 2026-10-28 (written before merge; the PR is open as a draft)
**Status**: Implementation complete; implementation and security reviews found no CRITICAL or HIGH issues; local tests passed on GB10 (spark-102); hosted CI in progress at the time of writing
**Languages**: Rust, Markdown
**Risk Level**: Medium (handshake wire version bump 3 to 4, so nodes built before and after this change cannot pair; a router whose peer list does not match the senders' bound identities now drops their result frames)

## Executive Summary

PR #2705 (closes #2692) closes five residuals that the security review of #2674 left on the authenticated cluster transport in `src/distributed/` and on the disaggregated router's result demux. None of them lets a party without the cluster secret read or inject frames. They are DoS, log-visibility and member-isolation hardening on a surface that already requires the secret, grouped into one PR because they share one subsystem and one verification pass.

1. The per-source pending-handshake limit keys IPv6 sources by /64 instead of by full address, so a host that owns a /64 can no longer bypass it by rotating addresses.
2. The unread `active_conns` counter in `accept_loop`, which leaked when a connection task panicked, is deleted. Every per-connection resource is now released by drop.
3. The single process-wide "5 warnings per minute" limiter is replaced by a new `handshake_warn` module with two independent budgets, `Noise` and `Diagnostic`, so junk connections cannot push a probable-secret-mismatch line down to debug.
4. The router refuses to build on a transport bound without the secret, and its result demux drops frames whose authenticated sender is not one of `--prefill-peers` or `--decode-peers`.
5. The `max_connections` check moves inside the handshake, after the client authenticates. A full listener now answers with a new `VERDICT_AT_CAPACITY` (0x02) byte and the client gets a typed `ServerAtCapacity` error instead of an accepted handshake followed by a closed socket. `PROTOCOL_VERSION` goes from 3 to 4.

The PR is 17 files, +1104 / -156, in two authored commits (`699d208d` implementation, `80974c3c` review wording) plus a merge from `main` (`9900bb89`).

## 1. Problem Statement

All five items were confirmed on `main` @ 54d63b0a when #2692 was filed.

### 1.1 IPv6 rotation bypassed the per-source pending limit

`PendingHandshakeLimiter::try_acquire` (from #2655) counted in-flight handshakes per canonical source address, capped by `max_pending_handshakes_per_ip` (default 8). An IPv6 host normally controls a whole /64 and can source each connection from a different address, so it could take every slot in the global pending pool (default 64) by itself. The module doc already admitted the gap.

### 1.2 A counter that leaked and was never read

`accept_loop` kept `active_conns: Arc<AtomicUsize>`, incremented on accept and decremented by hand on three exit paths. A panic inside a connection task, for example inside a registered RPC handler, skipped the decrement. The workspace builds with `panic = "unwind"`, so tokio drops the panicked task's future; the semaphore permit and the pending slot are RAII and were already released correctly. The only casualty was `active_conns`, which nothing read. It was a latent trap: the first metric or limit built on it would have drifted upward on every panic.

### 1.3 Junk connections hid the line an operator needs

`handshake_warn_allowed()` in `tcp_transport.rs` was one budget of 5 warnings per 60 s shared by every rejection: pending-pool refusals, all handshake failures, and frames that failed authentication. Anyone who can open sockets can spend that budget with bad magic bytes, malformed hellos or stalls to the 5 s timeout. A genuine `ClientAbortedAfterServerProof` ("the peer most likely holds a different cluster secret") arriving after a few junk sockets was logged only at debug.

### 1.4 The router demux trusted any secret holder

`spawn_result_demux` delivered a `ResultFrame` to whichever pending request matched its `request_id`, regardless of `from`. Request ids come from `next_id.fetch_add`, so they are sequential and guessable. Any holder of the secret (for example a pipeline stage sharing the same secret file) could inject tokens into another client's streamed response. Separately, `RouterState::build` accepted any `TcpTransport`; the only thing keeping the router's transport authenticated was the resolver upstream and a comment in `startup.rs`, and both router test fixtures bound without auth.

### 1.5 A full connection cap looked like a successful handshake

`server_handshake_inner` wrote `VERDICT_ACCEPT`, and only after it returned did `accept_loop` try `conn_semaphore.try_acquire_owned()`. On failure it dropped the socket. The client saw a successful handshake, then a broken connection on its first send, with nothing in the error to say the peer was full.

## 2. Change Summary

| Area | Change |
|------|--------|
| `src/distributed/pending_handshakes.rs` | New `pending_source_key(ip)`: canonicalize (IPv4-mapped IPv6 stays IPv4), return IPv4 as is, mask IPv6 to its /64 with `u128::from(v6) & !((1u128 << 64) - 1)`. Used by `try_acquire` and the test helper `pending_from`. `PendingSlotRefusal::PerIp` carries the key and prints IPv6 keys as `2001:db8:1:2::/64`. |
| `src/distributed/tcp_transport.rs` | `active_conns` and its three decrements removed with the `AtomicUsize` import. `accept_loop` calls `server_handshake_admitting(..., FrameChannel::Transport, \|\| conn_semaphore.try_acquire_owned().ok())` and logs `ConnectionLimitReached` unconditionally (only an authenticated peer can reach it). New `TcpTransport::is_authenticated()`. Old limiter deleted; call sites pass a class. |
| `src/distributed/handshake_warn.rs` (new, 202 lines) | `HandshakeWarnClass { Noise, Diagnostic }`, `handshake_warn_class(&anyhow::Error)` that walks the error chain for the five typed refusals, `WarnBudget` with a two-slot per-class window and an injectable `allow_at(class, now) -> WarnDecision`, and the process-wide `handshake_warn_allowed(class)` that also logs `N cluster handshake warnings (<class>) were logged at debug in the last 60s` on rollover. |
| `src/distributed/cluster_handshake.rs` | `VERDICT_AT_CAPACITY = 0x02`, `PROTOCOL_VERSION = 4`. New `pub(crate) server_handshake_admitting<P>(..., admit: impl FnOnce() -> Option<P>) -> Result<(String, SessionKeys, P)>`; `server_handshake_on` wraps it with `\|\| Some(())`. Client maps verdict 0x02 to `ServerAtCapacity`; server returns `ConnectionLimitReached { peer }`. Wire diagram and module doc updated. |
| `src/distributed/stage_assignment.rs` | Imports the new limiter; pending refusals draw `Noise`, handshake failures `handshake_warn_class(&err)`, rejected frames `Diagnostic`. Handshake admission unchanged (wrapper admits everything). |
| `src/distributed/connection_pool.rs`, `mod.rs` | Comment on the no-retry handshake path covers the transient `ServerAtCapacity` case; `ServerAtCapacity` and `ConnectionLimitReached` re-exported; `handshake_warn` is `pub(crate)`. |
| `src/server/router_front.rs` | `RouterState::build` bails when `!transport.is_authenticated()`. New `result_senders: HashSet<SocketAddr>` (prefill and decode peers with canonical IPs) and `accepts_result_from(&str)`. The demux drops other senders with a `router: dropping result frame` warn through the `Diagnostic` budget, using tracing fields rather than a formatted string. |
| `src/bin/mlx_server.rs`, `src/server/startup.rs` | `--prefill-peers` / `--decode-peers` help says each entry must be the address the peer connects from. Startup comment credits `RouterState::build` with enforcing auth. |
| `docs/distributed.md`, `docs/CONTINUOUS_BATCHING.md` | /64 counting and its shared-subnet consequence, the router sender check and what it does and does not authenticate, the at-capacity outcome, the two warning classes, version 4, and the residual risk that a stranger can still reach the `Diagnostic` class. |
| Tests (5 `_tests.rs` files, +523) | 13 new tests across `pending_handshakes`, `handshake_warn`, `cluster_handshake`, `tcp_transport::cluster_auth` and `server::router_front`; both router fixtures now bind with a random `ClusterSecret`. |

## 3. Technical Decisions

### 3.1 Key by /64, not a second per-prefix tier

The issue rejected adding a separate per-prefix limit with its own config field. One knob, keyed by a coarser source, covers the case. The cost is explicit in the docs: cluster nodes inside one IPv6 /64 now share the per-source budget of 8 in-flight handshakes, so a large IPv6 cluster in one /64 may need a higher `max_pending_handshakes_per_ip`. Since the pending pool only covers the few seconds of a handshake, this matters only during a mass reconnect. Canonicalizing before masking matters: without it an IPv4-mapped IPv6 address (`::ffff:a.b.c.d`) would be masked to `::/64` (its upper 64 bits are zero) and every IPv4 client on a dual-stack socket would share one key.

### 3.2 Delete the counter, do not fix it

Making the decrement panic-safe (a drop guard) would have kept a value nobody reads. The issue instead records that a future active-connection metric should derive from `max_connections - conn_semaphore.available_permits()`, which cannot drift because the permit is already RAII. The regression guard `panicking_rpc_handler_releases_its_connection_slot` passes before and after the change by design; it pins the property (a panicking handler leaks no slot) rather than the deletion.

The guard's observable had to change from the issue's text. `TcpTransport` parses `RpcResponse` frames only in `handle_connection`, which runs on accepted connections; the pooled outbound connection an `rpc_call` writes on is never read. So no TCP client can receive an RPC echo, and the test uses client B's `send` reaching `server.recv()` within 2 s instead. This is a pre-existing limitation of RPC over TCP (the only in-repo caller, `bench.rs`, ignores the result), recorded in the PR and the issue but not fixed here.

### 3.3 Two warning classes instead of per-source first-event logging

The alternative, always warning on the first event from each source, needs a per-source map that is either unbounded or evicting, and IPv6 rotation defeats both. Two fixed windows (a `Mutex<[Option<Window>; 2]>`) cost no per-source state. `WarnBudget::allow_at` takes `now` so the tests can drive 50 noise events and a 61 s rollover without sleeping; the process-wide wrapper is the only caller of `Instant::now()`.

Classification walks the `anyhow` chain with `downcast_ref` for `ClientAbortedAfterServerProof`, `HandshakeRejected`, `IdentityNotOwned`, `FrameSuiteMismatch` and `FrameRejected`. Everything untyped (bad magic, malformed hello, timeout, I/O) is `Noise`. Pending refusals are always `Noise`; post-handshake frame rejections and router drops are always `Diagnostic`.

The split is not a security boundary. `IdentityNotOwned` and `FrameSuiteMismatch` are decided from the client hello alone, and `ClientAbortedAfterServerProof` only needs a well-formed hello naming the attacker's own IP followed by a close. A stranger can therefore still saturate `Diagnostic`. The review commit corrected the docs to say so; the rollover summary line is the remaining signal in that case.

### 3.4 Admission as a callback inside the handshake

`server_handshake_admitting` takes `admit: impl FnOnce() -> Option<P>` and returns the permit with the keys. Three properties follow from that shape:

- `admit()` is called after the client authenticator verifies (a mismatch returns `VERDICT_REJECT` first), so only a secret holder can learn the capacity state. `admission_is_not_consulted_before_the_client_authenticates` pins this.
- The permit is a local that travels out with the result. If the `VERDICT_ACCEPT` write fails or the 5 s timeout fires after `admit()`, the permit is dropped with the future; nothing is counted by hand.
- Listeners that do not cap connections (the stage assignment channel) keep calling `server_handshake_on`, which passes `|| Some(())`, so their behavior is unchanged.

A generic permit type keeps `cluster_handshake.rs` free of any dependency on the semaphore in `tcp_transport.rs`.

### 3.5 `ServerAtCapacity` is not fatal, and not retried with backoff

`Transport::connect` treats `HandshakeRejected` as fatal because no retry fixes a secret mismatch. `ServerAtCapacity` is a separate type so `connect` logs a warning and continues. In `connect_with_retry` it takes the same no-backoff exit as every handshake failure: the error goes straight to the caller and the next `acquire` opens a fresh connection. The review commit adjusted the docs and the pool comment, which had first called this "retryable" without saying who retries.

### 3.6 Bump the wire version rather than reuse a reject byte

Before this PR, any verdict other than 0x01 was a rejection. An old client receiving 0x02 would report `HandshakeRejected` ("handshake was altered in transit"), which is misleading and fatal at `connect`. Bumping `PROTOCOL_VERSION` to 4 makes a mixed pair fail at the hello with the explicit version message. #2650, #2674, #2684 and this change all landed after v0.8.0-beta.1, so no released build speaks version 3. Builds from `main` between those PRs and this one cannot pair with builds after it, which matches the existing "mixed versions are unsupported" rule.

### 3.7 Router sender check on canonical IP and port

`result_senders` stores each configured peer as `SocketAddr::new(ip.to_canonical(), port)`, and `accepts_result_from` parses `from` the same way. That strips IPv4-mapped forms, scope ids and flowinfo on both sides; an unparseable `from` is refused, and a configured `0.0.0.0` entry matches nothing because a bound identity is never unspecified. Only the source IP is authenticated (the handshake binds it to the TCP connection); the port is the listener port the sender advertised. Two secret holders behind one IP can therefore claim each other's port. The docs state this, and it is accepted under the threat model in which secret holders are members.

Rejected: binding each request to the exact prefill or decode node it was routed to. That needs per-request routing state in the demux, and the protocol does not fix which role sends which phase. The check also covers only the router's result intake; the prefill request intake and the decode handoff intake keep trusting any authenticated member (out of scope in #2692).

The behavior change for operators: a peer whose bound identity differs from its configured address (a multi-homed node, or one bound to `0.0.0.0` that reaches the router over another interface) now has its frames dropped, and the request fails with the existing handoff timeout. The `router: dropping result frame` warn is the diagnostic.

## 4. Verification

From the PR body (GB10 spark-102, `--profile test-fast`, builds outside `gpu-lock`, runs under `gpu-lock run --tag issue-2692`):

- `cargo fmt --check` and `cargo clippy --profile test-fast --features cuda -p mlxcel --lib --tests -- -D warnings`: clean, rerun after the review commit.
- The 13 new or touched tests by name: 13 passed. `--lib distributed::`: 1473 passed, 0 failed, 4 ignored. `--lib server::router_front`: 35 passed. `--test zero_config_cluster_init`: 4 passed, 3 ignored. `tests/disaggregated_serving_e2e.rs`: 2 passed. `tests/disaggregated_router_e2e.rs`: 7 passed.
- Real-model parity (`MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1`, `--ignored`): `pipeline_remote_real_models` 10 passed and 1 failed, `pipeline_server_remote_real_models` 7 passed and 1 failed, `pipeline_ci_multi_stage_real_models` 2 passed. Every llama, gemma3, gemma4, qwen3, qwen3.5 and glm-4.7 parity test passed.
- The two failures are not a regression. They are the `glm_moe_dsa` tests panicking in `repo_model_dir` because the `glm5-4bit` checkpoint is absent: `mlx-community/GLM-5-4bit` is 418.6 GB and cannot be loaded on this 128 GB GB10 box, and the tests predate #2674.
- Manual three-process loopback (router, prefill and decode sharing one `--cluster-secret-file`, `qwen3-0.6b-4bit`, 127.0.0.1): `POST /v1/chat/completions` returned HTTP 200 with `"2 + 2 = 4."`, and the router log had no `dropping result frame` line. This shows the sender check does not drop legitimate frames in the loopback setup.

Failing-first evidence: the fix hunks were reverted temporarily on the branch and the tests run by name. `ipv6_sources_in_one_64_share_the_per_source_limit`, `ipv6_sources_in_different_64s_are_counted_separately`, `noise_flood_does_not_suppress_diagnostic_warnings`, `build_refuses_an_unauthenticated_transport` and `result_demux_drops_frames_from_unconfigured_senders` all failed. The admission tests (`server_at_capacity_sends_capacity_verdict`, `admission_is_not_consulted_before_the_client_authenticates`, `full_connection_cap_reports_capacity_to_the_client`) and the classifier tests do not compile without the new API. `rollover_reports_suppressed_count` and the panic guard pass on both sides by design.

Pre-merge gate on the merged head `9900bb89` (orchestrator log, partial at the time of writing): `verify-fmt`, `verify-versions`, `verify-env-var-docs`, `verify-kernel-dtype-keys`, `verify-kernel-port-dispatch`, `verify-binary-assets` and `verify-windows-paths` all exited 0. The contract test build was still compiling, so no test results from that run are cited here.

## 5. Remaining Gaps and Risks

- **Version 3 and 4 nodes cannot pair.** Rolling upgrades of a cluster built from `main` need every node updated together. No released version is affected.
- **The verdict byte is unauthenticated**, as it was before. An on-path attacker can swap 0x01 and 0x02, but it could already drop the connection, so this adds no capability.
- **The `Diagnostic` budget is reachable without the secret** (section 3.3). The two classes protect against cheap junk, not against a stranger who sends a well-formed hello.
- **Router drops share the process-wide `Diagnostic` budget** with handshake diagnostics. A misconfigured peer sends a stream of result frames per request, each one dropped and counted, so it can use up the 5 warnings per minute and push a concurrent secret-mismatch line to debug. The rollover summary still counts them, but it does not say which kind of event was suppressed.
- **The rollover summary is emitted lazily.** It is logged on the first event of the next window for that class. If a flood stops, its suppressed count is not reported until the next event of that class, possibly much later, and the message's "in the last 60s" wording is then inaccurate.
- **`ServerAtCapacity` has no backoff.** A caller that retries `send` in a tight loop against a full peer will run a full handshake per attempt, and `connect_with_retry` logs `Cluster handshake with peer ... failed` at warn on each one without rate limiting. Only secret holders are involved, and no in-repo caller is known to loop this way, but nothing prevents it.
- **Port spoofing between co-located members** (section 3.7) is accepted by design.
- **RPC over TCP cannot return responses** (section 3.2). Pre-existing, documented in the PR, not tracked by a dedicated issue as far as this report found.
- **Real-model parity for `glm_moe_dsa`** remains unverified on this hardware because the checkpoint does not fit.

## 6. Learning Points

- **Put admission inside the authenticated exchange.** Checking capacity after the handshake leaks nothing but gives a bad client experience; checking it before leaks capacity state to strangers. A callback invoked between "client verified" and "verdict written" gets both right, and returning the permit with the session keys keeps the resource lifetime in RAII.
- **Rate limiters that share a budget across trust levels invert their purpose.** The original limiter protected the log from strangers by silencing the operator. Splitting budgets by what the line is worth, not by who sent it, is cheap when the classes are few and fixed.
- **Canonicalize before masking.** IPv4-mapped IPv6 addresses must become IPv4 before any prefix logic, or every IPv4 client on a dual-stack listener collapses into one /64.
- **Delete unread state instead of hardening it.** A counter nobody reads still has to be kept correct by every future edit; derive the value from the RAII resource when it is needed.
- **Inject time into windowed logic.** `allow_at(class, now)` let the rollover and flood tests run deterministically with no sleeps.

## 7. References

- PR #2705, issue #2692
- Origin: #2674 (#2655 pre-auth pending limit and typed rejections), #2684 (#2656 per-frame AEAD), #2650 (cluster secret for distributed roles)
- Key files: `src/distributed/handshake_warn.rs`, `src/distributed/cluster_handshake.rs`, `src/distributed/pending_handshakes.rs`, `src/distributed/tcp_transport.rs`, `src/server/router_front.rs`, `docs/distributed.md`
