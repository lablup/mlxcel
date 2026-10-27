# PR #2674: Cap pre-auth handshake slots, type handshake rejections, authenticate disaggregated roles

**Date**: 2026-10-27
**Status**: Completed (report written on the PR branch before merge)
**Languages**: Rust, Markdown
**Risk Level**: High (breaking for operators of the disaggregated prefill, decode and router roles; handshake wire version bumped to 2)

## Executive Summary

PR #2650 put a mutual HMAC-SHA256 handshake with a pre-shared cluster secret in front of every pipeline-parallel TCP connection. Its review left three residuals in the same subsystem, which issue #2655 groups as items a, b and d (item c moved to #2658). This PR closes all three. (a) With auth on, a socket that has not finished the handshake no longer holds one of the 512 `max_connections` permits; it holds a slot in a separate pending pool (64 global, 8 per source IP) for the handshake only, so a flood of silent sockets cannot lock authenticated peers out of the connection cap. (b) A client whose advertised identity is not its source IP now receives a 9-byte rejection record and fails with a typed `HandshakeRejected` within 1 s, instead of seeing a bare EOF and only warning; the server log separates "identity not owned", "probable secret mismatch", "altered transcript" and "closed before the handshake". The wire version moves from 1 to 2. (d) The disaggregated `--node-role prefill`, `decode` and `router` transports now run the same handshake, and those roles refuse to start without a secret. Single-node serving (`hybrid` or no `--node-role`) is unchanged.

## 1. Problem Statement

Issue #2655 verified each residual against the code at the time (references below are to the pre-PR line numbers quoted in the issue).

1. **Pre-auth sockets exhaust the connection cap (item a).** `accept_loop` took a `max_connections` permit (default 512) with `try_acquire_owned` at accept time, before the spawned task ran `server_handshake`, and held it for the whole handshake (up to `HANDSHAKE_TIMEOUT`, 5 s). 512 sockets that connect and send nothing therefore held every permit, and a real stage connecting during that window was dropped before it could authenticate. The connecting side saw `PeerClosedBeforeHandshake`, which `Transport::connect` only warns about, so the coordinator started and then failed requests.
2. **Identity-binding failure was not a typed, distinguishable rejection (item b).** The client message "server rejected the cluster handshake (secret mismatch or identity binding failed)" was reached by neither case it named. A secret mismatch is detected by the client first (the server's tag does not verify) and the client hangs up, so the server only logged "connection closed before the client authenticator arrived", indistinguishable from a network drop. An identity-binding failure made the server return an untyped `anyhow` error and write nothing, so the client got EOF before the server hello (`PeerClosedBeforeHandshake`), `Transport::connect` warned instead of failing, and the existing test `handshake_rejects_client_advertising_foreign_ip` only finished because the client hit the 5 s timeout. The `VERDICT_REJECT` path was reachable only when the server's tag verified but the client's did not, which means the transcript was altered in transit.
3. **Disaggregated transports ran no handshake (item d).** The prefill/decode worker (`run_disaggregated_serving_role` in `model_worker.rs`) and the router (`start_server` in `startup.rs`) built `TcpTransportConfig` with `auth: None`. Any host that could reach a role port could put a frame (for example a `DecodeMetaFrame`) into the role loop. The docs described a trusted-segment model and said a follow-up would wire the secret in "as an opt-in".

Item c (`broadcast_beacon_loop` has no production sender, so `--pp-auto --cluster-discovery=mdns` always times out) is not part of this PR; it is tracked in #2658, which depends on this one.

## 2. Technical Decisions

### Item a: a separate pending pool, not a per-IP limit alone

`src/distributed/pending_handshakes.rs` adds `PendingHandshakeLimiter` (`pub(crate)`), and `TcpTransportConfig` gains `max_pending_handshakes` (default `DEFAULT_MAX_PENDING_HANDSHAKES = 64`) and `max_pending_handshakes_per_ip` (default `DEFAULT_MAX_PENDING_HANDSHAKES_PER_IP = 8`). With auth set, the accept path is now:

```text
accept -> pending.try_acquire(peer_ip)        refused: close socket, rate-limited warn
       -> spawn task: server_handshake (<= 5 s, wrapped in HANDSHAKE_TIMEOUT)
       -> drop pending slot (on every outcome)
       -> Ok(identity): conn_semaphore.try_acquire_owned()   full: close, "max connections reached"
       -> handle_connection
```

The issue rejected a per-IP limit on its own for two reasons: real stages and an attacker can share one NAT address, and a flood from many sources would still exhaust the main pool. The global pending pool is what keeps the 512 authenticated slots out of an unauthenticated peer's reach; the per-IP sub-limit only stops one source from taking the whole pending pool.

Implementation details a maintainer should keep:

- **Acquisition order and rollback.** `try_acquire` increments the per-IP count under a `std::sync::Mutex` first, wraps it in a `PerIpGuard`, then tries the global semaphore. If the global pool is full, returning the error drops the guard, which decrements the count again. The guard also removes a key when its count reaches zero, so the map never retains past sources. The lock is never held across an `.await`.
- **Key canonicalization.** The per-IP key is `peer_ip.to_canonical()`, so an IPv4-mapped IPv6 source (`::ffff:a.b.c.d`) counts against its IPv4 address, the same key `bind_advertised_identity` uses.
- **Slot lifetime.** The slot is dropped right after `server_handshake` returns, whatever the outcome, and `server_handshake` wraps the whole exchange in the 5 s timeout. The security review confirmed release on timeout and unwind as well, since it is plain RAII.
- **Zero bounds.** A bound of 0 would close every inbound socket at accept and produce a listener that silently accepts nobody. `PendingHandshakeLimiter::new` refuses it, and `TcpTransport::bind` builds the limiter before it binds the listener, so the error surfaces at startup. This only applies when `auth` is set; without auth the two fields are documented as ignored.
- **Without auth, nothing changes.** The `None` arm takes the `max_connections` permit at accept exactly as before and never constructs a limiter, which keeps `default_config_stays_unauthenticated` meaningful.
- **Reuse.** The limiter is `pub(crate)` with a two-method API so #2658's coordinator-to-peer listener can use the same accounting.

The trade-off is stated explicitly in the docs after review: a flood from 8 or more source IPs (at the defaults) fills the 64-slot pending pool, and while it is full a new connection from any source, including a real peer reconnecting, is closed at accept until a slot frees (at most 5 s per slot). The flood cannot take slots from established authenticated connections and cannot inject frames. The first draft of `docs/distributed.md` said such a flood "cannot starve authenticated ones"; the implementation review flagged that as overstated and the wording was corrected.

### Item b: an explicit rejection record before any server proof

The server cannot reuse the verdict byte to report an identity mismatch. The verdict comes after both tags, and the property from #2650 is that a client lying about its address never receives a server nonce or authenticator. The rejection therefore has to replace the server hello. The wire now allows:

```text
server -> client : REJECT_MAGIC "MLXCREJ\0"(8) | reason u8 (0x01 = REJECT_IDENTITY_NOT_OWNED)
```

- **Same length as `MAGIC`.** `read_hello` now reads exactly 8 bytes first (0 bytes still returns `PeerClosedBeforeHandshake`), then dispatches. In the client role, `REJECT_MAGIC` reads the reason byte and returns `HandshakeRejected("server refused this node's advertised identity <id>: the advertised IP must be the address the server sees this connection come from (check the bind or advertised address and any NAT or port forward between the nodes)")`. In the server role, `REJECT_MAGIC` is just unexpected bytes. A `HelloFrom` enum (`Server { local_identity }` or `Client`) carries the role and the identity named in the message, replacing the old `&str` role argument.
- **Constant content.** The record is 9 constant bytes and does not include the source IP the server observed. It tells the client only what it can learn for itself, and it reflects no peer-controlled data.
- **Tolerant decoding.** A truncated record (no reason byte) is still reported as the identity rejection, and an unknown reason code becomes `HandshakeRejected("... reason code 0x..")`, so a later reason can be added without the client misreporting it as an I/O error.
- **Typed, so policy follows.** `HandshakeRejected` is what `Transport::connect` treats as fatal (a misconfiguration never fixes itself), so a node with a wrong advertised address now fails at connect time instead of warning after 5 s.
- **Version bump.** `PROTOCOL_VERSION` goes from 1 to 2 (now `pub`). A mixed-version pair fails on the explicit version check in `read_hello` rather than on a confusing byte mismatch. Mixed versions were already unsupported (discovery beacons pin `CARGO_PKG_VERSION`).

Server-side classification, used by the existing rate-limited log in `accept_loop`:

| Class | Type | When | Log |
|---|---|---|---|
| Liveness probe | `PeerClosedBeforeHandshake` | 0 bytes before the client hello | debug |
| Identity not owned | `IdentityNotOwned { advertised, source_ip }` | concrete advertised IP differs from the source IP; the record is sent | warn (rate-limited) |
| Probable secret mismatch | `ClientAbortedAfterServerProof` | EOF, reset, abort or broken pipe while waiting for the client tag, after the server sent its tag | warn (rate-limited) |
| Client secret mismatch seen by server | `HandshakeRejected` | client tag does not verify | warn (rate-limited) |

The client-side message on `VERDICT_REJECT` now reads "server rejected this node's authenticator although the server's own authenticator verified; the handshake was altered in transit", which is the only way that branch is reached.

Changes from the security review (commit `865cbaaa`, originally `471320ca`):

- **An unparseable identity gets no record.** The first version sent `REJECT_IDENTITY_NOT_OWNED` for any `bind_advertised_identity` error, so a malformed identity was reported to the client as an IP mismatch. Now only a downcast to `IdentityNotOwned` writes the record. An identity that does not parse as a `SocketAddr` comes from no mlxcel client, so it is a protocol error and the connection closes without a reply.
- **Log escaping.** The parse-error message formats the identity with `{:?}`. It is unauthenticated peer input that reaches the operator log, and Debug formatting escapes control characters, so a peer cannot forge log lines.
- **Abortive closes.** `is_eof` became `is_peer_abort`, which walks the `anyhow` chain and matches `UnexpectedEof`, `ConnectionReset`, `ConnectionAborted` and `BrokenPipe`. A client that closes with unread data or `SO_LINGER` 0 produces a reset instead of a FIN; both now map to `ClientAbortedAfterServerProof`.

`server_handshake` writes the record itself rather than leaving it to `TcpTransport`, so any caller gets the behavior. `client_handshake`, `server_handshake`, `HANDSHAKE_TIMEOUT` and the four error types are re-exported from `mlxcel::distributed` for #2658.

### Item d: fail-closed, not opt-in

The #2650 docs promised a follow-up that would add the secret to the disaggregated roles "as an opt-in". This PR makes it mandatory instead, matching the pipeline roles. An opt-in would leave the default deployment unauthenticated, which is the state the issue set out to remove, and it would give the two families of distributed roles different security defaults.

- **Resolver.** `resolve_disaggregated_cluster_auth(sources, mode)` sits next to `resolve_pipeline_cluster_auth`. `Hybrid` returns `None` without reading any source. `PrefillOnly`, `DecodeOnly` and `Router` require a secret and otherwise fail with an error naming the role flag and `CLUSTER_SECRET_SOURCES_HELP` (every source). Sources are `ClusterSecretSources::from_env(--cluster-secret-file, [cluster] secret_file)`, so the precedence and the "both env vars set" error are the ones from #2650. The TOML is only parsed for a non-`Hybrid` mode.
- **Placement.** `start_server` runs the resolver before `resolve_distributed_startup`, the peer probe, any listener bind and the weight load. The serving-mode derivation was extracted from `build_server_config` into `resolve_serving_mode` so `start_server` can call it this early. `build_server_config` sets `cluster_auth: None` and reads no secret; `start_server` fills it.
- **Plumbing.** `ServerConfig::cluster_auth` feeds `ModelProvider` (three constructors gain a parameter), then `WorkerSchedulerConfig::cluster_auth`, then `run_disaggregated_serving_role`. The worker builds its transport through `disaggregated_role_transport_config(bind, Arc<ClusterAuth>)`, which takes a non-optional auth, and the coordinator test calls the same helper, so the test binds exactly what production binds. The router sets `auth: config.cluster_auth.clone()` on its own `TcpTransportConfig`; because `TcpTransport` carries `auth` into its `ConnectionPool`, outbound frames (router to prefill, prefill handoff to decode, result frames back) authenticate too. Both `mlxcel-server` and `mlxcel serve` reach `start_server`.
- **Second guard in the worker.** If `run_disaggregated_serving_role` ever sees `cluster_auth: None`, it logs an error and returns `false` (the existing fallback to the single-node scheduler loop for a misconfigured role) instead of binding an unauthenticated role listener. This is unreachable through `start_server` and exists to catch a future caller that skips the resolver.
- **Testability.** The decode loop's intake moved into `ServingCoordinator::next_decode_meta`, which `run_decode_role_networked` now calls. `networked_decode_role_rejects_unauthenticated_frame` writes a correctly framed `DecodeMetaFrame` from a raw socket and asserts both that the socket is closed and that `next_decode_meta` yields nothing within 500 ms, without needing a `BatchScheduler`.
- **Scope kept narrow.** Connection authentication keeps outsiders off the role ports. Per-frame source or correlation checks inside the disaggregated protocol (the analogue of #2650's coordinator checks) are out of scope; frames from authenticated members stay trusted, and `MLXCEL_DECODE_ALLOWLIST` (#389) is unchanged. The `router_front.rs` comment on continuation sequence numbers now says so: they are a liveness aid, not a tamper defense.
- **Identity binding applies here too.** Because the accept side binds the advertised identity to the TCP source IP, the docs now require `--serving-bind` to name the IP the peers see the node connect from, not `0.0.0.0` on a multi-homed host. A mismatch fails with the item b rejection.

The router health monitor's connect-and-close probe already lands in the `PeerClosedBeforeHandshake` debug branch and stays there.

## 3. Change Summary

| Item | Value |
|------|-------|
| Files changed | 23 |
| Lines added | +1,501 |
| Lines deleted | -186 |
| Tests added | 15 (plus 1 updated) |

- **New module**: `src/distributed/pending_handshakes.rs` (`PendingHandshakeLimiter`, `PendingHandshakeSlot`, `PendingSlotRefusal`, defaults) with `pending_handshakes_tests.rs`.
- **Handshake**: `cluster_handshake.rs` gains `REJECT_MAGIC`, `REJECT_IDENTITY_NOT_OWNED`, `IdentityNotOwned`, `ClientAbortedAfterServerProof`, `HelloFrom`, `read_rejection_reason`, `is_peer_abort`, and `PROTOCOL_VERSION = 2`; `distributed/mod.rs` re-exports the handshake API.
- **Transport**: `tcp_transport.rs` adds the two config fields, validates them in `bind`, and splits `accept_loop` into the unauthenticated path (unchanged) and the pending-slot path; Security Notes rewritten.
- **Disaggregated roles**: `disaggregated_role_transport_config` and `next_decode_meta` in `disaggregated/coordinator.rs`; `cluster_auth` on `ServerConfig`, `WorkerSchedulerConfig` and the `ModelProvider` constructors; `resolve_serving_mode`, `resolve_disaggregated_cluster_auth`, `toml_cluster_secret_file` and the router transport auth in `startup.rs`.
- **CLI and docs**: `--node-role` and `--cluster-secret-file` help on both binaries; `docs/distributed.md` (fail-closed scope, failure classes, pending-pool residual risk, disaggregated trust model and launch example); `docs/environment-variables.md`; `docs/CONTINUOUS_BATCHING.md` three-process example with the secret.
- **Test harness**: `tests/common::TestClusterSecretFile` (fresh random secret per test in a temp dir); `disaggregated_router_e2e.rs` spawns every role through a `ClusterRoles` helper that appends `--cluster-secret-file`; `disaggregated_serving_e2e.rs` gives its in-test client transport the same auth.

New tests: `client_reports_identity_rejection_as_typed_rejection`, `server_reports_client_abort_after_proof_as_probable_secret_mismatch`, `identity_rejection_sends_only_the_reject_record`, `unparseable_identity_closes_without_a_reject_record`, `server_reports_client_reset_after_proof_as_probable_secret_mismatch`, `peer_abort_classifies_eof_reset_and_broken_pipe` (handshake); `per_ip_limit_refuses_the_excess_and_releases_on_drop`, `global_limit_refuses_across_sources_and_does_not_leak_per_ip_counts`, `zero_bounds_are_refused_at_construction` (limiter); `pending_handshakes_do_not_consume_connection_slots`, `per_ip_pending_limit_closes_excess_sockets`, `pending_slot_is_released_after_handshake`, `bind_refuses_zero_pending_bounds_with_auth` (transport); `networked_decode_role_rejects_unauthenticated_frame` (coordinator); `disaggregated_roles_fail_closed_without_a_secret` (startup). `handshake_rejects_client_advertising_foreign_ip` now asserts `IdentityNotOwned` on the server and `HandshakeRejected` on the client.

| Hash (rebased) | Original | Message |
|---|---|---|
| `c2a4c783` | `0bd014c4` | fix(distributed): cap pre-auth slots, type rejections, auth DI roles |
| `865cbaaa` | `471320ca` | fix(distributed): tighten handshake reject and pending-pool edges |
| `5d4df5b2` | `6e3da1da` | docs: document handshake edge cases from the final #2655 code |

## 4. Breaking Change and Migration

`--node-role prefill`, `--node-role decode` and `--node-role router` exit at startup without a cluster secret, with `the disaggregated <role> role (--node-role <role>) requires a cluster secret ...` followed by every source. To migrate:

```bash
openssl rand -hex 32 > /etc/mlxcel/cluster.secret
chmod 600 /etc/mlxcel/cluster.secret
# copy the same file to every prefill, decode and router host, then add to each launch:
mlxcel-server ... --node-role decode --serving-bind 10.0.0.2:8304 \
    --cluster-secret-file /etc/mlxcel/cluster.secret
```

`MLXCEL_CLUSTER_SECRET_FILE`, `MLXCEL_CLUSTER_SECRET` or `[cluster] secret_file` in a TOML passed with `--distributed-config` work as well, with the same precedence as the pipeline roles. Set `--serving-bind` to the IP the peers see this node connect from. Upgrade every node together: the handshake wire version is now 2, so a pipeline or disaggregated pair with one node on the previous release fails with the version message.

## 5. Residual Risks

- **Pending pool flood delays new connections.** By design, a flood from 8 or more source IPs can fill the global pending pool and delay any new connection (including a real peer reconnecting) by up to 5 s per slot while it lasts. The connecting side sees the socket closed before the server hello, which is `PeerClosedBeforeHandshake` and only a warning in `Transport::connect`, not a typed failure.
- **IPv6 per-IP key is a full address.** One /64 can rotate source addresses past the per-IP limit. The global pool still bounds the total. Keying on a prefix is not done.
- **Warn budget is shared.** Pending-pool refusals and handshake failures share `handshake_warn_allowed` (5 warn lines per minute, process-wide). During a flood, the refusals can consume the budget and push a real secret-mismatch or identity-rejection line down to debug. Operators investigating a flood should look at debug logs.
- **`ClientAbortedAfterServerProof` is an inference.** A network drop or a client crash at exactly that point in the handshake is reported as "most likely holds a different cluster secret". The message hedges for that reason.
- **Main cap full after authentication.** If all 512 `max_connections` permits are in use, an authenticated peer has already received `VERDICT_ACCEPT` when the transport closes it with "max connections reached", so the client sees a later I/O error rather than a typed rejection. Previously it was closed before the handshake. Both are untyped; this needs 512 concurrent authenticated connections to occur.
- **`active_conns` on task panic.** The counter is not decremented if a connection task panics. This predates the PR.
- **Router has no second guard.** The worker refuses to bind a role transport without auth; the router sets `auth: config.cluster_auth.clone()` and relies on the resolver having failed closed earlier in `start_server`. A future refactor that builds the router transport without going through `start_server` would bind it unauthenticated.
- **Connection, not frame, authentication.** As in #2650: frames are neither encrypted nor MACed, every secret holder is fully trusted, and there are no per-frame source checks in the disaggregated protocol. The trusted-segment requirement for the role ports stays in the docs.
- **Item c is still open.** mDNS discovery has no beacon sender in production (#2658).

## 6. Validation

All results below come from the PR body and review comments, recorded on the pre-rebase heads (`471320ca` and `6e3da1da`) on the GB10 host spark-102 with `--profile test-fast --features cuda`, GPU work under `gpu-lock`. This report did not re-run any of it.

**Failing first.** Against the unfixed code, `client_reports_identity_rejection_as_typed_rejection` failed with "expected HandshakeRejected, got: peer closed the connection without starting the cluster handshake", `server_reports_client_abort_after_proof_as_probable_secret_mismatch` failed with "cluster handshake timed out after 5s", and `pending_handshakes_do_not_consume_connection_slots` failed with "cluster handshake with 127.0.0.1:... failed". The other new tests reference new fields or functions and do not compile against the old code.

**Lib suite and lint.** `distributed::` with `--test-threads=1`: 1390 passed, 0 failed, 4 ignored; `distributed::cluster_handshake::` repeated 8 times, 15 of 15 each time. `server::startup::`: 93 passed. `cargo clippy -p mlxcel --lib --tests -- -D warnings`, `cargo fmt --check` and `make verify-env-var-docs` were clean. The `disaggregated_router_e2e`, `disaggregated_serving_e2e`, `pipeline_remote_real_models` and `pipeline_server_remote_real_models` targets compile.

**Manual loopback (three processes, qwen3-0.6b-4bit, `target/test-fast/mlxcel-server`).**

- Shared `--cluster-secret-file` on router, prefill and decode: `POST /v1/chat/completions` returned HTTP 200 (16 prompt and 16 completion tokens).
- Different secret on the decode node: HTTP 500 `decode handoff to 127.0.0.1:19465 failed`. The prefill log showed the secret mismatch from the client side; the decode log showed `client closed the connection after receiving this server's authenticator; it most likely holds a different cluster secret`, which is the new item b classification working end to end.
- Decode and router roles with no secret: exit 1 with the fail-closed message.

**Real-model pipeline regression.** With `MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1`, `pipeline_remote_runtime_qwen3_real_model_parity_and_cleanup` and `pipeline_server_remote_coordinator_qwen3_matches_dense_baseline` passed on `6e3da1da`, which shows the version-2 handshake still carries the pipeline roles.

**Rebase.** The orchestrator rebased the branch onto main `f3b2efc0`. `git range-diff` shows all three commits patch-identical (`0bd014c4` to `c2a4c783`, `471320ca` to `865cbaaa`, `6e3da1da` to `5d4df5b2`). One upstream commit in that range, #2668 (CORS origin env vars), touches `src/server/startup.rs`, `src/main.rs`, `src/bin/mlx_server.rs` and `docs/environment-variables.md`; none touches `src/distributed/`; a pre-merge check on the rebased head was running when this report was written.

**Not verified.** The `#[ignore]` disaggregated e2e harnesses themselves were not run (only compiled); the manual loopback run is the equivalent three-process flow. No run used two physical hosts, a NAT or port forward between nodes, or IPv6 sources. No throughput or latency was measured; the added per-connection work on the accept path is one short mutex section and one semaphore `try_acquire`, and nothing changes per frame.

## 7. Learning Points

- **Count a connection against the cap only once it has earned it.** A shared cap that unauthenticated sockets can occupy turns the handshake timeout into a lockout window. A small, separate pre-auth pool, released the moment the handshake returns, keeps the expensive resource for peers that have proven themselves.
- **RAII makes partial acquisition safe.** Taking the per-IP count first and wrapping it in a guard means a failed global acquire rolls itself back on return, with no explicit undo path to forget.
- **A rejection must not leak what the protocol withholds.** The identity rejection replaces the server hello rather than following it, so a client that lies about its address still never sees a nonce or an authenticator, and the record carries only constant bytes.
- **Make the first read decide the message type.** Reading a fixed-length prefix before the rest lets one stream position carry either a hello or a rejection without ambiguity, and a same-length magic keeps the parser simple.
- **Classify errors by where in the protocol they happen.** An EOF while waiting for the client tag means something different from an EOF before the client hello. Typing them (`ClientAbortedAfterServerProof` versus `PeerClosedBeforeHandshake`) is what turned an indistinguishable log line into an actionable "probable secret mismatch".
- **Escape unauthenticated input before logging it.** `{:?}` on a peer-supplied string is a one-character fix for log-line forgery.
- **Prefer fail-closed over opt-in for a security default.** An opt-in secret leaves every existing deployment as it was; the cost of making it mandatory is one clearly worded startup error and a documented migration.
