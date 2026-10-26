# PR #2650: Authenticate cluster transports and correlate stage replies

**Date**: 2026-10-27
**Status**: Completed (report written on the PR branch before merge)
**Languages**: Rust, Markdown, Shell, YAML
**Risk Level**: High (security fix; breaking for operators of distributed pipeline roles)

## Executive Summary

`src/distributed` had no authentication layer. Any host that could reach a pipeline port could send a `pipeline_stage_response` frame that the coordinator returned as model output, drive or shut down a remote stage (and end its service thread with one malformed frame), and spoof a UDP discovery beacon into the cluster TOML that `--pp-auto` generates. This PR adds one pre-shared per-cluster secret and uses it at three layers: a mutual HMAC-SHA256 handshake on every TCP connection between pipeline roles, per-exchange correlation and source checks on every frame the coordinator and stages act on, and a MAC with a replay window on discovery beacons. Every pipeline role now refuses to start without the secret. Single-node serving and the disaggregated prefill/decode/router roles do not read it and are unchanged. Operators running `--pp-auto`, `--pp-peer`, or a `--distributed-config` with `pipeline_parallel_size >= 2` must supply a secret after upgrading (issue #2637).

## 1. Problem Statement

Issue #2637 reported three findings against the pipeline-parallel transports, all reachable from the network with no credential:

1. **Response injection.** `RemotePipelineRuntime::exchange` sent a command and returned the first `pipeline_stage_response` frame that `transport.recv()` produced. It ignored the sender (`let (_from, msg) = ...`) and had no request identifier to match. Any peer able to open a TCP connection to the coordinator's transport port could race the real stage with a `RunResult` carrying arbitrary logits, and those logits became the forward output that sampling ran on.
2. **Stage takeover and crash.** `RemoteStageService` executed every `pipeline_stage_command` it received regardless of origin, including `Shutdown`, `BeginDrain`, `CancelRequest` and `RunEntry`, and sent the reply to the `reply_addr` named inside the payload. `serde_json::from_slice(&payload)?` propagated a parse error out of the run loop, so a single malformed frame ended the service thread, and `serve_remote_pipeline_stage` kept parking on `ctrl_c` with no live service behind it.
3. **Discovery spoofing.** `discover_peers` in `--cluster-discovery=mdns` mode accepted any JSON datagram whose `cluster_name` matched (the default is the well-known `mlxcel-cluster`) and put its `peer_addr` into the generated TOML as a pipeline stage. The `version` field was documented as a compatibility gate but was not checked.

The transport's own Security Notes said it performed no authentication and that callers should bind to trusted interfaces or layer TLS on top. No caller did the latter.

## 2. Technical Decisions

### One secret, three uses

The PR introduces `src/distributed/cluster_auth.rs` with three types:

- `ClusterSecret`: at least 16 bytes after trimming exactly one trailing `\n` or `\r\n` (`MIN_CLUSTER_SECRET_BYTES`). `Debug` prints `ClusterSecret(<redacted>)`, there is no `Display`, and `Drop` scrubs the bytes with volatile writes (best effort, as the comment says). On unix, `from_file` warns when the file is group- or world-readable but still accepts it.
- `ClusterSecretSources`: collects every place the secret may come from and resolves them in one function, so the precedence is unit-testable without touching the process environment. Its `Debug` redacts the inline value (fixed during security review; it previously derived `Debug`).
- `ClusterAuth`: an HMAC-SHA256 key (`ring::hmac`) plus a `SystemRandom` CSPRNG. `sign(label, message)` computes `HMAC(K, label || 0x00 || message)`, and `verify` is constant-time through `ring::hmac::verify`. Every use passes its own label (`mlxcel-cluster-auth-v1/server`, `mlxcel-cluster-auth-v1/client`, `mlxcel-discovery-beacon-v1`), so a tag minted for one purpose never verifies for another.

`ring` was already compiled into every build through rustls and reqwest. `Cargo.toml` adds it as a direct dependency at `0.17` with the default feature set so it unifies with that existing instance and adds no new crate to the build.

A shared secret was chosen over per-node keys or TLS because it is the smallest change that closes all three findings without a PKI, and every role of one cluster is already operated as one trust domain. The cost is that there is no per-role credential (see section 5).

### Secret sources and precedence

`ClusterSecretSources::resolve` returns the first source that is set, in this order:

1. `--cluster-secret-file <PATH>` (on both `mlxcel-server` and `mlxcel serve`).
2. `MLXCEL_CLUSTER_SECRET_FILE=<PATH>`.
3. `MLXCEL_CLUSTER_SECRET=<VALUE>`, the secret itself.
4. `secret_file = "<PATH>"` under `[cluster]` in the cluster TOML. `ClusterConfig::from_file` resolves a relative path against the TOML's own directory, so a config and its secret can be copied together.

Setting both environment variables is an error rather than a silent pick, because the two can disagree without anyone noticing. Empty environment values count as unset. The TOML never carries the secret value: the `--pp-auto` coordinator writes the generated TOML to disk and operators copy it between hosts. When the secret came from a file, `--pp-auto` records that file in the generated TOML as an absolute path (`pp_auto_secret_file_reference`, using `std::path::absolute`). Recording the path as typed would have been wrong: a relative path typed on the command line is relative to the working directory, while readers of the TOML resolve it against the TOML's directory (by default `./.mlxcel/`). The implementation review caught that, and `pp_auto_records_secret_file_as_absolute_path_peers_resolve` covers it.

### Fail-closed scope and what stays unchanged

Three entry points refuse to start without a secret, and each error names every source (`CLUSTER_SECRET_SOURCES_HELP`, which also suggests `openssl rand -hex 32 > cluster.secret`):

- `--pp-auto`: `resolve_pp_auto_cluster_auth` runs before any discovery socket is bound or any TOML is written, so a coordinator without a secret never touches the network (`pp_auto_bring_up_fails_closed_before_discovery_or_toml_write`).
- `--distributed-config` with `pipeline_parallel_size >= 2`: `resolve_pipeline_cluster_auth` in `resolve_distributed_startup`. `--pp-peer` reaches this path too, because it requires `--distributed-config` pointing at the coordinator-emitted TOML.
- `discover_peers` in `Mdns` mode bails when it is given no `ClusterAuth`.

The type system backs this up. `RemotePipelineRuntimeConfig.auth` and `RemoteStageServiceConfig.auth` are `Arc<ClusterAuth>`, not `Option`, so no pipeline role can be constructed unauthenticated. `resolve_remote_pipeline_topology` takes `Option<&Arc<ClusterAuth>>` and treats `None` with a pipeline depth above 1 as an internal error.

Unchanged: default single-node serving, tensor parallelism inside one process, and a cluster config with `pipeline_parallel_size <= 1` need no secret (`resolve_pipeline_cluster_auth` returns `None`). `TcpTransportConfig::auth` defaults to `None`, and `ConnectionPool::new` builds a pool without auth, so the disaggregated prefill/decode transports and the router keep their previous unauthenticated behavior. Static discovery never touches the secret.

### The connect-time handshake

`src/distributed/cluster_handshake.rs` runs once per TCP connection, before any transport frame is read or written. All integers are big-endian:

```text
client -> server : MAGIC "MLXCAUTH"(8) | version u16 = 1 | nonce_c [32] | id_len u16 | identity
server -> client : MAGIC(8) | version u16 | nonce_s [32] | id_len u16 | identity
                   | tag_s = HMAC(K, "mlxcel-cluster-auth-v1/server", transcript)
client -> server : tag_c = HMAC(K, "mlxcel-cluster-auth-v1/client", transcript || tag_s)
server -> client : verdict u8 (0x01 accept; anything else or EOF is a rejection)
```

`transcript` is the exact bytes of the client hello followed by the server hello. Both nonces come from the CSPRNG, so each tag covers fresh randomness from both sides, and a recorded tag cannot be replayed on a new connection. The identities (each side's listener `host:port`, at most 255 bytes) are inside the transcript, so a tag is bound to the identities both sides claimed. The client's tag also covers `tag_s`, and the two directions use different labels, so the server's tag can never be reflected back as a client tag. A peer with the wrong secret fails whichever side opened the connection: the client checks `tag_s` before it sends anything secret-derived, and the server checks `tag_c` before it accepts.

The server sends its authenticator before the client has proven anything. That reveals a MAC over a transcript containing the server's own fresh nonce, which is useless outside that connection because of the label separation and the client's fresh nonce.

**Identity binding, accept side only.** The accept side needs an identity to attribute frames to: the ephemeral `ip:port` of an accepted socket names no configured peer. `bind_advertised_identity` takes the client's advertised listener address and binds it to the TCP source IP. An unspecified IP (`0.0.0.0`, which a listener bound on every interface advertises) is replaced by the source IP, and a concrete IP that differs from the source IP is refused. The server does this check before it replies, and on a mismatch it closes without answering, so a client that lies about its address never receives a server authenticator. The bound identity becomes the `from` of every frame `recv()` returns for that connection, and the runtime and stage service authorize against it.

The connect side deliberately does not bind the server's advertised identity to the socket IP. It already knows which address it dialed, and the server's tag proves possession of the secret. A stage reached through NAT or a port forward legitimately advertises an address other than the one the client sees. The first version of the PR did bind on the client side; the implementation review found the check had no consumer and broke forwarded stages, and it was removed (`client_accepts_server_advertising_a_forwarded_address`).

**Timeout.** Both sides wrap the whole handshake in `HANDSHAKE_TIMEOUT` (5 s), so a peer that connects and stalls holds a connection slot for at most that long. Before this PR, an idle unauthenticated socket held a slot indefinitely.

**Verdict byte.** Without it, the client would learn of a rejection only when its first frame hit a closed socket, as a generic I/O error some time later. The explicit verdict lets `client_handshake` return a typed `HandshakeRejected` at connect time. `TcpTransport::connect` (the pre-warm pass over configured peers) treats `HandshakeRejected` as fatal and any other error as a warning, because a peer that is not up yet may start later, but a secret mismatch never fixes itself. `RemotePipelineRuntime::new` calls `connect`, so the coordinator fails at startup on a mismatch instead of on the first request.

**Where it runs.** The accept side runs `server_handshake` inside the per-connection task, before `handle_connection` reads a frame, so a frame from an unauthenticated connection never reaches `recv()`, the RPC handler, or `rpc_pending`. The connect side runs `client_handshake` inside `ConnectionPool::connect_with_retry`, the only outbound connect path, so pooled and reconnected streams authenticate exactly once, when they are opened. Thunderbolt and RDMA wrap the same TCP core: `ThunderboltTransportConfig::from_bind_address_with_auth` and `RdmaTransportConfig.auth` forward the auth to it, and `bind_transport` takes `auth` for every backend.

**Logging.** A peer that connects and closes without sending a byte gets `PeerClosedBeforeHandshake`, logged at debug, because liveness probes (`probe_peers`, the router health monitor) do exactly that. The loopback E2E run found those probes being logged as rejected handshakes. Other failures log the peer IP (never secret material) at warn for the first 5 per minute, process-wide, and at debug afterwards (`handshake_warn_allowed`).

### Response correlation in the coordinator

`RemoteStageRequest` gained a `nonce: u64` drawn fresh from the CSPRNG per exchange (`ClusterAuth::random_u64`), and every `RemoteStageResponse` variant (`Ack`, `RunResult`, `State`, `Error`) echoes it, so forged acks and errors are rejected too, not only forged results. The entry stage stores the nonce in `PendingEntryReply` so the `RunResult` it sends after the reverse path carries the right value.

`exchange` now accepts a frame only when `ExchangeMatch::reject_reason` returns `None`:

- the authenticated `from` matches the stage the command was sent to (`PeerMatcher`),
- the echoed nonce equals this exchange's nonce,
- for a `RunEntry`, the `RunResult` names the same `request_id` and `micro_batch_id`.

`PeerMatcher` (`src/distributed/peer_identity.rs`) resolves a configured `host:port` once through the system resolver and compares canonical socket addresses, so a stage configured by hostname still matches its numeric identity. A configured `0.0.0.0:<port>`, which `--pp-auto` emits when the HTTP host is `0.0.0.0`, names no single host, so it matches any authenticated identity with that port. The handshake has already bound that identity to the real source IP.

A rejected frame is logged (the first three per exchange at warn, the rest at debug) and the loop keeps waiting. The wait uses an absolute deadline computed once (`deadline = now + timeout`), and each `recv` gets only the remaining time. Before, every non-matching frame restarted a full `timeout`, so a steady trickle of junk frames could hold an exchange open forever (`exchange_timeout_is_absolute_under_frame_flood`). The per-exchange nonce also means a late reply to an earlier timed-out exchange cannot satisfy the next one.

`forward_sequence` validates the logits before returning them: rank 3, batch 1, sequence length equal to the input's or 1 (last-position-only families), and the model's vocabulary size. `read_vocab_size` reads `vocab_size` from `config.json`, falling back to `text_config.vocab_size` for vision-language checkpoints, and applies `sanitize_config_json` like the other config readers. `startup.rs` and `PipelineServerModel::load_remote` both fill `RemotePipelineRuntimeConfig.vocab_size`. A checkpoint with no `vocab_size` still gets the rank, batch and sequence checks. A shape mismatch cleans up the request's state before returning the error.

### Remote stage service authorization

`RemoteStageServiceConfig` gained `coordinator_peer`, resolved from the cluster TOML as the first node whose role is not a pipeline stage (`pipeline_coordinator_address`). The service now enforces:

- `pipeline_stage_command` frames only from the coordinator. Commands from any other peer are dropped with a warning, authenticated or not, and get no reply.
- Forward activations only from the configured upstream stage, reverse activations only from the configured downstream stage. A stage with no such neighbour drops the activation.
- Replies go to the authenticated source of the command (`from`), never to `reply_addr`. The field stays in the envelope for wire compatibility, and a mismatch is logged at debug. Since only the coordinator's commands are accepted, the source is the coordinator by construction.

The run loop now distinguishes per-frame conditions from infrastructure failure. Malformed JSON, a bad wire tensor in `RunEntry` (answered with an `Error` response), and an activation with no pending reply are logged and the loop keeps serving. Only a closed transport channel, a coordinator `Shutdown`, or the local shutdown signal ends it. `RunEntry` handling moved into `handle_run_entry` for this.

The service thread now signals when it exits (`exited_rx`), and `serve_remote_pipeline_stage` selects on `ctrl_c` and `handle.exited()`. A stage whose service dies, for example because a neighbour rejected the handshake, now returns the error and the process exits instead of parking as a zombie that a supervisor would never restart.

The executor became injectable (`spawn_with_executor` with an `ExecutorFactory` built on the service thread, so non-`Send` MLX executors never cross threads), which lets the control-plane tests run without model weights.

The dormant RPC control plane, `install_stage_control_service` with `StageLifecycleRequest` and `StageLifecycleResponse`, was deleted along with its re-exports. Its only callers were tests, and leaving an unauthorized second control path in the library would have invited someone to wire it up later.

### Discovery beacon MAC

`src/distributed/discovery_beacon.rs` adds `timestamp_ms`, `nonce` and a hex `mac` to `DiscoveryBeacon`. The MAC covers a canonical, length-prefixed encoding of every other field (`canonical_bytes`), not a JSON re-serialization, so key order and whitespace cannot change it. `broadcast_beacon_loop` re-signs the beacon on every tick with a fresh timestamp and nonce. `BeaconVerifier::accept` checks, in order:

1. the datagram parses and names this cluster (anything else is treated as unrelated traffic and dropped silently),
2. `version` equals this build's `CARGO_PKG_VERSION` exactly,
3. the MAC verifies in constant time,
4. the timestamp is within 60 s (`BEACON_MAX_SKEW`) of the receiver's clock, in either direction,
5. the nonce has not been accepted within that window,
6. the advertised `peer_addr` IP equals the datagram source IP, or is unspecified and is replaced by it.

The version check is strict because the stage wire protocol is unversioned JSON and a mixed-version cluster has no defined behavior. The doc comment, which used to call it best-effort, now says it is enforced. Nonces enter the replay set only after the MAC verifies, so only secret holders can grow it, and entries are pruned once they leave the window. Rejections that name this cluster are logged at warn up to 20 per discovery run, then at debug.

The security review found that the first `unhex` sliced the attacker-controlled `mac` string at byte offsets, so one datagram with a multi-byte character at an odd offset panicked `--pp-auto` discovery before the MAC check. It now works on bytes and accepts exactly 64 ASCII hex characters (`verifier_rejects_non_ascii_mac_without_panicking`).

`broadcast_beacon_loop` has no in-tree production caller, before or after this PR: `--pp-peer` joins through the coordinator-emitted TOML. The signing side is exercised by tests (`broadcast_beacon_loop_resigns_each_tick_and_resolves_against_coordinator`).

### Listener coverage and scope

Covered by the handshake: the pipeline coordinator's control transport, the remote stage service transport, and the Thunderbolt and RDMA wrappers (same TCP core).

Out of scope, documented as such: the disaggregated prefill/decode role transports (`disaggregated/coordinator.rs`) and the router front (`router_front.rs`). They are a separate opt-in feature with their own role matrix and no secret plumbing today. `docs/distributed.md` now states that they keep the trusted-segment model and that wiring the same secret in as an opt-in is a follow-up. `discovery.rs` and the router health probe only connect. Heartbeat and KV-cache transfer use whatever transport they are handed. `bench_activation.rs` is a benchmark harness.

## 3. Change Summary

- New modules: `cluster_auth.rs` (secret, sources, HMAC and CSPRNG), `cluster_handshake.rs` (wire protocol, identity binding, typed errors), `discovery_beacon.rs` (canonical encoding, signing, `BeaconVerifier`), `peer_identity.rs` (`PeerMatcher`), each with a test file.
- Transport core: `TcpTransportConfig.auth`, the accept-side handshake and rate-limited logging in `tcp_transport.rs`; `ConnectionPool::with_auth` and the connect-side handshake in `connection_pool.rs`; auth forwarding in `thunderbolt_transport.rs`, `rdma_transport.rs` and `transport_factory.rs`.
- Pipeline: `runtime.rs` (`StagePeer`, `ExchangeMatch`, absolute deadline, `validate_logits_shape`, required `auth`, `vocab_size`); `remote_service.rs` (coordinator, upstream and downstream matchers, per-frame errors, reply to source, nonce echo, `spawn_with_executor`, `exited`); `server_runtime.rs` (vocab size fill); `activation_transfer.rs` (dormant RPC plane removed, -160 lines).
- Cluster config and bring-up: `ClusterMeta.secret_file` with TOML-relative resolution in `config.rs`; `ClusterInitRequest.secret_file`, the authenticated `discover_peers`, and re-signing in `broadcast_beacon_loop` in `cluster_init.rs`.
- Server startup and CLI: `--cluster-secret-file` on `mlxcel-server` and `mlxcel serve`; `resolve_pipeline_cluster_auth`, `pipeline_coordinator_address`, `pp_auto_secret_file_reference`, `resolve_pp_auto_cluster_auth`, and the `exited` select in `startup.rs`; `read_vocab_size` in `loading/mod.rs`.
- Docs and examples: a "Cluster authentication" section with residual risks in `docs/distributed.md`; both env vars in `docs/environment-variables.md`; `secret_file = "cluster.secret"` in both example TOMLs with a README note and a `.gitignore` entry; `scripts/benchmark_pipeline_remote_rollout.sh` generates a random secret with `umask 077` when missing.
- CI: the pipeline-parallel workflow's path filter and `scripts/ci/run-pp-two-host-logical.sh` include the new modules. Integration tests share a random secret through `tests/common::test_cluster_auth`, so no secret is committed.

The diff is 57 files, 4,532 lines added and 491 removed, most of the additions in tests.

## 4. Breaking Change and Migration

Every distributed pipeline role now exits at startup without a secret. To migrate, generate one secret per cluster, copy it to every host with owner-only permissions, and supply it through any source above:

```bash
openssl rand -hex 32 > /etc/mlxcel/cluster.secret
chmod 600 /etc/mlxcel/cluster.secret
mlxcel-server ... --distributed-config cluster.toml --cluster-secret-file /etc/mlxcel/cluster.secret
```

Mixed-version clusters stop working in two ways: an old build does not speak the handshake (the new side sees no `MLXCAUTH` magic), and `--pp-auto` discovery rejects beacons from a different `CARGO_PKG_VERSION`. All hosts must be upgraded together.

## 5. Residual Risks

- **Connection authentication, not per-frame authentication.** The handshake authenticates a connection once. Frames are neither encrypted nor MACed. An on-path attacker who can hijack an established TCP stream, or who can receive a peer's connection and relay the handshake to the real server, is out of scope. TLS or a per-frame MAC is the follow-up. Keep cluster ports on a trusted segment.
- **Pre-authentication slots.** Unauthenticated sockets share the 512-connection cap while their handshake is pending, bounded by the 5 s timeout. A flood can delay legitimate connections but cannot inject frames. Before this PR such sockets held slots with no bound at all. A per-IP pre-auth limit is the follow-up.
- **No retry for transient handshake failures.** `connect_with_retry` returns on any handshake error without using the remaining backoff attempts, so a handshake that times out or hits an I/O error is not retried at that point. The intent is to avoid retrying a secret mismatch. `Transport::connect` treats non-`HandshakeRejected` errors as warnings, and a later `send` opens a new connection, so the failure is not permanent, but it is not retried with backoff either.
- **Environment inheritance.** `MLXCEL_CLUSTER_SECRET` stays in the process environment and is inherited by child processes. The file sources are preferred, and the docs say so.
- **Coordinator stays up after an auth failure.** When the coordinator's model load fails on a handshake rejection, the HTTP server keeps running and answers through the existing no-model path (HTTP 501 in the loopback run). It does not serve completions, but it is not a process exit either, so a supervisor that only watches the process will not notice.
- **One secret means no per-role credential.** Every secret holder can complete the handshake. The role checks (commands only from the coordinator, activations only from neighbours) are bound to the source IP; the listener port inside the identity is self-asserted. They stop non-holders and misrouted peers, but not a compromised cluster member. A configured `0.0.0.0:<port>` coordinator address matches any authenticated identity on that port.
- **Coordinator selection.** `pipeline_coordinator_address` picks the first node that is not a pipeline stage. With several non-pipeline nodes in one TOML, a stage drops the real coordinator's commands unless it is listed first. The implementation review raised this as LOW, and it is not addressed in this PR.
- **Identity-binding failures are not typed on the client.** When the server refuses an identity it does not answer at all, so the client sees a closed connection rather than `HandshakeRejected`, and `Transport::connect` logs a warning instead of failing. This affects a multi-homed host whose outbound source IP differs from the concrete IP its listener is bound to.
- **Disaggregated serving is not covered** (see section 2).

## 6. Validation

All results below are from the PR body and review comments, recorded at the pre-rebase head `bbb11737`.

**Lib suite.** `cargo test --profile test-fast --features cuda -p mlxcel --lib -- --test-threads=1 distributed::` ran 1375 tests with 0 failures. `cargo clippy --profile test-fast --features cuda -p mlxcel --lib --tests -- -D warnings`, `cargo fmt --check`, and `make verify-env-var-docs` were clean.

**Revert proof.** A script switched one fix off, rebuilt, ran the named test, then restored the fix, rebuilt and reran. Each reverted run failed (exit 101), and the restored rerun of all five passed (exit 0):

| Finding | Fix reverted | Test that failed |
|---|---|---|
| 1, response injection | `ExchangeMatch::reject_reason` returns `None` | `runtime::correlation_tests::exchange_drops_run_result_from_non_stage_source` (forged logits from the attacker node became the forward output) |
| 2, unauthenticated control | accept-side handshake skipped and coordinator source check skipped | `remote_service::tests::unauthenticated_control_frame_is_rejected_before_dispatch` (a raw-socket `Shutdown` stopped the stage) |
| 2, malformed frame | per-frame `Err` arm returns `Err` again | `remote_service::tests::malformed_json_frame_does_not_end_service` |
| 3, spoofed beacon | MAC verification skipped | `cluster_init::tests::discover_peers_rejects_spoofed_beacon_without_valid_mac` and `discovery_beacon::tests::verifier_rejects_spoofed_beacon_without_mac` |

The review follow-ups were checked the same way: `verifier_rejects_non_ascii_mac_without_panicking` panics with the old decoder, and each test added in `4b6bd035` (rebased as `c077ecc5`) failed against the old production code. Sibling tests cover wrong nonce, wrong `request_id`, wrong vocab dimension, the absolute deadline under a frame flood, forged `State` frames, rogue authenticated peers sending `CancelRequest`, `BeginDrain`, `RunEntry` and `Shutdown` (no effect, no reply, executor never called), replies to the source rather than `reply_addr`, activations from non-neighbours, bad entry tensors, wrong secrets in both directions, plain-versus-authenticated mismatches, beacon tampering, stale and future timestamps, replayed nonces, source-IP mismatch, version mismatch, and secret precedence and redaction.

**Real models.** Under `gpu-lock`, with qwen3-0.6b-4bit: `pipeline_remote_runtime_qwen3_real_model_parity_and_cleanup` (two authenticated remote stages) and `pipeline_server_remote_coordinator_qwen3_matches_dense_baseline` passed. The second shows that the vocab-size check accepts real model output.

**Loopback E2E.** Coordinator on 127.0.0.1:19370, stages on 19371 and 19372, HTTP on 19380 to 19382, using the `target/test-fast` binaries:

- Shared `secret_file`: `/v1/chat/completions` returned HTTP 200 with a completion, and no handshake warnings were logged for liveness probes.
- Stage-1 with a different secret: stage-1 exited at startup with `cluster authentication with peer 127.0.0.1:19371 failed: ... (secret mismatch)` (it detected its correct-secret upstream's rejection), stage-0 logged `Rejecting connection from 127.0.0.1: cluster handshake failed`, and the coordinator could not reach the exited stage (HTTP 503). In the first run, before the two robustness changes (the `exited` select and debug-level logging of liveness probes), the stages logged the mismatch in both directions and the coordinator logged `Failed to load model: cluster authentication with peer 127.0.0.1:19372 failed` and answered HTTP 501 from the no-model path.
- No secret from any source: the coordinator exited 1 with the message naming `--cluster-secret-file`, both env vars and `secret_file`.

These real-model and E2E runs were repeated after the review changes with the rebuilt binary, with the same outcomes.

**Rebase.** The branch was rebased onto main `57b54b0f` before this report. `git range-diff` shows all four commits are patch-identical (`b5b43ffc` to `a0e175b9`, `9e78bf71` to `3dacaf7b`, `4b6bd035` to `c077ecc5`, `bbb11737` to `6a52a698`), and none of the upstream commits pulled in touch `src/distributed/`, `src/server/startup.rs`, `src/loading/`, or `Cargo.toml`. The runs above were not repeated on the rebased head when this report was written.

**Not verified.** No run used two physical hosts, Thunderbolt or RDMA hardware with auth enabled; those wrappers are covered only by the auth being forwarded to the shared TCP core. mDNS discovery was tested over local UDP sockets, not across a real broadcast domain. No throughput or latency was measured. By construction, the per-request cost is one 8-byte CSPRNG read per exchange plus the matcher comparisons, and the handshake runs only when the pool opens a connection.

## 7. Learning Points

- Authenticating a connection is not enough when the application still trusts its payload. The handshake stops outsiders, but the coordinator still needed source, nonce and request checks, and the stage still needed to stop replying to an address named in the frame. Each layer closes a gap the other leaves.
- Keep one parse failure from becoming a liveness failure. `?` on untrusted input inside a service loop turns any malformed frame into a remote kill switch. Separate per-frame errors (log and continue) from infrastructure errors (end the loop), and make the supervisor notice when the loop does end.
- A timeout that restarts on every unrelated message is not a timeout. Compute the deadline once and give each wait the remaining time.
- Bind identity where it is consumed. The accept side attributes frames to a peer, so it binds the advertised identity to the source IP. The connect side already knows what it dialed, and binding there only breaks NAT.
- Typed errors carry policy. `HandshakeRejected` versus a generic I/O error is what lets the transport fail fast on a secret mismatch while still tolerating a peer that has not started yet.
- Parse attacker-controlled text as bytes. Slicing a `&str` at byte offsets panics on multi-byte characters, and here that panic came before the MAC check.
- A relative path in a generated config must be relative to what the reader resolves against, not to the writer's working directory.
