# PR #2723: Authenticate handshake verdicts and bind reply targets

**Date**: 2026-10-29 (written before merge; the PR is open as a draft)
**Status**: Implementation complete at 71a8c6e0. The implementation review found no CRITICAL or HIGH issue. The security and performance review found one HIGH issue (a second secret holder could pre-seed the KV state of the coordinator's next sequence id), fixed in 71a8c6e0. The local gate passed on GB10 (spark-102) at 71a8c6e0. At the time of writing, a pre-merge test run was in progress and hosted CI was partly pending; the hosted `Pipeline Parallel CI / 2-host logical` job failed with a process-exit abort that also appears on other branches (section 4.3)
**Languages**: Rust, Markdown
**Risk Level**: Medium (wire break: `PROTOCOL_VERSION` 4 to 5, so every node of a cluster must be upgraded together; stages now refuse sequence commands from identities they accepted before; `PipelineModelRuntime::prepare_sequence_state` changes its signature)

## Executive Summary

PR #2723 (closes #2714) closes three LOW residuals that #2705 and #2708 left in the authenticated cluster transport, plus two items the orchestrator added after #2720 and #2721 landed on main. The handshake's post-authentication verdict byte now carries an HMAC (`tag_v`) over the whole transcript and the verdict, so an on-path party can no longer report a healthy server as full or a refused connection as accepted; this bumps the handshake to `PROTOCOL_VERSION` 5. The remote pipeline stage replaces the single `coordinator_identity` (overwritten by every command) with `sequence_owners`, a per-request-id map to the authenticated identity that sent the sequence's `PrepareSequence`; only that owner may run, cancel, release or re-prepare the sequence, and the final stage's `RunResult` and every activation `Error` go only to the owner. The router's result-sender port check stays unbound, with the reason now documented.

Because ownership lets a secret holder pre-claim a sequential `pp-seq-N` id, the PR also makes a refused `PrepareSequence` fail one request instead of aborting the coordinator (item a): the error travels from `RemotePipelineRuntime::prepare_sequence_state` through a new `LanguageModel::try_prepare_sequence_state` to `Engine::open` and the scheduler. Item b routes every failure exit of `handle_activation` through `report_activation_failure`, so a non-entry stage that refuses or fails an activation tells the coordinator at once instead of leaving it to wait out `stage_timeout`. A review addition makes the first `PrepareSequence` of an id discard caches that unowned commands left under it.

None of the five items lets a party without the cluster secret read or inject frames. Item 1 needs an on-path attacker; items 2, 3, a and the review fix need a second holder of the secret.

## 1. Problem Statement

#2714 collected residuals recorded as open in the #2705 report (section 3.7 and "Known limitations") and the #2708 report, re-checked on main at 752db4f9.

### 1.1 The handshake verdict was a bare byte (LOW, on-path)

After the client's authenticator verified, the server wrote one unauthenticated byte: `VERDICT_ACCEPT` 0x01, `VERDICT_AT_CAPACITY` 0x02 or `VERDICT_REJECT` 0x00. The client branched on it directly. An on-path party could always drop the connection, but by editing this byte it chose how the failure was reported: flipping 0x01 to 0x02 made a healthy server look full (the client gets `ServerAtCapacity` and backs off), and flipping 0x02 or 0x00 to 0x01 made the client treat a refused connection as up, after which it failed on its first send with a generic I/O error. Frame confidentiality and integrity were not affected, because no keys are derived from a connection the server refused.

### 1.2 The `RunResult` target followed the latest command (LOW, secret holder)

`RemoteStageService::coordinator_identity` was set to the authenticated source of every accepted command, and the final stage sent `RunResult` there. With the coordinator configured under an unspecified IP (a hand-written config or a copy of the `--pp-auto` TOML, which writes `0.0.0.0:<port>`), `PeerMatcher::matches` accepts any authenticated identity on that port. A secret holder listening on that port on another host could send one `ProbeState` and receive the logits of every in-flight sequence from then on. Zero-config peers get a concrete address from `localize_cluster` and were not affected.

### 1.3 The router result-sender port is not proven (LOW, secret holder)

`accepts_result_from` in `router_front.rs` checks the canonical IP and port of the sender, but the handshake proves only the IP (`bind_advertised_identity`). #2714 evaluated binding the port and decided against it (section 3.4); the deliverable is documentation.

### 1.4 A refused prepare aborted the coordinator (item a, added during implementation)

Once stages refuse a `PrepareSequence` for an id another identity owns, `RemotePipelineRuntime::prepare_sequence_state` would hit its `.expect("failed to prepare remote pipeline sequence state")` and abort the model worker; the draining check was an `assert!` with the same effect. #2714 first accepted that as an availability-only trade-off, but a secret holder can make it happen on demand because request ids are sequential (`pp-seq-N`). A `catch_unwind` probe on main panicked at `runtime.rs:605`. `request_id_for` also panicked on a missing mapping.

### 1.5 Non-entry activation refusals ended in a coordinator timeout (item b, added during implementation)

#2720 routed execution failures and failed downstream sends of `handle_activation` through `report_activation_failure`, but every other exit still used `?` and fell through to the service loop's `dropping frame` warning: the tensor-limit refusals that #2721 added, a missing downstream peer, `sequence_length`, request id parsing and serialization. The coordinator then waited out its full `stage_timeout` for an exchange the stage had already given up on. The #2721 report lists this as its first remaining gap.

## 2. Change Summary

| Item | Area | Change |
|---|---|---|
| 1 | `cluster_handshake.rs` | `VERDICT_LABEL = "mlxcel-cluster-auth-v1/verdict"`, `VERDICT_RECORD_LEN = 1 + TAG_LEN` (33 bytes). One `write_verdict` helper sends `verdict \|\| tag_v` for reject, at-capacity and accept, with `tag_v = sign(VERDICT_LABEL, client_hello \|\| server_hello \|\| tag_s \|\| tag_c \|\| verdict)`. The server extends its transcript with `tag_c` whether or not it verified, so the reject verdict is signed over the same bytes as the others. The client reads exactly 33 bytes, verifies `tag_v` before the `match`, and returns `HandshakeRejected("server's handshake verdict does not verify ...")` without deriving keys. `PROTOCOL_VERSION = 5`; the version history, wire diagram and module doc are updated. |
| 2 | `pipeline/remote_service.rs` | `coordinator_identity: Option<String>` becomes `sequence_owners: HashMap<String, String>`. `handle_command` first asks `sequence_of(&command)` (the id named by `PrepareSequence`, `RunEntry`, `CancelRequest`, `ReleaseSequence`; `None` for `BeginDrain`, `Shutdown`, `ProbeState`) and `owned_by_other(id, from)`; a hit is answered with `Error("sequence <id> was prepared by another coordinator identity")` and changes nothing. `PrepareSequence` records the owner. `finish_request` removes it; `Shutdown` clears the map. `reply_target` returns `Option<String>`: the owner or nothing, with no fallback to the configured address. |
| 2 (review) | `pipeline/remote_service.rs` | On the first `PrepareSequence` of an id (no owner recorded yet), caches already stored under that id are removed and released before the owner is recorded. A re-prepare by the owner keeps its caches. |
| 2 | `pipeline/remote_service.rs` | Final stage: without an owner, the logits are dropped with a warning and the request is finished. |
| 3 | `router_front.rs`, `docs/distributed.md` | Doc only: why the sender port stays a claim. |
| a | `pipeline/runtime.rs` | `PipelineModelRuntime::prepare_sequence_state` returns `Result<()>`. The remote runtime `bail!`s while draining; when `broadcast_ack(PrepareSequence)` fails it removes the id mapping, sends a best-effort `broadcast_ack(ReleaseSequence)`, and returns the error with context. `request_id_for` returns `Result` through a shared `request_id_of`; the forward paths propagate it. `request_id_for_sequence` builds `pp-seq-N` in one place. |
| a | `mlxcel-core/src/generate.rs`, `language_model_ref.rs`, `loaded_model.rs` | New `LanguageModel::try_prepare_sequence_state(&self, SequenceId) -> Result<(), String>`. The default calls the infallible hook and returns `Ok`; `&M` and `LoadedModel` forward it. |
| a | `pipeline/server_runtime.rs` | `PipelineServerModel` overrides the fallible hook to return the runtime's error. The infallible `prepare_sequence_state` keeps a panic for callers with no error path. |
| a | `mlxcel-core/src/engine/mod.rs` | `Engine::open` calls `try_prepare_sequence_state`; on `Err` it closes the pool entry it just allocated and returns `EngineError::Open("prepare sequence state for <id>: ...")`. The scheduler's `allocate_sequence_state` callers (admission, decode re-prefill, queued adoption, prompt cache, handoff) already fail only their own request on `Err`. |
| b | `pipeline/remote_service.rs` | `handle_activation` becomes a thin wrapper: the body moves to `run_activation`, which uses `?` for every failure, and any `Err` goes to `report_activation_failure`. That function reads the owner before `finish_request` removes it, then sends the `Error` with the activation's `reply_nonce` to the owner, or logs and drops it when there is no nonce or no owner. `send_run_error` is folded into it. |
| Docs | `docs/distributed.md` | "What the secret protects" describes sequence ownership, the first-prepare discard, the no-fallback rule and the failure reporting; the handshake section and the layer table describe the authenticated verdict and version 5. |
| Tests | 6 test files | 13 new tests; existing tests now read the 33-byte verdict record, prepare the sequence before sending activations, and require the `Error` reply where they used to accept silence (section 4.1). The remote-stage test harness gained a configurable coordinator address, a rogue transport on the coordinator port, and a cache-release counter. Call sites in `tests/pipeline_remote_*.rs` and `runtime_correlation_tests.rs` unwrap the new `Result`; the drain test asserts `is_err()` instead of `catch_unwind`. |

### Statistics

| Item | Value |
|---|---|
| Files changed | 16 (9 source, 6 test, 1 docs) |
| Lines added / deleted | +1324 / -244 against main |
| Tests added | 13 (`cluster_handshake_tests.rs` 5, `remote_service_tests.rs` 5, `runtime.rs` 1, `server_runtime.rs` 1, `engine/tests.rs` 1) |
| Commits | 666a851f (items 1, 2, 3, a, b), 71a8c6e0 (first-prepare discard from the security review) |

## 3. Technical Decisions

### 3.1 MAC the verdict inside the handshake, not in the first protected frame

**Context.** The verdict needs integrity, and the handshake already has a keyed primitive (`ClusterAuth::sign`/`verify`) with label-based domain separation.

| Option | Assessment |
|---|---|
| Deliver the verdict as the first AEAD-protected frame | Changes every caller (`tcp_transport`, `connection_pool`, `stage_assignment`), and the `HmacSha256` suite is not an AEAD, so it would need its own path anyway |
| Authenticate the pre-hello `REJECT_MAGIC` record too | That record is sent before the client proved anything; altering any hello byte already forces an equally terminal failure |
| **Chosen: append `tag_v` to the verdict byte** | Contained in `cluster_handshake.rs`, reuses `sign`/`verify` and the existing `HandshakeRejected` classification in `tcp_transport.rs`, `stage_assignment.rs` and `handshake_warn.rs` |

The signed message is the full transcript through `tag_c` followed by the verdict byte, so the tag binds the decision to this exchange (both nonces) and to the client's own authenticator. The security review checked the obvious reuse attack: a client that failed authentication receives a signed reject verdict, but every signed verdict message ends in its verdict byte and the label is separated from the message, so a reject tag can never verify as accept or at-capacity. The server signs the reject verdict over whatever `tag_c` the client sent, which is why the transcript is extended before the verification branch.

**Trade-off.** The record grows from 1 to 33 bytes, so version-4 and version-5 nodes cannot talk. The version field makes them fail with the explicit `speaks cluster handshake version 4, this build expects 5` message instead of a confusing read error. The authenticated handshake landed after v0.8.0-beta.1, so no release is affected.

### 3.2 Ownership is per sequence and set only by `PrepareSequence`

**Context.** The old design had one reply target for the whole stage, and any accepted command could move it.

The owner is the authenticated `from` of the `PrepareSequence` (the identity the handshake bound to the TCP source IP, not the payload's `reply_addr`). Only `PrepareSequence` creates a binding. `RunEntry` for an unowned id still runs and creates none, because every coordinator `RunEntry` follows `prepare_sequence_state`; letting `RunEntry` bind would let a rogue claim an id with any command. `ProbeState` and `BeginDrain` neither need nor touch ownership. `Shutdown` is not per sequence and stays available to any identity that passes the coordinator matcher, which is the existing trust model: a secret holder on the coordinator port can already shut a stage down.

`reply_target` has no fallback. The configured coordinator address may be `0.0.0.0:<port>`, which names no host, and sending there would recreate the problem. A result or error for a sequence nobody prepared on that stage has no legitimate receiver, so it is logged and dropped.

### 3.3 The first prepare discards unowned state (review fix)

The security review found that ownership protected only ids that already had an owner. A rogue that passes the coordinator matcher could send `RunEntry { pp-seq-N+1 }` before the coordinator prepared that id; the stage built caches under it, and the coordinator's later `PrepareSequence` reused them through `caches_by_request.entry(id).or_insert_with(..)`. The rogue's tokens became a hidden KV prefix of another client's request: an integrity problem, not only availability. 71a8c6e0 makes the first `PrepareSequence` release any caches stored under the id before recording the owner. The owner's own re-prepare (owner already recorded) keeps its caches, so the existing re-prepare semantics are unchanged.

### 3.4 Router result-sender port stays a claim

Proving the port would need per-process key material (for example a per-process Ed25519 key signed into both hellos and pinned by the router), because all members share one secret and nothing in the current handshake tells two secret holders on one IP apart. Accepting results only on router-dialed connections would need reads on pooled outbound connections, which `TcpTransport` never does. The gain is small: the demux accepts any configured peer for any request id, so the remaining party is an unconfigured secret holder sharing an IP with a configured peer, which the threat model already treats as a member. The PR records these three points in the `router_front.rs` module doc and `docs/distributed.md`.

### 3.5 A fallible hook beside the infallible one

**Context.** `LanguageModel::prepare_sequence_state` is infallible and is implemented or forwarded by many model wrappers (the VLM wrappers in `src/vision/` forward it to their text model).

| Option | Assessment |
|---|---|
| Make `prepare_sequence_state` return `Result` | Touches every implementation and forwarder for a failure only one model can produce |
| Panic and catch it in the scheduler | Unwinding across the model worker is not a supported control path |
| **Chosen: add `try_prepare_sequence_state` with a default** | One override (`PipelineServerModel`), two forwarders (`&M`, `LoadedModel`), one caller change (`Engine::open`) |

`Engine::open` is the single place the scheduler allocates sequence state, so routing the fallible hook through it covers admission, decode re-prefill after preemption, queued adoption, the prompt cache path and the disaggregated handoff, all of which already handle `Err` from `allocate_sequence_state` by failing only their request. On failure `open` calls `close(id)`, which releases the pool entry and calls `release_sequence_state_by_id`; the remote runtime already removed the mapping, so that call is a no-op there.

**Trade-off.** The infallible `PipelineServerModel::prepare_sequence_state` still panics. The only remaining callers are `CachePool::adopt` and `adopt_parked_paged`, which the model-owned pipeline layout never takes; the security review checked that this panic is not reachable from peer input.

### 3.6 Rollback on a refused prepare

When the broadcast fails, the runtime removes the id mapping first (so a later forward is an error, not a forward under a half-prepared id), then sends `ReleaseSequence` to the stages. It reuses `broadcast_ack`, which stops at the first error. That is enough in the common case: the broadcast prepare also stopped at the first refusing stage, so the stages before it prepared the id and the release reaches them before reaching the refusing one. The reviews recorded the cost (section 5).

### 3.7 One exit for activation failures

Rather than adding a `report_activation_failure` call at each fallible step, the PR splits `handle_activation` into a wrapper and a `run_activation` body that uses `?` everywhere. Any new failure point added later is reported automatically. `report_activation_failure` reads the owner before `finish_request` removes it, which is the ordering bug this split would otherwise invite.

## 4. Verification

### 4.1 Failing-first evidence

New and tightened tests were run against main 86409e76 (logs in the run's scratchpad `2714-failing-first/`):

| Test | main 86409e76 | branch |
|---|---|---|
| `flipped_accept_to_at_capacity_verdict_is_rejected_not_retried` (in-test TCP proxy flips the byte at `HELLO_HEAD_LEN + server_identity.len() + TAG_LEN`) | FAILED (`ServerAtCapacity`) | ok |
| `flipped_at_capacity_to_accept_verdict_is_rejected_not_accepted` | FAILED (client `Ok`) | ok |
| `flipped_verdict_tag_byte_is_rejected` | FAILED | ok |
| `handshake_rejects_forged_client_tag` (now reads the 33-byte record) | FAILED (EOF after 1 byte) | ok |
| `server_at_capacity_sends_authenticated_capacity_verdict_record` | FAILED (EOF after 1 byte) | ok |
| `version_4_hello_is_refused_with_the_version_message` | FAILED (v4 accepted) | ok |
| `second_authenticated_peer_cannot_redirect_run_result` | FAILED (rogue `PrepareSequence` acked) | ok |
| `non_owner_cannot_cancel_or_release_a_prepared_sequence` | FAILED (non-owner command acked) | ok |
| `final_stage_drops_result_of_unprepared_sequence` | FAILED (`RunResult` sent) | ok |
| `activation_outside_stage_limits_never_executes` (now requires the `Error` with nonce 42) | FAILED (no reply in 5 s) | ok |
| `post_execution_failure_reports_error_to_coordinator` | FAILED (no reply in 5 s) | ok |
| `first_prepare_discards_state_left_by_unowned_commands` (against 666a851f) | FAILED (0 releases, expected 1) | ok |
| `remote_runtime_prepare_failure_is_an_error_not_a_panic`, `pipeline_server_model_reports_prepare_failure_through_try_prepare`, `open_fails_and_releases_the_entry_when_prepare_is_refused` | do not compile on main (new API); a `catch_unwind` probe on main panicked at `runtime.rs:605` | ok |

The rogue tests use a dual-stack topology: the stage listens on `[::]`, the coordinator binds `[::1]:<port>` but is configured as `[::]:<port>`, and the rogue binds `127.0.0.1:<port>`. A rogue on `[::1]` dialing an IPv4 stage is refused by the handshake before the matcher runs, so the IPv4 rogue is what reaches the wildcard matcher path the issue describes.

### 4.2 Local gate

GB10 spark-102 at 71a8c6e0, `--profile test-fast`, under `gpu-lock` with `--test-threads=1`, real-model targets with `MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1`:

| Target | Result |
|---|---|
| `cargo fmt --all -- --check`; `cargo clippy --features cuda -p mlxcel --lib --tests -D warnings`; clippy on the touched real-model targets | clean |
| `--lib distributed::` | 1544 passed, 4 ignored |
| `--lib server::router_front` | 40 passed |
| mlxcel-core `engine::` | 71 passed |
| `zero_config_cluster_init --include-ignored` | 7 passed |
| `pipeline_remote_real_models` (llama, gemma3, gemma4, qwen3, qwen3.5 parity; llama drain and shutdown) | 6 passed |
| `pipeline_server_remote_real_models` (same five models) | 5 passed |
| `pipeline_ci_multi_stage_real_models --include-ignored` (two-host smoke, three-stage parity) | 4 passed |
| `disaggregated_serving_e2e`, `disaggregated_router_e2e` (`--include-ignored`) | 2, 7 passed |

Not run: `glm_moe_dsa` (the 418 GB checkpoint does not fit this box), and the gpt-oss, glm4, Thunderbolt and RDMA variants of the remote real-model suite. No performance measurement was taken. The per-token path gains one `HashMap` lookup per command and per final-stage result, and the verdict adds one HMAC per handshake.

### 4.3 Hosted CI

On 71a8c6e0 (and 666a851f), `Pipeline Parallel CI / 2-host logical (ubuntu-latest)` failed. The test itself, `pipeline_multi_stage_two_host_logical_smoke`, reported `ok`; the test binary then aborted at process exit with `pure virtual method called` and SIGABRT. The same exit abort appears on runs of other branches that day (#2716 at ed803065, #2712 at 89f5c196), while other runs of those branches passed, so it is an intermittent teardown abort in the test binary that predates this PR, not a protocol failure. It still blocks a green check and needs either a rerun or its own issue.

## 5. Remaining Gaps and Risks

- **Pre-claimed ids still fail requests (LOW, availability).** A secret holder that passes the coordinator matcher can `PrepareSequence` the next `pp-seq-N` ids before the coordinator does. Each such request now fails cleanly with an error instead of aborting the coordinator, but it still fails. Non-sequential (random) request ids would close this; they would change the id scheme every stage and the coordinator share.
- **`sequence_owners` has no cap (LOW).** Like `caches_by_request`, which it mirrors, the map grows with every id a coordinator-matching peer prepares and only shrinks on finish or shutdown. Unowned `RunEntry` commands can still create caches under arbitrary ids until a prepare or finish removes them.
- **Prepare-failure cleanup is best effort (LOW).** `broadcast_ack(ReleaseSequence)` stops at the first error and waits a full `stage_timeout` per unresponsive stage on the model-worker thread. If a prepare timed out on a stage that applied it but lost its `Ack`, that stage keeps the sequence until shutdown. `release_sequence_state_by_id` has the same semantics.
- **The client sees a misleading error label (LOW).** A refused prepare reaches the API client as `Server busy: open: prepare sequence state ...: remote stage <addr> failed request: ...`. It is not a capacity condition, and the text names an internal stage address. Before this PR the same condition aborted the process.
- **Middle stages still execute unowned activations (LOW).** Only the authenticated upstream can send a forward activation, so a middle stage computes and forwards hidden states for a sequence with no recorded owner; the final stage drops the logits and finishes the request.
- **Forward errors still panic in the infallible API (pre-existing).** `PipelineServerModel::forward_with_sequence_id` and `forward_batched_with_ids` turn a remote forward error into a panic because the `LanguageModel` forward API is infallible.
- **Owner identity must be stable for a sequence.** Ownership compares the handshake-bound identity string. A coordinator whose identity is derived from the source IP (configured on an unspecified IP) and whose route to a stage changes mid-sequence would see its own commands refused. Routes are normally stable for a connection's lifetime; no test covers a change.
- **Not run.** `glm_moe_dsa` and the gpt-oss, glm4, Thunderbolt and RDMA variants; the hosted `2-host logical` job is red for the exit abort described in 4.3.

## 6. Learning Points

### 6.1 Authenticate every decision a protocol branches on

An authenticated handshake that ends in an unauthenticated byte gives an on-path party the choice of how the connection ends. The rule the PR applies is: any field the peer branches on after keys exist (or would exist) is covered by a MAC over the transcript and that field.

```rust
let (verdict, tag_v) = record.split_at(1);
if !auth.verify(VERDICT_LABEL, &verdict_message(&transcript, verdict[0]), tag_v) {
    return Err(HandshakeRejected("server's handshake verdict does not verify ...".into()).into());
}
match verdict[0] { /* accept, at capacity, reject */ }
```

### 6.2 Bind a reply target to the resource, not to the latest request

A single "where replies go" field that every command can move is a redirect primitive. Recording the target per resource at creation time, and refusing commands on that resource from anyone else, turns it into an ownership check that the rest of the code can rely on.

### 6.3 Ownership needs a clean start, too

Refusing non-owners protects a resource only after it has an owner. If unowned commands can create state under an id, the first owner must start from scratch, or a rogue that acts first defines the owner's initial state. This is the same shape as a TOCTOU on a predictable name.

### 6.4 Add a fallible variant with a default instead of changing a widely implemented trait method

A trait method implemented across dozens of wrappers is expensive to change. A sibling `try_*` method whose default calls the infallible one keeps every existing implementation valid; only the one implementation that can fail overrides it, and only the one caller that can handle the error switches to it.

### 6.5 One error exit per handler

Moving a handler's body into a function that returns `Result` and reporting from a single wrapper guarantees that every failure, including ones added later, takes the same reporting path. It also makes ordering explicit: here the owner is read before `finish_request` deletes it.

## 7. References

- Issue #2714: authenticate handshake verdicts and bind reply targets
- PR #2705 (issue #2692): cluster pre-auth admission hardening and the at-capacity verdict (`PROTOCOL_VERSION` 4); its report's section 3.7 is the source of item 1
- PR #2708 (issue #2694): final stage sends `RunResult` straight to the coordinator; its report is the source of items 2 and 3
- PR #2720 (issue #2712): `report_activation_failure` for execution and forward failures; item b extends it to every exit
- PR #2721 (issue #2716): tensor decode limits; its first remaining gap is fixed by item b
- PRs #2674, #2684 (authenticated handshake, per-frame AEAD) and #2650 (cluster secret): why every item here requires the secret or an on-path position
