# Technical Report: PR #2680 - Push Stage Assignments to --pp-peer Processes

**Date**: 2026-10-27  
**Status**: Completed (report written before merge; PR open as draft)  
**Languages**: Rust  
**Risk Level**: Medium

## Executive Summary

PR #2680 makes zero-config pipeline bring-up work end to end. `mlxcel-server --pp-auto N` now pushes a stage assignment to N `mlxcel-server --pp-peer` processes over the authenticated cluster handshake, in both static and mdns discovery, and the peers no longer need a hand-copied cluster TOML or a guessed `--node-id`. The assignment protocol is a two-phase Offer/Accepted/Commit/Committed exchange in a new module, `src/distributed/stage_assignment.rs`; the existing stage startup path is reused unchanged. On the GB10 host, a three-process qwen3-0.6b-4bit cluster returned the same greedy `/v1/chat/completions` text as a single-node server in both discovery modes.

## 1. Problem Statement

Before this PR the zero-config flags described a flow that did not exist (issue #2658):

- **mdns discovery could never finish.** `broadcast_beacon_loop` had no production caller, so no process sent a beacon, and the coordinator always failed after the 30 s discovery timeout. Even with a sender, the coordinator bound its UDP receiver to the unicast control IP, and on Linux and macOS such a socket does not receive datagrams sent to `255.255.255.255`.
- **`--pp-peer` was a manual path in disguise.** It required `--distributed-config` (a TOML copied from the coordinator) and `--node-id`, which had to equal the `stage-{i}` that `plan_cluster` derives from the lexicographic order of peer addresses. The operator had to guess it.
- **The help text promised behavior that was not implemented**: registration against a coordinator and a stage assignment returned by it.
- **The coordinator's generated TOML was not portable.** It carried the coordinator's absolute `secret_file` path, and with `--host 0.0.0.0` it recorded the coordinator as `0.0.0.0:19000`, which a stage's `PeerMatcher` would accept from any authenticated member on that port.

The direction was a deliberate reversal: #2655 had planned to remove the mdns mode, and that removal was withdrawn in favor of making it work. The PR builds on the protocol v2 handshake from #2655 (typed identity rejection, `PendingHandshakeLimiter`) rather than extending it.

## 2. Technical Decisions

### 2.1 Push from the coordinator, not pull from the peer

In static mode only the coordinator knows the peer addresses (`--cluster-peers`), and in mdns mode the coordinator learns them from beacons. In both cases the coordinator holds the plan, so it dials each peer's listener and pushes the assignment. The peer only needs to know its own listen address. A pull model would have required peers to know the coordinator address in static mode, which is the reverse of how the flags are distributed.

### 2.2 Two-phase commit over the existing handshake

| Option | Outcome |
|---|---|
| Single Offer/Ack round | A failure on peer 2 leaves peer 1 running a stage for a coordinator that exits |
| **Chosen: Offer, then Commit only after every peer accepted** | A failed offer phase commits no peer; failures are reported per peer |
| Full 2PC with an abort message | Would also cover failures during the commit phase, at the cost of more protocol state |

The channel reuses `client_handshake`/`server_handshake`: the coordinator advertises its control address, the peer advertises its listener, and the peer binds the coordinator's identity to the TCP source IP exactly as the stage transport does. Frames after the handshake are a `u32` big-endian length plus JSON, capped at 64 KiB (checked before allocation). JSON was chosen over the binary stage protocol because the frames are small, rare, and need to evolve; `ASSIGNMENT_PROTOCOL_VERSION` versions them independently of the handshake.

The commit phase has a residual gap: if a peer fails between `Commit` and `Committed`, peers that already committed stay committed while the coordinator exits. This is documented as a known limitation instead of being solved with an abort frame.

### 2.3 The peer re-derives what it is told

The offer carries a `ModelIdentity` (SHA-256 of `config.json`, sorted `*.safetensors` names and byte sizes, layer count) and the layer range the coordinator computed for the stage. The TOML carries no layer range and each stage still derives its own partition from its checkpoint; the layer range in the offer exists only as a second identity check, so a config change that moves the partition is caught even if the hash check were weakened later. `ModelIdentity` reads no weight bytes, which keeps bring-up fast on large checkpoints but means two checkpoints with identical config and identically sized but different shards would match.

Verification runs in a fixed order: protocol, mlxcel version, cluster name, model, stage count bounded by layer count, layer range, `node_id == "stage-{index}"`, TOML structure (our stage at our listener port, exactly one coordinator node), and both the offer's `coordinator_control_addr` and the TOML's coordinator node against the handshake-bound identity. Protocol, version, model, and layer-range mismatches are fatal (`AssignmentRefused`) because a retry against the same coordinator cannot succeed; name, address, and TOML problems return the peer to listening so a misconfigured coordinator does not take down a correctly configured peer.

The own-node check compares only the port, not the IP. The coordinator may know the peer by an address that differs from the one the peer bound (for example `0.0.0.0` on the peer side), and `localize_cluster` replaces the peer's own address with the listener address afterwards.

### 2.4 The node id is a file name

The peer writes the received cluster to `.mlxcel/cluster-<node_id>.toml` with `create_dir_all`. The TOML validator accepts `/` and `..` in ids, so an unchecked id let any secret holder acting as coordinator write outside `.mlxcel/`. The security pass restricted the id to exactly `stage-<stage_index>` in `verify_offer`, and startup re-checks `[A-Za-z0-9_-]` before building the path. The same pass required the TOML's coordinator node to agree with the bound identity, since that address (not the offer field) is what the stage trusts and replies to.

### 2.5 Timeouts as a budget, not independent knobs

| Bound | Value | Notes |
|---|---|---|
| Handshake | 5 s | From #2655 |
| Frame I/O | 10 s | Every frame read except the commit wait |
| Peer wait for `Commit` after `Accepted` | 60 s | Read without the 10 s frame bound (fixed in c8d257cb) |
| Coordinator push | 60 s | Offer phase ends 10 s early (`commit_reserve`) |
| Peer wait for any assignment | 600 s | Then exits with a message naming address, mode, cluster |

Two defects in this area were found during review. The first version read `Commit` through `read_frame`, whose 10 s bound fired before the 60 s commit timeout, so an early peer gave up whenever the coordinator needed more than 10 s to reach a later peer (a static peer starting late). The second: offer and commit shared one deadline, so a late `Accepted` could leave no time for `Commit` and split the peers. `commit_reserve(timeout) = min(ASSIGNMENT_IO_TIMEOUT, timeout / 2)` now guarantees the commit phase a full frame window inside the overall 60 s.

### 2.6 Retrying connect and handshake

The coordinator retries connect plus handshake every 500 ms until the offer deadline and stops early only on `HandshakeRejected` (secret or identity mismatch). This covers two cases with one loop: a static peer that starts after the coordinator, and a peer whose 8-total/4-per-IP pre-auth slots are held by unauthenticated sockets and closes the coordinator's socket at accept. Without the retry, any host on the network could fail a bring-up by holding those slots.

### 2.7 Rejected alternatives recorded in the issue

- Handing the bound assignment listener to `RemoteStageServiceHandle::spawn`: it binds on its own thread and runtime and would need a non-`Clone` field in `RemoteStageServiceConfig`. The listener is dropped instead and the stage service rebinds the port (tokio sets `SO_REUSEADDR` on Unix). The coordinator's first stage connect can race that rebind, which `Transport::connect` already tolerates with a lazy reconnect.
- A coordinator-side broadcast target: the coordinator only listens, so `--cluster-discovery-target` is a peer flag.

## 3. Change Summary

| Area | Change |
|---|---|
| `distributed::stage_assignment` (new, 1292 lines) | Frames, `ModelIdentity`, `verify_offer`, `localize_cluster`, `await_stage_assignment` (peer), `push_stage_assignments` (coordinator), typed `AssignmentRefused` and `AssignmentPushFailed` |
| `distributed::cluster_init` | `broadcast_beacon_loop` takes a `target: IpAddr` instead of the hard-coded limited broadcast; doc no longer calls mdns "future zeroconf" |
| `distributed::tcp_transport` | `handshake_warn_allowed` is `pub(crate)` so the assignment listener shares its rate limit |
| `server::startup` | Coordinator: receiver bound to the unspecified address, loopback-control refusal with non-loopback peers, push between `plan_cluster` and the TOML write (skipped under `--dry-run`). Peer: fail-closed secret, listen-address resolution, assignment wait, localized TOML write, `distributed_config`/`node_id` set so `resolve_distributed_startup` is unchanged. `--pp-peer --dry-run` refused before binding |
| CLI (`mlxcel-server`, `mlxcel serve`) | New `--cluster-discovery-target` (default `255.255.255.255`); help for `--pp-auto`, `--pp-peer`, `--cluster-*` rewritten; `--cluster-control-addr` is the peer's listen address on `--pp-peer` |
| `docs/distributed.md` | "Zero-config bring-up" walkthrough: secret distribution, static and mdns three-host examples, firewall ports, timeouts, `--dry-run` restriction |
| Tests | 14 `stage_assignment` unit tests, 3 startup tests, 2 weightless loopback bring-ups, 2 ignored real-binary parity tests |

Behavior change for operators: `--pp-peer` without `--distributed-config` now waits for an assignment instead of failing; static mode requires `--cluster-control-addr` with a concrete port; every `--pp-peer` process needs the cluster secret before it binds anything. `--pp-peer --distributed-config --node-id` keeps the manual path. No new environment variable; the handshake protocol is unchanged.

Statistics: 16 files, +3107/-125.

| Commit | Message |
|---|---|
| `cce9b212` | feat(distributed): push stage assignments to --pp-peer processes |
| `c8d257cb` | fix(distributed): wait the full commit timeout for Commit on a peer |
| `0cb9fd38` | fix(security): bind assignment node ids and coordinator to the handshake |
| `482f5a5c` | fix(distributed): harden zero-config bring-up edge cases |
| `0e3fd2d2` | Merge origin/main |

## 4. Learning Points

**Broadcast reception depends on the bind address.** A UDP socket bound to a unicast IP does not see datagrams addressed to `255.255.255.255` on Linux or macOS; the receiver must bind the unspecified address. Limited broadcast also leaves only through the default-route interface and never crosses loopback, which is why `--cluster-discovery-target` accepts a directed subnet broadcast or a unicast IP, and why the loopback tests use `127.0.0.1` as the target.

**A real-broadcast run confirmed both halves.** With the default target, the coordinator's receiver on `0.0.0.0` did receive the peer's limited-broadcast beacon, but the datagram left through the LAN interface (source `10.100.64.122`) while advertising a loopback listener, so `BeaconVerifier` rejected it once per second ("beacon advertises 127.0.0.1 but was sent from 10.100.64.122") until the 30 s discovery timeout. That is the source-IP binding from #2650 working as designed; a full broadcast bring-up needs the stage listeners on a LAN address.

**Inner timeouts can silently shorten outer ones.** The c8d257cb bug is a general pattern: a helper with its own per-call timeout wrapped in a longer `tokio::time::timeout` gives the shorter bound, not the longer one. The fix keeps `read_frame` bounded everywhere else and adds `read_frame_unbounded` for the one call whose bound comes from outside.

**Commit claim with rollback.** The peer claims its single commit with a `compare_exchange` on an `AtomicBool` after reading `Commit`, and releases the claim if writing `Committed` fails, because a coordinator that never saw `Committed` fails its push and will not drive that stage.

## 5. Validation

Final head, GB10 spark-102, `--profile test-fast --features cuda`, `--test-threads=1`, GPU work under `gpu-lock run --tag issue-2658`:

- `distributed::stage_assignment` lib tests: 14 passed. `server::startup`: 97 passed.
- `zero_config_cluster_init`: 4 passed (static and mdns weightless bring-ups), 3 ignored.
- `cli_help_consistency`: 27 passed. Clippy `-D warnings`, `cargo fmt --check`, `make verify-env-var-docs`: clean.
- Earlier heads on this branch: `distributed::` lib 1411 passed; ignored `zero_config_cli_{static,mdns}_matches_single_node` passed with `MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1`.

Real-binary E2E on the final head (qwen3-0.6b-4bit, `MLXCEL_SDPA_DETERMINISTIC=1`, loopback ports 19500-19549, `temperature: 0`, `max_tokens: 24`): in both mdns (target `127.0.0.1`) and static modes, the single-node and cluster servers both returned HTTP 200 with the identical text "Three primary colors are **red, blue, and yellow**. These are the basic colors in the color wheel, representing". On loopback the whole push, from "Pushing stage assignments to 2 peer(s)" to both "committed" lines, took about 2 ms. The peer-written TOML contained the coordinator at `127.0.0.1:19500`, both stages, and no `secret_file`.

Not verified: a multi-host bring-up on a real LAN (every run used loopback), the mdns `0.0.0.0:0` default listen address through the real binary, macOS, and `--test distributed_integration` (it references none of the changed functions).

## 6. Remaining Items

Left open by design (recorded in the PR):

- Assignment and stage-transport handshakes carry no channel label, so they are interchangeable for a secret holder; there is no per-frame MAC.
- A beacon advertising `0.0.0.0:port` signs no IP; a LAN attacker who can spoof UDP can replay it from another source. The handshake still binds identity, so the effect is limited to a failed bring-up.
- A failure during the commit phase can leave some peers committed while the coordinator exits.
- A secret holder acting as coordinator can end a peer with a fatal rejection (for example a mismatched model); fatal reasons are limited to mismatches a retry cannot fix.

Minor observations from this review, not fixed in the PR:

- In `src/server/startup.rs`, the doc comment for `resolve_pp_peer_listen_addr` ("The listen address of a `--pp-peer` process ...") was merged into the doc block of `ensure_pp_peer_not_dry_run`, leaving `resolve_pp_peer_listen_addr` undocumented.
- A peer-written cluster TOML keeps the header "Generated by `mlxcel-server --pp-auto`", although the peer localized it.
- `tests/zero_config_cluster_init.rs` prefers this host's 19500-19549 port block and falls back to ephemeral ports elsewhere.

Related: #2658 (closed by this PR), #2655, #2650, #2637, #2654 (llama, gemma3, gemma4 remote parity, the reason the parity tests use qwen3).
