# PR #2746: Bound remote stage state, RPC replies and tensor axes

**Date**: 2026-10-30 (written before merge; the PR is open as a draft)
**Status**: Implementation complete at 4a18642b, with origin/main merged in at c46934f4. The security review found one gap (prepared request ids of any length were stored, so the count cap did not bound their memory), fixed in 4a18642b. The local gate in the PR body passed on GB10 (spark-102) at 4a18642b. At the time of writing, a second local gate was running on the merged head and hosted CI was partly pending (`2-host logical`, `cargo-clippy`, `CPU-only link`, `OpenXLA feature compile`, `WebUI installed artifact`)
**Languages**: Rust, Markdown
**Risk Level**: Low (no wire version bump; stages now refuse `RunEntry` and activations for unprepared ids and new prepares beyond 4096 sequences; the public `DimBound` and `ShapeBound::Uniform` types gain a variant and a field)

## Executive Summary

PR #2746 (closes #2726) closes four LOW security residuals that #2720, #2721 and #2723 left in the remote pipeline stage service, its tensor decode bounds and the TCP transport, plus one robustness item from the same module. A remote stage now refuses `RunEntry` and forward activations for any id no coordinator prepared on it, and holds at most `MAX_STAGE_SEQUENCES` (4096) owned sequences; together those two rules bound every per-sequence map. `PrepareSequence` also refuses ids that `RequestId::from_string` rejects, so the cap bounds memory, not only entry count. The remote coordinator's request ids gain a CSPRNG suffix (`pp-seq-{n}-{16 hex}`), so a second secret holder can no longer pre-claim the next id. `rpc_call` reads replies under a 1 MiB cap instead of the 256 MiB frame limit, sequence and mask axes refuse zero length, and reconnect backoff is jittered with the server's at-capacity warning moved onto the rate-limited warning budget.

Items 1 to 4 are reachable only by a peer holding the cluster secret (item 3 also by an on-path party). None of them lets a party without the secret read or inject frames.

## 1. Problem Statement

#2726 collected residuals verified against main 59c07652. It asked for all five items in one PR with one verification pass at the end.

### 1.1 Per-stage sequence state was unbounded (LOW, secret holder)

`RemoteStageService` keeps three per-sequence maps: `caches_by_request`, `active_requests` and `sequence_owners`. None had a count limit. `PrepareSequence` inserted an owner and a cache set for any new id, and `RunEntry` and forward activations built a cache set for ids nobody had prepared, because `owned_by_other` let unowned ids through. A secret holder passing the coordinator matcher could grow all three maps without bound, and a middle stage executed activations for unowned ids (the "middle stages still execute unowned activations" gap in the #2723 report). #2723 had papered over the integrity half of this by making the first `PrepareSequence` discard caches found under its id; the memory half stayed open.

### 1.2 Sequential remote request ids were predictable (LOW, secret holder)

`request_id_for_sequence` built `pp-seq-{N}` from the scheduler's sequential `SequenceId`. Since #2723 introduced per-sequence ownership, a secret holder that prepared `pp-seq-{N+1}` first made the coordinator's own prepare fail with "prepared by another coordinator identity". #2723 made that failure clean (the request fails, the coordinator survives), but every pre-claimed id still denied one request. The #2723 report lists this as its first remaining gap.

### 1.3 `rpc_call` replies were read under the full frame limit (LOW, secret holder or on-path)

`TcpTransport::rpc_call` read its reply with `config.max_frame_size`, 256 MiB by default. `read_frame` allocates `len + tag` before the frame authenticates, so a peer, or a party on the path that rewrites the length header, could force one 256 MiB allocation per in-flight call. The only caller is the latency benchmark (`bench.rs`), which exchanges 64 bytes.

### 1.4 Zero-length axes passed the tensor decode limits (LOW, secret holder)

`DimBound::AtMost(n)` admits 0, so hidden states `[1, 0, width]`, ids `[1, 0]` and logits `[1, 0, vocab]` passed `StageWireLimits` and reached the executor. `ShapeBound::Uniform` admitted zero mask axes. No honest coordinator or stage produces an empty sequence axis; such a tensor only reaches model code paths that were never written for it.

### 1.5 Lockstep reconnects and an unrated capacity warning (robustness)

`connect_with_retry` slept a fixed doubling delay after refused connects, timeouts and the at-capacity verdict, so clients refused together retried together. N concurrent acquires against a full peer made up to 6N handshakes (3N with `PoolConfig::router()`), and each wrote an unrated server warn ("max connections (N) reached").

## 2. Change Summary

| Item | Area | Change |
|---|---|---|
| 1 | `pipeline/remote_service.rs` | `handle_run_entry` refuses an id with no `sequence_owners` entry before decoding any tensor: `Error("sequence <id> was not prepared on this stage")` via `send_request_error`, nothing created. `run_activation` `ensure!`s an owner first, so an unowned activation goes through the existing `report_activation_failure` path (logged and dropped, since there is no owner to report to) without decoding or executing. |
| 1 | `pipeline/remote_service.rs` | `pub(crate) const MAX_STAGE_SEQUENCES: usize = 4096`, stored in a new `max_sequences` field set in `load`. A `PrepareSequence` for a new id while `sequence_owners.len() >= max_sequences` gets `Error("pipeline stage already holds {max} prepared sequences")` and changes nothing; the owner's re-prepare of an id it already holds is not a new entry and still gets `Ack`. A `#[cfg(test)]` `spawn_with_executor_and_sequence_cap` lowers the cap; `spawn_with_executor` and the new constructor share one private `spawn_service`. |
| 1 (review) | `pipeline/remote_service.rs` | `PrepareSequence` first validates its id with `RequestId::from_string` (1 to 256 bytes). An invalid id is answered with `Error("invalid request id (must be 1 to 256 bytes)")` with no request id attached, before the draining and cap checks, and nothing is stored. |
| 1 | `pipeline/remote_service.rs` | The first-prepare stale-cache discard from #2723 stays as defense in depth; the module doc and `owned_by_other` doc describe the new rules. |
| 2 | `pipeline/runtime.rs` | `RemotePipelineRuntime::prepare_sequence_state` builds `pp-seq-{n}-{auth.random_u64():016x}` and stores it in `request_ids`, from which every later command reads it. `InProcessPipelineRuntime` keeps `pp-seq-{n}` through `request_id_for_sequence`. The test `StubStageHandle` now records each `PrepareSequence` id. |
| 3 | `tcp_transport.rs` | `const MAX_RPC_RESPONSE_FRAME: usize = 1 << 20`; `rpc_call` reads with `MAX_RPC_RESPONSE_FRAME.min(config.max_frame_size)`. An oversize reply fails the call on its header and the existing error arm drops the stream. |
| 4 | `tensor_serialize.rs` | New `DimBound::Within { min, max }` (inclusive, displayed as "between {min} and {max}"). `ShapeBound::Uniform` gains `min_dim`; the check builds a `DimBound` (`AtMost` when `min_dim == 0`, else `Within`) and reports "axis {i} is {d}, expected {bound}". `TensorDecodeLimits::unrestricted` uses `min_dim: 0`. `AtMost` is kept. |
| 4 | `pipeline/wire_tensor.rs` | A `sequence_axis(max)` helper returns `Within { min: 1, max }`, used for the sequence axis of `hidden_states()`, `input_ids()` and `logits()`. `attention_mask()` uses `min_dim: 1`. `exact_or_unbounded(None)` becomes `Within { min: 1, max: u64::MAX }`. |
| 5 | `connection_pool.rs` | `fn jittered(delay) -> Duration` returns a uniform value in `[delay/2, delay]` from `ring::rand::SystemRandom`, falling back to `delay` if the RNG fails. `connect_with_retry` draws it once per attempt, logs it in the at-capacity debug line and sleeps it; the doubling of `delay` is unchanged. |
| 5 | `tcp_transport.rs` | The accept loop's `ConnectionLimitReached` branch logs at warn only when `handshake_warn_allowed(HandshakeWarnClass::AtCapacity)` allows it, otherwise at debug, like the neighbouring handshake-failure branches. |
| Docs | `docs/distributed.md` | "What the secret protects" describes the unowned-command refusal, the 4096 cap, the id validation and the random id suffix; the tensor bounds read "between 1 and 2^20"; the backoff paragraph mentions the jitter and the rate-limited capacity line; the residual-risk list notes the 1 MiB `rpc_call` cap. There is no Korean counterpart of this document. |
| Tests | 6 test files and `runtime.rs` | 8 new tests, 1 rewritten, several adjusted (section 4.1). `LogBuf` in `tcp_transport_frame_tests.rs` became `pub(super)` so the robustness tests can capture logs. |

### Statistics

| Item | Value |
|---|---|
| Files changed | 13 (6 source, 6 test, 1 docs) |
| Lines added / deleted | +725 / -95 against main 060b5051 |
| Tests added | 8 new (`remote_service_tests.rs` 3, `runtime.rs` 1, `wire_tensor_tests.rs` 1, `tcp_transport_robustness_tests.rs` 2, `connection_pool_tests.rs` 1), 1 rewritten |
| Commits | e8539643 (items 1 to 5), 4a18642b (prepare id validation from the security review, jittered delay logged as slept), c46934f4 (merge of origin/main) |

## 3. Technical Decisions

### 3.1 Refuse unowned commands instead of capping each map

**Context.** Three maps grew per id, and two different paths created entries: `PrepareSequence` (owned) and `RunEntry` or activations (unowned).

| Option | Assessment |
|---|---|
| Cap each map separately, evicting old entries | Eviction of an owned entry silently breaks a live sequence; three caps to keep consistent |
| Expire unowned state after a timeout | Needs a timer per id and still lets a rogue keep the maps full in steady state |
| **Chosen: no state without an owner, and one cap on owners** | Every entry in `caches_by_request` and `active_requests` now exists only for an owned id, so a cap on `sequence_owners` bounds all three |

Honest traffic is unaffected because every coordinator `RunEntry` follows a `PrepareSequence` broadcast to every stage (`runtime.rs`), and an activation for a sequence reaches a stage only after the coordinator prepared it there. The `RunEntry` refusal sits before the tensor decode, so a refused command costs a map lookup and a short error reply, not a decode. The activation path uses `ensure!` inside `run_activation` rather than a separate check, so it goes through the single failure exit #2723 introduced; with no owner, that exit logs and drops.

The cap refuses new ids instead of evicting. Refusing keeps every admitted sequence valid, and the coordinator already handles a refused prepare by failing only that request (#2723 item a).

**Trade-off.** The cap is stage-wide, not per identity. A secret holder that passes the coordinator matcher can fill all 4096 slots with its own prepares and never release them, after which every new honest prepare on that stage fails until those entries are released or the stage shuts down. The PR converts unbounded memory growth into a bounded, visible availability failure; it does not remove the availability attack, which needs per-identity accounting or expiry (section 5).

### 3.2 Validate the prepared id so the count cap is also a memory cap (review fix)

The security review noted that a stage stored each prepared id (as a `String` key or value) in three maps, and the id could be as long as a frame. 4096 entries of near-frame-size ids would bound the count but not the memory. 4a18642b refuses any id `RequestId::from_string` rejects (empty or over 256 bytes), which is exactly the set of ids an honest coordinator can never build, because it constructs ids with the same function. The check runs first, before the draining and cap checks, and the error reply carries no request id, so a large id is not echoed back across the wire. With the check, the stored ids are bounded at roughly 4096 x 3 x 256 bytes.

### 3.3 A random suffix, not a random id

**Context.** Item 2 needs ids a second secret holder cannot predict.

| Option | Assessment |
|---|---|
| Fully random id | Loses the sequence number, which logs and tests use to correlate a remote request with a scheduler sequence |
| Keep `pp-seq-{n}` and rely on ownership | Leaves the pre-claim denial from #2723's gap list open |
| **Chosen: `pp-seq-{n}-{random_u64:016x}`** | Unguessable (64 bits from the `ClusterAuth` CSPRNG already used for exchange nonces), still correlatable, and within the 256-byte id limit |

The id is generated once, stored in `request_ids`, and every later command reads it from there, so no other `runtime.rs` code changes. `InProcessPipelineRuntime` keeps the plain form because it has no peers to race with. Stages treat ids as opaque strings, so old stages accept the new ids and no wire version bump is needed.

### 3.4 A reply-sized cap for `rpc_call`, not a config field

`rpc_call` has one caller, the latency benchmark, with a 64-byte echo. A configurable cap would add a field no caller sets. The constant is combined with `min(config.max_frame_size)` so a transport configured with a smaller frame limit keeps it. The oversize error arises in `read_frame` from the header alone, before any allocation, and the existing `Ok(Err)` arm drops the stream so the connection is never returned to the pool with unread bytes.

### 3.5 `Within` beside `AtMost`, and `min_dim` on `Uniform`

**Context.** The bound types live in the public `mlxcel::distributed::tensor_serialize` module.

Changing `AtMost` to mean "1 to n" would silently change every caller's semantics, including `TensorDecodeLimits::unrestricted`, which must keep accepting empty tensors. A new `Within { min, max }` variant makes the intent explicit at each use site; the stage limits use it through a `sequence_axis` helper so the "minimum 1" decision lives in one place. For `Uniform`, a `min_dim` field keeps the uniform-rank form for masks, with `0` in the unrestricted limits. The `Uniform` check now constructs a `DimBound` and reuses `admits` and `Display`, so the two shape forms produce the same error wording.

**Trade-off.** Neither enum is `#[non_exhaustive]`, so the new variant and field are a source break for any external crate that matches exhaustively on `DimBound` or builds `ShapeBound::Uniform` with a struct literal. No code outside `src/distributed/` uses either type.

### 3.6 Jitter the sleep, keep the doubling

Equal-jitter (`[delay/2, delay]`) spreads clients refused together while keeping a guaranteed minimum wait, which the at-capacity path relies on to give a full server time to free a slot. The doubling of `delay` itself is unchanged, so the total budget's upper bound is the same as before. The jitter is drawn once per attempt before the connect, so the debug line prints the value that is actually slept (a review fix in 4a18642b; the first version logged the pre-jitter delay). `SystemRandom` is already a dependency through `ClusterAuth`.

The server's capacity warning moves onto the existing `HandshakeWarnClass::AtCapacity` budget instead of a new class. That class is also used by the connecting side's at-capacity line, so in a process that is both client and server the two share one budget; the operator still sees the first few lines per window and the per-minute suppressed-warning summary.

## 4. Verification

### 4.1 Failing-first evidence

The PR body records the new tests run against unchanged main 8385f40b:

| Test | main 8385f40b | branch |
|---|---|---|
| `unprepared_run_entry_is_refused_and_creates_no_state` | FAILED (got `RunResult`, expected `Error`) | ok |
| `unowned_run_entry_cannot_seed_state_before_first_prepare` (rewrite of #2723's `first_prepare_discards_state_left_by_unowned_commands`) | FAILED (got `RunResult`) | ok |
| `final_stage_drops_result_of_unprepared_sequence` (now also asserts zero executions) | FAILED (1 execution) | ok |
| `remote_runtime_request_ids_are_unpredictable` | FAILED ("expected pp-seq-21-<16 hex digits>, got pp-seq-21") | ok |
| `zero_length_axes_are_refused` | FAILED (`[1, 0, 8]` accepted) | ok |
| `rpc_call_refuses_oversize_response` | FAILED (2 MiB reply returned) | ok |
| `full_server_rate_limits_capacity_warnings` | FAILED (20 of 20 at warn) | ok |
| `prepare_with_invalid_request_id_is_refused` (review) | FAILED with the check disabled (300-byte id got `Ack`, `in_flight_requests: 1`) | ok |
| `prepare_beyond_the_sequence_cap_is_refused`, `jittered_delay_stays_in_half_to_full_range` | do not compile on main (new API) | ok |

Adjusted existing tests:

- `run_entry_with_bad_tensor_*` and `run_entry_with_out_of_bounds_ids_*` now prepare first; without that they would hit the new ownership refusal instead of the tensor check they exist for.
- `server_at_capacity_exhausts_retry_budget` lowers its elapsed lower bound from 60 ms to 30 ms, the minimum of two jittered sleeps of 20 and 40 ms.
- `full_server_rate_limits_capacity_warnings` asserts at most `2 * HANDSHAKE_WARN_PER_WINDOW` warns rather than the issue's "exactly one". The budget allows 5 per window, it is process wide and shared with the pool's own at-capacity line, and one window rollover can fall inside the test. All 20 lines must still appear (warn plus debug equals 20).
- Two wire-tensor message assertions now read "between 1 and N".

### 4.2 Local gate

GB10 spark-102 at 4a18642b, `--profile test-fast`, under `gpu-lock run --tag issue-2726` with `--test-threads=1`, `MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1`, binaries built outside the lock:

| Target | Result |
|---|---|
| `cargo fmt --all -- --check`; `cargo clippy --profile test-fast --features cuda,test-utils -p mlxcel --lib --tests -- -D warnings` | clean |
| `--lib distributed::` (`cuda,test-utils`) | 1552 passed, 4 ignored |
| `distributed_integration`, `pipeline_e2e` | 20, 30 passed |
| `pipeline_remote_real_models`, `pipeline_server_remote_real_models`, `pipeline_ci_multi_stage_real_models` (`--include-ignored --skip glm_moe_dsa`, no model skips) | 10, 7, 4 passed |
| `zero_config_cluster_init`, `disaggregated_serving_e2e` (`--include-ignored`) | 7, 2 passed |
| `disaggregated_router_e2e` (`--include-ignored`) | 7 passed |

The real-model targets are what prove item 1 and item 2 do not break honest coordinators: they prepare, run and release sequences under the new id format, against stages that now refuse unowned commands.

The first `disaggregated_router_e2e` run had one failure, `disaggregated_router_rejects_model_owned_family_and_keeps_serving` (a 503 on the first request, load average near 50). A reproduction with logs traced it to the router's health monitor: it probes 3 s after start, prefill was not yet listening, so it was marked down until the next tick, while the test sends as soon as prefill's TCP port opens. No pool or retry code is on that path. The same build passed the test 3 of 3 times at lower load and the full target passed 7/7. The race predates this PR and is not fixed here.

Not run: `glm_moe_dsa` (the 418 GB checkpoint does not fit this box), and the Thunderbolt and RDMA variants. No performance measurement was taken: the hot path gains one `HashMap::contains_key` per `RunEntry` and per activation, and each command frame carries 17 more id bytes.

### 4.3 After the merge of main

c46934f4 merges origin/main (060b5051) into the branch. The gate above predates it. At the time of writing a second local gate was running on the merged head and hosted CI was partly pending, so neither result is recorded here.

## 5. Remaining Gaps and Risks

- **The cap can be filled by a secret holder (LOW, availability).** `MAX_STAGE_SEQUENCES` is shared across identities, and owned entries only leave on release, finish or shutdown. A peer that passes the coordinator matcher can prepare 4096 ids and hold them, after which honest prepares on that stage fail cleanly per request. Per-identity quotas or idle expiry of owned sequences would close it.
- **The cap is a constant.** 4096 is far above a coordinator's scheduler cache pool today, but several coordinators sharing one stage, or a much larger cache pool, would hit it with no knob to raise it. The issue chose "not configurable" deliberately.
- **`rpc_call` cannot carry replies over 1 MiB.** Fine for the latency benchmark; a future caller with larger replies needs a new path or a parameter.
- **Shared warning budget.** The server's capacity line and the client's at-capacity line share `HandshakeWarnClass::AtCapacity`, so a busy client side can push the server's line to debug in the same process. The per-minute summary still reports suppressed counts.
- **Public type change.** `DimBound::Within` and `ShapeBound::Uniform::min_dim` break exhaustive matches and struct literals in external crates, if any exist.
- **Pre-existing router health race.** The `disaggregated_router_e2e` 503 at high load (4.2) is a test-timing race in the router health monitor, unrelated to this change, and has no issue of its own yet.
- **Not run.** `glm_moe_dsa`, Thunderbolt and RDMA variants; the merged head's gate and hosted CI were pending at the time of writing.

## 6. Learning Points

### 6.1 Bound state by refusing ownerless entries, then cap the owners

When several maps grow per key, capping each one invites inconsistency and eviction bugs. If every entry can be made to require an owner record first, one cap on the owner map bounds them all, and refusing new keys at the cap keeps every admitted key valid.

```rust
if !self.sequence_owners.contains_key(&request_id)
    && self.sequence_owners.len() >= self.max_sequences
{
    // refuse; nothing changes
}
```

### 6.2 A count cap is a memory cap only if entry size is bounded

The review fix is a general rule: a limit on the number of stored entries says nothing about memory when each entry's key is attacker-sized. Validate the key against the same constructor the honest client uses, and do it before anything is stored or echoed.

### 6.3 Unpredictable identifiers close pre-claim races

Ownership that is granted on first claim is only as strong as the unpredictability of the name being claimed. A sequential id plus first-claim ownership is a TOCTOU on a predictable name; a CSPRNG suffix removes the race while keeping the human-readable prefix for correlation.

### 6.4 Read caps should match the expected reply, not the transport maximum

A transport-wide frame limit exists for the largest legitimate frame (here, tensors). A request-response path whose replies are tiny should read under a reply-sized cap, because any allocation made before authentication is an allocation an attacker controls.

### 6.5 Equal jitter for backoff

Uniform jitter over `[delay/2, delay]` desynchronizes clients that were refused together while preserving a minimum wait, unlike full jitter (`[0, delay]`), which can retry immediately. Draw the value once per attempt so logs report what was slept.

## 7. References

- Issue #2726: bound remote pipeline stage state, RPC replies and tensor shapes
- PR #2723 (issue #2714): per-sequence ownership and fallible prepare; its "pre-claimed ids" and "sequence_owners has no cap" gaps are closed here
- PR #2721 (issue #2716): stage tensor decode limits; item 4 adds the lower bound
- PR #2720 (issue #2712): transport robustness follow-ups (capacity retry with backoff, `rpc_call` response on the same connection); item 5 jitters that backoff
- `docs/distributed.md`, "What the secret protects" and "Residual risks"
