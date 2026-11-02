# PR #2772: Re-probe down prefill peers before answering 503

**Date**: 2026-10-31 (written before merge; the PR is open as a draft)
**Status**: Implementation complete at 7726ef31 on top of base bebdd62c, then merged with origin/main at 57b9f9e5 (brings in #2769 at ab7b8509, which touches none of this PR's code). The local gate in the PR body passed on GB10 (spark-102) at 7726ef31. At the time of writing a second local gate was running in the worktree and hosted CI was partly pending (including `cargo-fmt`, `cargo-clippy`, `cargo-deny`, `CPU-only link`, `OpenXLA feature compile`, `python tooling tests`, `WebUI bundle`, `WebUI installed artifact`)
**Languages**: Rust, Markdown
**Risk Level**: Low (no wire, API or flag change; `/health` and the 503 text are unchanged. A refusal can now wait up to 500 ms for a re-probe when a prefill host silently drops connects)

## Executive Summary

PR #2772 (closes #2763) stops the disaggregated router from answering HTTP 503 while a prefill peer is already listening. Before it, the health monitor slept 3 s before its first probe and admission never re-checked a node the monitor had marked down, so a peer that bound its port just after a probe stayed refused for up to one 3 s cycle. The monitor now probes at startup, and `admission_reject` re-probes the prefill-capable nodes marked `Unreachable` before turning a `Queue` verdict into a 503. The re-probe is single-flight, starts at most once per 250 ms router-wide, gives each node one 500 ms connect, and shares its per-node probe and status transition (`probe_node` in the new `src/server/router_front_health.rs`) with the monitor.

## 1. Problem Statement

### 1.1 Background

During #2746 and #2747, `disaggregated_router_rejects_model_owned_family_and_keeps_serving` flaked: a request right after startup got "all serving nodes are at capacity or unreachable; retry shortly" although every peer was up and the router's `/health` answered 200. #2747 fixed the tests by making every router e2e test wait on `GET /router/stats` until all peers report `online` (`wait_for_router_peers_online`). Real clients get no such wait, and the #2747 review and report (section 4, "Trade-offs and Residuals": "The router's startup 503 window remains") left the router behavior as a follow-up, which became #2763.

### 1.2 The monitor's first probe came one interval late

`spawn_health_monitor` ran `sleep(HEALTH_PROBE_INTERVAL)` (3 s) at the top of its loop, so the first probe came 3 s after start, and each probe used a 2 s connect timeout. A peer still loading its model at that moment was marked `NodeStatus::Unreachable` and stayed down until the next cycle. The same window reopened after any later recovery: a peer that came back just after a cycle waited up to 3 s for the next one.

### 1.3 Admission trusted a stale verdict

`admission_reject` called `RequestRouter::apply_backpressure`, which returns `Queue` when no prefill node is online below `LoadLevel::Critical`. The router has no async queue, so `Queue` became an immediate 503. Nothing on that path re-checked the nodes the registry marked down, so the request was refused on the monitor's last observation, which could be up to 3 s old.

### 1.4 Risk of leaving it

| Risk | Impact | Likelihood |
|---|---|---|
| Clients of a freshly started or recovering router get 503 for up to about 3 s although the cluster can serve | Medium (clients that do not retry see a failed request; orchestrators that gate on `/health` see green and send traffic) | High on every rolling restart where the router starts before or alongside its peers |
| Test suites keep needing a router-specific readiness poll | Low | Certain |

## 2. Change Summary

| Area | File | Change |
|---|---|---|
| Probe module | `src/server/router_front_health.rs` (new, 242 lines) | `HEALTH_PROBE_INTERVAL` (3 s) and `HEALTH_PROBE_TIMEOUT` (2 s) move here; new `ON_DEMAND_PROBE_MIN_INTERVAL` (250 ms) and `ON_DEMAND_PROBE_TIMEOUT` (500 ms); `ProbeTrigger { Monitor, Admission }` for log tagging; `PeerProbe { on_demand: Arc<tokio::sync::Mutex<Option<Instant>>>, connect_attempts: AtomicU64 }`; `probe_node` (one connect plus the status transition, decided from the registry status re-read after the connect); `spawn_health_monitor` runs `run_probe_cycle` before its first sleep; `has_unreachable_prefill` and `reprobe_unreachable_prefill` |
| Router front | `src/server/router_front.rs` | Loads the module with `#[path = "router_front_health.rs"] mod health;` and re-exports `spawn_health_monitor`; `RouterState` gains `peer_probe: health::PeerProbe`; `admission_reject` becomes `async` and takes `&Arc<RouterState>`, re-probing on `Queue` when a prefill-capable node is `Unreachable`; `route_chat` and `route_completion` await it; module docs updated |
| Lib tests | `src/server/router_front_tests.rs` | 4 new tests: `first_health_probe_runs_at_startup`, `admission_reprobes_a_peer_marked_down_before_refusing` (chat and completions), `admission_reprobe_is_bounded`, `probe_decides_transition_from_current_registry_status` |
| E2E test | `tests/disaggregated_router_e2e.rs` | New `disaggregated_router_started_before_prefill_serves_first_request` and helper `wait_for_router_node_status`; the `wait_for_router_peers_online` doc now says why the wait is kept |
| Docs | `docs/distributed.md`, `docs/CONTINUOUS_BATCHING.md` | Startup probe, the re-probe's trigger and bounds, its cost when a host drops connects, `/health` remains a liveness check. No Korean counterpart of these documents exists |

### Statistics

| Item | Value |
|---|---|
| Files changed | 6 |
| Lines | +646 / -93 |
| New tests | 4 lib, 1 real-process e2e |
| Commits | 2, plus a merge of origin/main |

| Commit | Purpose |
|---|---|
| 0022227a | Startup probe, async admission re-probe, shared `probe_node`, three lib tests, the e2e case, docs |
| 7726ef31 | Review hardening: `probe_node` re-reads the registry status after the connect; the re-probe covers `Hybrid` through `nodes_with_role`; it runs in a spawned task that owns the gate; the 250 ms interval is stamped at completion; docs no longer say every prefill node must be down; `admission_reprobe_is_bounded` checks the spacing; new `probe_decides_transition_from_current_registry_status` |
| 57b9f9e5 | Merge of origin/main (ab7b8509, #2769). Its `docs/distributed.md` change is in the pipeline section; no conflict with this PR |

## 3. Technical Decisions

### 3.1 Probe at startup, sleep after each cycle

The monitor loop changed from `sleep; probe` to `probe; sleep`. A router whose peers are already listening never marks them down, and a router whose peers are not up yet learns that immediately instead of 3 s later (the e2e test relies on this to observe `unreachable` before starting the prefill node). This alone does not remove the window: a peer that binds just after the startup probe still waits a full cycle, which is why 3.2 exists.

### 3.2 Re-probe only on `Queue` with a prefill-capable node down

`admission_reject` re-probes only when `apply_backpressure` returns `Queue` and `has_unreachable_prefill` is true, then calls `apply_backpressure` again and answers 503 only if the second verdict is still `Queue`. The guards keep the healthy and overloaded paths unchanged:

- `Accept` never probes, so a working cluster pays nothing.
- `Reject` (prefill queue at capacity) never probes: a node coming back would not drain the queue.
- `Queue` with no node marked down (every prefill node at `LoadLevel::Critical`) never probes: there is nothing to restore.

The set of nodes re-probed is `Prefill` plus `Hybrid`, the same roles `apply_backpressure` counts through `get_prefill_load_infos`. The first commit covered `Prefill` only; the hardening commit added `Hybrid` so the trigger and the verdict agree. The router front registers no hybrid nodes today, so this is defensive.

| Option | Pros | Cons |
|---|---|---|
| `/health` returns 503 until every peer has been probed once | Orchestrators hold traffic until peers are known | Changes a documented liveness endpoint; still leaves the 3 s window for a peer that comes up after the first cycle or after a later failure |
| Shorten `HEALTH_PROBE_INTERVAL` | One-line change | Shrinks the window without closing it; multiplies background connects to every peer, decode nodes included |
| **Chosen: probe at startup and re-probe on demand before refusing** | Closes the window at the point a client is about to be refused; leaves `/health` and the interval alone | Refusals during a real outage can wait on a probe (see 3.3 and section 5) |

### 3.3 Bounding the on-demand path

Every refused request could otherwise trigger a connect to every down node, so the re-probe is bounded router-wide:

- **Single-flight.** `PeerProbe::on_demand` is a `tokio::sync::Mutex<Option<Instant>>`. Holding it means a re-probe is running. A request that arrives meanwhile waits for the lock, sees a recent stamp and returns, reusing the registry state the running probe left.
- **At most once per 250 ms, measured from completion.** The stamp is written after `join_all` returns. The first commit stamped at start; with a 500 ms timeout that let a new probe start right after a slow one finished. Stamping at completion guarantees 250 ms of cached verdicts between probes regardless of probe duration.
- **One 500 ms connect per down node, in parallel.** A request that triggers a probe waits at most about 500 ms, not 500 ms times the number of down nodes. The monitor keeps its 2 s timeout because no client waits on it.
- **No registry or request-map lock across the connect.** `unreachable_prefill_nodes` takes a snapshot (`nodes_with_role` clones under a read lock and releases it); `probe_node` takes the registry lock only after the connect returns. The tokio gate is the only lock held across an `.await`.

Client volume cannot raise the connect rate: the ceiling is one probe per (probe duration + 250 ms), and connects go only to configured peer addresses.

### 3.4 Run the re-probe in a spawned task that owns the gate

`reprobe_unreachable_prefill` spawns a task that acquires the gate with `lock_owned`, probes, stamps and releases, and the handler awaits its `JoinHandle`. If the client disconnects, axum drops the handler future; dropping a `JoinHandle` detaches the task rather than cancelling it, so the probe finishes and records its result. Without this, a disconnect mid-connect would drop the probe with no status transition and, with a start stamp, leave the next 250 ms of requests refused on no new information. A panicking probe task is logged at warn and treated as "no change"; the second `apply_backpressure` then decides.

### 3.5 One per-node probe, deciding from the current status

The connect and the transition (`Online` to `Unreachable` calls `handle_node_failure` and logs at warn; `Unreachable` to `Online` logs at info) moved out of the monitor loop into `probe_node`, used by both callers. Both log lines carry `trigger = "monitor"` or `"admission"`, so an operator can tell which path restored or failed a node.

The first version matched on `node.status` from the caller's snapshot. The monitor takes one snapshot per cycle and the connect can take 2 s, so if the admission path restored a node and the node then failed before the monitor reached it, the monitor saw `(false, Unreachable)` and left a dead node online for another cycle. `probe_node` now uses the snapshot for id and address only and re-reads the status with `registry.get_node` after the connect; a node removed in the meantime is left alone. `probe_decides_transition_from_current_registry_status` pins this by passing a snapshot that says `Unreachable` for a node the registry has `Online`.

### 3.6 Keep `/health` and the e2e readiness wait

`router_health` still returns `"ok"` unconditionally, as documented: it reports that the router process is alive, not that the cluster can serve. The 503 text is unchanged, so clients that match on it see no difference. The e2e suite keeps `wait_for_router_peers_online`; it is no longer needed to avoid a 503, but a peer that never starts listening now fails there with a named message instead of inside an unrelated request assertion.

## 4. Verification

### 4.1 Failing-first evidence

On main bebdd62c with the tests added and the router code unmodified:

| Test | On main | After |
|---|---|---|
| `admission_reprobes_a_peer_marked_down_before_refusing` | `/v1/chat/completions: refused although the peer is listening: all serving nodes are at capacity or unreachable; retry shortly` | 200 on chat and completions; node `Online` afterwards |
| `first_health_probe_runs_at_startup` | `the first health probe did not mark the closed peer down within 1 s of the monitor starting (status Some(Online))` | marked `Unreachable` within 1 s |
| Rest of `server::router_front::` | 43 passed, 2 failed | 47 passed |

`admission_reprobe_is_bounded` needs the `connect_attempts` hook and `probe_decides_transition_from_current_registry_status` needs `health::probe_node`, neither of which exists on main, so neither can run there. The bounded test fires 20 concurrent chat requests at a closed peer, asserts every response is the unchanged 503, and asserts `1 <= attempts <= 1 + elapsed / 250 ms`. It then checks that a request inside the window adds no attempt (asserted only when the measured elapsed time shows the window was still open) and that one after a 400 ms sleep adds exactly one.

### 4.2 Local gate (from the PR body, at 7726ef31)

GB10 (spark-102), `--profile test-fast --features cuda`, `--test-threads=1`; e2e runs under `gpu-lock` with `MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1`.

| Suite | Result |
|---|---|
| `server::router_front::` (lib) | 47 passed; the four new tests passed 10 repeated runs |
| `disaggregated_router_e2e --include-ignored` | 8/8 (7 existing plus the new startup-order case) |
| `disaggregated_serving_e2e --include-ignored` | 2/2 |
| `cargo clippy ... -D warnings` (lib and tests with `cuda`; `disaggregated_router_e2e` with `cuda,test-utils`), `cargo fmt --all -- --check` | clean |

The new e2e case is the manual check from #2763: router and decode node start first, `/router/stats` shows `prefill-0` as `unreachable`, the prefill node starts, and the first `/v1/completions` goes out as soon as `wait_for_tcp` sees its serving port accept. It gets 200, and `/router/stats` then reports the node `online`.

### 4.3 Not run

- The new e2e case was not run on main, so there is no failing-first evidence for it, and it does not prove which path restored the node. The gap between the port accepting and the request is milliseconds, so a monitor cycle landing in it is unlikely, but the test does not distinguish the two; the lib test `admission_reprobes_a_peer_marked_down_before_refusing` is the one that pins the admission path (it runs without the monitor).
- No test covers a host that silently drops SYNs (every test peer refuses or accepts at once), so the 500 ms wait in 3.3 is reasoned from the timeout, not measured.
- No test registers a `Hybrid` node; the `Hybrid` arm of the re-probe is covered by review only.
- Not run on Metal, ROCm or a multi-host deployment.

## 5. Remaining Gaps and Risks

- **A residual window of up to 250 ms.** A request that arrives within 250 ms after a re-probe finished reuses its verdict. If a peer starts accepting inside that window, those requests still get 503. This replaces a window of up to 3 s.
- **Refusals can be slower during a real outage.** Before this PR a 503 for "no prefill available" was immediate. Now, when a down prefill host refuses connects, the re-probe returns at once and nothing changes in practice; when it silently drops packets, the request that starts the re-probe and every request queued behind the gate wait up to 500 ms, then the next 250 ms of requests are refused immediately. Roughly two thirds of refusals in that state carry up to 500 ms of added latency. The docs state this cost.
- **Load shedding can wait on an unrelated down node.** If the online prefill nodes are at `LoadLevel::Critical` and some other prefill node is `Unreachable`, the `Queue` verdict still triggers a re-probe, so overload 503s inherit the latency above. This matches the issue's specification and was raised in review as MEDIUM; it was not changed.
- **The gate serializes waiters.** Each request in the `Queue`-with-a-node-down state spawns a task and takes the tokio mutex in turn. Each waiter only reads the stamp and returns, so the cost is small, but it is a router-wide serialization point under a burst of refused traffic.
- **Two registry clones per refused request.** `has_unreachable_prefill` collects the down nodes to test emptiness, and the spawned task collects them again. Registries are small (peer counts, not request counts); left as LOW.
- **Liveness is still "the port accepts".** The re-probe uses the same TCP-connect test as the monitor. A peer that binds its serving port before it can serve would be restored early by either path. This PR does not change that semantic.
- **`closed_loopback_addr` releases the port before the test connects**, so another process could bind it in between. The window is microseconds and repeated runs passed.

## 6. Learning Points

### 6.1 Re-check a cached health verdict at the point it would refuse work

A background monitor's view of peer health is a cache with a TTL equal to its interval. Shortening the interval only shortens the TTL. Re-validating the cache at the one moment a stale entry causes harm, here just before a 503, closes the window without adding background load, as long as the re-validation is bounded so that the refusal path cannot be turned into a connect amplifier.

### 6.2 Stamp a rate limit at completion when the work has variable duration

A "once per N ms" gate stamped at start lets a slow operation be followed immediately by another. Stamping at completion makes N the guaranteed quiet gap between operations, which is what bounds the load on the target. `admission_reprobe_is_bounded` checks both sides of that gap.

### 6.3 Decouple shared work from the caller's cancellation

When one caller does work on behalf of many (single-flight), the work should not die with that caller. Spawning it in a task that owns the gate, and having the caller await the `JoinHandle`, means a client disconnect cannot leave the shared state half updated. Dropping a tokio `JoinHandle` detaches the task; it does not abort it.

### 6.4 Decide a transition from current state, not from the snapshot that scheduled the check

A snapshot is fine for choosing what to probe (id and address do not change), but the transition must be decided from the state at the moment it is applied, because a slow operation in between can be raced by another writer. Re-reading after the `await` costs one map lookup.

## 7. References

- Issue #2763: Stop 503s while a just-started peer is still marked down
- PR #2747 and its report (`TECHNICAL_REPORTS/2747-router-reasoning-only-eos-parity-20261030.en.md`, section 4 "Trade-offs and Residuals"): the test-side `wait_for_router_peers_online` fix and the recorded follow-up
- Issue #201: Disaggregated serving: multi-node routing, load balancing, and failover (introduced the health monitor and admission control)
- `src/distributed/disaggregated/request_router.rs`: `RequestRouter::apply_backpressure`
- `src/distributed/registry.rs`: `NodeRegistry::nodes_with_role`, `get_node`, `set_node_status`
