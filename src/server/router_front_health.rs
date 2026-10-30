// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Peer liveness probing for the disaggregated router front (issues #201,
//! #2763).
//!
//! Two callers share one per-node probe ([`probe_node`]): the background
//! [`spawn_health_monitor`] loop, which runs its first cycle as soon as it
//! starts and then every [`HEALTH_PROBE_INTERVAL`], and the admission check,
//! which re-probes the prefill nodes marked down before it refuses a request
//! with HTTP 503 ([`reprobe_unreachable_prefill`]). Sharing the connect and
//! the status transition keeps the two paths from drifting apart.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::distributed::config::NodeRole;
use crate::distributed::registry::{NodeStatus, RegisteredNode};

use super::RouterState;

/// How often the background health monitor probes each registered serving peer
/// for liveness (issue #201). The first cycle runs when the monitor starts.
pub(super) const HEALTH_PROBE_INTERVAL: Duration = Duration::from_secs(3);

/// Per-probe TCP connect timeout for the health monitor. A peer that does not
/// accept a connection within this window is treated as unreachable.
pub(super) const HEALTH_PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// Minimum gap between two on-demand re-probes, router-wide (issue #2763). A
/// re-probe starts only when the previous one finished at least this long ago;
/// requests inside the gap reuse the registry state the last one left.
pub(super) const ON_DEMAND_PROBE_MIN_INTERVAL: Duration = Duration::from_millis(250);

/// Per-node TCP connect timeout for the on-demand re-probe. Shorter than
/// [`HEALTH_PROBE_TIMEOUT`] because a client request waits on it.
pub(super) const ON_DEMAND_PROBE_TIMEOUT: Duration = Duration::from_millis(500);

/// What started a probe, recorded on its status-transition log lines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbeTrigger {
    /// The background [`spawn_health_monitor`] cycle.
    Monitor,
    /// The admission check, before refusing a request with 503.
    Admission,
}

impl ProbeTrigger {
    fn as_str(self) -> &'static str {
        match self {
            Self::Monitor => "monitor",
            Self::Admission => "admission",
        }
    }
}

/// Router-wide probe state held in [`RouterState`].
#[derive(Debug, Default)]
pub(crate) struct PeerProbe {
    /// Single-flight gate for the on-demand re-probe. Holding the lock means a
    /// re-probe is running; the value is when the last one ran. Requests that
    /// arrive meanwhile wait for it and then reuse its result. No registry or
    /// request-map lock is ever taken while this is held across a connect.
    on_demand: tokio::sync::Mutex<Option<Instant>>,
    /// TCP connect attempts made by [`probe_node`], from the monitor and the
    /// on-demand path alike. Diagnostics and the test hook that bounds the
    /// on-demand re-probe rate.
    connect_attempts: AtomicU64,
}

impl PeerProbe {
    /// Number of peer connect attempts made so far (test hook).
    #[cfg(test)]
    #[must_use]
    pub(crate) fn connect_attempts(&self) -> u64 {
        self.connect_attempts.load(Ordering::Relaxed)
    }
}

/// Probe one node and apply the resulting status transition, returning whether
/// it accepted a connection within `timeout`.
///
/// `node` is a registry snapshot; its `status` decides the transition, as in
/// the original monitor loop. An online node that refuses is marked
/// [`NodeStatus::Unreachable`] and its in-flight requests are re-routed via
/// [`crate::distributed::disaggregated::RequestRouter::handle_node_failure`];
/// an unreachable node that accepts is restored to [`NodeStatus::Online`].
/// The connect runs with no registry or request-map lock held.
async fn probe_node(
    state: &RouterState,
    node: &RegisteredNode,
    timeout: Duration,
    trigger: ProbeTrigger,
) -> bool {
    state
        .peer_probe
        .connect_attempts
        .fetch_add(1, Ordering::Relaxed);
    let addr = node.config.address;
    let alive = matches!(
        tokio::time::timeout(timeout, tokio::net::TcpStream::connect(addr)).await,
        Ok(Ok(_))
    );
    match (alive, node.status) {
        (false, NodeStatus::Online) => {
            state
                .registry
                .set_node_status(&node.config.id, NodeStatus::Unreachable);
            let (rerouted, failed) = state.router.handle_node_failure(&node.config.id);
            tracing::warn!(
                node = %node.config.id, %addr, rerouted, failed, trigger = trigger.as_str(),
                "router health: peer unreachable; marked it down"
            );
        }
        (true, NodeStatus::Unreachable) => {
            state
                .registry
                .set_node_status(&node.config.id, NodeStatus::Online);
            tracing::info!(
                node = %node.config.id, %addr, trigger = trigger.as_str(),
                "router health: peer recovered; marked it online"
            );
        }
        _ => {}
    }
    alive
}

/// Spawn a background task that probes every registered serving peer for
/// liveness and keeps the [`crate::distributed::registry::NodeRegistry`]
/// status in sync (issues #201, #2763).
///
/// The first cycle runs immediately, so a router whose peers are already
/// listening never marks them down, and the task sleeps
/// [`HEALTH_PROBE_INTERVAL`] after each cycle. Each cycle TCP-connects to each
/// node's serving address (the same address the router routes to) through
/// `probe_node`. A node that does not accept a connection within
/// [`HEALTH_PROBE_TIMEOUT`] is marked [`NodeStatus::Unreachable`], so
/// `route_to_prefill` / `route_to_decode` skip it, and its in-flight requests
/// are re-routed. A previously unreachable node that starts accepting again is
/// restored to [`NodeStatus::Online`]. This catches the cases the per-send
/// transport-error path cannot: a decode node (which the router never sends to
/// directly) dying, and a node recovering. The task also purges terminal
/// tracked requests so the router's request map stays bounded.
///
/// Locking: `all_nodes()` returns a snapshot and releases the registry lock; no
/// registry or request-map lock is held across the connect `await`.
pub fn spawn_health_monitor(state: Arc<RouterState>) {
    tokio::spawn(async move {
        loop {
            run_probe_cycle(&state).await;
            tokio::time::sleep(HEALTH_PROBE_INTERVAL).await;
        }
    });
}

/// One monitor cycle: probe every registered node, then purge terminal tracked
/// requests older than the router config's auto-purge age.
async fn run_probe_cycle(state: &RouterState) {
    for node in state.registry.all_nodes() {
        probe_node(state, &node, HEALTH_PROBE_TIMEOUT, ProbeTrigger::Monitor).await;
    }
    let purged = state
        .router
        .purge_terminal(state.router.config().auto_purge_age);
    if purged > 0 {
        tracing::debug!(purged, "router health: purged terminal tracked requests");
    }
}

/// The prefill nodes the registry currently marks [`NodeStatus::Unreachable`].
fn unreachable_prefill_nodes(state: &RouterState) -> Vec<RegisteredNode> {
    state
        .registry
        .all_nodes()
        .into_iter()
        .filter(|node| {
            node.config.role == NodeRole::Prefill && node.status == NodeStatus::Unreachable
        })
        .collect()
}

/// Whether any prefill node is marked [`NodeStatus::Unreachable`], i.e. whether
/// an on-demand re-probe could change the admission verdict.
pub(super) fn has_unreachable_prefill(state: &RouterState) -> bool {
    !unreachable_prefill_nodes(state).is_empty()
}

/// Re-probe the prefill nodes marked down and restore those that accept, so the
/// caller can re-evaluate admission instead of refusing with 503 (issue #2763).
///
/// Bounded router-wide: single-flight, and a new re-probe starts only when the
/// last one finished at least [`ON_DEMAND_PROBE_MIN_INTERVAL`] ago. A caller
/// that arrives while one runs waits for it and then reuses its result, as
/// does one inside the gap after it. Each node gets one connect attempt with
/// [`ON_DEMAND_PROBE_TIMEOUT`], all nodes in parallel, so a re-probe costs a
/// request at most that timeout.
pub(super) async fn reprobe_unreachable_prefill(state: &RouterState) {
    let mut last = state.peer_probe.on_demand.lock().await;
    if last.is_some_and(|ran| ran.elapsed() < ON_DEMAND_PROBE_MIN_INTERVAL) {
        return;
    }
    // Stamp the start too, so a caller cancelled mid-probe (client gone) still
    // spaces the next re-probe from this one.
    *last = Some(Instant::now());
    let down = unreachable_prefill_nodes(state);
    futures::future::join_all(down.iter().map(|node| {
        probe_node(
            state,
            node,
            ON_DEMAND_PROBE_TIMEOUT,
            ProbeTrigger::Admission,
        )
    }))
    .await;
    *last = Some(Instant::now());
}
