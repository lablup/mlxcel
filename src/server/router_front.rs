// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Model-free HTTP router front-end for disaggregated serving (#126 B3b2b).
//!
//! The router loads a tokenizer and chat template but never loads model
//! weights. Incoming `/v1/chat/completions` and `/v1/completions` requests are
//! tokenized, forwarded to a prefill node via [`PrefillRequestFrame`], and the
//! two-part result (prefill first token + decode continuation) is merged and
//! returned to the client as a streaming SSE response or a single JSON object.
//! Both endpoints share the [`start_handoff`] dispatch body (issue #200); they
//! differ only in request parsing (chat template vs raw `prompt`) and the
//! response chunk shape (chat-completion chunk vs text-completion chunk). The
//! text-completion path reuses the single-node [`CompletionResponse`] /
//! [`CompletionChunk`] serializers so its output is byte-identical to
//! single-node `/v1/completions` for the same prompt and sampling.
//!
//! # Architecture
//!
//! ```text
//! Client ---> RouterState ---> prefill node (PrefillRequestFrame)
//!                          <-- ResultFrame{FirstToken}
//!                          <-- ResultFrame{Continuation}
//!   <-- SSE or JSON ------
//! ```
//!
//! The router binds its own [`TcpTransport`] at `--serving-bind` so that both
//! the prefill and decode nodes can return their [`ResultFrame`]s to it. A
//! background demux task (`spawn_result_demux`) routes each incoming frame to
//! the per-request channel keyed by `request_id`.
//!
//! # Output filtering (issue #198)
//!
//! Every decode text piece (and the prefill first token) passes through the
//! same [`StreamFilter`] the single-node chat route uses, so model-specific
//! structural markers (`<think>`, `<|channel>`, tool-call delimiters, stray
//! turn tokens) never leak to the client and thinking content is routed to
//! `delta.reasoning_content`. Tool-call parsing (accumulate-then-parse into
//! `tool_calls`) is NOT yet supported on the router path: the router emits
//! `content` and `reasoning_content` only, and the filter suppresses
//! tool-call delimiter markers.
//!
//! # Multi-node routing, health, and backpressure (issue #201)
//!
//! The router balances BOTH pools. It selects the prefill node with
//! `route_to_prefill` and the decode node with `route_to_decode`, then ships the
//! chosen decode node in [`PrefillRequestFrame`]'s `decode_target` so the prefill
//! node hands the KV cache to the router-balanced decode node rather than its own
//! static `--decode-peers` config (a frame without the field, from an older
//! router, leaves the prefill node on its config fallback). Both pools use a
//! round-robin strategy because the router has no live per-node load telemetry.
//!
//! Health and failover: a transport error when sending a prefill frame marks the
//! node unreachable, re-routes its in-flight requests via `handle_node_failure`,
//! and retries the request on a healthy node. A background [`spawn_health_monitor`]
//! task probes every peer's liveness (TCP connect) so a dead decode node (which
//! the router never sends to directly) is detected and skipped, and a recovered
//! node is restored to online.
//!
//! Backpressure: every request first passes `apply_backpressure` admission
//! control; when the prefill queue is full or no prefill node is available, the
//! router returns HTTP 503 instead of dispatching.

use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::distributed::backpressure::{BackpressureConfig, BackpressureMonitor};
use crate::distributed::config::{ClusterConfig, ClusterMeta, NodeConfig, NodeResources, NodeRole};
use crate::distributed::disaggregated::{
    BackpressureAction, DisaggRoutingStrategy, PrefillRequestFrame, RequestRouter, ResultFrame,
    RouterConfig, StreamBridge, TokenEvent, TokenSource, control_parts, sampling_to_serializable,
};
use crate::distributed::metrics::ClusterMetrics;
use crate::distributed::registry::{NodeRegistry, NodeStatus};
use crate::distributed::request_tracker::RequestId;
use crate::distributed::tcp_transport::TcpTransport;
use crate::distributed::transport::{Transport, TransportBackend};
use crate::server::ChatTemplateProcessor;
use crate::server::config::{ServerConfig, ServerGenerateOptions};
use crate::server::tool_calls::stream_filter::{FilterOutput, StreamFilter};
use crate::server::types::request::ChatCompletionRequest;
use crate::server::types::{CompletionChunk, CompletionRequest, CompletionResponse, ErrorResponse};
use crate::tokenizer::MlxcelTokenizer;

/// Timeout for waiting for the prefill first-token result and the decode
/// continuation from the serving nodes.
const HANDOFF_TIMEOUT_SECS: u64 = 120;

/// How often the background health monitor probes each registered serving peer
/// for liveness (issue #201).
const HEALTH_PROBE_INTERVAL: Duration = Duration::from_secs(3);

/// Per-probe TCP connect timeout for the health monitor. A peer that does not
/// accept a connection within this window is treated as unreachable.
const HEALTH_PROBE_TIMEOUT: Duration = Duration::from_secs(2);

// ── Router state ─────────────────────────────────────────────────────────

/// Shared state for the disaggregated router front-end.
///
/// Injected into axum handlers via `axum::extract::State<Arc<RouterState>>`.
pub struct RouterState {
    /// Transport bound to the router's own `--serving-bind` address; receives
    /// [`ResultFrame`]s from the prefill and decode nodes.
    pub transport: Arc<TcpTransport>,

    /// Demux map: `request_id -> per-request result channel sender`.
    /// Keyed by the same `u64` that goes into [`PrefillRequestFrame::request_id`].
    pub pending: Mutex<HashMap<u64, UnboundedSender<ResultFrame>>>,

    /// Monotonically increasing counter used to assign unique request ids.
    pub next_id: AtomicU64,

    /// The router's own serving address (`host:port`), sent to prefill nodes as
    /// the `reply_to` in every [`PrefillRequestFrame`].
    pub reply_to: String,

    /// Request router used to select the prefill node for each request.
    pub router: RequestRouter,

    /// Registry of known prefill and decode nodes.
    pub registry: NodeRegistry,

    /// Prefill node addresses used as a fallback when the router returns an
    /// error (e.g. because all nodes appear loaded).
    pub prefill_fallback: Vec<SocketAddr>,

    /// Per-node count of dispatched requests, keyed by prefill node id
    /// (`prefill-<i>`). Exposed via `GET /router/stats` so an operator (and the
    /// multi-node E2E) can see the load spread across the prefill pool.
    pub prefill_hits: Mutex<HashMap<String, u64>>,

    /// Per-node count of dispatched requests, keyed by decode node id
    /// (`decode-<i>`). Exposed via `GET /router/stats`.
    pub decode_hits: Mutex<HashMap<String, u64>>,

    /// Chat template processor for rendering the prompt.
    pub chat_template: Arc<ChatTemplateProcessor>,

    /// Tokenizer matching the served model.
    pub tokenizer: Arc<MlxcelTokenizer>,

    /// Server configuration (used for default sampling params etc.).
    pub config: Arc<ServerConfig>,

    /// Maximum time to wait for each half of the disaggregated response.
    pub handoff_timeout: Duration,
}

impl RouterState {
    /// Build a [`RouterState`] from resolved startup components.
    ///
    /// Registers every address from `config.prefill_peers` and
    /// `config.decode_peers` in the [`NodeRegistry`] as `Online` so
    /// `route_to_prefill` can find them.
    pub fn build(
        config: Arc<ServerConfig>,
        transport: Arc<TcpTransport>,
        reply_to: String,
        chat_template: Arc<ChatTemplateProcessor>,
        tokenizer: Arc<MlxcelTokenizer>,
    ) -> Result<Self> {
        // Build a registry of ONLY the prefill and decode peers the router can
        // route to. The router itself is deliberately NOT registered: it is a
        // front-end, not an inference worker, so registering it (as Hybrid)
        // would make `route_to_prefill` eligible to select the router's own
        // address and route a request back to itself.
        let mut nodes = Vec::with_capacity(config.prefill_peers.len() + config.decode_peers.len());
        for (i, addr) in config.prefill_peers.iter().enumerate() {
            nodes.push(NodeConfig {
                id: format!("prefill-{i}"),
                address: *addr,
                role: NodeRole::Prefill,
                stage: None,
                rank: None,
                resources: NodeResources::default(),
            });
        }
        for (i, addr) in config.decode_peers.iter().enumerate() {
            nodes.push(NodeConfig {
                id: format!("decode-{i}"),
                address: *addr,
                role: NodeRole::Decode,
                stage: None,
                rank: None,
                resources: NodeResources::default(),
            });
        }
        let cluster = ClusterConfig {
            cluster: ClusterMeta {
                name: "mlxcel-router".to_string(),
                tensor_parallel_size: 1,
                pipeline_parallel_size: 1,
                transport_backend: TransportBackend::default(),
            },
            nodes,
        };
        // `local_node_id` ("router") is intentionally absent from the node list,
        // so `from_config` marks every real peer `Joining`; flip them to `Online`
        // so routing can select them immediately (the router trusts its
        // `--prefill-peers` / `--decode-peers` config rather than probing).
        let registry = NodeRegistry::from_config(&cluster, "router");
        for node in &cluster.nodes {
            registry.set_node_status(&node.id, NodeStatus::Online);
        }

        // The router has no live per-node load or memory telemetry from the
        // worker pools, so the load-aware strategies (least-loaded /
        // memory-aware) would all collapse to "pick the first node" and never
        // balance. Round-robin is the strategy that actually spreads requests
        // across both pools with only the registry's online/offline view, which
        // is exactly what the router has. It picks both the prefill node and the
        // router-chosen decode node (issue #201).
        let router_config = RouterConfig {
            prefill_strategy: DisaggRoutingStrategy::RoundRobin,
            decode_strategy: DisaggRoutingStrategy::RoundRobin,
            ..RouterConfig::default()
        };
        let request_router = RequestRouter::new(
            router_config,
            registry.clone(),
            ClusterMetrics::new(),
            BackpressureMonitor::new(BackpressureConfig::default()),
        );

        let prefill_fallback = config.prefill_peers.clone();

        Ok(Self {
            transport,
            pending: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            reply_to,
            router: request_router,
            registry,
            prefill_fallback,
            prefill_hits: Mutex::new(HashMap::new()),
            decode_hits: Mutex::new(HashMap::new()),
            chat_template,
            tokenizer,
            config,
            handoff_timeout: Duration::from_secs(HANDOFF_TIMEOUT_SECS),
        })
    }

    /// Select a prefill node for a request, returning both its registry id (when
    /// the router picked one) and the address to send the frame to.
    ///
    /// Tracks the request in the [`RequestRouter`] (so decode routing and
    /// failover can find it) and applies the configured load-balancing strategy.
    /// `allow_fallback` controls the degenerate path: on the first attempt it is
    /// `true`, so a router with no usable registry entry still dispatches to the
    /// first configured `--prefill-peers` address (`node_id` is then `None`,
    /// since no registry node backs it). On a failover retry it is `false`: the
    /// caller wants a real, routed, healthy node and treats "no node" as a clean
    /// failure rather than re-sending to a possibly-dead static address.
    fn select_prefill(
        &self,
        rid: &RequestId,
        prompt_len: usize,
        allow_fallback: bool,
    ) -> Result<PrefillTarget> {
        match self.router.route_to_prefill(rid.clone(), prompt_len) {
            Ok(node_id) => self
                .registry
                .get_node(&node_id)
                .map(|n| PrefillTarget {
                    node_id: Some(node_id.clone()),
                    addr: n.config.address.to_string(),
                })
                .ok_or_else(|| anyhow::anyhow!("routed prefill node {node_id} not in registry")),
            Err(_) if allow_fallback => self
                .prefill_fallback
                .first()
                .map(|a| PrefillTarget {
                    node_id: None,
                    addr: a.to_string(),
                })
                .ok_or_else(|| anyhow::anyhow!("no prefill node available")),
            Err(e) => Err(anyhow::anyhow!("no healthy prefill node: {e}")),
        }
    }

    /// Select a decode node for an already-tracked request (router-driven decode
    /// selection, issue #201). Returns the decode node's registry id and
    /// address, or `None` when no decode node could be routed (no decode peers
    /// registered, all unreachable, or the request is not tracked because the
    /// prefill selection took the static fallback). A `None` here makes the
    /// router omit `decode_target` so the prefill node falls back to its own
    /// `--decode-peers` config.
    fn select_decode(&self, rid: &RequestId) -> Option<DecodeTarget> {
        match self.router.route_to_decode(rid) {
            Ok(node_id) => self.registry.get_node(&node_id).map(|n| DecodeTarget {
                node_id,
                addr: n.config.address.to_string(),
            }),
            Err(_) => None,
        }
    }

    /// Move a tracked request to a terminal phase so the router does not leak it.
    ///
    /// Derives the router's string request id from the response id (the same
    /// derivation [`start_handoff`] uses) and marks the request completed or
    /// failed. Terminal entries are purged by [`spawn_health_monitor`]; without
    /// this the `Prefilling` / `Decoding` entry would live forever. A no-op when
    /// the id cannot be reconstructed or the request was never tracked.
    fn finalize_request(&self, response_id: &str, completed: bool) {
        let Some(rid) = RequestId::from_string(response_id.to_string()) else {
            return;
        };
        if completed {
            let _ = self.router.mark_completed(&rid);
        } else {
            let _ = self.router.mark_failed(&rid, "router request failed");
        }
    }
}

/// A selected prefill node: its registry id (`None` when the static fallback
/// address was used) and the address to send the [`PrefillRequestFrame`] to.
struct PrefillTarget {
    node_id: Option<String>,
    addr: String,
}

/// A selected decode node: its registry id and the address the router puts in
/// [`PrefillRequestFrame::decode_target`] for the prefill node to hand off to.
struct DecodeTarget {
    node_id: String,
    addr: String,
}

// ── Background demux task ────────────────────────────────────────────────

/// Spawn a background task that forwards incoming [`ResultFrame`]s to the
/// per-request channels registered in `state.pending`.
///
/// The task exits when the transport's receive loop returns an error (e.g.
/// on shutdown).
pub fn spawn_result_demux(state: Arc<RouterState>) {
    tokio::spawn(async move {
        while let Ok((from, msg)) = state.transport.recv().await {
            tracing::debug!(%from, "router demux: received a transport frame");
            if let Ok((op, payload)) = control_parts(msg)
                && op == ResultFrame::OPERATION
                && let Ok(frame) = ResultFrame::decode(&payload)
            {
                let request_id = frame.request_id;
                let delivered = state
                    .pending
                    .lock()
                    .unwrap()
                    .get(&request_id)
                    .map(|tx| tx.send(frame).is_ok())
                    .unwrap_or(false);
                tracing::debug!(request_id, delivered, "router demux: routed a result frame");
            }
        }
    });
}

// ── Background health monitor ────────────────────────────────────────────

/// Spawn a background task that probes every registered serving peer for
/// liveness and keeps the [`NodeRegistry`] status in sync (issue #201).
///
/// Each cycle TCP-connects to each node's serving address (the same address the
/// router routes to). A node that does not accept a connection within
/// [`HEALTH_PROBE_TIMEOUT`] is marked [`NodeStatus::Unreachable`], so
/// `route_to_prefill` / `route_to_decode` skip it, and its in-flight requests
/// are re-routed via [`RequestRouter::handle_node_failure`]. A previously
/// unreachable node that starts accepting again is restored to
/// [`NodeStatus::Online`]. This catches the cases the per-send transport-error
/// path cannot: a decode node (which the router never sends to directly) dying,
/// and a node recovering. The task also purges terminal tracked requests so the
/// router's request map stays bounded.
///
/// Locking: `all_nodes()` returns a snapshot and releases the registry lock; no
/// registry or request-map lock is held across the connect `await`.
pub fn spawn_health_monitor(state: Arc<RouterState>) {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(HEALTH_PROBE_INTERVAL).await;
            for node in state.registry.all_nodes() {
                let addr = node.config.address.to_string();
                let alive = matches!(
                    tokio::time::timeout(
                        HEALTH_PROBE_TIMEOUT,
                        tokio::net::TcpStream::connect(&addr),
                    )
                    .await,
                    Ok(Ok(_))
                );
                match (alive, node.status) {
                    (false, NodeStatus::Online) => {
                        state
                            .registry
                            .set_node_status(&node.config.id, NodeStatus::Unreachable);
                        let (rerouted, failed) = state.router.handle_node_failure(&node.config.id);
                        tracing::warn!(
                            node = %node.config.id, %addr, rerouted, failed,
                            "router health: peer unreachable; marked it down"
                        );
                    }
                    (true, NodeStatus::Unreachable) => {
                        state
                            .registry
                            .set_node_status(&node.config.id, NodeStatus::Online);
                        tracing::info!(
                            node = %node.config.id, %addr,
                            "router health: peer recovered; marked it online"
                        );
                    }
                    _ => {}
                }
            }
            // Bound the tracked-request map: drop terminal entries older than the
            // router config's auto-purge age.
            let purged = state
                .router
                .purge_terminal(state.router.config().auto_purge_age);
            if purged > 0 {
                tracing::debug!(purged, "router health: purged terminal tracked requests");
            }
        }
    });
}

// ── HTTP handlers ─────────────────────────────────────────────────────────

/// GET /health
async fn router_health() -> &'static str {
    "ok"
}

/// Environment variable that opts the client-facing `GET /router/stats` into the
/// verbose (unredacted) view, including each node's raw `host:port` address
/// (issue #389).
const ROUTER_STATS_VERBOSE_ENV: &str = "MLXCEL_ROUTER_STATS_VERBOSE";

/// A registered node's identity for the `/router/stats` response. Carries the
/// stable router-assigned id and the raw address separately so the response
/// builder can redact the address on the public surface (issue #389).
struct StatsNode {
    id: String,
    role: String,
    status: String,
    address: String,
}

/// Whether the verbose (unredacted) `/router/stats` view is enabled, parsed from
/// the [`ROUTER_STATS_VERBOSE_ENV`] value (pure, for unit testing).
///
/// Truthy values are `1`, `true`, `yes`, `on` (case-insensitive, trimmed); any
/// other value, an empty string, or an unset variable keeps the redacted
/// default.
fn router_stats_verbose_from(raw: Option<&str>) -> bool {
    match raw {
        Some(v) => {
            let v = v.trim();
            v.eq_ignore_ascii_case("1")
                || v.eq_ignore_ascii_case("true")
                || v.eq_ignore_ascii_case("yes")
                || v.eq_ignore_ascii_case("on")
        }
        None => false,
    }
}

/// Read [`ROUTER_STATS_VERBOSE_ENV`] from the process environment.
fn router_stats_verbose() -> bool {
    router_stats_verbose_from(std::env::var(ROUTER_STATS_VERBOSE_ENV).ok().as_deref())
}

/// Build the `/router/stats` JSON body, redacting raw node addresses unless
/// `verbose` is set (issue #389).
///
/// The redacted default reports each node's stable router-assigned id, role, and
/// health, plus the per-node dispatch counts and the [`RouterMetrics`] snapshot,
/// but never the raw `host:port`. That is enough for an operator (and the
/// multi-node E2E) to confirm the load spread and that a failed node is marked
/// unreachable, without disclosing the internal cluster topology to any client
/// that can reach the inference port. The `verbose` view (opt-in via
/// [`ROUTER_STATS_VERBOSE_ENV`]) adds the raw address back for trusted-segment
/// debugging. `addresses_redacted` tells the caller which view it received.
fn router_stats_body(
    nodes: &[StatsNode],
    prefill_hits: &HashMap<String, u64>,
    decode_hits: &HashMap<String, u64>,
    metrics: &crate::distributed::disaggregated::RouterMetrics,
    verbose: bool,
) -> serde_json::Value {
    let nodes_json: Vec<serde_json::Value> = nodes
        .iter()
        .map(|n| {
            let mut obj = serde_json::json!({
                "id": n.id,
                "role": n.role,
                "status": n.status,
            });
            if verbose {
                obj["address"] = serde_json::Value::String(n.address.clone());
            }
            obj
        })
        .collect();
    serde_json::json!({
        "metrics": metrics,
        "prefill_hits": prefill_hits,
        "decode_hits": decode_hits,
        "nodes": nodes_json,
        "addresses_redacted": !verbose,
    })
}

/// GET /router/stats
///
/// Report the router's load distribution and routing metrics (issue #201): the
/// per-node dispatch counts for the prefill and decode pools, the registered
/// nodes with their current health status, and the [`RouterMetrics`] snapshot.
/// An operator (and the multi-node E2E) uses this to confirm requests spread
/// across the pools and that a failed node is marked unreachable.
///
/// This endpoint is mounted on the same client-facing axum app as
/// `/v1/chat/completions`, so by default it redacts each node's raw `host:port`
/// to avoid disclosing the internal cluster topology to an unauthenticated
/// client that can reach the inference port (issue #389). The node ids
/// (`prefill-<i>` / `decode-<i>`) and the dispatch counts are stable opaque
/// labels, not addresses, so they stay. Set [`ROUTER_STATS_VERBOSE_ENV`] to opt
/// a trusted-segment deployment back into the full address view.
async fn router_stats(State(state): State<Arc<RouterState>>) -> Json<serde_json::Value> {
    let prefill_hits = state.prefill_hits.lock().unwrap().clone();
    let decode_hits = state.decode_hits.lock().unwrap().clone();
    let nodes: Vec<StatsNode> = state
        .registry
        .all_nodes()
        .into_iter()
        .map(|n| StatsNode {
            id: n.config.id,
            role: n.config.role.to_string(),
            status: n.status.to_string(),
            address: n.config.address.to_string(),
        })
        .collect();
    let verbose = router_stats_verbose();
    Json(router_stats_body(
        &nodes,
        &prefill_hits,
        &decode_hits,
        &state.router.metrics(),
        verbose,
    ))
}

/// POST /v1/chat/completions
async fn router_chat_completions(
    State(state): State<Arc<RouterState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    match route_chat(state, request).await {
        Ok(resp) => resp,
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": {"message": e.to_string()}})),
        )
            .into_response(),
    }
}

/// Apply router-level admission control before dispatching (issue #201).
///
/// Consults [`RequestRouter::apply_backpressure`]. When the prefill queue is at
/// capacity (`Reject`) or no prefill node is currently available (`Queue`, e.g.
/// every prefill node is marked unreachable), the router has nowhere to send the
/// request and no async queue to park it in, so it returns HTTP 503 instead of
/// attempting a doomed dispatch. `Accept` returns `None` and the caller proceeds
/// exactly as before, so the healthy path is unchanged.
fn admission_reject(state: &RouterState) -> Option<Response> {
    match state.router.apply_backpressure() {
        BackpressureAction::Accept => None,
        BackpressureAction::Queue => Some(service_unavailable(
            "all serving nodes are at capacity or unreachable; retry shortly",
        )),
        BackpressureAction::Reject(reason) => Some(service_unavailable(&format!(
            "router rejected request: {reason}"
        ))),
    }
}

/// Build an HTTP 503 JSON error response for a backpressure rejection.
fn service_unavailable(message: &str) -> Response {
    (
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        Json(serde_json::json!({
            "error": {"message": message, "type": "service_unavailable"}
        })),
    )
        .into_response()
}

/// Core chat routing logic: tokenizes, sends to prefill, merges result.
async fn route_chat(state: Arc<RouterState>, request: ChatCompletionRequest) -> Result<Response> {
    // Admission control first: reject with 503 when the cluster cannot take the
    // request (issue #201), before spending work on template rendering.
    if let Some(resp) = admission_reject(&state) {
        return Ok(resp);
    }

    // Render the chat template and reject multimodal requests (the
    // disaggregated path is text-only for pool-backed Fp16 families).
    let prepared = super::chat_request::prepare_chat_request_with_cache(
        &state.chat_template,
        &request,
        state.config.chat_template_kwargs.as_ref(),
        false,
    )
    .await
    .map_err(|e| anyhow::anyhow!("{e}"))?;

    if !prepared.image_data.is_empty()
        || !prepared.audio_data.is_empty()
        || !prepared.videos.is_empty()
    {
        anyhow::bail!("the disaggregated router supports text-only requests");
    }

    let prompt = prepared.prompt;

    // Stream filter (issue #198): mirror the single-node chat route, including
    // the primed-open-thinking start state when the rendered prompt ends
    // inside an open thinking block (enable_thinking=true templates), so the
    // model's first emitted tokens route to `reasoning_content`.
    let primed_open_thinking = super::routes::chat::is_prompt_primed_open_thinking(&prompt);
    let stream_filter = if primed_open_thinking {
        StreamFilter::new_primed_open_thinking()
    } else {
        StreamFilter::new()
    };

    // Resolve sampling and token budget using the same defaults as the
    // model worker.
    let opts = super::routes::chat::build_generate_options(&request.params, &state.config);

    // Assign a request id and dispatch the prefill request through the shared
    // tokenize -> select_prefill -> send body (issue #200). The chat id scheme
    // (`chatcmpl-<n>`) is preserved exactly.
    let request_id = state.next_id.fetch_add(1, Ordering::Relaxed);
    let request_id_str = format!("chatcmpl-{request_id}");
    let (prompt_tokens, rx) =
        start_handoff(&state, &prompt, request_id, &request_id_str, &opts).await?;

    if request.stream {
        // Streaming: spawn a task that drives the handoff and feeds SSE events
        // into an unbounded channel; return the SSE response immediately. The
        // optional trailing usage chunk mirrors the router's own
        // `/v1/completions` streaming path and single-node chat (issue #398).
        let include_usage = wants_stream_usage(&request);
        let (chunk_tx, chunk_rx) =
            tokio::sync::mpsc::unbounded_channel::<Result<Event, Infallible>>();
        let state2 = state.clone();
        let request_id_str2 = request_id_str.clone();
        let model = request.model.clone();
        let handoff_timeout = state.handoff_timeout;
        let max_tokens = opts.max_tokens;

        let mut filter = stream_filter;
        tokio::spawn(async move {
            let _ = chunk_tx.send(Ok(sse_event(&chat_chunk_initial(&request_id_str2, &model))));

            let emit_filtered = |emit: FilterOutput| {
                if let Some(reasoning) = emit.reasoning
                    && !reasoning.is_empty()
                {
                    let _ = chunk_tx.send(Ok(sse_event(&chat_chunk_reasoning(
                        &request_id_str2,
                        &model,
                        &reasoning,
                    ))));
                }
                if let Some(content) = emit.content
                    && !content.is_empty()
                {
                    let _ = chunk_tx.send(Ok(sse_event(&chat_chunk_content(
                        &request_id_str2,
                        &model,
                        &content,
                    ))));
                }
            };

            // `frame_counted` is the emitted-piece fallback count; the resolved
            // count below prefers the worker's authoritative model-token count
            // (issue #387) so the finish_reason matches single-node even for
            // byte-fallback tokenizers.
            let mut frame_counted = 0usize;
            let result = drive_handoff_result(
                &mut { rx },
                &request_id_str2,
                handoff_timeout,
                max_tokens,
                |text| {
                    frame_counted += 1;
                    emit_filtered(filter.feed(text));
                },
            )
            .await;

            // Flush any text still buffered inside the filter (e.g. an
            // unterminated partial delimiter match at end of stream).
            emit_filtered(filter.flush());

            let completed = result.is_ok();
            // Resolve the authoritative completion-token count on success (issue
            // #387); `None` on a handoff failure, which also means "stop" for
            // `finish_reason` and no usage chunk below (issue #398), matching
            // the router's own `/v1/completions` streaming path.
            let completion_tokens = result
                .as_ref()
                .ok()
                .map(|outcome| resolve_completion_tokens(outcome, frame_counted, max_tokens));
            // Mirror single-node chat: "length" when the whole token budget was
            // generated, else "stop". On a handoff failure finish as "stop"
            // alongside the visible error delta emitted just above.
            let finish_reason = match completion_tokens {
                Some(n) if n >= max_tokens => "length",
                _ => "stop",
            };
            if let Err(e) = result {
                let _ = chunk_tx.send(Ok(sse_event(&chat_chunk_error(
                    &request_id_str2,
                    &model,
                    &e.to_string(),
                ))));
            }
            let _ = chunk_tx.send(Ok(sse_event(&chat_chunk_finish(
                &request_id_str2,
                &model,
                finish_reason,
            ))));

            // Emit the usage chunk only on success, matching single-node
            // `stream_chat_completion` (`if include_usage && let Ok(ref r) =
            // result`) and the router's own `/v1/completions` streaming path.
            // On a handoff failure (`completion_tokens` is `None`) no usage
            // chunk is sent.
            if include_usage && let Some(completion_tokens) = completion_tokens {
                let _ = chunk_tx.send(Ok(sse_event(&chat_chunk_usage(
                    &request_id_str2,
                    &model,
                    prompt_tokens,
                    completion_tokens,
                ))));
            }
            let _ = chunk_tx.send(Ok(Event::default().data("[DONE]")));

            state2.pending.lock().unwrap().remove(&request_id);
            state2.finalize_request(&request_id_str2, completed);
        });

        Ok(Sse::new(UnboundedReceiverStream::new(chunk_rx))
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        // Non-streaming: collect all tokens (filtered) then return a single
        // JSON object with `content`, `reasoning_content` when present, and a
        // `usage` block matching single-node's `ChatCompletionResponse` shape
        // (issue #398).
        let mut filter = stream_filter;
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut rx = rx;
        let finish_reason;
        let completion_tokens;
        {
            let mut frame_counted = 0usize;
            let mut absorb = |emit: FilterOutput| {
                if let Some(r) = emit.reasoning {
                    reasoning.push_str(&r);
                }
                if let Some(c) = emit.content {
                    content.push_str(&c);
                }
            };
            let r = drive_handoff_result(
                &mut rx,
                &request_id_str,
                state.handoff_timeout,
                opts.max_tokens,
                |text| {
                    frame_counted += 1;
                    absorb(filter.feed(text));
                },
            )
            .await;
            absorb(filter.flush());
            state.pending.lock().unwrap().remove(&request_id);
            state.finalize_request(&request_id_str, r.is_ok());
            let outcome = r?;
            // Mirror single-node chat finish_reason: "length" when the whole
            // token budget was generated, else "stop", using the worker's
            // authoritative count when present (issue #387). The same
            // resolved count feeds `usage.completion_tokens` below (issue #398).
            completion_tokens = resolve_completion_tokens(&outcome, frame_counted, opts.max_tokens);
            finish_reason = if completion_tokens >= opts.max_tokens {
                "length"
            } else {
                "stop"
            };
        }
        Ok(Json(chat_completion_json(
            &request_id_str,
            &request.model,
            &content,
            &reasoning,
            finish_reason,
            prompt_tokens,
            completion_tokens,
        ))
        .into_response())
    }
}

/// POST /v1/completions
async fn router_completions(
    State(state): State<Arc<RouterState>>,
    Json(request): Json<CompletionRequest>,
) -> Response {
    match route_completion(state, request).await {
        Ok(resp) => resp,
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": {"message": e.to_string()}})),
        )
            .into_response(),
    }
}

/// Core text-completion routing logic (issue #200).
///
/// Mirrors the single-node `/v1/completions` semantics: the raw `prompt` is
/// tokenized with the same `add_special` rule the worker uses (handled inside
/// [`start_handoff`]), generation is dispatched through the shared handoff body,
/// and the output is serialized with the SAME [`CompletionResponse`] /
/// [`CompletionChunk`] types the single-node route uses, so the wire shape is
/// byte-identical (modulo the volatile `id` / `created` fields) for the same
/// prompt and sampling.
///
/// # Parity notes
///
/// - `logprobs`, `response_format` (structured output), and explicit
///   reasoning/thinking budgets are rejected with a 400. The
///   [`PrefillRequestFrame`] carries only sampling and max_tokens, so the
///   worker-side behavior these options trigger (per-token logprob data, a
///   structured-output constraint, reasoning-budget enforcement) cannot be
///   reproduced on the router path. Rejecting keeps the router consistent with
///   single-node rather than returning 200 with silently divergent output.
/// - `completion_tokens` uses the worker's authoritative generated-token count
///   carried over the wire ([`ResultFrame::generated_tokens`], issue #387),
///   which is exact even for byte-fallback tokenizers (e.g. Gemma `<0xXX>` byte
///   sequences) where counting emitted detokenized text pieces under-counts.
///   `finish_reason` is derived from that authoritative count with the same
///   `count >= max_tokens` formula the worker uses, so it matches single-node.
///   Against a mixed-version cluster where a node predates the wire field, the
///   router falls back to counting emitted text pieces (the prior behavior).
async fn route_completion(state: Arc<RouterState>, request: CompletionRequest) -> Result<Response> {
    // Admission control first: reject with 503 when the cluster cannot take the
    // request (issue #201), before any per-request work.
    if let Some(resp) = admission_reject(&state) {
        return Ok(resp);
    }

    // The disaggregated result frames carry detokenized text only, with no
    // per-token logprob data, so the router cannot reproduce the single-node
    // `logprobs` object. Reject rather than emit a divergent null-logprobs
    // body (the chat path rejects multimodal requests on the same principle).
    if request.logprobs.is_some() {
        return Ok(ErrorResponse::new(
            "the disaggregated router does not support logprobs on /v1/completions",
            "invalid_request_error",
        )
        .into_response());
    }

    // The PrefillRequestFrame carries only sampling and max_tokens, so the
    // worker's structured-output constraint cannot be reproduced on the router
    // path. Reject rather than emit unconstrained output that silently diverges
    // from single-node (the same reject-what-the-frame-cannot-reproduce
    // principle as the logprobs guard above).
    if request.response_format.is_some() {
        return Ok(ErrorResponse::new(
            "the disaggregated router does not support response_format (structured output) on /v1/completions",
            "invalid_request_error",
        )
        .into_response());
    }

    // Reasoning/thinking-budget enforcement is worker-side and is not carried by
    // the PrefillRequestFrame, so it cannot be reproduced on the router path.
    // Reject an explicit budget rather than emit un-budgeted output that
    // diverges from single-node. A request with no budget alias set still works
    // (the default unbounded path needs no frame support).
    if crate::server::thinking_budget::pick_budget_alias(
        request.params.thinking_budget_tokens,
        request.params.thinking_token_budget,
        request.params.thinking_budget,
    )
    .is_some()
    {
        return Ok(ErrorResponse::new(
            "the disaggregated router does not support reasoning/thinking budgets on /v1/completions",
            "invalid_request_error",
        )
        .into_response());
    }

    let prompt = request.prompt.clone();
    // Same default/override resolution as the single-node completion route.
    let opts = super::routes::chat::build_generate_options(&request.params, &state.config);

    // Assign a request id and dispatch the prefill request through the shared
    // tokenize -> select_prefill -> send body. The completion id format
    // (`cmpl-<uuid>`) matches the single-node route's `format!("cmpl-{uuid}")`.
    let request_id = state.next_id.fetch_add(1, Ordering::Relaxed);
    let response_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let (prompt_tokens, rx) =
        start_handoff(&state, &prompt, request_id, &response_id, &opts).await?;

    if request.stream {
        // Streaming: spawn a task that drives the handoff and feeds text
        // completion chunks into an SSE channel; return the SSE response
        // immediately. The chunk shape, finish chunk, optional usage chunk, and
        // `[DONE]` sentinel match the single-node `stream_completion` path.
        let include_usage = request
            .stream_options
            .as_ref()
            .map(|o| o.include_usage)
            .unwrap_or(false);
        let (chunk_tx, chunk_rx) =
            tokio::sync::mpsc::unbounded_channel::<Result<Event, Infallible>>();
        let state2 = state.clone();
        let model = request.model.clone();
        let response_id2 = response_id.clone();
        let handoff_timeout = state.handoff_timeout;
        let max_tokens = opts.max_tokens;

        tokio::spawn(async move {
            // `frame_counted` is the number of emitted detokenized text pieces
            // (one `on_token` call per `ResultFrame` text entry). It is the
            // fallback usage count when the wire carries no authoritative count
            // (an older worker); the resolved `completion_tokens` below prefers
            // the worker's true model-token count (issue #387).
            let mut frame_counted = 0usize;
            let mut rx = rx;
            let result = drive_handoff_result(
                &mut rx,
                &response_id2,
                handoff_timeout,
                max_tokens,
                |text| {
                    frame_counted += 1;
                    let chunk = CompletionChunk::content(
                        response_id2.clone(),
                        model.clone(),
                        text.to_string(),
                    );
                    let _ = chunk_tx.send(Ok(sse_serialize(&chunk)));
                },
            )
            .await;

            // Resolve the request's completion token count: prefer the worker's
            // authoritative count carried over the wire (issue #387), which is
            // exact even for byte-fallback tokenizers (e.g. Gemma `<0xXX>` byte
            // sequences) where counting emitted text pieces under-counts; fall
            // back to the emitted-piece count for an older worker.
            let completion_tokens = match &result {
                Ok(outcome) => resolve_completion_tokens(outcome, frame_counted, max_tokens),
                Err(_) => frame_counted,
            };

            // Mirror the single-node finish_reason exactly: the worker reports
            // "length" when it generated the whole token budget, else "stop"
            // (model_worker.rs). The router applies the same formula to the
            // authoritative count, so it reproduces the single-node value. On a
            // handoff failure it reports "error" (matching single-node's
            // streaming error finish_reason).
            let finish_reason = if result.is_err() {
                "error"
            } else if completion_tokens >= max_tokens {
                "length"
            } else {
                "stop"
            };
            let finish = CompletionChunk::finish(
                response_id2.clone(),
                model.clone(),
                finish_reason.to_string(),
            );
            let _ = chunk_tx.send(Ok(sse_serialize(&finish)));

            // Emit the usage chunk only on success, matching single-node
            // `stream_completion` which guards `if include_usage && let Ok(ref r)
            // = result`. On a handoff failure (finish_reason "error") no usage
            // chunk is sent.
            if include_usage && result.is_ok() {
                let usage = CompletionChunk::usage(
                    response_id2.clone(),
                    model.clone(),
                    prompt_tokens,
                    completion_tokens,
                );
                let _ = chunk_tx.send(Ok(sse_serialize(&usage)));
            }
            let _ = chunk_tx.send(Ok(Event::default().data("[DONE]")));

            state2.pending.lock().unwrap().remove(&request_id);
            state2.finalize_request(&response_id2, result.is_ok());
        });

        Ok(Sse::new(UnboundedReceiverStream::new(chunk_rx))
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        // Non-streaming: collect all tokens, then return a single
        // `CompletionResponse` JSON object identical in shape to single-node.
        let mut text = String::new();
        let mut frame_counted = 0usize;
        let mut rx = rx;
        let r = drive_handoff_result(
            &mut rx,
            &response_id,
            state.handoff_timeout,
            opts.max_tokens,
            |piece| {
                text.push_str(piece);
                frame_counted += 1;
            },
        )
        .await;
        state.pending.lock().unwrap().remove(&request_id);
        state.finalize_request(&response_id, r.is_ok());
        let outcome = r?;

        // Prefer the worker's authoritative token count carried over the wire
        // (issue #387), which is exact even for byte-fallback tokenizers where
        // counting emitted text pieces under-counts; fall back to the emitted-
        // piece count for an older worker.
        let completion_tokens = resolve_completion_tokens(&outcome, frame_counted, opts.max_tokens);
        // Mirror the single-node finish_reason exactly (model_worker.rs):
        // "length" when the whole token budget was generated, else "stop".
        // logprobs are rejected up front, so always `None` here.
        let finish_reason = if completion_tokens >= opts.max_tokens {
            "length"
        } else {
            "stop"
        };
        let response = CompletionResponse::new_with_logprobs(
            response_id,
            request.model.clone(),
            text,
            prompt_tokens,
            completion_tokens,
            Some(finish_reason.to_string()),
            None,
        );
        Ok(Json(response).into_response())
    }
}

// ── Shared handoff dispatch ──────────────────────────────────────────────

/// Shared dispatch body for the router's chat and text-completion handlers
/// (issue #200).
///
/// Tokenizes the rendered `prompt` with the worker's `add_special` rule
/// (`!prompt.starts_with("<bos>") && !prompt.starts_with("<s>")`), routes it to
/// a prefill node, registers a per-request result channel keyed by the numeric
/// `request_id`, and sends the [`PrefillRequestFrame`]. The caller then drives
/// the returned receiver with [`drive_handoff_result`] and shapes the
/// per-endpoint response (chat-completion chunks vs text-completion chunks).
///
/// `response_id` is the request's display id (`chatcmpl-<n>` for chat,
/// `cmpl-<uuid>` for completions); it seeds the routing hash and the
/// [`StreamBridge`] id. Returns the prompt token count (for the completion
/// `usage` block) and the per-request receiver. On any failure before the frame
/// is sent the registered channel is removed so a failed request never leaks an
/// entry in `state.pending`.
async fn start_handoff(
    state: &Arc<RouterState>,
    prompt: &str,
    request_id: u64,
    response_id: &str,
    opts: &ServerGenerateOptions,
) -> Result<(usize, UnboundedReceiver<ResultFrame>)> {
    // Tokenize the rendered prompt. Match the worker's behavior: skip the
    // BOS special token when the prompt already starts with one.
    let add_special = !prompt.starts_with("<bos>") && !prompt.starts_with("<s>");
    let token_ids: Vec<i32> = state
        .tokenizer
        .encode(prompt, add_special)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
        .into_iter()
        .map(|t| t as i32)
        .collect();
    let prompt_tokens = token_ids.len();

    // The router tracks the request under the string id derived from
    // `response_id`, so `route_to_prefill` here and `route_to_decode` below
    // operate on the same tracked entry.
    let rid = RequestId::from_string(response_id.to_string()).unwrap_or_default();

    // Pick the prefill node (tracks the request) and, when possible, the decode
    // node too (issue #201). The router-chosen decode target rides the frame so
    // the prefill node hands the KV cache to the router-balanced decode node
    // instead of its own static `--decode-peers` config. A `None` decode target
    // (no decode peer routable) leaves the prefill node on its config fallback.
    let mut prefill = state.select_prefill(&rid, prompt_tokens, true)?;
    let decode = state.select_decode(&rid);

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<ResultFrame>();
    state.pending.lock().unwrap().insert(request_id, tx);

    // Build the prefill request frame once; re-encode per send attempt so a
    // failover retry can reuse it without requiring the transport message to be
    // cloneable.
    let frame = PrefillRequestFrame {
        request_id,
        prompt_tokens: token_ids,
        sampling: sampling_to_serializable(&opts.sampling),
        max_tokens: opts.max_tokens as u64,
        reply_to: state.reply_to.clone(),
        decode_target: decode.as_ref().map(|d| d.addr.clone()),
    };

    // Send to the prefill node, failing over to other healthy prefill nodes on a
    // transport error. A send error means the node is unreachable, so mark it
    // down in the registry (which makes `route_to_prefill` skip it), re-route any
    // other in-flight requests it held via `handle_node_failure`, and re-select a
    // healthy node for this request. Bounded by the prefill-node count so a fully
    // dead pool fails the request cleanly instead of spinning. No registry / map
    // lock is held across the `await`, mirroring the existing discipline.
    let max_attempts = state
        .registry
        .nodes_with_role(NodeRole::Prefill)
        .len()
        .max(1);
    let mut attempt = 0usize;
    loop {
        let encoded = match frame.encode() {
            Ok(encoded) => encoded,
            Err(e) => {
                state.pending.lock().unwrap().remove(&request_id);
                state.finalize_request(response_id, false);
                return Err(anyhow::anyhow!("{e}"));
            }
        };
        match state.transport.send(&prefill.addr, encoded).await {
            Ok(()) => break,
            Err(e) => {
                attempt += 1;
                if let Some(failed_id) = prefill.node_id.clone() {
                    state
                        .registry
                        .set_node_status(&failed_id, NodeStatus::Unreachable);
                    let (rerouted, failed) = state.router.handle_node_failure(&failed_id);
                    tracing::warn!(
                        node = %failed_id, addr = %prefill.addr, rerouted, failed,
                        "router: prefill node send failed; marked unreachable: {e}"
                    );
                }
                if attempt >= max_attempts {
                    state.pending.lock().unwrap().remove(&request_id);
                    state.finalize_request(response_id, false);
                    return Err(anyhow::anyhow!(
                        "send prefill request: no healthy prefill node after {attempt} attempt(s): {e}"
                    ));
                }
                // Re-select a healthy prefill node (the failed one is now
                // Unreachable, so the router skips it). No static fallback on
                // retry: a doomed re-send to a dead config address is not a fix.
                match state.select_prefill(&rid, prompt_tokens, false) {
                    Ok(next) => prefill = next,
                    Err(re) => {
                        state.pending.lock().unwrap().remove(&request_id);
                        state.finalize_request(response_id, false);
                        return Err(anyhow::anyhow!(
                            "send prefill request: {e}; no healthy prefill node to retry: {re}"
                        ));
                    }
                }
            }
        }
    }

    // Count the dispatched request against the nodes actually used, for the
    // `GET /router/stats` distribution view. Each lock is taken and released
    // without crossing an await.
    if let Some(id) = &prefill.node_id {
        *state
            .prefill_hits
            .lock()
            .unwrap()
            .entry(id.clone())
            .or_insert(0) += 1;
    }
    if let Some(d) = &decode {
        *state
            .decode_hits
            .lock()
            .unwrap()
            .entry(d.node_id.clone())
            .or_insert(0) += 1;
        // Reflect the in-flight decode phase for the metrics snapshot.
        let _ = state.router.mark_decoding(&rid, &d.node_id);
    }
    tracing::debug!(
        request_id,
        prefill = %prefill.addr,
        decode = decode.as_ref().map(|d| d.addr.as_str()).unwrap_or("<config-fallback>"),
        "router: sent prefill request frame"
    );
    Ok((prompt_tokens, rx))
}

// ── Handoff result driver ────────────────────────────────────────────────

/// The authoritative generation outcome the router extracts from the result
/// frames (issue #387), used to report `usage.completion_tokens` and to derive
/// `finish_reason` without under-counting byte-fallback tokenizers.
struct HandoffOutcome {
    /// Sum of the per-node authoritative model-token counts the workers reported
    /// (the prefill node's first token plus the decode node's continuation).
    /// `None` when no frame carried a count, which happens only against a
    /// mixed-version cluster running an older prefill or decode node that
    /// predates the wire field; the caller then falls back to counting emitted
    /// text pieces.
    generated_tokens: Option<u64>,
}

/// Resolve a request's `completion_tokens` from the disaggregated result.
///
/// Prefers the worker's authoritative generated-token count carried over the
/// wire (issue #387), which is exact even for byte-fallback tokenizers, and
/// falls back to `frame_counted` (the number of emitted detokenized text pieces)
/// when no node reported a count. The authoritative count is clamped to
/// `max_tokens`: the router bounds generation to its own budget regardless of
/// the remote `done` flag, so a larger reported count can only come from a buggy
/// or hostile node and must not inflate the usage figure.
fn resolve_completion_tokens(
    outcome: &HandoffOutcome,
    frame_counted: usize,
    max_tokens: usize,
) -> usize {
    outcome
        .generated_tokens
        .map(|n| (n as usize).min(max_tokens))
        .unwrap_or(frame_counted.min(max_tokens))
}

/// Whether the client requested the trailing streaming usage chunk via
/// `stream_options.include_usage`. Defaults to `false` when the request omits
/// `stream_options` entirely, or when the field is present but `false`,
/// matching single-node chat/completions (`chat.rs`, `completions.rs`) and
/// the router's own `/v1/completions` streaming path (issue #398).
fn wants_stream_usage(request: &ChatCompletionRequest) -> bool {
    request
        .stream_options
        .as_ref()
        .map(|o| o.include_usage)
        .unwrap_or(false)
}

/// Consume the two-part disaggregated result (prefill first token + decode
/// continuation), call `on_token` for every text piece in order, and return the
/// authoritative generation outcome once the stream is finalized (or an error
/// occurs).
///
/// Uses [`StreamBridge`] to enforce the prefill-decode phase ordering and
/// detect sequence gaps. The returned [`HandoffOutcome`] sums the worker-
/// reported model-token counts so the caller reports exact usage (issue #387).
async fn drive_handoff_result(
    rx: &mut UnboundedReceiver<ResultFrame>,
    request_id_str: &str,
    handoff_timeout: Duration,
    max_tokens: usize,
    mut on_token: impl FnMut(&str),
) -> Result<HandoffOutcome> {
    let bridge = StreamBridge::new(request_id_str.to_string(), handoff_timeout);

    // Accumulate the per-node authoritative model-token counts (issue #387): the
    // prefill first-token frame carries its count, the decode terminal frame
    // carries the continuation count. Summing any frame that carries one is
    // robust to which node ends the stream.
    let mut generated_tokens: Option<u64> = None;
    let mut add_count = |frame_count: Option<u64>| {
        if let Some(n) = frame_count {
            generated_tokens = Some(generated_tokens.unwrap_or(0).saturating_add(n));
        }
    };

    // Wait for the prefill node's first-token result.
    let first = tokio::time::timeout(handoff_timeout, rx.recv())
        .await
        .map_err(|_| anyhow::anyhow!("timed out waiting for the prefill first token"))?
        .ok_or_else(|| anyhow::anyhow!("prefill result channel closed before first token"))?;

    if let Some(e) = first.error {
        anyhow::bail!("prefill node error: {e}");
    }
    add_count(first.generated_tokens);
    if let Some(text) = first.tokens.first() {
        bridge
            .submit_first_token(&TokenEvent {
                token_id: 0,
                text: text.clone(),
                sequence_number: 0,
                source: TokenSource::Prefill,
                is_final: false,
            })
            .map_err(|e| anyhow::anyhow!("stream bridge: {e}"))?;
        on_token(text);
    }
    if first.done {
        bridge.finalize();
        return Ok(HandoffOutcome { generated_tokens });
    }

    // Transition to the decode phase and consume the incrementally streamed
    // continuation frames (issue #199) until the terminal `done` frame. The
    // timeout applies per frame, so long generations keep streaming as long
    // as the decode node makes progress.
    bridge
        .start_decode_stream()
        .map_err(|e| anyhow::anyhow!("stream bridge: {e}"))?;

    let mut seq: u64 = 1;
    loop {
        let cont = tokio::time::timeout(handoff_timeout, rx.recv())
            .await
            .map_err(|_| anyhow::anyhow!("timed out waiting for the decode continuation"))?
            .ok_or_else(|| anyhow::anyhow!("decode result channel closed before continuation"))?;

        if let Some(e) = cont.error {
            anyhow::bail!("decode node error: {e}");
        }
        add_count(cont.generated_tokens);
        // The router set the request's token budget itself; do not trust the
        // remote `done` flag to terminate the stream. A decode node that
        // exceeds the budget (buggy or hostile) is cut off here, which also
        // bounds the total frame count and, with the per-frame timeout, the
        // total wall-clock time per request.
        if seq as usize > max_tokens {
            anyhow::bail!(
                "decode node exceeded the request token budget ({max_tokens}) without \
                 a terminal frame"
            );
        }
        // Wire-level ordering check: a non-zero `start_sequence` must match
        // the next expected position (frames could in principle reorder
        // across pooled transport connections). This is a liveness and
        // debugging aid against benign loss or reordering, NOT a tamper
        // defense: the transport is unauthenticated and the disaggregated
        // deployment model assumes a trusted network segment.
        if cont.start_sequence != 0 && cont.start_sequence != seq {
            anyhow::bail!(
                "decode continuation frame out of order: expected sequence {seq}, \
                 got {}",
                cont.start_sequence
            );
        }
        for text in &cont.tokens {
            bridge
                .submit_decode_token(&TokenEvent {
                    token_id: 0,
                    text: text.clone(),
                    sequence_number: seq,
                    source: TokenSource::Decode,
                    is_final: false,
                })
                .map_err(|e| anyhow::anyhow!("stream bridge: {e}"))?;
            on_token(text);
            seq += 1;
        }
        if cont.done {
            break;
        }
    }
    bridge.finalize();
    Ok(HandoffOutcome { generated_tokens })
}

// ── Small JSON helpers ───────────────────────────────────────────────────

/// Build an SSE event from a JSON value.
fn sse_event(value: &serde_json::Value) -> Event {
    Event::default().data(serde_json::to_string(value).unwrap_or_default())
}

/// Build an SSE event from any serializable value (e.g. a typed
/// [`CompletionChunk`]). The serialized `data:` line matches the single-node
/// streaming path's `Event::default().data(serde_json::to_string(..))` framing.
fn sse_serialize<T: serde::Serialize>(value: &T) -> Event {
    Event::default().data(serde_json::to_string(value).unwrap_or_default())
}

/// Initial streaming chunk: sets `delta.role = "assistant"`.
fn chat_chunk_initial(id: &str, model: &str) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": null
        }]
    })
}

/// Streaming chunk carrying a content token.
fn chat_chunk_content(id: &str, model: &str, text: &str) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": text},
            "finish_reason": null
        }]
    })
}

/// Streaming chunk carrying a reasoning (thinking) token.
fn chat_chunk_reasoning(id: &str, model: &str, text: &str) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"reasoning_content": text},
            "finish_reason": null
        }]
    })
}

/// Final streaming chunk carrying the request's `finish_reason` ("stop" or
/// "length"); the router derives it from the worker's authoritative token count
/// (issue #387) so it matches the single-node chat route.
fn chat_chunk_finish(id: &str, model: &str, finish_reason: &str) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason
        }]
    })
}

/// Error chunk: injects a visible error message into the content delta.
fn chat_chunk_error(id: &str, model: &str, msg: &str) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": format!("[router error: {msg}]")},
            "finish_reason": "stop"
        }]
    })
}

/// Final usage chunk, sent only when the client requested
/// `stream_options.include_usage` and the handoff succeeded (issue #398).
/// `choices` is empty per the OpenAI streaming-usage-chunk convention.
/// `completion_tokens` is the caller's already-resolved authoritative count
/// (issue #387); mirrors the router's own `/v1/completions` streaming usage
/// chunk (`CompletionChunk::usage`) and single-node chat's
/// `ChatCompletionChunk::usage_with_cache` (without the prompt-cache
/// breakdown, which the router does not track).
fn chat_chunk_usage(
    id: &str,
    model: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    })
}

/// Non-streaming response body. `reasoning_content` is included only when
/// the filter routed any thinking text. `finish_reason` ("stop" or "length")
/// is derived from the worker's authoritative token count (issue #387) so it
/// matches the single-node chat route. `usage` mirrors single-node's
/// `ChatCompletionResponse` shape (`prompt_tokens`, `completion_tokens`,
/// `total_tokens`); `completion_tokens` is the caller's already-resolved
/// authoritative count (issue #398).
fn chat_completion_json(
    id: &str,
    model: &str,
    content: &str,
    reasoning: &str,
    finish_reason: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> serde_json::Value {
    let mut message = serde_json::json!({"role": "assistant", "content": content});
    if !reasoning.is_empty() {
        message["reasoning_content"] = serde_json::Value::String(reasoning.to_string());
    }
    serde_json::json!({
        "id": id,
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    })
}

// ── Router app factory ───────────────────────────────────────────────────

/// Build the axum `Router` for the disaggregated front-end.
///
/// Exposes:
/// - `GET /health` - liveness probe
/// - `GET /router/stats` - per-node dispatch distribution, node health, and
///   routing metrics (issue #201); raw node addresses are redacted on this
///   client-facing surface unless `MLXCEL_ROUTER_STATS_VERBOSE` is set
///   (issue #389)
/// - `POST /v1/chat/completions` - chat completions (streaming and non-streaming)
/// - `POST /v1/completions` - text completions (streaming and non-streaming)
pub fn create_router_app(state: Arc<RouterState>) -> Router {
    Router::new()
        .route("/health", get(router_health))
        .route("/router/stats", get(router_stats))
        .route("/v1/chat/completions", post(router_chat_completions))
        .route("/v1/completions", post(router_completions))
        .with_state(state)
}

#[cfg(test)]
#[path = "router_front_tests.rs"]
mod tests;
