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
//! weights. Incoming `/v1/chat/completions` requests are tokenized, forwarded
//! to a prefill node via [`PrefillRequestFrame`], and the two-part result
//! (prefill first token + decode continuation) is merged and returned to the
//! client as a streaming SSE response or a single JSON object.
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
    PrefillRequestFrame, RequestRouter, ResultFrame, RouterConfig, StreamBridge, TokenEvent,
    TokenSource, control_parts, sampling_to_serializable,
};
use crate::distributed::metrics::ClusterMetrics;
use crate::distributed::registry::{NodeRegistry, NodeStatus};
use crate::distributed::request_tracker::RequestId;
use crate::distributed::tcp_transport::TcpTransport;
use crate::distributed::transport::{Transport, TransportBackend};
use crate::server::ChatTemplateProcessor;
use crate::server::config::ServerConfig;
use crate::server::types::request::ChatCompletionRequest;
use crate::tokenizer::MlxcelTokenizer;

/// Timeout for waiting for the prefill first-token result and the decode
/// continuation from the serving nodes.
const HANDOFF_TIMEOUT_SECS: u64 = 120;

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

        let request_router = RequestRouter::new(
            RouterConfig::default(),
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
            chat_template,
            tokenizer,
            config,
            handoff_timeout: Duration::from_secs(HANDOFF_TIMEOUT_SECS),
        })
    }

    /// Resolve the prefill node address for a new request.
    ///
    /// Tries the [`RequestRouter`] first; falls back to the first entry in
    /// `prefill_fallback` on routing errors.
    fn select_prefill(&self, request_id_str: &str, prompt_len: usize) -> Result<String> {
        let rid = RequestId::from_string(request_id_str.to_string()).unwrap_or_default();
        match self.router.route_to_prefill(rid, prompt_len) {
            Ok(node_id) => self
                .registry
                .get_node(&node_id)
                .map(|n| n.config.address.to_string())
                .ok_or_else(|| anyhow::anyhow!("routed prefill node {node_id} not in registry")),
            Err(_) => self
                .prefill_fallback
                .first()
                .map(|a| a.to_string())
                .ok_or_else(|| anyhow::anyhow!("no prefill node available")),
        }
    }
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

// ── HTTP handlers ─────────────────────────────────────────────────────────

/// GET /health
async fn router_health() -> &'static str {
    "ok"
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

/// Core chat routing logic: tokenizes, sends to prefill, merges result.
async fn route_chat(state: Arc<RouterState>, request: ChatCompletionRequest) -> Result<Response> {
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

    // Tokenize the rendered prompt. Match the worker's behavior: skip the
    // BOS special token when the prompt already starts with one.
    let add_special = !prompt.starts_with("<bos>") && !prompt.starts_with("<s>");
    let token_ids: Vec<i32> = state
        .tokenizer
        .encode(&prompt, add_special)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
        .into_iter()
        .map(|t| t as i32)
        .collect();

    // Resolve sampling and token budget using the same defaults as the
    // model worker.
    let opts = super::routes::chat::build_generate_options(&request.params, &state.config);

    // Assign a request id and register a result channel.
    let request_id = state.next_id.fetch_add(1, Ordering::Relaxed);
    let request_id_str = format!("chatcmpl-{request_id}");
    let prefill_addr = state.select_prefill(&request_id_str, token_ids.len())?;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<ResultFrame>();
    state.pending.lock().unwrap().insert(request_id, tx);

    // Send the prefill request frame to the chosen prefill node.
    let frame = PrefillRequestFrame {
        request_id,
        prompt_tokens: token_ids,
        sampling: sampling_to_serializable(&opts.sampling),
        max_tokens: opts.max_tokens as u64,
        reply_to: state.reply_to.clone(),
    };
    if let Err(e) = state
        .transport
        .send(
            &prefill_addr,
            frame.encode().map_err(|e| anyhow::anyhow!("{e}"))?,
        )
        .await
    {
        state.pending.lock().unwrap().remove(&request_id);
        return Err(anyhow::anyhow!("send prefill request: {e}"));
    }
    tracing::debug!(request_id, %prefill_addr, "router: sent prefill request frame");

    if request.stream {
        // Streaming: spawn a task that drives the handoff and feeds SSE events
        // into an unbounded channel; return the SSE response immediately.
        let (chunk_tx, chunk_rx) =
            tokio::sync::mpsc::unbounded_channel::<Result<Event, Infallible>>();
        let state2 = state.clone();
        let request_id_str2 = request_id_str.clone();
        let model = request.model.clone();
        let handoff_timeout = state.handoff_timeout;

        tokio::spawn(async move {
            let _ = chunk_tx.send(Ok(sse_event(&chat_chunk_initial(&request_id_str2, &model))));

            let result =
                drive_handoff_result(&mut { rx }, &request_id_str2, handoff_timeout, |text| {
                    let _ = chunk_tx.send(Ok(sse_event(&chat_chunk_content(
                        &request_id_str2,
                        &model,
                        text,
                    ))));
                })
                .await;

            if let Err(e) = result {
                let _ = chunk_tx.send(Ok(sse_event(&chat_chunk_error(
                    &request_id_str2,
                    &model,
                    &e.to_string(),
                ))));
            }
            let _ = chunk_tx.send(Ok(sse_event(&chat_chunk_finish(&request_id_str2, &model))));
            let _ = chunk_tx.send(Ok(Event::default().data("[DONE]")));

            state2.pending.lock().unwrap().remove(&request_id);
        });

        Ok(Sse::new(UnboundedReceiverStream::new(chunk_rx))
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        // Non-streaming: collect all tokens then return a single JSON object.
        let mut content = String::new();
        let mut rx = rx;
        let r = drive_handoff_result(&mut rx, &request_id_str, state.handoff_timeout, |text| {
            content.push_str(text);
        })
        .await;
        state.pending.lock().unwrap().remove(&request_id);
        r?;
        Ok(Json(chat_completion_json(
            &request_id_str,
            &request.model,
            &content,
        ))
        .into_response())
    }
}

// ── Handoff result driver ────────────────────────────────────────────────

/// Consume the two-part disaggregated result (prefill first token + decode
/// continuation), call `on_token` for every text piece in order, and return
/// once the stream is finalized or an error occurs.
///
/// Uses [`StreamBridge`] to enforce the prefill-decode phase ordering and
/// detect sequence gaps.
async fn drive_handoff_result(
    rx: &mut UnboundedReceiver<ResultFrame>,
    request_id_str: &str,
    handoff_timeout: Duration,
    mut on_token: impl FnMut(&str),
) -> Result<()> {
    let bridge = StreamBridge::new(request_id_str.to_string(), handoff_timeout);

    // Wait for the prefill node's first-token result.
    let first = tokio::time::timeout(handoff_timeout, rx.recv())
        .await
        .map_err(|_| anyhow::anyhow!("timed out waiting for the prefill first token"))?
        .ok_or_else(|| anyhow::anyhow!("prefill result channel closed before first token"))?;

    if let Some(e) = first.error {
        anyhow::bail!("prefill node error: {e}");
    }
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
        return Ok(());
    }

    // Transition to the decode phase and wait for the continuation.
    bridge
        .start_decode_stream()
        .map_err(|e| anyhow::anyhow!("stream bridge: {e}"))?;

    let cont = tokio::time::timeout(handoff_timeout, rx.recv())
        .await
        .map_err(|_| anyhow::anyhow!("timed out waiting for the decode continuation"))?
        .ok_or_else(|| anyhow::anyhow!("decode result channel closed before continuation"))?;

    if let Some(e) = cont.error {
        anyhow::bail!("decode node error: {e}");
    }
    let mut seq: u64 = 1;
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
    bridge.finalize();
    Ok(())
}

// ── Small JSON helpers ───────────────────────────────────────────────────

/// Build an SSE event from a JSON value.
fn sse_event(value: &serde_json::Value) -> Event {
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

/// Final streaming chunk with `finish_reason = "stop"`.
fn chat_chunk_finish(id: &str, model: &str) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
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

/// Non-streaming response body.
fn chat_completion_json(id: &str, model: &str, content: &str) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }]
    })
}

// ── Router app factory ───────────────────────────────────────────────────

/// Build the axum `Router` for the disaggregated front-end.
///
/// Exposes:
/// - `GET /health` - liveness probe
/// - `POST /v1/chat/completions` - chat completions (streaming and non-streaming)
pub fn create_router_app(state: Arc<RouterState>) -> Router {
    Router::new()
        .route("/health", get(router_health))
        .route("/v1/chat/completions", post(router_chat_completions))
        .with_state(state)
}
