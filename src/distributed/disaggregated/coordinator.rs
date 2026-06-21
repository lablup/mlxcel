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

//! Serving-role coordinator skeleton for disaggregated paged KV handoff (#126
//! B2a).
//!
//! A [`ServingCoordinator`] binds a node's [`ServingMode`] to a concrete
//! [`Transport`] and the address of the peer it hands off to (a decode peer for
//! a prefill node) or receives from (a prefill peer for a decode node). It is
//! the seam between the local execution engine (the batch scheduler, which owns
//! the model and the `CachePool`) and the cross-node transport: a prefill node
//! extracts a finished sequence's KV and sends the frame to its decode peer; a
//! decode node receives the frame and ingests it onto a fresh local pool slot.
//!
//! # Scope of this skeleton (B2a)
//!
//! This is the additive skeleton. It carries the serving mode and owns the
//! transport seam through [`ServingCoordinator::send_handoff`] /
//! [`ServingCoordinator::recv_handoff`] (thin role-aware wrappers over the
//! [`super::handoff_impl`] async byte bridge, which already round-trips a
//! serialized cache frame over any [`Transport`]). What it deliberately does
//! NOT yet hold is a [`BatchScheduler`]: the scheduler-driving run loop (pull a
//! finished prefill -> `extract_sequence_handoff` -> [`Self::send_handoff`] on
//! the prefill side; [`Self::recv_handoff`] -> `ingest_sequence_handoff` ->
//! decode -> emit on the decode side) is the next step (B2b), where the
//! scheduler serving-role driver entry points land. Keeping the scheduler out
//! of this skeleton lets it stay model-free and unit-testable over an in-process
//! [`MockTransport`].
//!
//! The CLI/startup branch that constructs a coordinator for a non-hybrid
//! `--node-role` is plumbed separately ([`ServingMode`] now threads from
//! `--node-role` to the worker), but the live worker still runs the standard
//! single-node scheduler loop: a coordinator only becomes the live serving path
//! once B2b supplies the scheduler driver and B3 supplies a real network
//! transport (the in-process [`MockTransport`] connects two roles only within a
//! single process, which is how B2c exercises a real-model 2-role handoff).
//!
//! [`BatchScheduler`]: crate::server::batch::BatchScheduler
//! [`MockTransport`]: crate::distributed::mock_transport::MockTransport
//! [`Transport`]: crate::distributed::transport::Transport

use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

use anyhow::{Context, Result};
use mlxcel_core::generate::SamplingConfig;

use super::handoff_impl::{recv_handoff_payload, send_handoff_payload};
use super::serving::ServingMode;
use super::serving_protocol::{
    DecodeMetaFrame, PrefillRequestFrame, ResultFrame, ResultPhase, control_parts,
    sampling_from_serializable,
};
use crate::distributed::tcp_transport::{TcpTransport, TcpTransportConfig};
use crate::distributed::transport::Transport;
use crate::server::batch::BatchScheduler;
use crate::server::model_provider::GenerateEvent;

/// Set to `true` the first time the no-allowlist warning is emitted so
/// subsequent requests in the same process do not flood the log.
static ALLOWLIST_UNCONFIGURED_WARNED: AtomicBool = AtomicBool::new(false);

/// Allowlist of decode node addresses a prefill node will ship a KV cache
/// handoff to (issue #389, defense-in-depth on top of issue #201).
///
/// The router picks the decode node per request and ships it in
/// [`PrefillRequestFrame::decode_target`]; the prefill node then connects to
/// that address to deliver the KV cache (which encodes the prompt). Under the
/// trusted-network-segment model that is safe (the router only emits addresses
/// from its own configured registry), but a forged request frame from a breached
/// segment could redirect the handoff off-cluster. This allowlist lets the
/// prefill node reject a target it was not configured to know.
///
/// CRITICAL design constraint: the set must cover EVERY decode node the router
/// may select, not just this prefill's static handoff peer. Validating against a
/// single node would silently reject router-balanced targets and break the
/// multi-node decode balancing issue #201 added. The allowlist is therefore
/// decoupled from `--decode-peers` (whose first entry is only the static handoff
/// fallback) and comes from the dedicated `MLXCEL_DECODE_ALLOWLIST` env input: a
/// comma-separated `host:port` list an operator sets to the full pool of
/// router-selectable decode nodes (the shared cluster config). See
/// [`decode_allowlist_from_env`]. When that source is unset (an empty set), the
/// prefill node stays permissive-with-warning so router-driven balancing is never
/// silently disabled.
#[derive(Debug, Clone, Default)]
pub(crate) struct DecodeAllowlist {
    peers: HashSet<SocketAddr>,
}

/// The decision a prefill node makes for a router-chosen `decode_target`
/// (issue #389).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DecodeTargetDecision {
    /// The target is on the allowlist: forward the KV handoff.
    Allow,
    /// No allowlist is configured: forward but warn, so router-driven balancing
    /// is never silently broken by a missing allowlist source.
    AllowUnchecked,
    /// An allowlist is configured and the target is off-list: reject the handoff.
    Reject,
}

impl DecodeAllowlist {
    /// Build an allowlist from a node's configured decode peers.
    pub(crate) fn from_peers(peers: &[SocketAddr]) -> Self {
        Self {
            peers: peers.iter().copied().collect(),
        }
    }

    /// Decide whether a router-chosen `decode_target` may be connected to.
    ///
    /// The router only ever emits `SocketAddr`-formatted addresses (its registry
    /// is built from a `Vec<SocketAddr>`), so the target is parsed to a
    /// [`SocketAddr`] for a canonical comparison that is robust to textual
    /// differences in how the same address is spelled. A target that does not
    /// parse cannot match any allowlisted peer and is rejected when an allowlist
    /// is configured.
    pub(crate) fn decide(&self, decode_target: &str) -> DecodeTargetDecision {
        if self.peers.is_empty() {
            return DecodeTargetDecision::AllowUnchecked;
        }
        match decode_target.parse::<SocketAddr>() {
            Ok(addr) if self.peers.contains(&addr) => DecodeTargetDecision::Allow,
            _ => DecodeTargetDecision::Reject,
        }
    }
}

/// Build the decode-target allowlist from the dedicated `MLXCEL_DECODE_ALLOWLIST`
/// env input (issue #389).
///
/// `raw` is the comma-separated `host:port` list an operator sets to the FULL set
/// of router-selectable decode nodes (the shared cluster config). It is
/// independent of this node's `--decode-peers`, which stays the static handoff
/// fallback only. Each entry is trimmed; empty entries are skipped; an entry that
/// does not parse to a [`SocketAddr`] is skipped with a one-line warning rather
/// than failing startup. An empty or `None` input yields an empty allowlist,
/// which is the permissive-with-warning ([`DecodeTargetDecision::AllowUnchecked`])
/// path, so router-driven decode balancing is never silently broken when the
/// allowlist is left unconfigured.
pub(crate) fn decode_allowlist_from_env(raw: Option<&str>) -> DecodeAllowlist {
    let Some(raw) = raw else {
        return DecodeAllowlist::default();
    };
    let mut peers = Vec::new();
    for entry in raw.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        match entry.parse::<SocketAddr>() {
            Ok(addr) => peers.push(addr),
            Err(e) => {
                tracing::warn!(
                    "MLXCEL_DECODE_ALLOWLIST entry {entry:?} is not a valid host:port \
                     address ({e}); skipping it"
                );
            }
        }
    }
    DecodeAllowlist::from_peers(&peers)
}

/// Binds a serving role to a transport and its handoff peer.
///
/// Constructed by the startup serving-role branch (for a `PrefillOnly` or
/// `DecodeOnly` node) and, in tests, directly over a [`MockTransport`]. See the
/// module docs for the B2a scope boundary.
///
/// [`MockTransport`]: crate::distributed::mock_transport::MockTransport
pub struct ServingCoordinator {
    /// The serving mode this coordinator drives. A coordinator is only
    /// meaningful for a non-hybrid role (`PrefillOnly` / `DecodeOnly`); the
    /// `Hybrid` and `Router` modes never construct one (hybrid serves locally,
    /// router has no local KV to hand off).
    mode: ServingMode,

    /// The transport this node uses to move handoff frames. A
    /// [`MockTransport`] for the in-process B2/B2c path, a real network
    /// transport in B3.
    ///
    /// [`MockTransport`]: crate::distributed::mock_transport::MockTransport
    transport: Box<dyn Transport>,

    /// Address of the peer this node hands off to or receives from.
    ///
    /// For a prefill node this is the decode peer the extracted frame is sent
    /// to. For a decode node this is the prefill peer it is paired with (the
    /// inbound frame's sender is reported by the transport on receive, so a
    /// decode node does not need the peer address to accept a handoff, but it
    /// is retained for symmetric construction and the B3 control path).
    peer: String,

    /// Allowlist of decode addresses this prefill node will hand a KV cache off
    /// to (issue #389). Empty for the decode role, the legacy in-process role
    /// loops, and the unit-test coordinators; populated for the live prefill role
    /// from the dedicated `MLXCEL_DECODE_ALLOWLIST` full-pool env input so a
    /// router-chosen `decode_target` is validated before the prefill connects to
    /// it. Empty (env unset) means permissive-with-warning, never a hard reject.
    decode_allowlist: DecodeAllowlist,
}

impl std::fmt::Debug for ServingCoordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServingCoordinator")
            .field("mode", &self.mode)
            .field("peer", &self.peer)
            .field("decode_allowlist", &self.decode_allowlist)
            .finish()
    }
}

impl ServingCoordinator {
    /// Build a coordinator for `mode` over `transport`, handing off to / pairing
    /// with `peer`.
    pub fn new(mode: ServingMode, transport: Box<dyn Transport>, peer: impl Into<String>) -> Self {
        Self {
            mode,
            transport,
            peer: peer.into(),
            decode_allowlist: DecodeAllowlist::default(),
        }
    }

    /// Attach the decode-target allowlist the prefill role validates each
    /// router-chosen handoff target against before connecting (issue #389).
    pub(crate) fn with_decode_allowlist(mut self, allowlist: DecodeAllowlist) -> Self {
        self.decode_allowlist = allowlist;
        self
    }

    /// The serving mode this coordinator drives.
    pub fn mode(&self) -> ServingMode {
        self.mode
    }

    /// The address of the handoff peer.
    pub fn peer(&self) -> &str {
        &self.peer
    }

    /// Borrow the underlying transport (e.g. to inspect counters in a test or
    /// to drive a stream in B3).
    pub fn transport(&self) -> &dyn Transport {
        &*self.transport
    }

    /// Prefill side: send an extracted handoff frame to the decode peer.
    ///
    /// `payload` is a serialized cache state produced by
    /// [`super::handoff_impl::extract_sequence_handoff`] (the scheduler driver,
    /// B2b, produces it). The frame is tagged so the receiver rejects a
    /// mismatched message kind.
    pub async fn send_handoff(&self, payload: &[u8]) -> Result<()> {
        send_handoff_payload(&*self.transport, &self.peer, payload).await
    }

    /// Decode side: receive the next inbound handoff frame, returning the
    /// sender address and the raw serialized cache bytes.
    ///
    /// The bytes are handed to the scheduler driver's `ingest_sequence_handoff`
    /// (B2b), which validates and reconstructs them onto a fresh local pool
    /// slot. A non-handoff transport message is rejected by the underlying
    /// bridge rather than mis-restored.
    pub async fn recv_handoff(&self) -> Result<(String, Vec<u8>)> {
        recv_handoff_payload(&*self.transport).await
    }

    /// Prefill role loop (#126 B3a): drain prefill requests, prefill each on
    /// `scheduler`, and send the extracted handoff frame to the decode peer.
    ///
    /// Each request drives the standard full-prefill + extract entry
    /// ([`BatchScheduler::prefill_text_request_for_handoff`]); the resulting
    /// frame is shipped over the transport seam to the decode peer. A request
    /// that finishes at prefill (immediate EOS) produces no frame: the prefill
    /// node already emitted its terminal event on the request's own channel, so
    /// there is nothing to hand off. The loop returns when the request channel
    /// closes (a graceful drain).
    ///
    /// The scheduler is borrowed, not owned: a worker pairs its one scheduler
    /// with the role loop, and the coordinator stays model-free so the transport
    /// seam keeps its lightweight unit tests. The future is `!Send` (the
    /// scheduler holds the per-process MLX cache pool), so a caller drives it on
    /// a current-thread runtime (the worker thread, or a `LocalSet` in tests).
    ///
    /// `dead_code` is allowed until the worker flip wires the live caller (B3b);
    /// the in-process two-node parity test drives it now.
    #[allow(dead_code)]
    pub(crate) async fn run_prefill_role(
        &self,
        scheduler: &mut BatchScheduler,
        mut requests: tokio::sync::mpsc::Receiver<PrefillRoleRequest>,
    ) -> Result<()> {
        while let Some(req) = requests.recv().await {
            let frame = scheduler.prefill_text_request_for_handoff(
                req.prompt_tokens,
                req.sampling,
                req.max_tokens,
                req.response_tx,
                req.cancelled,
            )?;
            if let Some(bytes) = frame {
                self.send_handoff(&bytes)
                    .await
                    .context("prefill role loop: send handoff frame to the decode peer")?;
            }
        }
        Ok(())
    }

    /// Decode role loop (#126 B3a): receive handoff frames from the prefill peer,
    /// reconstruct each onto a fresh pool slot, and decode it to completion.
    ///
    /// Each iteration pairs the next inbound frame with the next coordination
    /// metadata (the per-request budget, sampling policy, and client output
    /// channel, which travel with the node holding the client connection rather
    /// than inside the KV frame) and drives
    /// [`BatchScheduler::ingest_handoff_as_active`] then
    /// [`BatchScheduler::decode_handoff_until_idle`], which streams the decode
    /// tokens out on the request's channel. The loop returns when the metadata
    /// channel closes.
    ///
    /// Metadata is paired with frames in FIFO arrival order in this in-process
    /// step; tagging frames with a request id for out-of-order delivery is a
    /// later step (the router in a real two-process deployment). The future is
    /// `!Send` for the same reason as [`Self::run_prefill_role`].
    ///
    /// `dead_code` is allowed until the worker flip wires the live caller (B3b);
    /// the in-process two-node parity test drives it now.
    #[allow(dead_code)]
    pub(crate) async fn run_decode_role(
        &self,
        scheduler: &mut BatchScheduler,
        mut handoffs: tokio::sync::mpsc::Receiver<DecodeRoleHandoff>,
    ) -> Result<()> {
        while let Some(meta) = handoffs.recv().await {
            let (_from, bytes) = self
                .recv_handoff()
                .await
                .context("decode role loop: receive handoff frame from the prefill peer")?;
            scheduler
                .ingest_handoff_as_active(&bytes, meta.max_tokens, meta.sampling, meta.response_tx)
                .context("decode role loop: ingest handoff frame as an active sequence")?;
            scheduler.decode_handoff_until_idle();
        }
        Ok(())
    }

    /// Live prefill role loop (#126 B3b2a): drive prefill from networked request
    /// frames and return results over the network.
    ///
    /// Unlike [`Self::run_prefill_role`] (driven by an in-process channel for the
    /// single-process parity test), this is the loop a real prefill worker runs:
    /// requests arrive as [`PrefillRequestFrame`] control frames over the
    /// transport, and the prefill node returns its first token to the request's
    /// `reply_to` and forwards the KV handoff (a [`DecodeMetaFrame`] followed by
    /// the KV frame) to the decode peer ([`Self::peer`]). A request that finishes
    /// at prefill (immediate EOS) produces no handoff: its first-token result is
    /// the whole stream and is marked done.
    ///
    /// The loop returns when the transport's inbound channel closes (shutdown).
    /// A non-request control frame is logged and skipped. The future is `!Send`
    /// (the scheduler holds the per-process MLX pool), so a caller drives it on a
    /// current-thread runtime ([`serve_prefill_role_networked_blocking`]).
    pub(crate) async fn run_prefill_role_networked(
        &self,
        scheduler: &mut BatchScheduler,
    ) -> Result<()> {
        loop {
            let message = match self.transport.recv().await {
                Ok((from, message)) => {
                    tracing::debug!(%from, "prefill role: received a transport frame");
                    message
                }
                // Inbound channel closed: the node is shutting down. A graceful
                // return rather than a hard error.
                Err(_) => break,
            };
            let (operation, payload) = match control_parts(message) {
                Ok(parts) => parts,
                Err(e) => {
                    tracing::warn!("prefill role: ignoring non-control frame: {e}");
                    continue;
                }
            };
            if operation != PrefillRequestFrame::OPERATION {
                tracing::warn!(
                    "prefill role: ignoring unexpected control op '{operation}' \
                     (expected '{}')",
                    PrefillRequestFrame::OPERATION
                );
                continue;
            }
            let request = PrefillRequestFrame::decode(&payload)
                .context("prefill role: decode prefill request frame")?;

            // Drive the standard full-prefill + extract entry, capturing the
            // first token on a local channel so it can be returned over the wire.
            let (token_tx, token_rx) = mpsc::channel();
            let frame = scheduler.prefill_text_request_for_handoff(
                request.prompt_tokens,
                sampling_from_serializable(&request.sampling),
                request.max_tokens as usize,
                token_tx,
                Arc::new(AtomicBool::new(false)),
            )?;
            let drained = drain_generation_events(&token_rx);

            // The prefill node's authoritative count of model tokens it
            // generated for this request (issue #387). When it finished here
            // (immediate EOS or a single-token budget) the terminal `Done`
            // reports the exact count (0 or 1); when it produced a handoff and
            // continues on the decode node, it sampled exactly one first token
            // and emitted no `Done`, so the count is 1. Carrying this lets the
            // router count a byte-fallback first token (which surfaces as no
            // text piece) correctly instead of under-counting it.
            let prefill_generated = drained.completion_tokens.unwrap_or(1);

            // Return the prefill node's first token to the client. When prefill
            // produced no handoff frame (immediate EOS), this first-token result
            // is the entire stream, so mark it done. A send failure here means
            // the router/client went away; log and move to the next request
            // rather than tearing down the worker loop that serves everyone.
            let first_result = ResultFrame {
                request_id: request.request_id,
                phase: ResultPhase::FirstToken,
                tokens: drained.tokens,
                start_sequence: 0,
                done: drained.done || frame.is_none(),
                error: drained.error,
                generated_tokens: Some(prefill_generated),
            };
            if let Err(e) = self
                .transport
                .send(&request.reply_to, first_result.encode()?)
                .await
            {
                tracing::warn!(
                    request_id = request.request_id,
                    reply_to = %request.reply_to,
                    "prefill role: failed to return the first token: {e}; dropping request"
                );
                continue;
            }

            // Forward the handoff to the decode node: the coordination metadata
            // first, then the KV frame (the decode loop reads them in that
            // order). The router picks the decode node and ships it in
            // `decode_target` (issue #201); a frame without it (an older router)
            // falls back to this node's statically configured `--decode-peers`
            // peer. A router-chosen target is validated against this node's
            // decode allowlist before connecting (issue #389): an off-list target
            // is rejected (the KV cache, which encodes the prompt, never leaves
            // for an unknown address) and the request fails cleanly. A handoff
            // failure (e.g. a dead decode node) fails only this one request: the
            // prefill loop logs it, tells the client so it does not wait for a
            // continuation that never comes, and keeps serving the rest. The
            // router's health monitor then marks the dead decode node down so
            // later requests route elsewhere.
            if let Some(bytes) = frame {
                // Resolve the decode handoff target. A router-chosen
                // `decode_target` is validated against the allowlist; an absent
                // target falls back to this node's own configured peer, which
                // needs no validation (operator config, not a wire address).
                let decode_peer = match request
                    .decode_target
                    .as_deref()
                    .filter(|addr| !addr.is_empty())
                {
                    Some(target) => match self.decode_allowlist.decide(target) {
                        DecodeTargetDecision::Allow => target.to_string(),
                        DecodeTargetDecision::AllowUnchecked => {
                            // Warn at most once per process: in the default multi-node
                            // config (MLXCEL_DECODE_ALLOWLIST unset, router sends
                            // decode_target) every request would otherwise emit a WARN
                            // and bury real warnings. The security posture is unchanged
                            // (still permissive; still forwards). Set
                            // MLXCEL_DECODE_ALLOWLIST to enforce.
                            if !ALLOWLIST_UNCONFIGURED_WARNED.swap(true, Ordering::Relaxed) {
                                tracing::warn!(
                                    "prefill role: MLXCEL_DECODE_ALLOWLIST is not configured; \
                                     decode_target validation is permissive and this node will \
                                     forward KV handoffs to any router-chosen address unchecked. \
                                     Set MLXCEL_DECODE_ALLOWLIST to the full pool of \
                                     router-selectable decode nodes (numeric IP:port, \
                                     comma-separated) to enforce the allowlist. This warning \
                                     appears once per process."
                                );
                            }
                            target.to_string()
                        }
                        DecodeTargetDecision::Reject => {
                            tracing::warn!(
                                request_id = request.request_id,
                                decode_target = %target,
                                "prefill role: rejecting KV handoff to a decode_target outside \
                                 the configured decode allowlist; failing the request"
                            );
                            let err_result = ResultFrame {
                                request_id: request.request_id,
                                phase: ResultPhase::Continuation,
                                tokens: Vec::new(),
                                start_sequence: 0,
                                done: true,
                                error: Some(format!(
                                    "decode_target {target} is not in this prefill node's \
                                     decode allowlist; rejecting the KV handoff"
                                )),
                                generated_tokens: None,
                            };
                            let _ = self
                                .transport
                                .send(&request.reply_to, err_result.encode()?)
                                .await;
                            continue;
                        }
                    },
                    None => self.peer().to_string(),
                };
                let meta = DecodeMetaFrame {
                    request_id: request.request_id,
                    max_tokens: request.max_tokens,
                    sampling: request.sampling,
                    reply_to: request.reply_to.clone(),
                };
                if let Err(e) = self.forward_handoff_to(&decode_peer, &meta, &bytes).await {
                    tracing::warn!(
                        request_id = request.request_id,
                        decode_peer = %decode_peer,
                        "prefill role: decode handoff failed: {e:#}; failing the request"
                    );
                    let err_result = ResultFrame {
                        request_id: request.request_id,
                        phase: ResultPhase::Continuation,
                        tokens: Vec::new(),
                        start_sequence: 0,
                        done: true,
                        error: Some(format!("decode handoff to {decode_peer} failed: {e}")),
                        generated_tokens: None,
                    };
                    let _ = self
                        .transport
                        .send(&request.reply_to, err_result.encode()?)
                        .await;
                }
            }
        }
        Ok(())
    }

    /// Forward a request's decode handoff to a specific decode node: the
    /// [`DecodeMetaFrame`] first, then the KV frame (the decode loop reads them
    /// in that order). `decode_peer` is the router-chosen target (issue #201) or
    /// this node's configured peer when the router did not specify one.
    async fn forward_handoff_to(
        &self,
        decode_peer: &str,
        meta: &DecodeMetaFrame,
        bytes: &[u8],
    ) -> Result<()> {
        self.transport
            .send(decode_peer, meta.encode()?)
            .await
            .context("prefill role: forward decode metadata to the decode node")?;
        send_handoff_payload(&*self.transport, decode_peer, bytes)
            .await
            .context("prefill role: ship the KV handoff to the decode node")?;
        Ok(())
    }

    /// Live decode role loop (#126 B3b2a): reconstruct networked handoffs and
    /// return the continuation over the network.
    ///
    /// The decode counterpart of [`Self::run_prefill_role_networked`]: each
    /// iteration reads the prefill node's [`DecodeMetaFrame`] then the KV frame,
    /// reconstructs the sequence on a fresh pool slot, decodes it to completion,
    /// and returns the continuation tokens to the metadata's `reply_to` (the
    /// client). The decode node has no fixed handoff peer: it replies to whatever
    /// address each request carries, so the merged client stream is the prefill
    /// node's first token plus this continuation.
    ///
    /// The loop returns when the transport's inbound channel closes. The future
    /// is `!Send` for the same reason as the prefill loop.
    pub(crate) async fn run_decode_role_networked(
        &self,
        scheduler: &mut BatchScheduler,
    ) -> Result<()> {
        loop {
            let message = match self.transport.recv().await {
                Ok((_from, message)) => message,
                Err(_) => break,
            };
            let (operation, payload) = match control_parts(message) {
                Ok(parts) => parts,
                Err(e) => {
                    tracing::warn!("decode role: ignoring non-control frame before KV: {e}");
                    continue;
                }
            };
            if operation != DecodeMetaFrame::OPERATION {
                tracing::warn!(
                    "decode role: ignoring unexpected control op '{operation}' \
                     (expected '{}')",
                    DecodeMetaFrame::OPERATION
                );
                continue;
            }
            let meta =
                DecodeMetaFrame::decode(&payload).context("decode role: decode metadata frame")?;

            // The KV handoff frame follows its metadata frame on the wire.
            let (_from, bytes) = self
                .recv_handoff()
                .await
                .context("decode role: receive the KV handoff after its metadata")?;

            let (token_tx, token_rx) = mpsc::channel();
            scheduler
                .ingest_handoff_as_active(
                    &bytes,
                    meta.max_tokens as usize,
                    sampling_from_serializable(&meta.sampling),
                    token_tx,
                )
                .context("decode role: ingest the handoff as an active sequence")?;

            // Stream the continuation incrementally (issue #199): after every
            // decode tick, drain the tokens that tick produced and ship them
            // as a non-terminal Continuation frame, so the client sees tokens
            // during decode instead of after the whole generation finishes.
            // The sends are awaited sequentially, so frames leave in order;
            // `start_sequence` tags each frame's first token (1-based, the
            // prefill first token is sequence 0) so the router can detect
            // gaps or reordering on the wire.
            let mut next_sequence: u64 = 1;
            // The decode node's authoritative count of model tokens it generated
            // after the handoff (issue #387), accumulated from whichever tick's
            // terminal `Done` event carries it. The decode `decode_state` is
            // seeded with the prompt and the prefill node's first token, so this
            // count covers only the decode continuation; the router adds the
            // prefill node's first-token count for the request total.
            let mut decode_generated: Option<u64> = None;
            loop {
                let active = scheduler.decode_handoff_step();
                let drained = drain_generation_events(&token_rx);
                if let Some(count) = drained.completion_tokens {
                    decode_generated = Some(count);
                }
                if !drained.tokens.is_empty() || drained.error.is_some() {
                    let frame_tokens = drained.tokens.len() as u64;
                    let result = ResultFrame {
                        request_id: meta.request_id,
                        phase: ResultPhase::Continuation,
                        tokens: drained.tokens,
                        start_sequence: next_sequence,
                        done: false,
                        error: drained.error,
                        // Only the terminal frame carries the count; intermediate
                        // continuation frames leave it None.
                        generated_tokens: None,
                    };
                    self.transport
                        .send(&meta.reply_to, result.encode()?)
                        .await
                        .context("decode role: stream a continuation frame to reply_to")?;
                    next_sequence += frame_tokens;
                }
                if !active {
                    break;
                }
            }

            // Terminal frame. The per-tick drain above runs after each
            // finalize_completed, so this drain is normally empty; it exists
            // as a defensive catch-all so no event can be dropped. The
            // authoritative decode count rides this terminal frame even when the
            // count's `Done` event arrived on an earlier tick.
            let drained = drain_generation_events(&token_rx);
            if let Some(count) = drained.completion_tokens {
                decode_generated = Some(count);
            }
            let result = ResultFrame {
                request_id: meta.request_id,
                phase: ResultPhase::Continuation,
                start_sequence: if drained.tokens.is_empty() {
                    0
                } else {
                    next_sequence
                },
                tokens: drained.tokens,
                done: true,
                error: drained.error,
                generated_tokens: decode_generated,
            };
            self.transport
                .send(&meta.reply_to, result.encode()?)
                .await
                .context("decode role: return the terminal continuation frame to reply_to")?;
        }
        Ok(())
    }
}

/// The result of draining a finished generation channel: the text pieces, the
/// terminal flag, any error, and the worker's authoritative generated-token
/// count when a terminal [`GenerateEvent::Done`] was emitted (issue #387).
struct DrainedGeneration {
    /// Detokenized text pieces, in generation order.
    tokens: Vec<String>,
    /// `true` when a terminal [`GenerateEvent::Done`] was seen.
    done: bool,
    /// A generation error message, if one was emitted.
    error: Option<String>,
    /// The worker's authoritative count of model tokens generated for this half
    /// of the stream, taken from the terminal [`GenerateEvent::Done`]'s
    /// [`GenerationResult::completion_tokens`]. `None` when this half ended
    /// without a `Done` event (a prefill node that handed off mid-stream returns
    /// its single first token without a `Done`).
    ///
    /// [`GenerationResult::completion_tokens`]: crate::server::model_provider::GenerationResult::completion_tokens
    completion_tokens: Option<u64>,
}

/// Drain a finished generation channel into its text pieces, terminal flag, any
/// error, and the authoritative generated-token count.
///
/// The serving-role scheduler entries
/// ([`BatchScheduler::prefill_text_request_for_handoff`] /
/// [`BatchScheduler::decode_handoff_until_idle`]) run synchronously to
/// completion and emit their [`GenerateEvent`]s on a local channel before
/// returning, so a non-blocking drain after the call collects the whole half of
/// the stream. The terminal `Done` event carries the worker's true model-token
/// count, which the caller forwards on the wire so the router can report exact
/// usage even for byte-fallback tokenizers (issue #387).
fn drain_generation_events(rx: &mpsc::Receiver<GenerateEvent>) -> DrainedGeneration {
    let mut tokens = Vec::new();
    let mut done = false;
    let mut error = None;
    let mut completion_tokens = None;
    while let Ok(event) = rx.try_recv() {
        match event {
            GenerateEvent::Token(t) | GenerateEvent::TokenWithLogprobs(t, _) => tokens.push(t),
            GenerateEvent::Done(result) => {
                done = true;
                completion_tokens = Some(result.completion_tokens as u64);
            }
            GenerateEvent::Error(e) => error = Some(e),
        }
    }
    DrainedGeneration {
        tokens,
        done,
        error,
        completion_tokens,
    }
}

/// A prefill-role work item the serving-role prefill loop turns into a handoff.
///
/// Carries a request's raw parts; the disaggregated path is text-only over the
/// pool-backed Fp16 families, so a request is fully described by its prompt
/// token ids, sampling policy, and per-request token budget. `response_tx` is
/// the channel the prefill node emits its first token on (the decode node emits
/// the continuation); `cancelled` is the client's cancellation flag, polled by
/// the scheduler to abort an orphaned sequence.
pub struct PrefillRoleRequest {
    /// The prompt token ids to prefill.
    pub prompt_tokens: Vec<i32>,
    /// Sampling policy for the request.
    pub sampling: SamplingConfig,
    /// Maximum tokens to generate (counted across the prefill first token and
    /// the decode continuation).
    pub max_tokens: usize,
    /// Output channel for the prefill node's half of the stream (the first
    /// sampled token).
    pub response_tx: std::sync::mpsc::Sender<GenerateEvent>,
    /// Client cancellation flag.
    pub cancelled: Arc<AtomicBool>,
}

/// The per-request coordination metadata a decode node pairs with an inbound
/// handoff frame.
///
/// The handoff frame carries the KV cache, the prompt token history, and the
/// prefill node's generated token(s). The request's budget, sampling policy,
/// and output stream stay with the node holding the client connection (the
/// router in a real deployment, the test harness in this in-process step), so
/// the decode loop supplies them alongside each frame.
pub struct DecodeRoleHandoff {
    /// Maximum tokens to generate (matches the originating request budget).
    pub max_tokens: usize,
    /// Sampling policy for the decode continuation.
    pub sampling: SamplingConfig,
    /// Output channel for the decode node's half of the stream (the
    /// continuation after the prefill node's first token).
    pub response_tx: std::sync::mpsc::Sender<GenerateEvent>,
}

/// Drive the prefill serving role to completion on a fresh current-thread tokio
/// runtime, binding the node's real TCP transport (#126 B3b1).
///
/// The model worker is a plain `std::thread` with no ambient tokio runtime, and
/// the role-loop future is `!Send` (the scheduler owns the per-process MLX
/// pool), so this builds a current-thread runtime and drives the loop on it via
/// `block_on` (a current-thread runtime accepts a `!Send` future; the transport
/// accept loop it spawns is `Send` and runs on the same runtime). `bind` is the
/// node's own listener config, `peer` the decode node it hands off to, and
/// `requests` the intake seam a networked request source feeds. When `ready` is
/// set, the bound local address is reported once the listener is up so a caller
/// that bound an ephemeral port can wire it as a peer (tests use this; the
/// worker passes `None`).
///
/// `dead_code` is allowed until the worker flip wires the live caller (B3b2);
/// the in-process two-node parity test drives it now.
#[allow(dead_code)]
pub(crate) fn serve_prefill_role_blocking(
    bind: TcpTransportConfig,
    peer: String,
    scheduler: &mut BatchScheduler,
    requests: tokio::sync::mpsc::Receiver<PrefillRoleRequest>,
    ready: Option<std::sync::mpsc::Sender<String>>,
) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("serve prefill role: build current-thread runtime")?;
    runtime.block_on(async move {
        let transport = TcpTransport::bind(bind)
            .await
            .context("serve prefill role: bind transport")?;
        if let Some(ready) = ready {
            let _ = ready.send(transport.local_addr()?);
        }
        let coordinator =
            ServingCoordinator::new(ServingMode::PrefillOnly, Box::new(transport), peer);
        coordinator.run_prefill_role(scheduler, requests).await
    })
}

/// Drive the decode serving role to completion on a fresh current-thread tokio
/// runtime, binding the node's real TCP transport (#126 B3b1).
///
/// The decode counterpart of [`serve_prefill_role_blocking`]: `peer` is the
/// prefill node it is paired with and `handoffs` the per-request coordination
/// metadata the decode loop pairs with inbound frames. See that function for the
/// runtime and `ready` rationale.
///
/// `dead_code` is allowed until the worker flip wires the live caller (B3b2);
/// the in-process two-node parity test drives it now.
#[allow(dead_code)]
pub(crate) fn serve_decode_role_blocking(
    bind: TcpTransportConfig,
    peer: String,
    scheduler: &mut BatchScheduler,
    handoffs: tokio::sync::mpsc::Receiver<DecodeRoleHandoff>,
    ready: Option<std::sync::mpsc::Sender<String>>,
) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("serve decode role: build current-thread runtime")?;
    runtime.block_on(async move {
        let transport = TcpTransport::bind(bind)
            .await
            .context("serve decode role: bind transport")?;
        if let Some(ready) = ready {
            let _ = ready.send(transport.local_addr()?);
        }
        let coordinator =
            ServingCoordinator::new(ServingMode::DecodeOnly, Box::new(transport), peer);
        coordinator.run_decode_role(scheduler, handoffs).await
    })
}

/// Drive the prefill serving role over a real network transport on a fresh
/// current-thread runtime, returning when the transport closes (#126 B3b2a).
///
/// The live counterpart of [`serve_prefill_role_blocking`]: instead of an
/// in-process request channel, prefill requests arrive as
/// [`PrefillRequestFrame`] control frames over the bound transport and results
/// are returned over the network, so a model worker can run this directly. `bind`
/// is the node's own role-transport listener, `decode_peers` the node's
/// configured decode peers, whose first entry is the static handoff fallback used
/// when the router does not pick a decode node (an older router). The allowlist a
/// router-chosen `decode_target` is validated against before this node connects to
/// it (issue #389) is read separately from the dedicated `MLXCEL_DECODE_ALLOWLIST`
/// env input (the full pool of router-selectable decode nodes); when that is unset
/// the prefill stays permissive-with-warning so balancing is never broken. When
/// `ready` is set the bound local address is reported once the listener is up
/// (tests learn an ephemeral port this way; the worker passes `None`).
///
/// [`PrefillRequestFrame`]: super::serving_protocol::PrefillRequestFrame
pub(crate) fn serve_prefill_role_networked_blocking(
    bind: TcpTransportConfig,
    decode_peers: Vec<SocketAddr>,
    scheduler: &mut BatchScheduler,
    ready: Option<std::sync::mpsc::Sender<String>>,
) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("serve prefill role: build current-thread runtime")?;
    runtime.block_on(async move {
        let transport = TcpTransport::bind(bind)
            .await
            .context("serve prefill role: bind transport")?;
        if let Some(ready) = ready {
            let _ = ready.send(transport.local_addr()?);
        }
        // The first configured decode peer is the static handoff fallback, used
        // when the router does not pick a decode node (issue #389).
        let static_peer = decode_peers
            .first()
            .map(|addr| addr.to_string())
            .unwrap_or_default();
        // The allowlist for router-chosen targets comes from the dedicated
        // MLXCEL_DECODE_ALLOWLIST full-pool env input, not from --decode-peers, so
        // it never rejects router-balanced targets. Unset stays permissive (issue
        // #389).
        let allowlist =
            decode_allowlist_from_env(std::env::var("MLXCEL_DECODE_ALLOWLIST").ok().as_deref());
        let coordinator =
            ServingCoordinator::new(ServingMode::PrefillOnly, Box::new(transport), static_peer)
                .with_decode_allowlist(allowlist);
        coordinator.run_prefill_role_networked(scheduler).await
    })
}

/// Drive the decode serving role over a real network transport on a fresh
/// current-thread runtime (#126 B3b2a).
///
/// The live counterpart of [`serve_decode_role_blocking`]: handoff frames (a
/// [`DecodeMetaFrame`] then the KV frame) arrive over the bound transport and the
/// continuation is returned to each request's `reply_to`. A decode node has no
/// fixed handoff peer (it replies per request), so the coordinator is built with
/// an empty peer. See [`serve_prefill_role_networked_blocking`] for the runtime
/// and `ready` rationale.
///
/// [`DecodeMetaFrame`]: super::serving_protocol::DecodeMetaFrame
pub(crate) fn serve_decode_role_networked_blocking(
    bind: TcpTransportConfig,
    scheduler: &mut BatchScheduler,
    ready: Option<std::sync::mpsc::Sender<String>>,
) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("serve decode role: build current-thread runtime")?;
    runtime.block_on(async move {
        let transport = TcpTransport::bind(bind)
            .await
            .context("serve decode role: bind transport")?;
        if let Some(ready) = ready {
            let _ = ready.send(transport.local_addr()?);
        }
        let coordinator =
            ServingCoordinator::new(ServingMode::DecodeOnly, Box::new(transport), String::new());
        coordinator.run_decode_role_networked(scheduler).await
    })
}

#[cfg(test)]
#[path = "coordinator_tests.rs"]
mod tests;
