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

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::{Context, Result};
use mlxcel_core::generate::SamplingConfig;

use super::handoff_impl::{recv_handoff_payload, send_handoff_payload};
use super::serving::ServingMode;
use crate::distributed::transport::Transport;
use crate::server::batch::BatchScheduler;
use crate::server::model_provider::GenerateEvent;

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
}

impl std::fmt::Debug for ServingCoordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServingCoordinator")
            .field("mode", &self.mode)
            .field("peer", &self.peer)
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
        }
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

#[cfg(test)]
#[path = "coordinator_tests.rs"]
mod tests;
