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

use anyhow::Result;

use super::handoff_impl::{recv_handoff_payload, send_handoff_payload};
use super::serving::ServingMode;
use crate::distributed::transport::Transport;

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
}

#[cfg(test)]
#[path = "coordinator_tests.rs"]
mod tests;
