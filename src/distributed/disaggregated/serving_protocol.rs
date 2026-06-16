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

//! Control-frame wire protocol for live disaggregated serving (#126 B3b2a).
//!
//! The KV cache itself rides a [`TransportMessage::TensorData`] handoff frame
//! (see [`super::handoff_impl`]). This module adds the small JSON control frames
//! that surround it so a request can flow client -> prefill node -> decode node
//! -> client across real processes:
//!
//! 1. [`PrefillRequestFrame`] (`serve.prefill_request`): the client (a router,
//!    or the test harness standing in for one) sends a request's prompt token
//!    ids, sampling policy, token budget, and a `reply_to` address to a prefill
//!    node.
//! 2. [`DecodeMetaFrame`] (`serve.decode_meta`): the prefill node forwards the
//!    per-request coordination metadata (budget, sampling, `reply_to`) to the
//!    decode node, sent immediately before the KV handoff frame so the decode
//!    node knows how to continue the sequence and where to return its tokens.
//! 3. [`ResultFrame`] (`serve.result`): a node returns generated tokens to the
//!    request's `reply_to`. The prefill node returns its
//!    [`ResultPhase::FirstToken`]; the decode node returns the
//!    [`ResultPhase::Continuation`]. The client concatenates the two halves, the
//!    same split a [`StreamBridge`] merges in a full router deployment (B3b2b).
//!
//! All three are carried as [`TransportMessage::Control`] with a JSON payload,
//! matching the existing convention of riding cache metadata inside JSON (the KV
//! tensor bytes stay on the binary `TensorData` frame). The control payloads are
//! tiny (token ids and sampling scalars), so JSON keeps them debuggable without
//! a measurable cost.
//!
//! [`TransportMessage::TensorData`]: crate::distributed::transport::TransportMessage::TensorData
//! [`TransportMessage::Control`]: crate::distributed::transport::TransportMessage::Control
//! [`StreamBridge`]: super::stream_bridge::StreamBridge

use anyhow::{Result, bail};
use bytes::Bytes;
use serde::{Deserialize, Serialize};

use mlxcel_core::generate::SamplingConfig;

use crate::distributed::kv_cache_serde::SerializableSamplingState;
use crate::distributed::transport::TransportMessage;

/// Control operation tag for a [`PrefillRequestFrame`].
pub const OP_PREFILL_REQUEST: &str = "serve.prefill_request";
/// Control operation tag for a [`DecodeMetaFrame`].
pub const OP_DECODE_META: &str = "serve.decode_meta";
/// Control operation tag for a [`ResultFrame`].
pub const OP_RESULT: &str = "serve.result";

/// A prefill-role work request a client (router) sends to a prefill node.
///
/// The disaggregated path is text-only over the pool-backed Fp16 families, so a
/// request is fully described by its prompt token ids, sampling policy, and
/// per-request token budget. `reply_to` is the listener address the node returns
/// its [`ResultFrame`]s to (the router, or the test harness standing in for one),
/// and `request_id` correlates the frames that belong to one request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefillRequestFrame {
    /// Correlates the request's frames (decode meta, results) across nodes.
    pub request_id: u64,
    /// Prompt token ids to prefill.
    pub prompt_tokens: Vec<i32>,
    /// Sampling policy for the request (mirrors the decode-relevant
    /// [`SamplingConfig`] fields; see [`sampling_to_serializable`]).
    pub sampling: SerializableSamplingState,
    /// Maximum tokens to generate (counted across the prefill first token and
    /// the decode continuation).
    pub max_tokens: u64,
    /// Listener address (`host:port`) the nodes return [`ResultFrame`]s to.
    pub reply_to: String,
}

/// The per-request coordination metadata a prefill node forwards to a decode
/// node, sent immediately before the KV handoff frame.
///
/// The KV handoff frame carries the cache, the prompt token history, and the
/// prefill node's first generated token. The request's budget, sampling policy,
/// and `reply_to` travel with the node holding the client connection, so the
/// prefill node forwards them here for the decode node to continue with.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodeMetaFrame {
    /// Correlates with the originating [`PrefillRequestFrame::request_id`].
    pub request_id: u64,
    /// Maximum tokens to generate (matches the originating request budget).
    pub max_tokens: u64,
    /// Sampling policy for the decode continuation.
    pub sampling: SerializableSamplingState,
    /// Listener address the decode node returns its [`ResultFrame`] to.
    pub reply_to: String,
}

/// Which half of the generated stream a [`ResultFrame`] carries.
///
/// A disaggregated request's output is split: the prefill node samples the first
/// token, the decode node generates the continuation. The client orders the two
/// halves by phase before concatenating, the same split a [`StreamBridge`] merges
/// in a full router deployment.
///
/// [`StreamBridge`]: super::stream_bridge::StreamBridge
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResultPhase {
    /// The prefill node's first sampled token.
    FirstToken,
    /// The decode node's continuation (tokens after the first), ending the
    /// stream.
    Continuation,
}

/// Generated tokens a node returns to a request's `reply_to`.
///
/// `tokens` are the detokenized text pieces in order; `done` marks the terminal
/// frame of the request (the decode node sets it on the continuation); `error`
/// carries a generation error message instead of tokens when one occurred.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultFrame {
    /// Correlates with the originating [`PrefillRequestFrame::request_id`].
    pub request_id: u64,
    /// Which half of the stream this frame carries.
    pub phase: ResultPhase,
    /// Detokenized text pieces, in generation order.
    pub tokens: Vec<String>,
    /// Stream sequence number of the FIRST token in this frame (issue #199).
    ///
    /// The decode role streams its continuation incrementally as multiple
    /// frames; the router uses this tag to detect gaps or reordering across
    /// the transport. Numbering starts at 1 for the first continuation token
    /// (sequence 0 is the prefill first token). `0` on frames that carry no
    /// continuation position (the first-token result, terminal/error frames
    /// without tokens, and frames from older senders via `serde(default)`),
    /// which the receiver treats as "unchecked".
    #[serde(default)]
    pub start_sequence: u64,
    /// `true` on the terminal frame of the request.
    pub done: bool,
    /// A generation error message, if the node hit one.
    pub error: Option<String>,
}

macro_rules! json_control_frame {
    ($ty:ty, $op:expr) => {
        impl $ty {
            /// The control operation tag this frame is carried under.
            pub const OPERATION: &'static str = $op;

            /// Encode this frame as a [`TransportMessage::Control`] ready to send.
            pub fn encode(&self) -> Result<TransportMessage> {
                let payload = serde_json::to_vec(self)?;
                Ok(TransportMessage::Control {
                    operation: Self::OPERATION.to_string(),
                    payload: Bytes::from(payload),
                })
            }

            /// Decode this frame from a control payload.
            pub fn decode(payload: &[u8]) -> Result<Self> {
                Ok(serde_json::from_slice(payload)?)
            }
        }
    };
}

json_control_frame!(PrefillRequestFrame, OP_PREFILL_REQUEST);
json_control_frame!(DecodeMetaFrame, OP_DECODE_META);
json_control_frame!(ResultFrame, OP_RESULT);

/// Split a [`TransportMessage::Control`] into its operation tag and payload,
/// rejecting a [`TransportMessage::TensorData`] frame (which belongs on the KV
/// handoff path, not the control path).
pub fn control_parts(message: TransportMessage) -> Result<(String, Bytes)> {
    match message {
        TransportMessage::Control { operation, payload } => Ok((operation, payload)),
        TransportMessage::TensorData { tensor_id, .. } => {
            bail!("expected a control frame, got TensorData('{tensor_id}')")
        }
    }
}

/// Copy the decode-relevant fields of a live [`SamplingConfig`] into the
/// serializable mirror carried on the wire.
///
/// The server-wide `token_bias` map is intentionally dropped: it is a node-local
/// policy applied at sampling time, not a per-request handoff field, and both
/// nodes resolve their own.
pub fn sampling_to_serializable(config: &SamplingConfig) -> SerializableSamplingState {
    SerializableSamplingState {
        temperature: config.temperature,
        top_k: config.top_k,
        top_p: config.top_p,
        min_p: config.min_p,
        seed: config.seed,
        repetition_penalty: config.repetition_penalty,
        dry_multiplier: config.dry_multiplier,
        dry_base: config.dry_base,
        dry_allowed_length: config.dry_allowed_length,
        dry_penalty_last_n: config.dry_penalty_last_n,
        dry_sequence_breakers: config.dry_sequence_breakers.clone(),
        frequency_penalty: config.frequency_penalty,
        presence_penalty: config.presence_penalty,
        stop_token_ids: config.stop_token_ids.clone(),
    }
}

/// Reconstruct a live [`SamplingConfig`] from the serializable wire mirror.
///
/// The inverse of [`sampling_to_serializable`]; `token_bias` is left at its
/// default (the receiving node applies its own server-wide bias, if any).
pub fn sampling_from_serializable(state: &SerializableSamplingState) -> SamplingConfig {
    SamplingConfig {
        temperature: state.temperature,
        top_k: state.top_k,
        top_p: state.top_p,
        min_p: state.min_p,
        seed: state.seed,
        repetition_penalty: state.repetition_penalty,
        dry_multiplier: state.dry_multiplier,
        dry_base: state.dry_base,
        dry_allowed_length: state.dry_allowed_length,
        dry_penalty_last_n: state.dry_penalty_last_n,
        dry_sequence_breakers: state.dry_sequence_breakers.clone(),
        frequency_penalty: state.frequency_penalty,
        presence_penalty: state.presence_penalty,
        stop_token_ids: state.stop_token_ids.clone(),
        token_bias: Default::default(),
    }
}

#[cfg(test)]
#[path = "serving_protocol_tests.rs"]
mod tests;
