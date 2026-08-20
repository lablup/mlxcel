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

//! Adaptive MTP policy read endpoint (issue #1257).
//!
//! `GET /v1/internal/mtp-policy` reports the adaptive B=1 MTP policy's state
//! for the pairing this server is running: whether the verdict has settled,
//! which way it settled, and how far a profiling window has to go.
//!
//! ## Why this exists
//!
//! The verdict already lived in a per-pairing hint file under
//! `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/mtp-policy/`, and a host
//! application that wanted to show it had no choice but to read that private
//! on-disk format from another process. That coupling breaks silently: the
//! hint format is versioned and free to change in a patch release, and a
//! consumer that stops recognising it degrades to a blank surface neither
//! project detects. This endpoint is the supported alternative, served by the
//! process that owns the decision, over the transport the consumer already
//! uses. The hint files are unchanged and keep working.
//!
//! ## Versioning and compatibility
//!
//! The body carries [`MTP_POLICY_SCHEMA_VERSION`]. Within one schema version
//! the contract is: existing fields keep their names, types, and meanings, and
//! the `state`, `reason`, and `verdict` label sets only grow. New fields may
//! be added, so a consumer must ignore unknown fields and must treat an
//! unrecognised label as "no verdict I can render" rather than as an error.
//! Anything that breaks those promises (a removed or renamed field, a changed
//! meaning, a narrowed label set) bumps `schema_version`. See
//! `docs/mtp-policy-api.md` for the full contract.
//!
//! The endpoint is always mounted, and answers with a well-formed body even
//! when no policy is running, so a consumer polls unconditionally. Gating it
//! behind a flag would reproduce the very failure it exists to remove: a
//! consumer that cannot tell "nothing to report" from "this server does not
//! answer".

use axum::{Json, extract::State};
use serde::{Deserialize, Serialize};

use crate::server::AppState;
use crate::server::batch::MtpPolicySnapshot;

/// Wire schema version of the `GET /v1/internal/mtp-policy` body.
///
/// Bumped only for a breaking change; see the module docs and
/// `docs/mtp-policy-api.md`.
pub const MTP_POLICY_SCHEMA_VERSION: u32 = 1;

/// Body of `GET /v1/internal/mtp-policy`.
///
/// Every field is present in every state; the ones that do not apply are
/// `null`. That keeps a consumer's parse unconditional, and makes "the policy
/// has nothing to report" a value rather than an absence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MtpPolicyResponse {
    /// See [`MTP_POLICY_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// `"settled"`, `"profiling"`, `"forced"`, or `"unavailable"`.
    ///
    /// `"forced"` is not `"settled"`: it means `MLXCEL_ENABLE_MTP_B1` pinned
    /// the decision, so nothing was measured on this machine and rendering it
    /// as a measured verdict would be a lie.
    pub state: String,
    /// Why the policy is unavailable: `"no_mtp_dispatch"`,
    /// `"adaptive_disabled"`, or `"worker_not_ready"`. `null` in every other
    /// state.
    pub reason: Option<String>,
    /// `"enable"` or `"decline"` once settled, `null` otherwise. Matches the
    /// `verdict` values in the persisted hint file.
    pub verdict: Option<String>,
    /// Whether the B=1 MTP burst runs right now. `true` while profiling, since
    /// profiling forces MTP on to collect the sample, so this is the live gate
    /// and not the verdict.
    pub mtp_enabled: Option<bool>,
    /// Served model directory basename.
    pub target: Option<String>,
    /// Draft model directory basename.
    pub drafter: Option<String>,
    /// Coarse hardware-class label, e.g. `"M5-16c"`.
    pub hardware: Option<String>,
    /// Draft block size (K) the pairing is keyed on. A consumer showing a
    /// verdict should confirm this matches the K it is showing, because a
    /// verdict profiled at one K does not carry to another.
    pub block_size: Option<u32>,
    /// Coarse measured acceptance rate: running while profiling, final once
    /// settled, `null` when nothing was measured.
    pub acceptance_rate: Option<f64>,
    /// Qualifying samples accumulated so far, or behind the settled verdict.
    /// Zero when forced or unavailable.
    pub samples: usize,
    /// Qualifying samples a profiling window needs before it settles.
    pub samples_required: usize,
    /// Qualifying samples still needed, present only while profiling.
    pub samples_remaining: Option<usize>,
}

/// `GET /v1/internal/mtp-policy` - report the adaptive MTP policy state.
pub async fn mtp_policy(State(state): State<AppState>) -> Json<MtpPolicyResponse> {
    Json(build_mtp_policy_response(
        state.batch_observability.mtp_policy_snapshot(),
    ))
}

/// Pure helper: project a published snapshot onto the wire body.
///
/// Extracted so route-level tests can drive the mapping without constructing a
/// full [`AppState`] (which would require loading a real model), mirroring
/// `cache::build_stats_response`.
///
/// `None` means no worker has published anything yet, which is reported as
/// `unavailable` / `worker_not_ready` rather than as an error: a server whose
/// model is still loading has a real answer to give, and a 404 or a 500 here
/// would be indistinguishable from a server that does not implement the
/// endpoint at all.
pub(crate) fn build_mtp_policy_response(snapshot: Option<MtpPolicySnapshot>) -> MtpPolicyResponse {
    use crate::server::batch::MtpPolicyUnavailableReason;

    let snapshot = snapshot.unwrap_or_else(|| {
        MtpPolicySnapshot::unavailable(MtpPolicyUnavailableReason::WorkerNotReady, None)
    });
    let samples_remaining = snapshot.samples_remaining();
    MtpPolicyResponse {
        schema_version: MTP_POLICY_SCHEMA_VERSION,
        state: snapshot.status.as_str().to_string(),
        reason: snapshot.reason.map(|r| r.as_str().to_string()),
        verdict: snapshot
            .verdict
            .map(|v| if v.runs() { "enable" } else { "decline" }.to_string()),
        mtp_enabled: snapshot.mtp_enabled,
        target: snapshot.target,
        drafter: snapshot.drafter,
        hardware: snapshot.hardware,
        block_size: snapshot.block_size,
        acceptance_rate: snapshot.acceptance_rate,
        samples: snapshot.samples,
        samples_required: snapshot.samples_required,
        samples_remaining,
    }
}

#[cfg(test)]
#[path = "mtp_policy_tests.rs"]
mod tests;
