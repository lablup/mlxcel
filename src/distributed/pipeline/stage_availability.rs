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

//! The remote pipeline coordinator's availability gate (#2782 item 1).
//!
//! Every pipeline request needs every stage, so one stage that stops
//! answering (a stopped process, a partition, an on-path party dropping
//! frames, none of which needs the cluster secret) made each new request
//! block the single model worker for a full `stage_timeout` (30 s) before
//! failing. The gate remembers that a stage did not answer within the
//! stage timeout, or could not be sent a command (#2790), and refuses new
//! requests at once while the stage stays silent. It is not a fixed
//! cooldown: one dropped reply must not refuse the next 10 s of requests,
//! so the first request after a timeout probes every stage with a short
//! wait and is served normally when they all answer. Requests inside the
//! probe interval are refused without sending anything.
//!
//! Vocabulary: the gate is *open* while it refuses requests (a stage is
//! known silent) and *closed* while requests flow, like a circuit breaker.
//! This module holds the pure state machine; `RemotePipelineRuntime` does
//! the I/O around it and owns the locking (see
//! `RemotePipelineRuntime::ensure_stages_available`).
//!
//! Used by: `pipeline::runtime` (the gate), `pipeline::server_runtime`
//! (logs a [`PipelineUnavailable`] refusal at debug).

use std::time::{Duration, Instant};

/// How long the gate stays open between probes once a stage went silent.
/// While a stage is down, new requests cost at most one short probe wait
/// per interval instead of one `stage_timeout` each.
pub(crate) const STAGE_PROBE_INTERVAL: Duration = Duration::from_secs(10);

/// The fast-fail error of a request refused while a stage is silent.
/// Carried in `anyhow` so callers can `downcast_ref` it: the server model
/// logs it at debug because the gate's own open and close lines carry the
/// operator signal.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub struct PipelineUnavailable {
    /// The stage whose exchange failed (or whose probe went unanswered);
    /// `None` after a pipeline run timed out and before a probe named the
    /// silent stage (#2790).
    pub stage: Option<String>,
    /// How long the gate has been open.
    pub since: Duration,
}

impl std::fmt::Display for PipelineUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let since = self.since;
        match &self.stage {
            Some(stage) => write!(f, "remote pipeline stage {stage} has not answered"),
            None => write!(
                f,
                "an unidentified stage of the remote pipeline has not answered"
            ),
        }?;
        write!(
            f,
            " for {since:?}; refusing the request until a probe reaches every stage"
        )
    }
}

/// `stage` for a log line: its address, or "an unidentified stage" while a
/// pipeline run timeout has not been attributed yet.
pub(crate) fn stage_label(stage: Option<&str>) -> &str {
    stage.unwrap_or("an unidentified stage")
}

/// What the gate decided for one request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GateDecision {
    /// The gate is closed: serve the request.
    Pass,
    /// The gate is open and no probe ran in the last probe interval: the
    /// caller probes every stage, then reports with
    /// [`StageAvailability::close_after_probe`] or
    /// [`StageAvailability::probe_failed`]. `generation` identifies the
    /// outage the probe is for and `stage` the stage that went silent,
    /// `None` when a pipeline run timed out without naming one.
    Probe {
        generation: u64,
        stage: Option<String>,
    },
    /// The gate is open and was probed recently: refuse without sending.
    Refuse(PipelineUnavailable),
}

/// How [`StageAvailability::close_after_probe`] applied a probe in which
/// every stage answered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ProbeClose {
    /// This probe closed the gate after an outage of the given length.
    Closed(Duration),
    /// A concurrent probe for the same outage already closed the gate and
    /// nothing failed since: serve the request, log nothing.
    AlreadyClosed,
    /// A newer timeout or failed probe replaced the outage this probe was
    /// for: the gate stays as it is and the request is refused.
    Superseded,
}

/// How [`StageAvailability::probe_failed`] applied a probe that a stage did
/// not answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProbeFailure {
    /// The refusal for the request that probed.
    pub refusal: PipelineUnavailable,
    /// `true` when this failure reopened a gate that a concurrent probe for
    /// the same outage had closed; the caller logs the open line again.
    pub reopened: bool,
}

/// Whether the stages are believed reachable, and when they were last
/// probed. One instance per runtime, behind a mutex.
#[derive(Debug, Default)]
pub(crate) struct StageAvailability {
    /// `Some` while the gate is open: when the current outage began.
    unavailable_since: Option<Instant>,
    /// The stage that last failed to answer. `Some` only while open, and
    /// `None` while open after a pipeline run timed out until a probe
    /// names the silent stage (#2790).
    failed_peer: Option<String>,
    /// When the last probe of the current outage was sent.
    last_probe: Option<Instant>,
    /// Bumped on every new timeout and failed probe, so a probe result can
    /// tell whether the outage it probed is still the current one.
    generation: u64,
}

impl StageAvailability {
    /// Record that a full-`stage_timeout` exchange failed because of a
    /// stage: `peer`, or `None` when the failure cannot name one (a
    /// pipeline run spans every stage, #2790). Returns `true` on the
    /// closed-to-open transition (the caller logs it once). An outage
    /// already open keeps its start time, so the outage length stays
    /// honest, but the next request probes again.
    pub(crate) fn mark_unavailable(&mut self, peer: Option<&str>, now: Instant) -> bool {
        let transition = self.unavailable_since.is_none();
        if transition {
            self.unavailable_since = Some(now);
        }
        self.failed_peer = peer.map(str::to_string);
        self.last_probe = None;
        self.generation = self.generation.wrapping_add(1);
        transition
    }

    /// Decide for a request arriving at `now`. A `Probe` decision records
    /// the probe time before the probe is sent, so concurrent requests see
    /// a recent probe and are refused instead of each probing too.
    pub(crate) fn decide(&mut self, now: Instant, probe_interval: Duration) -> GateDecision {
        let Some(since) = self.unavailable_since else {
            return GateDecision::Pass;
        };
        let stage = self.failed_peer.clone();
        let probe_due = self
            .last_probe
            .is_none_or(|last| now.saturating_duration_since(last) >= probe_interval);
        if probe_due {
            self.last_probe = Some(now);
            GateDecision::Probe {
                generation: self.generation,
                stage,
            }
        } else {
            GateDecision::Refuse(PipelineUnavailable {
                stage,
                since: now.saturating_duration_since(since),
            })
        }
    }

    /// Every stage answered the probe issued for `generation`. Probes can
    /// overlap (one that outlasts the probe interval lets the next request
    /// probe too), so the result depends on what happened meanwhile; see
    /// [`ProbeClose`]. A close does not bump the generation, so a sibling
    /// probe of the same outage that also succeeded is told
    /// [`ProbeClose::AlreadyClosed`] instead of being refused.
    pub(crate) fn close_after_probe(&mut self, generation: u64, now: Instant) -> ProbeClose {
        if self.generation != generation {
            return ProbeClose::Superseded;
        }
        let Some(since) = self.unavailable_since.take() else {
            return ProbeClose::AlreadyClosed;
        };
        self.failed_peer = None;
        self.last_probe = None;
        ProbeClose::Closed(now.saturating_duration_since(since))
    }

    /// The probe issued for `generation` reached `stage` and it did not
    /// answer.
    ///
    /// - Same generation, gate open: `stage` is the silent stage now, and
    ///   the generation moves on so a still-running sibling probe cannot
    ///   close the gate on this outage.
    /// - Same generation, gate closed: a sibling probe closed it while this
    ///   one was finding `stage` silent. The newer observation wins: the
    ///   gate reopens as a fresh outage and `reopened` asks the caller to
    ///   log the open line, so the log never shows a closed gate that is
    ///   in fact refusing requests.
    /// - Older generation: a newer timeout or failed probe already
    ///   describes the current state (open, or closed by a newer probe that
    ///   succeeded); it is left alone and only this request is refused.
    pub(crate) fn probe_failed(
        &mut self,
        generation: u64,
        stage: &str,
        now: Instant,
    ) -> ProbeFailure {
        if self.generation != generation {
            let refusal = if self.unavailable_since.is_some() {
                self.refusal(now)
            } else {
                PipelineUnavailable {
                    stage: Some(stage.to_string()),
                    since: Duration::ZERO,
                }
            };
            return ProbeFailure {
                refusal,
                reopened: false,
            };
        }
        let reopened = self.unavailable_since.is_none();
        if reopened {
            self.unavailable_since = Some(now);
            self.last_probe = Some(now);
        }
        self.failed_peer = Some(stage.to_string());
        self.generation = self.generation.wrapping_add(1);
        ProbeFailure {
            refusal: self.refusal(now),
            reopened,
        }
    }

    /// The current refusal. While closed (which no caller asks for), or
    /// open with no stage named yet, the stage is `None`; while closed the
    /// outage length is zero.
    pub(crate) fn refusal(&self, now: Instant) -> PipelineUnavailable {
        PipelineUnavailable {
            stage: self.failed_peer.clone(),
            since: self
                .unavailable_since
                .map_or(Duration::ZERO, |since| now.saturating_duration_since(since)),
        }
    }

    /// The silent stage while the gate is open and one is named, `None`
    /// while closed or while a pipeline run timeout is unattributed.
    pub(crate) fn failed_peer(&self) -> Option<&str> {
        self.failed_peer.as_deref()
    }
}

#[cfg(test)]
#[path = "stage_availability_tests.rs"]
mod tests;
