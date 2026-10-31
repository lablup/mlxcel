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
//! failing. The gate remembers that an exchange timed out and refuses new
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
#[error(
    "remote pipeline stage {stage} has not answered for {since:?}; refusing the request until a \
     probe reaches every stage"
)]
pub struct PipelineUnavailable {
    /// The stage whose exchange timed out (or whose probe went unanswered).
    pub stage: String,
    /// How long the gate has been open.
    pub since: Duration,
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
    /// outage the probe is for and `stage` the stage that went silent.
    Probe { generation: u64, stage: String },
    /// The gate is open and was probed recently: refuse without sending.
    Refuse(PipelineUnavailable),
}

/// Whether the stages are believed reachable, and when they were last
/// probed. One instance per runtime, behind a mutex.
#[derive(Debug, Default)]
pub(crate) struct StageAvailability {
    /// `Some` while the gate is open: when the current outage began.
    unavailable_since: Option<Instant>,
    /// The stage that last failed to answer; `Some` exactly while open.
    failed_peer: Option<String>,
    /// When the last probe of the current outage was sent.
    last_probe: Option<Instant>,
    /// Bumped on every new timeout and failed probe, so a probe result can
    /// tell whether the outage it probed is still the current one.
    generation: u64,
}

impl StageAvailability {
    /// Record that a full-`stage_timeout` exchange with `peer` failed.
    /// Returns `true` on the closed-to-open transition (the caller logs it
    /// once). An outage already open keeps its start time, so the outage
    /// length stays honest, but the next request probes again.
    pub(crate) fn mark_unavailable(&mut self, peer: &str, now: Instant) -> bool {
        let transition = self.unavailable_since.is_none();
        if transition {
            self.unavailable_since = Some(now);
        }
        self.failed_peer = Some(peer.to_string());
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
        let stage = self.failed_peer.clone().unwrap_or_default();
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

    /// Every stage answered the probe issued for `generation`: close the
    /// gate and return the outage length. `None` when a newer timeout or
    /// failed probe replaced that outage while the probe was in flight; the
    /// gate stays open and the caller refuses the request.
    pub(crate) fn close_after_probe(&mut self, generation: u64, now: Instant) -> Option<Duration> {
        if self.generation != generation {
            return None;
        }
        let since = self.unavailable_since.take()?;
        self.failed_peer = None;
        self.last_probe = None;
        Some(now.saturating_duration_since(since))
    }

    /// The probe reached `stage` and it did not answer: it is the silent
    /// stage now. Returns the refusal for the request that probed.
    pub(crate) fn probe_failed(&mut self, stage: &str, now: Instant) -> PipelineUnavailable {
        if self.unavailable_since.is_none() {
            // Opened by the probe itself: an exchange failed between the
            // decision and the probe and was closed meanwhile. Treat it as
            // a fresh outage.
            self.unavailable_since = Some(now);
        }
        self.failed_peer = Some(stage.to_string());
        self.generation = self.generation.wrapping_add(1);
        self.refusal(now)
    }

    /// The current refusal. While closed (which no caller asks for) the
    /// stage is empty and the outage length zero.
    pub(crate) fn refusal(&self, now: Instant) -> PipelineUnavailable {
        PipelineUnavailable {
            stage: self.failed_peer.clone().unwrap_or_default(),
            since: self
                .unavailable_since
                .map_or(Duration::ZERO, |since| now.saturating_duration_since(since)),
        }
    }

    /// The silent stage while the gate is open, `None` while closed.
    pub(crate) fn failed_peer(&self) -> Option<&str> {
        self.failed_peer.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INTERVAL: Duration = Duration::from_secs(10);

    #[test]
    fn closed_gate_passes_every_request() {
        let mut gate = StageAvailability::default();
        let now = Instant::now();
        assert_eq!(gate.decide(now, INTERVAL), GateDecision::Pass);
        assert_eq!(gate.failed_peer(), None);
    }

    #[test]
    fn first_request_after_a_timeout_probes_and_the_next_is_refused() {
        let mut gate = StageAvailability::default();
        let t0 = Instant::now();
        assert!(gate.mark_unavailable("stage-a", t0), "closed to open");
        assert!(
            !gate.mark_unavailable("stage-a", t0 + Duration::from_millis(1)),
            "a second timeout is not a transition"
        );
        assert_eq!(gate.failed_peer(), Some("stage-a"));

        let t1 = t0 + Duration::from_secs(1);
        let GateDecision::Probe { generation, stage } = gate.decide(t1, INTERVAL) else {
            panic!("the first request after a timeout probes");
        };
        assert_eq!(stage, "stage-a");

        // Inside the interval: refused, with the outage length so far.
        match gate.decide(t1 + Duration::from_secs(2), INTERVAL) {
            GateDecision::Refuse(err) => {
                assert_eq!(err.stage, "stage-a");
                assert_eq!(err.since, Duration::from_secs(3));
            }
            other => panic!("expected a refusal, got {other:?}"),
        }

        // The probe succeeded: closed, with the outage length.
        let t2 = t1 + Duration::from_secs(4);
        assert_eq!(
            gate.close_after_probe(generation, t2),
            Some(Duration::from_secs(5))
        );
        assert_eq!(gate.decide(t2, INTERVAL), GateDecision::Pass);
        assert_eq!(gate.failed_peer(), None);
    }

    #[test]
    fn probe_is_due_again_after_the_interval_and_a_failed_probe_keeps_it_open() {
        let mut gate = StageAvailability::default();
        let t0 = Instant::now();
        gate.mark_unavailable("stage-a", t0);
        let GateDecision::Probe { .. } = gate.decide(t0, INTERVAL) else {
            panic!("probe");
        };
        let refusal = gate.probe_failed("stage-b", t0 + Duration::from_secs(2));
        assert_eq!(refusal.stage, "stage-b");
        assert_eq!(refusal.since, Duration::from_secs(2));
        assert_eq!(gate.failed_peer(), Some("stage-b"));

        assert!(matches!(
            gate.decide(t0 + Duration::from_secs(9), INTERVAL),
            GateDecision::Refuse(_)
        ));
        assert!(matches!(
            gate.decide(t0 + INTERVAL, INTERVAL),
            GateDecision::Probe { .. }
        ));
    }

    #[test]
    fn a_probe_for_a_replaced_outage_does_not_close_the_gate() {
        let mut gate = StageAvailability::default();
        let t0 = Instant::now();
        gate.mark_unavailable("stage-a", t0);
        let GateDecision::Probe { generation, .. } = gate.decide(t0, INTERVAL) else {
            panic!("probe");
        };
        // Another exchange timed out while the probe was in flight.
        gate.mark_unavailable("stage-b", t0 + Duration::from_secs(1));
        assert_eq!(
            gate.close_after_probe(generation, t0 + Duration::from_secs(2)),
            None
        );
        assert_eq!(gate.failed_peer(), Some("stage-b"));
        assert_eq!(gate.refusal(t0 + Duration::from_secs(2)).stage, "stage-b");
    }
}
