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

//! Unit tests of the availability gate state machine (#2782, #2790).

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
    assert!(gate.mark_unavailable(Some("stage-a"), t0), "closed to open");
    assert!(
        !gate.mark_unavailable(Some("stage-a"), t0 + Duration::from_millis(1)),
        "a second timeout is not a transition"
    );
    assert_eq!(gate.failed_peer(), Some("stage-a"));

    let t1 = t0 + Duration::from_secs(1);
    let GateDecision::Probe { generation, stage } = gate.decide(t1, INTERVAL) else {
        panic!("the first request after a timeout probes");
    };
    assert_eq!(stage.as_deref(), Some("stage-a"));

    // Inside the interval: refused, with the outage length so far.
    match gate.decide(t1 + Duration::from_secs(2), INTERVAL) {
        GateDecision::Refuse(err) => {
            assert_eq!(err.stage.as_deref(), Some("stage-a"));
            assert_eq!(err.since, Duration::from_secs(3));
        }
        other => panic!("expected a refusal, got {other:?}"),
    }

    // The probe succeeded: closed, with the outage length.
    let t2 = t1 + Duration::from_secs(4);
    assert_eq!(
        gate.close_after_probe(generation, t2),
        ProbeClose::Closed(Duration::from_secs(5))
    );
    assert_eq!(gate.decide(t2, INTERVAL), GateDecision::Pass);
    assert_eq!(gate.failed_peer(), None);
}

#[test]
fn probe_is_due_again_after_the_interval_and_a_failed_probe_keeps_it_open() {
    let mut gate = StageAvailability::default();
    let t0 = Instant::now();
    gate.mark_unavailable(Some("stage-a"), t0);
    let GateDecision::Probe { generation, .. } = gate.decide(t0, INTERVAL) else {
        panic!("probe");
    };
    let ProbeFailure { refusal, reopened } =
        gate.probe_failed(generation, "stage-b", t0 + Duration::from_secs(2));
    assert!(!reopened, "the gate was open already");
    assert_eq!(refusal.stage.as_deref(), Some("stage-b"));
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
    gate.mark_unavailable(Some("stage-a"), t0);
    let GateDecision::Probe { generation, .. } = gate.decide(t0, INTERVAL) else {
        panic!("probe");
    };
    // Another exchange timed out while the probe was in flight.
    gate.mark_unavailable(Some("stage-b"), t0 + Duration::from_secs(1));
    assert_eq!(
        gate.close_after_probe(generation, t0 + Duration::from_secs(2)),
        ProbeClose::Superseded
    );
    assert_eq!(gate.failed_peer(), Some("stage-b"));
    assert_eq!(
        gate.refusal(t0 + Duration::from_secs(2)).stage.as_deref(),
        Some("stage-b")
    );
}

/// Two probes for the same outage overlap: one that outlasts the probe
/// interval lets the next request probe too.
fn overlapping_probes(gate: &mut StageAvailability, t0: Instant) -> u64 {
    gate.mark_unavailable(Some("stage-a"), t0);
    let GateDecision::Probe {
        generation: first, ..
    } = gate.decide(t0, INTERVAL)
    else {
        panic!("first probe");
    };
    let GateDecision::Probe {
        generation: second, ..
    } = gate.decide(t0 + INTERVAL, INTERVAL)
    else {
        panic!("second probe once the interval passed");
    };
    assert_eq!(first, second, "both probes are for the same outage");
    first
}

#[test]
fn a_second_succeeded_probe_for_a_closed_outage_is_served() {
    let mut gate = StageAvailability::default();
    let t0 = Instant::now();
    let generation = overlapping_probes(&mut gate, t0);
    let t1 = t0 + INTERVAL + Duration::from_secs(1);
    assert_eq!(
        gate.close_after_probe(generation, t1),
        ProbeClose::Closed(Duration::from_secs(11))
    );
    assert_eq!(
        gate.close_after_probe(generation, t1 + Duration::from_millis(5)),
        ProbeClose::AlreadyClosed,
        "the sibling probe also reached every stage; its request is served"
    );
    assert_eq!(gate.decide(t1, INTERVAL), GateDecision::Pass);
}

#[test]
fn a_failed_probe_after_a_sibling_closed_the_gate_reopens_it_visibly() {
    let mut gate = StageAvailability::default();
    let t0 = Instant::now();
    let generation = overlapping_probes(&mut gate, t0);
    let t1 = t0 + INTERVAL + Duration::from_secs(1);
    assert!(matches!(
        gate.close_after_probe(generation, t1),
        ProbeClose::Closed(_)
    ));
    let failure = gate.probe_failed(generation, "stage-b", t1 + Duration::from_secs(1));
    assert!(failure.reopened, "the caller logs the open line again");
    assert_eq!(failure.refusal.stage.as_deref(), Some("stage-b"));
    assert_eq!(failure.refusal.since, Duration::ZERO);
    assert_eq!(gate.failed_peer(), Some("stage-b"));
    // The reopen counts as a probe: the next request inside the
    // interval is refused without sending anything.
    assert!(matches!(
        gate.decide(t1 + Duration::from_secs(2), INTERVAL),
        GateDecision::Refuse(_)
    ));
}

#[test]
fn a_stale_failed_probe_does_not_touch_a_newer_outage_or_a_newer_close() {
    let mut gate = StageAvailability::default();
    let t0 = Instant::now();
    let stale = overlapping_probes(&mut gate, t0);
    // The sibling probe failed first and moved the outage on.
    let t1 = t0 + INTERVAL + Duration::from_secs(1);
    let first = gate.probe_failed(stale, "stage-b", t1);
    assert!(!first.reopened);
    let late = gate.probe_failed(stale, "stage-c", t1 + Duration::from_secs(1));
    assert!(!late.reopened);
    assert_eq!(
        late.refusal.stage.as_deref(),
        Some("stage-b"),
        "the newer failure is kept"
    );
    assert_eq!(gate.failed_peer(), Some("stage-b"));

    // A newer probe closes the gate; a still older failure leaves it
    // closed and refuses only its own request.
    let t2 = t1 + INTERVAL + Duration::from_secs(1);
    let GateDecision::Probe { generation, .. } = gate.decide(t2, INTERVAL) else {
        panic!("probe after the interval");
    };
    assert!(matches!(
        gate.close_after_probe(generation, t2),
        ProbeClose::Closed(_)
    ));
    let older = gate.probe_failed(stale, "stage-c", t2);
    assert!(!older.reopened);
    assert_eq!(older.refusal.stage.as_deref(), Some("stage-c"));
    assert_eq!(gate.decide(t2, INTERVAL), GateDecision::Pass);
    assert_eq!(gate.failed_peer(), None);
}

/// A pipeline run timeout opens the gate without naming a stage
/// (#2790): the probe it triggers carries no stage, the refusals say so,
/// and the failed probe then names the stage that is really silent.
#[test]
fn an_unattributed_outage_is_named_by_the_failed_probe() {
    let mut gate = StageAvailability::default();
    let t0 = Instant::now();
    assert!(gate.mark_unavailable(None, t0), "closed to open");
    assert_eq!(gate.failed_peer(), None, "no stage is named yet");
    let refusal = gate.refusal(t0 + Duration::from_secs(1));
    assert_eq!(refusal.stage, None);
    assert!(
        refusal.to_string().contains("an unidentified stage"),
        "{refusal}"
    );

    let GateDecision::Probe { generation, stage } = gate.decide(t0, INTERVAL) else {
        panic!("the first request after the timeout probes");
    };
    assert_eq!(stage, None);
    match gate.decide(t0 + Duration::from_secs(1), INTERVAL) {
        GateDecision::Refuse(err) => assert_eq!(err.stage, None),
        other => panic!("expected a refusal, got {other:?}"),
    }

    let failure = gate.probe_failed(generation, "stage-c", t0 + Duration::from_secs(2));
    assert!(!failure.reopened, "the gate was open already");
    assert_eq!(failure.refusal.stage.as_deref(), Some("stage-c"));
    assert_eq!(gate.failed_peer(), Some("stage-c"));
    assert_eq!(
        gate.refusal(t0 + Duration::from_secs(3)).stage.as_deref(),
        Some("stage-c")
    );
}

/// An unattributed timeout while a stage is named replaces the name:
/// the newer outage is the one the gate describes.
#[test]
fn an_unattributed_timeout_clears_an_older_name() {
    let mut gate = StageAvailability::default();
    let t0 = Instant::now();
    gate.mark_unavailable(Some("stage-a"), t0);
    assert!(!gate.mark_unavailable(None, t0 + Duration::from_secs(1)));
    assert_eq!(gate.failed_peer(), None);
    assert_eq!(gate.refusal(t0 + Duration::from_secs(2)).stage, None);
}

#[test]
fn a_named_refusal_keeps_its_text() {
    let refusal = PipelineUnavailable {
        stage: Some("10.0.0.3:9000".to_string()),
        since: Duration::from_secs(4),
    };
    assert_eq!(
        refusal.to_string(),
        "remote pipeline stage 10.0.0.3:9000 has not answered for 4s; refusing the request \
         until a probe reaches every stage"
    );
    assert_eq!(stage_label(None), "an unidentified stage");
    assert_eq!(stage_label(Some("10.0.0.3:9000")), "10.0.0.3:9000");
}
