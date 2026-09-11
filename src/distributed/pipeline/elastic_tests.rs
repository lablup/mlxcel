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

//! Unit tests for the elastic pipeline repartition coordinator.

use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::*;
use crate::distributed::pipeline::partition::ModelProfile;

#[derive(Debug, Default)]
struct FakeDriverState {
    begin_drain_calls: usize,
    apply_new_plan_calls: Vec<Vec<Range<usize>>>,
    release_drain_calls: usize,
    in_flight_seq: Vec<usize>,
    plan_error: Option<String>,
}

#[derive(Debug)]
struct FakeDriver {
    state: Mutex<FakeDriverState>,
    ranges: Mutex<Vec<Range<usize>>>,
    budget: Vec<u64>,
    devices: Vec<String>,
}

impl FakeDriver {
    fn new(ranges: Vec<Range<usize>>, budget: Vec<u64>, devices: Vec<String>) -> Self {
        Self {
            state: Mutex::new(FakeDriverState {
                in_flight_seq: vec![0],
                ..Default::default()
            }),
            ranges: Mutex::new(ranges),
            budget,
            devices,
        }
    }

    fn with_in_flight(self, seq: Vec<usize>) -> Self {
        self.state.lock().unwrap().in_flight_seq = seq;
        self
    }

    fn with_plan_error(self, msg: &str) -> Self {
        self.state.lock().unwrap().plan_error = Some(msg.to_string());
        self
    }

    fn begin_drain_calls(&self) -> usize {
        self.state.lock().unwrap().begin_drain_calls
    }

    fn apply_calls(&self) -> Vec<Vec<Range<usize>>> {
        self.state.lock().unwrap().apply_new_plan_calls.clone()
    }

    fn release_calls(&self) -> usize {
        self.state.lock().unwrap().release_drain_calls
    }
}

impl ElasticRuntimeDriver for FakeDriver {
    fn begin_drain(&self) -> anyhow::Result<()> {
        self.state.lock().unwrap().begin_drain_calls += 1;
        Ok(())
    }

    fn in_flight_requests(&self) -> anyhow::Result<usize> {
        let mut s = self.state.lock().unwrap();
        if s.in_flight_seq.is_empty() {
            Ok(0)
        } else {
            Ok(s.in_flight_seq.remove(0))
        }
    }

    fn apply_new_plan(&self, new_ranges: &[Range<usize>]) -> anyhow::Result<()> {
        {
            let mut s = self.state.lock().unwrap();
            if let Some(msg) = s.plan_error.take() {
                anyhow::bail!(msg);
            }
            s.apply_new_plan_calls.push(new_ranges.to_vec());
        }
        *self.ranges.lock().unwrap() = new_ranges.to_vec();
        Ok(())
    }

    fn release_drain(&self) -> anyhow::Result<()> {
        self.state.lock().unwrap().release_drain_calls += 1;
        Ok(())
    }

    fn current_ranges(&self) -> anyhow::Result<Vec<Range<usize>>> {
        Ok(self.ranges.lock().unwrap().clone())
    }

    fn per_stage_budget_bytes(&self) -> anyhow::Result<Vec<u64>> {
        Ok(self.budget.clone())
    }

    fn device_ids(&self) -> anyhow::Result<Vec<String>> {
        Ok(self.devices.clone())
    }
}

fn test_profile(num_layers: usize, layer_bytes: u64) -> ModelProfile {
    ModelProfile::uniform(num_layers, layer_bytes, 1_000_000, 1_000_000)
}

#[test]
fn disabled_config_rejects_construction() {
    let profile = test_profile(8, 1_000_000);
    let driver = Arc::new(FakeDriver::new(
        vec![0..4, 4..8],
        vec![10_000_000, 10_000_000],
        vec!["a".into(), "b".into()],
    ));
    let sink = Arc::new(RecordingEventSink::new());
    let err = RepartitionCoordinator::new(ElasticPpConfig::default(), driver, profile, sink)
        .expect_err("disabled elastic config must fail to construct");
    assert!(err.to_string().contains("disabled"));
}

#[test]
fn explicit_trigger_drives_idle_to_idle_and_emits_events() {
    let profile = test_profile(8, 1_000_000);
    let driver = Arc::new(FakeDriver::new(
        vec![0..4, 4..8],
        vec![10_000_000, 10_000_000],
        vec!["a".into(), "b".into()],
    ));
    let sink = Arc::new(RecordingEventSink::new());
    let coord = RepartitionCoordinator::new(
        ElasticPpConfig::enabled().with_drain_timeout(Duration::from_secs(2)),
        Arc::clone(&driver) as Arc<dyn ElasticRuntimeDriver>,
        profile,
        Arc::clone(&sink) as Arc<dyn RepartitionEventSink>,
    )
    .expect("elastic coord");

    assert_eq!(coord.state(), RepartitionState::Idle);

    let outcome = coord
        .run(RepartitionTrigger::Explicit)
        .expect("explicit run");
    assert_eq!(outcome, RepartitionOutcome::Completed);
    assert_eq!(coord.state(), RepartitionState::Idle);

    // The driver should have received drain, apply, release in order.
    assert_eq!(driver.begin_drain_calls(), 1);
    assert_eq!(driver.apply_calls().len(), 1);
    assert_eq!(driver.release_calls(), 1);

    // Events must include Draining, Rebalancing, Resuming, Idle (final).
    let events = sink.events();
    let states: Vec<_> = events.iter().map(|e| e.to_state).collect();
    assert_eq!(
        states,
        vec![
            RepartitionState::Draining,
            RepartitionState::Rebalancing,
            RepartitionState::Resuming,
            RepartitionState::Idle,
        ]
    );

    // Only the final event carries an outcome.
    assert!(events[..3].iter().all(|e| e.outcome.is_none()));
    assert_eq!(events[3].outcome, Some(RepartitionOutcome::Completed));

    // Final event captures before/after ranges.
    assert_eq!(events[3].ranges_before, vec![0..4, 4..8]);
    assert!(!events[3].ranges_after.is_empty());
}

#[test]
fn drain_timeout_transitions_to_failed_and_emits_failed_event() {
    let profile = test_profile(8, 1_000_000);
    // Driver always reports 1 request in flight.
    let driver = Arc::new(
        FakeDriver::new(
            vec![0..4, 4..8],
            vec![10_000_000, 10_000_000],
            vec!["a".into(), "b".into()],
        )
        .with_in_flight(vec![1; 128]),
    );
    let sink = Arc::new(RecordingEventSink::new());
    let coord = RepartitionCoordinator::new(
        ElasticPpConfig::enabled().with_drain_timeout(Duration::from_millis(150)),
        Arc::clone(&driver) as Arc<dyn ElasticRuntimeDriver>,
        profile,
        Arc::clone(&sink) as Arc<dyn RepartitionEventSink>,
    )
    .expect("elastic coord");

    let err = coord
        .run(RepartitionTrigger::Explicit)
        .expect_err("drain timeout must bubble up");
    assert!(err.to_string().contains("drain timeout"));
    assert_eq!(coord.state(), RepartitionState::Failed);

    let events = sink.events();
    // First event: Draining. Last event: Failed with Outcome::Failed.
    assert_eq!(events.first().unwrap().to_state, RepartitionState::Draining);
    assert_eq!(events.last().unwrap().to_state, RepartitionState::Failed);
    assert_eq!(
        events.last().unwrap().outcome,
        Some(RepartitionOutcome::Failed)
    );
    // release_drain still called once on the failure path (best-effort).
    assert_eq!(driver.release_calls(), 1);
}

#[test]
fn pressure_trigger_below_threshold_is_rejected() {
    let profile = test_profile(8, 1_000_000);
    let driver = Arc::new(FakeDriver::new(
        vec![0..4, 4..8],
        vec![10_000_000, 10_000_000],
        vec!["a".into(), "b".into()],
    ));
    let sink = Arc::new(RecordingEventSink::new());
    let coord = RepartitionCoordinator::new(
        ElasticPpConfig::enabled().with_trigger_memory_fraction(0.95),
        Arc::clone(&driver) as Arc<dyn ElasticRuntimeDriver>,
        profile,
        Arc::clone(&sink) as Arc<dyn RepartitionEventSink>,
    )
    .expect("elastic coord");

    let err = coord
        .run(RepartitionTrigger::MemoryPressure {
            stage_index: 1,
            fraction: 0.80,
        })
        .expect_err("trigger below threshold must be rejected");
    assert!(err.to_string().contains("below threshold"));
    assert_eq!(coord.state(), RepartitionState::Idle);

    // Exactly one event emitted describing the aborted trigger.
    let events = sink.events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].outcome, Some(RepartitionOutcome::Aborted));
}

#[test]
fn pressure_trigger_within_cool_down_is_rejected() {
    let profile = test_profile(8, 1_000_000);
    let driver = Arc::new(FakeDriver::new(
        vec![0..4, 4..8],
        vec![10_000_000, 10_000_000],
        vec!["a".into(), "b".into()],
    ));
    let sink = Arc::new(RecordingEventSink::new());
    let coord = RepartitionCoordinator::new(
        ElasticPpConfig::enabled()
            .with_cool_down(Duration::from_secs(60))
            .with_trigger_memory_fraction(0.50),
        Arc::clone(&driver) as Arc<dyn ElasticRuntimeDriver>,
        profile,
        Arc::clone(&sink) as Arc<dyn RepartitionEventSink>,
    )
    .expect("elastic coord");

    // First pressure trigger: accepted.
    coord
        .run(RepartitionTrigger::MemoryPressure {
            stage_index: 0,
            fraction: 0.95,
        })
        .expect("first pressure trigger");

    // Second pressure trigger on the same stage, still within cool-down:
    // rejected.
    let err = coord
        .run(RepartitionTrigger::MemoryPressure {
            stage_index: 0,
            fraction: 0.95,
        })
        .expect_err("second pressure trigger must be debounced");
    assert!(err.to_string().contains("cool-down"));
}

#[test]
fn explicit_trigger_bypasses_cool_down() {
    let profile = test_profile(8, 1_000_000);
    let driver = Arc::new(FakeDriver::new(
        vec![0..4, 4..8],
        vec![10_000_000, 10_000_000],
        vec!["a".into(), "b".into()],
    ));
    let sink = Arc::new(RecordingEventSink::new());
    let coord = RepartitionCoordinator::new(
        ElasticPpConfig::enabled()
            .with_cool_down(Duration::from_secs(60))
            .with_trigger_memory_fraction(0.50),
        Arc::clone(&driver) as Arc<dyn ElasticRuntimeDriver>,
        profile,
        Arc::clone(&sink) as Arc<dyn RepartitionEventSink>,
    )
    .expect("elastic coord");

    coord
        .run(RepartitionTrigger::MemoryPressure {
            stage_index: 0,
            fraction: 0.95,
        })
        .expect("first pressure trigger");
    // Explicit retry should bypass the debounce.
    coord
        .run(RepartitionTrigger::Explicit)
        .expect("explicit trigger bypasses cool-down");
}

#[test]
fn apply_new_plan_error_transitions_to_failed() {
    let profile = test_profile(8, 1_000_000);
    let driver = Arc::new(
        FakeDriver::new(
            vec![0..4, 4..8],
            vec![10_000_000, 10_000_000],
            vec!["a".into(), "b".into()],
        )
        .with_plan_error("synthetic reload failure"),
    );
    let sink = Arc::new(RecordingEventSink::new());
    let coord = RepartitionCoordinator::new(
        ElasticPpConfig::enabled(),
        Arc::clone(&driver) as Arc<dyn ElasticRuntimeDriver>,
        profile,
        Arc::clone(&sink) as Arc<dyn RepartitionEventSink>,
    )
    .expect("elastic coord");

    let err = coord
        .run(RepartitionTrigger::Explicit)
        .expect_err("reload failure must surface");
    assert!(err.to_string().contains("synthetic reload failure"));
    assert_eq!(coord.state(), RepartitionState::Failed);
    // Subsequent pressure triggers are rejected until the failure is cleared.
    let rejected = coord
        .run(RepartitionTrigger::MemoryPressure {
            stage_index: 0,
            fraction: 0.99,
        })
        .expect_err("failed state must reject pressure triggers");
    assert!(rejected.to_string().contains("failed state"));
    // After clearance, a new explicit trigger can start the drain path (which
    // will succeed now that the synthetic error has been consumed).
    coord.clear_failure().expect("clear failure");
    coord
        .run(RepartitionTrigger::Explicit)
        .expect("explicit retry after clearance");
    assert_eq!(coord.state(), RepartitionState::Idle);
}

#[test]
fn noop_sink_does_not_panic() {
    let sink = NoopEventSink;
    let event = RepartitionEvent {
        trigger: RepartitionTrigger::Explicit,
        to_state: RepartitionState::Idle,
        drain_duration: Duration::ZERO,
        total_duration: Duration::ZERO,
        outcome: Some(RepartitionOutcome::Completed),
        ranges_before: Vec::new(),
        ranges_after: Vec::new(),
    };
    sink.record_event(&event);
}
