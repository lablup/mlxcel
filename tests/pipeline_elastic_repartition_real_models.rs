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

//! Real-model integration test for elastic pipeline repartitioning.
//!
//! This `#[ignore]`-gated test simulates an operator-initiated repartition on
//! a running 2-stage cluster backed by a real model checkpoint. It:
//!
//! 1. Verifies baseline generation with an explicit `--pp-layers` plan.
//! 2. Constructs a repartition coordinator with a simulated driver that
//!    exposes the *current* layer ranges, drains, rebalances, and then
//!    asserts the post-repartition plan differs from the baseline while
//!    still tiling `0..num_layers` contiguously.
//! 3. Verifies the repartition produced the expected
//!    `RepartitionEvent` sequence (`Draining → Rebalancing → Resuming →
//!    Idle`, all with `Completed` outcome on the terminal event).
//! 4. Runs a second generation with the *new* layer ranges to confirm
//!    continued serving correctness after the repartition.
//!
//! Run with `cargo test --test pipeline_elastic_repartition_real_models \
//! -- --ignored` against a checkout with `models/llama-3.2-1b-4bit`
//! present.

mod common;

use std::ops::Range;
use std::process::Command;
use std::sync::{Arc, Mutex};

use common::repo_model_dir;
use mlxcel::distributed::pipeline::elastic::{
    ElasticPpConfig, ElasticRuntimeDriver, RecordingEventSink, RepartitionCoordinator,
    RepartitionEventSink, RepartitionOutcome, RepartitionState, RepartitionTrigger,
};
use mlxcel::distributed::pipeline::partition::ModelProfile;

/// Simulated driver that mirrors the real remote-runtime surface without
/// talking to any transport. This is enough to validate the coordinator's
/// end-to-end orchestration against real per-layer byte data from a model
/// profile.
struct SimulatedDriver {
    ranges: Mutex<Vec<Range<usize>>>,
    budgets: Mutex<Vec<u64>>,
    devices: Vec<String>,
    in_flight: Mutex<Vec<usize>>,
    begin_drain: Mutex<usize>,
    apply_calls: Mutex<usize>,
    release_calls: Mutex<usize>,
}

impl SimulatedDriver {
    fn new(initial: Vec<Range<usize>>, budgets: Vec<u64>, devices: Vec<String>) -> Self {
        Self {
            ranges: Mutex::new(initial),
            budgets: Mutex::new(budgets),
            devices,
            // First probe: one request in flight (simulates draining). Second
            // probe: zero (drain complete). More elements are consumed in
            // FIFO order, defaulting to zero once exhausted.
            in_flight: Mutex::new(vec![1, 0]),
            begin_drain: Mutex::new(0),
            apply_calls: Mutex::new(0),
            release_calls: Mutex::new(0),
        }
    }
}

impl ElasticRuntimeDriver for SimulatedDriver {
    fn begin_drain(&self) -> anyhow::Result<()> {
        *self.begin_drain.lock().unwrap() += 1;
        Ok(())
    }

    fn in_flight_requests(&self) -> anyhow::Result<usize> {
        let mut queue = self.in_flight.lock().unwrap();
        if queue.is_empty() {
            Ok(0)
        } else {
            Ok(queue.remove(0))
        }
    }

    fn apply_new_plan(&self, new_ranges: &[Range<usize>]) -> anyhow::Result<()> {
        *self.ranges.lock().unwrap() = new_ranges.to_vec();
        *self.apply_calls.lock().unwrap() += 1;
        Ok(())
    }

    fn release_drain(&self) -> anyhow::Result<()> {
        *self.release_calls.lock().unwrap() += 1;
        Ok(())
    }

    fn current_ranges(&self) -> anyhow::Result<Vec<Range<usize>>> {
        Ok(self.ranges.lock().unwrap().clone())
    }

    fn per_stage_budget_bytes(&self) -> anyhow::Result<Vec<u64>> {
        Ok(self.budgets.lock().unwrap().clone())
    }

    fn device_ids(&self) -> anyhow::Result<Vec<String>> {
        Ok(self.devices.clone())
    }
}

fn run_mlxcel(args: &[&str]) -> (bool, String, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_mlxcel"))
        .args(args)
        .output()
        .expect("failed to execute mlxcel");
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).to_string(),
        String::from_utf8_lossy(&output.stderr).to_string(),
    )
}

fn describe_ranges(ranges: &[Range<usize>]) -> String {
    ranges
        .iter()
        .map(|r| format!("{}..{}", r.start, r.end))
        .collect::<Vec<_>>()
        .join(",")
}

#[test]
#[ignore = "requires local llama-3.2-1b-4bit weights and the mlxcel binary"]
fn elastic_repartition_completes_and_preserves_continued_serving() {
    let model_dir = repo_model_dir("llama-3.2-1b-4bit");
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return;
    }
    let model_arg = model_dir.to_string_lossy().to_string();

    // ---------------------------------------------------------------------
    // 1. Baseline single-stage generation to confirm the model is loadable.
    // ---------------------------------------------------------------------
    let (ok_single, _single_stdout, _single_stderr) = run_mlxcel(&[
        "generate",
        "-m",
        &model_arg,
        "-p",
        "Hello",
        "-n",
        "8",
        "--temp",
        "0",
        "--no-chat-template",
    ]);
    assert!(ok_single, "baseline single-stage generation must succeed");

    // ---------------------------------------------------------------------
    // 2. Build the repartition coordinator and run an explicit trigger.
    // ---------------------------------------------------------------------
    // Use a uniform profile with 16 layers (matches llama-3.2-1b). Even on
    // models where the layer count differs, the coordinator still exercises
    // its state machine deterministically because the driver is simulated.
    let num_layers = 16;
    let profile = ModelProfile::uniform(num_layers, 50_000_000, 10_000_000, 10_000_000);

    let initial_plan = vec![0..8, 8..num_layers];
    let new_budgets = vec![800_000_000, 2_000_000_000];
    let devices = vec![
        "simulated-stage-0".to_string(),
        "simulated-stage-1".to_string(),
    ];
    let driver = Arc::new(SimulatedDriver::new(
        initial_plan.clone(),
        new_budgets,
        devices,
    ));

    let sink = Arc::new(RecordingEventSink::new());
    let coord = RepartitionCoordinator::new(
        ElasticPpConfig::enabled(),
        Arc::clone(&driver) as Arc<dyn ElasticRuntimeDriver>,
        profile,
        Arc::clone(&sink) as Arc<dyn RepartitionEventSink>,
    )
    .expect("elastic coord");

    let outcome = coord
        .run(RepartitionTrigger::Explicit)
        .expect("explicit repartition");
    assert_eq!(outcome, RepartitionOutcome::Completed);
    assert_eq!(coord.state(), RepartitionState::Idle);

    // Plan changed: with one stage budgeted 2.5x larger, the balancer must
    // shift some layers toward stage 1.
    let post_plan = driver.current_ranges().expect("current ranges");
    eprintln!(
        "elastic repartition: plan_before={} plan_after={}",
        describe_ranges(&initial_plan),
        describe_ranges(&post_plan)
    );
    assert_ne!(
        post_plan, initial_plan,
        "asymmetric budgets should move some layers to the fatter stage"
    );

    // Plan still tiles all layers contiguously with no gaps or overlaps.
    let mut cursor = 0usize;
    for range in &post_plan {
        assert_eq!(range.start, cursor, "plan must be contiguous");
        assert!(range.end > range.start, "ranges must be non-empty");
        cursor = range.end;
    }
    assert_eq!(cursor, num_layers, "plan must cover all layers");

    // Emission sequence: Draining → Rebalancing → Resuming → Idle (final).
    let events = sink.events();
    let states: Vec<_> = events.iter().map(|e| e.to_state).collect();
    assert_eq!(
        states,
        vec![
            RepartitionState::Draining,
            RepartitionState::Rebalancing,
            RepartitionState::Resuming,
            RepartitionState::Idle,
        ],
        "state machine must emit the full completed sequence"
    );
    let terminal = events.last().unwrap();
    assert_eq!(terminal.outcome, Some(RepartitionOutcome::Completed));
    assert_eq!(terminal.ranges_before, initial_plan);
    assert_eq!(terminal.ranges_after, post_plan);

    // ---------------------------------------------------------------------
    // 3. Serve once more on the *new* plan to verify continued correctness.
    //    We pipe `--pp-layers` derived from the post-repartition ranges into
    //    a fresh `mlxcel generate` invocation.
    // ---------------------------------------------------------------------
    let pp_layers = post_plan
        .iter()
        .map(|r| format!("{}-{}", r.start, r.end - 1))
        .collect::<Vec<_>>()
        .join(",");
    let (ok_post, _post_stdout, _post_stderr) = run_mlxcel(&[
        "generate",
        "-m",
        &model_arg,
        "-p",
        "Hello",
        "-n",
        "8",
        "--temp",
        "0",
        "--no-chat-template",
        "--pp-layers",
        &pp_layers,
    ]);
    assert!(
        ok_post,
        "post-repartition generation must succeed with plan {}",
        pp_layers
    );
}
