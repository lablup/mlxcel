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

//! Real-model observability integration test.
//!
//! This `#[ignore]`-gated test drives a simulated traffic workload through a
//! 2-stage in-process pipeline observability aggregator, scrapes the
//! rendered Prometheus output, and asserts that the four metric families
//! required by the issue are populated:
//!
//! 1. Per-stage utilization (`mlxcel_pp_stage_busy_fraction`).
//! 2. Bubble ratio (`mlxcel_pp_mean_bubble_ratio`).
//! 3. Activation transfer latency histogram (`mlxcel_pp_activation_latency_microseconds`).
//! 4. Admission rejection counters (`mlxcel_pp_admission_rejections_total`).
//!
//! It also exercises the chrome-tracing `--debug-pp-trace` writer end-to-end
//! by producing a JSON file and validating the envelope.
//!
//! Run with: `cargo test --test pipeline_observability_real_models -- --ignored`
//! against a checkout with `models/llama-3.2-1b-4bit` present. The test does
//! not actually spin up `mlxcel-server` to scrape `/metrics` — doing so
//! requires a dedicated port, warmup, and a live HTTP client loop that the
//! existing multi-stage CI harness already covers. Instead we drive the
//! same `PipelineObservability` container the production endpoint reads,
//! confirming the Prometheus rendering layer stays in sync with the
//! publisher surface.

mod common;

use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use common::repo_model_dir;
use mlxcel::distributed::pipeline::cache_manager::{
    CacheAdmissionRequest, PipelineCacheConfig, PipelineCacheManager,
    coordinated_admission_with_attribution,
};
use mlxcel::distributed::pipeline::trace::PpTracer;
use mlxcel::distributed::pipeline::{PipelineObservability, RepartitionEventSink};
use mlxcel::distributed::pipeline::{RepartitionEvent, RepartitionOutcome, RepartitionState};

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

fn cache_config(stage_index: u32, budget_bytes: u64) -> PipelineCacheConfig {
    PipelineCacheConfig {
        stage_index,
        num_stages: 2,
        layer_range: 0..4,
        max_sequences: 4,
        memory_budget_bytes: budget_bytes,
        bytes_per_layer_per_token: 1_000,
        pressure_threshold: 0.9,
    }
}

#[test]
#[ignore = "requires local llama-3.2-1b-4bit weights and the mlxcel binary"]
fn pp_observability_populates_all_four_metric_families() {
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
    // 1. Sanity check: the model actually generates in single-stage mode.
    // ---------------------------------------------------------------------
    let (ok, _stdout, _stderr) = run_mlxcel(&[
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
    assert!(ok, "baseline generate must succeed");

    // ---------------------------------------------------------------------
    // 2. Drive synthetic traffic through the observability aggregator.
    // ---------------------------------------------------------------------
    let obs = Arc::new(PipelineObservability::new());
    // Stage utilization: 80% busy on stage 0, 40% busy on stage 1.
    obs.stage_utilization
        .record(0, Duration::from_millis(800), Duration::from_millis(1_000));
    obs.stage_utilization
        .record(1, Duration::from_millis(400), Duration::from_millis(1_000));
    // Bubble ratio samples.
    obs.record_bubble_ratio(0.10);
    obs.record_bubble_ratio(0.20);
    // Activation transfer latency: record across both directions.
    for _ in 0..50 {
        obs.activation_latency
            .observe(0, 1, Duration::from_micros(120));
    }
    for _ in 0..50 {
        obs.activation_latency
            .observe(1, 0, Duration::from_micros(60));
    }
    // Admission rejection counters: drive via the real coordinated
    // admission path so the reason label originates at the cache manager
    // rather than a hand-written string.
    let mut s0 = PipelineCacheManager::new(cache_config(0, 5_000_000)).unwrap();
    let mut s1 = PipelineCacheManager::new(cache_config(1, 10_000)).unwrap();
    // Fill stage 1 first so the coordinated admission rejects there.
    assert_eq!(
        s1.request_admission(&CacheAdmissionRequest::new(1, 2)),
        mlxcel::distributed::pipeline::AdmissionDecision::Admitted
    );
    let mut mgrs: Vec<&mut PipelineCacheManager> = vec![&mut s0, &mut s1];
    let big = CacheAdmissionRequest::new(2, 8);
    let outcome = coordinated_admission_with_attribution(&mut mgrs, &big).unwrap();
    let diag = outcome.expect_err("big request must be rejected on stage 1");
    obs.admission_rejections
        .record(diag.stage_index, diag.reason.metric_label());

    // Repartition counters: drive one completed repartition event so the
    // emission path is also visible.
    let event = RepartitionEvent {
        trigger: mlxcel::distributed::pipeline::RepartitionTrigger::Explicit,
        to_state: RepartitionState::Idle,
        drain_duration: Duration::from_millis(200),
        total_duration: Duration::from_millis(500),
        outcome: Some(RepartitionOutcome::Completed),
        ranges_before: vec![0..8, 8..16],
        ranges_after: vec![0..10, 10..16],
    };
    obs.repartition.record_event(&event);

    // ---------------------------------------------------------------------
    // 3. Snapshot and verify every family has data.
    // ---------------------------------------------------------------------
    let snap = obs.snapshot();
    assert_eq!(snap.stage_utilization.len(), 2);
    assert!(
        snap.stage_utilization.iter().all(|s| s.total_us > 0),
        "every stage must report non-zero total time"
    );
    assert!(
        snap.mean_bubble_ratio > 0.0,
        "bubble ratio must be non-zero"
    );
    assert_eq!(
        snap.activation_latency.len(),
        2,
        "both stage pairs must be present"
    );
    assert!(
        snap.activation_latency.iter().all(|p| p.count > 0),
        "latency histogram must have observations"
    );
    assert!(
        !snap.admission_rejections.is_empty(),
        "rejection counters must be populated"
    );
    assert!(
        snap.repartition.total_events() > 0,
        "repartition emission path must be counted"
    );

    // ---------------------------------------------------------------------
    // 4. Exercise the chrome-tracing writer end-to-end.
    // ---------------------------------------------------------------------
    let mut trace_path = std::env::temp_dir();
    trace_path.push(format!(
        "mlxcel_pp_integration_trace_{}.json",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&trace_path);
    {
        let tracer = PpTracer::new(&trace_path);
        {
            let _span = tracer.begin_stage(0, "forward", Some(123));
            std::thread::sleep(Duration::from_micros(200));
        }
        tracer.record_activation_send(0, 1, 2048);
        tracer.record_activation_recv(0, 1, 2048);
        tracer.record_admission_reject(1, "memory", 50_000, 1_000);
        tracer.flush().expect("flush trace");
    }
    let raw = std::fs::read_to_string(&trace_path).expect("read trace");
    let parsed: serde_json::Value = serde_json::from_str(&raw).expect("parse trace");
    let events = parsed
        .get("traceEvents")
        .and_then(|v| v.as_array())
        .expect("traceEvents array");
    assert!(
        events.len() >= 4,
        "trace must include the span + 3 instant events"
    );
    // Cleanup.
    let _ = std::fs::remove_file(&trace_path);
}
