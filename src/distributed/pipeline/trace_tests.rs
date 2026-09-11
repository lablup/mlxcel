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

//! Unit tests for the chrome-tracing writer.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use super::*;

fn unique_trace_path() -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let idx = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let mut path = std::env::temp_dir();
    path.push(format!("mlxcel_pp_trace_{pid}_{idx}.json"));
    let _ = fs::remove_file(&path);
    path
}

#[test]
fn flush_writes_chrome_tracing_envelope() {
    let path = unique_trace_path();
    let tracer = PpTracer::new(&path);
    tracer.record_batch_arrival(0, 4, 2);
    tracer.flush().expect("flush");

    let raw = fs::read_to_string(&path).expect("read");
    let json: serde_json::Value = serde_json::from_str(&raw).expect("parse");
    let events = json
        .get("traceEvents")
        .and_then(|v| v.as_array())
        .expect("traceEvents array");
    assert_eq!(events.len(), 1);
    let ev = &events[0];
    assert_eq!(
        ev.get("name").and_then(|v| v.as_str()),
        Some("batch_arrival")
    );
    assert_eq!(ev.get("ph").and_then(|v| v.as_str()), Some("i"));
    assert_eq!(ev.get("cat").and_then(|v| v.as_str()), Some("mlxcel-pp"));
    assert_eq!(ev.get("tid").and_then(|v| v.as_u64()), Some(0));
    let args = ev.get("args").expect("args");
    assert_eq!(args.get("batch_size").and_then(|v| v.as_str()), Some("4"));
    assert_eq!(args.get("queue_depth").and_then(|v| v.as_str()), Some("2"));
}

#[test]
fn stage_span_emits_duration_event_on_drop() {
    let path = unique_trace_path();
    let tracer = PpTracer::new(&path);
    {
        let _span = tracer.begin_stage(2, "forward", Some(7));
        // Let the clock move at least a tick.
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    tracer.flush().expect("flush");
    let raw = fs::read_to_string(&path).expect("read");
    let json: serde_json::Value = serde_json::from_str(&raw).expect("parse");
    let events = json
        .get("traceEvents")
        .and_then(|v| v.as_array())
        .expect("traceEvents");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].get("ph").and_then(|v| v.as_str()), Some("X"));
    assert_eq!(events[0].get("tid").and_then(|v| v.as_u64()), Some(2));
    assert!(
        events[0].get("dur").and_then(|v| v.as_u64()).unwrap_or(0) > 0,
        "duration must be non-zero"
    );
    let args = events[0].get("args").expect("span args");
    assert_eq!(args.get("sequence_id").and_then(|v| v.as_str()), Some("7"));
}

#[test]
fn activation_send_and_recv_pair_carries_direction_metadata() {
    let path = unique_trace_path();
    let tracer = PpTracer::new(&path);
    tracer.record_activation_send(0, 1, 1024);
    tracer.record_activation_recv(0, 1, 1024);
    tracer.flush().expect("flush");
    let raw = fs::read_to_string(&path).expect("read");
    let json: serde_json::Value = serde_json::from_str(&raw).expect("parse");
    let events = json
        .get("traceEvents")
        .and_then(|v| v.as_array())
        .expect("traceEvents");
    let names: Vec<_> = events
        .iter()
        .map(|e| e.get("name").and_then(|v| v.as_str()).unwrap_or(""))
        .collect();
    assert_eq!(names, vec!["activation_send", "activation_recv"]);
    let send_tid = events[0].get("tid").and_then(|v| v.as_u64()).unwrap();
    let recv_tid = events[1].get("tid").and_then(|v| v.as_u64()).unwrap();
    assert_eq!(send_tid, 0, "send must land on source stage");
    assert_eq!(recv_tid, 1, "recv must land on destination stage");
}

#[test]
fn admission_reject_records_reason_and_occupancy() {
    let path = unique_trace_path();
    let tracer = PpTracer::new(&path);
    tracer.record_admission_reject(3, "insufficient_memory", 10_000, 4_000);
    tracer.flush().expect("flush");
    let raw = fs::read_to_string(&path).expect("read");
    let json: serde_json::Value = serde_json::from_str(&raw).expect("parse");
    let events = json
        .get("traceEvents")
        .and_then(|v| v.as_array())
        .expect("traceEvents");
    assert_eq!(events.len(), 1);
    let args = events[0].get("args").expect("args");
    assert_eq!(
        args.get("reason").and_then(|v| v.as_str()),
        Some("insufficient_memory")
    );
    assert_eq!(
        args.get("required_bytes").and_then(|v| v.as_str()),
        Some("10000")
    );
    assert_eq!(
        args.get("available_bytes").and_then(|v| v.as_str()),
        Some("4000")
    );
}

#[test]
fn flush_is_idempotent() {
    let path = unique_trace_path();
    let tracer = PpTracer::new(&path);
    tracer.record_batch_arrival(0, 1, 0);
    tracer.flush().expect("first flush");
    tracer.record_batch_arrival(0, 2, 1);
    // Second flush is a no-op — we keep the first successful write.
    tracer.flush().expect("second flush");
    let raw = fs::read_to_string(&path).expect("read");
    let json: serde_json::Value = serde_json::from_str(&raw).expect("parse");
    let events = json
        .get("traceEvents")
        .and_then(|v| v.as_array())
        .expect("traceEvents");
    // Only the first flush's event was persisted.
    assert_eq!(events.len(), 1);
    let args = events[0].get("args").expect("args");
    assert_eq!(args.get("batch_size").and_then(|v| v.as_str()), Some("1"));
}
