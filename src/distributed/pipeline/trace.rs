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
//
//! Chrome-tracing compatible trace writer for pipeline-parallel scheduler
//! actions.
//!
//! When `mlxcel-server --debug-pp-trace <file>` is passed the runtime
//! constructs a [`PpTracer`] and hands it out as an `Arc` so every stage
//! worker, activation sender, scheduler, and admission path can record
//! events without touching the hot-path serializer. Events are buffered in
//! memory until `PpTracer::flush` (or `Drop`) writes the JSON array to disk
//! using the Chrome-tracing "traceEvents" format.
//!
//! The format is intentionally minimal: we emit duration events (`ph: "X"`)
//! for things we have a bracket pair for (stage enter/exit), and instant
//! events (`ph: "i"`) for singletons such as batch arrival, activation
//! send/receive, and admission rejection. Chrome and Perfetto accept both
//! shapes without additional metadata, so no viewer configuration is needed.
//!
//! Used by: scheduler (`schedule.rs`), stage workers (`stage_worker.rs`,
//! `stage_executor/*`), cache admission (`cache_manager.rs`), server startup
//! (`server/startup.rs`).

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Serialize, Serializer};

/// Category name used for every event emitted by this tracer.
const TRACE_CATEGORY: &str = "mlxcel-pp";

/// Pseudo-process id used for every stage so Chrome-tracing groups them
/// under a single process row.
const TRACE_PID: u64 = 1;

/// Chrome-tracing phase for instant events.
const PHASE_INSTANT: &str = "i";

/// Chrome-tracing phase for complete duration events.
const PHASE_DURATION: &str = "X";

/// Serializable trace event in Chrome-tracing format.
///
/// Fields use the canonical `ts`/`dur` microsecond convention (u64). Keys
/// are renamed to the short Chrome-tracing names via serde rename so viewers
/// pick them up without extra metadata.
#[derive(Debug, Clone, Serialize)]
struct TraceEvent {
    /// Event name.
    name: String,
    /// Chrome-tracing category tag. Always `mlxcel-pp` in this crate.
    #[serde(rename = "cat")]
    category: &'static str,
    /// Phase: `i` (instant) or `X` (duration).
    #[serde(rename = "ph")]
    phase: &'static str,
    /// Timestamp since tracer construction, in microseconds.
    #[serde(rename = "ts")]
    timestamp_us: u64,
    /// Duration, microseconds. Only populated for `X` phase events.
    #[serde(rename = "dur", skip_serializing_if = "Option::is_none")]
    duration_us: Option<u64>,
    /// Pseudo-pid. Always 1 in this crate.
    #[serde(rename = "pid")]
    pid: u64,
    /// Pseudo-tid — used here to group by stage index.
    #[serde(rename = "tid")]
    tid: u64,
    /// Extra labelled dimensions (sequence id, rejection reason, etc.).
    #[serde(rename = "args", skip_serializing_if = "Option::is_none")]
    args: Option<TraceArgs>,
}

/// Key-value arguments attached to a trace event. Kept `String`-typed so
/// downstream viewers treat every attribute as free-form text.
#[derive(Debug, Clone, Default, Serialize)]
struct TraceArgs {
    #[serde(flatten, serialize_with = "serialize_args_map")]
    fields: Vec<(String, String)>,
}

fn serialize_args_map<S>(fields: &[(String, String)], serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    use serde::ser::SerializeMap;
    let mut map = serializer.serialize_map(Some(fields.len()))?;
    for (k, v) in fields {
        map.serialize_entry(k, v)?;
    }
    map.end()
}

/// Scoped handle returned by [`PpTracer::begin_stage`]. On drop, emits a
/// duration event for the elapsed span so callers do not have to pair the
/// begin/end manually.
///
/// Held exclusively by a single task — the tracer itself is shareable, but
/// spans are intentionally not `Sync` because Chrome-tracing duration events
/// carry a wall-clock `ts` that must be paired with a single timeline.
pub struct StageSpan<'a> {
    tracer: &'a PpTracer,
    stage_index: u32,
    started_at: Instant,
    name: String,
    sequence_id: Option<u64>,
}

impl<'a> StageSpan<'a> {
    /// Finish the span manually. Equivalent to dropping the handle.
    pub fn finish(self) {
        // Drop runs the finalizer.
        drop(self);
    }
}

impl Drop for StageSpan<'_> {
    fn drop(&mut self) {
        let duration = self.started_at.elapsed();
        let ts = self.tracer.elapsed_us(self.started_at);
        let mut args = TraceArgs::default();
        if let Some(seq) = self.sequence_id {
            args.fields
                .push(("sequence_id".to_string(), seq.to_string()));
        }
        let event = TraceEvent {
            name: std::mem::take(&mut self.name),
            category: TRACE_CATEGORY,
            phase: PHASE_DURATION,
            timestamp_us: ts,
            duration_us: Some(duration_as_us(duration)),
            pid: TRACE_PID,
            tid: u64::from(self.stage_index),
            args: if args.fields.is_empty() {
                None
            } else {
                Some(args)
            },
        };
        self.tracer.push_event(event);
    }
}

/// Chrome-tracing writer shared across the pipeline. Buffers events in
/// memory and flushes them to disk on demand or on drop.
pub struct PpTracer {
    /// Instant the tracer was constructed — every emitted `ts` is relative
    /// to this.
    started_at: Instant,
    /// Destination file path. Only written at [`PpTracer::flush`] / drop.
    output_path: PathBuf,
    /// Accumulated events.
    events: Mutex<Vec<TraceEvent>>,
    /// Once `flush` has written to disk we stop buffering, so repeated
    /// flushes are idempotent and `Drop` does not double-write.
    flushed: Mutex<bool>,
}

impl std::fmt::Debug for PpTracer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PpTracer")
            .field("output_path", &self.output_path)
            .field(
                "events",
                &self.events.lock().map(|v| v.len()).unwrap_or_else(|_| 0),
            )
            .finish()
    }
}

impl PpTracer {
    /// Construct a new tracer that will flush to `output_path`.
    pub fn new(output_path: impl Into<PathBuf>) -> Self {
        Self {
            started_at: Instant::now(),
            output_path: output_path.into(),
            events: Mutex::new(Vec::new()),
            flushed: Mutex::new(false),
        }
    }

    /// Destination path.
    pub fn output_path(&self) -> &Path {
        &self.output_path
    }

    /// Number of events buffered so far. Useful for assertions in tests.
    pub fn event_count(&self) -> usize {
        self.events.lock().expect("pp tracer poisoned").len()
    }

    /// Open a scoped span for a stage's forward pass. The span records a
    /// duration event when dropped.
    pub fn begin_stage(
        &self,
        stage_index: u32,
        name: impl Into<String>,
        sequence_id: Option<u64>,
    ) -> StageSpan<'_> {
        StageSpan {
            tracer: self,
            stage_index,
            started_at: Instant::now(),
            name: name.into(),
            sequence_id,
        }
    }

    /// Record an instant event for a batch arrival in the scheduler queue.
    pub fn record_batch_arrival(&self, stage_index: u32, batch_size: usize, queue_depth: usize) {
        let mut args = TraceArgs::default();
        args.fields
            .push(("batch_size".to_string(), batch_size.to_string()));
        args.fields
            .push(("queue_depth".to_string(), queue_depth.to_string()));
        self.push_instant(stage_index, "batch_arrival", Some(args));
    }

    /// Record an instant event for an activation send from `src_stage`.
    pub fn record_activation_send(&self, src_stage: u32, dst_stage: u32, bytes: usize) {
        let mut args = TraceArgs::default();
        args.fields
            .push(("dst_stage".to_string(), dst_stage.to_string()));
        args.fields.push(("bytes".to_string(), bytes.to_string()));
        self.push_instant(src_stage, "activation_send", Some(args));
    }

    /// Record an instant event for an activation receive on `dst_stage`.
    pub fn record_activation_recv(&self, src_stage: u32, dst_stage: u32, bytes: usize) {
        let mut args = TraceArgs::default();
        args.fields
            .push(("src_stage".to_string(), src_stage.to_string()));
        args.fields.push(("bytes".to_string(), bytes.to_string()));
        self.push_instant(dst_stage, "activation_recv", Some(args));
    }

    /// Record an instant event for a KV cache admission rejection.
    pub fn record_admission_reject(
        &self,
        stage_index: u32,
        reason: &str,
        required_bytes: u64,
        available_bytes: u64,
    ) {
        let mut args = TraceArgs::default();
        args.fields.push(("reason".to_string(), reason.to_string()));
        args.fields
            .push(("required_bytes".to_string(), required_bytes.to_string()));
        args.fields
            .push(("available_bytes".to_string(), available_bytes.to_string()));
        self.push_instant(stage_index, "admission_reject", Some(args));
    }

    /// Force the accumulated events to disk. Idempotent — subsequent calls
    /// after the first successful flush are no-ops.
    pub fn flush(&self) -> Result<()> {
        let mut flushed = self.flushed.lock().expect("pp tracer flush flag poisoned");
        if *flushed {
            return Ok(());
        }
        let events = self
            .events
            .lock()
            .expect("pp tracer event buffer poisoned")
            .clone();
        let payload = serde_json::json!({ "traceEvents": events });
        let rendered = serde_json::to_string(&payload)
            .context("failed to serialise chrome-tracing payload")?;
        if let Some(parent) = self.output_path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create trace directory {parent:?}"))?;
        }
        std::fs::write(&self.output_path, rendered)
            .with_context(|| format!("failed to write trace file {:?}", self.output_path))?;
        *flushed = true;
        Ok(())
    }

    fn push_instant(&self, stage_index: u32, name: &str, args: Option<TraceArgs>) {
        let ts = self.elapsed_us(Instant::now());
        let event = TraceEvent {
            name: name.to_string(),
            category: TRACE_CATEGORY,
            phase: PHASE_INSTANT,
            timestamp_us: ts,
            duration_us: None,
            pid: TRACE_PID,
            tid: u64::from(stage_index),
            args,
        };
        self.push_event(event);
    }

    fn push_event(&self, event: TraceEvent) {
        self.events
            .lock()
            .expect("pp tracer event buffer poisoned")
            .push(event);
    }

    fn elapsed_us(&self, at: Instant) -> u64 {
        duration_as_us(at.duration_since(self.started_at))
    }
}

impl Drop for PpTracer {
    fn drop(&mut self) {
        if let Err(err) = self.flush() {
            // `Drop` cannot propagate errors; surface via tracing and
            // continue. We intentionally do not panic because losing a
            // trace should never crash production.
            tracing::warn!(
                target: "pp-trace",
                path = %self.output_path.display(),
                error = %err,
                "failed to flush chrome-tracing file on drop"
            );
        }
    }
}

fn duration_as_us(duration: Duration) -> u64 {
    duration.as_micros().min(u128::from(u64::MAX)) as u64
}

#[cfg(test)]
#[path = "trace_tests.rs"]
mod tests;
