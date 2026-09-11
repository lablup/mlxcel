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

//! Activation-transfer benchmark harness for pipeline-parallel transport
//! backends.
//!
//! Unlike the generic [`super::bench`] harness, this module focuses on the
//! exact payload shape the pipeline runtime actually ships between stages:
//! an [`super::transport::TransportMessage::TensorData`] carrying a
//! contiguous buffer sized to a realistic hidden-state micro-batch. The
//! harness measures p50/p95/p99 one-way latency and bandwidth for a 2-node
//! setup and returns a structured [`ActivationTransferReport`] that the
//! benchmark binary can turn into a markdown report for
//! `docs_internal/performance/`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use bytes::Bytes;
use tokio::sync::Mutex;

use super::transport::{Transport, TransportBackend, TransportMessage};

/// Parameters for an activation-transfer benchmark pass.
#[derive(Debug, Clone)]
pub struct ActivationBenchConfig {
    /// Activation payload size in bytes (e.g. `hidden * seq * bytes_per_elem`).
    pub payload_bytes: usize,
    /// Number of measured iterations.
    pub iterations: usize,
    /// Warmup iterations (not counted in the final statistics).
    pub warmup: usize,
    /// Optional label for the measurement (`"2-node loopback"` etc.).
    pub label: String,
}

impl Default for ActivationBenchConfig {
    fn default() -> Self {
        Self {
            payload_bytes: 4 * 1024 * 1024, // 4 MiB hidden-state micro-batch
            iterations: 200,
            warmup: 20,
            label: "default".to_string(),
        }
    }
}

/// Latency / bandwidth statistics derived from a benchmark pass.
#[derive(Debug, Clone)]
pub struct ActivationTransferReport {
    pub backend: TransportBackend,
    pub label: String,
    pub payload_bytes: usize,
    pub iterations: usize,
    pub p50: Duration,
    pub p95: Duration,
    pub p99: Duration,
    pub min: Duration,
    pub max: Duration,
    pub mean: Duration,
    pub bandwidth_gib_per_s: f64,
}

impl ActivationTransferReport {
    /// Markdown row for rendering comparison tables in the performance
    /// report. Leaves out the backend column so callers can embed multiple
    /// rows grouped by scenario.
    pub fn markdown_row(&self) -> String {
        format!(
            "| {} | {:>10} | {:>8} | {:>8.3} ms | {:>8.3} ms | {:>8.3} ms | {:>8.2} GiB/s |",
            self.backend,
            human_bytes(self.payload_bytes),
            self.iterations,
            dur_ms(self.p50),
            dur_ms(self.p95),
            dur_ms(self.p99),
            self.bandwidth_gib_per_s,
        )
    }

    /// Markdown table header matching [`Self::markdown_row`].
    pub fn markdown_header() -> &'static str {
        "| Backend | Payload | Iters | p50 | p95 | p99 | Bandwidth |\n\
         |---------|--------:|------:|----:|----:|----:|----------:|"
    }
}

/// Run a one-way activation transfer benchmark against the provided
/// transport. The peer must already be connected and a receiver task must be
/// draining inbound messages — otherwise the transport will backpressure the
/// sender and the measurement will track drain latency, not transport
/// latency.
pub async fn measure_activation_transfer(
    transport: &dyn Transport,
    peer: &str,
    config: &ActivationBenchConfig,
) -> Result<ActivationTransferReport> {
    let payload = Bytes::from(vec![0xABu8; config.payload_bytes]);

    for i in 0..config.warmup {
        let msg = TransportMessage::TensorData {
            tensor_id: format!("warmup-{i}"),
            shape: vec![config.payload_bytes],
            data: payload.clone(),
        };
        transport
            .send(peer, msg)
            .await
            .with_context(|| format!("warmup send #{i} failed"))?;
    }

    let mut samples: Vec<Duration> = Vec::with_capacity(config.iterations);
    let run_start = Instant::now();
    for i in 0..config.iterations {
        let start = Instant::now();
        let msg = TransportMessage::TensorData {
            tensor_id: format!("iter-{i}"),
            shape: vec![config.payload_bytes],
            data: payload.clone(),
        };
        transport
            .send(peer, msg)
            .await
            .with_context(|| format!("measured send #{i} failed"))?;
        samples.push(start.elapsed());
    }
    let total = run_start.elapsed();

    samples.sort();
    let n = samples.len();
    let min = samples[0];
    let max = samples[n - 1];
    let sum_ns: u128 = samples.iter().map(|d| d.as_nanos()).sum();
    let mean = Duration::from_nanos((sum_ns / n as u128) as u64);
    let p50 = samples[(n as f64 * 0.50) as usize];
    let p95 = samples[((n as f64 * 0.95) as usize).min(n - 1)];
    let p99 = samples[((n as f64 * 0.99) as usize).min(n - 1)];

    let total_bytes = (config.payload_bytes * n) as f64;
    let total_secs = total.as_secs_f64();
    let bandwidth_gib_per_s = if total_secs > 0.0 {
        total_bytes / total_secs / (1024.0 * 1024.0 * 1024.0)
    } else {
        0.0
    };

    Ok(ActivationTransferReport {
        backend: transport.backend(),
        label: config.label.clone(),
        payload_bytes: config.payload_bytes,
        iterations: n,
        p50,
        p95,
        p99,
        min,
        max,
        mean,
        bandwidth_gib_per_s,
    })
}

/// Shared drain: spawns a task that keeps calling `recv()` on the transport
/// so the sender is not blocked on backpressure during measurement.
///
/// Returns a shutdown handle that halts the drain task cleanly.
pub fn spawn_drain_task(transport: Arc<dyn Transport>) -> DrainHandle {
    let stop = Arc::new(Mutex::new(false));
    let stop_clone = stop.clone();
    let handle = tokio::spawn(async move {
        loop {
            if *stop_clone.lock().await {
                return;
            }
            match tokio::time::timeout(Duration::from_millis(500), transport.recv()).await {
                Ok(Ok(_)) => {}
                Ok(Err(_)) => return,
                Err(_) => {}
            }
        }
    });
    DrainHandle { stop, handle }
}

/// Handle controlling a spawned [`spawn_drain_task`].
pub struct DrainHandle {
    stop: Arc<Mutex<bool>>,
    handle: tokio::task::JoinHandle<()>,
}

impl DrainHandle {
    pub async fn shutdown(self) {
        *self.stop.lock().await = true;
        let _ = self.handle.await;
    }
}

/// Human-readable byte count for markdown rendering.
fn human_bytes(n: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;
    if n >= GB {
        format!("{:.1} GiB", n as f64 / GB as f64)
    } else if n >= MB {
        format!("{:.1} MiB", n as f64 / MB as f64)
    } else if n >= KB {
        format!("{:.1} KiB", n as f64 / KB as f64)
    } else {
        format!("{n} B")
    }
}

/// Duration to fractional milliseconds for markdown rendering.
fn dur_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Render a comparison report as a markdown document suitable for
/// `docs_internal/performance/`.
pub fn render_markdown_report(
    title: &str,
    reports: &[ActivationTransferReport],
    methodology: &str,
) -> String {
    use std::fmt::Write;

    let mut out = String::new();
    let _ = writeln!(out, "# {title}");
    let _ = writeln!(out);
    let _ = writeln!(out, "## Methodology");
    let _ = writeln!(out);
    let _ = writeln!(out, "{methodology}");
    let _ = writeln!(out);
    let _ = writeln!(out, "## Results");
    let _ = writeln!(out);
    let _ = writeln!(out, "{}", ActivationTransferReport::markdown_header());
    for report in reports {
        let _ = writeln!(out, "{}", report.markdown_row());
    }
    let _ = writeln!(out);
    out
}

#[cfg(test)]
#[path = "bench_activation_tests.rs"]
mod tests;
