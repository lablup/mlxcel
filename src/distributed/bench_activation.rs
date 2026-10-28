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
//!
//! Two measurement shapes are supported. The default records how long
//! `send()` takes to hand each frame to the kernel, which is the sender-side
//! cost on the pipeline's critical path. With
//! [`ActivationBenchConfig::round_trip`] the peer echoes every message back
//! ([`spawn_echo_task`]) and half the round trip is recorded as the one-way
//! latency, which includes the receiver's deserialization and, with cluster
//! authentication, both ends' per-frame protection work.
//!
//! [`FrameProtectionMode`] and [`measure_frame_protection_candidates`] run
//! the same harness over authenticated loopback transports for every
//! per-frame protection candidate of #2656 (unprotected baseline, HMAC-SHA256,
//! AES-256-GCM, ChaCha20-Poly1305) on the real `TcpTransport`, so the numbers
//! that pick the production suite come from the code path that ships.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use bytes::Bytes;
use tokio::sync::Mutex;

use super::cluster_auth::{ClusterAuth, ClusterSecret};
use super::frame_protection::FrameSuite;
use super::tcp_transport::{TcpTransport, TcpTransportConfig};
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
    /// Wait for the peer to echo each message back (see
    /// [`spawn_echo_task`]) and record half the round trip as the one-way
    /// latency. Off by default: the plain mode records send completion only.
    pub round_trip: bool,
}

impl Default for ActivationBenchConfig {
    fn default() -> Self {
        Self {
            payload_bytes: 4 * 1024 * 1024, // 4 MiB hidden-state micro-batch
            iterations: 200,
            warmup: 20,
            label: "default".to_string(),
            round_trip: false,
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
            "| {} | {} | {:>10} | {:>8} | {:>8.3} ms | {:>8.3} ms | {:>8.3} ms | {:>8.2} GiB/s |",
            self.backend,
            self.label,
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
        "| Backend | Label | Payload | Iters | p50 | p95 | p99 | Bandwidth |\n\
         |---------|-------|--------:|------:|----:|----:|----:|----------:|"
    }
}

/// Upper bound on one echo in round-trip mode; a lost echo fails the run
/// instead of hanging it.
const ECHO_TIMEOUT: Duration = Duration::from_secs(30);

/// Run a one-way activation transfer benchmark against the provided
/// transport. The peer must already be connected and a receiver task must be
/// draining inbound messages (or echoing them, in round-trip mode); otherwise
/// the transport will backpressure the sender and the measurement will track
/// drain latency, not transport latency.
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
        if config.round_trip {
            await_echo(transport, i).await?;
        }
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
        if config.round_trip {
            await_echo(transport, i).await?;
            samples.push(start.elapsed() / 2);
        } else {
            samples.push(start.elapsed());
        }
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

    let directions = if config.round_trip { 2 } else { 1 };
    let total_bytes = (config.payload_bytes * n * directions) as f64;
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

/// Wait for the echo of iteration `i` on the sender's own transport.
async fn await_echo(transport: &dyn Transport, i: usize) -> Result<()> {
    tokio::time::timeout(ECHO_TIMEOUT, transport.recv())
        .await
        .map_err(|_| anyhow::anyhow!("no echo for iteration {i} within {ECHO_TIMEOUT:?}"))?
        .with_context(|| format!("echo recv #{i} failed"))?;
    Ok(())
}

/// Echo peer for round-trip measurements: every message received on
/// `transport` is sent back unchanged to `reply_to` (the sender's listener
/// address), so the sender's `recv()` completes one round trip. The reply
/// crosses a second authenticated connection in the reverse direction, so
/// round-trip mode exercises both ends' frame protection.
pub fn spawn_echo_task(transport: Arc<dyn Transport>, reply_to: String) -> DrainHandle {
    let stop = Arc::new(Mutex::new(false));
    let stop_clone = stop.clone();
    let handle = tokio::spawn(async move {
        loop {
            if *stop_clone.lock().await {
                return;
            }
            match tokio::time::timeout(Duration::from_millis(500), transport.recv()).await {
                Ok(Ok((_, msg))) => {
                    if transport.send(&reply_to, msg).await.is_err() {
                        return;
                    }
                }
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

/// One per-frame protection candidate of #2656, as measured on the real
/// `TcpTransport`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameProtectionMode {
    /// No cluster authentication: the frame layout before #2656, used as the
    /// baseline (the handshake itself costs nothing per frame).
    Unprotected,
    /// Cluster authentication with the given per-frame suite.
    Protected(FrameSuite),
}

impl FrameProtectionMode {
    /// Baseline followed by every suite.
    pub const ALL: [FrameProtectionMode; 4] = [
        FrameProtectionMode::Unprotected,
        FrameProtectionMode::Protected(FrameSuite::HmacSha256),
        FrameProtectionMode::Protected(FrameSuite::Aes256Gcm),
        FrameProtectionMode::Protected(FrameSuite::ChaCha20Poly1305),
    ];

    pub fn label(self) -> String {
        match self {
            Self::Unprotected => "unprotected".to_string(),
            Self::Protected(suite) => suite.label().to_string(),
        }
    }
}

/// Two loopback `TcpTransport`s configured for `mode`, the client already
/// connected to the server.
pub struct LoopbackPair {
    pub client: Arc<TcpTransport>,
    pub server: Arc<TcpTransport>,
    pub client_addr: String,
    pub server_addr: String,
}

impl LoopbackPair {
    pub async fn bind(mode: FrameProtectionMode) -> Result<Self> {
        let auth = match mode {
            FrameProtectionMode::Unprotected => None,
            FrameProtectionMode::Protected(_) => {
                Some(Arc::new(ClusterAuth::new(&ClusterSecret::random())))
            }
        };
        let frame_suite = match mode {
            FrameProtectionMode::Protected(suite) => suite,
            FrameProtectionMode::Unprotected => FrameSuite::default(),
        };
        let config = |auth: Option<Arc<ClusterAuth>>| TcpTransportConfig {
            bind_address: "127.0.0.1:0".to_string(),
            auth,
            frame_suite,
            ..Default::default()
        };
        let server = Arc::new(TcpTransport::bind(config(auth.clone())).await?);
        let client = Arc::new(TcpTransport::bind(config(auth)).await?);
        let server_addr = server.local_addr()?;
        let client_addr = client.local_addr()?;
        client.connect(std::slice::from_ref(&server_addr)).await?;
        Ok(Self {
            client,
            server,
            client_addr,
            server_addr,
        })
    }

    pub async fn shutdown(self) -> Result<()> {
        self.client.shutdown().await?;
        self.server.shutdown().await?;
        Ok(())
    }
}

/// Measure every [`FrameProtectionMode`] over loopback for each
/// `(payload_bytes, iterations)` scenario in round-trip mode (one-way latency
/// includes both ends' frame work). Reports are labelled with the mode and
/// ordered mode-major, scenario-minor.
pub async fn measure_frame_protection_candidates(
    scenarios: &[(usize, usize)],
    warmup: usize,
) -> Result<Vec<ActivationTransferReport>> {
    let mut reports = Vec::with_capacity(FrameProtectionMode::ALL.len() * scenarios.len());
    for mode in FrameProtectionMode::ALL {
        let pair = LoopbackPair::bind(mode).await?;
        let echo = spawn_echo_task(
            Arc::clone(&pair.server) as Arc<dyn Transport>,
            pair.client_addr.clone(),
        );
        for &(payload_bytes, iterations) in scenarios {
            let config = ActivationBenchConfig {
                payload_bytes,
                iterations,
                warmup,
                label: mode.label(),
                round_trip: true,
            };
            reports.push(
                measure_activation_transfer(pair.client.as_ref(), &pair.server_addr, &config)
                    .await
                    .with_context(|| format!("{} at {payload_bytes} bytes", mode.label()))?,
            );
        }
        echo.shutdown().await;
        pair.shutdown().await?;
    }
    Ok(reports)
}

#[cfg(test)]
#[path = "bench_activation_tests.rs"]
mod tests;
