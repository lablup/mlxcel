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

//! Benchmarking harness for transport throughput and latency measurement.
//!
//! Provides utilities to measure the performance characteristics of any
//! [`Transport`] implementation between two nodes.

use std::time::{Duration, Instant};

use anyhow::Result;
use bytes::Bytes;

use super::transport::{Transport, TransportMessage};

/// Configuration for a benchmark run.
#[derive(Debug, Clone)]
pub struct BenchConfig {
    /// Size of each payload in bytes.
    pub payload_size: usize,
    /// Number of messages to send in the throughput test.
    pub num_messages: usize,
    /// Number of RPC roundtrips for the latency test.
    pub num_rpc_roundtrips: usize,
    /// Number of warmup iterations before measuring.
    pub warmup_iterations: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            payload_size: 1024 * 1024, // 1 MiB
            num_messages: 100,
            num_rpc_roundtrips: 100,
            warmup_iterations: 5,
        }
    }
}

/// Results from a benchmark run.
#[derive(Debug, Clone)]
pub struct BenchResult {
    /// Transport backend name.
    pub backend: String,
    /// Throughput test results (if run).
    pub throughput: Option<ThroughputResult>,
    /// Latency test results (if run).
    pub latency: Option<LatencyResult>,
}

/// Throughput measurement results.
#[derive(Debug, Clone)]
pub struct ThroughputResult {
    /// Payload size per message in bytes.
    pub payload_size: usize,
    /// Number of messages sent.
    pub num_messages: usize,
    /// Total bytes transferred.
    pub total_bytes: u64,
    /// Wall-clock duration.
    pub duration: Duration,
    /// Throughput in bytes per second.
    pub bytes_per_second: f64,
    /// Throughput in megabytes per second.
    pub mb_per_second: f64,
    /// Throughput in gigabits per second.
    pub gbps: f64,
}

impl std::fmt::Display for ThroughputResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Throughput: {:.2} MB/s ({:.2} Gbps) | {} messages x {} bytes in {:.2}s",
            self.mb_per_second,
            self.gbps,
            self.num_messages,
            self.payload_size,
            self.duration.as_secs_f64()
        )
    }
}

/// Latency measurement results.
#[derive(Debug, Clone)]
pub struct LatencyResult {
    /// Number of roundtrips measured.
    pub num_roundtrips: usize,
    /// Minimum roundtrip time.
    pub min: Duration,
    /// Maximum roundtrip time.
    pub max: Duration,
    /// Mean roundtrip time.
    pub mean: Duration,
    /// Median roundtrip time.
    pub median: Duration,
    /// 95th percentile roundtrip time.
    pub p95: Duration,
    /// 99th percentile roundtrip time.
    pub p99: Duration,
}

impl std::fmt::Display for LatencyResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Latency ({} roundtrips): min={:.2?} mean={:.2?} median={:.2?} p95={:.2?} p99={:.2?} max={:.2?}",
            self.num_roundtrips, self.min, self.mean, self.median, self.p95, self.p99, self.max,
        )
    }
}

/// Run a throughput benchmark by sending `config.num_messages` tensor payloads
/// through the transport.
///
/// The caller must ensure that a receiver is consuming messages on the other
/// end (or the transport will block/drop depending on backpressure).
///
/// Returns the throughput measurement.
pub async fn measure_throughput(
    transport: &dyn Transport,
    peer: &str,
    config: &BenchConfig,
) -> Result<ThroughputResult> {
    let payload = Bytes::from(vec![0xABu8; config.payload_size]);

    // Warmup.
    for _ in 0..config.warmup_iterations {
        let msg = TransportMessage::TensorData {
            tensor_id: "bench_warmup".to_string(),
            shape: vec![config.payload_size],
            data: payload.clone(),
        };
        transport.send(peer, msg).await?;
    }

    // Measured run.
    let start = Instant::now();
    for i in 0..config.num_messages {
        let msg = TransportMessage::TensorData {
            tensor_id: format!("bench_{i}"),
            shape: vec![config.payload_size],
            data: payload.clone(),
        };
        transport.send(peer, msg).await?;
    }
    let duration = start.elapsed();

    let total_bytes = (config.payload_size * config.num_messages) as u64;
    let secs = duration.as_secs_f64();
    let bytes_per_second = if secs > 0.0 {
        total_bytes as f64 / secs
    } else {
        0.0
    };

    Ok(ThroughputResult {
        payload_size: config.payload_size,
        num_messages: config.num_messages,
        total_bytes,
        duration,
        bytes_per_second,
        mb_per_second: bytes_per_second / (1024.0 * 1024.0),
        gbps: bytes_per_second * 8.0 / 1_000_000_000.0,
    })
}

/// Run a latency benchmark using RPC roundtrips.
///
/// The peer must have an RPC handler installed that echoes back the request.
pub async fn measure_latency(
    transport: &dyn Transport,
    peer: &str,
    config: &BenchConfig,
) -> Result<LatencyResult> {
    let request = vec![0u8; 64]; // Small payload for latency measurement.

    // Warmup.
    for _ in 0..config.warmup_iterations {
        let _ = transport.rpc_call(peer, &request).await;
    }

    // Measured run.
    let mut latencies = Vec::with_capacity(config.num_rpc_roundtrips);
    for _ in 0..config.num_rpc_roundtrips {
        let start = Instant::now();
        transport.rpc_call(peer, &request).await?;
        latencies.push(start.elapsed());
    }

    latencies.sort();
    let n = latencies.len();

    let min = latencies[0];
    let max = latencies[n - 1];
    let mean = Duration::from_nanos(
        (latencies.iter().map(|d| d.as_nanos()).sum::<u128>() / n as u128) as u64,
    );
    let median = latencies[n / 2];
    let p95 = latencies[((n as f64 * 0.95) as usize).min(n - 1)];
    let p99 = latencies[((n as f64 * 0.99) as usize).min(n - 1)];

    Ok(LatencyResult {
        num_roundtrips: n,
        min,
        max,
        mean,
        median,
        p95,
        p99,
    })
}

/// Run both throughput and latency benchmarks and return combined results.
pub async fn run_benchmark(
    transport: &dyn Transport,
    peer: &str,
    config: &BenchConfig,
) -> Result<BenchResult> {
    let backend = transport.backend().to_string();

    tracing::info!("Starting transport benchmark ({backend})...");
    tracing::info!(
        "  Throughput: {} messages x {} bytes",
        config.num_messages,
        config.payload_size
    );
    tracing::info!("  Latency: {} RPC roundtrips", config.num_rpc_roundtrips);

    let throughput = match measure_throughput(transport, peer, config).await {
        Ok(t) => {
            tracing::info!("  {t}");
            Some(t)
        }
        Err(e) => {
            tracing::warn!("  Throughput measurement failed: {e}");
            None
        }
    };

    let latency = match measure_latency(transport, peer, config).await {
        Ok(l) => {
            tracing::info!("  {l}");
            Some(l)
        }
        Err(e) => {
            tracing::warn!("  Latency measurement failed: {e}");
            None
        }
    };

    Ok(BenchResult {
        backend,
        throughput,
        latency,
    })
}

/// Print a human-readable summary of benchmark results.
pub fn print_results(results: &BenchResult) {
    println!("=== Transport Benchmark: {} ===", results.backend);
    if let Some(ref t) = results.throughput {
        println!("  {t}");
    }
    if let Some(ref l) = results.latency {
        println!("  {l}");
    }
    println!("===");
}

#[cfg(test)]
#[path = "bench_tests.rs"]
mod tests;
