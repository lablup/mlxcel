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

//! Tests for the activation-transfer benchmark harness.
//!
//! These tests exercise the statistics and markdown rendering on synthetic
//! data so they stay fast and hermetic — the real bench binary is exercised
//! separately when operators commit a report to
//! `docs_internal/performance/`.

use std::sync::Arc;
use std::time::Duration;

use super::{
    ActivationBenchConfig, ActivationTransferReport, measure_activation_transfer,
    render_markdown_report, spawn_drain_task,
};
use crate::distributed::rdma_transport::{RdmaTransport, RdmaTransportConfig};
use crate::distributed::transport::{Transport, TransportBackend};

#[test]
fn markdown_row_renders_stable_shape() {
    let report = ActivationTransferReport {
        backend: TransportBackend::Rdma,
        label: "2-node loopback".to_string(),
        payload_bytes: 4 * 1024 * 1024,
        iterations: 100,
        p50: Duration::from_micros(1500),
        p95: Duration::from_micros(2400),
        p99: Duration::from_micros(3200),
        min: Duration::from_micros(1100),
        max: Duration::from_micros(4500),
        mean: Duration::from_micros(1700),
        bandwidth_gib_per_s: 12.5,
    };
    let row = report.markdown_row();
    assert!(row.contains("rdma"));
    assert!(row.contains("4.0 MiB"));
    assert!(row.contains("1.500 ms"));
    assert!(row.contains("2.400 ms"));
    assert!(row.contains("3.200 ms"));
    assert!(row.contains("12.50 GiB/s"));
}

#[test]
fn render_markdown_report_composes_title_methodology_and_rows() {
    let reports = vec![ActivationTransferReport {
        backend: TransportBackend::Tcp,
        label: "2-node loopback".to_string(),
        payload_bytes: 1024,
        iterations: 10,
        p50: Duration::from_micros(100),
        p95: Duration::from_micros(200),
        p99: Duration::from_micros(300),
        min: Duration::from_micros(80),
        max: Duration::from_micros(400),
        mean: Duration::from_micros(130),
        bandwidth_gib_per_s: 0.01,
    }];
    let md = render_markdown_report(
        "Activation Transfer Benchmark",
        &reports,
        "2-node loopback over `127.0.0.1`, payload 1 KiB.",
    );
    assert!(md.contains("# Activation Transfer Benchmark"));
    assert!(md.contains("## Methodology"));
    assert!(md.contains("## Results"));
    assert!(md.contains("| Backend |"));
    assert!(md.contains("tcp"));
    assert!(md.contains("2-node loopback"));
}

#[tokio::test]
async fn measure_activation_transfer_yields_ordered_percentiles() {
    // Spin up a small 2-transport loopback so the harness can drain in the
    // background while we measure the sender latency distribution.
    let server = RdmaTransport::bind(RdmaTransportConfig {
        bind_address: "127.0.0.1:0".to_string(),
        prefer_thunderbolt: false,
        tcp_config: None,
        max_negotiation_failures: 4,
    })
    .await
    .unwrap();
    let server_addr = server.local_addr().unwrap();
    let server = Arc::new(server) as Arc<dyn Transport>;

    let client = RdmaTransport::bind(RdmaTransportConfig {
        bind_address: "127.0.0.1:0".to_string(),
        prefer_thunderbolt: false,
        tcp_config: None,
        max_negotiation_failures: 4,
    })
    .await
    .unwrap();
    client
        .connect(std::slice::from_ref(&server_addr))
        .await
        .unwrap();

    let drain = spawn_drain_task(server.clone());

    let report = measure_activation_transfer(
        &client,
        &server_addr,
        &ActivationBenchConfig {
            payload_bytes: 16 * 1024,
            iterations: 32,
            warmup: 4,
            label: "unit-test".to_string(),
        },
    )
    .await
    .unwrap();

    assert_eq!(report.iterations, 32);
    assert_eq!(report.payload_bytes, 16 * 1024);
    assert_eq!(report.backend, TransportBackend::Rdma);
    assert!(report.min <= report.p50);
    assert!(report.p50 <= report.p95);
    assert!(report.p95 <= report.p99);
    assert!(report.p99 <= report.max);

    drain.shutdown().await;
    client.shutdown().await.unwrap();
    server.shutdown().await.unwrap();
}
