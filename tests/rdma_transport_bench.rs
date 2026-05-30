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

//! 2-node loopback activation-transfer benchmark harness.
//!
//! Run with:
//!   cargo test --release --test rdma_transport_bench rdma_vs_tcp_activation_transfer \
//!       -- --ignored --nocapture
//!
//! The harness exercises `TcpTransport` and `RdmaTransport` back-to-back
//! against a loopback peer and prints a markdown table suitable for pasting
//! into `docs_internal/performance/`.

use std::sync::Arc;
use std::time::Duration;

use mlxcel::distributed::bench_activation::{
    ActivationBenchConfig, measure_activation_transfer, render_markdown_report, spawn_drain_task,
};
use mlxcel::distributed::tcp_transport::{TcpTransport, TcpTransportConfig};
use mlxcel::distributed::{RdmaTransport, RdmaTransportConfig, Transport};

async fn run_bench_on(
    transport: &dyn Transport,
    peer: &str,
    label: &str,
    payload: usize,
    iterations: usize,
) -> mlxcel::distributed::bench_activation::ActivationTransferReport {
    let config = ActivationBenchConfig {
        payload_bytes: payload,
        iterations,
        warmup: 16,
        label: label.to_string(),
    };
    measure_activation_transfer(transport, peer, &config)
        .await
        .unwrap()
}

#[tokio::test]
#[ignore = "2-node loopback micro-benchmark, run manually for docs_internal/performance/"]
async fn rdma_vs_tcp_activation_transfer() {
    // Each scenario mirrors the activation shapes the pipeline runtime sees
    // in practice: small (control), medium (prefill micro-batch), large
    // (per-stage hidden state). Keep the iteration count moderate so the
    // test still completes in seconds on CI-class hardware.
    let scenarios: &[(usize, usize)] = &[
        (64 * 1024, 256),       // 64 KiB — RPC-sized handoff
        (1024 * 1024, 128),     // 1 MiB  — decode-step activation
        (8 * 1024 * 1024, 64),  // 8 MiB  — prefill-sized activation
        (32 * 1024 * 1024, 16), // 32 MiB — stage hand-off worst case
    ];

    // ----- TCP baseline -----
    let tcp_server = TcpTransport::bind(TcpTransportConfig {
        bind_address: "127.0.0.1:0".to_string(),
        ..Default::default()
    })
    .await
    .unwrap();
    let tcp_server_addr = tcp_server.local_addr().unwrap();
    let tcp_server: Arc<dyn Transport> = Arc::new(tcp_server);
    let tcp_client = TcpTransport::bind(TcpTransportConfig {
        bind_address: "127.0.0.1:0".to_string(),
        ..Default::default()
    })
    .await
    .unwrap();
    tcp_client
        .connect(std::slice::from_ref(&tcp_server_addr))
        .await
        .unwrap();
    let drain_tcp = spawn_drain_task(tcp_server.clone());

    let mut reports = Vec::new();
    for (payload, iters) in scenarios.iter().copied() {
        let label = format!("2-node loopback ({payload} B)");
        let report = run_bench_on(&tcp_client, &tcp_server_addr, &label, payload, iters).await;
        reports.push(report);
    }

    drain_tcp.shutdown().await;
    tcp_client.shutdown().await.unwrap();
    tcp_server.shutdown().await.unwrap();

    // Small settle so the OS releases the previous sockets.
    tokio::time::sleep(Duration::from_millis(50)).await;

    // ----- RDMA wrapper -----
    let rdma_server = RdmaTransport::bind(RdmaTransportConfig {
        bind_address: "127.0.0.1:0".to_string(),
        prefer_thunderbolt: false,
        tcp_config: None,
        max_negotiation_failures: 4,
    })
    .await
    .unwrap();
    let rdma_server_addr = rdma_server.local_addr().unwrap();
    let rdma_server: Arc<dyn Transport> = Arc::new(rdma_server);
    let rdma_client = RdmaTransport::bind(RdmaTransportConfig {
        bind_address: "127.0.0.1:0".to_string(),
        prefer_thunderbolt: false,
        tcp_config: None,
        max_negotiation_failures: 4,
    })
    .await
    .unwrap();
    rdma_client
        .connect(std::slice::from_ref(&rdma_server_addr))
        .await
        .unwrap();
    let drain_rdma = spawn_drain_task(rdma_server.clone());

    for (payload, iters) in scenarios.iter().copied() {
        let label = format!("2-node loopback ({payload} B)");
        let report = run_bench_on(&rdma_client, &rdma_server_addr, &label, payload, iters).await;
        reports.push(report);
    }

    drain_rdma.shutdown().await;
    rdma_client.shutdown().await.unwrap();
    rdma_server.shutdown().await.unwrap();

    let methodology = "\
Each scenario sends a single-stream activation payload from a loopback \
client to a loopback server. The server runs a dedicated drain task to \
avoid backpressure accounting skew; the client records one-way send \
latency (header frame + `write_all`). Percentiles are computed post-hoc \
from the sorted sample set; bandwidth is `total_bytes / wall_clock`. \
Both backends run with `release` optimisation against 127.0.0.1 on the \
same host, so the numbers reflect transport overhead above the raw \
loopback write path — not a cross-machine run.";
    let md = render_markdown_report("RDMA vs TCP Activation Transfer", &reports, methodology);
    println!("{md}");
}
