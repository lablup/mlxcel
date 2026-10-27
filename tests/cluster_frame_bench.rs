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

//! Loopback micro-benchmark of the per-frame protection candidates (#2656).
//!
//! Runs the activation-transfer harness in round-trip mode over authenticated
//! loopback `TcpTransport`s for the unprotected baseline and every
//! `FrameSuite`, at the three payload sizes the issue names: 8 KiB (one decode
//! token, hidden 4096, fp16), 64 KiB, and 16 MiB (a 2048-token prefill), plus
//! 1 MiB and 2 MiB for the JSON-encoded per-token logits reply.
//! Prints the markdown table used in `docs/distributed.md`. Run it alone on a
//! quiet machine (and under `gpu-lock` on GB10 hosts):
//!
//! ```bash
//! cargo test --profile test-fast --features cuda -p mlxcel \
//!   --test cluster_frame_bench -- --ignored --nocapture --test-threads=1
//! ```

use mlxcel::distributed::bench_activation::{
    measure_frame_protection_candidates, render_markdown_report,
};

#[tokio::test]
#[ignore = "benchmark: run manually on a quiet machine"]
async fn frame_protection_candidates_loopback() {
    // 1 MiB and 2 MiB bracket the JSON-encoded `RunResult` frame the
    // coordinator receives per decode token from a 150k-vocabulary model
    // (f16 and f32 logits, about 3.6 JSON characters per byte).
    let scenarios: &[(usize, usize)] = &[
        (8 * 1024, 4000),
        (64 * 1024, 2000),
        (1024 * 1024, 400),
        (2 * 1024 * 1024, 200),
        (16 * 1024 * 1024, 60),
    ];
    let reports = measure_frame_protection_candidates(scenarios, 50)
        .await
        .unwrap();
    println!(
        "{}",
        render_markdown_report(
            "Per-frame protection candidates (loopback, round trip / 2)",
            &reports,
            "Two authenticated `TcpTransport`s on 127.0.0.1; the server echoes every \
             `TensorData` message back and half the round trip is recorded as the one-way \
             latency, so the numbers include both ends' per-frame work.",
        )
    );
    for report in &reports {
        println!(
            "FRAME_BENCH mode={} payload={} iters={} p50_us={:.1} p95_us={:.1} p99_us={:.1} gib_s={:.3}",
            report.label,
            report.payload_bytes,
            report.iterations,
            report.p50.as_secs_f64() * 1e6,
            report.p95.as_secs_f64() * 1e6,
            report.p99.as_secs_f64() * 1e6,
            report.bandwidth_gib_per_s,
        );
    }
}
