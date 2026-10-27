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

//! End-to-end greedy decode throughput of the two-stage remote pipeline over
//! loopback TCP (#2656). Both stages and the coordinator run in this process,
//! but every activation crosses an authenticated `TcpTransport` connection, so
//! the number includes the per-frame protection cost of the cluster transport.
//!
//! Run it only on a quiet machine (and under `gpu-lock` on GB10 hosts):
//!
//! ```bash
//! cargo test --profile test-fast --features cuda -p mlxcel \
//!   --test pipeline_remote_decode_bench -- --ignored --nocapture --test-threads=1
//! ```

mod common;

use std::net::IpAddr;
use std::time::{Duration, Instant};

use common::{repo_model_dir, test_cluster_auth};
use mlxcel::distributed::TransportBackend;
use mlxcel::distributed::pipeline::{
    PipelineModelRuntime, RemotePipelineRuntime, RemotePipelineRuntimeConfig,
    RemoteStageServiceConfig, RemoteStageServiceHandle, StageAssignment,
    resolve_in_process_pipeline_num_layers,
};
use mlxcel_core::cache::SequenceId;

const MODEL: &str = "qwen3-0.6b-4bit";
const PROMPT: &[i32] = &[
    785, 6722, 315, 9625, 374, 12095, 13, 576, 6722, 315, 9856, 374, 19846, 13, 576, 6722,
];
const DECODE_TOKENS: usize = 128;
const REPEATS: usize = 7;
/// Fixed loopback ports (coordinator, stage 0, stage 1): the bench is meant to
/// run alone on a quiet machine, so ephemeral reservation is not needed.
const PORT_BASE: u16 = 19590;

fn two_stage_assignments(num_layers: usize, split: usize) -> [StageAssignment; 2] {
    [
        StageAssignment {
            stage_index: 0,
            device_id: "stage-0".to_string(),
            layer_range: 0..split,
            has_embedding: true,
            has_lm_head: false,
            estimated_memory_bytes: 0,
        },
        StageAssignment {
            stage_index: 1,
            device_id: "stage-1".to_string(),
            layer_range: split..num_layers,
            has_embedding: false,
            has_lm_head: true,
            estimated_memory_bytes: 0,
        },
    ]
}

fn argmax_last_token(logits: &mlxcel_core::MlxArray) -> i32 {
    let shape = mlxcel_core::array_shape(logits);
    let last = mlxcel_core::slice(logits, &[0, shape[1] - 1, 0], &[1, shape[1], shape[2]]);
    let argmax = mlxcel_core::argmax_last_axis(&last);
    mlxcel_core::eval(&argmax);
    mlxcel_core::item_i32(&argmax)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

#[test]
#[ignore = "benchmark: requires local model weights and a quiet machine"]
fn pipeline_remote_qwen3_decode_throughput() {
    let model_dir = repo_model_dir(MODEL);
    if !model_dir.exists() {
        if std::env::var_os("MLXCEL_REQUIRE_MODELS").is_some() {
            panic!("model directory not found at {}", model_dir.display());
        }
        eprintln!("Skipping bench: {} not found", model_dir.display());
        return;
    }

    let bind_ip: IpAddr = "127.0.0.1".parse().unwrap();
    let num_layers = resolve_in_process_pipeline_num_layers(&model_dir).unwrap();
    let assignments = two_stage_assignments(num_layers, num_layers / 2);
    let auth = test_cluster_auth();
    let coordinator_addr = format!("{bind_ip}:{PORT_BASE}");
    let stage0_addr = format!("{bind_ip}:{}", PORT_BASE + 1);
    let stage1_addr = format!("{bind_ip}:{}", PORT_BASE + 2);

    let stage1 = RemoteStageServiceHandle::spawn(RemoteStageServiceConfig {
        model_dir: model_dir.clone(),
        bind_address: stage1_addr.clone(),
        transport_backend: TransportBackend::Tcp,
        stage_assignment: assignments[1].clone(),
        num_stages: 2,
        coordinator_peer: coordinator_addr.clone(),
        upstream_peer: Some(stage0_addr.clone()),
        downstream_peer: None,
        auth: std::sync::Arc::clone(&auth),
    })
    .unwrap();
    let stage0 = RemoteStageServiceHandle::spawn(RemoteStageServiceConfig {
        model_dir: model_dir.clone(),
        bind_address: stage0_addr.clone(),
        transport_backend: TransportBackend::Tcp,
        stage_assignment: assignments[0].clone(),
        num_stages: 2,
        coordinator_peer: coordinator_addr.clone(),
        upstream_peer: None,
        downstream_peer: Some(stage1_addr.clone()),
        auth: std::sync::Arc::clone(&auth),
    })
    .unwrap();
    let runtime = RemotePipelineRuntime::new(RemotePipelineRuntimeConfig {
        stage_peers: vec![stage0_addr, stage1_addr],
        transport_backend: TransportBackend::Tcp,
        bind_address: coordinator_addr,
        stage_timeout: Duration::from_secs(30),
        auth,
        vocab_size: mlxcel::read_vocab_size(&model_dir),
    })
    .unwrap();

    let prompt_ids = mlxcel_core::from_slice_i32(PROMPT, &[1, PROMPT.len() as i32]);
    let mut rates = Vec::with_capacity(REPEATS);
    // Repeat 0 is a warm-up (kernel compilation, connection setup) and is
    // reported but excluded from the median.
    for repeat in 0..=REPEATS {
        let seq_id = SequenceId::from_raw(9_000 + repeat as u64);
        runtime.prepare_sequence_state(seq_id);
        let logits = runtime.forward_sequence(seq_id, &prompt_ids, None).unwrap();
        let mut token = argmax_last_token(&logits);
        let mut generated = Vec::with_capacity(DECODE_TOKENS);
        let start = Instant::now();
        for _ in 0..DECODE_TOKENS {
            let ids = mlxcel_core::from_slice_i32(&[token], &[1, 1]);
            let logits = runtime.forward_sequence(seq_id, &ids, None).unwrap();
            token = argmax_last_token(&logits);
            generated.push(token);
        }
        let elapsed = start.elapsed().as_secs_f64();
        runtime.release_sequence_state_by_id(seq_id);
        let tps = DECODE_TOKENS as f64 / elapsed;
        println!(
            "DECODE_BENCH repeat={repeat} tokens={DECODE_TOKENS} secs={elapsed:.4} tok_per_s={tps:.2} first_tokens={:?}",
            &generated[..8]
        );
        if repeat > 0 {
            rates.push(tps);
        }
    }
    println!(
        "DECODE_BENCH median_tok_per_s={:.2} repeats={REPEATS}",
        median(&mut rates)
    );

    runtime.shutdown().unwrap();
    stage0.shutdown().unwrap();
    stage1.shutdown().unwrap();
}
