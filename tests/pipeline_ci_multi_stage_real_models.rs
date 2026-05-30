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

//! Multi-stage pipeline-parallel integration tests driven by the
//! `pipeline-parallel-ci.yml` workflow.
//!
//! These tests exist specifically so the CI harness has a stable,
//! versioned entry point for:
//!
//! - the 2-host logical smoke that exercises the full remote-stage path
//!   (coordinator + 2 stages, all running on loopback TCP within one
//!   runner process) against a real model checkpoint;
//! - the 3-host real-model parity test that the self-hosted lab runs on
//!   demand, also against a real checkpoint;
//! - two heterogeneous-memory partition regressions that run on every PR
//!   without needing a model checkout, so the partitioner and the KV-cache
//!   admission-control decisions stay correct when stage 0 has strictly
//!   less memory than stage 1.
//!
//! The tests that need model weights are `#[ignore]`d so `cargo test`
//! defaults stay cheap; the partition and admission regressions run by
//! default, which is how the heterogeneous-memory scenario from issue
//! becomes a true guardrail on every PR.

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::repo_model_dir;

use mlxcel::SamplingConfig;
use mlxcel::distributed::TransportBackend;
use mlxcel::distributed::pipeline::{
    AdmissionDecision, CacheAdmissionRequest, DeviceSpec, ModelProfile, PipelineCacheConfig,
    PipelineCacheManager, RejectionReason, RemotePipelineRuntimeConfig, RemoteStageServiceConfig,
    RemoteStageServiceHandle, StageAssignment, auto_partition,
    resolve_in_process_pipeline_num_layers, validate_memory_fit, validate_partition,
};
use mlxcel::server::batch::{BatchObservability, RequestPriority};
use mlxcel::server::{
    BatchMetrics, ModelProvider, PipelineParallelRuntimeConfig, ServerConfig, ServerGenerateOptions,
};

/// Reserve an ephemeral loopback TCP port and release it so the caller can
/// bind it next. Same technique as `tests/pipeline_server_remote_real_models.rs`.
fn reserve_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let port = listener.local_addr().expect("local addr").port();
    drop(listener);
    port
}

fn wait_for_loaded(provider: &ModelProvider) {
    let deadline = std::time::Instant::now() + Duration::from_secs(180);
    while std::time::Instant::now() < deadline {
        if provider.is_loaded() {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("model provider did not finish loading within 180s");
}

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

fn three_stage_assignments(num_layers: usize) -> [StageAssignment; 3] {
    assert!(
        num_layers >= 3,
        "three-stage partition requires at least 3 layers, got {num_layers}",
    );
    let third = num_layers / 3;
    let two_thirds = 2 * num_layers / 3;
    [
        StageAssignment {
            stage_index: 0,
            device_id: "stage-0".to_string(),
            layer_range: 0..third.max(1),
            has_embedding: true,
            has_lm_head: false,
            estimated_memory_bytes: 0,
        },
        StageAssignment {
            stage_index: 1,
            device_id: "stage-1".to_string(),
            layer_range: third.max(1)..two_thirds.max(third.max(1) + 1),
            has_embedding: false,
            has_lm_head: false,
            estimated_memory_bytes: 0,
        },
        StageAssignment {
            stage_index: 2,
            device_id: "stage-2".to_string(),
            layer_range: two_thirds.max(third.max(1) + 1)..num_layers,
            has_embedding: false,
            has_lm_head: true,
            estimated_memory_bytes: 0,
        },
    ]
}

/// Helper used by both the CI 2-host smoke and the 3-host real-model parity
/// paths. Spins up `N` remote stage services on loopback and asserts that
/// the remote coordinator returns the same greedy completion as the dense
/// single-device runtime.
fn assert_multi_stage_coordinator_matches_dense_baseline(
    model_name: &str,
    prompt: &str,
    num_stages: usize,
) {
    assert!(num_stages >= 2, "need at least 2 stages");

    let model_dir = repo_model_dir(model_name);
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return;
    }

    let num_layers = resolve_in_process_pipeline_num_layers(&model_dir).unwrap();
    if num_layers < num_stages {
        eprintln!(
            "Skipping test: {model_name} has only {num_layers} layers, which is less than {num_stages} stages",
        );
        return;
    }

    let assignments: Vec<StageAssignment> = if num_stages == 2 {
        two_stage_assignments(num_layers, num_layers / 2).to_vec()
    } else if num_stages == 3 {
        three_stage_assignments(num_layers).to_vec()
    } else {
        // Evenly spread for the general case (only 2 and 3 are wired today).
        let mut v = Vec::with_capacity(num_stages);
        let per = num_layers / num_stages;
        for i in 0..num_stages {
            let start = i * per;
            let end = if i == num_stages - 1 {
                num_layers
            } else {
                (i + 1) * per
            };
            v.push(StageAssignment {
                stage_index: i,
                device_id: format!("stage-{i}"),
                layer_range: start..end,
                has_embedding: i == 0,
                has_lm_head: i + 1 == num_stages,
                estimated_memory_bytes: 0,
            });
        }
        v
    };

    let coordinator_addr = format!("127.0.0.1:{}", reserve_port());
    let stage_addrs: Vec<String> = (0..num_stages)
        .map(|_| format!("127.0.0.1:{}", reserve_port()))
        .collect();

    // Spawn stages in reverse order so each upstream stage knows the listen
    // address of its downstream neighbour when it issues the first connect.
    let mut stage_handles: Vec<RemoteStageServiceHandle> = Vec::with_capacity(num_stages);
    for i in (0..num_stages).rev() {
        let upstream = if i == 0 {
            None
        } else {
            Some(stage_addrs[i - 1].clone())
        };
        let downstream = if i + 1 == num_stages {
            None
        } else {
            Some(stage_addrs[i + 1].clone())
        };
        let handle = RemoteStageServiceHandle::spawn(RemoteStageServiceConfig {
            model_dir: model_dir.clone(),
            bind_address: stage_addrs[i].clone(),
            transport_backend: TransportBackend::Tcp,
            stage_assignment: assignments[i].clone(),
            num_stages: num_stages as u32,
            upstream_peer: upstream,
            downstream_peer: downstream,
        })
        .unwrap();
        stage_handles.push(handle);
    }
    // Re-order back to stage-order so later shutdown can be stage-ordered.
    stage_handles.reverse();

    let dense_provider = ModelProvider::new_with_server_config(
        model_dir.clone(),
        None,
        &ServerConfig::default(),
        Arc::new(BatchMetrics::new()),
        Arc::new(BatchObservability::new()),
    )
    .unwrap();
    let remote_provider = ModelProvider::new_with_server_config(
        model_dir.clone(),
        None,
        &ServerConfig {
            pipeline_parallel_runtime: Some(PipelineParallelRuntimeConfig::RemoteCoordinator(
                RemotePipelineRuntimeConfig {
                    stage_peers: stage_addrs.clone(),
                    transport_backend: TransportBackend::Tcp,
                    bind_address: coordinator_addr,
                    stage_timeout: Duration::from_secs(60),
                },
            )),
            ..ServerConfig::default()
        },
        Arc::new(BatchMetrics::new()),
        Arc::new(BatchObservability::new()),
    )
    .unwrap();

    wait_for_loaded(&dense_provider);
    wait_for_loaded(&remote_provider);

    let options = ServerGenerateOptions {
        max_tokens: 8,
        sampling: SamplingConfig::greedy(),
        stop_sequences: None,
        priority: RequestPriority::Normal,
        logprobs: Default::default(),
        reasoning_budget: Default::default(),
        thinking_enter_block_on_start: false,
        prompt_cache_ctx: None,
        structured: None,
    };

    let dense = dense_provider
        .generate(prompt.to_string(), options.clone())
        .unwrap();
    let remote = remote_provider
        .generate(prompt.to_string(), options)
        .unwrap();

    drop(remote_provider);
    drop(dense_provider);
    for handle in stage_handles {
        handle.shutdown().unwrap();
    }

    assert_eq!(
        dense.text, remote.text,
        "{}-stage remote coordinator diverged from dense baseline",
        num_stages
    );
    assert_eq!(
        dense.completion_tokens, remote.completion_tokens,
        "{}-stage remote coordinator token count diverged from dense baseline",
        num_stages
    );
}

/// 2-host logical smoke: two PP stages as two processes in one runner.
/// Driven by `scripts/ci/run-pp-two-host-logical.sh`.
///
/// The target model is resolved from `MLXCEL_CI_PP_MODEL` (the env var
/// the workflow sets after extracting the CI fixture). When the env is
/// unset — e.g. running locally — we fall back to `llama-3.2-1b-4bit`,
/// which matches the historical default and what most developer setups
/// keep under `models/`.
#[test]
#[ignore = "requires local model weights; CI drives with --ignored"]
fn pipeline_multi_stage_two_host_logical_smoke() {
    let model =
        std::env::var("MLXCEL_CI_PP_MODEL").unwrap_or_else(|_| "llama-3.2-1b-4bit".to_string());
    assert_multi_stage_coordinator_matches_dense_baseline(&model, "Hello", 2);
}

/// 3-host real-model parity: validates that the 3-stage remote pipeline
/// produces the same greedy output as the dense baseline. Driven by
/// `scripts/ci/run-pp-three-host.sh` from the self-hosted runner lab.
#[test]
#[ignore = "requires local model weights and is driven by the on-demand self-hosted 3-host CI job"]
fn pipeline_multi_stage_three_host_real_model_parity() {
    assert_multi_stage_coordinator_matches_dense_baseline("llama-3.2-1b-4bit", "Hello", 3);
}

/// Heterogeneous-memory partition regression.
///
/// Scenario from the acceptance list: stage 0 sits on a
/// memory-smaller machine, stage 1 sits on a memory-larger machine. The
/// auto-partitioner must assign fewer layers to stage 0 than to stage 1
/// and the resulting assignment must pass `validate_partition` +
/// `validate_memory_fit`.
#[test]
fn pipeline_heterogeneous_memory_partition_is_stable() {
    // Synthetic model: 32 transformer layers, ~250 MiB per layer, ~300 MiB
    // embedding, ~300 MiB lm_head. Numbers are illustrative; the property
    // under test is the relative assignment shape, not the absolute sizes.
    let profile =
        ModelProfile::uniform(32, 250 * 1024 * 1024, 300 * 1024 * 1024, 300 * 1024 * 1024);
    let devices = vec![
        DeviceSpec {
            device_id: "small-memory-stage-0".to_string(),
            available_memory_bytes: 16 * 1024 * 1024 * 1024, // 16 GiB
            compute_units: 10,
        },
        DeviceSpec {
            device_id: "large-memory-stage-1".to_string(),
            available_memory_bytes: 64 * 1024 * 1024 * 1024, // 64 GiB
            compute_units: 40,
        },
    ];

    let assignments = auto_partition(&profile, &devices).expect("auto-partition must succeed");
    assert_eq!(assignments.len(), 2, "exactly one stage per device");

    // Layer assignment validation.
    validate_partition(&assignments, profile.num_layers)
        .expect("partition must cover all layers without gaps or overlaps");
    validate_memory_fit(&assignments, &devices)
        .expect("auto-partition must fit each stage in its device's memory budget");

    // Relative shape: the 16 GiB stage must not be given more raw layers
    // than the 64 GiB stage. Embedding is still on stage 0, so we compare
    // the *layer* counts and the *memory* estimates separately.
    let stage0_layers = assignments[0].layer_range.end - assignments[0].layer_range.start;
    let stage1_layers = assignments[1].layer_range.end - assignments[1].layer_range.start;
    assert!(
        stage0_layers <= stage1_layers,
        "small-memory stage 0 was assigned more layers ({stage0_layers}) than stage 1 ({stage1_layers})",
    );
    assert!(assignments[0].has_embedding);
    assert!(!assignments[0].has_lm_head);
    assert!(!assignments[1].has_embedding);
    assert!(assignments[1].has_lm_head);
}

/// Heterogeneous-memory admission regression.
///
/// Exercises `PipelineCacheManager::request_admission` in a scenario where
/// stage 0 (the memory-constrained machine) cannot admit a long-prompt
/// sequence but stage 1 (the larger machine) can. The expected behaviour is
/// that stage 0 returns `Rejected(InsufficientMemory)` so the coordinator
/// can reject the sequence cluster-wide before stage 1 ever commits
/// resources. A regression here would let long prompts silently over-admit
/// on stage 0 and OOM mid-generation.
#[test]
fn pipeline_heterogeneous_memory_admission_rejects_oom() {
    let bytes_per_layer_per_token: u64 = 2 * 1024; // 2 KiB/layer/token (synthetic)

    // Stage 0: small budget, 8 layers. Stage 1: large budget, 24 layers.
    let mut stage0 = PipelineCacheManager::new(PipelineCacheConfig {
        stage_index: 0,
        num_stages: 2,
        layer_range: 0..8,
        max_sequences: 4,
        memory_budget_bytes: 1024 * 1024, // 1 MiB, tiny on purpose
        bytes_per_layer_per_token,
        pressure_threshold: 0.9,
    })
    .unwrap();
    let mut stage1 = PipelineCacheManager::new(PipelineCacheConfig {
        stage_index: 1,
        num_stages: 2,
        layer_range: 8..32,
        max_sequences: 4,
        memory_budget_bytes: 1024 * 1024 * 1024, // 1 GiB, generous
        bytes_per_layer_per_token,
        pressure_threshold: 0.9,
    })
    .unwrap();

    // A long prompt that exceeds stage 0's budget but fits on stage 1.
    let long_prompt_tokens = 4096;
    let request = CacheAdmissionRequest::new(42, long_prompt_tokens)
        .with_estimated_max_tokens(long_prompt_tokens + 64);

    let decision0 = stage0.request_admission(&request);
    match decision0 {
        AdmissionDecision::Rejected(RejectionReason::InsufficientMemory { .. }) => {}
        other => {
            panic!("stage 0 should reject the long prompt with InsufficientMemory, got {other:?}",)
        }
    }

    let decision1 = stage1.request_admission(&request);
    assert_eq!(
        decision1,
        AdmissionDecision::Admitted,
        "stage 1 with a larger budget must admit the same sequence",
    );
}
