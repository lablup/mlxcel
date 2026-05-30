mod common;

use std::net::{IpAddr, TcpListener};
use std::time::Duration;

use common::repo_model_dir;
use mlxcel::distributed::pipeline::{
    RemotePipelineRuntime, RemotePipelineRuntimeConfig, RemoteStageResponse,
    RemoteStageServiceConfig, RemoteStageServiceHandle, StageAssignment,
    resolve_in_process_pipeline_num_layers,
};
use mlxcel::distributed::{ThunderboltTransport, TransportBackend};
use mlxcel::{LanguageModel, distributed::pipeline::PipelineModelRuntime};
use mlxcel_core::cache::SequenceId;

fn reserve_port(bind_ip: IpAddr) -> u16 {
    let listener = TcpListener::bind((bind_ip, 0)).expect("bind ephemeral port");
    let port = listener.local_addr().expect("local addr").port();
    drop(listener);
    port
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

fn spawn_remote_runtime(
    model_dir: &std::path::Path,
    bind_ip: IpAddr,
    transport_backend: TransportBackend,
    split_override: Option<usize>,
) -> (
    RemotePipelineRuntime,
    RemoteStageServiceHandle,
    RemoteStageServiceHandle,
) {
    let num_layers = resolve_in_process_pipeline_num_layers(model_dir).unwrap();
    let assignments = two_stage_assignments(num_layers, split_override.unwrap_or(num_layers / 2));

    let stage0_addr = format!("{}:{}", bind_ip, reserve_port(bind_ip));
    let stage1_addr = format!("{}:{}", bind_ip, reserve_port(bind_ip));

    let stage1 = RemoteStageServiceHandle::spawn(RemoteStageServiceConfig {
        model_dir: model_dir.to_path_buf(),
        bind_address: stage1_addr.clone(),
        transport_backend,
        stage_assignment: assignments[1].clone(),
        num_stages: 2,
        upstream_peer: Some(stage0_addr.clone()),
        downstream_peer: None,
    })
    .unwrap();
    let stage0 = RemoteStageServiceHandle::spawn(RemoteStageServiceConfig {
        model_dir: model_dir.to_path_buf(),
        bind_address: stage0_addr.clone(),
        transport_backend,
        stage_assignment: assignments[0].clone(),
        num_stages: 2,
        upstream_peer: None,
        downstream_peer: Some(stage1_addr.clone()),
    })
    .unwrap();

    assert_eq!(stage0.local_addr(), stage0_addr);
    assert_eq!(stage1.local_addr(), stage1_addr);

    let runtime = RemotePipelineRuntime::new(RemotePipelineRuntimeConfig {
        stage_peers: vec![stage0_addr, stage1_addr],
        transport_backend,
        bind_address: format!("{}:{}", bind_ip, reserve_port(bind_ip)),
        stage_timeout: Duration::from_secs(30),
    })
    .unwrap();

    (runtime, stage0, stage1)
}

fn assert_remote_runtime_matches_full_model(
    model_name: &str,
    prompt: &[i32],
    decode_token: i32,
    seq_raw: u64,
    split_override: Option<usize>,
) {
    let model_dir = repo_model_dir(model_name);
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return;
    }

    let (model, _) = mlxcel::load_model(&model_dir).unwrap();
    let (runtime, stage0, stage1) = spawn_remote_runtime(
        &model_dir,
        "127.0.0.1".parse().unwrap(),
        TransportBackend::Tcp,
        split_override,
    );

    let prompt_ids = mlxcel_core::from_slice_i32(prompt, &[1, prompt.len() as i32]);
    let decode_ids = mlxcel_core::from_slice_i32(&[decode_token], &[1, 1]);

    let mut full_caches = model.make_caches();
    let full_prefill = model.forward(&prompt_ids, &mut full_caches, None);
    let full_decode = model.forward(&decode_ids, &mut full_caches, None);

    let seq_id = SequenceId::from_raw(seq_raw);
    runtime.prepare_sequence_state(seq_id);
    let remote_prefill = runtime.forward_sequence(seq_id, &prompt_ids, None).unwrap();
    let remote_decode = runtime.forward_sequence(seq_id, &decode_ids, None).unwrap();

    let atol = 1e-4f64;
    assert!(mlxcel_core::item_bool(&mlxcel_core::allclose(
        &full_prefill,
        &remote_prefill,
        atol,
        atol
    )));
    assert!(mlxcel_core::item_bool(&mlxcel_core::allclose(
        &full_decode,
        &remote_decode,
        atol,
        atol
    )));

    runtime.release_sequence_state_by_id(seq_id);
    runtime.shutdown().unwrap();
    stage0.shutdown().unwrap();
    stage1.shutdown().unwrap();
}

#[test]
#[ignore = "requires local model weights and TCP-bound remote pipeline stages"]
fn pipeline_remote_runtime_llama_real_model_parity_and_cleanup() {
    let model_dir = repo_model_dir("llama-3.2-1b-4bit");
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return;
    }

    let (model, _) = mlxcel::load_model(&model_dir).unwrap();
    let (runtime, stage0, stage1) = spawn_remote_runtime(
        &model_dir,
        "127.0.0.1".parse().unwrap(),
        TransportBackend::Tcp,
        None,
    );

    let prompt_ids = mlxcel_core::from_slice_i32(&[128000, 9906], &[1, 2]);
    let decode_ids = mlxcel_core::from_slice_i32(&[13], &[1, 1]);

    let mut full_caches = model.make_caches();
    let full_prefill = model.forward(&prompt_ids, &mut full_caches, None);
    let full_decode = model.forward(&decode_ids, &mut full_caches, None);

    let seq_id = SequenceId::from_raw(42);
    runtime.prepare_sequence_state(seq_id);
    let remote_prefill = runtime.forward_sequence(seq_id, &prompt_ids, None).unwrap();
    let remote_decode = runtime.forward_sequence(seq_id, &decode_ids, None).unwrap();

    let active_states = runtime.probe_stages().unwrap();
    assert!(active_states.iter().all(|state| matches!(
        state,
        RemoteStageResponse::State {
            state,
            pending_entry_replies: _,
            ..
        } if state.in_flight_requests == 1
    )));

    let atol = 1e-4f64;
    assert!(mlxcel_core::item_bool(&mlxcel_core::allclose(
        &full_prefill,
        &remote_prefill,
        atol,
        atol
    )));
    assert!(mlxcel_core::item_bool(&mlxcel_core::allclose(
        &full_decode,
        &remote_decode,
        atol,
        atol
    )));

    runtime.release_sequence_state_by_id(seq_id);
    let released_states = runtime.probe_stages().unwrap();
    assert!(released_states.iter().all(|state| matches!(
        state,
        RemoteStageResponse::State {
            state,
            pending_entry_replies: 0,
            ..
        } if state.in_flight_requests == 0
    )));

    stage0.shutdown().unwrap();
    stage1.shutdown().unwrap();
}

#[test]
#[ignore = "requires local model weights and TCP-bound remote pipeline stages"]
fn pipeline_remote_runtime_llama_drain_and_shutdown_transition() {
    let model_dir = repo_model_dir("llama-3.2-1b-4bit");
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return;
    }

    let (runtime, stage0, stage1) = spawn_remote_runtime(
        &model_dir,
        "127.0.0.1".parse().unwrap(),
        TransportBackend::Tcp,
        None,
    );

    let seq_id = SequenceId::from_raw(77);
    runtime.prepare_sequence_state(seq_id);
    runtime.begin_drain().unwrap();

    let draining_states = runtime.probe_stages().unwrap();
    assert!(draining_states.iter().all(|state| matches!(
        state,
        RemoteStageResponse::State { state, .. } if state.draining && !state.shutdown
    )));

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        runtime.prepare_sequence_state(SequenceId::from_raw(78));
    }));
    assert!(panic.is_err(), "draining runtime must reject new sequences");

    runtime.release_sequence_state_by_id(seq_id);
    runtime.shutdown().unwrap();

    stage0.shutdown().unwrap();
    stage1.shutdown().unwrap();
}

#[test]
#[ignore = "requires local model weights and an active Thunderbolt Bridge interface"]
fn pipeline_remote_runtime_llama_thunderbolt_bridge_parity() {
    let model_dir = repo_model_dir("llama-3.2-1b-4bit");
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return;
    }

    let interfaces = ThunderboltTransport::available_interfaces().unwrap_or_default();
    let Some(bind_ip) = interfaces
        .iter()
        .flat_map(|interface| interface.addresses.iter().copied())
        .find(|addr| addr.is_ipv4())
    else {
        eprintln!("Skipping test: no Thunderbolt Bridge interface with an IPv4 address");
        return;
    };

    let (model, _) = mlxcel::load_model(&model_dir).unwrap();
    let (runtime, stage0, stage1) =
        spawn_remote_runtime(&model_dir, bind_ip, TransportBackend::Thunderbolt, None);

    let prompt_ids = mlxcel_core::from_slice_i32(&[128000, 9906], &[1, 2]);
    let decode_ids = mlxcel_core::from_slice_i32(&[13], &[1, 1]);

    let mut full_caches = model.make_caches();
    let full_prefill = model.forward(&prompt_ids, &mut full_caches, None);
    let full_decode = model.forward(&decode_ids, &mut full_caches, None);

    let seq_id = SequenceId::from_raw(99);
    runtime.prepare_sequence_state(seq_id);
    let remote_prefill = runtime.forward_sequence(seq_id, &prompt_ids, None).unwrap();
    let remote_decode = runtime.forward_sequence(seq_id, &decode_ids, None).unwrap();

    let atol = 1e-4f64;
    assert!(mlxcel_core::item_bool(&mlxcel_core::allclose(
        &full_prefill,
        &remote_prefill,
        atol,
        atol
    )));
    assert!(mlxcel_core::item_bool(&mlxcel_core::allclose(
        &full_decode,
        &remote_decode,
        atol,
        atol
    )));

    runtime.release_sequence_state_by_id(seq_id);
    runtime.shutdown().unwrap();
    stage0.shutdown().unwrap();
    stage1.shutdown().unwrap();
}

#[test]
#[ignore = "requires local model weights and TCP-bound remote pipeline stages"]
fn pipeline_remote_runtime_gpt_oss_real_model_parity_and_cleanup() {
    assert_remote_runtime_matches_full_model("gpt-oss-20b-mxfp4", &[42, 43], 44, 420, None);
}

#[test]
#[ignore = "requires local model weights and TCP-bound remote pipeline stages"]
fn pipeline_remote_runtime_gemma3_real_model_parity_and_cleanup() {
    assert_remote_runtime_matches_full_model("gemma3-1b-4bit", &[2, 3], 4, 430, None);
}

#[test]
#[ignore = "requires local model weights and TCP-bound remote pipeline stages"]
fn pipeline_remote_runtime_gemma4_real_model_parity_and_cleanup() {
    assert_remote_runtime_matches_full_model("gemma-4-e2b-it-4bit", &[2, 3], 4, 440, Some(13));
}

#[test]
#[ignore = "requires local model weights and TCP-bound remote pipeline stages"]
fn pipeline_remote_runtime_qwen3_real_model_parity_and_cleanup() {
    assert_remote_runtime_matches_full_model("qwen3-0.6b-4bit", &[2, 3], 4, 450, None);
}

#[test]
#[ignore = "requires local model weights and TCP-bound remote pipeline stages"]
fn pipeline_remote_runtime_qwen35_real_model_parity_and_cleanup() {
    assert_remote_runtime_matches_full_model("qwen3.5-0.8b-4bit", &[2, 3], 4, 460, None);
}

#[test]
#[ignore = "requires local model weights and TCP-bound remote pipeline stages"]
fn pipeline_remote_runtime_glm4_real_model_parity_and_cleanup() {
    assert_remote_runtime_matches_full_model("glm4-flash-4bit", &[2, 3], 4, 470, None);
}

#[test]
#[ignore = "requires local model weights and TCP-bound remote pipeline stages"]
fn pipeline_remote_runtime_glm_moe_dsa_real_model_parity_and_cleanup() {
    assert_remote_runtime_matches_full_model("glm5-4bit", &[2, 3], 4, 480, None);
}

// ---------------------------------------------------------------------------
// RDMA backend real-model coverage
// ---------------------------------------------------------------------------
//
// This test runs the full 2-stage remote pipeline over the RDMA-aware
// transport. The RDMA backend negotiates an OS-level zero-copy primitive at
// bind time and falls back transparently to TCP when the primitive is not
// available, so the test is valid on every host: on a machine where
// io_uring / kqueue-registered-buffers are reachable the accelerated path is
// exercised, while on a locked-down box it validates the silent-but-logged
// TCP fallback.
//
// The assertion contract is identical to the TCP parity test — the
// abstraction hides the backend from the activation-transfer consumer.

#[test]
#[ignore = "requires local model weights and RDMA-capable or fallback-enabled loopback"]
fn pipeline_remote_runtime_llama_rdma_real_model_parity_and_cleanup() {
    let model_dir = repo_model_dir("llama-3.2-1b-4bit");
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return;
    }

    let (model, _) = mlxcel::load_model(&model_dir).unwrap();
    let (runtime, stage0, stage1) = spawn_remote_runtime(
        &model_dir,
        "127.0.0.1".parse().unwrap(),
        TransportBackend::Rdma,
        None,
    );

    let prompt_ids = mlxcel_core::from_slice_i32(&[128000, 9906], &[1, 2]);
    let decode_ids = mlxcel_core::from_slice_i32(&[13], &[1, 1]);

    let mut full_caches = model.make_caches();
    let full_prefill = model.forward(&prompt_ids, &mut full_caches, None);
    let full_decode = model.forward(&decode_ids, &mut full_caches, None);

    let seq_id = SequenceId::from_raw(343);
    runtime.prepare_sequence_state(seq_id);
    let remote_prefill = runtime.forward_sequence(seq_id, &prompt_ids, None).unwrap();
    let remote_decode = runtime.forward_sequence(seq_id, &decode_ids, None).unwrap();

    let atol = 1e-4f64;
    assert!(mlxcel_core::item_bool(&mlxcel_core::allclose(
        &full_prefill,
        &remote_prefill,
        atol,
        atol
    )));
    assert!(mlxcel_core::item_bool(&mlxcel_core::allclose(
        &full_decode,
        &remote_decode,
        atol,
        atol
    )));

    runtime.release_sequence_state_by_id(seq_id);
    runtime.shutdown().unwrap();
    stage0.shutdown().unwrap();
    stage1.shutdown().unwrap();
}
