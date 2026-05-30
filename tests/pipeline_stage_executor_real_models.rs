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

mod common;

use std::path::Path;

use common::repo_model_dir;
use mlxcel::distributed::RequestId;
use mlxcel::distributed::pipeline::{
    ChannelConfig, InProcessStageWorkerLoop, LoadedStageExecutor, PipelineConfig,
    PipelineWorkerInput, StageAssignment, StageExecutionInput,
};
use mlxcel::{LanguageModel, distributed::pipeline::StageExecutor};

fn two_stage_assignments(total_layers: usize, split: usize) -> [StageAssignment; 2] {
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
            layer_range: split..total_layers,
            has_embedding: false,
            has_lm_head: true,
            estimated_memory_bytes: 0,
        },
    ]
}

fn assert_two_stage_model_matches_full_model(
    model_dir: &Path,
    prompt: &[i32],
    decode_token: i32,
    split_override: Option<usize>,
) {
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return;
    }

    let (model, _) = mlxcel::load_model(model_dir).unwrap();
    let total_layers = model.num_layers();
    let assignments =
        two_stage_assignments(total_layers, split_override.unwrap_or(total_layers / 2));
    let stage0 = LoadedStageExecutor::load(model_dir, &assignments[0]).unwrap();
    let stage1 = LoadedStageExecutor::load(model_dir, &assignments[1]).unwrap();

    let prompt_ids = mlxcel_core::from_slice_i32(prompt, &[1, prompt.len() as i32]);
    let decode_ids = mlxcel_core::from_slice_i32(&[decode_token], &[1, 1]);

    let mut full_caches = model.make_caches();
    let full_prefill = model.forward(&prompt_ids, &mut full_caches, None);
    let full_decode = model.forward(&decode_ids, &mut full_caches, None);

    let mut stage0_caches = stage0.make_caches();
    let mut stage1_caches = stage1.make_caches();
    let stage0_prefill = stage0
        .execute(
            StageExecutionInput::TokenIds(&prompt_ids),
            &mut stage0_caches,
            None,
        )
        .unwrap()
        .into_hidden_states()
        .unwrap();
    let stage_prefill = stage1
        .execute(
            StageExecutionInput::HiddenStates(stage0_prefill.as_ref().unwrap()),
            &mut stage1_caches,
            None,
        )
        .unwrap()
        .into_logits()
        .unwrap();
    let stage0_decode = stage0
        .execute(
            StageExecutionInput::TokenIds(&decode_ids),
            &mut stage0_caches,
            None,
        )
        .unwrap()
        .into_hidden_states()
        .unwrap();
    let stage_decode = stage1
        .execute(
            StageExecutionInput::HiddenStates(stage0_decode.as_ref().unwrap()),
            &mut stage1_caches,
            None,
        )
        .unwrap()
        .into_logits()
        .unwrap();

    let atol = 1e-4f64;
    let prefill_close = mlxcel_core::allclose(&full_prefill, &stage_prefill, atol, atol);
    let decode_close = mlxcel_core::allclose(&full_decode, &stage_decode, atol, atol);
    assert!(
        mlxcel_core::item_bool(&prefill_close),
        "prefill logits mismatch for {}",
        model_dir.display()
    );
    assert!(
        mlxcel_core::item_bool(&decode_close),
        "decode logits mismatch for {}",
        model_dir.display()
    );
}

fn assert_two_stage_model_worker_loop_matches_full_model(
    model_dir: &Path,
    prompt: &[i32],
    decode_token: i32,
    split_override: Option<usize>,
) {
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return;
    }

    let (model, _) = mlxcel::load_model(model_dir).unwrap();
    let total_layers = model.num_layers();
    let assignments =
        two_stage_assignments(total_layers, split_override.unwrap_or(total_layers / 2));
    let executors: Vec<Box<dyn StageExecutor>> = vec![
        Box::new(LoadedStageExecutor::load(model_dir, &assignments[0]).unwrap()),
        Box::new(LoadedStageExecutor::load(model_dir, &assignments[1]).unwrap()),
    ];
    let mut worker_loop = InProcessStageWorkerLoop::new(
        PipelineConfig::new(2, 1).unwrap(),
        executors,
        ChannelConfig::default(),
    )
    .unwrap();

    let prompt_ids = mlxcel_core::from_slice_i32(prompt, &[1, prompt.len() as i32]);
    let decode_ids = mlxcel_core::from_slice_i32(&[decode_token], &[1, 1]);
    let request_id = RequestId::from_string("llama-worker-loop".to_string()).unwrap();

    let mut full_caches = model.make_caches();
    let full_prefill = model.forward(&prompt_ids, &mut full_caches, None);
    let full_decode = model.forward(&decode_ids, &mut full_caches, None);

    let prefill = worker_loop
        .run_to_completion(vec![PipelineWorkerInput::new(
            request_id.clone(),
            prompt_ids,
        )])
        .unwrap();
    let decode = worker_loop
        .run_to_completion(vec![PipelineWorkerInput::new(request_id, decode_ids)])
        .unwrap();

    let atol = 1e-4f64;
    let prefill_close = mlxcel_core::allclose(
        &full_prefill,
        prefill[0].logits.as_ref().unwrap(),
        atol,
        atol,
    );
    let decode_close =
        mlxcel_core::allclose(&full_decode, decode[0].logits.as_ref().unwrap(), atol, atol);
    assert!(
        mlxcel_core::item_bool(&prefill_close),
        "prefill worker loop logits mismatch for {}",
        model_dir.display()
    );
    assert!(
        mlxcel_core::item_bool(&decode_close),
        "decode worker loop logits mismatch for {}",
        model_dir.display()
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn pipeline_stage_executor_llama_real_model_parity() {
    assert_two_stage_model_matches_full_model(
        &repo_model_dir("llama-3.2-1b-4bit"),
        &[128000, 9906],
        13,
        None,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn pipeline_stage_worker_loop_llama_real_model_parity() {
    assert_two_stage_model_worker_loop_matches_full_model(
        &repo_model_dir("llama-3.2-1b-4bit"),
        &[128000, 9906],
        13,
        None,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn pipeline_stage_executor_gpt_oss_real_model_parity() {
    assert_two_stage_model_matches_full_model(
        &repo_model_dir("gpt-oss-20b-mxfp4"),
        &[42, 43],
        44,
        None,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn pipeline_stage_worker_loop_gpt_oss_real_model_parity() {
    assert_two_stage_model_worker_loop_matches_full_model(
        &repo_model_dir("gpt-oss-20b-mxfp4"),
        &[42, 43],
        44,
        None,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn pipeline_stage_executor_gemma3_real_model_parity() {
    assert_two_stage_model_matches_full_model(&repo_model_dir("gemma3-1b-4bit"), &[2, 3], 4, None);
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn pipeline_stage_worker_loop_gemma3_real_model_parity() {
    assert_two_stage_model_worker_loop_matches_full_model(
        &repo_model_dir("gemma3-1b-4bit"),
        &[2, 3],
        4,
        None,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn pipeline_stage_executor_gemma4_real_model_parity() {
    assert_two_stage_model_matches_full_model(
        &repo_model_dir("gemma-4-e2b-it-4bit"),
        &[2, 3],
        4,
        Some(13),
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn pipeline_stage_worker_loop_gemma4_real_model_parity() {
    assert_two_stage_model_worker_loop_matches_full_model(
        &repo_model_dir("gemma-4-e2b-it-4bit"),
        &[2, 3],
        4,
        Some(13),
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn pipeline_stage_executor_qwen3_real_model_parity() {
    assert_two_stage_model_matches_full_model(&repo_model_dir("qwen3-0.6b-4bit"), &[2, 3], 4, None);
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn pipeline_stage_worker_loop_qwen3_real_model_parity() {
    assert_two_stage_model_worker_loop_matches_full_model(
        &repo_model_dir("qwen3-0.6b-4bit"),
        &[2, 3],
        4,
        None,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn pipeline_stage_executor_qwen35_real_model_parity() {
    assert_two_stage_model_matches_full_model(
        &repo_model_dir("qwen3.5-0.8b-4bit"),
        &[2, 3],
        4,
        None,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn pipeline_stage_worker_loop_qwen35_real_model_parity() {
    assert_two_stage_model_worker_loop_matches_full_model(
        &repo_model_dir("qwen3.5-0.8b-4bit"),
        &[2, 3],
        4,
        None,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn pipeline_stage_executor_glm4_real_model_parity() {
    assert_two_stage_model_matches_full_model(&repo_model_dir("glm4-flash-4bit"), &[2, 3], 4, None);
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn pipeline_stage_worker_loop_glm4_real_model_parity() {
    assert_two_stage_model_worker_loop_matches_full_model(
        &repo_model_dir("glm4-flash-4bit"),
        &[2, 3],
        4,
        None,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn pipeline_stage_executor_glm_moe_dsa_real_model_parity() {
    assert_two_stage_model_matches_full_model(&repo_model_dir("glm5-4bit"), &[2, 3], 4, None);
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn pipeline_stage_worker_loop_glm_moe_dsa_real_model_parity() {
    assert_two_stage_model_worker_loop_matches_full_model(
        &repo_model_dir("glm5-4bit"),
        &[2, 3],
        4,
        None,
    );
}

// families — stage executor parity against the non-PP reference
// path on a short prompt. Each test is gated behind `#[ignore]` because it
// requires real model weights to be present under `models/`; operators can
// opt in by downloading the listed HuggingFace MLX Community checkpoint.

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn pipeline_stage_executor_mistral_real_model_parity() {
    // mlx-community/Mistral-7B-Instruct-v0.3-4bit (or any base mistral model)
    assert_two_stage_model_matches_full_model(
        &repo_model_dir("mistral-7b-instruct-v0.3-4bit"),
        &[1, 2],
        3,
        None,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn pipeline_stage_worker_loop_mistral_real_model_parity() {
    assert_two_stage_model_worker_loop_matches_full_model(
        &repo_model_dir("mistral-7b-instruct-v0.3-4bit"),
        &[1, 2],
        3,
        None,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn pipeline_stage_executor_mixtral_real_model_parity() {
    // mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit
    assert_two_stage_model_matches_full_model(
        &repo_model_dir("mixtral-8x7b-instruct-v0.1-4bit"),
        &[1, 2],
        3,
        None,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn pipeline_stage_worker_loop_mixtral_real_model_parity() {
    assert_two_stage_model_worker_loop_matches_full_model(
        &repo_model_dir("mixtral-8x7b-instruct-v0.1-4bit"),
        &[1, 2],
        3,
        None,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn pipeline_stage_executor_deepseek_v3_real_model_parity() {
    // mlx-community/DeepSeek-V3-0324-4bit — note the 61-layer MTP trailer:
    // the pipeline partitioner treats num_hidden_layers - 1 as the real
    // transformer depth.
    assert_two_stage_model_matches_full_model(
        &repo_model_dir("deepseek-v3-0324-4bit"),
        &[1, 2],
        3,
        None,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn pipeline_stage_worker_loop_deepseek_v3_real_model_parity() {
    assert_two_stage_model_worker_loop_matches_full_model(
        &repo_model_dir("deepseek-v3-0324-4bit"),
        &[1, 2],
        3,
        None,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn pipeline_stage_executor_llama4_real_model_parity() {
    // mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit — TEXT-only tower.
    assert_two_stage_model_matches_full_model(
        &repo_model_dir("llama-4-scout-17b-16e-instruct-4bit"),
        &[128000, 1],
        2,
        None,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn pipeline_stage_worker_loop_llama4_real_model_parity() {
    assert_two_stage_model_worker_loop_matches_full_model(
        &repo_model_dir("llama-4-scout-17b-16e-instruct-4bit"),
        &[128000, 1],
        2,
        None,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn pipeline_stage_executor_jamba_real_model_parity() {
    // mlx-community/Jamba-v0.1-4bit — hybrid Mamba + Transformer. Every
    // stage internally keeps the full model loaded (see the design note at
    // the top of src/distributed/pipeline/stage_executor/jamba.rs); the
    // parity check verifies that executing the stage-assigned layer range
    // on each stage matches the non-PP model's logits end-to-end.
    assert_two_stage_model_matches_full_model(&repo_model_dir("jamba-v0.1-4bit"), &[1, 2], 3, None);
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn pipeline_stage_worker_loop_jamba_real_model_parity() {
    assert_two_stage_model_worker_loop_matches_full_model(
        &repo_model_dir("jamba-v0.1-4bit"),
        &[1, 2],
        3,
        None,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn pipeline_stage_executor_nemotron_h_real_model_parity() {
    // mlx-community/nvidia-Nemotron-H-8B-Base-8bit — hybrid Mamba2 +
    // Transformer + MoE.
    assert_two_stage_model_matches_full_model(
        &repo_model_dir("nvidia-nemotron-h-8b-base-8bit"),
        &[1, 2],
        3,
        None,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn pipeline_stage_worker_loop_nemotron_h_real_model_parity() {
    assert_two_stage_model_worker_loop_matches_full_model(
        &repo_model_dir("nvidia-nemotron-h-8b-base-8bit"),
        &[1, 2],
        3,
        None,
    );
}
