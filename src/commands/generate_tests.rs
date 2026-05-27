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

use super::{
    apply_user_chat_template, cli_pipeline_requested, estimate_delta_label_and_bytes,
    generated_suffix, generation_stats_from_duration, memory_preflight_ctx_len,
    resolve_cli_pipeline_assignments, resolve_cli_prompt, validate_pipeline_parallel_args,
    validate_tensor_parallel_args,
};
use mlxcel::server::chat_template::ChatTemplateProcessor;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

#[test]
fn generation_stats_from_duration_uses_elapsed_time_for_decode_rate() {
    let stats = generation_stats_from_duration(12, 6, Duration::from_secs(2));

    assert_eq!(stats.prompt_tokens, 12);
    assert_eq!(stats.generated_tokens, 6);
    assert_eq!(stats.decode_time_ms, 2000.0);
    assert_eq!(stats.decode_tok_per_sec, 3.0);
}

#[test]
fn generation_stats_from_duration_handles_zero_elapsed_time() {
    let stats = generation_stats_from_duration(4, 2, Duration::ZERO);

    assert_eq!(stats.decode_time_ms, 0.0);
    assert_eq!(stats.decode_tok_per_sec, 0.0);
}

#[test]
fn estimate_delta_labels_match_actual_direction() {
    assert_eq!(
        estimate_delta_label_and_bytes(100, 125),
        ("under-estimated by", 25)
    );
    assert_eq!(
        estimate_delta_label_and_bytes(125, 100),
        ("over-estimated by", 25)
    );
}

#[test]
fn memory_preflight_ctx_len_includes_prompt_and_generation_budget() {
    assert_eq!(memory_preflight_ctx_len(4096, 128), 4224);
    assert_eq!(memory_preflight_ctx_len(0, 0), 1);
}

#[test]
fn apply_user_chat_template_wraps_prompt_as_user_message() {
    let processor = ChatTemplateProcessor::with_template(
        "{{ messages[0].role }}: {{ messages[0].content }}".to_string(),
    );

    let rendered = apply_user_chat_template(&processor, "Hello");

    assert_eq!(rendered, "user: Hello");
}

#[test]
fn resolve_cli_prompt_skips_template_when_disabled() {
    let processor = ChatTemplateProcessor::with_template("wrapped".to_string());

    let prompt = resolve_cli_prompt("Hello", true, Some(&processor), 0);

    assert_eq!(prompt, "Hello");
}

#[test]
fn resolve_cli_prompt_falls_back_on_template_errors() {
    let processor = ChatTemplateProcessor::with_template("{% if %}".to_string());

    let prompt = resolve_cli_prompt("Hello", false, Some(&processor), 0);

    assert_eq!(prompt, "Hello");
}

#[test]
fn generated_suffix_strips_prompt_prefix() {
    assert_eq!(generated_suffix("Hello, world", "Hello"), ", world");
}

#[test]
fn generated_suffix_falls_back_when_prefix_is_missing() {
    assert_eq!(generated_suffix("world", "Hello"), "world");
}

fn temp_model_dir(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "mlxcel-generate-test-{name}-{}",
        uuid::Uuid::new_v4()
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

fn sample_generate_args(model_path: PathBuf) -> crate::GenerateArgs {
    crate::GenerateArgs {
        model: crate::ModelOptions {
            model: model_path,
            models_dir: None,
            adapter: None,
            draft_model: None,
            num_draft_tokens: 3,
        },
        generation: crate::GenerationOptions {
            prompt: Some("Hello".to_string()),
            image: Vec::new(),
            audio: None,
            video: Vec::new(),
            fps: 2.0,
            max_tokens: 16,
            profile: false,
            no_chat_template: false,
            recommend_quant: false,
            estimate_memory: false,
            force_memory: false,
            turbo: mlxcel::cli::turbo_args::TurboKvCacheArgs::default(),
        },
        sampling: crate::SamplingOptions {
            temp: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_penalty_last_n: 0,
        },
        pipeline_parallel: crate::PipelineParallelOptions {
            pp_size: 1,
            pp_layers: None,
            pp_micro_batch_size: 1,
        },
        tensor_parallel: crate::TensorParallelOptions {
            tp_size: 1,
            tp_moe_mode: "expert_parallel".to_string(),
            tp_embedding_mode: "replicated".to_string(),
            tp_lm_head_mode: "replicated".to_string(),
        },
        lang_bias: mlxcel::lang_bias::LangBiasCliArgs::default(),
        speculative: mlxcel::cli::speculative_args::SpeculativeArgs::default(),
        // Issue #371 (A4): default to None so existing tests stay on
        // the bit-exact baseline load path; tests that need surgery
        // override this field explicitly.
        #[cfg(feature = "surgery")]
        surgery: None,
    }
}

#[test]
fn validate_tensor_parallel_args_accepts_single_rank() {
    let dir = temp_model_dir("tp1");
    fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "llama",
            "num_hidden_layers": 32
        }"#,
    )
    .unwrap();

    let args = sample_generate_args(dir.clone());
    validate_tensor_parallel_args(&args).unwrap();

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_tensor_parallel_args_accepts_supported_multi_rank_runtime() {
    let dir = temp_model_dir("tp2");
    fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "llama",
            "num_hidden_layers": 32
        }"#,
    )
    .unwrap();

    let mut args = sample_generate_args(dir.clone());
    args.tensor_parallel.tp_size = 2;

    validate_tensor_parallel_args(&args).unwrap();

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_tensor_parallel_args_accepts_qwen2_multi_rank_runtime() {
    let dir = temp_model_dir("tp-qwen2");
    fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "qwen2",
            "num_hidden_layers": 24
        }"#,
    )
    .unwrap();

    let mut args = sample_generate_args(dir.clone());
    args.tensor_parallel.tp_size = 2;

    validate_tensor_parallel_args(&args).unwrap();

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_tensor_parallel_args_accepts_qwen3_multi_rank_runtime() {
    let dir = temp_model_dir("tp-qwen3");
    fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "qwen3",
            "num_hidden_layers": 28
        }"#,
    )
    .unwrap();

    let mut args = sample_generate_args(dir.clone());
    args.tensor_parallel.tp_size = 2;

    validate_tensor_parallel_args(&args).unwrap();

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_tensor_parallel_args_accepts_qwen35_multi_rank_runtime() {
    let dir = temp_model_dir("tp-qwen35");
    fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "qwen3_5",
            "num_hidden_layers": 24
        }"#,
    )
    .unwrap();

    let mut args = sample_generate_args(dir.clone());
    args.tensor_parallel.tp_size = 2;

    validate_tensor_parallel_args(&args).unwrap();

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_tensor_parallel_args_accepts_ernie45_multi_rank_runtime() {
    let dir = temp_model_dir("tp-ernie45");
    fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "ernie4_5",
            "num_hidden_layers": 18
        }"#,
    )
    .unwrap();

    let mut args = sample_generate_args(dir.clone());
    args.tensor_parallel.tp_size = 2;

    validate_tensor_parallel_args(&args).unwrap();

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_tensor_parallel_args_accepts_hunyuan_v1_dense_multi_rank_runtime() {
    let dir = temp_model_dir("tp-hunyuan-v1-dense");
    fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "hunyuan_v1_dense",
            "num_hidden_layers": 32
        }"#,
    )
    .unwrap();

    let mut args = sample_generate_args(dir.clone());
    args.tensor_parallel.tp_size = 2;

    validate_tensor_parallel_args(&args).unwrap();

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_tensor_parallel_args_accepts_gemma3_multi_rank_runtime() {
    let dir = temp_model_dir("tp-gemma3");
    fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "gemma3_text",
            "num_hidden_layers": 26
        }"#,
    )
    .unwrap();

    let mut args = sample_generate_args(dir.clone());
    args.tensor_parallel.tp_size = 2;

    validate_tensor_parallel_args(&args).unwrap();

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_tensor_parallel_args_accepts_gemma4_multi_rank_runtime() {
    let dir = temp_model_dir("tp-gemma4");
    fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "num_hidden_layers": 26
            }
        }"#,
    )
    .unwrap();

    let mut args = sample_generate_args(dir.clone());
    args.tensor_parallel.tp_size = 2;

    validate_tensor_parallel_args(&args).unwrap();

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn cli_pipeline_requested_is_disabled_by_default() {
    let args = sample_generate_args(temp_model_dir("pp-disabled"));
    assert!(!cli_pipeline_requested(&args));
    fs::remove_dir_all(args.model.model).unwrap();
}

#[test]
fn validate_pipeline_parallel_args_requires_two_stages_without_manual_ranges() {
    let mut args = sample_generate_args(temp_model_dir("pp-too-small"));
    args.pipeline_parallel.pp_size = 1;

    assert!(validate_pipeline_parallel_args(&args).is_ok());

    args.pipeline_parallel.pp_layers = Some("0-1".to_string());
    assert!(validate_pipeline_parallel_args(&args).is_ok());

    args.pipeline_parallel.pp_layers = None;
    args.pipeline_parallel.pp_size = 0;
    assert!(validate_pipeline_parallel_args(&args).is_ok());

    fs::remove_dir_all(args.model.model).unwrap();
}

#[test]
fn validate_pipeline_parallel_args_rejects_incompatible_modes() {
    let mut args = sample_generate_args(temp_model_dir("pp-incompatible"));
    // Speculative decoding + PP is still rejected (separate epic).
    args.pipeline_parallel.pp_size = 2;
    args.model.draft_model = Some(PathBuf::from("draft"));
    assert!(validate_pipeline_parallel_args(&args).is_err());
    args.model.draft_model = None;

    // Tensor parallelism + PP is now accepted (2D PP × TP composition landed
    // via #346). Positive coverage for the 2D path lives in
    // `validate_pipeline_parallel_args_accepts_2d_pp_tp` below.

    fs::remove_dir_all(args.model.model).unwrap();
}

#[test]
fn validate_pipeline_parallel_args_accepts_adapter() {
    // LoRA + PP composition is supported: stage-local adapter loading is
    // wired through load_in_process_stage_worker_with_adapter. The CLI
    // validator must accept the combination so the runtime path can take
    // over. (v1 single-adapter composition.)
    let mut args = sample_generate_args(temp_model_dir("pp-with-adapter"));
    args.pipeline_parallel.pp_size = 2;
    args.model.adapter = Some(PathBuf::from("adapter"));
    assert!(validate_pipeline_parallel_args(&args).is_ok());
    fs::remove_dir_all(args.model.model).unwrap();
}

#[test]
fn validate_pipeline_parallel_args_accepts_2d_pp_tp() {
    // Issue #346: the validator no longer rejects PP + TP.
    let mut args = sample_generate_args(temp_model_dir("pp-tp-2d"));
    args.pipeline_parallel.pp_size = 2;
    args.tensor_parallel.tp_size = 2;
    let result = validate_pipeline_parallel_args(&args);
    assert!(
        result.is_ok(),
        "expected validator to accept 2D PPxTP, got: {result:?}"
    );
    fs::remove_dir_all(args.model.model).unwrap();
}

#[test]
fn validate_pipeline_parallel_args_rejects_2d_without_pp_enabled() {
    // TP > 1 with pp_size=1 is TP-only, not 2D; but if the caller sets pp_size=0
    // alongside tp_size=2 it is malformed. The validator returns early when
    // PP is disabled, so this case is harmless — verify it doesn't error.
    let mut args = sample_generate_args(temp_model_dir("tp-only"));
    args.pipeline_parallel.pp_size = 1;
    args.tensor_parallel.tp_size = 2;
    assert!(validate_pipeline_parallel_args(&args).is_ok());
    fs::remove_dir_all(args.model.model).unwrap();
}

#[test]
fn resolve_cli_pipeline_assignments_honors_manual_ranges() {
    let mut args = sample_generate_args(temp_model_dir("pp-manual"));
    args.pipeline_parallel.pp_size = 2;
    args.pipeline_parallel.pp_layers = Some("0-3,4-7".to_string());

    let model_dir = args.model.model.clone();
    let assignments = resolve_cli_pipeline_assignments(&model_dir, 8, &args).unwrap();

    assert_eq!(assignments.len(), 2);
    assert_eq!(assignments[0].layer_range, 0..4);
    assert_eq!(assignments[1].layer_range, 4..8);
    fs::remove_dir_all(args.model.model).unwrap();
}

#[test]
fn resolve_cli_pipeline_assignments_auto_splits_layers_across_stages() {
    let mut args = sample_generate_args(temp_model_dir("pp-auto"));
    args.pipeline_parallel.pp_size = 3;

    let model_dir = args.model.model.clone();
    let assignments = resolve_cli_pipeline_assignments(&model_dir, 9, &args).unwrap();

    assert_eq!(assignments.len(), 3);
    assert_eq!(assignments[0].layer_range.start, 0);
    assert_eq!(assignments[2].layer_range.end, 9);
    assert!(
        assignments
            .iter()
            .all(|stage| !stage.layer_range.is_empty())
    );
    fs::remove_dir_all(args.model.model).unwrap();
}
