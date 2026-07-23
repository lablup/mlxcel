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
    apply_user_chat_template, apply_vlm_chat_template, cli_pipeline_requested,
    estimate_delta_label_and_bytes, generated_suffix, generation_stats_from_duration,
    memory_preflight_ctx_len, resolve_cli_pipeline_assignments, resolve_cli_prompt,
    should_route_offline_mtp, strip_trailing_eos, validate_pipeline_parallel_args,
    validate_tensor_parallel_args,
};
use mlxcel::server::chat_template::ChatTemplateProcessor;
use mlxcel_core::drafter::DrafterKind;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

// issue #166: the offline MTP loop is constructed only when the operator
// explicitly passes `--draft-kind mtp`. An auto-detected MTP shape (no explicit
// flag) keeps the classic SpeculativeGenerator path, and DFlash / InternalMtp
// explicit kinds fall through to the deferred-error branch. These pins guard the
// routing decision without loading a model.
#[test]
fn should_route_offline_mtp_only_for_explicit_mtp() {
    assert!(should_route_offline_mtp(true, DrafterKind::Mtp));
}

#[test]
fn should_route_offline_mtp_rejects_auto_detected_mtp() {
    // Auto-detect resolved to MTP but no explicit `--draft-kind`: stay on the
    // classic path for backward compatibility.
    assert!(!should_route_offline_mtp(false, DrafterKind::Mtp));
}

#[test]
fn should_route_offline_mtp_rejects_other_explicit_kinds() {
    assert!(!should_route_offline_mtp(true, DrafterKind::Dflash));
    assert!(!should_route_offline_mtp(true, DrafterKind::InternalMtp));
    assert!(!should_route_offline_mtp(false, DrafterKind::Dflash));
}

// issue #166: the offline MTP loop must exclude the terminal EOS / stop token so
// its output is byte-identical to the non-speculative `mlxcel generate` path,
// which breaks on EOS BEFORE pushing. `strip_trailing_eos` enforces that without
// loading a model.
#[test]
fn strip_trailing_eos_drops_terminal_stop_token() {
    // Common temp-0 case: the generator leaked the terminal EOS at the end.
    let tokens = vec![10, 20, 30, 2];
    assert_eq!(strip_trailing_eos(tokens, &[2]), vec![10, 20, 30]);
}

#[test]
fn strip_trailing_eos_leaves_output_without_eos_unchanged() {
    // max_tokens stop (no EOS emitted): nothing to strip.
    let tokens = vec![10, 20, 30];
    assert_eq!(strip_trailing_eos(tokens, &[2]), vec![10, 20, 30]);
}

#[test]
fn strip_trailing_eos_first_token_eos_yields_empty() {
    // EOS as the only / first token truncates to an empty output, mirroring the
    // server breaking at the first EOS occurrence.
    assert_eq!(strip_trailing_eos(vec![2], &[2]), Vec::<i32>::new());
    assert_eq!(strip_trailing_eos(vec![2, 10, 20], &[2]), Vec::<i32>::new());
}

#[test]
fn strip_trailing_eos_empty_eos_set_is_noop() {
    let tokens = vec![10, 20, 2, 30];
    assert_eq!(strip_trailing_eos(tokens, &[]), vec![10, 20, 2, 30]);
}

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

    let prompt = resolve_cli_prompt("Hello", true, Some(&processor), 0, 0, 0);

    assert_eq!(prompt, "Hello");
}

#[test]
fn resolve_cli_prompt_falls_back_on_template_errors() {
    let processor = ChatTemplateProcessor::with_template("{% if %}".to_string());

    let prompt = resolve_cli_prompt("Hello", false, Some(&processor), 0, 0, 0);

    assert_eq!(prompt, "Hello");
}

#[test]
fn vlm_chat_template_renders_video_content_part_in_user_turn() {
    // A template that handles both image and video content items. The video
    // marker must land alongside the text (inside the user turn), not before
    // it. That placement is what lets the Gemma 4 Unified video path produce a
    // grounded answer (issue #164).
    let template = "user: {% for item in messages[0]['content'] %}\
        {% if item['type'] == 'image' %}<IMG>\
        {% elif item['type'] == 'video' %}<VID>\
        {% elif item['type'] == 'text' %}{{ item['text'] }}\
        {% endif %}{% endfor %}"
        .to_string();
    let processor = ChatTemplateProcessor::with_template(template);

    // One video + the question: marker precedes the text within the turn.
    let prompt = apply_vlm_chat_template(&processor, "Describe this video.", 0, 1, 0);
    assert_eq!(prompt, "user: <VID>Describe this video.");

    // Image + video together render in image-then-video order.
    let mixed = apply_vlm_chat_template(&processor, "Q", 1, 1, 0);
    assert_eq!(mixed, "user: <IMG><VID>Q");
}

#[test]
fn vlm_chat_template_omits_video_when_template_lacks_video_support() {
    // A template that handles images but NOT video must not emit a video
    // content part (it would render the raw item as text). The ViT-backed
    // gemma4 VLM path relies on this: it keeps its splice-after-BOS behavior.
    let template = "user: {% for item in messages[0]['content'] %}\
        {% if item['type'] == 'image' %}<IMG>\
        {% elif item['type'] == 'text' %}{{ item['text'] }}\
        {% endif %}{% endfor %}"
        .to_string();
    let processor = ChatTemplateProcessor::with_template(template);
    assert!(!processor.supports_video_content());

    // num_videos > 0 but the template has no video branch: no marker emitted.
    let prompt = apply_vlm_chat_template(&processor, "Q", 0, 1, 0);
    assert_eq!(prompt, "user: Q");
}

#[test]
fn vlm_chat_template_renders_audio_content_part_in_user_turn() {
    // A Gemma-4-style template that handles image and audio content items. The
    // `<|audio|>` marker must land inside the user turn (an audio block in the
    // model turn forces an immediate EOS, issue #436) and, for Gemma 4, AFTER
    // the prompt text (issue #797): upstream mlx-vlm's `_format_list_with_image_type`
    // builds the user content as `[image]*n + [text] + [audio]*n`, and the
    // server audio path splices the block right before the user turn's closing
    // `<end_of_turn>`. Both land audio after the text.
    let template = "user: {% for item in messages[0]['content'] %}\
        {% if item['type'] == 'image' %}<IMG>\
        {% elif item['type'] == 'audio' %}<|audio|>\
        {% elif item['type'] == 'text' %}{{ item['text'] }}\
        {% endif %}{% endfor %}"
        .to_string();
    let processor = ChatTemplateProcessor::with_template(template);
    assert!(processor.supports_audio_content());

    // One audio clip + the question: the audio marker follows the text within
    // the user turn, so `expand_gemma4_audio_tokens` wraps it right after the
    // prompt, matching the server splice and the reference frame.
    let prompt = apply_vlm_chat_template(&processor, "Transcribe this audio.", 0, 0, 1);
    assert_eq!(prompt, "user: Transcribe this audio.<|audio|>");

    // Image + audio together render as image (before text) then text then audio.
    let mixed = apply_vlm_chat_template(&processor, "Transcribe this audio.", 1, 0, 1);
    assert_eq!(mixed, "user: <IMG>Transcribe this audio.<|audio|>");
}

#[test]
fn vlm_chat_template_omits_audio_when_template_lacks_audio_support() {
    // A template that handles images but NOT audio must not emit an audio
    // content part (it would render the raw item as text and break the prompt).
    let template = "user: {% for item in messages[0]['content'] %}\
        {% if item['type'] == 'image' %}<IMG>\
        {% elif item['type'] == 'text' %}{{ item['text'] }}\
        {% endif %}{% endfor %}"
        .to_string();
    let processor = ChatTemplateProcessor::with_template(template);
    assert!(!processor.supports_audio_content());

    // num_audios > 0 but the template has no audio branch: no marker emitted.
    let prompt = apply_vlm_chat_template(&processor, "Q", 0, 0, 1);
    assert_eq!(prompt, "user: Q");
}

#[test]
fn resolve_cli_prompt_routes_audio_through_vlm_template() {
    // Audio-bearing request on a Gemma-4-style template (image + audio content
    // support) renders the `<|audio|>` marker in the user turn (the VLM path),
    // even with no images or videos. `apply_vlm_chat_template` takes the
    // multimodal (content-list) path because the template handles image
    // content, as the real Gemma 4 template does.
    let template = "user: {% for item in messages[0]['content'] %}\
        {% if item['type'] == 'image' %}<IMG>\
        {% elif item['type'] == 'audio' %}<|audio|>\
        {% elif item['type'] == 'text' %}{{ item['text'] }}\
        {% endif %}{% endfor %}"
        .to_string();
    let processor = ChatTemplateProcessor::with_template(template);

    // Audio follows the text within the user turn (issue #797), matching the
    // reference frame and the server audio-after-text splice.
    let prompt = resolve_cli_prompt("Transcribe.", false, Some(&processor), 0, 0, 1);
    assert_eq!(prompt, "user: Transcribe.<|audio|>");
}

#[test]
fn resolve_cli_prompt_keeps_text_path_for_audio_on_plain_template() {
    // A plain (string-content) template that does NOT handle audio content must
    // keep the byte-identical text-only path even when an audio clip is present
    // (no regression for non-audio models). The per-family token expansion then
    // places the audio block instead.
    let processor = ChatTemplateProcessor::with_template(
        "{{ messages[0].role }}: {{ messages[0].content }}".to_string(),
    );
    assert!(!processor.supports_audio_content());

    let with_audio = resolve_cli_prompt("Hello", false, Some(&processor), 0, 0, 1);
    let without_audio = resolve_cli_prompt("Hello", false, Some(&processor), 0, 0, 0);
    assert_eq!(with_audio, without_audio);
    assert_eq!(with_audio, "user: Hello");
}

#[test]
fn gemma4_unified_audio_prompt_matches_reference_framing() {
    // Regression for issue #797. On the Gemma 4 12B Unified audio path the CLI
    // used to render the `<|audio|>` marker BEFORE the prompt text, an
    // out-of-distribution frame that deterministically flipped the model from
    // transcription into answering the perceived content on acoustically hard
    // clips. The reference (upstream mlx-vlm `_format_list_with_image_type` for
    // `gemma4`) and the server audio path both place the audio block AFTER the
    // text. This pins the rendered CLI prompt for a Gemma 4 12B Unified audio
    // request using the real template shape: BOS, the system-block guard, the
    // content-parts loop, and the closed thinking-channel scaffold that the
    // generation prompt emits when thinking defaults OFF (issue #686).
    let template = r#"{{- bos_token -}}
{%- if (enable_thinking is defined and enable_thinking) or tools or messages[0]['role'] in ['system', 'developer'] -%}
{{- '<|turn>system\n<turn|>\n' -}}
{%- endif -%}
{%- for message in messages -%}
{{- '<|turn>' + message['role'] + '\n' -}}
{%- if message['content'] is string -%}
{{- message['content'] | trim -}}
{%- else -%}
{%- for item in message['content'] -%}
{%- if item['type'] == 'text' -%}{{- item['text'] | trim -}}
{%- elif item['type'] == 'image' -%}{{- '<|image|>' -}}
{%- elif item['type'] == 'audio' -%}{{- '<|audio|>' -}}
{%- endif -%}
{%- endfor -%}
{%- endif -%}
{{- '<turn|>\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- '<|turn>model\n' -}}
{%- if not enable_thinking | default(false) -%}{{- '<|channel>thought\n<channel|>' -}}{%- endif -%}
{%- endif -%}"#
        .to_string();

    let processor = ChatTemplateProcessor::with_template(template);
    assert!(processor.supports_image_content());
    assert!(processor.supports_audio_content());

    const PROMPT: &str = "이 음성을 들리는 그대로 한국어로 받아쓰기 하세요.";
    let prompt = apply_vlm_chat_template(&processor, PROMPT, 0, 0, 1);

    // The audio marker lands AFTER the prompt text (issue #797).
    let text_pos = prompt.find("받아쓰기").expect("prompt text present");
    let audio_pos = prompt.find("<|audio|>").expect("audio marker present");
    assert!(
        text_pos < audio_pos,
        "audio marker must follow the prompt text: {prompt}"
    );

    // Thinking defaults OFF: the closed `<|channel>thought\n<channel|>` scaffold
    // primes a direct answer and no system turn is emitted (issue #686).
    assert!(
        prompt.contains("<|turn>model\n<|channel>thought\n<channel|>"),
        "closed thinking-channel scaffold must render: {prompt}"
    );
    assert!(
        !prompt.contains("<|turn>system"),
        "no system turn without a system message: {prompt}"
    );

    // The full rendered frame, pinned exactly.
    assert_eq!(
        prompt,
        "<|turn>user\n\
         이 음성을 들리는 그대로 한국어로 받아쓰기 하세요.<|audio|><turn|>\n\
         <|turn>model\n<|channel>thought\n<channel|>"
    );

    // CLI / server parity: the server renders the chat text-only and splices the
    // BOA/AUDIO/EOA block right before the user turn's closing `<end_of_turn>`
    // (`expand_gemma4_audio_tokens_for_server`). At the marker level that is the
    // text-only prompt with `<|audio|>` inserted before the user turn's first
    // `<turn|>`. It must equal the CLI audio prompt.
    let text_only = apply_user_chat_template(&processor, PROMPT);
    let server_equiv = text_only.replacen("<turn|>", "<|audio|><turn|>", 1);
    assert_eq!(
        prompt, server_equiv,
        "CLI and server must frame the audio user turn identically"
    );
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
            image_soft_tokens: None,
            audio: None,
            video: Vec::new(),
            fps: 2.0,
            output_audio: None,
            speaker: "ethan".to_string(),
            max_tokens: 16,
            profile: false,
            no_chat_template: false,
            show_reasoning: false,
            recommend_quant: false,
            estimate_memory: false,
            force_memory: false,
            turbo: mlxcel::cli::turbo_args::TurboKvCacheArgs::default(),
            diffusion: crate::DiffusionCliOptions::default(),
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
            seed: None,
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
        // (A4): default to None so existing tests stay on
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

    // Tensor parallelism + PP is now accepted (2D PP × TP composition landed). Positive coverage for the 2D path lives in
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
    // the validator no longer rejects PP + TP.
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
