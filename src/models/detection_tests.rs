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

use super::ModelType;
use super::detection::{detect_hunyuan_model_type, detect_text_or_vlm, has_vision_config};
use serde_json::json;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn has_vision_config_detects_vlm_configs() {
    assert!(has_vision_config(&json!({ "vision_config": {} })));
    assert!(!has_vision_config(&json!({ "text_config": {} })));
}

#[test]
fn detect_text_or_vlm_prefers_vlm_when_vision_config_exists() {
    let vlm = detect_text_or_vlm(
        &json!({ "vision_config": {} }),
        ModelType::Gemma3,
        ModelType::Gemma3VLM,
    );
    let text = detect_text_or_vlm(&json!({}), ModelType::Gemma3, ModelType::Gemma3VLM);

    assert_eq!(vlm, ModelType::Gemma3VLM);
    assert_eq!(text, ModelType::Gemma3);
}

#[test]
fn detect_hunyuan_model_type_uses_num_experts() {
    assert_eq!(
        detect_hunyuan_model_type(&json!({ "num_experts": 4 })),
        ModelType::HunyuanMoe
    );
    assert_eq!(
        detect_hunyuan_model_type(&json!({ "num_experts": 1 })),
        ModelType::HunyuanV1Dense
    );
    assert_eq!(
        detect_hunyuan_model_type(&json!({})),
        ModelType::HunyuanV1Dense
    );
}

fn temp_path(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("mlxcel_detection_test_{name}_{nanos}"))
}

#[test]
fn inkling_model_type_aliases_are_detected() {
    for model_type in ["inkling_mm_model", "inkling"] {
        let model_dir = temp_path(model_type);
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(
            model_dir.join("config.json"),
            serde_json::to_vec(&json!({
                "architectures": ["InklingForConditionalGeneration"],
                "model_type": model_type,
                "text_config": {"model_type": "inkling_text"}
            }))
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            super::detection::get_model_type(&model_dir).unwrap(),
            ModelType::Inkling
        );
        fs::remove_dir_all(model_dir).unwrap();
    }
}

#[test]
fn isolated_inkling_mtp_download_is_not_a_standalone_model() {
    use std::io::Write;

    let model_dir = temp_path("inkling_mtp_only");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        json!({
            "model_type": "inkling_mm_model",
            "text_config": {"model_type": "inkling"},
            "mtp_config": {"num_nextn_predict_layers": 3}
        })
        .to_string(),
    )
    .unwrap();
    let name = "model.mtp.layers.0.embed_norm.weight";
    let mut header =
        format!("{{\"{name}\":{{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}}}");
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut file = fs::File::create(model_dir.join("mtp.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    file.write_all(&1.0_f32.to_le_bytes()).unwrap();
    let error = super::detection::get_model_type(&model_dir).unwrap_err();
    assert!(error.to_string().contains("isolated Inkling MTP drafter"));
    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn deepseek_v4_model_type_is_detected() {
    let model_dir = temp_path("deepseek_v4");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "model_type": "deepseek_v4",
            "architectures": ["DeepseekV4ForCausalLM"],
            "vocab_size": 129280,
            "hidden_size": 4096,
            "num_hidden_layers": 43
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::DeepSeekV4);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn whisper_model_type_is_detected() {
    let model_dir = temp_path("whisper_asr");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "model_type": "whisper",
            "num_mel_bins": 80,
            "d_model": 384,
            "encoder_attention_heads": 6,
            "encoder_layers": 4,
            "decoder_attention_heads": 6,
            "decoder_layers": 4,
            "vocab_size": 51865
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::Whisper);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn florence2_model_type_is_detected() {
    // Florence-2 declares `model_type: "florence2"` at the top level. The
    // real checkpoint's `vision_config.model_type` is an empty string, so
    // detection must key off the top-level value alone (a `vision_config`
    // is present but never consulted for this family).
    let model_dir = temp_path("florence2_vlm");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "model_type": "florence2",
            "architectures": ["Florence2ForConditionalGeneration"],
            "is_encoder_decoder": true,
            "projection_dim": 768,
            "text_config": {
                "model_type": "florence2_language",
                "d_model": 768,
                "encoder_layers": 6,
                "decoder_layers": 6,
                "vocab_size": 51289
            },
            "vision_config": {
                "model_type": "",
                "dim_embed": [128, 256, 512, 1024]
            }
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::Florence2VLM);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn muse_glimmer_model_type_is_detected() {
    let model_dir = temp_path("muse_glimmer_vlm");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "model_type": "muse_glimmer",
            "architectures": ["MuseGlimmerForConditionalGeneration"],
            "image_token_id": 200092,
            "video_token_id": 200091,
            "text_config": {
                "model_type": "muse_glimmer_text",
                "hidden_size": 6656,
                "intermediate_size": 19968,
                "num_hidden_layers": 52,
                "num_attention_heads": 32,
                "num_key_value_heads": 2,
                "head_dim": 128,
                "rms_norm_eps": 1e-5,
                "vocab_size": 202048
            },
            "vision_config": {
                "model_type": "muse_glimmer_vision_model"
            }
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::MuseGlimmerVLM);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn gpt2_model_type_is_detected() {
    // GPT-2 configs use the original OpenAI field names (`n_embd` / `n_head` /
    // `n_layer`), so detection must key off `model_type` alone.
    let model_dir = temp_path("gpt2_text");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "model_type": "gpt2",
            "architectures": ["GPT2LMHeadModel"],
            "n_embd": 768,
            "n_head": 12,
            "n_layer": 12,
            "n_positions": 1024,
            "n_ctx": 1024,
            "layer_norm_epsilon": 1e-05,
            "vocab_size": 50257,
            "activation_function": "gelu_new"
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::Gpt2);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn gpt_bigcode_model_type_is_detected() {
    // GPT-BigCode reuses GPT-2's config field names and its `architectures`
    // entry starts with the same `GPT` prefix, so detection must key off
    // `model_type` alone and must not fall through to the `gpt2` arm.
    let model_dir = temp_path("gpt_bigcode_text");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "model_type": "gpt_bigcode",
            "architectures": ["GPTBigCodeForCausalLM"],
            "n_embd": 2048,
            "n_head": 16,
            "n_inner": 8192,
            "n_layer": 24,
            "n_positions": 2048,
            "layer_norm_epsilon": 1e-05,
            "vocab_size": 49280,
            "multi_query": true,
            "activation_function": "gelu_pytorch_tanh"
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::GptBigCode);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn gpt_neox_model_type_is_detected() {
    // GPT-NeoX shares the `GPT` prefix in `architectures` with GPT-2 and
    // GPT-BigCode but uses the modern `hidden_size` / `num_attention_heads`
    // config naming, so detection must key off `model_type` alone and must not
    // fall through to either GPT-2-lineage arm.
    let model_dir = temp_path("gpt_neox_text");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "model_type": "gpt_neox",
            "architectures": ["GPTNeoXForCausalLM"],
            "hidden_size": 2048,
            "num_attention_heads": 8,
            "num_hidden_layers": 16,
            "intermediate_size": 8192,
            "max_position_embeddings": 2048,
            "layer_norm_eps": 1e-05,
            "vocab_size": 50304,
            "rotary_emb_base": 10000,
            "rotary_pct": 0.25,
            "use_parallel_residual": true,
            "tie_word_embeddings": false,
            "hidden_act": "gelu"
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::GptNeoX);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn helium_model_type_is_detected() {
    // Helium's config is field-for-field a Llama config apart from `model_type`
    // and the smaller `rms_norm_eps`, so detection must key off `model_type`
    // alone and must not fall through to the Llama arm. The two decode
    // differently: Helium rotates interleaved RoPE pairs, Llama split-half ones.
    let model_dir = temp_path("helium_text");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "model_type": "helium",
            "architectures": ["HeliumForCausalLM"],
            "hidden_size": 2560,
            "num_attention_heads": 20,
            "num_key_value_heads": 20,
            "num_hidden_layers": 24,
            "intermediate_size": 7040,
            "head_dim": 128,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-08,
            "rope_theta": 100000.0,
            "attention_bias": false,
            "mlp_bias": false,
            "tie_word_embeddings": false,
            "vocab_size": 48000
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::Helium);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn fastvlm_model_type_is_detected() {
    for model_type in ["fastvlm", "llava_qwen2"] {
        let model_dir = temp_path(&format!("fastvlm_{model_type}"));
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(
            model_dir.join("config.json"),
            format!(
                r#"{{
                "model_type": "{model_type}",
                "hidden_size": 896,
                "num_hidden_layers": 24,
                "num_attention_heads": 14,
                "mm_projector_type": "mlp2x_gelu",
                "vision_config": {{ "image_size": 1024 }}
            }}"#
            ),
        )
        .unwrap();

        let detected = super::detection::get_model_type(&model_dir).unwrap();
        assert_eq!(detected, ModelType::FastVLM, "for model_type {model_type}");

        fs::remove_dir_all(model_dir).unwrap();
    }
}

#[test]
fn llava_qwen2_hyphen_stays_bunny() {
    // The hyphenated `llava-qwen2` is the Bunny family and must not route to
    // FastVLM (which owns the underscore `llava_qwen2`).
    let model_dir = temp_path("llava_qwen2_hyphen");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{ "model_type": "llava-qwen2", "hidden_size": 896 }"#,
    )
    .unwrap();
    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::LlavaBunnyVLM);
    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn qwen3_omni_moe_model_type_is_detected() {
    let model_dir = temp_path("qwen3_omni_moe");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "model_type": "qwen3_omni_moe",
            "thinker_config": {
                "text_config": { "num_hidden_layers": 48, "hidden_size": 2048 },
                "vision_config": { "depth": 27 },
                "audio_config": { "d_model": 1280 }
            }
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::Qwen3OmniMoe);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn hunyuan_vl_model_type_is_detected() {
    let model_dir = temp_path("hunyuan_vl");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "model_type": "hunyuan_vl",
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "vision_config": { "hidden_size": 1152, "num_hidden_layers": 27 }
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::HunyuanVLM);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn ernie4_5_moe_vl_model_type_is_detected() {
    let model_dir = temp_path("ernie4_5_moe_vl");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "model_type": "ernie4_5_moe_vl",
            "hidden_size": 2560,
            "num_hidden_layers": 28,
            "num_attention_heads": 20,
            "moe_num_experts": [64, 64],
            "vision_config": { "depth": 32, "embed_dim": 1280 }
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::Ernie45MoeVLM);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn deepseek_vl2_model_type_is_detected() {
    let model_dir = temp_path("deepseek_vl2");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "model_type": "deepseek_vl_v2",
            "tile_tag": "2D",
            "global_view_pos": "head",
            "candidate_resolutions": [[384, 384]],
            "language_config": { "model_type": "deepseek_v2", "hidden_size": 2048 },
            "vision_config": { "model_type": "vision", "width": 1152, "layers": 27, "patch_size": 14 },
            "projector_config": { "model_type": "mlp_projector" }
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::DeepSeekVL2);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn mllama_model_type_is_detected() {
    // Llama 3.2 Vision: a `mllama` checkpoint must resolve to the VLM route
    // instead of erroring with "Unsupported model type".
    let model_dir = temp_path("llama_3_2_vision");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "model_type": "mllama",
            "image_token_index": 128256,
            "text_config": {
                "model_type": "mllama",
                "hidden_size": 4096,
                "num_hidden_layers": 40,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "cross_attention_layers": [3, 8, 13, 18, 23, 28, 33, 38]
            },
            "vision_config": {
                "image_size": 560,
                "patch_size": 14,
                "hidden_size": 1280,
                "num_hidden_layers": 32,
                "num_global_layers": 8
            }
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::MllamaVLM);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn cohere2_moe_model_type_is_detected() {
    // A `cohere2_moe` checkpoint must resolve to the Command MoE runtime instead
    // of erroring with "Unsupported model type", and must not collide with the
    // dense `cohere2` arm.
    let model_dir = temp_path("cohere2_moe");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "model_type": "cohere2_moe",
            "hidden_size": 1024,
            "head_dim": 128,
            "num_hidden_layers": 36,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_num_shared_experts": 4,
            "vocab_size": 256000
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::Cohere2Moe);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn mellum_model_type_is_detected() {
    let model_dir = temp_path("mellum_code");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "model_type": "mellum",
            "architectures": ["MellumForCausalLM"],
            "hidden_size": 2304,
            "head_dim": 128,
            "num_hidden_layers": 28,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "num_experts": 64,
            "vocab_size": 98304
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::Mellum);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn gemma4_detection_stays_on_text_route_without_vision_weights() {
    let model_dir = temp_path("gemma4_text_route");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "model_type": "gemma4",
            "vision_config": {},
            "text_config": { "model_type": "gemma4_text" }
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::Gemma4);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn gemma4_detection_uses_vlm_route_when_vision_weights_exist() {
    let model_dir = temp_path("gemma4_vlm_route");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "model_type": "gemma4",
            "vision_config": {},
            "text_config": { "model_type": "gemma4_text" }
        }"#,
    )
    .unwrap();
    fs::write(
        model_dir.join("model.safetensors.index.json"),
        r#"{
            "weight_map": {
                "vision_tower.encoder.layers.0.input_layernorm.weight": "model-00001-of-00001.safetensors"
            }
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::Gemma4VLM);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn gemma4_detection_uses_vlm_route_for_nvfp4_prefixed_vision_weights() {
    // ModelOpt NVFP4 exports nest the vision front-end under a leading
    // `model.` (`model.vision_tower.` / `model.embed_vision.`) instead of the
    // MLX-community unprefixed `vision_tower.` / `embed_vision.` keys. Both
    // prefixed forms must also route to Gemma4VLM so `normalize_nvfp4_keys`
    // gets a chance to strip the prefix later (issue #749).
    for weight_key in [
        "model.vision_tower.encoder.layers.0.input_layernorm.weight",
        "model.embed_vision.embedding_projection.weight",
    ] {
        let model_dir = temp_path(&format!(
            "gemma4_vlm_route_nvfp4_{}",
            weight_key.replace(['.', '/'], "_")
        ));
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(
            model_dir.join("config.json"),
            r#"{
                "model_type": "gemma4",
                "vision_config": {},
                "text_config": { "model_type": "gemma4_text" }
            }"#,
        )
        .unwrap();
        fs::write(
            model_dir.join("model.safetensors.index.json"),
            format!(
                r#"{{
                    "weight_map": {{
                        "{weight_key}": "model-00001-of-00001.safetensors"
                    }}
                }}"#
            ),
        )
        .unwrap();

        let detected = super::detection::get_model_type(&model_dir).unwrap();
        assert_eq!(
            detected,
            ModelType::Gemma4VLM,
            "for weight key {weight_key}"
        );

        fs::remove_dir_all(model_dir).unwrap();
    }
}

#[test]
fn idefics3_smolvlm_instruct_model_type_is_detected() {
    // SmolVLM-Instruct ships as an Idefics3 checkpoint: top-level
    // `model_type: "idefics3"` (`Idefics3ForConditionalGeneration`) with a Llama
    // `text_config` and a SigLIP-style `vision_config` that is itself tagged
    // `idefics3`. It must resolve to the SmolVLM runtime instead of erroring with
    // "Unsupported model type: idefics3". Config shape mirrors the released
    // HuggingFaceTB/SmolVLM-Instruct config.json.
    let model_dir = temp_path("smolvlm_instruct_idefics3");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "architectures": ["Idefics3ForConditionalGeneration"],
            "model_type": "idefics3",
            "image_token_id": 49153,
            "image_seq_len": 81,
            "scale_factor": 3,
            "tie_word_embeddings": false,
            "text_config": {
                "model_type": "llama",
                "hidden_size": 2048,
                "intermediate_size": 8192,
                "num_hidden_layers": 24,
                "num_attention_heads": 32,
                "num_key_value_heads": 32,
                "head_dim": 64,
                "rms_norm_eps": 1e-05,
                "rope_theta": 273768.0,
                "vocab_size": 49155,
                "tie_word_embeddings": false
            },
            "vision_config": {
                "model_type": "idefics3",
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "num_hidden_layers": 27,
                "num_attention_heads": 16,
                "patch_size": 14,
                "image_size": 384
            },
            "vocab_size": 49155
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::SmolVLM);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn lfm2_vl_model_type_is_detected_both_spellings() {
    // LFM2-VL ships model_type "lfm2-vl" (hyphen); the underscore alias must also
    // resolve. Both map to the LFM2-VL runtime, not "Unsupported model type".
    for mt in ["lfm2-vl", "lfm2_vl"] {
        let model_dir = temp_path(&format!("lfm2_vl_{}", mt.replace('-', "_")));
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(
            model_dir.join("config.json"),
            format!(
                r#"{{
                    "model_type": "{mt}",
                    "image_token_index": 396,
                    "downsample_factor": 2,
                    "text_config": {{ "model_type": "lfm2", "hidden_size": 1024, "num_hidden_layers": 16 }},
                    "vision_config": {{ "model_type": "siglip2_vision_model", "hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "patch_size": 16, "num_patches": 256 }}
                }}"#
            ),
        )
        .unwrap();
        let detected = super::detection::get_model_type(&model_dir).unwrap();
        assert_eq!(detected, ModelType::Lfm2VL, "spelling {mt}");
    }
}

/// Text-only Youtu-LLM ships under two vendor labels: `youtu` on
/// `tencent/Youtu-LLM-2B` and `youtu_llm` on one community conversion. Both
/// must reach the Youtu MLA decoder rather than the unsupported-model arm.
#[test]
fn youtu_llm_model_type_is_detected_for_both_vendor_labels() {
    for mt in ["youtu", "youtu_llm"] {
        let model_dir = temp_path(&format!("youtu_llm_{mt}"));
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(
            model_dir.join("config.json"),
            format!(
                r#"{{
                    "model_type": "{mt}",
                    "architectures": ["YoutuForCausalLM"],
                    "vocab_size": 128256,
                    "hidden_size": 2048,
                    "intermediate_size": 6144,
                    "num_hidden_layers": 32,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 16,
                    "kv_lora_rank": 512,
                    "q_lora_rank": 1536,
                    "qk_rope_head_dim": 64,
                    "qk_nope_head_dim": 128,
                    "v_head_dim": 128,
                    "rope_theta": 1600000,
                    "rope_interleave": true,
                    "tie_word_embeddings": true
                }}"#
            ),
        )
        .unwrap();

        let detected = super::detection::get_model_type(&model_dir).unwrap();
        assert_eq!(detected, ModelType::YoutuLLM, "label {mt}");

        fs::remove_dir_all(model_dir).unwrap();
    }
}

/// `mlx-community/Youtu-LLM-2B-4bit` relabels itself `deepseek_v2` so mlx-lm
/// can load it, while keeping `architectures: ["YoutuForCausalLM"]` and an
/// `auto_map` that points at the vendor's own modules. It stays on the
/// DeepSeek-V2 route anyway.
///
/// This is a guard against re-adding an architecture-string split to the
/// `deepseek_v2` arm. #1371 first added one, on the belief that the DeepSeek-V2
/// decoder mishandled that export. Greedy decode of the real checkpoint through
/// both decoders, against an mlx-lm `deepseek_v2` oracle on the same weights,
/// showed otherwise: on a chat-templated prompt the DeepSeek-V2 route matches
/// the oracle for all 32 tokens, and on a raw prompt it matches for 18. A split
/// would move working checkpoints onto a different decoder for no gain, so the
/// label decides the route and only the two vendor labels are new.
#[test]
fn deepseek_v2_label_keeps_the_deepseek_v2_route() {
    let cases: [(&str, Option<&str>); 3] = [
        ("deepseek_v2_youtu_relabel", Some("YoutuForCausalLM")),
        ("deepseek_v2_genuine", Some("DeepseekV2ForCausalLM")),
        ("deepseek_v2_no_architectures", None),
    ];

    for (name, architecture) in cases {
        let model_dir = temp_path(name);
        fs::create_dir_all(&model_dir).unwrap();
        let architectures = match architecture {
            Some(arch) => format!(r#""architectures": ["{arch}"],"#),
            None => String::new(),
        };
        fs::write(
            model_dir.join("config.json"),
            format!(
                r#"{{
                    "model_type": "deepseek_v2",
                    {architectures}
                    "vocab_size": 102400,
                    "hidden_size": 2048,
                    "num_hidden_layers": 27,
                    "num_attention_heads": 16,
                    "kv_lora_rank": 512,
                    "qk_rope_head_dim": 64,
                    "qk_nope_head_dim": 128,
                    "v_head_dim": 128
                }}"#
            ),
        )
        .unwrap();

        let detected = super::detection::get_model_type(&model_dir).unwrap();
        assert_eq!(detected, ModelType::DeepSeekV2, "case {name}");

        fs::remove_dir_all(model_dir).unwrap();
    }
}

#[test]
fn granite_vision_model_type_is_detected() {
    // MLX conversions ship `model_type: "granite_vision"`.
    let model_dir = temp_path("granite_vision");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "model_type": "granite_vision",
            "image_token_index": 49155,
            "vision_feature_layer": [-24, -20, -12, -1],
            "text_config": {"model_type": "granite", "hidden_size": 2048},
            "vision_config": {"model_type": "siglip_vision_model", "num_hidden_layers": 27,
                "hidden_size": 1152, "intermediate_size": 4304, "num_attention_heads": 16,
                "patch_size": 14}
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::GraniteVisionVLM);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn llava_next_with_granite_text_routes_to_granite_vision() {
    // The original IBM checkpoint ships `llava_next` + a `granite` text config.
    let model_dir = temp_path("llava_next_granite");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "model_type": "llava_next",
            "image_token_index": 49155,
            "text_config": {"model_type": "granite", "hidden_size": 2048},
            "vision_config": {"model_type": "siglip_vision_model", "num_hidden_layers": 27,
                "hidden_size": 1152, "intermediate_size": 4304, "num_attention_heads": 16,
                "patch_size": 14}
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::GraniteVisionVLM);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn llava_next_without_granite_stays_llava() {
    // A vanilla LLaVA-Next (llama/mistral/qwen2 text) must still route to LLaVA.
    let model_dir = temp_path("llava_next_vanilla");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "model_type": "llava_next",
            "text_config": {"model_type": "llama", "hidden_size": 4096},
            "vision_config": {"model_type": "clip_vision_model"}
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::LlavaVLM);

    fs::remove_dir_all(model_dir).unwrap();
}

// =============================================================================
// DFlash speculative drafter rejection (#1168)
// =============================================================================

/// `config.json` of the published `qwen3.5-27b-dflash` drafter, trimmed to the
/// fields detection reads. The point of the fixture is that `model_type` says
/// `qwen3`: without the structural DFlash arm this routes to `ModelType::Qwen3`
/// and then to `Qwen3Model::load`, which dies on
/// `Weight not found: model.embed_tokens.weight`.
const DFLASH_DRAFTER_CONFIG: &str = r#"{
    "architectures": ["DFlashDraftModel"],
    "auto_map": {"AutoModel": "dflash.DFlashDraftModel"},
    "block_size": 16,
    "dflash_config": {"mask_token_id": 248070, "target_layer_ids": [1, 16, 31, 46, 61]},
    "hidden_size": 5120,
    "model_type": "qwen3",
    "num_hidden_layers": 5,
    "num_target_layers": 64,
    "vocab_size": 248320
}"#;

#[test]
fn dflash_drafter_is_rejected_as_a_standalone_model() {
    let model_dir = temp_path("dflash_drafter");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(model_dir.join("config.json"), DFLASH_DRAFTER_CONFIG).unwrap();

    let error = super::detection::get_model_type(&model_dir)
        .expect_err("a DFlash drafter is not a standalone model")
        .to_string();

    assert!(
        error.contains("DFlash speculative drafter"),
        "the error must name the real problem, got: {error}",
    );
    assert!(
        error.contains("--draft-model"),
        "the error must point at the flag that does take a drafter, got: {error}",
    );
    assert!(
        !error.contains("Weight not found"),
        "the weight-lookup symptom must not be what a user sees, got: {error}",
    );

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn dflash_drafter_is_rejected_on_either_marker_alone() {
    for (name, config) in [
        (
            "architectures_only",
            r#"{"architectures": ["DFlashDraftModel"], "model_type": "qwen3"}"#,
        ),
        (
            "dflash_config_only",
            r#"{"model_type": "qwen3", "dflash_config": {"mask_token_id": 248070}}"#,
        ),
    ] {
        let model_dir = temp_path(name);
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("config.json"), config).unwrap();

        let error = super::detection::get_model_type(&model_dir)
            .expect_err("marker alone is sufficient")
            .to_string();
        assert!(
            error.contains("DFlash speculative drafter"),
            "{name}: {error}"
        );

        fs::remove_dir_all(model_dir).unwrap();
    }
}

#[test]
fn ordinary_qwen3_full_model_still_detects_as_qwen3() {
    // Non-regression control for the classic `--draft-model` path: a small
    // Qwen 3 full model used as a classic drafter carries neither DFlash
    // marker and must keep loading exactly as before. It resolves to
    // `DrafterKind::Dflash` by default, which is why the rejection above keys
    // on checkpoint structure and not on the resolved drafter kind.
    let model_dir = temp_path("ordinary_qwen3_drafter");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "hidden_size": 1024,
            "num_hidden_layers": 28
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::Qwen3);

    fs::remove_dir_all(model_dir).unwrap();
}

// ---------------------------------------------------------------------------
// Embedding detection (#1353): encoder-only `model_type`, embedding
// `architectures[0]`, `modules.json` Pooling entry, `1_Pooling/config.json`,
// and the negative cases that must keep routing to the generators.
// ---------------------------------------------------------------------------

/// Write `config.json` plus any extra files (relative path, contents) into a
/// fresh temp checkpoint directory and run detection on it.
fn detect_layout(
    name: &str,
    config: &serde_json::Value,
    extra_files: &[(&str, &str)],
) -> anyhow::Result<ModelType> {
    let model_dir = temp_path(name);
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(model_dir.join("config.json"), config.to_string()).unwrap();
    for (relative, contents) in extra_files {
        let path = model_dir.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, contents).unwrap();
    }
    let result = super::detection::get_model_type(&model_dir);
    fs::remove_dir_all(model_dir).unwrap();
    result
}

const POOLING_MODULES: &str = r#"[
    {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
    {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
    {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"}
]"#;

const LOGIT_SCORE_MODULES: &str = r#"[
    {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
    {"idx": 1, "name": "1", "path": "1_LogitScore", "type": "custom.LogitScore"}
]"#;

#[test]
fn encoder_only_model_types_detect_as_embedding_without_pooling_files() {
    let cases = [
        ("bert", "BertForMaskedLM", ModelType::Bert),
        (
            "xlm-roberta",
            "XLMRobertaForMaskedLM",
            ModelType::XlmRoberta,
        ),
        ("modernbert", "ModernBertForMaskedLM", ModelType::ModernBert),
        ("siglip", "SiglipModel", ModelType::SiglipText),
    ];
    for (model_type, arch, expected) in cases {
        let detected = detect_layout(
            &format!("encoder_{model_type}"),
            &json!({"model_type": model_type, "architectures": [arch], "hidden_size": 8}),
            &[],
        )
        .unwrap();
        assert_eq!(detected, expected, "{model_type}");
    }
}

#[test]
fn embedding_architectures_route_generator_model_types_to_embedding_variants() {
    let cases = [
        ("llama", "LlamaBidirectionalModel", ModelType::LlamaBidirec),
        (
            "llama_nemotron_vl",
            "LlamaNemotronVLModel",
            ModelType::LlamaNemotronVLEmbedding,
        ),
        ("lfm2", "Lfm2BidirectionalModel", ModelType::Lfm2Embedding),
        ("idefics3", "ColIdefics3", ModelType::ColIdefics3),
        ("qwen2_5_vl", "ColQwen2_5", ModelType::ColQwen25),
        ("qwen2_5_vl", "ColQwen2ForRetrieval", ModelType::ColQwen25),
        ("siglip", "SiglipTextModel", ModelType::SiglipText),
        ("bert", "BertModel", ModelType::Bert),
        ("xlm-roberta", "XLMRobertaModel", ModelType::XlmRoberta),
        ("modernbert", "ModernBertModel", ModelType::ModernBert),
    ];
    for (model_type, arch, expected) in cases {
        let detected = detect_layout(
            &format!("arch_{arch}"),
            &json!({"model_type": model_type, "architectures": [arch], "hidden_size": 8}),
            &[],
        )
        .unwrap();
        assert_eq!(detected, expected, "{arch}");
    }
}

#[test]
fn gemma3_text_model_is_embedding_only_with_bidirectional_attention() {
    let bidirectional = detect_layout(
        "embeddinggemma",
        &json!({
            "model_type": "gemma3_text",
            "architectures": ["Gemma3TextModel"],
            "use_bidirectional_attention": true,
            "hidden_size": 8
        }),
        &[],
    )
    .unwrap();
    assert_eq!(bidirectional, ModelType::Gemma3Embedding);

    let causal = detect_layout(
        "gemma3_text_causal",
        &json!({
            "model_type": "gemma3_text",
            "architectures": ["Gemma3TextModel"],
            "hidden_size": 8
        }),
        &[],
    )
    .unwrap();
    assert_eq!(causal, ModelType::Gemma3);
}

#[test]
fn ministral3_model_is_embedding_only_when_not_causal() {
    let bidirectional = detect_layout(
        "ministral3_embed",
        &json!({
            "model_type": "ministral3",
            "architectures": ["Ministral3Model"],
            "is_causal": false,
            "hidden_size": 8
        }),
        &[],
    )
    .unwrap();
    assert_eq!(bidirectional, ModelType::Ministral3Embedding);

    let causal = detect_layout(
        "ministral3_causal",
        &json!({
            "model_type": "ministral3",
            "architectures": ["Ministral3ForCausalLM"],
            "hidden_size": 8
        }),
        &[],
    )
    .unwrap();
    assert_eq!(causal, ModelType::Ministral3);
}

#[test]
fn modules_json_pooling_entry_routes_qwen3_to_qwen3_embedding() {
    let detected = detect_layout(
        "qwen3_modules_pooling",
        &json!({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], "hidden_size": 8}),
        &[("modules.json", POOLING_MODULES)],
    )
    .unwrap();
    assert_eq!(detected, ModelType::Qwen3Embedding);
}

#[test]
fn one_pooling_config_routes_qwen3_and_qwen3_vl_to_embedding_variants() {
    let detected = detect_layout(
        "qwen3_one_pooling",
        &json!({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], "hidden_size": 8}),
        &[(
            "1_Pooling/config.json",
            r#"{"pooling_mode_lasttoken": true}"#,
        )],
    )
    .unwrap();
    assert_eq!(detected, ModelType::Qwen3Embedding);

    let detected = detect_layout(
        "qwen3_vl_one_pooling",
        &json!({"model_type": "qwen3_vl", "architectures": ["Qwen3VLForConditionalGeneration"]}),
        &[(
            "1_Pooling/config.json",
            r#"{"pooling_mode_lasttoken": true}"#,
        )],
    )
    .unwrap();
    assert_eq!(detected, ModelType::Qwen3VLEmbedding);
}

#[test]
fn qwen3_for_causal_lm_without_pooling_layout_stays_qwen3() {
    let detected = detect_layout(
        "qwen3_plain",
        &json!({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], "hidden_size": 8}),
        &[],
    )
    .unwrap();
    assert_eq!(detected, ModelType::Qwen3);
}

#[test]
fn sequence_classification_checkpoints_are_never_embedding() {
    // A reranker keeps its generator routing even with a pooling layout.
    let detected = detect_layout(
        "qwen3_reranker",
        &json!({
            "model_type": "qwen3",
            "architectures": ["Qwen3ForSequenceClassification"],
            "hidden_size": 8
        }),
        &[("modules.json", POOLING_MODULES)],
    )
    .unwrap();
    assert_eq!(detected, ModelType::Qwen3);
}

#[test]
fn cross_encoder_checkpoints_detect_as_the_reranker_family() {
    // #1356: a one-label `ForSequenceClassification` export on one of the
    // three encoder families is a reranker, so `-m <checkpoint>` can serve
    // `/v1/rerank` without `--reranker-model`. A pooling layout does not turn
    // it back into an embedder.
    for (model_type, architecture) in [
        ("bert", "BertForSequenceClassification"),
        ("xlm-roberta", "XLMRobertaForSequenceClassification"),
        ("xlm_roberta", "XLMRobertaForSequenceClassification"),
        ("modernbert", "ModernBertForSequenceClassification"),
    ] {
        let detected = detect_layout(
            &format!("cross_encoder_{model_type}"),
            &json!({
                "model_type": model_type,
                "architectures": [architecture],
                "hidden_size": 8,
            }),
            &[("modules.json", POOLING_MODULES)],
        )
        .unwrap_or_else(|err| panic!("{model_type}: {err}"));
        assert_eq!(detected, ModelType::SequenceClassifier, "{model_type}");
    }
}

#[test]
fn a_classifier_on_an_unported_family_keeps_its_generator_routing() {
    // Only the three encoder families have a cross-encoder head port. A
    // classifier on anything else must not be claimed by the reranker family;
    // it stays on the existing dispatch, which is what reports it.
    let err = detect_layout(
        "deberta_classifier",
        &json!({
            "model_type": "deberta-v2",
            "architectures": ["DebertaV2ForSequenceClassification"],
            "hidden_size": 8,
        }),
        &[],
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("Unsupported model type"), "{err}");
    assert!(err.contains("deberta-v2"), "{err}");
}

#[test]
fn modules_json_with_only_logit_score_does_not_trigger_embedding() {
    let detected = detect_layout(
        "qwen3_vl_reranker",
        &json!({"model_type": "qwen3_vl", "architectures": ["Qwen3VLForConditionalGeneration"]}),
        &[("modules.json", LOGIT_SCORE_MODULES)],
    )
    .unwrap();
    assert_eq!(detected, ModelType::Qwen3VL);
}

#[test]
fn embedding_layout_with_unknown_family_is_reported_not_misrouted() {
    let err = detect_layout(
        "mpnet_pooling",
        &json!({"model_type": "mpnet", "architectures": ["MPNetModel"], "hidden_size": 8}),
        &[(
            "1_Pooling/config.json",
            r#"{"pooling_mode_mean_tokens": true}"#,
        )],
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("embedding checkpoint"), "{err}");
    assert!(err.contains("mpnet"), "{err}");
}
