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

//! Unit tests for the VoiceChat config, the duplex LLM glue and the
//! load-time consistency checks (random weights, no checkpoint).

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::VoiceChatConfig;
use super::llm::{VoiceChatLanguageModel, take_llm_weights};
use super::model::check_sub_configs;
use super::tts::TtsConfig;
use crate::audio::nemotron_codec::CodecConfig;
use crate::models::nemotron_h::NemotronHConfig;

const VOCAB: i32 = 16;
const HIDDEN: i32 = 8;

fn ramp(shape: &[i32], scale: f32) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| ((i * 7 % 13) as f32 - 6.0) * scale)
        .collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

fn tiny_text_config() -> NemotronHConfig {
    serde_json::from_value(serde_json::json!({
        "model_type": "nemotron_h",
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "mamba_num_heads": 1,
        "mamba_head_dim": 4,
        "ssm_state_size": 4,
        "conv_kernel": 4,
        "n_groups": 1,
        "hybrid_override_pattern": "*-",
    }))
    .expect("tiny config parses")
}

/// A tiny checkpoint in the VoiceChat `stt_model.*` layout: one attention
/// layer, one MLP layer, the shared embedding table and both heads.
fn tiny_voicechat_weights() -> WeightMap {
    let mut w = WeightMap::new();
    let mut put = |k: &str, v: UniquePtr<MlxArray>| {
        w.insert(k.to_string(), v);
    };
    put("stt_model.embed_tokens.weight", ramp(&[VOCAB, HIDDEN], 0.1));
    put("stt_model.lm_head.weight", ramp(&[VOCAB, HIDDEN], 0.05));
    put(
        "stt_model.function_head.weight",
        ramp(&[VOCAB, HIDDEN], -0.07),
    );
    put(
        "stt_model.llm.norm_f.weight",
        mlxcel_core::ones(&[HIDDEN], mlxcel_core::dtype::FLOAT32),
    );
    for layer in 0..2 {
        put(
            &format!("stt_model.llm.layers.{layer}.norm.weight"),
            mlxcel_core::ones(&[HIDDEN], mlxcel_core::dtype::FLOAT32),
        );
    }
    for (i, proj) in ["q_proj", "k_proj", "v_proj", "o_proj"].iter().enumerate() {
        put(
            &format!("stt_model.llm.layers.0.mixer.{proj}.weight"),
            ramp(&[HIDDEN, HIDDEN], 0.03 * (i as f32 + 1.0)),
        );
    }
    put(
        "stt_model.llm.layers.1.mixer.up_proj.weight",
        ramp(&[16, HIDDEN], 0.02),
    );
    put(
        "stt_model.llm.layers.1.mixer.down_proj.weight",
        ramp(&[HIDDEN, 16], 0.02),
    );
    put("stt_model.perception.proj.weight", ramp(&[HIDDEN, 4], 0.01));
    w
}

fn host_f32(a: &MlxArray) -> Vec<f32> {
    let a = mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&a);
    mlxcel_core::array_to_raw_bytes(&a)
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[test]
fn take_llm_weights_renames_onto_nemotron_h_layout() {
    let mut weights = tiny_voicechat_weights();
    let llm = take_llm_weights(&mut weights);
    assert!(llm.contains_key("backbone.embeddings.weight"));
    assert!(llm.contains_key("backbone.layers.0.mixer.q_proj.weight"));
    assert!(llm.contains_key("backbone.norm_f.weight"));
    assert!(llm.contains_key("lm_head.weight"));
    assert!(llm.contains_key("stt_model.function_head.weight"));
    // Non-LLM keys stay behind for the other loaders.
    assert!(weights.contains_key("stt_model.perception.proj.weight"));
    assert!(!weights.keys().any(|k| k.starts_with("stt_model.llm.")));
}

#[test]
fn forward_embeds_to_hidden_matches_forward_on_embedded_ids() {
    let mut weights = tiny_voicechat_weights();
    let lm = VoiceChatLanguageModel::from_weights(tiny_text_config(), None, &mut weights, 2.0)
        .expect("tiny VoiceChat LM loads");
    let backbone = lm.backbone();
    let ids = [3, 7, 1, 12];

    let mut caches = backbone.make_caches();
    let hidden = backbone.forward_embeds_to_hidden(&lm.embed(&ids), &mut caches);
    let via_embeds = host_f32(&backbone.apply_lm_head(&hidden));

    let id_arr = mlxcel_core::from_slice_i32(&ids, &[1, ids.len() as i32]);
    let mut caches = backbone.make_caches();
    let via_ids = host_f32(&backbone.forward_stage(&id_arr, 0..2, true, true, &mut caches));

    assert_eq!(via_embeds.len(), via_ids.len());
    let max = via_embeds
        .iter()
        .zip(&via_ids)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(max < 1e-5, "max abs diff {max}");
}

#[test]
fn fused_step_uses_both_heads_greedily() {
    let mut weights = tiny_voicechat_weights();
    let lm = VoiceChatLanguageModel::from_weights(tiny_text_config(), None, &mut weights, 2.0)
        .expect("tiny VoiceChat LM loads");
    let audio = ramp(&[1, 1, HIDDEN], 0.2);
    let fused = lm.fused_input(12, &audio, 12);
    let mut caches = lm.make_caches();
    let out = lm.step(&fused, &mut caches);
    assert!((0..VOCAB).contains(&out.text));
    assert!((0..VOCAB).contains(&out.function));
    // The function head has different weights, so over a few positions the
    // two channels must not be a copy of each other.
    let mut differs = out.text != out.function;
    for t in 0..4 {
        let fused = lm.fused_input(t, &audio, t + 1);
        let step = lm.step(&fused, &mut caches);
        differs |= step.text != step.function;
    }
    assert!(differs, "text and function heads produced identical ids");
}

#[test]
fn config_defaults_and_validation() {
    let config = VoiceChatConfig::from_json(r#"{"text_config": {}}"#).expect("defaults parse");
    assert_eq!(config.frame_samples(), 1280);
    assert_eq!(config.special_text_ids(), [12, 11, 1, 2]);
    assert_eq!(config.default_quantization(), None);

    let quantized = VoiceChatConfig::from_json(
        r#"{"text_config": {}, "quantization": {"group_size": 64, "bits": 8}}"#,
    )
    .expect("quantized parses");
    assert_eq!(quantized.default_quantization(), Some((64, 8)));

    let voice = VoiceChatConfig::from_json(r#"{"text_config": {}, "speaker": "Other"}"#);
    assert!(voice.is_err_and(|e| e.contains("Aria")));
    let rate = VoiceChatConfig::from_json(r#"{"text_config": {}, "input_sample_rate": 8000}"#);
    assert!(rate.is_err_and(|e| e.contains("16000")));
}

#[test]
fn sub_config_mismatch_is_an_error() {
    let text = tiny_text_config();
    let tts = TtsConfig::default();
    let codec = CodecConfig::default();
    assert!(check_sub_configs(&text, &tts, &codec).is_ok());
    let codec = CodecConfig {
        num_quantizers: 16,
        ..CodecConfig::default()
    };
    assert!(check_sub_configs(&text, &tts, &codec).is_err());
}
