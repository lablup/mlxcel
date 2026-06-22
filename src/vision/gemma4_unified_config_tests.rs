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

//! Unit tests for the Gemma 4 Unified config schema.

use super::*;
use crate::models::gemma4::ModelArgs;

/// A representative `gemma4_unified` config matching the reference checkpoint
/// (`mlx-community/gemma-4-12b-it-4bit`) field shapes, trimmed for the test.
fn reference_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "gemma4_unified",
        "architectures": ["Gemma4UnifiedForConditionalGeneration"],
        "image_token_id": 258880,
        "audio_token_id": 258881,
        "video_token_id": 258884,
        "boi_token_id": 255999,
        "eoi_token_id": 258882,
        "boa_token_id": 256000,
        "eoa_token_index": 258883,
        "tie_word_embeddings": true,
        "quantization": {
            "group_size": 64,
            "bits": 4,
            "mode": "affine",
            "language_model.model.layers.0.mlp.gate_proj": { "group_size": 64, "bits": 8 },
            "language_model.model.layers.0.mlp.down_proj": { "group_size": 64, "bits": 8 }
        },
        "text_config": {
            "model_type": "gemma4_unified_text",
            "hidden_size": 3840,
            "num_hidden_layers": 48,
            "intermediate_size": 15360,
            "num_attention_heads": 16,
            "head_dim": 256,
            "global_head_dim": 512,
            "num_key_value_heads": 8,
            "num_global_key_value_heads": 1,
            "sliding_window": 1024,
            "sliding_window_pattern": 6,
            "attention_k_eq_v": true,
            "use_double_wide_mlp": false,
            "use_bidirectional_attention": "vision",
            "vocab_size": 262144,
            "vocab_size_per_layer_input": 262144,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 131072,
            "final_logit_softcapping": 30.0,
            "layer_types": [
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention"
            ],
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional"
                },
                "sliding_attention": { "rope_theta": 10000.0, "rope_type": "default" }
            }
        },
        "vision_config": {
            "model_type": "gemma4_unified_vision",
            "patch_size": 16,
            "pooling_kernel_size": 3,
            "model_patch_size": 48,
            "mm_embed_dim": 3840,
            "mm_posemb_size": 1120,
            "num_soft_tokens": 280,
            "output_proj_dims": 3840,
            "rms_norm_eps": 1e-6
        },
        "audio_config": {
            "model_type": "gemma4_unified_audio",
            "audio_samples_per_token": 640,
            "audio_embed_dim": 640,
            "hidden_size": 640,
            "output_proj_dims": 640,
            "rms_norm_eps": 1e-6
        }
    })
}

#[test]
fn parses_three_subconfigs() {
    let cfg: Gemma4UnifiedConfig = serde_json::from_value(reference_config()).unwrap();
    assert_eq!(cfg.model_type, "gemma4_unified");

    // Vision sub-config.
    assert_eq!(cfg.vision_config.model_patch_size, 48);
    assert_eq!(cfg.vision_config.num_soft_tokens, 280);
    assert_eq!(cfg.vision_config.mm_embed_dim, 3840);
    assert_eq!(cfg.vision_config.mm_posemb_size, 1120);
    assert_eq!(cfg.vision_config.output_proj_dims, 3840);

    // Audio sub-config.
    let audio = cfg.audio_config.as_ref().expect("audio_config present");
    assert_eq!(audio.audio_samples_per_token, 640);
    assert_eq!(audio.output_proj_dims, 640);

    // Text sub-config parses through the shared Gemma 4 text path and exposes
    // the new bidirectional flag.
    let text: ModelArgs = serde_json::from_value(reference_config()).unwrap();
    let text = text.text_args();
    assert_eq!(text.hidden_size, 3840);
    assert_eq!(text.num_hidden_layers, 48);
    assert!(text.uses_bidirectional_vision_attention());
}

#[test]
fn eoa_token_index_fallback() {
    // Reference: eoa_token_id absent, eoa_token_index present.
    let cfg: Gemma4UnifiedConfig = serde_json::from_value(reference_config()).unwrap();
    assert_eq!(cfg.eoa_token_id, None);
    assert_eq!(cfg.eoa_token_index, Some(258883));
    assert_eq!(cfg.resolve_eoa_token_id(), 258883);

    // Explicit eoa_token_id wins over the index.
    let mut value = reference_config();
    value["eoa_token_id"] = serde_json::json!(999);
    let cfg2: Gemma4UnifiedConfig = serde_json::from_value(value).unwrap();
    assert_eq!(cfg2.resolve_eoa_token_id(), 999);

    // Neither present → documented default.
    let mut value = reference_config();
    value.as_object_mut().unwrap().remove("eoa_token_index");
    let cfg3: Gemma4UnifiedConfig = serde_json::from_value(value).unwrap();
    assert_eq!(cfg3.resolve_eoa_token_id(), 258883);
}

#[test]
fn vision_soft_tokens_per_video_frame_defaults_to_70() {
    // The reference checkpoint omits the field; it must default to 70
    // (issue #164). An explicit value overrides the default.
    let cfg: Gemma4UnifiedConfig = serde_json::from_value(reference_config()).unwrap();
    assert_eq!(cfg.vision_soft_tokens_per_video_frame, 70);

    let mut value = reference_config();
    value["vision_soft_tokens_per_video_frame"] = serde_json::json!(140);
    let cfg2: Gemma4UnifiedConfig = serde_json::from_value(value).unwrap();
    assert_eq!(cfg2.vision_soft_tokens_per_video_frame, 140);
}

#[test]
fn token_ids_match_reference() {
    let cfg: Gemma4UnifiedConfig = serde_json::from_value(reference_config()).unwrap();
    assert_eq!(cfg.image_token_id, 258880);
    assert_eq!(cfg.audio_token_id, 258881);
    assert_eq!(cfg.video_token_id, 258884);
    assert_eq!(cfg.boi_token_id, 255999);
    assert_eq!(cfg.eoi_token_id, 258882);
    assert_eq!(cfg.boa_token_id, 256000);
    assert_eq!(cfg.tie_word_embeddings, Some(true));
}

#[test]
fn mixed_precision_quant_inherited_by_text_config() {
    // The root quantization map's 4-bit default must flow into the text
    // config (the per-tensor 8-bit overrides are resolved later by tensor
    // shape inference at load time, not by config parsing).
    let text: ModelArgs = serde_json::from_value(reference_config()).unwrap();
    let text = text.text_args();
    let quant = text.quantization.expect("quant inherited from root");
    assert_eq!(quant.group_size, 64);
    assert_eq!(quant.bits, 4);
}

#[test]
fn audio_config_optional() {
    // A text+vision-only checkpoint (no audio_config) parses with audio None.
    let mut value = reference_config();
    value.as_object_mut().unwrap().remove("audio_config");
    let cfg: Gemma4UnifiedConfig = serde_json::from_value(value).unwrap();
    assert!(cfg.audio_config.is_none());
}

#[test]
fn placeholder_tokens_cover_all_seven_markers() {
    // issue #350: the config's placeholder ids feed the output-suppression set.
    let cfg: Gemma4UnifiedConfig = serde_json::from_value(reference_config()).unwrap();
    let suppressed = cfg.placeholder_tokens().suppressed_ids();
    // audio 258881, image 258880, video 258884, boa 256000, boi 255999,
    // eoa 258883 (via eoa_token_index fallback), eoi 258882, returned sorted.
    assert_eq!(
        suppressed,
        vec![
            255_999, 256_000, 258_880, 258_881, 258_882, 258_883, 258_884
        ]
    );
}

#[test]
fn output_suppression_masks_placeholders_but_not_eos() {
    // The end-to-end shape used at generation time: derive the suppressed set
    // from the gemma4_unified config and force it into a TokenBiasMap. Every
    // placeholder id must become -inf; the real EOS ids ([1, 106, 50] in the
    // gemma-4-12b checkpoint) must stay untouched so end-of-sequence detection
    // and normal text generation are unaffected (issue #350 acceptance).
    let mut value = reference_config();
    value["eos_token_id"] = serde_json::json!([1, 106, 50]);
    let cfg: Gemma4UnifiedConfig = serde_json::from_value(value).unwrap();

    let mut bias = mlxcel_core::TokenBiasMap::default();
    assert!(bias.is_empty(), "baseline starts empty (zero-cost path)");
    bias.suppress_tokens(&cfg.placeholder_tokens().suppressed_ids());

    for placeholder in [
        255_999, 256_000, 258_880, 258_881, 258_882, 258_883, 258_884,
    ] {
        let b = bias
            .get(&placeholder)
            .copied()
            .unwrap_or_else(|| panic!("placeholder {placeholder} must be in the bias map"));
        assert!(
            b.is_infinite() && b.is_sign_negative(),
            "placeholder {placeholder} must be masked to -inf, got {b}"
        );
    }
    for eos in [1, 106, 50] {
        assert!(
            bias.get(&eos).is_none(),
            "real EOS id {eos} must never be suppressed"
        );
    }
}

#[test]
fn vision_config_defaults_apply() {
    // Minimal vision_config falls back to documented defaults.
    let value = serde_json::json!({
        "model_type": "gemma4_unified",
        "text_config": reference_config()["text_config"].clone(),
        "vision_config": { "model_type": "gemma4_unified_vision" }
    });
    let cfg: Gemma4UnifiedConfig = serde_json::from_value(value).unwrap();
    assert_eq!(cfg.vision_config.model_patch_size, 48);
    assert_eq!(cfg.vision_config.num_soft_tokens, 280);
    assert_eq!(cfg.vision_config.mm_embed_dim, 3840);
    // Token ids fall back to documented defaults when absent.
    assert_eq!(cfg.image_token_id, 258880);
    assert_eq!(cfg.resolve_eoa_token_id(), 258883);
}
