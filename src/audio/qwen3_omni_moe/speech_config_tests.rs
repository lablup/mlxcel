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

//! Unit tests for the Qwen3-Omni speech-output configuration parser.

use super::speech_config::*;

fn root_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "qwen3_omni_moe",
        "enable_audio_output": true,
        "tts_bos_token_id": 151672,
        "tts_eos_token_id": 151673,
        "tts_pad_token_id": 151671,
        "im_start_token_id": 151644,
        "system_token_id": 8948,
        "user_token_id": 872,
        "assistant_token_id": 77091,
        "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
        "thinker_config": {"image_token_id": 151655, "video_token_id": 151656,
                            "audio_token_id": 151675},
        "talker_config": {
            "accept_hidden_layer": 24,
            "num_code_groups": 16,
            "thinker_hidden_size": 2048,
            "codec_bos_id": 2149,
            "codec_eos_token_id": 2150,
            "codec_nothink_id": 2155,
            "codec_pad_id": 2148,
            "codec_think_bos_id": 2156,
            "codec_think_eos_id": 2157,
            "speaker_id": {"chelsie": 2301, "ethan": 2302, "aiden": 2303},
            "text_config": {
                "hidden_size": 1024,
                "intermediate_size": 2048,
                "num_hidden_layers": 20,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "head_dim": 128,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000,
                "vocab_size": 3072,
                "num_experts": 128,
                "num_experts_per_tok": 6,
                "moe_intermediate_size": 384,
                "shared_expert_intermediate_size": 768,
                "norm_topk_prob": true,
                "sliding_window": null
            },
            "code_predictor_config": {
                "num_hidden_layers": 5,
                "hidden_size": 1024,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 2048,
                "num_code_groups": 16,
                "rope_scaling": null,
                "architectures": null
            }
        },
        "code2wav_config": {
            "hidden_size": 1024,
            "num_hidden_layers": 8,
            "decoder_dim": 1536,
            "codebook_size": 2048,
            "num_quantizers": 16,
            "upsample_rates": [8, 5, 4, 3],
            "upsampling_ratios": [2, 2],
            "sliding_window": 72
        }
    })
}

#[test]
fn parses_nested_talker_and_code2wav_configs_from_root() {
    let cfg = Qwen3OmniSpeechConfig::parse(&root_config(), None).unwrap();
    assert_eq!(cfg.talker.text_config.num_hidden_layers, 20);
    assert_eq!(cfg.talker.text_config.num_key_value_heads, 2);
    assert_eq!(cfg.talker.text_config.num_experts, 128);
    assert_eq!(cfg.talker.code_predictor_config.num_hidden_layers, 5);
    assert_eq!(cfg.talker.code_predictor_config.num_code_groups, 16);
    assert_eq!(cfg.talker.accept_hidden_layer, 24);
    assert_eq!(cfg.talker.codec_eos_token_id, 2150);
    assert_eq!(cfg.talker.speaker_id.get("ethan"), Some(&2302));
    assert_eq!(cfg.code2wav.decoder_dim, 1536);
    assert_eq!(cfg.code2wav.samples_per_frame(), 1920);
    assert_eq!(cfg.group_size, 64);
    assert_eq!(cfg.bits, 4);
    assert!(cfg.enable_audio_output);
}

#[test]
fn tts_and_role_token_ids_default_when_absent() {
    let mut root = root_config();
    let map = root.as_object_mut().unwrap();
    for key in [
        "tts_bos_token_id",
        "tts_eos_token_id",
        "tts_pad_token_id",
        "im_start_token_id",
        "system_token_id",
        "user_token_id",
        "assistant_token_id",
    ] {
        map.remove(key);
    }
    let cfg = Qwen3OmniSpeechConfig::parse(&root, None).unwrap();
    assert_eq!(cfg.ids.tts_bos_token_id, 151_672);
    assert_eq!(cfg.ids.tts_eos_token_id, 151_673);
    assert_eq!(cfg.ids.tts_pad_token_id, 151_671);
    assert_eq!(cfg.ids.im_start_token_id, 151_644);
    assert_eq!(cfg.ids.assistant_token_id, 77_091);
    assert_eq!(
        cfg.ids.multimodal_token_ids,
        vec![151_655, 151_656, 151_675]
    );
}

#[test]
fn talker_sampling_reads_generation_config_fields() {
    let g = serde_json::json!({
        "talker_max_new_tokens": 2048,
        "talker_temperature": 0.7,
        "talker_top_p": 0.95,
        "talker_top_k": 50,
        "talker_repetition_penalty": 1.05
    });
    let cfg = Qwen3OmniSpeechConfig::parse(&root_config(), Some(&g)).unwrap();
    assert_eq!(cfg.sampling.max_frames, 2048);
    assert!((cfg.sampling.temperature - 0.7).abs() < 1e-6);
    assert!((cfg.sampling.top_p - 0.95).abs() < 1e-6);
    // Defaults when generation_config.json is absent.
    let cfg = Qwen3OmniSpeechConfig::parse(&root_config(), None).unwrap();
    assert_eq!(cfg.sampling.max_frames, 4096);
    assert!((cfg.sampling.temperature - 0.9).abs() < 1e-6);
    assert!((cfg.sampling.top_p - 1.0).abs() < 1e-6);
}

#[test]
fn rejects_mismatched_talker_and_code2wav_group_counts() {
    let mut root = root_config();
    root["code2wav_config"]["num_quantizers"] = serde_json::json!(15);

    let err = Qwen3OmniSpeechConfig::parse(&root, None).unwrap_err();
    assert!(err.contains("code2wav_config.num_quantizers (15)"));
    assert!(err.contains("talker_config.num_code_groups (16)"));
}

#[test]
fn rejects_invalid_sampling_limits_from_generation_config() {
    let generation = serde_json::json!({
        "talker_max_new_tokens": 0,
        "talker_temperature": -0.1
    });

    let err = Qwen3OmniSpeechConfig::parse(&root_config(), Some(&generation)).unwrap_err();
    assert!(err.contains("talker_max_new_tokens must be nonzero"));
}
