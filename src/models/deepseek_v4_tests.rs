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

//! Unit tests for the DeepSeek-V4 backbone config port.
//!
//! These pin the V4 config defaults against the Python reference
//! (`references/mlx-vlm/mlx_vlm/models/deepseek_v4/config.py`) and verify that
//! [`DeepSeekV4Config::to_dsv3_config`] threads every backbone field onto the
//! DeepSeek-V3 config. HiSA and the shared-expert `swiglu_limit` clamp are out
//! of scope for this backbone (tracked as follow-ups); the deferred-field test
//! confirms they are parsed but not applied to the V3 backbone.

use super::DeepSeekV4Config;

/// Minimal `deepseek_v4` config: only the required fields, everything else
/// falls back to the serde defaults (which track the reference `config.py`).
fn minimal_v4_config() -> DeepSeekV4Config {
    let json = r#"{
        "model_type": "deepseek_v4",
        "vocab_size": 129280,
        "hidden_size": 4096,
        "intermediate_size": 18432,
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 1
    }"#;
    serde_json::from_str(json).expect("parse minimal deepseek_v4 config")
}

#[test]
fn deepseek_v4_config_defaults_match_reference() {
    // Values pinned against references/mlx-vlm/.../deepseek_v4/config.py.
    let cfg = minimal_v4_config();
    assert_eq!(cfg.model_type, "deepseek_v4");
    assert_eq!(cfg.moe_intermediate_size, 2048);
    assert_eq!(cfg.routed_scaling_factor, 1.5);
    assert_eq!(cfg.q_lora_rank, 1024);
    assert_eq!(cfg.qk_rope_head_dim, 64);
    assert_eq!(cfg.num_experts_per_tok, 6);
    assert_eq!(cfg.max_position_embeddings, 1048576);
    assert_eq!(cfg.rms_norm_eps, 1e-6);
    assert_eq!(cfg.rope_theta, 10000.0);
    assert_eq!(cfg.scoring_func, "sqrtsoftplus");
    assert_eq!(cfg.topk_method, "noaux_tc");
    assert!(cfg.norm_topk_prob);
    assert!(!cfg.tie_word_embeddings);
    assert!(!cfg.attention_bias);
    // Group-limited routing defaults to ungrouped (matches DeepSeek-V3).
    assert_eq!(cfg.n_group, 1);
    assert_eq!(cfg.topk_group, 1);
    // Deferred V4-only fields (HiSA attention + shared-expert clamp).
    assert_eq!(cfg.head_dim, 512);
    assert_eq!(cfg.swiglu_limit, 10.0);
    assert_eq!(cfg.num_nextn_predict_layers, 1);
    // MLA-latent dims are absent from the V4 reference config; the backbone
    // reuses V3 MLA, so they fall back to the V3 defaults.
    assert_eq!(cfg.kv_lora_rank, 512);
    assert_eq!(cfg.qk_nope_head_dim, 128);
    assert_eq!(cfg.v_head_dim, 128);
}

#[test]
fn deepseek_v4_config_maps_to_v3_backbone() {
    // A representative deepseek_v4 config maps field-for-field onto the V3
    // backbone config the wrapper drives. This is the core scope contract:
    // "reuse the V3/V32 MLA + group-limited MoE + unclamped shared-expert MLP".
    let json = r#"{
        "model_type": "deepseek_v4",
        "vocab_size": 129280,
        "hidden_size": 4096,
        "intermediate_size": 18432,
        "moe_intermediate_size": 2048,
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "n_shared_experts": 1,
        "n_routed_experts": 256,
        "routed_scaling_factor": 1.5,
        "q_lora_rank": 1024,
        "qk_rope_head_dim": 64,
        "num_experts_per_tok": 6,
        "norm_topk_prob": true,
        "max_position_embeddings": 1048576,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "scoring_func": "sqrtsoftplus"
    }"#;
    let cfg: DeepSeekV4Config = serde_json::from_str(json).expect("parse deepseek_v4 config");
    let v3 = cfg.to_dsv3_config();

    assert_eq!(v3.model_type, "deepseek_v4");
    assert_eq!(v3.vocab_size, 129280);
    assert_eq!(v3.hidden_size, 4096);
    assert_eq!(v3.intermediate_size, 18432);
    assert_eq!(v3.moe_intermediate_size, 2048);
    assert_eq!(v3.num_hidden_layers, 43);
    assert_eq!(v3.num_attention_heads, 64);
    assert_eq!(v3.num_key_value_heads, 1);
    assert_eq!(v3.n_shared_experts, Some(1));
    assert_eq!(v3.n_routed_experts, Some(256));
    assert_eq!(v3.routed_scaling_factor, 1.5);
    assert_eq!(v3.q_lora_rank, 1024);
    assert_eq!(v3.qk_rope_head_dim, 64);
    assert_eq!(v3.num_experts_per_tok, 6);
    assert_eq!(v3.max_position_embeddings, 1048576);
    assert_eq!(v3.rope_theta, 10000.0);
    assert_eq!(v3.scoring_func, "sqrtsoftplus");
    assert!(v3.norm_topk_prob);
    // MLA-latent dims fall through the V4 defaults into the V3 backbone.
    assert_eq!(v3.kv_lora_rank, 512);
    assert_eq!(v3.qk_nope_head_dim, 128);
    assert_eq!(v3.v_head_dim, 128);
    // q_head_dim = qk_nope_head_dim + qk_rope_head_dim.
    assert_eq!(v3.q_head_dim(), 192);
    // Ungrouped by default; no dense-layer prefix.
    assert_eq!(v3.n_group, 1);
    assert_eq!(v3.topk_group, 1);
    assert_eq!(v3.first_k_dense_replace, 0);
}

#[test]
fn deepseek_v4_backbone_reuses_group_limited_moe() {
    // Explicit group-limited routing threads through, and the resulting V3
    // config reports MoE layers with V3 semantics (n_routed_experts set,
    // beyond first_k_dense_replace, on the moe_layer_freq cadence).
    let json = r#"{
        "model_type": "deepseek_v4",
        "vocab_size": 129280,
        "hidden_size": 4096,
        "intermediate_size": 18432,
        "num_hidden_layers": 8,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
        "n_group": 8,
        "topk_group": 4,
        "first_k_dense_replace": 3
    }"#;
    let cfg: DeepSeekV4Config = serde_json::from_str(json).expect("parse deepseek_v4 config");
    let v3 = cfg.to_dsv3_config();

    assert_eq!(v3.n_group, 8);
    assert_eq!(v3.topk_group, 4);
    assert_eq!(v3.first_k_dense_replace, 3);
    // Layers 0..2 are dense (below first_k_dense_replace); 3.. are MoE.
    assert!(!v3.is_moe_layer(0));
    assert!(!v3.is_moe_layer(2));
    assert!(v3.is_moe_layer(3));
    assert!(v3.is_moe_layer(7));
}

#[test]
fn deepseek_v4_defers_hisa_and_swiglu_limit() {
    // The V4-only features are parsed (so a genuine deepseek_v4 config loads and
    // the follow-up issues have the values) but never applied to the backbone:
    // the V3 config the wrapper drives has no HiSA/swiglu_limit surface, so the
    // shared-expert MLP stays unclamped and attention stays dense MLA.
    let json = r#"{
        "model_type": "deepseek_v4",
        "vocab_size": 129280,
        "hidden_size": 4096,
        "intermediate_size": 18432,
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "swiglu_limit": 7.5,
        "index_topk": 512,
        "num_nextn_predict_layers": 1
    }"#;
    let cfg: DeepSeekV4Config = serde_json::from_str(json).expect("parse deepseek_v4 config");
    assert_eq!(cfg.head_dim, 512);
    assert_eq!(cfg.swiglu_limit, 7.5);
    assert_eq!(cfg.num_nextn_predict_layers, 1);

    // Conversion succeeds and simply drops the deferred V4-only fields; the V3
    // backbone config exposes no clamp/HiSA knobs to carry them.
    let v3 = cfg.to_dsv3_config();
    assert_eq!(v3.num_hidden_layers, 43);
    assert_eq!(v3.moe_intermediate_size, cfg.moe_intermediate_size);
}

#[test]
fn deepseek_v4_quantization_and_rope_scaling_thread_through() {
    let json = r#"{
        "model_type": "deepseek_v4",
        "vocab_size": 129280,
        "hidden_size": 4096,
        "intermediate_size": 18432,
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "quantization": { "group_size": 64, "bits": 4 },
        "rope_scaling": {
            "type": "yarn",
            "factor": 40.0,
            "original_max_position_embeddings": 4096
        }
    }"#;
    let cfg: DeepSeekV4Config = serde_json::from_str(json).expect("parse deepseek_v4 config");
    let v3 = cfg.to_dsv3_config();

    assert_eq!(v3.group_size(), 64);
    assert_eq!(v3.bits(), 4);
    let q = v3
        .quantization
        .as_ref()
        .expect("quantization threaded through");
    assert_eq!(q.group_size, 64);
    assert_eq!(q.bits, 4);

    let rope = v3
        .rope_scaling
        .as_ref()
        .expect("rope_scaling threaded through");
    assert_eq!(
        rope.get("type").and_then(|v| v.as_str()),
        Some("yarn"),
        "rope_scaling type must survive conversion"
    );
    assert_eq!(
        rope.get("factor").and_then(|v| v.as_f64()),
        Some(40.0),
        "rope_scaling factor must survive conversion"
    );
}
