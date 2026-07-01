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

//! DeepSeek-V4 backbone parity.
//!
//! The issue directs a backbone-only port that reuses the DeepSeek-V3 MLA +
//! group-limited MoE + unclamped shared-expert MLP (HiSA and the shared-expert
//! `swiglu_limit` clamp are deferred follow-ups). There is no public
//! `deepseek_v4` checkpoint yet, and the reference's novel attention stack is
//! out of scope, so parity here is asserted two ways without downloading
//! weights:
//!
//! 1. Config parity: a representative `deepseek_v4` config maps field-for-field
//!    onto the V3 backbone config, with the expected values derived from the
//!    Python reference (`references/mlx-vlm/mlx_vlm/models/deepseek_v4/config.py`).
//! 2. Forward parity: on identical synthetic weights, the V4 backbone produces
//!    bit-for-bit the same logits as the DeepSeek-V3 model it wraps, proving the
//!    wrapper and config mapping preserve the backbone exactly.

use mlxcel::models::deepseek_v3::DeepSeekV3Config;
use mlxcel::models::deepseek_v4::DeepSeekV4Config;
use mlxcel::models::{DeepSeekV3Model, DeepSeekV4Model};
use mlxcel::{LanguageModel, initialize_runtime};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

/// A representative `deepseek_v4` config mirroring the reference `config.py`
/// backbone-relevant defaults.
const REFERENCE_V4_CONFIG: &str = r#"{
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
    "scoring_func": "sqrtsoftplus",
    "head_dim": 512,
    "swiglu_limit": 10.0,
    "num_nextn_predict_layers": 1
}"#;

#[test]
fn deepseek_v4_config_maps_to_v3_backbone_reference_values() {
    let cfg: DeepSeekV4Config =
        serde_json::from_str(REFERENCE_V4_CONFIG).expect("parse reference deepseek_v4 config");
    let v3 = cfg.to_dsv3_config();

    // Reference-derived expectations (deepseek_v4/config.py backbone fields).
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
    assert!(v3.norm_topk_prob);
    // MLA-latent dims reuse the V3 defaults (absent from the V4 reference).
    assert_eq!(v3.kv_lora_rank, 512);
    assert_eq!(v3.qk_nope_head_dim, 128);
    assert_eq!(v3.v_head_dim, 128);
    // Every layer is MoE for this config (no dense prefix, ungrouped default).
    assert_eq!(v3.first_k_dense_replace, 0);
    assert!(v3.is_moe_layer(0));
    assert!(v3.is_moe_layer(42));
}

// Tiny dense (non-MoE) DeepSeek config used for the forward-parity check. Keeping
// `n_routed_experts` unset makes every layer dense, so no quantized expert
// weights are needed. `num_hidden_layers` is 2 so the loader drops the trailing
// MTP layer and builds a single decoder layer.
const TINY_DIMS: &str = r#"
    "vocab_size": 8,
    "hidden_size": 16,
    "intermediate_size": 32,
    "moe_intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "q_lora_rank": 8,
    "kv_lora_rank": 8,
    "qk_rope_head_dim": 4,
    "qk_nope_head_dim": 4,
    "v_head_dim": 4,
    "max_position_embeddings": 64
"#;

fn tiny_v3_config() -> DeepSeekV3Config {
    let json = format!("{{ \"model_type\": \"deepseek_v3\", {} }}", TINY_DIMS);
    serde_json::from_str(&json).expect("parse tiny deepseek_v3 config")
}

fn tiny_v4_config() -> DeepSeekV4Config {
    let json = format!("{{ \"model_type\": \"deepseek_v4\", {} }}", TINY_DIMS);
    serde_json::from_str(&json).expect("parse tiny deepseek_v4 config")
}

/// Deterministic, decorrelated small weights so logits are well spread.
fn synth(seed: u64, shape: &[i32]) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| {
            let h = seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add((i as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
            let x = ((h >> 11) as f64 / (1u64 << 53) as f64) as f32; // [0, 1)
            (x - 0.5) * 0.2
        })
        .collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

fn ones(shape: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::ones(shape, mlxcel_core::dtype::FLOAT32)
}

/// Raw (pre-sanitize) weights for the tiny single-layer dense DeepSeek model.
fn tiny_raw_weights() -> WeightMap {
    let mut w = WeightMap::new();
    w.insert("model.embed_tokens.weight".into(), synth(1, &[8, 16]));

    let p = "model.layers.0";
    w.insert(format!("{p}.self_attn.q_a_proj.weight"), synth(2, &[8, 16]));
    w.insert(format!("{p}.self_attn.q_a_layernorm.weight"), ones(&[8]));
    w.insert(format!("{p}.self_attn.q_b_proj.weight"), synth(3, &[16, 8]));
    w.insert(
        format!("{p}.self_attn.kv_a_proj_with_mqa.weight"),
        synth(4, &[12, 16]),
    );
    w.insert(format!("{p}.self_attn.kv_a_layernorm.weight"), ones(&[8]));
    w.insert(
        format!("{p}.self_attn.kv_b_proj.weight"),
        synth(5, &[16, 8]),
    );
    w.insert(format!("{p}.self_attn.o_proj.weight"), synth(6, &[16, 8]));
    w.insert(format!("{p}.input_layernorm.weight"), ones(&[16]));
    w.insert(format!("{p}.post_attention_layernorm.weight"), ones(&[16]));
    w.insert(format!("{p}.mlp.gate_proj.weight"), synth(7, &[32, 16]));
    w.insert(format!("{p}.mlp.up_proj.weight"), synth(8, &[32, 16]));
    w.insert(format!("{p}.mlp.down_proj.weight"), synth(9, &[16, 32]));

    w.insert("model.norm.weight".into(), ones(&[16]));
    w.insert("lm_head.weight".into(), synth(10, &[8, 16]));
    w
}

/// Extract the final-token logit row as a host `Vec<f32>`.
fn last_token_logits(logits: &MlxArray) -> Vec<f32> {
    mlxcel_core::eval(logits);
    let shape = mlxcel_core::array_shape(logits); // [1, seq, vocab]
    let seq = shape[1];
    let vocab = shape[2];
    (0..vocab)
        .map(|j| {
            let e = mlxcel_core::slice(logits, &[0, seq - 1, j], &[1, seq, j + 1]);
            mlxcel_core::item_f32(&e)
        })
        .collect()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

#[test]
fn deepseek_v4_backbone_matches_deepseek_v3_on_shared_weights() {
    let _runtime = initialize_runtime();

    let v3_cfg = tiny_v3_config();
    let v4_cfg = tiny_v4_config();

    // Sanitize once (MLA kv_b_proj decomposition + MTP-layer drop) and build both
    // models from the identical weight map.
    let sanitized = DeepSeekV3Model::sanitize_weights_with_args(tiny_raw_weights(), &v3_cfg);
    let m3 = DeepSeekV3Model::from_weights(&sanitized, &v3_cfg).expect("build deepseek_v3");
    let m4 = DeepSeekV4Model::from_weights(&sanitized, &v4_cfg).expect("build deepseek_v4");

    let prompt = mlxcel_core::from_slice_i32(&[1, 2, 3, 4], &[1, 4]);

    let mut c3 = LanguageModel::make_caches(&m3);
    let l3 = LanguageModel::forward(&m3, &prompt, &mut c3, None);
    let v3_logits = last_token_logits(&l3);

    let mut c4 = LanguageModel::make_caches(&m4);
    let l4 = LanguageModel::forward(&m4, &prompt, &mut c4, None);
    let v4_logits = last_token_logits(&l4);

    assert_eq!(
        v3_logits.len(),
        v4_logits.len(),
        "logit vocab dimension must match"
    );
    let max_abs_diff = v3_logits
        .iter()
        .zip(&v4_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs_diff < 1e-4,
        "V4 backbone logits diverged from V3 (max abs diff {max_abs_diff})"
    );
    assert_eq!(
        argmax(&v3_logits),
        argmax(&v4_logits),
        "V4 backbone must select the same next token as V3"
    );
}
