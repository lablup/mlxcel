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

//! Checkpoint-free tests for the MiMo-V2 loader: flat-config building, the
//! fused-QKV unfuse at tensor-parallel degrees 1 and 4 (including the padded
//! 108-row grid), the MXFP4 expert remap, and detection/registration.

use super::{MimoV2Config, load_mimo_v2_vlm};
use crate::models::ModelType;
use crate::models::mimo_v2::{
    candidate_degrees, convert_mxfp4_experts, sanitize_text_weights, unfuse_qkv,
};
use crate::models::mimo_v2_flash::ModelArgs;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

/// E4M3 byte for 1.0, 2.0 and 4.0.
const E4M3_ONE: u32 = 0x38;
const E4M3_TWO: u32 = 0x40;
const E4M3_FOUR: u32 = 0x48;

const V26_CONFIG: &str = include_str!("../../tests/fixtures/mimo_v2_v26_config.json");

fn full_attention_args() -> ModelArgs {
    ModelArgs {
        model_type: "mimo_v2".to_string(),
        num_hidden_layers: 1,
        hybrid_layer_pattern: vec![0],
        moe_layer_freq: vec![0],
        num_attention_heads: 64,
        num_key_value_heads: 4,
        head_dim: 192,
        v_head_dim: 128,
        ..ModelArgs::default()
    }
}

fn uint8_array(data: &[u32], shape: &[i32]) -> UniquePtr<MlxArray> {
    let values = mlxcel_core::from_slice_u32(data, shape);
    mlxcel_core::astype(&values, dtype::UINT8)
}

fn read_all(array: &MlxArray) -> Vec<f32> {
    let flat = mlxcel_core::astype(array, dtype::FLOAT32);
    mlxcel_core::eval(&flat);
    let bytes = mlxcel_core::array_to_raw_bytes(&flat);
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn assert_constant(array: &MlxArray, expected: f32) {
    let values = read_all(array);
    assert!(!values.is_empty());
    for value in values {
        assert!(
            (value - expected).abs() < 1e-6,
            "expected every element to be {expected}, got {value}"
        );
    }
}

#[test]
fn flat_v26_config_builds_the_flash_decoder_and_parses_string_audio() {
    let config = MimoV2Config::from_json_str(V26_CONFIG).expect("the V2.6 config builds");
    assert_eq!(config.text.model_type, "mimo_v2");
    assert_eq!(config.text.hidden_size, 4096);
    assert_eq!(config.text.num_hidden_layers, 48);
    assert_eq!(config.text.num_attention_heads, 64);
    assert_eq!(config.text.num_key_value_heads, 4);
    assert_eq!(config.text.head_dim, 192);
    assert_eq!(config.text.v_head_dim, 128);
    assert_eq!(config.text.swa_num_key_value_heads, 8);
    assert_eq!(config.text.n_routed_experts, Some(256));
    assert_eq!(config.text.num_experts_per_tok, 8);
    assert_eq!(config.text.rope_theta, 10_000_000.0);
    assert_eq!(config.text.swa_rope_theta, 10_000.0);

    let vision = config.vision.expect("vision_config");
    // `in_chans` is the released spelling and must land in `in_channels`.
    assert_eq!(vision.in_channels, 3);
    assert_eq!(vision.depth, 28);
    assert_eq!(vision.qk_channels, 64);
    assert_eq!(vision.fullatt_block_indexes, vec![0, 9, 18, 27]);
    assert_eq!(vision.visual_token_window_size, 64);

    // The two string-typed audio scalars are coerced to integers.
    let audio = config.audio.expect("audio_config");
    assert_eq!(audio.speech_vocab_size, 1280);
    assert_eq!(audio.speech_zeroemb_idx, 1024);

    assert_eq!(config.tokens.image_token_id, 151_655);
    assert_eq!(config.tokens.video_token_id, 151_656);
    assert_eq!(config.tokens.audio_token_id, 151_669);
    assert_eq!(config.tokens.vision_start_token_id, 151_652);
    assert_eq!(config.tokens.vision_end_token_id, 151_653);
    assert!(!config.skip_vision);
}

#[test]
fn detection_and_registration_route_v26_to_the_vlm_family() {
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::write(dir.path().join("config.json"), V26_CONFIG).expect("write config");
    assert_eq!(
        crate::models::get_model_type(dir.path()).expect("detects"),
        ModelType::MiMoV2VLM
    );
    assert!(crate::model_metadata::is_vlm_model_type(
        ModelType::MiMoV2VLM
    ));
    assert!(crate::models::model_type_has_native_video(
        ModelType::MiMoV2VLM
    ));
    assert_eq!(ModelType::MiMoV2VLM.registry_id(), "mimo_v2_vlm");
    assert_eq!(ModelType::MiMoV2VLM.family(), "Xiaomi");
    assert!(
        crate::models::ALL_MODEL_TYPES.contains(&ModelType::MiMoV2VLM),
        "mlxcel arch must list the family"
    );
}

#[test]
fn skip_vision_loads_text_only() {
    let dir = tempfile::tempdir().expect("temp dir");
    let mut config: serde_json::Value = serde_json::from_str(V26_CONFIG).expect("json");
    config["skip_vision"] = serde_json::Value::Bool(true);
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::to_string(&config).expect("encode"),
    )
    .expect("write config");
    // Detection stays on the VLM family; the loader is what honors the flag.
    assert_eq!(
        crate::models::get_model_type(dir.path()).expect("detects"),
        ModelType::MiMoV2VLM
    );
    let parsed = MimoV2Config::from_json_str(V26_CONFIG).expect("parses");
    assert!(!parsed.skip_vision);
    let parsed = MimoV2Config::from_json_str(&serde_json::to_string(&config).unwrap())
        .expect("parses the skip_vision variant");
    assert!(parsed.skip_vision);
}

// ---------------------------------------------------------------------------
// Fused QKV unfuse
// ---------------------------------------------------------------------------

/// Build a fused `qkv_proj.weight` + `weight_scale_inv` pair for one full
/// attention layer of the published geometry, sharded at `degree`.
fn fused_checkpoint(degree: usize, grid_rows: i32) -> WeightMap {
    let hidden = 32i32;
    let (heads, kv_heads, head_dim, v_head_dim) = (64usize, 4usize, 192usize, 128usize);
    let q_rows = heads / degree * head_dim;
    let k_rows = kv_heads / degree * head_dim;
    let v_rows = kv_heads / degree * v_head_dim;
    let shard = q_rows + k_rows + v_rows;
    let total_rows = shard * degree;

    let mut weights = WeightMap::new();
    if grid_rows == 0 {
        // No scale grid means the fused tensor is already a dense projection,
        // which is the degree-1 path upstream takes (`sharded == false`).
        let mut values = Vec::with_capacity(total_rows * hidden as usize);
        for _ in 0..degree {
            values.extend(std::iter::repeat_n(1.0f32, q_rows * hidden as usize));
            values.extend(std::iter::repeat_n(2.0f32, k_rows * hidden as usize));
            values.extend(std::iter::repeat_n(4.0f32, v_rows * hidden as usize));
        }
        weights.insert(
            "model.layers.0.self_attn.qkv_proj.weight".into(),
            mlxcel_core::from_slice_f32(&values, &[total_rows as i32, hidden]),
        );
        return weights;
    }
    let mut values = Vec::with_capacity(total_rows * hidden as usize);
    for _ in 0..degree {
        values.extend(std::iter::repeat_n(E4M3_ONE, q_rows * hidden as usize));
        values.extend(std::iter::repeat_n(E4M3_TWO, k_rows * hidden as usize));
        values.extend(std::iter::repeat_n(E4M3_FOUR, v_rows * hidden as usize));
    }
    weights.insert(
        "model.layers.0.self_attn.qkv_proj.weight".into(),
        uint8_array(&values, &[total_rows as i32, hidden]),
    );
    if grid_rows > 0 {
        weights.insert(
            "model.layers.0.self_attn.qkv_proj.weight_scale_inv".into(),
            mlxcel_core::ones(&[grid_rows, 1], dtype::FLOAT32),
        );
    }
    weights
}

#[test]
fn candidate_degrees_sees_the_padded_degree_four_grid() {
    // A degree-4 full-attention shard is 3392 rows, which pads to 27 blocks;
    // four shards therefore ship a 108-row grid for a 106-block weight.
    assert_eq!(
        candidate_degrees(13_568, 64, 4, 192, 128, 108),
        vec![4],
        "only degree 4 explains the padded grid"
    );
    assert_eq!(
        candidate_degrees(13_568, 64, 4, 192, 128, 106),
        vec![1, 2],
        "a 106-row grid cannot distinguish degree 1 from 2; the released \
         checkpoints are sharded at 4 and the padded 108-row grid pins that, \
         while a scale-less fused tensor is the degree-1 path"
    );
}

#[test]
fn unfuse_degree_one_recovers_q_k_v() {
    let args = full_attention_args();
    let weights = fused_checkpoint(1, 0);
    let out = unfuse_qkv(weights, &args).expect("degree 1 unfuses");
    assert!(!out.contains_key("model.layers.0.self_attn.qkv_proj.weight"));
    let q = out
        .get("model.layers.0.self_attn.q_proj.weight")
        .expect("q_proj");
    let k = out
        .get("model.layers.0.self_attn.k_proj.weight")
        .expect("k_proj");
    let v = out
        .get("model.layers.0.self_attn.v_proj.weight")
        .expect("v_proj");
    assert_eq!(mlxcel_core::array_shape(q), [64 * 192, 32]);
    assert_eq!(mlxcel_core::array_shape(k), [4 * 192, 32]);
    assert_eq!(mlxcel_core::array_shape(v), [4 * 128, 32]);
    assert_constant(q, 1.0);
    assert_constant(k, 2.0);
    assert_constant(v, 4.0);
}

#[test]
fn unfuse_degree_four_with_padded_grid_recovers_q_k_v() {
    let args = full_attention_args();
    let weights = fused_checkpoint(4, 108);
    let out = unfuse_qkv(weights, &args).expect("degree 4 unfuses");
    assert!(!out.contains_key("model.layers.0.self_attn.qkv_proj.weight_scale_inv"));
    let q = out
        .get("model.layers.0.self_attn.q_proj.weight")
        .expect("q_proj");
    let k = out
        .get("model.layers.0.self_attn.k_proj.weight")
        .expect("k_proj");
    let v = out
        .get("model.layers.0.self_attn.v_proj.weight")
        .expect("v_proj");
    assert_eq!(mlxcel_core::array_shape(q), [64 * 192, 32]);
    assert_eq!(mlxcel_core::array_shape(k), [4 * 192, 32]);
    assert_eq!(mlxcel_core::array_shape(v), [4 * 128, 32]);
    assert_constant(q, 1.0);
    assert_constant(k, 2.0);
    assert_constant(v, 4.0);
}

#[test]
fn unfuse_rejects_a_grid_no_degree_explains() {
    let args = full_attention_args();
    let weights = fused_checkpoint(1, 107);
    let error = unfuse_qkv(weights, &args)
        .err()
        .expect("107 rows explain no degree");
    assert!(
        error.contains("no tensor-parallel degree"),
        "unhelpful error: {error}"
    );
}

// ---------------------------------------------------------------------------
// MXFP4 experts
// ---------------------------------------------------------------------------

#[test]
fn mxfp4_experts_are_repacked_to_uint32_weights_and_scales() {
    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.1.mlp.switch_mlp.gate_proj.weight".into(),
        uint8_array(&vec![E4M3_ONE; 4 * 8 * 16], &[4, 8, 16]),
    );
    weights.insert(
        "model.layers.1.mlp.switch_mlp.gate_proj.weight_scale".into(),
        uint8_array(&[0u32; 32], &[4, 8, 1]),
    );
    let out = convert_mxfp4_experts(weights);
    let weight = out
        .get("model.layers.1.mlp.switch_mlp.gate_proj.weight")
        .expect("packed weight");
    assert_eq!(mlxcel_core::array_dtype(weight), dtype::UINT32);
    assert_eq!(mlxcel_core::array_shape(weight), [4, 8, 4]);
    assert!(
        out.contains_key("model.layers.1.mlp.switch_mlp.gate_proj.scales"),
        "the weight_scale sidecar becomes .scales"
    );
    assert!(
        !out.contains_key("model.layers.1.mlp.switch_mlp.gate_proj.weight_scale"),
        "the sidecar key must not survive"
    );
}

#[test]
fn sanitize_strips_prefixes_audio_and_mtp_and_keeps_vision_separate() {
    let args = full_attention_args();
    let mut weights = WeightMap::new();
    weights.insert(
        "language_model.model.norm.weight".into(),
        mlxcel_core::ones(&[32], dtype::FLOAT32),
    );
    weights.insert(
        "audio_encoder.projection.mlp.0.weight".into(),
        mlxcel_core::ones(&[4, 4], dtype::FLOAT32),
    );
    weights.insert(
        "vision_tower.blocks.0.attn.sinks".into(),
        mlxcel_core::zeros(&[2], dtype::FLOAT32),
    );
    weights.insert(
        "model.mtp.layers.0.self_attn.q_proj.weight".into(),
        mlxcel_core::ones(&[4, 4], dtype::FLOAT32),
    );
    let text = sanitize_text_weights(weights, &args).expect("sanitizes");
    assert!(text.contains_key("model.norm.weight"));
    assert!(!text.keys().any(|key| key.starts_with("audio_encoder.")));
    assert!(!text.keys().any(|key| key.starts_with("vision_tower.")));
    assert!(!text.keys().any(|key| key.contains(".mtp.")));
}

/// The loader entry point is wired, but a real checkpoint is 173.8 GB and not
/// available here. This asserts the dispatch reaches the MiMo loader rather
/// than falling through to "unsupported"; the load itself fails on the missing
/// weights, which is the expected error for an empty directory.
#[test]
fn loader_entry_point_is_wired() {
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::write(dir.path().join("config.json"), V26_CONFIG).expect("write config");
    let error = load_mimo_v2_vlm(dir.path())
        .err()
        .expect("no weight shards exist");
    let message = format!("{error:#}");
    assert!(
        message.contains("Failed to load MiMo-V2 text model")
            || message.contains("weight")
            || message.contains("config")
            || message.contains("safetensors")
            || message.contains("files"),
        "unexpected loader error: {message}"
    );
}

#[test]
fn native_layout_is_dequantized_and_stacked_into_switch_mlp() {
    let mut args = full_attention_args();
    args.num_hidden_layers = 3;
    args.hybrid_layer_pattern = vec![0, 1, 1];
    args.moe_layer_freq = vec![0, 1, 1];
    args.n_routed_experts = Some(2);
    args.num_experts_per_tok = 1;
    args.moe_intermediate_size = 16;

    let mut weights = WeightMap::new();
    // Dense prefix MLP: block FP8 with a 1x1 inverse-scale grid.
    weights.insert(
        "model.layers.0.mlp.gate_proj.weight".into(),
        uint8_array(&vec![E4M3_ONE; 8 * 32], &[8, 32]),
    );
    weights.insert(
        "model.layers.0.mlp.gate_proj.weight_scale_inv".into(),
        mlxcel_core::ones(&[1, 1], dtype::FLOAT32),
    );
    // Layer 1: V2.6 MXFP4 experts, per-expert weight + weight_scale.
    for expert in 0..2 {
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            weights.insert(
                format!("model.layers.1.mlp.experts.{expert}.{projection}.weight"),
                uint8_array(&vec![E4M3_ONE; 8 * 16], &[8, 16]),
            );
            weights.insert(
                format!("model.layers.1.mlp.experts.{expert}.{projection}.weight_scale"),
                uint8_array(&[0u32; 8], &[8, 1]),
            );
        }
    }
    // Layer 2: V2.5 FP8 experts, per-expert weight + weight_scale_inv.
    for expert in 0..2 {
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            weights.insert(
                format!("model.layers.2.mlp.experts.{expert}.{projection}.weight"),
                uint8_array(&vec![E4M3_ONE; 8 * 32], &[8, 32]),
            );
            weights.insert(
                format!("model.layers.2.mlp.experts.{expert}.{projection}.weight_scale_inv"),
                mlxcel_core::ones(&[1, 1], dtype::FLOAT32),
            );
        }
    }

    let out = sanitize_text_weights(weights, &args).expect("native layout sanitizes");
    let dense = out
        .get("model.layers.0.mlp.gate_proj.weight")
        .expect("dense plane");
    assert_eq!(mlxcel_core::array_dtype(dense), dtype::BFLOAT16);
    assert_eq!(mlxcel_core::array_shape(dense), [8, 32]);
    assert_constant(dense, 1.0);

    let mxfp4 = out
        .get("model.layers.1.mlp.switch_mlp.gate_proj.weight")
        .expect("V2.6 stacked plane");
    assert_eq!(mlxcel_core::array_dtype(mxfp4), dtype::UINT32);
    assert_eq!(mlxcel_core::array_shape(mxfp4), [2, 8, 4]);
    let scales = out
        .get("model.layers.1.mlp.switch_mlp.gate_proj.scales")
        .expect("V2.6 stacked scales");
    assert_eq!(mlxcel_core::array_shape(scales), [2, 8, 1]);

    let fp8 = out
        .get("model.layers.2.mlp.switch_mlp.gate_proj.weight")
        .expect("V2.5 stacked plane");
    assert_eq!(mlxcel_core::array_dtype(fp8), dtype::BFLOAT16);
    assert_eq!(mlxcel_core::array_shape(fp8), [2, 8, 32]);
    assert_constant(fp8, 1.0);

    assert!(
        !out.keys().any(|key| key.contains(".experts.")),
        "per-expert planes must not survive stacking"
    );
}
