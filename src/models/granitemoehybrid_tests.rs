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

//! Unit tests for Granite 4.x (`granitemoehybrid`) config parsing and the
//! config-derived geometry.
//!
//! These cover the checkpoint-free surface: that the `granitemoehybrid`
//! `config.json` deserializes into `ModelArgs` with the four Granite multipliers,
//! that the `layer_types` interleave classifies Mamba vs attention layers, that
//! `use_moe` follows `num_local_experts`, that the NoPE flag disables RoPE, that
//! the Mamba2 inner dimensions and the `conv_dim` / `projection_size`
//! derivations match the reference, and that the EOS default is the Granite
//! `<|end_of_text|>` id (100257). The numeric correctness of the interleaved
//! SSM + attention hybrid is validated end-to-end against the real
//! `mlx-community/granite-4.0-h-350m-4bit` checkpoint and is not exercised here
//! (no Metal device is assumed in unit tests).

use super::granitemoehybrid::{FeedForward, GraniteMoeHybridMoE, ModelArgs, build_feed_forward};
use crate::models::switch_layers::{
    HONEST_EXPERT_GROUP_SIZE, insert_honest_affine_swiglu_experts, moe_weighted_sum,
};
use mlxcel_core::weights::WeightMap;

fn values(len: usize, start: f32, step: f32) -> Vec<f32> {
    (0..len).map(|idx| start + idx as f32 * step).collect()
}

fn insert_tensor(weights: &mut WeightMap, name: &str, values: Vec<f32>, shape: &[i32]) {
    weights.insert(
        name.to_string(),
        mlxcel_core::from_slice_f32(&values, shape),
    );
}

/// A trimmed `granite-4.0-h-350m` config (the dense-MLP hybrid). Values mirror
/// the real checkpoint, including the full 32-entry `layer_types` interleave.
const GRANITE_4_350M_CONFIG: &str = r#"{
    "model_type": "granitemoehybrid",
    "attention_bias": false,
    "attention_multiplier": 0.015625,
    "embedding_multiplier": 12,
    "eos_token_id": 100257,
    "hidden_size": 768,
    "intermediate_size": 2048,
    "logits_scaling": 3,
    "layer_types": [
        "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "mamba",
        "mamba", "mamba", "attention", "mamba", "mamba", "attention", "mamba",
        "mamba", "mamba", "attention", "mamba", "mamba", "mamba", "mamba",
        "mamba", "mamba", "mamba", "mamba", "mamba", "attention", "mamba",
        "mamba", "mamba", "mamba"
    ],
    "mamba_conv_bias": true,
    "mamba_d_conv": 4,
    "mamba_d_head": 32,
    "mamba_d_state": 128,
    "mamba_n_groups": 1,
    "mamba_n_heads": 48,
    "mamba_proj_bias": false,
    "max_position_embeddings": 32768,
    "num_attention_heads": 12,
    "num_experts_per_tok": 0,
    "num_hidden_layers": 32,
    "num_key_value_heads": 4,
    "num_local_experts": 0,
    "position_embedding_type": "nope",
    "residual_multiplier": 0.246,
    "rms_norm_eps": 1e-05,
    "rope_theta": 10000,
    "shared_intermediate_size": 2048,
    "tie_word_embeddings": true,
    "vocab_size": 100352,
    "quantization": { "group_size": 64, "bits": 4 }
}"#;

#[test]
fn granite_hybrid_config_parses_scalar_multipliers() {
    let args: ModelArgs = serde_json::from_str(GRANITE_4_350M_CONFIG).expect("parse config");
    assert_eq!(args.model_type, "granitemoehybrid");
    assert_eq!(args.embedding_multiplier, 12.0);
    assert_eq!(args.attention_multiplier, 0.015625);
    assert_eq!(args.residual_multiplier, 0.246);
    assert_eq!(args.logits_scaling, 3.0);
    assert!(args.tie_word_embeddings);
}

#[test]
fn granite_hybrid_layer_types_interleave() {
    let args: ModelArgs = serde_json::from_str(GRANITE_4_350M_CONFIG).expect("parse config");

    // The 350m checkpoint places attention at layers 10, 13, 17, 27; the rest
    // are Mamba (28 Mamba + 4 attention).
    let attention_layers: Vec<usize> = (0..args.num_hidden_layers)
        .filter(|&i| !args.is_mamba_layer(i))
        .collect();
    assert_eq!(attention_layers, vec![10, 13, 17, 27]);

    assert!(args.is_mamba_layer(0));
    assert!(!args.is_mamba_layer(10));
    assert!(!args.is_mamba_layer(27));
    assert!(args.is_mamba_layer(31));
}

#[test]
fn granite_hybrid_use_moe_follows_num_local_experts() {
    // num_local_experts == 0 -> dense MLP mode.
    let dense: ModelArgs = serde_json::from_str(GRANITE_4_350M_CONFIG).expect("parse config");
    assert!(!dense.use_moe());

    // A positive expert count switches on the MoE feed-forward.
    let moe_cfg = GRANITE_4_350M_CONFIG
        .replace("\"num_experts_per_tok\": 0,", "\"num_experts_per_tok\": 8,")
        .replace("\"num_local_experts\": 0,", "\"num_local_experts\": 64,");
    let moe: ModelArgs = serde_json::from_str(&moe_cfg).expect("parse moe config");
    assert!(moe.use_moe());
    assert_eq!(moe.num_local_experts, 64);
    assert_eq!(moe.num_experts_per_tok, 8);
}

#[test]
fn granite_hybrid_nope_disables_rope() {
    // position_embedding_type == "nope" -> no RoPE.
    let nope: ModelArgs = serde_json::from_str(GRANITE_4_350M_CONFIG).expect("parse config");
    assert_eq!(nope.position_embedding_type, "nope");
    assert!(!nope.use_rope());

    // An explicit "rope" re-enables it.
    let rope_cfg = GRANITE_4_350M_CONFIG.replace(
        "\"position_embedding_type\": \"nope\",",
        "\"position_embedding_type\": \"rope\",",
    );
    let rope: ModelArgs = serde_json::from_str(&rope_cfg).expect("parse rope config");
    assert!(rope.use_rope());
}

#[test]
fn granite_hybrid_head_dim_and_mamba_dims() {
    let args: ModelArgs = serde_json::from_str(GRANITE_4_350M_CONFIG).expect("parse config");

    // Attention head_dim = hidden_size / num_attention_heads = 768 / 12 = 64.
    assert_eq!(args.head_dim(), 64);

    // Mamba inner width = mamba_n_heads * mamba_d_head = 48 * 32 = 1536.
    assert_eq!(args.mamba_intermediate(), 1536);
    assert_eq!(args.mamba_n_heads, 48);
    assert_eq!(args.mamba_d_head, 32);
    assert_eq!(args.mamba_d_state, 128);
    assert_eq!(args.mamba_d_conv, 4);
    assert_eq!(args.mamba_n_groups, 1);
    assert!(args.mamba_conv_bias);
}

#[test]
fn granite_hybrid_conv_and_projection_derivations() {
    let args: ModelArgs = serde_json::from_str(GRANITE_4_350M_CONFIG).expect("parse config");

    // conv_dim = mamba_intermediate + 2 * n_groups * d_state = 1536 + 2*1*128.
    assert_eq!(args.conv_dim(), 1792);
    assert_eq!(
        args.conv_dim(),
        args.mamba_intermediate() + 2 * args.mamba_n_groups * args.mamba_d_state
    );

    // projection_size = mamba_intermediate + conv_dim + mamba_n_heads.
    assert_eq!(args.projection_size(), 1536 + 1792 + 48);
    assert_eq!(args.projection_size(), 3376);
}

#[test]
fn granite_hybrid_time_step_limit_defaults_to_ssm_update_default() {
    // Granite passes no explicit limit, so it falls back to (0.001, 100.0).
    let args: ModelArgs = serde_json::from_str(GRANITE_4_350M_CONFIG).expect("parse config");
    assert_eq!(args.time_step_limit, (0.001, 100.0));
}

#[test]
fn granite_hybrid_eos_token_id_scalar_and_missing() {
    // Scalar form (the 350m checkpoint ships eos_token_id == 100257).
    let scalar: ModelArgs = serde_json::from_str(GRANITE_4_350M_CONFIG).expect("parse config");
    assert_eq!(scalar.eos_token_ids(), vec![100257]);

    // Missing -> default to the Granite `<|end_of_text|>` id (100257).
    let no_eos = GRANITE_4_350M_CONFIG.replace("\"eos_token_id\": 100257,", "");
    let missing: ModelArgs = serde_json::from_str(&no_eos).expect("parse no-eos config");
    assert_eq!(missing.eos_token_ids(), vec![100257]);
}

#[test]
fn granite_hybrid_quantization_read_from_config() {
    let args: ModelArgs = serde_json::from_str(GRANITE_4_350M_CONFIG).expect("parse config");
    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 4);

    // Dropping the quantization block falls back to the defaults.
    let no_quant = GRANITE_4_350M_CONFIG.replace(
        ",\n    \"quantization\": { \"group_size\": 64, \"bits\": 4 }",
        "",
    );
    let bf16: ModelArgs = serde_json::from_str(&no_quant).expect("parse unquantized config");
    assert!(bf16.quantization.is_none());
    assert_eq!(bf16.group_size(), 64);
    assert_eq!(bf16.bits(), 4);
}

/// Hidden width of the fused-kernel geometry below: one quantization group per
/// input row, so one token carries 64 elements. Reuses the honest affine 4-bit
/// expert triple the Qwen3MoE fused-geometry tests use (issue #1884).
const FUSED_HIDDEN: i32 = HONEST_EXPERT_GROUP_SIZE;
const FUSED_EXPERTS: i32 = 2;

/// A granite `GraniteMoeHybridMoE` over two honest affine 4-bit bf16 experts of
/// intermediate width `dff`, loaded through the family's own `build_feed_forward`,
/// so the fused-kernel decision under test runs on the exact type and stored
/// triple a converted granite-4.0-h checkpoint produces.
fn fused_moe_block() -> GraniteMoeHybridMoE {
    let mut weights = WeightMap::new();
    insert_honest_affine_swiglu_experts(
        &mut weights,
        "model.layers.0.block_sparse_moe.switch_mlp",
        FUSED_EXPERTS,
        FUSED_HIDDEN,
        FUSED_HIDDEN,
    );
    // A non-quantized router and shared expert, so the declared pair can only
    // reach the routed experts.
    insert_tensor(
        &mut weights,
        "model.layers.0.block_sparse_moe.router.layer.weight",
        values((FUSED_EXPERTS * FUSED_HIDDEN) as usize, -0.2, 0.004),
        &[FUSED_EXPERTS, FUSED_HIDDEN],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.shared_mlp.input_linear.weight",
        values((2 * FUSED_HIDDEN * FUSED_HIDDEN) as usize, 0.01, 0.002),
        &[2 * FUSED_HIDDEN, FUSED_HIDDEN],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.shared_mlp.output_linear.weight",
        values((FUSED_HIDDEN * FUSED_HIDDEN) as usize, 0.02, 0.002),
        &[FUSED_HIDDEN, FUSED_HIDDEN],
    );
    let args: ModelArgs = serde_json::from_value(serde_json::json!({
        "model_type": "granitemoehybrid",
        "vocab_size": 8,
        "hidden_size": FUSED_HIDDEN,
        "intermediate_size": FUSED_HIDDEN,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-5,
        "embedding_multiplier": 1.0,
        "attention_multiplier": 1.0,
        "logits_scaling": 1.0,
        "residual_multiplier": 1.0,
        "layer_types": ["attention"],
        "num_local_experts": FUSED_EXPERTS,
        "num_experts_per_tok": FUSED_EXPERTS,
        "shared_intermediate_size": FUSED_HIDDEN,
        "quantization": { "group_size": 64, "bits": 4 },
    }))
    .expect("fused-geometry granite config must parse");
    let ff = build_feed_forward(&weights, &args, "model.layers.0")
        .expect("honest affine 4-bit granite experts must load");
    match ff {
        FeedForward::Moe { moe, .. } => moe,
        _ => panic!("num_local_experts 2 must build the MoE feed-forward"),
    }
}

/// Before issue #2245 granite's `block_sparse_moe` never called
/// `forward_fused_kernel`, so the fused single-token MoE kernel (#268) never
/// ran for the family and every decode step paid the `gather_qmm` fallback
/// (25.4% of decode GPU time on gfx1151, per the #2061 profile). The kernel
/// must decline nothing on an honest affine 4-bit expert triple of dff 64 and
/// compute the `forward` + `moe_weighted_sum` function within the tolerance
/// that test pins. It launches the kernel, which on a backend without a port
/// refuses across the bridge, so it runs only when
/// `fused_moe_kernels_available()` says both ports exist.
#[test]
fn granite_hybrid_fused_kernel_matches_the_gather_path_at_one_token() {
    if !mlxcel_core::fused_moe_kernels_available() {
        eprintln!("skipping: this backend has no fused MoE kernel port (#1803)");
        return;
    }
    let moe = fused_moe_block();
    let x = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(
            &values(FUSED_HIDDEN as usize, -0.4, 0.0125),
            &[1, FUSED_HIDDEN],
        ),
        mlxcel_core::dtype::BFLOAT16,
    );
    let indices = mlxcel_core::from_slice_i32(&[0, 1], &[1, FUSED_EXPERTS]);
    let scores = mlxcel_core::from_slice_f32(&[0.25, 0.75], &[1, FUSED_EXPERTS]);

    let fused = moe
        .switch_mlp
        .forward_fused_kernel(&x, &indices, &scores)
        .expect("honest affine 4-bit experts of dff 64 must dispatch the fused kernel");
    let fused = mlxcel_core::reshape(&fused, &[1, FUSED_HIDDEN]);
    let gathered = moe_weighted_sum(
        &moe.switch_mlp.forward(&x, &indices),
        &scores,
        mlxcel_core::dtype::BFLOAT16,
    );
    // A structural check, not a numeric parity bound (that lives in
    // `fused_moe_parity_tests`): swapped planes or a dropped expert land near
    // 1, the bf16 quantized jitter between the two paths two orders lower.
    let fused = mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(
        &fused,
        mlxcel_core::dtype::FLOAT32,
    ));
    let gathered = mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(
        &gathered,
        mlxcel_core::dtype::FLOAT32,
    ));
    let diff_sq: f64 = fused
        .iter()
        .zip(&gathered)
        .map(|(a, b)| f64::from(a - b).powi(2))
        .sum();
    let ref_sq: f64 = gathered.iter().map(|b| f64::from(*b).powi(2)).sum();
    let nrms = (diff_sq / ref_sq.max(1e-20)).sqrt();
    assert!(
        fused.iter().all(|v| v.is_finite()) && nrms < 0.1,
        "the fused kernel must compute the gather path's function: nrms {nrms:e}"
    );
}

/// Regression for the `slice_update`-is-functional footgun: `inject_at_positions`
/// MUST write the visual features into `h` at the image-token positions. A
/// dropped `slice_update` return leaves the image slots zero and silently makes
/// the VLM blind (Granite 4 Vision #541). Runs on the default device.
#[test]
fn inject_at_positions_writes_features_into_hidden_state() {
    use super::granitemoehybrid::inject_at_positions;

    let (seq, hidden) = (5i32, 4i32);
    // Image slots start zeroed, exactly as `get_input_embeddings` produces them.
    let h = mlxcel_core::zeros(&[1, seq, hidden], mlxcel_core::dtype::FLOAT32);
    // Image tokens occupy positions 1, 2, 3 (a contiguous single-image run).
    let mask = mlxcel_core::from_slice_i32(&[0, 1, 1, 1, 0], &[1, seq]);
    // Three feature rows, each a distinct non-zero constant.
    let feats = mlxcel_core::from_slice_f32(
        &[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0],
        &[3, hidden],
    );

    let out = inject_at_positions(&h, &mask, &feats);
    mlxcel_core::eval(&out);
    let val = |pos: i32, col: i32| {
        let c = mlxcel_core::slice(&out, &[0, pos, col], &[1, pos + 1, col + 1]);
        mlxcel_core::eval(&c);
        mlxcel_core::item_f32(&c)
    };

    // Non-image positions stay zero; image positions carry their feature row.
    assert_eq!(val(0, 0), 0.0, "non-image position must be untouched");
    assert_eq!(
        val(1, 0),
        1.0,
        "image position 1 must receive feature row 0"
    );
    assert_eq!(
        val(2, 2),
        2.0,
        "image position 2 must receive feature row 1"
    );
    assert_eq!(
        val(3, 3),
        3.0,
        "image position 3 must receive feature row 2"
    );
    assert_eq!(val(4, 0), 0.0, "non-image position must be untouched");
}
