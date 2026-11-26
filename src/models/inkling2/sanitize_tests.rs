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

use std::collections::BTreeMap;

use mlxcel_core::dtype;
use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::weights::WeightMap;

use super::*;

const H: i32 = 8;
const E: i32 = 2;
const I: i32 = 8;

fn bf16(shape: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::zeros(shape, dtype::BFLOAT16)
}

fn bytes(shape: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::zeros(shape, dtype::UINT8)
}

/// A native key set: layer 0 sliding (NVFP4 experts, no attention input),
/// layer 1 global in the loop core (MXFP8 experts, controllers), the attn_res
/// queries, the two cross-pass tensors, the visual backbone and `mtp.*`.
fn native() -> WeightMap {
    let mut w = WeightMap::new();
    for layer in 0..2 {
        let p = format!("model.llm.layers.{layer}");
        for name in [
            "wq_du",
            "wk_dv",
            "wv_dv",
            "wr_du",
            "wo_ud",
            "gate_attn",
            "attn_sink_proj",
        ] {
            w.insert(format!("{p}.attn.{name}.weight"), bf16(&[H, H]));
        }
        w.insert(format!("{p}.attn.q_l2_clip.scale"), bf16(&[2, 4]));
        w.insert(format!("{p}.attn.rel_l2_clip.scale"), bf16(&[2, 2]));
        w.insert(format!("{p}.attn.rel_logits_proj.proj"), bf16(&[2, 16]));
        w.insert(format!("{p}.attn_sconv.weight"), bf16(&[H, 1, 4]));
        w.insert(format!("{p}.mlp_sconv.weight"), bf16(&[H, 1, 4]));
        w.insert(format!("{p}.mlp.gate.weight"), bf16(&[E, H]));
        w.insert(
            format!("{p}.mlp.gate.bias"),
            mlxcel_core::zeros(&[E], dtype::FLOAT32),
        );
        w.insert(format!("{p}.mlp_norm.weight"), bf16(&[H]));
    }
    let sliding = "model.llm.layers.0";
    w.insert(format!("{sliding}.attn.k_sconv.weight"), bf16(&[4, 1, 4]));
    w.insert(format!("{sliding}.attn.v_sconv.weight"), bf16(&[4, 1, 4]));
    w.insert(format!("{sliding}.attn_norm.weight"), bf16(&[H]));
    // NVFP4: packed bytes [E, 2I, H/2], E4M3 scales [E, 2I, H/16 -> 1].
    w.insert(
        format!("{sliding}.mlp.experts.w13_weight"),
        bytes(&[E, 2 * I, H / 2]),
    );
    w.insert(
        format!("{sliding}.mlp.experts.w13_weight_scale"),
        bytes(&[E, 2 * I, 1]),
    );
    w.insert(
        format!("{sliding}.mlp.experts.w13_weight_amax"),
        mlxcel_core::from_slice_f32(&[6.0, 12.0], &[2]),
    );
    w.insert(
        format!("{sliding}.mlp.experts.w13_weight_outer_scale_denominator"),
        mlxcel_core::from_slice_f32(&[3.0], &[1]),
    );
    w.insert(
        format!("{sliding}.mlp.experts.w2_weight"),
        bytes(&[E, H, I / 2]),
    );
    w.insert(
        format!("{sliding}.mlp.experts.w2_weight_scale"),
        bytes(&[E, H, 1]),
    );
    w.insert(
        format!("{sliding}.mlp.experts.w2_weight_amax"),
        mlxcel_core::from_slice_f32(&[1.0, 2.0], &[2]),
    );
    w.insert(
        format!("{sliding}.mlp.experts.w2_weight_outer_scale_denominator"),
        mlxcel_core::from_slice_f32(&[4.0], &[1]),
    );

    let global = "model.llm.layers.1";
    for kind in ["k", "v"] {
        w.insert(
            format!("{global}.attn.{kind}_sconv.static_w"),
            bf16(&[4, 4]),
        );
        w.insert(
            format!("{global}.attn.{kind}_sconv.weight_proj"),
            bf16(&[4, H]),
        );
    }
    for norm in ["q_input_norm", "k_input_norm", "v_input_norm"] {
        w.insert(format!("{global}.{norm}.weight"), bf16(&[H]));
    }
    for controller in ["attn", "k", "v", "mlp"] {
        w.insert(
            format!("{global}.controllers.{controller}.down_weight"),
            bf16(&[2, 2 * H]),
        );
        w.insert(
            format!("{global}.controllers.{controller}.forget_weight"),
            bf16(&[H, 2]),
        );
        w.insert(
            format!("{global}.controllers.{controller}.gain_weight"),
            bf16(&[H, 2]),
        );
    }
    // MXFP8: E4M3 bytes [E, 2I, H] packed 4 per uint32, UE8M0 scales.
    w.insert(
        format!("{global}.mlp.experts.w13_weight"),
        bytes(&[E, 2 * I, H]),
    );
    w.insert(
        format!("{global}.mlp.experts.w13_weight_scale"),
        bytes(&[E, 2 * I, 1]),
    );
    w.insert(format!("{global}.mlp.experts.w2_weight"), bytes(&[E, H, I]));
    w.insert(
        format!("{global}.mlp.experts.w2_weight_scale"),
        bytes(&[E, H, 1]),
    );

    w.insert(
        "model.llm.attn_res.attn_queries.0".into(),
        mlxcel_core::zeros(&[H], dtype::FLOAT32),
    );
    w.insert(
        "model.llm.attn_res.mlp_queries.0".into(),
        mlxcel_core::zeros(&[H], dtype::FLOAT32),
    );
    w.insert(
        "model.llm.attn_res.mlp_queries.1".into(),
        mlxcel_core::zeros(&[H], dtype::FLOAT32),
    );
    w.insert(
        "model.llm.attn_res.output_query".into(),
        mlxcel_core::zeros(&[H], dtype::FLOAT32),
    );
    // One core layer: [-1, 2, h] stored flat as [2, h]; distinct row values.
    let qa: Vec<f32> = (0..2 * H).map(|i| (i / H) as f32 + 1.0).collect();
    let kv: Vec<f32> = (0..2 * H).map(|i| (i / H) as f32 + 10.0).collect();
    w.insert(
        "model.llm.middle_cycle_cross_pass_queries".into(),
        mlxcel_core::from_slice_f32(&qa, &[2, H]),
    );
    w.insert(
        "model.llm.middle_cycle_cross_pass_kv_queries".into(),
        mlxcel_core::from_slice_f32(&kv, &[2, H]),
    );
    w.insert("model.llm.embed.weight".into(), bf16(&[16, H]));
    w.insert("model.llm.embed_norm.weight".into(), bf16(&[H]));
    w.insert("model.llm.norm.weight".into(), bf16(&[H]));
    w.insert("model.llm.unembed.weight".into(), bf16(&[16, H]));
    for key in [
        "model.visual.backbone.embeddings.class_embedding",
        "model.visual.backbone.ln_pre.weight",
        "model.visual.backbone.layers.0.ln_1.weight",
        "model.visual.backbone.layers.0.ln_2.bias",
        "model.visual.backbone.layers.0.ls_1.gamma",
        "model.visual.backbone.layers.0.ls_2.gamma",
        "model.visual.backbone.layers.0.mlp.c_fc.weight",
        "model.visual.backbone.layers.0.mlp.c_proj.bias",
        "model.visual.backbone.layers.0.self_attn.q_proj.weight",
        "model.visual.proj_in.weight",
    ] {
        w.insert(key.into(), bf16(&[H]));
    }
    w.insert("mtp.layers.0.input_layernorm.weight".into(), bf16(&[H]));
    w.insert("model.mtp.fuse_weight".into(), bf16(&[H]));
    w
}

fn shapes(weights: &WeightMap) -> BTreeMap<String, Vec<i32>> {
    weights
        .iter()
        .map(|(key, value)| (key.clone(), mlxcel_core::array_shape(value)))
        .collect()
}

#[test]
fn native_keys_map_to_exact_names_and_shapes() {
    let out = sanitize_weights(native(), H as usize).unwrap();
    let s = shapes(&out);
    let t = "language_model.model.layers";
    let expected: &[(&str, &[i32])] = &[
        (
            "language_model.model.layers.0.self_attn.q_proj.weight",
            &[H, H],
        ),
        (
            "language_model.model.layers.0.self_attn.sink_proj.weight",
            &[H, H],
        ),
        (
            "language_model.model.layers.0.self_attn.gate_proj.weight",
            &[H, H],
        ),
        ("language_model.model.layers.0.self_attn.q_scale", &[2, 4]),
        ("language_model.model.layers.0.self_attn.rel_scale", &[2, 2]),
        ("language_model.model.layers.0.self_attn.rel_proj", &[2, 16]),
        // Native [C, 1, K] conv weights become MLX [C, K, 1].
        (
            "language_model.model.layers.0.self_attn.k_sconv.conv.weight",
            &[4, 4, 1],
        ),
        (
            "language_model.model.layers.0.attn_sconv.conv.weight",
            &[H, 4, 1],
        ),
        ("language_model.model.layers.0.input_layernorm.weight", &[H]),
        (
            "language_model.model.layers.0.post_attention_layernorm.weight",
            &[H],
        ),
        ("language_model.model.layers.0.mlp.gate_weight", &[E, H]),
        (
            "language_model.model.layers.0.mlp.e_score_correction_bias",
            &[E],
        ),
        (
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
            &[E, I, H / 8],
        ),
        (
            "language_model.model.layers.0.mlp.switch_mlp.up_proj.scales",
            &[E, I, 1],
        ),
        (
            "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight",
            &[E, H, I / 8],
        ),
        (
            "language_model.model.layers.0.mlp.switch_mlp.down_proj.native_global_scale",
            &[E],
        ),
        (
            "language_model.model.layers.1.self_attn.k_sconv.static_weight",
            &[4, 4],
        ),
        (
            "language_model.model.layers.1.self_attn.v_sconv.weight_proj.weight",
            &[4, H],
        ),
        (
            "language_model.model.layers.1.q_input_layernorm.weight",
            &[H],
        ),
        (
            "language_model.model.layers.1.controllers.mlp.down_weight",
            &[2, 2 * H],
        ),
        (
            "language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight",
            &[E, I, H / 4],
        ),
        (
            "language_model.model.layers.1.mlp.switch_mlp.down_proj.weight",
            &[E, H, I / 4],
        ),
        ("language_model.model.layers.1.attention_input.query", &[H]),
        ("language_model.model.layers.0.mlp_input.query", &[H]),
        ("language_model.model.layers.1.mlp_input.query", &[H]),
        ("language_model.model.final_input.query", &[H]),
        ("language_model.model.message_queries", &[1, 4, H]),
        ("language_model.model.embed_tokens.weight", &[16, H]),
        ("language_model.model.embed_norm.weight", &[H]),
        ("language_model.model.norm.weight", &[H]),
        ("language_model.lm_head.weight", &[16, H]),
        ("vision_tower.embeddings.class_embedding", &[H]),
        ("vision_tower.pre_layernorm.weight", &[H]),
        ("vision_tower.layers.0.layernorm_before.weight", &[H]),
        ("vision_tower.layers.0.layernorm_after.bias", &[H]),
        ("vision_tower.layers.0.lambda_1", &[H]),
        ("vision_tower.layers.0.lambda_2", &[H]),
        ("vision_tower.layers.0.mlp.fc1.weight", &[H]),
        ("vision_tower.layers.0.mlp.fc2.bias", &[H]),
        ("vision_tower.layers.0.self_attn.q_proj.weight", &[H]),
        ("vision_tower.proj_in.weight", &[H]),
    ];
    for (key, shape) in expected {
        assert_eq!(s.get(*key).map(Vec::as_slice), Some(*shape), "{key}");
    }
    // MXFP8 layers carry no tensor scale; NVFP4 gate/up share one.
    assert!(!s.contains_key(&format!(
        "{t}.1.mlp.switch_mlp.gate_proj.native_global_scale"
    )));
    assert!(s.contains_key(&format!("{t}.0.mlp.switch_mlp.up_proj.native_global_scale")));
    // Nothing native, nothing MTP.
    assert!(
        s.keys()
            .all(|key| !key.starts_with("model.") && !key.starts_with("mtp.")),
        "{s:?}"
    );
    assert!(!s.contains_key("language_model.model.layers.0.attention_input.query"));
}

#[test]
fn message_queries_are_attn_k_v_mlp() {
    let out = sanitize_weights(native(), H as usize).unwrap();
    let fused = out.get("language_model.model.message_queries").unwrap();
    let values = array_to_vec_f32(&mlxcel_core::astype(fused, dtype::FLOAT32));
    // qa rows are [1, 2] (attn, mlp), kv rows are [10, 11] (k, v).
    let rows: Vec<f32> = values.chunks(H as usize).map(|row| row[0]).collect();
    assert_eq!(rows, [1.0, 10.0, 11.0, 2.0]);
}

#[test]
fn weight_scale_2_is_amax_over_denominator() {
    let out = sanitize_weights(native(), H as usize).unwrap();
    let scale = out
        .get("language_model.model.layers.0.mlp.switch_mlp.gate_proj.native_global_scale")
        .unwrap();
    let values = array_to_vec_f32(scale);
    // weight_scale_2 = [6, 12] / 3 = [2, 4]; MLX takes 2688 * weight_scale_2.
    assert_eq!(values, [2.0 * 2688.0, 4.0 * 2688.0]);
    let down = out
        .get("language_model.model.layers.0.mlp.switch_mlp.down_proj.native_global_scale")
        .unwrap();
    assert_eq!(array_to_vec_f32(down), [0.25 * 2688.0, 0.5 * 2688.0]);
}

#[test]
fn remap_is_a_no_op_on_converted_keys() {
    let once = sanitize_weights(native(), H as usize).unwrap();
    let before = shapes(&once);
    let twice = sanitize_weights(once, H as usize).unwrap();
    assert_eq!(shapes(&twice), before);
}

#[test]
fn unmapped_llm_key_is_an_error_naming_it() {
    let mut weights = native();
    weights.insert("model.llm.layers.0.attn.mystery.weight".into(), bf16(&[H]));
    let error = sanitize_weights(weights, H as usize)
        .err()
        .expect("must fail");
    assert!(
        error.contains("model.llm.layers.0.attn.mystery.weight"),
        "{error}"
    );

    let mut weights = native();
    weights.insert("model.llm.surprise".into(), bf16(&[H]));
    let error = sanitize_weights(weights, H as usize)
        .err()
        .expect("must fail");
    assert!(error.contains("model.llm.surprise"), "{error}");
}

#[test]
fn transformers_layout_maps_to_the_same_names() {
    let mut w = WeightMap::new();
    let t = "model.language_model.layers.6";
    w.insert(
        format!("{t}.mlp.experts.gate_up_proj.weight"),
        mlxcel_core::zeros(&[E, 2 * I, 1], dtype::UINT32),
    );
    w.insert(
        format!("{t}.mlp.experts.gate_up_proj.scales"),
        bytes(&[E, 2 * I, 1]),
    );
    w.insert(
        format!("{t}.mlp.experts.gate_up_proj.global_scale"),
        mlxcel_core::from_slice_f32(&[1.0, 2.0], &[2]),
    );
    w.insert(
        format!("{t}.mlp.experts.down_proj.global_scale"),
        mlxcel_core::zeros(&[E], dtype::FLOAT32),
    );
    w.insert(
        format!("{t}.q_message_gate.message_gate_proj.weight"),
        bf16(&[H, 2]),
    );
    w.insert(
        format!("{t}.mlp_message_gate.down_proj.weight"),
        bf16(&[2, 2 * H]),
    );
    w.insert(
        format!("{t}.post_attention_convolution.conv1d.weight"),
        bf16(&[H, 4, 1]),
    );
    w.insert(
        format!("{t}.self_attn.k_sconv.conv1d.weight"),
        bf16(&[4, 4, 1]),
    );
    w.insert(format!("{t}.self_attn.q_l2_clip.scale"), bf16(&[2, 4]));
    w.insert(
        format!("{t}.mlp.gate.e_score_correction_bias"),
        mlxcel_core::zeros(&[E], dtype::FLOAT32),
    );
    w.insert(
        "model.language_model.message_input.query".into(),
        mlxcel_core::zeros(&[1, 4, H], dtype::FLOAT32),
    );
    w.insert(
        "model.language_model.embed_tokens.weight".into(),
        bf16(&[16, H]),
    );
    w.insert(
        "model.vision_tower.merger.linear_1.weight".into(),
        bf16(&[H, H]),
    );
    w.insert(
        "model.vision_tower.embeddings.position_embedding.weight".into(),
        bf16(&[H, H]),
    );
    w.insert("lm_head.scales".into(), bf16(&[16, 1]));
    let s = shapes(&sanitize_weights(w, H as usize).unwrap());
    let l = "language_model.model.layers.6";
    for key in [
        format!("{l}.mlp.switch_mlp.gate_proj.weight"),
        format!("{l}.mlp.switch_mlp.up_proj.scales"),
        format!("{l}.mlp.switch_mlp.up_proj.native_global_scale"),
        format!("{l}.mlp.switch_mlp.down_proj.native_global_scale"),
        format!("{l}.controllers.attn.gain_weight"),
        format!("{l}.controllers.mlp.down_weight"),
        format!("{l}.attn_sconv.conv.weight"),
        format!("{l}.self_attn.k_sconv.conv.weight"),
        format!("{l}.self_attn.q_scale"),
        format!("{l}.mlp.e_score_correction_bias"),
        "language_model.model.message_queries".into(),
        "language_model.model.embed_tokens.weight".into(),
        "vision_tower.proj_in.weight".into(),
        "vision_tower.embeddings.position_embedding".into(),
        "language_model.lm_head.scales".into(),
    ] {
        assert!(s.contains_key(&key), "{key} missing from {s:?}");
    }
    assert_eq!(
        s[&format!("{l}.mlp.switch_mlp.gate_proj.weight")],
        [E, I, 1]
    );
}
