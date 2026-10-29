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

use super::*;

#[test]
fn backbone_keys_lose_the_wrapper_and_take_the_bailing_norm_names() {
    assert_eq!(
        mllm_key("model.model.layers.3.attention.q_norm.weight").as_deref(),
        Some("model.layers.3.attention.query_layernorm.weight")
    );
    assert_eq!(
        mllm_key("model.model.layers.3.attention.k_norm.weight").as_deref(),
        Some("model.layers.3.attention.key_layernorm.weight")
    );
    assert_eq!(
        mllm_key("model.model.layers.1.mlp.image_gate.expert_bias").as_deref(),
        Some("model.layers.1.mlp.image_gate.expert_bias")
    );
    assert_eq!(
        mllm_key("model.model.word_embeddings.weight").as_deref(),
        Some("model.word_embeddings.weight")
    );
    for unread in [
        "model.model.layers.1.mlp.audio_gate.weight",
        "model.lm_head.weight",
        "vision.blocks.0.attn.qkv.weight",
        "linear_proj.weight",
    ] {
        assert_eq!(mllm_key(unread), None, "{unread}");
    }
}

#[test]
fn bridge_keys_map_to_the_upstream_modules() {
    assert_eq!(
        bridge_key("query_tokens_dict.16x16", 16).as_deref(),
        Some("query_tokens")
    );
    assert_eq!(bridge_key("query_tokens_dict.8x8", 16), None);
    assert_eq!(
        bridge_key("proj_in.weight", 16).as_deref(),
        Some("proj_in.weight")
    );
    assert_eq!(
        bridge_key("proj_directvlm.0.weight", 16).as_deref(),
        Some("directvlm_norm.weight")
    );
    assert_eq!(
        bridge_key("proj_directvlm.1.bias", 16).as_deref(),
        Some("directvlm_proj.bias")
    );
}

#[test]
fn experts_stack_along_a_leading_axis_in_order() {
    let mut w = WeightMap::new();
    for e in 0..3 {
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            let v = vec![e as f32; 6];
            w.insert(
                format!("m.experts.{e}.{proj}.weight"),
                mlxcel_core::from_slice_f32(&v, &[2, 3]),
            );
        }
    }
    stack_experts(&mut w, "m", 3).unwrap();
    assert_eq!(w.len(), 3);
    let s = &w["m.switch_mlp.up_proj.weight"];
    assert_eq!(mlxcel_core::array_shape(s), [3, 2, 3]);
    let v = crate::imagegen::testing::values(s);
    assert_eq!((v[0], v[6], v[12]), (0.0, 1.0, 2.0));
    // A missing expert is an error, not a short stack.
    let mut short = WeightMap::new();
    short.insert(
        "m.experts.0.up_proj.weight".into(),
        mlxcel_core::from_slice_f32(&[0.0], &[1, 1]),
    );
    assert!(stack_experts(&mut short, "m", 2).is_err());
    // An expert of another shape is an error naming it, not an FFI abort.
    let mut odd = WeightMap::new();
    for (e, shape) in [[2, 3], [3, 2]].iter().enumerate() {
        odd.insert(
            format!("m.experts.{e}.up_proj.weight"),
            mlxcel_core::from_slice_f32(&[0.0; 6], shape),
        );
    }
    let err = stack_experts(&mut odd, "m", 2).unwrap_err();
    assert!(format!("{err:#}").contains("experts.1.up_proj"), "{err:#}");
}

#[test]
fn only_the_bf16_input_path_keeps_bf16_weights() {
    for bf16 in [
        "x_embedder.weight",
        "t_embedder.linear1.bias",
        "cap_embedder.1.weight",
        "layers.7.adaLN_modulation.0.weight",
        "final_layer.adaLN_modulation.0.bias",
        "noise_refiner.0.attention.to_q.weight",
        "noise_refiner.0.attention_norm1.weight",
        "context_refiner.0.attention.norm_k.weight",
    ] {
        assert!(reads_bf16_activation(bf16), "{bf16}");
    }
    for f32_stream in [
        "noise_refiner.0.attention.to_out.0.weight",
        "noise_refiner.0.feed_forward.w1.weight",
        "noise_refiner.1.attention.to_q.weight",
        "context_refiner.0.attention_norm2.weight",
        "layers.0.attention.to_q.weight",
        "final_layer.linear.weight",
    ] {
        assert!(!reads_bf16_activation(f32_stream), "{f32_stream}");
    }
}

#[test]
fn a_component_without_an_mlx_marker_is_the_raw_layout() {
    let dir = tempfile::tempdir().unwrap();
    assert!(!is_mlx_native(dir.path()).unwrap());
    assert!(read_quant(dir.path()).unwrap().is_none());
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"mlx_format": true, "quantization": {"bits": 4, "group_size": 64}}"#,
    )
    .unwrap();
    assert!(is_mlx_native(dir.path()).unwrap());
    let q = read_quant(dir.path()).unwrap().unwrap();
    assert_eq!((q.bits, q.group_size), (4, 64));
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"mlx_format": true, "quantization": {"bits": 7, "group_size": 64}}"#,
    )
    .unwrap();
    assert!(read_quant(dir.path()).is_err());
}

#[test]
fn the_raw_vae_keeps_the_last_frame_of_each_3d_kernel() {
    let mut raw = WeightMap::new();
    // [O=1, I=1, T=3, 1, 2]: frames 0, 1, 2 hold 0.., 10.., 20..
    raw.insert(
        "decoder.conv_in.weight".into(),
        mlxcel_core::from_slice_f32(&[0.0, 1.0, 10.0, 11.0, 20.0, 21.0], &[1, 1, 3, 1, 2]),
    );
    raw.insert(
        "decoder.norm_out.gamma".into(),
        mlxcel_core::from_slice_f32(&[1.0, 2.0], &[2, 1, 1, 1]),
    );
    raw.insert(
        "decoder.head.bias".into(),
        mlxcel_core::from_slice_f32(&[5.0], &[1]),
    );
    let out = sanitize_vae_weights(raw);
    let conv = &out["decoder.conv_in.weight"];
    assert_eq!(mlxcel_core::array_shape(conv), [1, 1, 1, 2]);
    assert_eq!(crate::imagegen::testing::values(conv), [20.0, 21.0]);
    assert_eq!(
        mlxcel_core::array_shape(&out["decoder.norm_out.gamma"]),
        [2]
    );
    assert_eq!(mlxcel_core::array_shape(&out["decoder.head.bias"]), [1]);
}
