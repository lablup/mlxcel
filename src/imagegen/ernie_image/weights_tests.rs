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

use serde_json::json;

use super::*;
use crate::imagegen::testing::values;

fn meta(v: Value) -> Map<String, Value> {
    v.as_object().unwrap().clone()
}

#[test]
fn layout_comes_from_metadata_and_defaults_to_pytorch() {
    assert_eq!(tensor_layout(&meta(json!({}))), ConvLayout::PytorchNchw);
    assert_eq!(
        tensor_layout(&meta(json!({"format": "pt"}))),
        ConvLayout::PytorchNchw
    );
    assert_eq!(
        tensor_layout(&meta(json!({"format": "mlx"}))),
        ConvLayout::MlxNhwc
    );
    assert_eq!(
        tensor_layout(&meta(json!({"mflux_version": "0.9"}))),
        ConvLayout::MlxNhwc
    );
    assert_eq!(
        tensor_layout(&meta(
            json!({"tensor_layout": "pytorch_nchw", "format": "mlx"})
        )),
        ConvLayout::PytorchNchw
    );
}

#[test]
fn the_patch_embed_kernel_is_permuted_once() {
    let data: Vec<f32> = (0..2 * 3).map(|v| v as f32).collect();
    // PyTorch [O=2, C=3, 1, 1] -> MLX [2, 1, 1, 3].
    let torch = mlxcel_core::from_slice_f32(&data, &[2, 3, 1, 1]);
    let target = [2, 1, 1, 3];
    let mlx = match_conv_layout(&torch, &target, "k", ConvLayout::PytorchNchw).unwrap();
    assert_eq!(mlxcel_core::array_shape(&mlx), target);
    assert_eq!(values(&mlx), data);
    // An already-permuted kernel is kept as stored when the metadata says MLX...
    let kept = match_conv_layout(&mlx, &target, "k", ConvLayout::MlxNhwc).unwrap();
    assert_eq!(values(&kept), data);
    // ...and refused, not permuted a second time, when labelled PyTorch.
    let Err(err) = match_conv_layout(&mlx, &target, "k", ConvLayout::PytorchNchw) else {
        panic!("an MLX kernel labelled PyTorch was permuted again");
    };
    assert!(err.to_string().contains("shape mismatch"), "{err}");
    // A PyTorch kernel mislabelled as MLX is refused too.
    assert!(match_conv_layout(&torch, &target, "k", ConvLayout::MlxNhwc).is_err());
}

#[test]
fn sanitize_renames_casts_and_rejects_quantized_checkpoints() {
    let config = ErnieTransformerConfig {
        hidden_size: 4,
        in_channels: 2,
        num_attention_heads: 1,
        rope_axes_dim: vec![2, 2, 0],
        ..ErnieTransformerConfig::default()
    };
    let mut raw = WeightMap::new();
    raw.insert(
        "adaLN_modulation.1.weight".into(),
        mlxcel_core::zeros(&[24, 4], dtype::FLOAT32),
    );
    raw.insert(
        "layers.0.self_attention.to_out.0.weight".into(),
        mlxcel_core::zeros(&[4, 4], dtype::FLOAT16),
    );
    raw.insert(
        "x_embedder.proj.weight".into(),
        mlxcel_core::zeros(&[4, 2, 1, 1], dtype::BFLOAT16),
    );
    let w = sanitize_transformer_weights(raw, &config, ConvLayout::PytorchNchw).unwrap();
    assert_eq!(
        mlxcel_core::array_dtype(&w["adaln_modulation.weight"]),
        TRANSFORMER_DTYPE
    );
    assert!(w.contains_key("layers.0.self_attention.to_out.weight"));
    assert_eq!(
        mlxcel_core::array_shape(&w["x_embedder.proj.weight"]),
        [4, 1, 1, 2]
    );
    let mut quantized = WeightMap::new();
    quantized.insert(
        "layers.0.mlp.up_proj.scales".into(),
        mlxcel_core::zeros(&[4, 1], dtype::FLOAT16),
    );
    let Err(err) = sanitize_transformer_weights(quantized, &config, ConvLayout::PytorchNchw) else {
        panic!("a quantized checkpoint was accepted");
    };
    assert!(err.to_string().contains("quantized"), "{err}");
}
