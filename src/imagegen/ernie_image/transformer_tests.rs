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

//! A tiny random-weight transformer: shapes, positions and the padding mask.

use super::*;
use crate::imagegen::testing::values;

const D: i32 = 16;
const C: i32 = 8;
const TEXT: i32 = 6;

pub(super) fn tiny_config() -> ErnieTransformerConfig {
    ErnieTransformerConfig {
        hidden_size: D,
        ffn_hidden_size: 32,
        in_channels: C,
        out_channels: C,
        num_layers: 2,
        num_attention_heads: 2,
        patch_size: 1,
        qk_layernorm: true,
        rope_axes_dim: vec![2, 2, 4],
        rope_theta: 256.0,
        text_in_dim: TEXT,
        eps: 1e-6,
    }
}

fn randn(seed: u64, shape: &[i32], scale: f32) -> UniquePtr<MlxArray> {
    let key = mlxcel_core::random_key(seed);
    // SAFETY: `key` is alive for the call.
    let x = unsafe { mlxcel_core::random_normal(shape, dtype::FLOAT32, &*key) };
    mlxcel_core::multiply(&x, &mlxcel_core::full_f32(&[1], scale, dtype::FLOAT32))
}

pub(super) fn tiny_weights() -> WeightMap {
    let mut w = WeightMap::new();
    let mut seed = 100_u64;
    let mut put = |w: &mut WeightMap, key: &str, shape: &[i32], scale: f32| {
        seed += 1;
        w.insert(key.to_string(), randn(seed, shape, scale));
    };
    put(&mut w, "x_embedder.proj.weight", &[D, 1, 1, C], 0.3);
    put(&mut w, "x_embedder.proj.bias", &[D], 0.1);
    put(&mut w, "text_proj.weight", &[D, TEXT], 0.3);
    for l in ["time_embedding.linear_1", "time_embedding.linear_2"] {
        put(&mut w, &format!("{l}.weight"), &[D, D], 0.05);
        put(&mut w, &format!("{l}.bias"), &[D], 0.1);
    }
    put(&mut w, "adaln_modulation.weight", &[6 * D, D], 0.1);
    put(&mut w, "adaln_modulation.bias", &[6 * D], 0.1);
    for i in 0..2 {
        let p = format!("layers.{i}");
        for n in ["adaLN_sa_ln", "adaLN_mlp_ln"] {
            w.insert(
                format!("{p}.{n}.weight"),
                mlxcel_core::ones(&[D], dtype::FLOAT32),
            );
        }
        for n in ["to_q", "to_k", "to_v", "to_out"] {
            put(
                &mut w,
                &format!("{p}.self_attention.{n}.weight"),
                &[D, D],
                0.25,
            );
        }
        for n in ["norm_q", "norm_k"] {
            w.insert(
                format!("{p}.self_attention.{n}.weight"),
                mlxcel_core::ones(&[8], dtype::FLOAT32),
            );
        }
        put(&mut w, &format!("{p}.mlp.gate_proj.weight"), &[32, D], 0.25);
        put(&mut w, &format!("{p}.mlp.up_proj.weight"), &[32, D], 0.25);
        put(
            &mut w,
            &format!("{p}.mlp.linear_fc2.weight"),
            &[D, 32],
            0.25,
        );
    }
    put(&mut w, "final_norm.linear.weight", &[2 * D, D], 0.1);
    put(&mut w, "final_norm.linear.bias", &[2 * D], 0.1);
    put(&mut w, "final_linear.weight", &[C, D], 0.25);
    put(&mut w, "final_linear.bias", &[C], 0.1);
    w
}

#[test]
fn positions_put_image_tokens_after_the_text_length() {
    let p = values(&positions(2, 1, 2, 3, &[1, 3]));
    // Row 0: two image cells at (1, 0, x), then text (i, 0, 0).
    assert_eq!(
        &p[..15],
        &[1., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 2., 0., 0.]
    );
    // Row 1 uses its own text length.
    assert_eq!(&p[15..21], &[3., 0., 0., 3., 0., 1.]);
}

#[test]
fn the_key_mask_hides_only_padding() {
    let m = values(&key_mask(2, 2, 3, &[1, 3], dtype::FLOAT32));
    assert_eq!(
        &m[..5],
        &[0.0, 0.0, 0.0, f32::NEG_INFINITY, f32::NEG_INFINITY]
    );
    assert_eq!(&m[5..], &[0.0; 5]);
}

#[test]
fn forward_shape_and_padding_never_reaches_a_real_row() {
    let t = ErnieTransformer::from_weights(tiny_config(), &tiny_weights()).unwrap();
    let latents = randn(1, &[2, C, 3, 4], 1.0);
    let timestep = mlxcel_core::from_slice_f32(&[900.0, 900.0], &[2]);
    let text_a = randn(2, &[2, 4, TEXT], 1.0);
    // Same valid tokens in row 0 (2 of 4), different padding content.
    let text_b = {
        let other = randn(3, &[2, 4, TEXT], 1.0);
        let keep = mlxcel_core::slice(&text_a, &[0, 0, 0], &[1, 2, TEXT]);
        let pad = mlxcel_core::slice(&other, &[0, 2, 0], &[1, 4, TEXT]);
        let row0 = mlxcel_core::concatenate(&keep, &pad, 1);
        let row1 = mlxcel_core::slice(&text_a, &[1, 0, 0], &[2, 4, TEXT]);
        mlxcel_core::concatenate(&row0, &row1, 0)
    };
    let a = t.forward(&latents, &timestep, &text_a, &[2, 4]).unwrap();
    assert_eq!(mlxcel_core::array_shape(&a), [2, C, 3, 4]);
    let b = t.forward(&latents, &timestep, &text_b, &[2, 4]).unwrap();
    let row = |x: &MlxArray| values(&mlxcel_core::slice(x, &[0, 0, 0, 0], &[1, C, 3, 4]));
    assert_eq!(row(&a), row(&b), "padded text tokens changed a real row");
    // Without the mask the padding would matter.
    let c = t.forward(&latents, &timestep, &text_b, &[4, 4]).unwrap();
    assert_ne!(row(&a), row(&c));
}

#[test]
fn bad_inputs_are_errors() {
    let t = ErnieTransformer::from_weights(tiny_config(), &tiny_weights()).unwrap();
    let latents = randn(1, &[1, C, 2, 2], 1.0);
    let timestep = mlxcel_core::from_slice_f32(&[900.0], &[1]);
    let text = randn(2, &[1, 3, TEXT], 1.0);
    assert!(t.forward(&latents, &timestep, &text, &[4]).is_err());
    assert!(t.forward(&latents, &timestep, &text, &[1, 1]).is_err());
    let mut config = tiny_config();
    config.patch_size = 2;
    assert!(ErnieTransformer::from_weights(config, &tiny_weights()).is_err());
}
