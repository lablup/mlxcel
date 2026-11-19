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

fn zeros(shape: &[i32]) -> UniquePtr<MlxArray> {
    let len: i32 = shape.iter().product();
    mlxcel_core::from_slice_f32(&vec![0.0; len as usize], shape)
}

fn tiny_config() -> TransformerConfig {
    TransformerConfig {
        dim: 16,
        ffn_dim: 32,
        num_heads: 2,
        num_layers: 23,
        text_dim: 12,
        freq_dim: 8,
        latent_channels: 4,
        rope_base: 10000.0,
    }
}

#[test]
fn the_rope_table_covers_image_then_text_tokens() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    // The shape of the PR's original out-of-bounds panic, scaled down.
    let (sin, cos) = rope_table(4, 3, 5, 16, 10000.0, dtype::FLOAT32);
    assert_eq!(mlxcel_core::array_shape(&sin), vec![1, 17, 1, 16]);
    assert_eq!(mlxcel_core::array_shape(&cos), vec![1, 17, 1, 16]);
}

#[test]
fn a_quantized_projection_reports_its_logical_input_width() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let mut w = WeightMap::new();
    // 8-bit packs four values per u32 word: 64 inputs -> 16 words.
    w.insert("p.weight".to_string(), zeros(&[32, 16]));
    w.insert("p.scales".to_string(), zeros(&[32, 1]));
    let quant = LinearQuant {
        group_size: 64,
        bits: 8,
        mode: "affine",
    };
    assert_eq!(linear_dims(&w, "p", Some(quant)).unwrap(), (32, 64));
    let err = linear_dims(&w, "p", None).unwrap_err().to_string();
    assert!(err.contains("no quantization metadata"), "{err}");
}

#[test]
fn a_mis_shaped_projection_is_an_error_not_an_abort() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let mut w = WeightMap::new();
    w.insert("video_embedder.weight".to_string(), zeros(&[16, 5]));
    let err = MossTransformer::load(&w, tiny_config(), None)
        .err()
        .expect("a 5-channel embedder under a 4-channel config must be refused")
        .to_string();
    assert!(err.contains("video_embedder"), "{err}");
    assert!(err.contains("expected [16, 4]"), "{err}");
}

#[test]
fn a_norm_of_the_wrong_width_is_refused() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let mut w = WeightMap::new();
    w.insert("b.norm1.weight".to_string(), zeros(&[16]));
    w.insert("b.norm2.weight".to_string(), zeros(&[16]));
    w.insert("b.q_norm.weight".to_string(), zeros(&[16]));
    let err = expect_block_norms(&w, "b", 16, 8).unwrap_err().to_string();
    assert!(err.contains("b.q_norm.weight"), "{err}");
}
