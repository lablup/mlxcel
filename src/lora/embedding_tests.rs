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

//! Unit tests for the embedding pair layouts (issue #2421).

use super::*;

fn values(array: &MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(array)
}

#[test]
fn embedding_targets_are_recognised_by_their_last_path_segment() {
    assert!(is_input_embedding_weight("model.embed_tokens.weight"));
    assert!(is_input_embedding_weight(
        "language_model.model.embed_tokens.weight"
    ));
    assert!(is_input_embedding_weight("embed_tokens.weight"));
    assert!(!is_input_embedding_weight("lm_head.weight"));
    assert!(!is_input_embedding_weight("model.embed_tokens_norm.weight"));
    assert!(!is_input_embedding_weight(
        "model.layers.0.self_attn.q_proj.weight"
    ));
}

#[test]
fn each_layout_is_detected_against_the_vocabulary() {
    // vocab 32, hidden 8, rank 2.
    assert_eq!(
        detect_embedding_layout("e", &[32, 2], &[2, 8], 32).unwrap(),
        (EmbeddingPairLayout::MlxLmEmbedding, 8)
    );
    assert_eq!(
        detect_embedding_layout("e", &[8, 2], &[2, 32], 32).unwrap(),
        (EmbeddingPairLayout::LinearMlxLm, 8)
    );
    assert_eq!(
        detect_embedding_layout("e", &[2, 8], &[32, 2], 32).unwrap(),
        (EmbeddingPairLayout::LinearPeft, 8)
    );
    let err = detect_embedding_layout("e", &[7, 2], &[2, 8], 32)
        .unwrap_err()
        .to_string();
    assert!(err.contains("input embedding of 32 rows"), "{err}");
    let err = detect_embedding_layout("e", &[4, 32, 2], &[2, 8], 32)
        .unwrap_err()
        .to_string();
    assert!(err.contains("expected 2-D matrices"), "{err}");
}

#[test]
fn a_square_table_reads_the_mlx_lm_layout() {
    // vocab == hidden: `a = [16, 2]`, `b = [2, 16]` fits both the mlx-lm
    // embedding layout and the linear one; the mlx-lm reading wins, so the
    // delta is `a @ b`, not its transpose.
    assert_eq!(
        detect_embedding_layout("e", &[16, 2], &[2, 16], 16).unwrap(),
        (EmbeddingPairLayout::MlxLmEmbedding, 16)
    );
    let base = mlxcel_core::from_slice_f32(&[0.0; 4], &[2, 2]);
    // a = [[1], [0]], b = [[0, 1]]: a @ b = [[0, 1], [0, 0]], the transpose
    // would be [[0, 0], [1, 0]].
    let a = mlxcel_core::from_slice_f32(&[1.0, 0.0], &[2, 1]);
    let b = mlxcel_core::from_slice_f32(&[0.0, 1.0], &[1, 2]);
    let fused = fuse_embedding_pair("e", "e.weight", &base, &a, &b, 1.0).unwrap();
    assert_eq!(values(&fused), vec![0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn every_layout_canonicalizes_to_the_same_delta() {
    // A = [3, 1] (vocab 3, rank 1), B = [1, 2] (hidden 2): delta = A @ B.
    let a_vals = [1.0f32, 2.0, 3.0];
    let b_vals = [10.0f32, 20.0];
    let expected = vec![10.0, 20.0, 20.0, 40.0, 30.0, 60.0];
    let base = mlxcel_core::from_slice_f32(&[0.0; 6], &[3, 2]);
    let cases = [
        // mlx-lm embedding: a = A, b = B.
        (
            mlxcel_core::from_slice_f32(&a_vals, &[3, 1]),
            mlxcel_core::from_slice_f32(&b_vals, &[1, 2]),
        ),
        // linear, mlx-lm orientation: a = Bᵀ [2, 1], b = Aᵀ [1, 3].
        (
            mlxcel_core::from_slice_f32(&b_vals, &[2, 1]),
            mlxcel_core::from_slice_f32(&a_vals, &[1, 3]),
        ),
        // linear, PEFT orientation: a = B [1, 2], b = A [3, 1].
        (
            mlxcel_core::from_slice_f32(&b_vals, &[1, 2]),
            mlxcel_core::from_slice_f32(&a_vals, &[3, 1]),
        ),
    ];
    for (lora_a, lora_b) in &cases {
        let fused = fuse_embedding_pair("e", "e.weight", &base, lora_a, lora_b, 1.0).unwrap();
        assert_eq!(values(&fused), expected);
        let (layout, _) = detect_embedding_layout(
            "e",
            &mlxcel_core::array_shape(lora_a),
            &mlxcel_core::array_shape(lora_b),
            3,
        )
        .unwrap();
        let (a, b) = canonical_embedding_factors(lora_a, lora_b, layout);
        assert_eq!(mlxcel_core::array_shape(&a), vec![3, 1]);
        assert_eq!(mlxcel_core::array_shape(&b), vec![1, 2]);
        assert_eq!(values(&mlxcel_core::matmul(&a, &b)), expected);
    }
}

#[test]
fn the_fused_table_keeps_the_base_dtype_and_applies_the_scale() {
    // mlx_lm.fuse writes `W + (scale * a @ b).astype(W.dtype)`; a fused linear
    // promotes to f32, an embedding table must not.
    let base = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&[1.0; 4], &[2, 2]),
        mlxcel_core::dtype::BFLOAT16,
    );
    let a = mlxcel_core::from_slice_f32(&[1.0, 1.0], &[2, 1]);
    let b = mlxcel_core::from_slice_f32(&[1.0, 2.0], &[1, 2]);
    let fused = fuse_embedding_pair("e", "e.weight", &base, &a, &b, 0.5).unwrap();
    assert_eq!(
        mlxcel_core::array_dtype(&fused),
        mlxcel_core::dtype::BFLOAT16
    );
    assert_eq!(values(&fused), vec![1.5, 2.0, 1.5, 2.0]);
}

#[test]
fn a_packed_table_is_not_fused() {
    // A quantized table's packed columns hide the hidden width, so the fused
    // path keeps its dense-base-only limit with a shape error.
    let packed = mlxcel_core::from_slice_f32(&[0.0; 6], &[3, 2]);
    let a = mlxcel_core::from_slice_f32(&[1.0; 3], &[3, 1]);
    let b = mlxcel_core::from_slice_f32(&[1.0; 16], &[1, 16]);
    let Err(err) = fuse_embedding_pair("e", "e.weight", &packed, &a, &b, 1.0) else {
        panic!("a packed table must not fuse");
    };
    let err = err.to_string();
    assert!(err.contains("LoRA shape mismatch"), "{err}");
}

#[test]
fn the_fusion_loop_refuses_a_quantized_embedding_table_by_its_scales() {
    // A delta whose width equals the packed column count would otherwise pass
    // the shape check and be added to the packed words.
    let mut base = mlxcel_core::weights::WeightMap::new();
    base.insert(
        "model.embed_tokens.weight".to_string(),
        mlxcel_core::from_slice_f32(&[0.0; 6], &[3, 2]),
    );
    base.insert(
        "model.embed_tokens.scales".to_string(),
        mlxcel_core::from_slice_f32(&[1.0; 3], &[3, 1]),
    );
    let mut adapter = mlxcel_core::weights::WeightMap::new();
    adapter.insert(
        "model.embed_tokens.lora_a".to_string(),
        mlxcel_core::from_slice_f32(&[1.0; 3], &[3, 1]),
    );
    adapter.insert(
        "model.embed_tokens.lora_b".to_string(),
        mlxcel_core::from_slice_f32(&[1.0; 2], &[1, 2]),
    );
    let err = crate::lora::fuse_lora_weights_into(&mut base, &adapter, 1.0)
        .expect_err("a quantized table must not fuse")
        .to_string();
    assert!(err.contains("quantized embedding table"), "{err}");
    assert!(err.contains("--lora"), "{err}");
}
