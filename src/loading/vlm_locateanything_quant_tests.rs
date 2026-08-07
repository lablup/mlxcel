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

//! Unit tests for the LocateAnything mixed-precision QKV normalization.

use super::*;

fn one() -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    mlxcel_core::ones(&[1], mlxcel_core::dtype::FLOAT32)
}

// --- mixed 4/8-bit fused-QKV normalization -----------------------------

const GROUP: i32 = 64;

/// A deterministic dense plane of shape `[out, GROUP]`.
fn dense_plane(out: i32, seed: i32) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let n = (out * GROUP) as usize;
    let data: Vec<f32> = (0..n)
        .map(|i| (((i as i32 * 37 + seed * 11) % 97) as f32 - 48.0) * 0.01)
        .collect();
    mlxcel_core::from_slice_f32(&data, &[out, GROUP])
}

/// Insert `{prefix}.{weight,scales,biases}` quantized at `bits`.
fn insert_quantized(weights: &mut WeightMap, prefix: &str, out: i32, seed: i32, bits: i32) {
    let dense = dense_plane(out, seed);
    let q = mlxcel_core::quantize_weights_with_mode(&dense, GROUP, bits, "affine");
    weights.insert(
        format!("{prefix}.weight"),
        mlxcel_core::quantized_weights_w(&q),
    );
    weights.insert(
        format!("{prefix}.scales"),
        mlxcel_core::quantized_weights_scales(&q),
    );
    if mlxcel_core::quantized_weights_has_biases(&q) {
        weights.insert(
            format!("{prefix}.biases"),
            mlxcel_core::quantized_weights_biases(&q),
        );
    }
}

/// Dequantize `{prefix}` back to dense using the width its shapes imply.
fn dequantize_plane(
    weights: &WeightMap,
    prefix: &str,
) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let w = weights.get(&format!("{prefix}.weight")).unwrap();
    let s = weights.get(&format!("{prefix}.scales")).unwrap();
    let layout = mlxcel_core::layers::reconcile_quantization_layout(
        &mlxcel_core::array_shape(w),
        &mlxcel_core::array_shape(s),
        GROUP,
        4,
        "affine",
    )
    .expect("layout");
    let b = weights.get(&format!("{prefix}.biases"));
    let b_ptr = match b {
        Some(b) => b.as_ref().map(|r| r as *const mlxcel_core::MlxArray),
        None => None,
    }
    .unwrap_or(std::ptr::null());
    // SAFETY: both arrays outlive the call; `b_ptr` is null or borrowed live.
    unsafe { mlxcel_core::dequantize(w, s, b_ptr, GROUP, layout.bits, "affine") }
}

fn max_abs_diff(a: &mlxcel_core::MlxArray, b: &mlxcel_core::MlxArray) -> f32 {
    let d = mlxcel_core::subtract(a, b);
    let d = mlxcel_core::abs(&d);
    let m = mlxcel_core::max_all(&d);
    mlxcel_core::eval(&m);
    mlxcel_core::item_f32(&m)
}

/// The released checkpoint is `mixed_4_8`: layer 0 stores `q_proj`/`k_proj` at
/// 4 bits and `v_proj` at 8. Concatenating those packed planes along axis 0 is
/// a hard shape error inside MLX, so the loader must dequantize the layer's
/// three planes first.
#[test]
fn densifies_mixed_precision_qkv_layers() {
    let prefix = "language_model.model.layers.0.self_attn";
    let mut weights = WeightMap::new();
    insert_quantized(&mut weights, &format!("{prefix}.q_proj"), 8, 1, 4);
    insert_quantized(&mut weights, &format!("{prefix}.k_proj"), 4, 2, 4);
    insert_quantized(&mut weights, &format!("{prefix}.v_proj"), 4, 3, 8);
    // The real checkpoint also carries a true linear bias on q/k/v; it must
    // survive because Qwen2 attention needs it.
    weights.insert(format!("{prefix}.q_proj.bias"), one());

    let before_q = dequantize_plane(&weights, &format!("{prefix}.q_proj"));
    let before_k = dequantize_plane(&weights, &format!("{prefix}.k_proj"));
    let before_v = dequantize_plane(&weights, &format!("{prefix}.v_proj"));
    mlxcel_core::eval(&before_q);
    mlxcel_core::eval(&before_k);
    mlxcel_core::eval(&before_v);

    let converted =
        densify_mixed_precision_qkv(&mut weights, GROUP, 4, "affine").expect("normalize");
    assert_eq!(converted, 1, "exactly one layer needed converting");

    for proj in ["q_proj", "k_proj", "v_proj"] {
        // All three become dense so `FusedQKVLinear` takes its non-quantized
        // branch (it decides that from `q_proj.scales` alone).
        assert!(
            !weights.contains_key(&format!("{prefix}.{proj}.scales")),
            "{proj} must no longer look quantized"
        );
        assert!(!weights.contains_key(&format!("{prefix}.{proj}.biases")));
        let w = weights.get(&format!("{prefix}.{proj}.weight")).unwrap();
        assert_eq!(
            mlxcel_core::array_shape(w).last().copied(),
            Some(GROUP),
            "{proj} must be dense with the unpacked in-features width"
        );
    }
    assert!(
        weights.contains_key(&format!("{prefix}.q_proj.bias")),
        "the true linear bias must survive"
    );

    // Dequantization is the stored representation's definition, so no value
    // the model computes with changes.
    let after_q = weights.get(&format!("{prefix}.q_proj.weight")).unwrap();
    let after_k = weights.get(&format!("{prefix}.k_proj.weight")).unwrap();
    let after_v = weights.get(&format!("{prefix}.v_proj.weight")).unwrap();
    assert_eq!(
        max_abs_diff(&before_q, after_q),
        0.0,
        "q_proj must be exact"
    );
    assert_eq!(
        max_abs_diff(&before_k, after_k),
        0.0,
        "k_proj must be exact"
    );
    assert_eq!(
        max_abs_diff(&before_v, after_v),
        0.0,
        "v_proj must be exact"
    );
}

/// Positive control: a layer whose q/k/v already agree must stay packed, so the
/// guard cannot pass by dequantizing everything.
#[test]
fn leaves_a_uniform_precision_layer_quantized() {
    let prefix = "language_model.model.layers.4.self_attn";
    let mut weights = WeightMap::new();
    for (proj, out, seed) in [("q_proj", 8, 1), ("k_proj", 4, 2), ("v_proj", 4, 3)] {
        insert_quantized(&mut weights, &format!("{prefix}.{proj}"), out, seed, 4);
    }

    let converted =
        densify_mixed_precision_qkv(&mut weights, GROUP, 4, "affine").expect("normalize");
    assert_eq!(converted, 0, "a uniform layer needs no conversion");

    for proj in ["q_proj", "k_proj", "v_proj"] {
        assert!(weights.contains_key(&format!("{prefix}.{proj}.scales")));
        let w = weights.get(&format!("{prefix}.{proj}.weight")).unwrap();
        assert_eq!(
            mlxcel_core::array_shape(w).last().copied(),
            Some(8),
            "{proj} must stay packed at 4 bits (GROUP * 4 / 32 = 8)"
        );
    }
}

/// A layer whose q/k/v are all 8-bit is also uniform and must stay packed.
#[test]
fn leaves_a_uniform_8bit_layer_quantized() {
    let prefix = "language_model.model.layers.7.self_attn";
    let mut weights = WeightMap::new();
    for (proj, out, seed) in [("q_proj", 8, 1), ("k_proj", 4, 2), ("v_proj", 4, 3)] {
        insert_quantized(&mut weights, &format!("{prefix}.{proj}"), out, seed, 8);
    }
    let converted =
        densify_mixed_precision_qkv(&mut weights, GROUP, 4, "affine").expect("normalize");
    assert_eq!(converted, 0);
    assert!(weights.contains_key(&format!("{prefix}.q_proj.scales")));
}

/// A non-quantized attention layer has no `.scales` and must be skipped rather
/// than treated as a degenerate quantized triple.
#[test]
fn skips_layers_without_quantized_planes() {
    let mut weights = WeightMap::new();
    weights.insert(
        "language_model.model.layers.0.self_attn.q_proj.weight".to_string(),
        one(),
    );
    let converted =
        densify_mixed_precision_qkv(&mut weights, GROUP, 4, "affine").expect("normalize");
    assert_eq!(converted, 0);
}
