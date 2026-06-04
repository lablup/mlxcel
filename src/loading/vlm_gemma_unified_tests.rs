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

//! Unit tests for the Gemma 4 Unified loader sanitize remaps.

use super::{normalize_gemma4_unified_key, sanitize_gemma4_unified_weights};
use mlxcel_core::dtype;
use mlxcel_core::utils::slice_axis;
use mlxcel_core::weights::WeightMap;

/// MLX affine-quantization parameters for the synthetic fused-MoE fixtures.
const Q_BITS: i32 = 4;
const Q_GROUP_SIZE: i32 = 32;

/// Build a synthetic quantized fused `gate_up_proj` triplet in the on-disk
/// layout the runtime expects: `weight`/`scales`/`biases` shaped
/// `[num_experts, 2*ffn, …]` with MLX affine grouping along the **last**
/// (input/contract) axis. The doubled output axis (`2*ffn`) is axis 1, which is
/// the axis the sanitize split must partition.
///
/// Shapes for `(num_experts=e, doubled=2*ffn, in_features)` with `bits`/`gs`:
/// * `weight`  (packed u32): `[e, 2*ffn, in_features * bits / 32]`
/// * `scales`  (f32):        `[e, 2*ffn, in_features / gs]`
/// * `biases`  (f32):        `[e, 2*ffn, in_features / gs]`
///
/// Values are deterministic but **distinct per output row** (and per expert) so
/// that a wrong-axis or group-straddling split would change the dequantized
/// result, making the equivalence gate non-vacuous.
fn make_quantized_gate_up(
    e: i32,
    doubled: i32,
    in_features: i32,
) -> (
    mlxcel_core::UniquePtr<mlxcel_core::MlxArray>,
    mlxcel_core::UniquePtr<mlxcel_core::MlxArray>,
    mlxcel_core::UniquePtr<mlxcel_core::MlxArray>,
) {
    let packed_cols = (in_features * Q_BITS / 32) as usize;
    let groups = (in_features / Q_GROUP_SIZE) as usize;

    let mut weight_data: Vec<u32> = Vec::with_capacity(e as usize * doubled as usize * packed_cols);
    let mut scales_data: Vec<f32> = Vec::with_capacity(e as usize * doubled as usize * groups);
    let mut biases_data: Vec<f32> = Vec::with_capacity(e as usize * doubled as usize * groups);

    for ei in 0..e as u32 {
        for r in 0..doubled as u32 {
            // Distinct packed bit pattern per (expert, output-row, column).
            for c in 0..packed_cols as u32 {
                weight_data.push(0x1234_5678u32.wrapping_add(ei * 131 + r * 17 + c));
            }
            // Distinct, non-trivial scale/bias per (expert, output-row, group).
            for g in 0..groups as u32 {
                scales_data
                    .push(0.05 + 0.01 * (ei as f32) + 0.001 * (r as f32) + 0.0003 * g as f32);
                biases_data
                    .push(-0.2 + 0.02 * (ei as f32) - 0.003 * (r as f32) + 0.0007 * g as f32);
            }
        }
    }

    let weight = mlxcel_core::from_slice_u32(&weight_data, &[e, doubled, packed_cols as i32]);
    let scales = mlxcel_core::from_slice_f32(&scales_data, &[e, doubled, groups as i32]);
    let biases = mlxcel_core::from_slice_f32(&biases_data, &[e, doubled, groups as i32]);
    (weight, scales, biases)
}

/// Affine-dequantize a quantized triplet to full precision (f32).
fn dequant(
    weight: &mlxcel_core::MlxArray,
    scales: &mlxcel_core::MlxArray,
    biases: &mlxcel_core::MlxArray,
) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    unsafe {
        mlxcel_core::dequantize(
            weight,
            scales,
            biases as *const _,
            Q_GROUP_SIZE,
            Q_BITS,
            "affine",
        )
    }
}

/// Max absolute difference between two arrays as an f32 scalar.
fn max_abs_diff(a: &mlxcel_core::MlxArray, b: &mlxcel_core::MlxArray) -> f32 {
    let diff = mlxcel_core::subtract(a, b);
    let abs = mlxcel_core::abs(&diff);
    let max = mlxcel_core::max_all(&abs);
    mlxcel_core::item_f32(&max)
}

#[test]
fn unified_key_normalization_prefixes() {
    // Bare language_model.X -> language_model.model.X.
    assert_eq!(
        normalize_gemma4_unified_key("language_model.layers.0.input_layernorm.weight"),
        "language_model.model.layers.0.input_layernorm.weight"
    );
    // Already-normalized key is left untouched.
    assert_eq!(
        normalize_gemma4_unified_key("language_model.model.norm.weight"),
        "language_model.model.norm.weight"
    );
    // model.language_model.X -> language_model.model.X.
    assert_eq!(
        normalize_gemma4_unified_key("model.language_model.norm.weight"),
        "language_model.model.norm.weight"
    );
    // Leading model. is stripped from the multimodal prefixes.
    assert_eq!(
        normalize_gemma4_unified_key("model.vision_embedder.patch_dense.weight"),
        "vision_embedder.patch_dense.weight"
    );
    assert_eq!(
        normalize_gemma4_unified_key("model.embed_vision.embedding_projection.weight"),
        "embed_vision.embedding_projection.weight"
    );
    // Already-clean multimodal keys are untouched.
    assert_eq!(
        normalize_gemma4_unified_key("vision_embedder.pos_embedding"),
        "vision_embedder.pos_embedding"
    );
}

#[test]
fn unified_sanitize_drops_rotary_and_lm_head() {
    let mut raw = WeightMap::new();
    raw.insert(
        "language_model.model.layers.0.self_attn.rotary_emb.inv_freq".to_string(),
        mlxcel_core::ones(&[8], dtype::FLOAT32),
    );
    raw.insert(
        "lm_head.weight".to_string(),
        mlxcel_core::ones(&[4, 4], dtype::FLOAT32),
    );
    raw.insert(
        "vision_embedder.pos_embedding".to_string(),
        mlxcel_core::ones(&[1120, 2, 8], dtype::FLOAT32),
    );

    let out = sanitize_gemma4_unified_weights(raw, true);
    assert!(!out.contains_key("language_model.model.layers.0.self_attn.rotary_emb.inv_freq"));
    assert!(!out.contains_key("lm_head.weight"));
    assert!(out.contains_key("vision_embedder.pos_embedding"));
}

#[test]
fn unified_sanitize_drops_audio_when_absent() {
    let mut raw = WeightMap::new();
    raw.insert(
        "embed_audio.embedding_projection.weight".to_string(),
        mlxcel_core::ones(&[4, 4], dtype::FLOAT32),
    );
    raw.insert(
        "embed_vision.embedding_projection.weight".to_string(),
        mlxcel_core::ones(&[4, 4], dtype::FLOAT32),
    );

    // has_audio = false drops embed_audio.*, keeps embed_vision.*.
    let out = sanitize_gemma4_unified_weights(raw, false);
    assert!(!out.contains_key("embed_audio.embedding_projection.weight"));
    assert!(out.contains_key("embed_vision.embedding_projection.weight"));
}

#[test]
fn unified_sanitize_splits_moe_switch_glu() {
    // Fused experts.gate_up_proj [num_experts=2, in=3, 2*ffn=8] splits into
    // gate/up [2, 4, 3] (axes swapped, doubled dim halved); down_proj renamed.
    let mut raw = WeightMap::new();
    raw.insert(
        "language_model.model.layers.0.mlp.experts.gate_up_proj".to_string(),
        mlxcel_core::ones(&[2, 3, 8], dtype::FLOAT32),
    );
    raw.insert(
        "language_model.model.layers.0.mlp.experts.down_proj".to_string(),
        mlxcel_core::ones(&[2, 4, 3], dtype::FLOAT32),
    );

    let out = sanitize_gemma4_unified_weights(raw, true);

    assert!(!out.contains_key("language_model.model.layers.0.mlp.experts.gate_up_proj"));
    let gate = out
        .get("language_model.model.layers.0.mlp.experts.switch_glu.gate_proj.weight")
        .expect("gate_proj split present");
    let up = out
        .get("language_model.model.layers.0.mlp.experts.switch_glu.up_proj.weight")
        .expect("up_proj split present");
    assert_eq!(mlxcel_core::array_shape(gate), vec![2, 4, 3]);
    assert_eq!(mlxcel_core::array_shape(up), vec![2, 4, 3]);
    assert!(
        out.contains_key("language_model.model.layers.0.mlp.experts.switch_glu.down_proj.weight"),
        "down_proj renamed under switch_glu"
    );
}

#[test]
fn unified_sanitize_moe_split_normalizes_model_prefix() {
    // A `model.`-prefixed fused expert key must land under the normalized
    // `language_model.model.…` namespace, like every other sanitized tensor —
    // the split branch runs its output keys through `normalize_gemma4_unified_key`.
    let mut raw = WeightMap::new();
    raw.insert(
        "model.language_model.layers.0.mlp.experts.gate_up_proj".to_string(),
        mlxcel_core::ones(&[2, 3, 8], dtype::FLOAT32),
    );
    raw.insert(
        "model.language_model.layers.0.mlp.experts.down_proj".to_string(),
        mlxcel_core::ones(&[2, 4, 3], dtype::FLOAT32),
    );

    let out = sanitize_gemma4_unified_weights(raw, true);

    // Split keys are normalized (no leftover raw `model.language_model.` prefix).
    assert!(
        out.contains_key("language_model.model.layers.0.mlp.experts.switch_glu.gate_proj.weight"),
        "gate_proj split present under normalized prefix"
    );
    assert!(
        out.contains_key("language_model.model.layers.0.mlp.experts.switch_glu.up_proj.weight"),
        "up_proj split present under normalized prefix"
    );
    assert!(
        out.contains_key("language_model.model.layers.0.mlp.experts.switch_glu.down_proj.weight"),
        "down_proj renamed under normalized prefix"
    );
    assert!(
        !out.keys().any(|k| k.starts_with("model.language_model.")),
        "no raw model.language_model.* prefix should survive the split"
    );
}

#[test]
fn unified_sanitize_splits_quantized_moe_switch_glu() {
    // Quantized fused triplet: gate_up_proj.{weight,scales,biases} in the
    // [num_experts=2, 2*ffn=8, in/…] on-disk layout, plus a down_proj triplet.
    // `in_features = 64` ⇒ packed weight cols = 64*4/32 = 8, groups = 64/32 = 2.
    let (gu_w, gu_s, gu_b) = make_quantized_gate_up(2, 8, 64);
    let prefix = "language_model.model.layers.0.mlp.experts";

    let mut raw = WeightMap::new();
    raw.insert(format!("{prefix}.gate_up_proj.weight"), gu_w);
    raw.insert(format!("{prefix}.gate_up_proj.scales"), gu_s);
    raw.insert(format!("{prefix}.gate_up_proj.biases"), gu_b);
    // down_proj triplet (renamed, not split): [num_experts=2, ffn=4, in_packed].
    raw.insert(
        format!("{prefix}.down_proj.weight"),
        mlxcel_core::from_slice_u32(&vec![0u32; 2 * 4 * 8], &[2, 4, 8]),
    );
    raw.insert(
        format!("{prefix}.down_proj.scales"),
        mlxcel_core::ones(&[2, 4, 2], dtype::FLOAT32),
    );
    raw.insert(
        format!("{prefix}.down_proj.biases"),
        mlxcel_core::ones(&[2, 4, 2], dtype::FLOAT32),
    );

    let out = sanitize_gemma4_unified_weights(raw, true);

    // Fused/bare keys must be gone for every component.
    for comp in [".weight", ".scales", ".biases"] {
        assert!(
            !out.contains_key(&format!("{prefix}.gate_up_proj{comp}")),
            "fused gate_up_proj{comp} should be split away"
        );
    }

    // gate/up split: weight [2,4,8], scales/biases [2,4,2] (output axis halved).
    for proj in ["gate_proj", "up_proj"] {
        let w = out
            .get(&format!("{prefix}.switch_glu.{proj}.weight"))
            .unwrap_or_else(|| panic!("{proj}.weight present"));
        let s = out
            .get(&format!("{prefix}.switch_glu.{proj}.scales"))
            .unwrap_or_else(|| panic!("{proj}.scales present"));
        let b = out
            .get(&format!("{prefix}.switch_glu.{proj}.biases"))
            .unwrap_or_else(|| panic!("{proj}.biases present"));
        assert_eq!(mlxcel_core::array_shape(w), vec![2, 4, 8], "{proj}.weight");
        assert_eq!(mlxcel_core::array_shape(s), vec![2, 4, 2], "{proj}.scales");
        assert_eq!(mlxcel_core::array_shape(b), vec![2, 4, 2], "{proj}.biases");
    }

    // down_proj triplet is renamed unchanged (no split).
    for comp in [".weight", ".scales", ".biases"] {
        assert!(
            out.contains_key(&format!("{prefix}.switch_glu.down_proj{comp}")),
            "down_proj{comp} renamed under switch_glu"
        );
        assert!(
            !out.contains_key(&format!("{prefix}.down_proj{comp}")),
            "original down_proj{comp} removed"
        );
    }
}

#[test]
fn unified_sanitize_quantized_split_dequant_equivalence() {
    // Numerical gate: splitting the quantized fused tensor along the output axis
    // must be lossless and axis-aligned, i.e.
    //     dequantize(split(W)) == split(dequantize(W))   (along output axis 1).
    // This proves the sanitize split partitions weight/scales/biases at the same
    // half boundary with no group straddling — without needing a real
    // quantized-MoE gemma4_unified checkpoint.
    let (gu_w, gu_s, gu_b) = make_quantized_gate_up(2, 8, 64);

    // Reference: dequantize the FULL fused tensor, then split along axis 1.
    let full = dequant(&gu_w, &gu_s, &gu_b); // [2, 8, 64]
    let half = mlxcel_core::array_shape(&full)[1] / 2; // 4
    let ref_gate = slice_axis(&full, 1, 0, half); // [2, 4, 64]
    let ref_up = slice_axis(&full, 1, half, -1); // [2, 4, 64]

    // Under test: run sanitize, then dequantize each split leg.
    let prefix = "language_model.model.layers.0.mlp.experts";
    let mut raw = WeightMap::new();
    raw.insert(format!("{prefix}.gate_up_proj.weight"), gu_w);
    raw.insert(format!("{prefix}.gate_up_proj.scales"), gu_s);
    raw.insert(format!("{prefix}.gate_up_proj.biases"), gu_b);
    let out = sanitize_gemma4_unified_weights(raw, true);

    let gate = dequant(
        out.get(&format!("{prefix}.switch_glu.gate_proj.weight"))
            .unwrap(),
        out.get(&format!("{prefix}.switch_glu.gate_proj.scales"))
            .unwrap(),
        out.get(&format!("{prefix}.switch_glu.gate_proj.biases"))
            .unwrap(),
    );
    let up = dequant(
        out.get(&format!("{prefix}.switch_glu.up_proj.weight"))
            .unwrap(),
        out.get(&format!("{prefix}.switch_glu.up_proj.scales"))
            .unwrap(),
        out.get(&format!("{prefix}.switch_glu.up_proj.biases"))
            .unwrap(),
    );

    // Shapes match the reference split, and values are bit-identical (the split
    // never touches the quantized data, only selects output rows).
    assert_eq!(mlxcel_core::array_shape(&gate), vec![2, 4, 64]);
    assert_eq!(mlxcel_core::array_shape(&up), vec![2, 4, 64]);
    assert_eq!(
        max_abs_diff(&gate, &ref_gate),
        0.0,
        "dequantize(split(gate)) must equal split(dequantize)(gate)"
    );
    assert_eq!(
        max_abs_diff(&up, &ref_up),
        0.0,
        "dequantize(split(up)) must equal split(dequantize)(up)"
    );
}
