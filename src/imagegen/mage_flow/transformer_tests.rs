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

//! Parity of the RoPE tables and of the whole transformer against upstream's
//! `image_rope_frequencies` and `MageFlowTransformer`.
//!
//! `tests/fixtures/imagegen/mage_flow_tiny_refs.safetensors` holds upstream
//! outputs (`generate_mage_flow_fixtures.py tiny`); the tiny model's weights
//! are derived on both sides from the closed-form [`det`] generator (`fill`
//! in `tests/fixtures/generate_imagegen_fixtures.py`), named here with the
//! checkpoint's Diffusers keys so the key sanitization is exercised too.

use std::path::PathBuf;

use super::*;
use crate::imagegen::mage_flow::weights::{infer_quantization, sanitize_transformer_key};
use crate::imagegen::testing::det;
use mlxcel_core::weights::{WeightMap, load_safetensors};

const DIM: i32 = 64;
const IN: i32 = 8;
const CTX: i32 = 24;
const HEAD_DIM: i32 = 32;

pub(crate) fn refs() -> WeightMap {
    load_safetensors(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/imagegen/mage_flow_tiny_refs.safetensors"),
    )
    .expect("fixture loads")
}

pub(crate) fn to_vec(a: &MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(a, dtype::FLOAT32))
}

pub(crate) fn rel_rms(a: &MlxArray, b: &MlxArray) -> f32 {
    assert_eq!(mlxcel_core::array_shape(a), mlxcel_core::array_shape(b));
    let (a, b) = (to_vec(a), to_vec(b));
    let num: f64 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| f64::from(x - y).powi(2))
        .sum();
    let den: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum();
    (num / den.max(1e-30)).sqrt() as f32
}

fn tiny_config() -> TransformerConfig {
    TransformerConfig {
        in_channels: IN,
        out_channels: IN,
        context_in_dim: CTX,
        hidden_size: DIM,
        num_heads: 2,
        depth: 2,
        axes_dim: [8, 12, 12],
        theta: 10000.0,
    }
}

/// `(upstream module key, shape)` of every parameter of the tiny model.
fn parameters() -> Vec<(String, Vec<i32>)> {
    let mut p: Vec<(String, Vec<i32>)> = Vec::new();
    let linear = |p: &mut Vec<(String, Vec<i32>)>, key: &str, out: i32, inn: i32| {
        p.push((format!("{key}.weight"), vec![out, inn]));
        p.push((format!("{key}.bias"), vec![out]));
    };
    linear(&mut p, "img_in", DIM, IN);
    p.push(("txt_norm.weight".into(), vec![CTX]));
    linear(&mut p, "txt_in", DIM, CTX);
    linear(
        &mut p,
        "time_text_embed.timestep_embedder.linear_1",
        DIM,
        256,
    );
    linear(
        &mut p,
        "time_text_embed.timestep_embedder.linear_2",
        DIM,
        DIM,
    );
    for i in 0..2 {
        let b = format!("transformer_blocks.{i}");
        linear(&mut p, &format!("{b}.img_mod.linear"), 6 * DIM, DIM);
        linear(&mut p, &format!("{b}.txt_mod.linear"), 6 * DIM, DIM);
        for proj in [
            "to_q",
            "to_k",
            "to_v",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_out",
            "to_add_out",
        ] {
            linear(&mut p, &format!("{b}.attn.{proj}"), DIM, DIM);
        }
        for norm in ["norm_q", "norm_k", "norm_added_q", "norm_added_k"] {
            p.push((format!("{b}.attn.{norm}.weight"), vec![HEAD_DIM]));
        }
        for mlp in ["img_mlp", "txt_mlp"] {
            linear(&mut p, &format!("{b}.{mlp}.linear_in"), 4 * DIM, DIM);
            linear(&mut p, &format!("{b}.{mlp}.linear_out"), DIM, 4 * DIM);
        }
    }
    linear(&mut p, "norm_out.linear", 2 * DIM, DIM);
    linear(&mut p, "proj_out", IN, DIM);
    p
}

/// The value `fill(model, "mft")` assigns to `key`.
fn fill(key: &str, shape: &[i32]) -> UniquePtr<MlxArray> {
    let name = format!("mft.{key}");
    let parts: Vec<&str> = key.split('.').collect();
    let owner = if parts.len() > 1 {
        parts[parts.len() - 2]
    } else {
        ""
    };
    if key.ends_with("norm_q.weight")
        || key.ends_with("norm_k.weight")
        || (owner.contains("norm") && key.ends_with(".weight"))
    {
        det(&name, shape, 0.2, 1.0)
    } else if key.ends_with(".bias") {
        det(&name, shape, 0.1, 0.0)
    } else {
        let fan = if shape.len() > 1 {
            shape[shape.len() - 1]
        } else {
            1
        };
        det(&name, shape, 0.8 / f64::from(fan).sqrt(), 0.0)
    }
}

/// The published checkpoint's name for an upstream module key (the inverse
/// of [`sanitize_transformer_key`]).
fn raw_name(key: &str) -> String {
    key.replace(".img_mod.linear.", ".img_mod.1.")
        .replace(".txt_mod.linear.", ".txt_mod.1.")
        .replace(".img_mlp.linear_in.", ".img_mlp.net.0.proj.")
        .replace(".img_mlp.linear_out.", ".img_mlp.net.2.")
        .replace(".txt_mlp.linear_in.", ".txt_mlp.net.0.proj.")
        .replace(".txt_mlp.linear_out.", ".txt_mlp.net.2.")
        .replace(".attn.to_out.", ".attn.to_out.0.")
}

fn sanitized_weights(dt: i32) -> WeightMap {
    parameters()
        .into_iter()
        .map(|(key, shape)| {
            let raw = raw_name(&key);
            (
                sanitize_transformer_key(&raw),
                mlxcel_core::astype(&fill(&key, &shape), dt),
            )
        })
        .collect()
}

fn tiny_model(dt: i32) -> MageFlowTransformer {
    MageFlowTransformer::from_weights(tiny_config(), &sanitized_weights(dt), None)
        .expect("tiny transformer builds")
}

#[test]
fn checkpoint_keys_sanitize_onto_the_module_layout() {
    let mut got: Vec<String> = sanitized_weights(dtype::FLOAT32).into_keys().collect();
    let mut want: Vec<String> = parameters().into_iter().map(|(k, _)| k).collect();
    got.sort();
    want.sort();
    assert_eq!(got, want);
    // Unsanitized checkpoint keys do not load.
    let raw: WeightMap = parameters()
        .into_iter()
        .map(|(k, s)| (raw_name(&k), fill(&k, &s)))
        .collect();
    let err = MageFlowTransformer::from_weights(tiny_config(), &raw, None)
        .err()
        .expect("raw keys need sanitizing")
        .to_string();
    assert!(err.contains("Mage-Flow weight not found"), "{err}");
}

#[test]
fn rope_positions_of_a_32x48_grid_match_upstream() {
    let refs = refs();
    let want = to_vec(&refs["rope.grid_positions"]);
    let got: Vec<f32> = rope_positions(&[(1, 32, 48)])
        .into_iter()
        .flatten()
        .collect();
    assert_eq!(got, want);
    // Centered: rows -16..15 and columns -24..23.
    assert_eq!(&got[..3], [0.0, -16.0, -24.0]);
    let edit = to_vec(&refs["rope.edit_positions"]);
    let got: Vec<f32> = rope_positions(&[(1, 4, 6); 3])
        .into_iter()
        .flatten()
        .collect();
    assert_eq!(got, edit);
}

/// Every other column of every `rows_step`-th row of a `[1, L, D]` table
/// whose frequencies are duplicated, as upstream's `[L, D / 2]`.
fn half_table(t: &MlxArray, rows_step: usize) -> Vec<f32> {
    let s = mlxcel_core::array_shape(t);
    let (l, d) = (s[1] as usize, s[2] as usize);
    let v = to_vec(t);
    (0..l)
        .step_by(rows_step)
        .flat_map(|r| (0..d / 2).map(move |c| (r, c)))
        .map(|(r, c)| v[r * d + 2 * c])
        .collect()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn rope_tables_match_upstream() {
    let refs = refs();
    let (cos, sin) = rope_tables(&[(1, 32, 48)], [16, 56, 56], 10000.0).unwrap();
    let e = max_abs(
        &half_table(&cos, 16),
        &to_vec(&refs["rope.grid_cos_rows16"]),
    );
    assert!(e < 1e-5, "grid cos max abs {e}");
    let e = max_abs(
        &half_table(&sin, 16),
        &to_vec(&refs["rope.grid_sin_rows16"]),
    );
    assert!(e < 1e-5, "grid sin max abs {e}");
    let (cos, sin) = rope_tables(&[(1, 4, 6); 3], [16, 56, 56], 10000.0).unwrap();
    let e = max_abs(&half_table(&cos, 1), &to_vec(&refs["rope.edit_cos"]));
    assert!(e < 1e-5, "edit cos max abs {e}");
    let e = max_abs(&half_table(&sin, 1), &to_vec(&refs["rope.edit_sin"]));
    assert!(e < 1e-5, "edit sin max abs {e}");
}

fn inputs(
    dt: i32,
) -> (
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
) {
    let cast = |a: UniquePtr<MlxArray>| mlxcel_core::astype(&a, dt);
    (
        cast(det("mft.img", &[1, 12, IN], 1.0, 0.0)),
        cast(det("mft.txt", &[1, 5, CTX], 1.0, 0.0)),
        cast(det("mft.edit_img", &[1, 12, IN], 1.0, 0.0)),
    )
}

#[test]
fn f32_generate_and_edit_layouts_match_upstream() {
    let refs = refs();
    let model = tiny_model(dtype::FLOAT32);
    let (img, txt, edit_img) = inputs(dtype::FLOAT32);
    let out = model.forward(&img, &txt, 0.7, &[(1, 3, 4)]).unwrap();
    let err = rel_rms(&out, &refs["tf.f32_generate"]);
    assert!(err < 1e-4, "generate rel-RMS {err}");
    // Two frames: the reference grid sits at frame 1 of the RoPE table.
    let out = model
        .forward(&edit_img, &txt, 0.7, &[(1, 2, 3), (1, 2, 3)])
        .unwrap();
    let err = rel_rms(&out, &refs["tf.f32_edit"]);
    assert!(err < 1e-4, "edit rel-RMS {err}");
}

#[test]
fn bf16_forward_matches_upstream() {
    let refs = refs();
    let model = tiny_model(dtype::BFLOAT16);
    let (img, txt, _) = inputs(dtype::BFLOAT16);
    let out = model.forward(&img, &txt, 0.7, &[(1, 3, 4)]).unwrap();
    assert_eq!(mlxcel_core::array_dtype(&out), dtype::BFLOAT16);
    let err = rel_rms(&out, &refs["tf.bf16_generate"]);
    // bf16 end to end on both sides (one bf16 ulp is 3.9e-3 relative).
    assert!(err < 2e-2, "bf16 rel-RMS {err}");
}

#[test]
fn bad_inputs_are_rejected() {
    let model = tiny_model(dtype::FLOAT32);
    let (img, txt, _) = inputs(dtype::FLOAT32);
    // The shapes cover 9 tokens, the image has 12.
    assert!(model.forward(&img, &txt, 0.7, &[(1, 3, 3)]).is_err());
    let flat = mlxcel_core::reshape(&img, &[12, IN]);
    assert!(model.forward(&flat, &txt, 0.7, &[(1, 3, 4)]).is_err());
}

/// Quantize every block projection of `weights` (affine, `bits`,
/// `group_size`) the way mlx-vlm's converter does, leaving the rest dense.
fn quantize_blocks(weights: &mut WeightMap, group_size: i32, bits: i32) -> Vec<String> {
    let keys: Vec<String> = weights
        .keys()
        .filter(|k| {
            k.starts_with("transformer_blocks.")
                && k.ends_with(".weight")
                && mlxcel_core::array_ndim(&weights[*k]) == 2
        })
        .cloned()
        .collect();
    for key in &keys {
        let dense = weights.remove(key).unwrap();
        let q = mlxcel_core::quantize_weights(&dense, group_size, bits);
        let prefix = key.strip_suffix(".weight").unwrap();
        weights.insert(key.clone(), mlxcel_core::quantized_weights_w(&q));
        weights.insert(
            format!("{prefix}.scales"),
            mlxcel_core::quantized_weights_scales(&q),
        );
        weights.insert(
            format!("{prefix}.biases"),
            mlxcel_core::quantized_weights_biases(&q),
        );
    }
    keys
}

#[test]
fn quantized_block_projections_load_and_run() {
    let mut weights = sanitized_weights(dtype::FLOAT32);
    let keys = quantize_blocks(&mut weights, 32, 8);
    assert!(!keys.is_empty());
    // No metadata: the packed tensors are refused.
    let err = infer_quantization(&weights, &serde_json::Map::new())
        .expect_err("missing metadata")
        .to_string();
    assert!(err.contains("quantization mode"), "{err}");
    // Packed tensors without metadata do not build a dense model either.
    assert!(MageFlowTransformer::from_weights(tiny_config(), &weights, None).is_err());

    let metadata = serde_json::json!({
        "quantization": {"mode": "affine", "bits": 8, "group_size": 32}
    });
    let metadata = metadata.as_object().unwrap();
    let quant = infer_quantization(&weights, metadata)
        .unwrap()
        .expect("quantized");
    assert_eq!(
        (quant.bits, quant.group_size, quant.mode),
        (8, 32, "affine")
    );
    let model = MageFlowTransformer::from_weights(tiny_config(), &weights, Some(quant)).unwrap();
    assert!(model.blocks[0].attn.to_q.is_quantized());
    assert!(model.blocks[1].img_mod.is_quantized());
    assert!(!model.img_in.is_quantized());
    let (img, txt, _) = inputs(dtype::FLOAT32);
    let out = model.forward(&img, &txt, 0.7, &[(1, 3, 4)]).unwrap();
    // 8-bit block weights: close to the dense upstream output, not equal.
    let err = rel_rms(&out, &refs()["tf.f32_generate"]);
    assert!(err > 0.0 && err < 5e-2, "quantized rel-RMS {err}");

    // Scales that do not match their packed weight are rejected before MLX
    // sees them.
    let prefix = keys[0].strip_suffix(".weight").unwrap().to_string();
    weights.insert(
        format!("{prefix}.scales"),
        mlxcel_core::zeros(&[3, 1], dtype::FLOAT32),
    );
    let err = infer_quantization(&weights, metadata)
        .expect_err("bad scales")
        .to_string();
    assert!(err.contains(&prefix), "{err}");
}

#[test]
fn quantization_metadata_is_read_like_upstream() {
    let mut weights = sanitized_weights(dtype::FLOAT32);
    quantize_blocks(&mut weights, 64, 4);
    // Top-level keys (an mlx-vlm index metadata spelling), numeric strings.
    let flat = serde_json::json!({
        "quantization_mode": "AFFINE", "quantization_level": "4",
        "quantization_group_size": 64
    });
    let q = infer_quantization(&weights, flat.as_object().unwrap())
        .unwrap()
        .unwrap();
    assert_eq!((q.bits, q.group_size, q.mode), (4, 64, "affine"));
    let nested = serde_json::json!({"quantization_config": {"q_mode": "affine", "q_bits": 4, "q_group_size": 64}});
    assert!(
        infer_quantization(&weights, nested.as_object().unwrap())
            .unwrap()
            .is_some()
    );
    let bad_mode = serde_json::json!({"mode": "int4", "bits": 4, "group_size": 64});
    let err = infer_quantization(&weights, bad_mode.as_object().unwrap())
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("Unsupported Mage-Flow quantization"), "{err}");
    let no_bits = serde_json::json!({"mode": "affine", "group_size": 64});
    assert!(infer_quantization(&weights, no_bits.as_object().unwrap()).is_err());
    // A dense map needs no metadata.
    assert!(
        infer_quantization(&sanitized_weights(dtype::FLOAT32), &serde_json::Map::new())
            .unwrap()
            .is_none()
    );
}
