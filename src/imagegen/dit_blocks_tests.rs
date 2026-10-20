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

//! Parity of the Z-Image blocks against the upstream Python.
//!
//! `tests/fixtures/imagegen/dit_blocks_refs.safetensors` holds the upstream
//! outputs for fixed inputs. Weights and inputs are not stored: both sides
//! derive them from the same closed-form generator ([`det`]), so the fixture is
//! outputs only. `tests/fixtures/generate_imagegen_fixtures.py` regenerates it.

use super::*;
use crate::imagegen::testing::{det, fnv1a32};
use mlxcel_core::weights::load_safetensors;
use std::path::PathBuf;

const DIM: i32 = 32;
const HEADS: i32 = 2;
const HEAD_DIM: i32 = 16;
const MLP: i32 = 48;
const ADALN: i32 = 24;
const OUT: i32 = 12;
const TOL: f32 = 2e-4;
/// `det("x", [1], 1, 0)[0]`, computed by the Python generator.
const FIRST_X: f32 = 0.478_709_3;

/// The weight value for `prefix.key`, classified as the Python `fill` does.
fn param(prefix: &str, key: &str, shape: &[i32]) -> UniquePtr<MlxArray> {
    let name = format!("{prefix}.{key}");
    let parts: Vec<&str> = key.split('.').collect();
    let owner = parts
        .get(parts.len().saturating_sub(2))
        .copied()
        .unwrap_or("");
    if key.ends_with("norm_q.weight")
        || key.ends_with("norm_k.weight")
        || (owner.contains("norm") && key.ends_with(".weight"))
    {
        det(&name, shape, 0.2, 1.0)
    } else if key.ends_with(".bias") {
        det(&name, shape, 0.1, 0.0)
    } else {
        let fan = f64::from(*shape.last().unwrap());
        det(&name, shape, 0.8 / fan.sqrt(), 0.0)
    }
}

fn put(map: &mut WeightMap, prefix: &str, key: &str, shape: &[i32]) {
    map.insert(format!("{prefix}.{key}"), param(prefix, key, shape));
}

fn linear_keys(map: &mut WeightMap, prefix: &str, key: &str, out: i32, inn: i32, bias: bool) {
    put(map, prefix, &format!("{key}.weight"), &[out, inn]);
    if bias {
        put(map, prefix, &format!("{key}.bias"), &[out]);
    }
}

fn feed_forward_keys(map: &mut WeightMap, prefix: &str, sub: &str) {
    let dotted = |k: &str| {
        if sub.is_empty() {
            k.to_string()
        } else {
            format!("{sub}.{k}")
        }
    };
    linear_keys(map, prefix, &dotted("w1"), MLP, DIM, false);
    linear_keys(map, prefix, &dotted("w2"), DIM, MLP, false);
    linear_keys(map, prefix, &dotted("w3"), MLP, DIM, false);
}

fn block_weights(prefix: &str, modulation: bool) -> WeightMap {
    let mut m = WeightMap::new();
    for k in ["to_q", "to_k", "to_v"] {
        linear_keys(&mut m, prefix, &format!("attention.{k}"), DIM, DIM, false);
    }
    linear_keys(&mut m, prefix, "attention.to_out.0", DIM, DIM, false);
    put(&mut m, prefix, "attention.norm_q.weight", &[HEAD_DIM]);
    put(&mut m, prefix, "attention.norm_k.weight", &[HEAD_DIM]);
    feed_forward_keys(&mut m, prefix, "feed_forward");
    for k in [
        "attention_norm1",
        "attention_norm2",
        "ffn_norm1",
        "ffn_norm2",
    ] {
        put(&mut m, prefix, &format!("{k}.weight"), &[DIM]);
    }
    if modulation {
        linear_keys(&mut m, prefix, "adaLN_modulation.0", 4 * DIM, ADALN, true);
    }
    m
}

fn refs() -> WeightMap {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/imagegen/dit_blocks_refs.safetensors");
    load_safetensors(path).expect("fixture loads")
}

fn to_vec(a: &MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(a)
}

fn rel_rms(a: &MlxArray, b: &MlxArray) -> f32 {
    assert_eq!(
        mlxcel_core::array_shape(a),
        mlxcel_core::array_shape(b),
        "shape"
    );
    let (a, b) = (to_vec(a), to_vec(b));
    let num: f64 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| f64::from(x - y).powi(2))
        .sum();
    let den: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum();
    (num / den.max(1e-30)).sqrt() as f32
}

fn assert_matches(label: &str, got: &MlxArray, want: &MlxArray) {
    let err = rel_rms(got, want);
    assert!(err < TOL, "{label}: rel-RMS {err}");
}

fn positions() -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_f32(
        &[
            1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 7.0, 0.0, 0.0, 7.0, 0.0, 1.0, 7.0, 1.0,
            0.0,
        ],
        &[1, 6, 3],
    )
}

fn rope() -> MRoPE {
    MRoPE {
        sections: vec![4, 6, 6],
        theta: 256.0,
    }
}

#[test]
fn timestep_embedder_matches_upstream() {
    let mut w = WeightMap::new();
    linear_keys(&mut w, "t_embedder", "linear1", 1024, 16, true);
    linear_keys(&mut w, "t_embedder", "linear2", ADALN, 1024, true);
    let te = TimestepEmbedder::load(&w, "t_embedder", 16).unwrap();
    let t = mlxcel_core::from_slice_f32(&[0.25 * 1000.0, 0.75 * 1000.0], &[2]);
    assert_matches("t_embed", &te.forward(&t), &refs()["t_embed"]);
}

#[test]
fn feed_forward_matches_upstream() {
    let mut w = WeightMap::new();
    feed_forward_keys(&mut w, "feed_forward", "");
    let ff = FeedForward::load(&w, "feed_forward").unwrap();
    let x = det("x", &[1, 6, DIM], 1.0, 0.0);
    assert_matches("feed_forward", &ff.forward(&x), &refs()["feed_forward"]);
}

#[test]
fn mrope_frequencies_match_upstream() {
    let (cos, sin) = rope().compute_freqs(&positions()).unwrap();
    assert_eq!(mlxcel_core::array_shape(&cos), [1, 6, 16]);
    assert_matches("rope_cos", &cos, &refs()["rope_cos"]);
    assert_matches("rope_sin", &sin, &refs()["rope_sin"]);
}

#[test]
fn mrope_rejects_bad_inputs() {
    assert!(
        rope()
            .compute_freqs(&mlxcel_core::zeros(&[1, 6, 2], dtype::FLOAT32))
            .is_err()
    );
    assert!(
        rope()
            .compute_freqs(&mlxcel_core::zeros(&[6, 3], dtype::FLOAT32))
            .is_err()
    );
    let odd = MRoPE {
        sections: vec![4, 5, 6],
        theta: 256.0,
    };
    assert!(odd.compute_freqs(&positions()).is_err());
}

#[test]
fn apply_rotary_matches_upstream() {
    let (cos, sin) = rope().compute_freqs(&positions()).unwrap();
    let q = det("q", &[1, 6, HEADS, HEAD_DIM], 1.0, 0.0);
    let out = apply_rotary(&q, &cos, &sin).unwrap();
    assert_matches("apply_rotary", &out, &refs()["apply_rotary"]);
}

#[test]
fn apply_rotary_is_a_rotation_and_position_zero_is_the_identity() {
    let q = det("q", &[1, 6, HEADS, HEAD_DIM], 1.0, 0.0);
    let zeros = mlxcel_core::zeros(&[1, 6, 3], dtype::FLOAT32);
    let (cos, sin) = rope().compute_freqs(&zeros).unwrap();
    let same = apply_rotary(&q, &cos, &sin).unwrap();
    assert!(rel_rms(&same, &q) < 1e-6);

    let (cos, sin) = rope().compute_freqs(&positions()).unwrap();
    let rotated = apply_rotary(&q, &cos, &sin).unwrap();
    let norm = |a: &MlxArray| to_vec(a).iter().map(|v| v * v).sum::<f32>();
    assert!(
        (norm(&rotated) - norm(&q)).abs() / norm(&q) < 1e-5,
        "rotation preserves length"
    );
    assert!(apply_rotary(&q, &cos, &sin).is_ok());
    assert!(apply_rotary(&mlxcel_core::zeros(&[6, 16], dtype::FLOAT32), &cos, &sin).is_err());
}

#[test]
fn modulated_block_matches_upstream() {
    let w = block_weights("layers.0", true);
    let block = ZImageTransformerBlock::load(
        &w,
        "layers.0",
        HEADS,
        HEAD_DIM,
        true,
        RopePrecision::KeepPromoted,
    )
    .unwrap();
    assert!(block.has_modulation());
    let (cos, sin) = rope().compute_freqs(&positions()).unwrap();
    let x = det("x", &[1, 6, DIM], 1.0, 0.0);
    let adaln = det("adaln", &[1, ADALN], 1.0, 0.0);
    let out = block.forward(&x, &cos, &sin, Some(&adaln), None).unwrap();
    assert_matches("block_mod", &out, &refs()["block_mod"]);
}

#[test]
fn attention_mask_is_honored() {
    let w = block_weights("layers.0", true);
    let block = ZImageTransformerBlock::load(
        &w,
        "layers.0",
        HEADS,
        HEAD_DIM,
        true,
        RopePrecision::KeepPromoted,
    )
    .unwrap();
    let (cos, sin) = rope().compute_freqs(&positions()).unwrap();
    let x = det("x", &[1, 6, DIM], 1.0, 0.0);
    let adaln = det("adaln", &[1, ADALN], 1.0, 0.0);
    let mut mask = vec![0.0_f32; 36];
    for r in 0..6 {
        for c in (r + 1)..6 {
            mask[r * 6 + c] = -1e4;
        }
    }
    let mask = mlxcel_core::from_slice_f32(&mask, &[1, 1, 6, 6]);
    let out = block
        .forward(&x, &cos, &sin, Some(&adaln), Some(&mask))
        .unwrap();
    assert_matches("block_mod_masked", &out, &refs()["block_mod_masked"]);
    let unmasked = block.forward(&x, &cos, &sin, Some(&adaln), None).unwrap();
    assert!(
        rel_rms(&out, &unmasked) > 1e-4,
        "the mask must change the result"
    );
}

#[test]
fn plain_block_matches_upstream_and_needs_no_adaln() {
    let w = block_weights("context_refiner.0", false);
    let block = ZImageTransformerBlock::load(
        &w,
        "context_refiner.0",
        HEADS,
        HEAD_DIM,
        false,
        RopePrecision::KeepPromoted,
    )
    .unwrap();
    assert!(!block.has_modulation());
    let (cos, sin) = rope().compute_freqs(&positions()).unwrap();
    let x = det("x", &[1, 6, DIM], 1.0, 0.0);
    let out = block.forward(&x, &cos, &sin, None, None).unwrap();
    assert_matches("block_plain", &out, &refs()["block_plain"]);
}

#[test]
fn a_modulated_block_without_adaln_input_is_an_error() {
    let w = block_weights("layers.0", true);
    let block = ZImageTransformerBlock::load(
        &w,
        "layers.0",
        HEADS,
        HEAD_DIM,
        true,
        RopePrecision::KeepPromoted,
    )
    .unwrap();
    let (cos, sin) = rope().compute_freqs(&positions()).unwrap();
    let x = det("x", &[1, 6, DIM], 1.0, 0.0);
    let err = block
        .forward(&x, &cos, &sin, None, None)
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("adaln_input"), "{err}");
}

#[test]
fn loading_a_modulated_block_from_unmodulated_weights_names_the_key() {
    let w = block_weights("layers.0", false);
    let err = ZImageTransformerBlock::load(
        &w,
        "layers.0",
        HEADS,
        HEAD_DIM,
        true,
        RopePrecision::KeepPromoted,
    )
    .err()
    .unwrap();
    assert!(format!("{err:#}").contains("adaLN_modulation.0"), "{err:#}");
}

#[test]
fn final_layer_matches_upstream() {
    let mut w = WeightMap::new();
    linear_keys(&mut w, "final_layer", "linear", OUT, DIM, true);
    linear_keys(
        &mut w,
        "final_layer",
        "adaLN_modulation.0",
        DIM,
        ADALN,
        true,
    );
    let layer = FinalLayer::load(&w, "final_layer").unwrap();
    let x = det("x", &[1, 6, DIM], 1.0, 0.0);
    let adaln = det("adaln", &[1, ADALN], 1.0, 0.0);
    assert_matches(
        "final_layer",
        &layer.forward(&x, &adaln),
        &refs()["final_layer"],
    );
}

#[test]
fn rope_precision_decides_the_block_output_dtype_for_half_precision() {
    let (cos, sin) = rope().compute_freqs(&positions()).unwrap();
    let to_half = |a: UniquePtr<MlxArray>| mlxcel_core::astype(&a, dtype::FLOAT16);
    let w: WeightMap = block_weights("layers.0", true)
        .into_iter()
        .map(|(k, v)| (k, mlxcel_core::astype(&v, dtype::FLOAT16)))
        .collect();
    let x = to_half(det("x", &[1, 6, DIM], 1.0, 0.0));
    let adaln = to_half(det("adaln", &[1, ADALN], 1.0, 0.0));

    let round = ZImageTransformerBlock::load(
        &w,
        "layers.0",
        HEADS,
        HEAD_DIM,
        true,
        RopePrecision::RoundToInput,
    )
    .unwrap()
    .forward(&x, &cos, &sin, Some(&adaln), None)
    .unwrap();
    assert_eq!(mlxcel_core::array_dtype(&round), dtype::FLOAT16);
    assert!(to_vec(&round).iter().all(|v| v.is_finite()));

    let keep = ZImageTransformerBlock::load(
        &w,
        "layers.0",
        HEADS,
        HEAD_DIM,
        true,
        RopePrecision::KeepPromoted,
    )
    .unwrap()
    .forward(&x, &cos, &sin, Some(&adaln), None)
    .unwrap();
    assert!(to_vec(&keep).iter().all(|v| v.is_finite()));
    // Both stay close to the f32 reference at half precision.
    let reference = &refs()["block_mod"];
    assert!(rel_rms(&mlxcel_core::astype(&round, dtype::FLOAT32), reference) < 5e-2);
    assert!(rel_rms(&mlxcel_core::astype(&keep, dtype::FLOAT32), reference) < 5e-2);
}

#[test]
fn sanitize_maps_the_source_names() {
    let marker = |n: f32| mlxcel_core::from_slice_f32(&[n], &[1]);
    let mut w = WeightMap::new();
    for (i, key) in [
        "all_final_layer.2-1.adaLN_modulation.1.weight",
        "all_final_layer.2-1.linear.bias",
        "all_x_embedder.2-1.weight",
        "t_embedder.mlp.0.weight",
        "t_embedder.mlp.2.bias",
        "layers.0.attention.to_q.weight",
    ]
    .iter()
    .enumerate()
    {
        w.insert((*key).to_string(), marker(i as f32));
    }
    let s = sanitize_transformer_weights(w);
    let mut keys: Vec<&str> = s.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        [
            "final_layer.adaLN_modulation.0.weight",
            "final_layer.linear.bias",
            "layers.0.attention.to_q.weight",
            "t_embedder.linear1.weight",
            "t_embedder.linear2.bias",
            "x_embedder.weight",
        ]
    );
    assert_eq!(
        to_vec(&s["x_embedder.weight"]),
        [2.0],
        "values travel with their keys"
    );
}

#[test]
fn det_is_the_closed_form_the_python_generator_uses() {
    // Pinned so a change to the generator is noticed on both sides.
    assert_eq!(fnv1a32("x"), 0xFD0C_5087);
    let first = to_vec(&det("x", &[1], 1.0, 0.0))[0];
    assert!((first - FIRST_X).abs() < 1e-7, "{first}");
    assert_eq!(
        to_vec(&det("x", &[4], 1.0, 0.0)),
        to_vec(&det("x", &[4], 1.0, 0.0))
    );
    assert_ne!(
        to_vec(&det("y", &[4], 1.0, 0.0)),
        to_vec(&det("x", &[4], 1.0, 0.0))
    );
}
