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

//! Parity of the top-level transformer against upstream's `ZImageTransformer`.
//!
//! `tests/fixtures/imagegen/z_image_tiny_refs.safetensors` holds upstream
//! outputs for a tiny config; the weights are derived on both sides from the
//! closed-form [`det`] generator (`fill` in
//! `tests/fixtures/generate_imagegen_fixtures.py`), named here with the raw
//! Diffusers keys so the load path's key sanitization is exercised too.
//! `tests/fixtures/generate_z_image_fixtures.py tiny` regenerates the file.

use std::path::PathBuf;

use super::*;
use crate::imagegen::dit_blocks::sanitize_transformer_weights;
use crate::imagegen::testing::det;
use mlxcel_core::dtype;
use mlxcel_core::weights::load_safetensors;

const DIM: i32 = 32;
const MLP: i32 = 48;
const ADALN: i32 = 24;
const CAP: i32 = 12;
const PATCH_IN: i32 = 16;
const HEAD_DIM: i32 = 16;
const TOL: f32 = 2e-4;

fn tiny_config() -> ZImageTransformerConfig {
    ZImageTransformerConfig {
        hidden_size: DIM,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        in_channels: 4,
        text_embed_dim: CAP,
        num_hidden_layers: 2,
        n_refiner_layers: 1,
        n_context_refiner_layers: 1,
        patch_size: 2,
        f_patch_size: 1,
        adaln_embed_dim: ADALN,
        rope_sections: vec![4, 6, 6],
        rope_theta: 256.0,
        norm_eps: 1e-5,
        timestep_scale: 1000.0,
    }
}

/// `(upstream module key, shape)` of every parameter of the tiny model.
fn parameters() -> Vec<(String, Vec<i32>)> {
    let mut p: Vec<(String, Vec<i32>)> = vec![
        ("t_embedder.linear1.weight".into(), vec![1024, 256]),
        ("t_embedder.linear1.bias".into(), vec![1024]),
        ("t_embedder.linear2.weight".into(), vec![ADALN, 1024]),
        ("t_embedder.linear2.bias".into(), vec![ADALN]),
        ("cap_embedder.0.weight".into(), vec![CAP]),
        ("cap_embedder.1.weight".into(), vec![DIM, CAP]),
        ("cap_embedder.1.bias".into(), vec![DIM]),
        ("final_layer.linear.weight".into(), vec![PATCH_IN, DIM]),
        ("final_layer.linear.bias".into(), vec![PATCH_IN]),
        (
            "final_layer.adaLN_modulation.0.weight".into(),
            vec![DIM, ADALN],
        ),
        ("final_layer.adaLN_modulation.0.bias".into(), vec![DIM]),
        ("x_embedder.weight".into(), vec![DIM, PATCH_IN]),
        ("x_embedder.bias".into(), vec![DIM]),
        ("x_pad_token".into(), vec![1, DIM]),
        ("cap_pad_token".into(), vec![1, DIM]),
    ];
    let blocks = [
        ("noise_refiner.0", true),
        ("context_refiner.0", false),
        ("layers.0", true),
        ("layers.1", true),
    ];
    for (prefix, modulated) in blocks {
        let mut add = |k: &str, s: Vec<i32>| p.push((format!("{prefix}.{k}"), s));
        for proj in ["to_q", "to_k", "to_v", "to_out.0"] {
            add(&format!("attention.{proj}.weight"), vec![DIM, DIM]);
        }
        add("attention.norm_q.weight", vec![HEAD_DIM]);
        add("attention.norm_k.weight", vec![HEAD_DIM]);
        add("feed_forward.w1.weight", vec![MLP, DIM]);
        add("feed_forward.w2.weight", vec![DIM, MLP]);
        add("feed_forward.w3.weight", vec![MLP, DIM]);
        for norm in [
            "attention_norm1",
            "attention_norm2",
            "ffn_norm1",
            "ffn_norm2",
        ] {
            add(&format!("{norm}.weight"), vec![DIM]);
        }
        if modulated {
            add("adaLN_modulation.0.weight", vec![4 * DIM, ADALN]);
            add("adaLN_modulation.0.bias", vec![4 * DIM]);
        }
    }
    p
}

/// The value `fill(model, "zt")` assigns to `key`.
fn fill(key: &str, shape: &[i32]) -> UniquePtr<MlxArray> {
    let name = format!("zt.{key}");
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

/// The source checkpoint's name for an upstream module key.
fn raw_name(key: &str) -> String {
    if let Some(rest) = key.strip_prefix("final_layer.") {
        format!(
            "all_final_layer.2-1.{}",
            rest.replace("adaLN_modulation.0.", "adaLN_modulation.1.")
        )
    } else if let Some(rest) = key.strip_prefix("x_embedder.") {
        format!("all_x_embedder.2-1.{rest}")
    } else {
        key.replace("t_embedder.linear1.", "t_embedder.mlp.0.")
            .replace("t_embedder.linear2.", "t_embedder.mlp.2.")
    }
}

fn raw_weights() -> WeightMap {
    parameters()
        .into_iter()
        .map(|(key, shape)| (raw_name(&key), fill(&key, &shape)))
        .collect()
}

fn refs() -> WeightMap {
    load_safetensors(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/imagegen/z_image_tiny_refs.safetensors"),
    )
    .expect("fixture loads")
}

fn rel_rms(a: &MlxArray, b: &MlxArray) -> f32 {
    assert_eq!(mlxcel_core::array_shape(a), mlxcel_core::array_shape(b));
    let a = mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(a, dtype::FLOAT32));
    let b = mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(b, dtype::FLOAT32));
    let num: f64 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| f64::from(x - y).powi(2))
        .sum();
    let den: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum();
    (num / den.max(1e-30)).sqrt() as f32
}

fn tiny_model() -> ZImageTransformer {
    ZImageTransformer::from_weights(tiny_config(), &sanitize_transformer_weights(raw_weights()))
        .expect("tiny transformer builds")
}

#[test]
fn raw_checkpoint_keys_sanitize_onto_the_module_layout() {
    let sanitized = sanitize_transformer_weights(raw_weights());
    let mut got: Vec<String> = sanitized.into_keys().collect();
    let mut want: Vec<String> = parameters().into_iter().map(|(k, _)| k).collect();
    got.sort();
    want.sort();
    assert_eq!(got, want);
    // Unsanitized raw keys do not load.
    let err = ZImageTransformer::from_weights(tiny_config(), &raw_weights())
        .err()
        .expect("raw keys need sanitizing")
        .to_string();
    assert!(err.contains("Z-Image"), "{err}");
}

#[test]
fn padded_streams_match_upstream() {
    let refs = refs();
    let model = tiny_model();
    // 15 image and 5 caption tokens: both streams are padded to 32.
    let x = det("latents", &[1, 4, 1, 6, 10], 1.0, 0.0);
    let cap = det("cap", &[1, 5, CAP], 1.0, 0.0);
    let t = mlxcel_core::from_slice_f32(&[0.3], &[1]);
    let out = model.forward(&x, &t, &cap).unwrap();
    assert_eq!(mlxcel_core::array_shape(&out), [1, 4, 1, 6, 10]);
    assert_eq!(mlxcel_core::array_dtype(&out), dtype::FLOAT32);
    let err = rel_rms(&out, &refs["out"]);
    assert!(err < TOL, "rel-RMS {err}");
}

#[test]
fn a_shared_caption_broadcasts_over_the_batch() {
    let refs = refs();
    let model = tiny_model();
    let x = det("latents_b2", &[2, 4, 1, 4, 4], 1.0, 0.0);
    let cap = det("cap", &[1, 5, CAP], 1.0, 0.0);
    let t = mlxcel_core::from_slice_f32(&[0.75], &[1]);
    let out = model.forward(&x, &t, &cap).unwrap();
    let err = rel_rms(&out, &refs["out_b2"]);
    assert!(err < TOL, "rel-RMS {err}");
}

#[test]
fn position_ids_follow_upstream_layout() {
    let (cap, img) = build_position_ids(2, 1, 2, 3);
    assert_eq!(cap, [1.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
    let rows: Vec<&[f32]> = img.chunks(3).collect();
    assert_eq!(rows.len(), 6);
    assert_eq!(rows[0], [3.0, 0.0, 0.0]);
    assert_eq!(rows[2], [3.0, 0.0, 2.0]);
    assert_eq!(rows[3], [3.0, 1.0, 0.0]);
    assert_eq!(rows[5], [3.0, 1.0, 2.0]);
}

#[test]
fn bad_inputs_are_rejected() {
    let model = tiny_model();
    let t = mlxcel_core::from_slice_f32(&[0.3], &[1]);
    let cap = det("cap", &[1, 5, CAP], 1.0, 0.0);
    let odd = det("x", &[1, 4, 1, 5, 6], 1.0, 0.0);
    assert!(model.forward(&odd, &t, &cap).is_err());
    let four_d = det("x", &[1, 4, 6, 6], 1.0, 0.0);
    assert!(model.forward(&four_d, &t, &cap).is_err());
    let x = det("x", &[1, 4, 1, 4, 4], 1.0, 0.0);
    let wide = det("cap", &[1, 5, CAP + 1], 1.0, 0.0);
    assert!(model.forward(&x, &t, &wide).is_err());
}
