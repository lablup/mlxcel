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

//! Parity of the transformer against upstream's `MingImageTransformer`.
//!
//! `tests/fixtures/imagegen/ming_image_tiny_refs.safetensors` holds upstream
//! outputs of a tiny bf16 model; the weights are derived on both sides from
//! the closed-form [`det`] generator (`fill` in
//! `tests/fixtures/generate_imagegen_fixtures.py`), named here with the raw
//! Diffusers keys and loaded through the same sanitize and dtype split as a
//! real checkpoint. `tests/fixtures/generate_ming_image_fixtures.py tiny`
//! regenerates the file.

use std::path::PathBuf;

use mlxcel_core::dtype;
use mlxcel_core::weights::{WeightMap, load_safetensors};

use super::*;
use crate::imagegen::dit_blocks::sanitize_transformer_weights;
use crate::imagegen::ming_image::weights::reads_bf16_activation;
use crate::imagegen::testing::{det, rel_rms, values};
use crate::imagegen::z_image::weights::cast_floats;

const DIM: i32 = 64;
const MLP: i32 = 96;
const ADALN: i32 = 32;
const CAP: i32 = 24;
const PATCH_IN: i32 = 16;
const HEAD_DIM: i32 = 32;
/// rel-RMS bound against upstream's bf16 output: measured 2.5e-3 with and
/// without `cap_feats_2`; dropping `cap_feats_2` moves the output 4.5e-2.
const TOL: f64 = 5e-3;

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
        rope_sections: vec![8, 12, 12],
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

/// The value `fill(model, "tf")` assigns to `key`, in bf16 (upstream then
/// calls `set_dtype(bfloat16)`).
fn fill(key: &str, shape: &[i32]) -> UniquePtr<MlxArray> {
    let name = format!("tf.{key}");
    let parts: Vec<&str> = key.split('.').collect();
    let owner = if parts.len() > 1 {
        parts[parts.len() - 2]
    } else {
        ""
    };
    let v = if key.ends_with("norm_q.weight")
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
    };
    mlxcel_core::astype(&v, dtype::BFLOAT16)
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

/// The tiny weights after the real load path's sanitize and dtype split.
pub(crate) fn tiny_weights() -> WeightMap {
    let raw: WeightMap = parameters()
        .into_iter()
        .map(|(key, shape)| (raw_name(&key), fill(&key, &shape)))
        .collect();
    let mut weights = sanitize_transformer_weights(raw);
    let widened = cast_floats(&mut weights, reads_bf16_activation);
    assert!(widened.is_empty(), "{widened:?}");
    weights
}

fn refs() -> WeightMap {
    load_safetensors(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/imagegen/ming_image_tiny_refs.safetensors"),
    )
    .expect("fixture loads")
}

fn bf16(x: UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
    mlxcel_core::astype(&x, dtype::BFLOAT16)
}

struct Inputs {
    x: UniquePtr<MlxArray>,
    t: UniquePtr<MlxArray>,
    cap: UniquePtr<MlxArray>,
    cap2: UniquePtr<MlxArray>,
}

fn inputs() -> Inputs {
    Inputs {
        x: bf16(det("x", &[1, 4, 1, 6, 10], 1.0, 0.0)),
        t: bf16(mlxcel_core::from_slice_f32(&[0.375], &[1])),
        cap: bf16(det("cap", &[1, 7, CAP], 1.0, 0.0)),
        cap2: bf16(det("cap2", &[1, 5, DIM], 1.0, 0.0)),
    }
}

fn err(got: &MlxArray, want: &MlxArray) -> f64 {
    assert_eq!(
        mlxcel_core::array_shape(got),
        mlxcel_core::array_shape(want)
    );
    rel_rms(&values(got), &values(want))
}

#[test]
fn both_caption_streams_match_upstream() {
    let refs = refs();
    let model = MingImageTransformer::from_weights(tiny_config(), &tiny_weights()).unwrap();
    let i = inputs();
    // 15 image tokens padded to 32; 7 + 5 caption tokens padded to 32.
    let out = model.forward(&i.x, &i.t, &i.cap, Some(&i.cap2)).unwrap();
    assert_eq!(mlxcel_core::array_shape(&out), [1, 4, 1, 6, 10]);
    let with = err(&out, &refs["out"]);
    let out = model.forward(&i.x, &i.t, &i.cap, None).unwrap();
    let without = err(&out, &refs["out_no_cap2"]);
    eprintln!("tiny Ming-Image transformer rel-RMS: with cap2 {with:.3e}, without {without:.3e}");
    assert!(with < TOL, "with cap_feats_2: rel-RMS {with}");
    assert!(without < TOL, "without cap_feats_2: rel-RMS {without}");
    // The second caption stream is not ignored.
    let skipped = err(&out, &refs["out"]);
    assert!(
        skipped > 5.0 * TOL,
        "dropping cap_feats_2 moved the output by only {skipped}"
    );
}

#[test]
fn key_masks_mark_the_valid_prefix_of_each_segment() {
    assert!(key_mask(&[(32, 32)]).is_none());
    let m = key_mask(&[(2, 4), (1, 2)]).unwrap();
    assert_eq!(mlxcel_core::array_shape(&m), [1, 1, 1, 6]);
    assert_eq!(
        values(&mlxcel_core::astype(&m, dtype::FLOAT32)),
        [1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    );
}

#[test]
fn mismatched_caption_streams_are_rejected() {
    let model = MingImageTransformer::from_weights(tiny_config(), &tiny_weights()).unwrap();
    let i = inputs();
    let narrow = bf16(det("cap2", &[1, 5, DIM - 1], 1.0, 0.0));
    assert!(model.forward(&i.x, &i.t, &i.cap, Some(&narrow)).is_err());
    let wide = bf16(det("cap", &[1, 7, CAP + 1], 1.0, 0.0));
    assert!(model.forward(&i.x, &i.t, &wide, Some(&i.cap2)).is_err());
}
