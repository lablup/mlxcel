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

//! Limite numerics on synthetic tensors (issue #2360): the dtype epsilon, the
//! bf16 fold, the partial interleaved RoPE, the MUDD accumulation order, the
//! 1025-key sliding cache, the config contract, and the model-owned cache
//! contracts shared with the other mixed sliding/global families.

use super::limite::{LimiteModel, LimiteWrapper, ModelArgs};
use super::limite_layers::{
    MuddWay, PartialRope, finfo_eps, fold, mudd_combine, positions_from_offset, rms_norm,
};
use super::mixed_cache_batched_decode_tests::contracts;
use crate::models::last_row_logits_harness::last_row_case;
use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

fn values(x: &MlxArray) -> Vec<f32> {
    let x = mlxcel_core::astype(x, dtype::FLOAT32);
    mlxcel_core::eval(&x);
    mlxcel_core::utils::array_to_vec_f32(&x)
}

/// Round an f32 to bf16 (round to nearest, ties to even).
fn bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(rounded)
}

fn bf16_array(data: &[f32], shape: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::astype(&mlxcel_core::from_slice_f32(data, shape), dtype::BFLOAT16)
}

#[test]
fn rms_norm_uses_the_dtype_finfo_epsilon() {
    assert_eq!(finfo_eps(dtype::BFLOAT16), 0.0078125);
    assert_eq!(finfo_eps(dtype::FLOAT16), 0.0009765625);
    assert_eq!(finfo_eps(dtype::FLOAT32), 1.1920929e-07);
    // Mean square 1e-4 is far below the bf16 epsilon, so the epsilon
    // dominates and the usual 1e-6 would give an output near 1.
    let data = [0.01f32, -0.01, 0.01, -0.01];
    for (dt, eps) in [
        (dtype::FLOAT32, 1.1920929e-07f64),
        (dtype::FLOAT16, 0.0009765625),
        (dtype::BFLOAT16, 0.0078125),
    ] {
        let x = mlxcel_core::astype(&mlxcel_core::from_slice_f32(&data, &[1, 4]), dt);
        let got = values(&rms_norm(&x));
        let xin = values(&x)[0] as f64;
        let want = xin / (xin * xin + eps).sqrt();
        let tol = if dt == dtype::FLOAT32 { 1e-5 } else { 1e-2 };
        assert!(
            (got[0] as f64 - want).abs() < tol * want.abs(),
            "dtype {dt}: got {} want {want}",
            got[0]
        );
    }
    let x = bf16_array(&data, &[1, 4]);
    let wrong = values(&mlxcel_core::fast_rms_norm_no_weight(&x, 1e-6));
    assert!((values(&rms_norm(&x))[0] - wrong[0]).abs() > 0.1);
}

#[test]
fn fold_rounds_the_scalar_to_the_weight_dtype_first() {
    let scale = 1.005f32;
    let w: Vec<f32> = (1..=64).map(|i| bf16(i as f32 * 0.37)).collect();
    let weight = bf16_array(&w, &[8, 8]);
    let scalar = mlxcel_core::full_f32(&[], scale, dtype::FLOAT32);
    let got = values(&fold(&scalar, &weight));
    assert_eq!(
        mlxcel_core::array_dtype(&fold(&scalar, &weight)),
        dtype::BFLOAT16
    );
    let mut differs = 0;
    for (i, &wi) in w.iter().enumerate() {
        let want = bf16(bf16(scale) * wi);
        assert_eq!(got[i], want, "element {i}");
        if bf16(scale * wi) != want {
            differs += 1;
        }
    }
    // The f32 product rounded once is a different answer somewhere.
    assert!(differs > 0, "the fixture must separate the two roundings");
}

/// Reference RoPE in f32 over `pairs` rotating pairs, bf16 cos/sin.
fn reference_rope(x: &[f32], head_dim: usize, pos: f32, n_pairs: usize, pairs: usize) -> Vec<f32> {
    let mut out = x.to_vec();
    for p in 0..pairs {
        let exp = if n_pairs > 1 {
            p as f32 / (n_pairs - 1) as f32
        } else {
            0.0
        };
        let freq = (1.0f32 / 1024.0).powf(exp);
        let theta = pos * freq;
        let (c, s) = (bf16(theta.cos()), bf16(theta.sin()));
        for h in 0..x.len() / head_dim {
            let (a, b) = (x[h * head_dim + 2 * p], x[h * head_dim + 2 * p + 1]);
            out[h * head_dim + 2 * p] = a * c + b * s;
            out[h * head_dim + 2 * p + 1] = b * c - a * s;
        }
    }
    out
}

#[test]
fn partial_interleaved_rope_rotates_only_the_declared_pairs() {
    let (heads, head_dim, n_pairs) = (2usize, 128usize, 32usize);
    let rope = PartialRope::new(n_pairs as i32, head_dim as i32, 1024.0);
    let x: Vec<f32> = (0..heads * head_dim)
        .map(|i| ((i * 13 % 29) as f32 - 14.0) * 0.07)
        .collect();
    for pos in [0.0f32, 1.0, 7.0, 300.0] {
        let input = mlxcel_core::from_slice_f32(&x, &[1, 1, heads as i32, head_dim as i32]);
        let positions = positions_from_offset(pos as i32, 1);
        let got = values(&rope.apply(&input, &positions));
        let want = reference_rope(&x, head_dim, pos, n_pairs, n_pairs);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() < 1e-4, "pos {pos} lane {i}: {g} vs {w}");
        }
        // Lanes past the rotated half pass through unchanged.
        for h in 0..heads {
            for lane in 2 * n_pairs..head_dim {
                assert_eq!(got[h * head_dim + lane], x[h * head_dim + lane]);
            }
        }
        if pos > 0.0 {
            // The same rotation over all 64 pairs (frequencies spread over 64).
            let all = reference_rope(&x, head_dim, pos, head_dim / 2, head_dim / 2);
            let max_diff = got
                .iter()
                .zip(&all)
                .map(|(g, a)| (g - a).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff > 1e-3,
                "rotating all 64 pairs must differ (pos {pos})"
            );
        }
    }
    // The odd-lane sign: (1, 0) -> (cos, -sin).
    let mut unit = vec![0.0f32; head_dim];
    unit[0] = 1.0;
    let input = mlxcel_core::from_slice_f32(&unit, &[1, 1, 1, head_dim as i32]);
    let got = values(&rope.apply(&input, &positions_from_offset(1, 1)));
    assert!((got[0] - bf16(1.0f32.cos())).abs() < 1e-6);
    assert!((got[1] + bf16(1.0f32.sin())).abs() < 1e-6);
}

#[test]
fn mudd_accumulates_three_taps_left_to_right_in_the_activation_dtype() {
    let hidden = 4;
    // dense2 = 0 makes the per-tap weight exactly the bias.
    let dense1 = bf16_array(&[0.1; 8], &[2, hidden]);
    let x = bf16_array(&[1.0, 2.0, 3.0, 4.0], &[1, 1, hidden]);
    let way = |bias: [f32; 3]| MuddWay {
        dense2: bf16_array(&[0.0; 6], &[3, 2]),
        bias: bf16_array(&bias, &[3]),
    };
    // bf16 spacing at 256 is 2: (256 + 1) + 1 rounds to 256 twice, while a
    // stacked or right-to-left sum reaches 258.
    let v0 = bf16_array(&[256.0; 4], &[1, 1, hidden]);
    let v1 = bf16_array(&[1.0; 4], &[1, 1, hidden]);
    let v2 = bf16_array(&[1.0; 4], &[1, 1, hidden]);
    let got = values(&mudd_combine(
        &[&v0, &v1, &v2],
        &x,
        &dense1,
        &way([1.0, 1.0, 1.0]),
    ));
    assert_eq!(got, vec![256.0; 4], "left-to-right bf16 accumulation");
    let stacked = bf16(256.0 + 1.0 + 1.0);
    assert_eq!(stacked, 258.0);

    // Each tap takes its own weight.
    let v0 = bf16_array(&[2.0; 4], &[1, 1, hidden]);
    let v1 = bf16_array(&[3.0; 4], &[1, 1, hidden]);
    let v2 = bf16_array(&[5.0; 4], &[1, 1, hidden]);
    let got = values(&mudd_combine(
        &[&v0, &v1, &v2],
        &x,
        &dense1,
        &way([0.5, 2.0, -1.0]),
    ));
    assert_eq!(got, vec![2.0; 4]);
}

fn patterned(shape: &[i32], seed: usize, scale: f32) -> UniquePtr<MlxArray> {
    let len: i32 = shape.iter().product();
    let data: Vec<f32> = (0..len as usize)
        .map(|i| (((i * 37 + seed * 17) % 101) as f32 - 50.0) * scale)
        .collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

pub(super) fn tiny_config(sliding_window: usize) -> ModelArgs {
    serde_json::from_value(serde_json::json!({
        "model_type": "limite",
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "vocab_size": 16,
        "attn_gate_channels": 8,
        "ve_dim": 8,
        "ve_gate_channels": 4,
        "ve_stored_heads": 2,
        "ve_layers": [1],
        "xsa_layers": [0, 1, 2, 3],
        "global_layers": [3],
        "sliding_window": sliding_window,
        "rope_n_pairs": 2,
        "mudd_taps": 3,
        "mudd_inter": 4,
        "mudd_layers": [2, 3],
        "mudd_tap_idx": {"2": [0, 1, 2], "3": [0, 2, 3]},
        // The released head is (23, 5, 7.5); a smaller `a` keeps the logits
        // near 1 so the shared contracts' absolute 1e-5 tolerance measures
        // f32 rounding rather than the head's scale.
        "softcap_logits": {"kind": "sigmoid", "a": 2.0, "b": 0.5, "c": 1.5},
        "rope_style": "interleaved_pairs_odd_lane_sign_flip",
        "rms_norm_eps_mode": "torch_finfo_default",
        "eos_token_id": 15
    }))
    .expect("tiny Limite config")
}

pub(super) fn tiny_weights(args: &ModelArgs) -> WeightMap {
    let mut w = WeightMap::new();
    let (h, kv, d) = (16, 8, 8);
    let mut seed = 0;
    let mut put = |w: &mut WeightMap, name: String, shape: &[i32], scale: f32| {
        seed += 1;
        w.insert(name, patterned(shape, seed, scale));
    };
    put(&mut w, "model.embed_tokens.weight".into(), &[16, h], 0.02);
    put(
        &mut w,
        "model.value_embeds.weight".into(),
        &[16, 2 * d],
        0.02,
    );
    for l in 0..args.num_hidden_layers {
        let p = format!("model.layers.{l}");
        for (name, shape) in [
            ("self_attn.q_proj.weight", [h, h]),
            ("self_attn.k_proj.weight", [kv, h]),
            ("self_attn.v_proj.weight", [kv, h]),
            ("self_attn.o_proj.weight", [h, h]),
            ("mlp.gate_proj.weight", [32, h]),
            ("mlp.up_proj.weight", [32, h]),
            ("mlp.down_proj.weight", [h, 32]),
        ] {
            put(&mut w, format!("{p}.{name}"), &shape, 0.004);
        }
        put(&mut w, format!("{p}.self_attn.attn_gate"), &[2, 8], 0.01);
        put(&mut w, format!("{p}.self_attn.ve_gate"), &[2, 4], 0.01);
        put(&mut w, format!("{p}.self_attn.xsa_alpha"), &[2], 0.01);
        for (name, value) in [
            ("self_attn.qkv_scale", 1.3f32),
            ("self_attn.o_scale", 0.9),
            ("resid_lambda_attn", 1.0),
            ("post_lambda_attn", 0.6),
            ("resid_lambda_mlp", 1.0),
            ("post_lambda_mlp", 0.4),
        ] {
            w.insert(
                format!("{p}.{name}"),
                mlxcel_core::full_f32(&[], value, dtype::FLOAT32),
            );
        }
    }
    put(&mut w, "model.mudd.dense1".into(), &[4, h], 0.01);
    put(&mut w, "model.mudd.dense2".into(), &[4, 3, 4], 0.01);
    put(&mut w, "model.mudd.dense2_mlp".into(), &[4, 3, 4], 0.01);
    w.insert(
        "model.mudd.bias".into(),
        mlxcel_core::from_slice_f32(&[0.4, 0.3, 0.3].repeat(4), &[4, 3]),
    );
    w.insert(
        "model.mudd.bias_mlp".into(),
        mlxcel_core::from_slice_f32(&[0.2, 0.2, 0.6].repeat(4), &[4, 3]),
    );
    w
}

pub(super) fn tiny_wrapper() -> LimiteWrapper {
    let args = tiny_config(3);
    LimiteWrapper::new(LimiteModel::from_weights(&tiny_weights(&args), &args).unwrap())
}

#[test]
fn sliding_cache_keeps_sliding_window_plus_one_keys() {
    let args = tiny_config(1024);
    let model = LimiteWrapper::new(LimiteModel::from_weights(&tiny_weights(&args), &args).unwrap());
    let id = SequenceId::from_raw(3);
    model.prepare_sequence_state(id);
    let prompt: Vec<i32> = (0..1100).map(|i| i % 15).collect();
    let input = mlxcel_core::from_slice_i32(&prompt, &[1, prompt.len() as i32]);
    mlxcel_core::eval(&model.forward_with_sequence_id(&input, Some(id), &mut [], None));
    let step = mlxcel_core::from_slice_i32(&[4], &[1, 1]);
    mlxcel_core::eval(&model.forward_with_sequence_id(&step, Some(id), &mut [], None));
    let lens = model.sequence_visible_lens(id);
    // Layers 0-2 slide (k >= q - 1024 inclusive: 1025 keys); layer 3 is global.
    assert_eq!(lens, vec![1025, 1025, 1025, 1101], "visible keys per layer");
    model.release_sequence_state_by_id(id);
}

#[test]
fn head_returns_sigmoid_softcapped_values() {
    let model = tiny_wrapper();
    let input = mlxcel_core::from_slice_i32(&[1, 2, 3], &[1, 3]);
    let logits = model.forward_with_sequence_id(&input, None, &mut [], None);
    assert_eq!(mlxcel_core::array_dtype(&logits), dtype::FLOAT32);
    let v = values(&logits);
    assert_eq!(v.len(), 3 * 16);
    assert!(v.iter().all(|&x| x > 0.0 && x < 2.0), "{v:?}");
}

#[test]
fn config_refuses_conventions_it_does_not_implement() {
    let base = serde_json::to_value(serde_json::json!({
        "model_type": "limite", "hidden_size": 16, "intermediate_size": 32,
        "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 8, "vocab_size": 16, "rope_n_pairs": 2,
        "attn_gate_channels": 8, "ve_gate_channels": 4
    }))
    .unwrap();
    let parse = |patch: serde_json::Value| {
        let mut v = base.clone();
        for (k, val) in patch.as_object().unwrap() {
            v[k] = val.clone();
        }
        serde_json::from_value::<ModelArgs>(v).unwrap().validate()
    };
    assert!(parse(serde_json::json!({})).is_ok());
    for patch in [
        serde_json::json!({"rope_style": "neox"}),
        serde_json::json!({"rms_norm_eps_mode": "fixed_1e-6"}),
        serde_json::json!({"sliding_window_convention": "k > q - sliding_window"}),
        serde_json::json!({"mudd_accumulation": "stacked"}),
        serde_json::json!({"rms_norm_has_weight": true}),
        serde_json::json!({"rope_per_layer": true}),
        serde_json::json!({"mlp_type": "relu2"}),
        serde_json::json!({"softcap_logits": {"kind": "tanh", "a": 1.0}}),
        serde_json::json!({"global_window": 512}),
        serde_json::json!({"quantization": {"group_size": 64, "bits": 4}}),
        serde_json::json!({"mudd_layers": [2], "mudd_tap_idx": {"2": [0, 3]}}),
        serde_json::json!({"mudd_layers": [2], "mudd_tap_idx": {"3": [0]}}),
        serde_json::json!({"mudd": false, "mudd_layers": [2], "mudd_tap_idx": {"2": [0, 2]}}),
    ] {
        assert!(parse(patch.clone()).is_err(), "{patch} must be refused");
    }
}

#[test]
fn limite_native_isolation() {
    contracts(tiny_wrapper, |m, id| m.sequence_offsets(id));
}

last_row_case!(limite_last_row_logits, tiny_wrapper);
