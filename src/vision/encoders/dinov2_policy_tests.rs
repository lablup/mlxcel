// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
#![allow(clippy::excessive_precision)]
use super::super::{Attention, Mlp};
use super::*;
use anyhow::Result;
use mlxcel_core::{self as mx, layers::Linear, weights::WeightMap};

// Authenticated MLX 0.32.4 oracle, already used by Sapiens2. The source JSON
// SHA256 is ab93240df9362f37bdc00fa5a179cf850630af7a9903c13d2eae52a2098e35b8,
// pinned in tests/fixtures/sapiens2/reference-output-hashes.json. No new oracle
// values are inferred from the implementation under test.
const INPUT: [f32; 4] = [-0.447265625, 0.84375, 1.140625, -0.08935546875];
const WEIGHT: [f32; 12] = [
    0.80078125,
    -0.1640625,
    0.267578125,
    -1.3125,
    0.76171875,
    -0.73046875,
    0.97265625,
    -0.57421875,
    1.015625,
    -1.8203125,
    2.765625,
    0.6484375,
];
const BIAS: [f32; 3] = [1.3671875, -0.70703125, 0.82421875];
const FUSED: [f32; 3] = [1.296875, -0.50390625, 1.9296875];
const SEPARATE: [f32; 3] = [1.296875, -0.50390625, 1.9375];

fn bf16(values: &[f32], shape: &[i32]) -> UniquePtr<MlxArray> {
    mx::astype(&mx::from_slice_f32(values, shape), mx::dtype::BFLOAT16)
}
fn exact(actual: &MlxArray, expected: &[f32]) {
    assert_eq!(mx::array_dtype(actual), mx::dtype::BFLOAT16);
    assert_eq!(mx::utils::array_to_vec_f32(actual), expected);
}
fn padded(values: &[f32], length: usize) -> Vec<f32> {
    let mut out = values.to_vec();
    out.resize(length, 0.0);
    out
}
fn identity() -> Vec<f32> {
    (0..16)
        .map(|i| if i / 4 == i % 4 { 1.0 } else { 0.0 })
        .collect()
}
fn put(w: &mut WeightMap, name: &str, values: &[f32], shape: &[i32]) {
    w.insert(name.into(), bf16(values, shape));
}

#[test]
fn objects_dino_linear_matches_authentic_bf16_addmm_oracle() {
    let layer = Linear::new(bf16(&WEIGHT, &[3, 4]), Some(bf16(&BIAS, &[3])));
    let x = bf16(&INPUT, &[1, 4]);
    exact(&linear_with_policy(&layer, &x, true), &FUSED);
    exact(&linear_with_policy(&layer, &x, false), &SEPARATE);
    // Bias-free layers retain matmul semantics in either policy.
    let unbiased = Linear::new(bf16(&WEIGHT, &[3, 4]), None);
    assert_eq!(
        mx::array_to_raw_bytes(&linear_with_policy(&unbiased, &x, true)),
        mx::array_to_raw_bytes(&unbiased.forward(&x))
    );
}

fn attention_routing(qkv_is_oracle: bool) -> Result<()> {
    let mut w = WeightMap::new();
    let mut qkv_weight = vec![0.; 48];
    let mut qkv_bias = vec![0.; 12];
    if qkv_is_oracle {
        qkv_weight[32..44].copy_from_slice(&WEIGHT);
        qkv_bias[8..11].copy_from_slice(&BIAS);
    } else {
        qkv_bias[8..12].copy_from_slice(&INPUT);
    }
    put(&mut w, "a.qkv.weight", &qkv_weight, &[12, 4]);
    put(&mut w, "a.qkv.bias", &qkv_bias, &[12]);
    let proj_weight = if qkv_is_oracle {
        identity()
    } else {
        padded(&WEIGHT, 16)
    };
    let proj_bias = if qkv_is_oracle {
        vec![0.; 4]
    } else {
        padded(&BIAS, 4)
    };
    put(&mut w, "a.proj.weight", &proj_weight, &[4, 4]);
    put(&mut w, "a.proj.bias", &proj_bias, &[4]);
    let attention = Attention::from_weights(&w, "a", 4, 1, true).map_err(anyhow::Error::msg)?;
    let x = bf16(&INPUT, &[1, 1, 4]);
    // One-token attention returns V exactly, isolating the selected projection.
    exact(&*attention.forward_objects(&x)?, &padded(&FUSED, 4));
    exact(&*attention.forward(&x)?, &padded(&SEPARATE, 4));
    Ok(())
}
#[test]
fn objects_dino_attention_routes_qkv_through_addmm() -> Result<()> {
    attention_routing(true)
}
#[test]
fn objects_dino_attention_routes_proj_through_addmm() -> Result<()> {
    attention_routing(false)
}

#[test]
fn objects_dino_mlp_routes_fc1_through_addmm() -> Result<()> {
    let mut w = WeightMap::new();
    put(&mut w, "m.fc1.weight", &padded(&WEIGHT, 16), &[4, 4]);
    put(&mut w, "m.fc1.bias", &padded(&BIAS, 4), &[4]);
    put(&mut w, "m.fc2.weight", &identity(), &[4, 4]);
    put(&mut w, "m.fc2.bias", &[0.; 4], &[4]);
    let mlp = Mlp::from_weights(&w, "m", 4, 4).map_err(anyhow::Error::msg)?;
    let expected = mx::compiled_objects_gelu(&bf16(&padded(&FUSED, 4), &[1, 1, 4]))?;
    let wrong = mx::compiled_objects_gelu(&bf16(&padded(&SEPARATE, 4), &[1, 1, 4]))?;
    let actual = mlp.forward_objects(&bf16(&INPUT, &[1, 1, 4]))?;
    assert_eq!(
        mx::array_to_raw_bytes(&actual),
        mx::array_to_raw_bytes(&expected)
    );
    assert_ne!(
        mx::array_to_raw_bytes(&actual),
        mx::array_to_raw_bytes(&wrong)
    );
    Ok(())
}

#[test]
fn objects_dino_mlp_routes_fc2_through_addmm() -> Result<()> {
    let mut w = WeightMap::new();
    put(&mut w, "m.fc1.weight", &[0.; 16], &[4, 4]);
    put(&mut w, "m.fc1.bias", &[8., 16., 32., 64.], &[4]);
    let mut weight = [0.; 16];
    weight[0] = 0.125;
    weight[1] = 1. / 4096.;
    put(&mut w, "m.fc2.weight", &weight, &[4, 4]);
    put(&mut w, "m.fc2.bias", &[1. / 256., 0., 0., 0.], &[4]);
    let mlp = Mlp::from_weights(&w, "m", 4, 4).map_err(anyhow::Error::msg)?;
    // GELU is exactly x at these BF16 inputs; dot=1+1/256 is a halfway
    // rounding case. Fused bias yields 1+1/128; separate rounding yields 1.
    let activation = mx::compiled_objects_gelu(&bf16(&[8., 16., 32., 64.], &[1, 1, 4]))?;
    exact(&activation, &[8., 16., 32., 64.]);
    exact(&mlp.fc2.forward(&activation), &[1., 0., 0., 0.]);
    exact(
        &*mlp.forward_objects(&bf16(&[0.; 4], &[1, 1, 4]))?,
        &[1.0078125, 0., 0., 0.],
    );
    Ok(())
}
