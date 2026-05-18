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

//! Unit tests for [`super::ScaleOp`] against a synthetic
//! [`mlxcel_core::weights::WeightMap`].
//!
//! Covers acceptance criterion (a):
//! - matched tensors are multiplied,
//! - non-matching tensors are left untouched,
//! - dtype and shape are preserved,
//! - zero-match errors out,
//! - quantized triplets route to scales/biases rather than the
//!   packed payload.

use super::*;
use mlxcel_core::dtype;
use mlxcel_core::MlxArray;
use mlxcel_core::UniquePtr;
use std::sync::Arc;

/// Construct a tiny f32 tensor from a literal slice. The shape is
/// 1-D for simplicity.
fn f32_tensor(values: &[f32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_f32(values, &[values.len() as i32])
}

/// Read a 1-D f32 tensor back to a `Vec<f32>` for value assertions.
fn read_f32(tensor: &MlxArray) -> Vec<f32> {
    mlxcel_core::eval(tensor);
    let bytes = mlxcel_core::array_to_raw_bytes(tensor);
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("4-byte chunk")))
        .collect()
}

/// Read a 1-D f16 tensor back to a `Vec<f32>` (f16 reinterpreted to
/// f32). f16 has no native Rust type, so we cast via the host
/// representation `u16 -> f32`.
fn read_f16(tensor: &MlxArray) -> Vec<f32> {
    mlxcel_core::eval(tensor);
    let bytes = mlxcel_core::array_to_raw_bytes(tensor);
    bytes
        .chunks_exact(2)
        .map(|chunk| f16_bits_to_f32(u16::from_ne_bytes(chunk.try_into().expect("2-byte chunk"))))
        .collect()
}

/// Convert IEEE-754 binary16 (half precision) bits to f32 — the
/// minimal portable implementation that does not require the
/// `half` crate.
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 0x1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        if mantissa == 0 {
            sign << 31
        } else {
            // Subnormal — normalize.
            let mut m = mantissa;
            let mut e = 1u32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            (sign << 31) | ((127u32 - 15 + 1 - e) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xffu32 << 23) | (mantissa << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mantissa << 13)
    };
    f32::from_bits(f32_bits)
}

#[test]
fn rejects_non_finite_factor() {
    let err = ScaleOp::new("*", f32::NAN).expect_err("NaN must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("scale") && msg.contains("finite"),
        "error must mention finite: {msg}"
    );

    let err = ScaleOp::new("*", f32::INFINITY).expect_err("inf must be rejected");
    let msg = format!("{err}");
    assert!(msg.contains("finite"), "error must mention finite: {msg}");
}

#[test]
fn rejects_malformed_glob() {
    let err = ScaleOp::new("model.layers.{0", 1.0).expect_err("bad glob must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("glob") || msg.contains("pattern"),
        "error must mention glob: {msg}"
    );
}

#[test]
fn from_spec_accepts_scale_variant_only() {
    let op = ScaleOp::from_spec(OpSpec::Scale {
        pattern: "*".to_string(),
        factor: 2.5,
    })
    .expect("scale variant accepted");
    assert_eq!(op.name(), "scale");
    assert_eq!(op.factor(), 2.5);
    assert_eq!(op.pattern(), "*");

    let err = ScaleOp::from_spec(OpSpec::Add {
        pattern: "*".to_string(),
        source: "/tmp/x".into(),
        alpha: 1.0,
    })
    .expect_err("add variant must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("from_spec") || msg.contains("non-Scale"),
        "error must mention misuse: {msg}"
    );
}

#[test]
fn zero_match_returns_error() {
    let op = ScaleOp::new("nothing.matches.this.pattern", 2.0).expect("construct");
    let mut weights = WeightMap::new();
    weights.insert(
        "model.embed_tokens.weight".to_string(),
        f32_tensor(&[1.0, 2.0]),
    );

    let err = op
        .apply(&mut weights, &serde_json::Value::Null)
        .expect_err("zero match must error");
    let msg = format!("{err}");
    assert!(
        msg.contains("matched zero tensors") || msg.contains("zero"),
        "zero-match error message: {msg}"
    );

    // Original tensor must be untouched.
    let after = weights.get("model.embed_tokens.weight").unwrap();
    assert_eq!(read_f32(after), vec![1.0, 2.0]);
}

#[test]
fn scales_matched_tensors_and_leaves_others_alone() {
    let op = ScaleOp::new("model.layers.0.self_attn.o_proj.weight", 2.0).expect("construct");
    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.self_attn.o_proj.weight".to_string(),
        f32_tensor(&[1.0, -2.0, 3.0, -4.0]),
    );
    weights.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        f32_tensor(&[10.0, 20.0]),
    );
    weights.insert(
        "model.embed_tokens.weight".to_string(),
        f32_tensor(&[100.0]),
    );

    op.apply(&mut weights, &serde_json::Value::Null)
        .expect("apply succeeds");

    let scaled = weights
        .get("model.layers.0.self_attn.o_proj.weight")
        .unwrap();
    assert_eq!(read_f32(scaled), vec![2.0, -4.0, 6.0, -8.0]);
    assert_eq!(mlxcel_core::array_dtype(scaled), dtype::FLOAT32);
    assert_eq!(mlxcel_core::array_shape(scaled), vec![4]);

    // Other tensors must be bit-identical to their input.
    let q = weights
        .get("model.layers.0.self_attn.q_proj.weight")
        .unwrap();
    assert_eq!(read_f32(q), vec![10.0, 20.0]);
    let embed = weights.get("model.embed_tokens.weight").unwrap();
    assert_eq!(read_f32(embed), vec![100.0]);
}

#[test]
fn glob_wildcard_matches_multiple_layers() {
    let op = ScaleOp::new("model.layers.*.self_attn.o_proj.weight", 0.5).expect("construct");
    let mut weights = WeightMap::new();
    for layer in 0..3 {
        weights.insert(
            format!("model.layers.{layer}.self_attn.o_proj.weight"),
            f32_tensor(&[2.0, 4.0]),
        );
        // Sibling tensor that must remain untouched.
        weights.insert(
            format!("model.layers.{layer}.self_attn.q_proj.weight"),
            f32_tensor(&[10.0, 20.0]),
        );
    }

    op.apply(&mut weights, &serde_json::Value::Null)
        .expect("apply");

    for layer in 0..3 {
        let o = weights
            .get(&format!("model.layers.{layer}.self_attn.o_proj.weight"))
            .unwrap();
        assert_eq!(read_f32(o), vec![1.0, 2.0], "layer {layer} o_proj");
        let q = weights
            .get(&format!("model.layers.{layer}.self_attn.q_proj.weight"))
            .unwrap();
        assert_eq!(read_f32(q), vec![10.0, 20.0], "layer {layer} q_proj");
    }
}

#[test]
fn preserves_dtype_and_shape_for_f16_tensor() {
    let op = ScaleOp::new("*", 3.0).expect("construct");
    // Build a 2-D f16 tensor by casting an f32 array.
    let f32_arr = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let f16_arr = mlxcel_core::astype(&f32_arr, dtype::FLOAT16);
    let mut weights = WeightMap::new();
    weights.insert("model.layer.weight".to_string(), f16_arr);

    op.apply(&mut weights, &serde_json::Value::Null)
        .expect("apply");

    let after = weights.get("model.layer.weight").unwrap();
    assert_eq!(
        mlxcel_core::array_dtype(after),
        dtype::FLOAT16,
        "dtype must remain FLOAT16"
    );
    assert_eq!(
        mlxcel_core::array_shape(after),
        vec![2, 2],
        "shape must remain (2, 2)"
    );
    let values = read_f16(after);
    assert_eq!(values, vec![3.0, 6.0, 9.0, 12.0]);
}

#[test]
fn quantized_affine_routes_to_scales_and_biases_not_packed_codes() {
    // Synthetic quantized layer:
    //   model.layer.weight  = packed payload (u32 codes — keep verbatim)
    //   model.layer.scales  = f32 (real layout is bf16 but values are
    //                          carried as fp; the routing logic is
    //                          dtype-agnostic)
    //   model.layer.biases  = f32 (ditto)
    // A user pattern that matches the `.weight` key MUST scale
    // scales+biases instead, and leave the packed codes untouched.
    let op = ScaleOp::new("model.layer.weight", 4.0).expect("construct");
    let mut weights = WeightMap::new();
    let packed_codes: Vec<u32> = vec![0xDEAD_BEEF, 0x1234_5678, 0x0000_0001, 0xFFFF_FFFF];
    weights.insert(
        "model.layer.weight".to_string(),
        mlxcel_core::from_slice_u32(&packed_codes, &[packed_codes.len() as i32]),
    );
    weights.insert(
        "model.layer.scales".to_string(),
        f32_tensor(&[0.25, 0.5, 1.0, 2.0]),
    );
    weights.insert("model.layer.biases".to_string(), f32_tensor(&[10.0, -20.0]));

    op.apply(&mut weights, &serde_json::Value::Null)
        .expect("apply");

    // Packed codes — bit-identical.
    let packed = weights.get("model.layer.weight").unwrap();
    mlxcel_core::eval(packed);
    let packed_bytes = mlxcel_core::array_to_raw_bytes(packed);
    let recovered: Vec<u32> = packed_bytes
        .chunks_exact(4)
        .map(|c| u32::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(
        recovered, packed_codes,
        "packed quantized codes must not be touched"
    );

    // Scales — multiplied by 4.0.
    let scales = weights.get("model.layer.scales").unwrap();
    assert_eq!(read_f32(scales), vec![1.0, 2.0, 4.0, 8.0]);

    // Biases — multiplied by 4.0.
    let biases = weights.get("model.layer.biases").unwrap();
    assert_eq!(read_f32(biases), vec![40.0, -80.0]);
}

#[test]
fn quantized_mxfp4_layer_without_biases_scales_only_scales() {
    // mxfp4 / nvfp4 / mxfp8 do not carry a `.biases` tensor.
    // Pattern matches the packed `.weight`, must scale only
    // `.scales`, and the absence of biases must not error out.
    let op = ScaleOp::new("model.embed.weight", 0.5).expect("construct");
    let mut weights = WeightMap::new();
    weights.insert(
        "model.embed.weight".to_string(),
        mlxcel_core::from_slice_u32(&[0xCAFE_F00D], &[1]),
    );
    weights.insert(
        "model.embed.scales".to_string(),
        f32_tensor(&[2.0, -4.0, 8.0]),
    );

    op.apply(&mut weights, &serde_json::Value::Null)
        .expect("apply");

    let packed = weights.get("model.embed.weight").unwrap();
    mlxcel_core::eval(packed);
    let packed_bytes = mlxcel_core::array_to_raw_bytes(packed);
    let recovered: Vec<u32> = packed_bytes
        .chunks_exact(4)
        .map(|c| u32::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(recovered, vec![0xCAFE_F00D]);

    let scales = weights.get("model.embed.scales").unwrap();
    assert_eq!(read_f32(scales), vec![1.0, -2.0, 4.0]);
}

#[test]
fn pattern_matching_scales_directly_is_pass_through() {
    // When the user explicitly targets a `.scales` key (no sibling
    // packed `.weight` in the same prefix), the op must scale it
    // directly. This exercises the "scales without a matching
    // weight" fallback in resolve_effective_targets.
    let op = ScaleOp::new("model.layer.scales", 10.0).expect("construct");
    let mut weights = WeightMap::new();
    weights.insert(
        "model.layer.scales".to_string(),
        f32_tensor(&[0.1, 0.2, 0.3]),
    );

    op.apply(&mut weights, &serde_json::Value::Null)
        .expect("apply");

    let after = weights.get("model.layer.scales").unwrap();
    // Note: f32 multiply has rounding; assert with tolerance.
    let values = read_f32(after);
    assert_eq!(values.len(), 3);
    for (got, want) in values.iter().zip([1.0_f32, 2.0, 3.0].iter()) {
        assert!(
            (got - want).abs() < 1e-5,
            "expected {want}, got {got} (diff={})",
            (got - want).abs()
        );
    }
}

#[test]
fn wildcard_matching_quantized_triplet_does_not_double_scale() {
    // Pattern `model.layer.*` matches `.weight`, `.scales`, and
    // `.biases`. The packed `.weight` route resolves to {scales,
    // biases}; the standalone `.scales` and `.biases` matches
    // resolve to themselves. The dedup logic must collapse to a
    // single scale of each metadata tensor — NOT double-scale.
    let op = ScaleOp::new("model.layer.*", 2.0).expect("construct");
    let mut weights = WeightMap::new();
    weights.insert(
        "model.layer.weight".to_string(),
        mlxcel_core::from_slice_u32(&[0xABCD_1234], &[1]),
    );
    weights.insert("model.layer.scales".to_string(), f32_tensor(&[3.0]));
    weights.insert("model.layer.biases".to_string(), f32_tensor(&[-5.0]));

    op.apply(&mut weights, &serde_json::Value::Null)
        .expect("apply");

    let scales = weights.get("model.layer.scales").unwrap();
    assert_eq!(
        read_f32(scales),
        vec![6.0],
        "scales must be scaled exactly once, not twice"
    );
    let biases = weights.get("model.layer.biases").unwrap();
    assert_eq!(
        read_f32(biases),
        vec![-10.0],
        "biases must be scaled exactly once, not twice"
    );

    // Packed codes are bit-identical.
    let packed = weights.get("model.layer.weight").unwrap();
    mlxcel_core::eval(packed);
    let bytes = mlxcel_core::array_to_raw_bytes(packed);
    let codes: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(codes, vec![0xABCD_1234]);
}

#[test]
fn factor_one_is_a_value_preserving_pass() {
    // `factor=1.0` is a no-op in math but still has to go through
    // the FFI multiply path — this regression-pins the case where
    // a user "disables" a scale op by setting `1.0` without
    // commenting it out. The values should remain bit-identical
    // after one f32 multiply by 1.0 (per IEEE-754, x * 1.0 == x
    // for all finite x).
    let op = ScaleOp::new("*", 1.0).expect("construct");
    let mut weights = WeightMap::new();
    weights.insert(
        "model.layer.weight".to_string(),
        f32_tensor(&[1.5, -2.5, 0.0, f32::MIN_POSITIVE]),
    );

    op.apply(&mut weights, &serde_json::Value::Null)
        .expect("apply");

    let after = weights.get("model.layer.weight").unwrap();
    let values = read_f32(after);
    assert_eq!(
        values,
        vec![1.5, -2.5, 0.0, f32::MIN_POSITIVE],
        "x * 1.0 must be bit-exact x for finite f32"
    );
}

#[test]
fn scaleop_is_send_and_sync_for_pipeline_sharing() {
    // Compile-time witness that ScaleOp meets the SurgeryOp
    // bound. If this ever stops compiling, the pipeline's
    // `Arc<dyn SurgeryOp>` storage would break.
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ScaleOp>();

    // And that it really materializes through the trait object.
    let op = ScaleOp::new("*", 1.0).expect("construct");
    let _arc: Arc<dyn SurgeryOp> = Arc::new(op);
}
