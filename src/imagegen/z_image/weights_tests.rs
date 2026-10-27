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

use std::path::PathBuf;

use super::*;
use crate::imagegen::testing::det;
use crate::imagegen::vae::{Flux2Vae, Flux2VaeConfig};
use mlxcel_core::dtype;
use mlxcel_core::weights::load_safetensors;
use serde_json::json;

fn meta(v: Value) -> Map<String, Value> {
    v.as_object().unwrap().clone()
}

fn max_abs_diff(a: &mlxcel_core::MlxArray, b: &mlxcel_core::MlxArray) -> f32 {
    let a = mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(a, dtype::FLOAT32));
    let b = mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(b, dtype::FLOAT32));
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn quantization_metadata_follows_upstream_table() {
    let ok = |mode: &str, bits: &str, group: &str| {
        quant_spec(&meta(json!({
            "quantization_mode": mode, "quantization_level": bits,
            "quantization_group_size": group
        })))
    };
    assert_eq!(
        ok("affine", "4", "64").unwrap(),
        QuantSpec {
            mode: "affine".into(),
            bits: 4,
            group_size: 64
        }
    );
    assert!(ok("affine", "8", "32").is_ok());
    assert!(ok("mxfp4", "4", "32").is_ok());
    assert!(ok("mxfp8", "8", "32").is_ok());
    assert!(ok("nvfp4", "4", "16").is_ok());
    for (mode, bits, group) in [
        ("mxfp4", "4", "64"),
        ("nvfp4", "4", "32"),
        ("mxfp8", "4", "32"),
        ("gguf", "4", "32"),
        ("", "0", "0"),
    ] {
        let err = ok(mode, bits, group).unwrap_err().to_string();
        assert!(
            err.contains("Unsupported Z-Image quantization metadata"),
            "{err}"
        );
    }
    // Numbers are accepted as well as the strings upstream writes.
    assert!(
        quant_spec(&meta(json!({
            "quantization_mode": "affine", "quantization_level": 4,
            "quantization_group_size": 64
        })))
        .is_ok()
    );
}

#[test]
fn dequantize_replaces_packed_triples_and_keeps_dense_tensors() {
    let dense = det("layers.0.attention.to_q.weight", &[16, 64], 0.5, 0.0);
    let q = mlxcel_core::quantize_weights_with_mode(&dense, 32, 8, "affine");
    let mut weights = WeightMap::new();
    weights.insert(
        "layers.0.attention.to_q.weight".into(),
        mlxcel_core::quantized_weights_w(&q),
    );
    weights.insert(
        "layers.0.attention.to_q.scales".into(),
        mlxcel_core::quantized_weights_scales(&q),
    );
    weights.insert(
        "layers.0.attention.to_q.biases".into(),
        mlxcel_core::quantized_weights_biases(&q),
    );
    weights.insert("x_pad_token".into(), det("x_pad_token", &[1, 16], 1.0, 0.0));
    let metadata = meta(json!({
        "mlx_vlm_format": "z_image", "quantization_mode": "affine",
        "quantization_level": "8", "quantization_group_size": "32"
    }));
    let out = dequantize_weights(weights, &metadata).unwrap();
    let mut keys: Vec<_> = out.keys().cloned().collect();
    keys.sort();
    assert_eq!(keys, ["layers.0.attention.to_q.weight", "x_pad_token"]);
    let restored = &out["layers.0.attention.to_q.weight"];
    assert_eq!(mlxcel_core::array_shape(restored), [16, 64]);
    assert!(max_abs_diff(restored, &dense) < 0.01);

    // A quantized tensor with bad metadata is an error, a dense map is not.
    let mut bad = WeightMap::new();
    bad.insert("a.weight".into(), mlxcel_core::quantized_weights_w(&q));
    bad.insert("a.scales".into(), mlxcel_core::quantized_weights_scales(&q));
    assert!(dequantize_weights(bad, &Map::new()).is_err());
    let mut plain = WeightMap::new();
    plain.insert("a.weight".into(), mlxcel_core::copy(&dense));
    assert_eq!(dequantize_weights(plain, &Map::new()).unwrap().len(), 1);
}

#[test]
fn index_metadata_marks_a_converted_directory() {
    let dir = tempfile::TempDir::new().unwrap();
    assert!(index_metadata(dir.path()).unwrap().is_empty());
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        r#"{"metadata": {"mlx_vlm_format": "z_image", "quantization_level": "4"}, "weight_map": {}}"#,
    )
    .unwrap();
    let metadata = index_metadata(dir.path()).unwrap();
    assert!(is_mlx_vlm_format(&metadata));
    assert!(!is_mlx_vlm_format(&meta(json!({"format": "pt"}))));
}

fn vae_fixture() -> WeightMap {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/imagegen/vae_tiny.safetensors");
    load_safetensors(path)
        .unwrap()
        .into_iter()
        .filter(|(k, _)| !k.starts_with("ref."))
        .collect()
}

fn vae_refs() -> WeightMap {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/imagegen/vae_tiny.safetensors");
    load_safetensors(path)
        .unwrap()
        .into_iter()
        .filter_map(|(k, v)| k.strip_prefix("ref.").map(|k| (k.to_string(), v)))
        .collect()
}

fn tiny_vae_config() -> Flux2VaeConfig {
    Flux2VaeConfig {
        latent_channels: 4,
        block_out_channels: vec![4, 4, 8, 8],
        norm_num_groups: 2,
        ..Flux2VaeConfig::default()
    }
}

#[test]
fn vae_sanitize_is_a_no_op_on_a_converted_mlx_layout_directory() {
    // An mlx-vlm converted VAE: MLX conv layout and upstream's `to_out` name.
    let converted: WeightMap = vae_fixture()
        .into_iter()
        .map(|(k, v)| {
            let v = if mlxcel_core::array_shape(&v).len() == 4 {
                mlxcel_core::transpose_axes(&v, &[0, 2, 3, 1])
            } else {
                v
            };
            (k.replace(".to_out.0.", ".to_out."), v)
        })
        .collect();
    let shapes: std::collections::BTreeMap<String, Vec<i32>> = converted
        .iter()
        .map(|(k, v)| (k.clone(), mlxcel_core::array_shape(v)))
        .collect();
    let sanitized = sanitize_vae_weights(converted);
    let after: std::collections::BTreeMap<String, Vec<i32>> = sanitized
        .iter()
        .map(|(k, v)| (k.clone(), mlxcel_core::array_shape(v)))
        .collect();
    assert_eq!(after, shapes, "keys and shapes must be untouched");
    // The VAE then decodes it exactly like the PyTorch-layout original.
    let refs = vae_refs();
    let vae =
        Flux2Vae::from_raw_weights(tiny_vae_config(), sanitized, true, dtype::FLOAT32).unwrap();
    let decoded = vae.decode(&refs["latents"]).unwrap();
    assert!(max_abs_diff(&decoded, &refs["decode"]) < 1e-3);
}

#[test]
fn vae_sanitize_renames_mflux_style_paths() {
    let mut weights = WeightMap::new();
    for key in [
        "encoder.conv_in.conv2d.weight",
        "decoder.conv_out.conv.bias",
        "decoder.conv_norm_out.norm.weight",
    ] {
        weights.insert(key.into(), mlxcel_core::zeros(&[1], dtype::FLOAT32));
    }
    let mut keys: Vec<_> = sanitize_vae_weights(weights).into_keys().collect();
    keys.sort();
    assert_eq!(
        keys,
        [
            "decoder.conv_norm_out.weight",
            "decoder.conv_out.bias",
            "encoder.conv_in.weight"
        ]
    );
}

#[test]
fn vae_encoder_without_quant_conv_returns_the_moments_mean() {
    // `use_quant_conv: false`: dropping an identity quant_conv must not change
    // the encoding.
    let refs = vae_refs();
    let mut identity = vae_fixture();
    let channels = mlxcel_core::array_shape(&identity["quant_conv.weight"])[0];
    let eye: Vec<f32> = (0..channels * channels)
        .map(|i| {
            if i / channels == i % channels {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    identity.insert(
        "quant_conv.weight".into(),
        mlxcel_core::from_slice_f32(&eye, &[channels, channels, 1, 1]),
    );
    identity.insert(
        "quant_conv.bias".into(),
        mlxcel_core::zeros(&[channels], dtype::FLOAT32),
    );
    let mut without = vae_fixture();
    without.remove("quant_conv.weight");
    without.remove("quant_conv.bias");
    let with_id =
        Flux2Vae::from_raw_weights(tiny_vae_config(), identity, true, dtype::FLOAT32).unwrap();
    let none =
        Flux2Vae::from_raw_weights(tiny_vae_config(), without, true, dtype::FLOAT32).unwrap();
    let a = with_id.encode(&refs["image"]).unwrap();
    let b = none.encode(&refs["image"]).unwrap();
    assert_eq!(mlxcel_core::array_shape(&b), [1, 4, 4, 4]);
    assert!(max_abs_diff(&a, &b) < 1e-5);
}

/// The reason for [`WEIGHT_DTYPE`]: f16 with f32 promotes to f32 on every
/// backend, while the CUDA build's patched promotion table gives bf16 for
/// bf16 with f32.
#[test]
fn f16_weights_promote_with_f32_to_f32() {
    let x = mlxcel_core::from_slice_f32(&[1.5, -2.25], &[2]);
    let promoted = |dt: i32| {
        let w = mlxcel_core::astype(&x, dt);
        mlxcel_core::array_dtype(&mlxcel_core::multiply(&w, &x))
    };
    assert_eq!(promoted(dtype::FLOAT16), dtype::FLOAT32);
    #[cfg(feature = "cuda")]
    assert_eq!(promoted(dtype::BFLOAT16), dtype::BFLOAT16);
}

#[test]
fn f16_round_trip_error_measures_range_and_subnormal_loss() {
    let exact = mlxcel_core::astype(&det("w", &[64], 2.0, 0.0), dtype::BFLOAT16);
    assert_eq!(f16_round_trip_error(&exact), 0.0);
    assert!(fits_f16(&exact));
    // Above f16's 65504: the cast overflows to infinity.
    let big = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&[1.0, 7.0e4], &[2]),
        dtype::BFLOAT16,
    );
    assert!(f16_round_trip_error(&big).is_infinite());
    assert!(!fits_f16(&big));
    // Below f16's smallest subnormal (6e-8): flushed to zero.
    let tiny = mlxcel_core::astype(
        &mlxcel_core::full_f32(&[8], 1.0e-8, dtype::FLOAT32),
        dtype::BFLOAT16,
    );
    assert!(f16_round_trip_error(&tiny) > 0.5);
    assert!(!fits_f16(&tiny));
    // A non-finite source is never declared safe.
    let nan = mlxcel_core::from_slice_f32(&[1.0, f32::NAN], &[2]);
    assert!(!fits_f16(&nan));
    assert_eq!(
        f16_round_trip_error(&mlxcel_core::full_f32(&[4], 0.0, dtype::FLOAT32)),
        0.0
    );
}

#[test]
fn an_f32_tensor_is_kept_f32_because_f16_rounds_it() {
    let mut weights = WeightMap::new();
    weights.insert("layers.0.w".into(), det("a", &[64], 1.0, 0.0));
    assert_eq!(cast_floats(&mut weights, |_| false), ["layers.0.w"]);
    assert_eq!(
        mlxcel_core::array_dtype(&weights["layers.0.w"]),
        dtype::FLOAT32
    );
}

#[test]
fn cast_floats_widens_what_f16_cannot_hold_and_keeps_selected_keys() {
    let bf16 = |a: &mlxcel_core::MlxArray| mlxcel_core::astype(a, dtype::BFLOAT16);
    let mut weights = WeightMap::new();
    weights.insert("layers.0.w".into(), bf16(&det("a", &[32], 1.0, 0.0)));
    weights.insert(
        "layers.1.w".into(),
        bf16(&mlxcel_core::from_slice_f32(&[1.0, 7.0e4], &[2])),
    );
    weights.insert(
        "layers.2.w".into(),
        bf16(&mlxcel_core::full_f32(&[4], 1.0e-8, dtype::FLOAT32)),
    );
    weights.insert(
        "cap_embedder.0.weight".into(),
        bf16(&det("c", &[8], 1.0, 0.0)),
    );
    weights.insert(
        "layers.0.idx".into(),
        mlxcel_core::astype(&det("i", &[4], 4.0, 4.0), dtype::INT32),
    );
    let reference = mlxcel_core::astype(&weights["layers.0.w"], dtype::FLOAT32);

    let widened = cast_floats(&mut weights, reads_caption_stream);

    assert_eq!(widened, ["layers.1.w", "layers.2.w"]);
    let dt = |k: &str| mlxcel_core::array_dtype(&weights[k]);
    assert_eq!(dt("layers.0.w"), WEIGHT_DTYPE);
    assert_eq!(max_abs_diff(&weights["layers.0.w"], &reference), 0.0);
    assert_eq!(dt("layers.1.w"), dtype::FLOAT32);
    assert_eq!(dt("layers.2.w"), dtype::FLOAT32);
    assert_eq!(dt("cap_embedder.0.weight"), dtype::BFLOAT16);
    assert_eq!(dt("layers.0.idx"), dtype::INT32);
}

#[test]
fn the_vae_falls_back_to_f32_when_any_tensor_does_not_fit_f16() {
    // The fixture is f32 (which f16 cannot hold to 1e-5); as a bf16 checkpoint
    // with weights of ordinary magnitude it fits.
    let mut weights: WeightMap = vae_fixture()
        .into_iter()
        .map(|(k, v)| (k, mlxcel_core::astype(&v, dtype::BFLOAT16)))
        .collect();
    assert_eq!(vae_weight_dtype(&weights), (WEIGHT_DTYPE, Vec::new()));
    weights.insert(
        "decoder.conv_in.bias".into(),
        mlxcel_core::astype(
            &mlxcel_core::from_slice_f32(&[0.0, 1.0e5, 0.0, 0.0], &[4]),
            dtype::BFLOAT16,
        ),
    );
    let (dt, offending) = vae_weight_dtype(&weights);
    assert_eq!(dt, dtype::FLOAT32);
    assert_eq!(offending, ["decoder.conv_in.bias"]);
}

fn quantized_map(w: &mlxcel_core::MlxArray, group: i32, bits: i32, mode: &str) -> WeightMap {
    let q = mlxcel_core::quantize_weights_with_mode(w, group, bits, mode);
    let mut map = WeightMap::new();
    map.insert("a.weight".into(), mlxcel_core::quantized_weights_w(&q));
    map.insert("a.scales".into(), mlxcel_core::quantized_weights_scales(&q));
    if mode == "affine" {
        map.insert("a.biases".into(), mlxcel_core::quantized_weights_biases(&q));
    }
    map
}

fn quant_meta(mode: &str, bits: i32, group: i32) -> Map<String, Value> {
    meta(json!({
        "quantization_mode": mode, "quantization_level": bits,
        "quantization_group_size": group
    }))
}

fn rejection(weights: WeightMap, metadata: &Map<String, Value>) -> String {
    match dequantize_weights(weights, metadata) {
        Ok(_) => panic!("a malformed quantized tensor was accepted"),
        Err(e) => format!("{e:#}"),
    }
}

#[test]
fn quantization_metadata_rejects_what_mlx_dequantize_would_throw_on() {
    for (bits, group) in [(1, 64), (7, 64), (9, 64), (4, 16), (4, 48), (4, 256)] {
        let err = quant_spec(&quant_meta("affine", bits, group))
            .unwrap_err()
            .to_string();
        assert!(err.contains("Unsupported Z-Image quantization"), "{err}");
    }
    for bits in [2, 3, 4, 5, 6, 8] {
        for group in [32, 64, 128] {
            assert!(quant_spec(&quant_meta("affine", bits, group)).is_ok());
        }
    }
}

#[test]
fn dequantize_accepts_each_valid_mode() {
    let dense = det("w", &[8, 128], 0.5, 0.0);
    for (mode, bits, group) in [
        ("affine", 4, 64),
        ("mxfp4", 4, 32),
        ("mxfp8", 8, 32),
        ("nvfp4", 4, 16),
    ] {
        let out = dequantize_weights(
            quantized_map(&dense, group, bits, mode),
            &quant_meta(mode, bits, group),
        )
        .unwrap_or_else(|e| panic!("{mode}: {e:#}"));
        assert_eq!(mlxcel_core::array_shape(&out["a.weight"]), [8, 128]);
    }
}

#[test]
fn dequantize_rejects_malformed_quantized_tensors_before_mlx() {
    let dense = det("w", &[8, 128], 0.5, 0.0);
    let meta4 = quant_meta("affine", 4, 64);

    // Packed weight that is not uint32.
    let mut m = quantized_map(&dense, 64, 4, "affine");
    m.insert(
        "a.weight".into(),
        mlxcel_core::astype(&dense, dtype::FLOAT32),
    );
    assert!(rejection(m, &meta4).contains("a.weight must be uint32"));

    // Packed weight with a single dimension.
    let mut m = quantized_map(&dense, 64, 4, "affine");
    m.insert("a.weight".into(), mlxcel_core::zeros(&[16], dtype::UINT32));
    assert!(rejection(m, &meta4).contains("invalid packed shape"));

    // Scales whose group count disagrees with the group size.
    let m = quantized_map(&dense, 64, 4, "affine");
    assert!(rejection(m, &quant_meta("affine", 4, 32)).contains("scales shape"));

    // Scales with a wrong leading dimension.
    let mut m = quantized_map(&dense, 64, 4, "affine");
    m.insert(
        "a.scales".into(),
        mlxcel_core::zeros(&[4, 2], dtype::FLOAT32),
    );
    assert!(rejection(m, &meta4).contains("scales shape"));

    // Affine without biases, and biases of the wrong shape or dtype.
    let mut m = quantized_map(&dense, 64, 4, "affine");
    m.remove("a.biases");
    assert!(rejection(m, &meta4).contains("no biases"));
    let mut m = quantized_map(&dense, 64, 4, "affine");
    m.insert(
        "a.biases".into(),
        mlxcel_core::zeros(&[8, 1], dtype::FLOAT32),
    );
    assert!(rejection(m, &meta4).contains("biases do not match"));
    let mut m = quantized_map(&dense, 64, 4, "affine");
    let scales_dtype = mlxcel_core::array_dtype(&m["a.scales"]);
    let other = if scales_dtype == dtype::FLOAT16 {
        dtype::FLOAT32
    } else {
        dtype::FLOAT16
    };
    let bias = mlxcel_core::astype(&m["a.biases"], other);
    m.insert("a.biases".into(), bias);
    assert!(rejection(m, &meta4).contains("biases do not match"));

    // Non-float affine scales.
    let mut m = quantized_map(&dense, 64, 4, "affine");
    m.insert("a.scales".into(), mlxcel_core::zeros(&[8, 2], dtype::INT32));
    assert!(rejection(m, &meta4).contains("floating dtype"));

    // Packed width that is not whole at the bit width (3-bit needs a multiple of 3).
    let mut m = quantized_map(&dense, 64, 4, "affine");
    m.insert(
        "a.weight".into(),
        mlxcel_core::zeros(&[8, 16], dtype::UINT32),
    );
    assert!(rejection(m, &quant_meta("affine", 3, 64)).contains("not whole"));

    // Input width that is not a multiple of the group size.
    let mut m = quantized_map(&dense, 64, 4, "affine");
    m.insert(
        "a.weight".into(),
        mlxcel_core::zeros(&[8, 12], dtype::UINT32),
    );
    assert!(rejection(m, &meta4).contains("multiple of group size"));

    // mx modes: scales are uint8 and there are no biases.
    let mut m = quantized_map(&dense, 32, 4, "mxfp4");
    m.insert(
        "a.scales".into(),
        mlxcel_core::zeros(&[8, 4], dtype::FLOAT32),
    );
    assert!(rejection(m, &quant_meta("mxfp4", 4, 32)).contains("must be uint8"));
    let mut m = quantized_map(&dense, 32, 4, "mxfp4");
    m.insert(
        "a.biases".into(),
        mlxcel_core::zeros(&[8, 4], dtype::FLOAT32),
    );
    assert!(rejection(m, &quant_meta("mxfp4", 4, 32)).contains("has biases"));
}
