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

//! Tests for the compressed-tensors `float-quantized` FP8 route (issue #2411).

use super::{
    CT_FP8_INPUT_SCALE_SUFFIX, CT_FP8_SCALE_SUFFIX, Fp8CtQuantization, Fp8CtScaleLayout,
    compressed_tensors_fp8_quantization, has_compressed_tensors_fp8_weights,
    requantize_compressed_tensors_fp8_weights,
};
use crate::models::fp8_block::{MXFP8_BITS, MXFP8_GROUP_SIZE, MXFP8_MODE};
use crate::models::sanitize::{f8_e4m3_to_f32, f32_to_f8_e4m3};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, dtype};
use serde_json::{Value, json};

/// Deterministic xorshift so a failing case is reproducible from the test name
/// alone. `rand` is not a dependency of this crate's model layer.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 32) as u32
    }

    /// Uniform in `[-4, 4)`.
    fn next_value(&mut self) -> f32 {
        let unit = f64::from(self.next_u32()) / f64::from(u32::MAX);
        (unit * 8.0 - 4.0) as f32
    }
}

fn read_f32(array: &MlxArray) -> Vec<f32> {
    let contiguous = mlxcel_core::astype(array, dtype::FLOAT32);
    mlxcel_core::eval(&contiguous);
    mlxcel_core::array_to_raw_bytes(&contiguous)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn raw_bytes(array: &mlxcel_core::UniquePtr<MlxArray>) -> Vec<u8> {
    let inner = array.as_ref().expect("array handle");
    mlxcel_core::eval(inner);
    mlxcel_core::array_to_raw_bytes(inner)
}

/// Half an E4M3 step at `byte`, expressed in the checkpoint's units.
///
/// The source value was rounded to the nearest E4M3 code of `value / scale`,
/// so its round-trip error is bounded by half the wider of the two gaps around
/// that code, multiplied back by `scale`. Measured from the neighbouring codes
/// rather than from the exponent, so the subnormal range and the binade
/// boundaries are bounded by the same expression.
fn half_e4m3_step_at(byte: u8, scale: f32) -> f32 {
    // E4M3 is symmetric, so the spacing depends only on the magnitude code;
    // the sign bit is stripped before reading the neighbours.
    let code = byte & 0x7F;
    let current = f8_e4m3_to_f32(code);
    let lower = if code == 0 {
        0.0
    } else {
        f8_e4m3_to_f32(code - 1)
    };
    let upper = if code >= 0x7E {
        current + (current - lower)
    } else {
        f8_e4m3_to_f32(code + 1)
    };
    // The relative term covers the f32 division that produced `value / scale`.
    0.5 * (current - lower).max(upper - current) * scale * (1.0 + 1e-6)
}

fn expect_requantize_error(weights: WeightMap, layout: Fp8CtScaleLayout) -> String {
    match requantize_compressed_tensors_fp8_weights(weights, layout) {
        Ok(_) => panic!("expected a failure"),
        Err(error) => error,
    }
}

struct CtFixture {
    weights: WeightMap,
    /// The f32 values the checkpoint was quantized from.
    source: Vec<f32>,
    /// `e4m3_decode(byte) * scale`, the exact reconstruction target.
    expected: Vec<f32>,
    /// Per-element round-to-nearest bound of the fixture's own E4M3
    /// quantization, in the checkpoint's units.
    source_bounds: Vec<f32>,
    rows: usize,
    cols: usize,
}

/// Quantize a random f32 matrix to E4M3 and build the checkpoint's
/// `<prefix>.weight` / `<prefix>.weight_scale` / `<prefix>.input_scale` triple.
///
/// `layout` decides whether the scale is per row or per block; `column_vector`
/// covers the `[out, 1]` spelling llm-compressor writes for channel strategy.
fn ct_fixture(
    prefix: &str,
    rows: usize,
    cols: usize,
    layout: Fp8CtScaleLayout,
    column_vector: bool,
    seed: u64,
) -> CtFixture {
    let mut rng = Rng::new(seed);
    let source: Vec<f32> = (0..rows * cols).map(|_| rng.next_value()).collect();

    let e4m3_max = f8_e4m3_to_f32(0x7E);
    let mut bytes = vec![0u8; rows * cols];
    let mut expected = vec![0f32; rows * cols];
    let mut source_bounds = vec![0f32; rows * cols];

    match layout {
        Fp8CtScaleLayout::Channel => {
            let mut scales = Vec::with_capacity(rows);
            for r in 0..rows {
                let row = &source[r * cols..(r + 1) * cols];
                let amax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
                let scale = if amax == 0.0 { 1.0 } else { amax / e4m3_max };
                for c in 0..cols {
                    let byte = f32_to_f8_e4m3(safe_divide(row[c], scale));
                    bytes[r * cols + c] = byte;
                    expected[r * cols + c] = f8_e4m3_to_f32(byte) * scale;
                    source_bounds[r * cols + c] = half_e4m3_step_at(byte, scale);
                }
                scales.push(scale);
            }
            let mut weights = WeightMap::new();
            weights.insert(
                format!("{prefix}.weight"),
                mlxcel_core::from_bytes(&bytes, &[rows as i32, cols as i32], dtype::UINT8),
            );
            let shape = if column_vector {
                vec![rows as i32, 1]
            } else {
                vec![rows as i32]
            };
            weights.insert(
                format!("{prefix}{CT_FP8_SCALE_SUFFIX}"),
                mlxcel_core::from_slice_f32(&scales, &shape),
            );
            weights.insert(
                format!("{prefix}{CT_FP8_INPUT_SCALE_SUFFIX}"),
                mlxcel_core::from_slice_f32(&[0.125], &[1]),
            );
            CtFixture {
                weights,
                source,
                expected,
                source_bounds,
                rows,
                cols,
            }
        }
        Fp8CtScaleLayout::Block { rows: br, cols: bc } => {
            let row_blocks = rows.div_ceil(br);
            let col_blocks = cols.div_ceil(bc);
            let mut scales = vec![0f32; row_blocks * col_blocks];
            for rb in 0..row_blocks {
                for cb in 0..col_blocks {
                    let mut amax = 0f32;
                    for r in (rb * br)..((rb + 1) * br).min(rows) {
                        for c in (cb * bc)..((cb + 1) * bc).min(cols) {
                            amax = amax.max(source[r * cols + c].abs());
                        }
                    }
                    scales[rb * col_blocks + cb] = if amax == 0.0 { 1.0 } else { amax / e4m3_max };
                }
            }
            for r in 0..rows {
                for c in 0..cols {
                    let scale = scales[(r / br) * col_blocks + c / bc];
                    let byte = f32_to_f8_e4m3(safe_divide(source[r * cols + c], scale));
                    bytes[r * cols + c] = byte;
                    expected[r * cols + c] = f8_e4m3_to_f32(byte) * scale;
                    source_bounds[r * cols + c] = half_e4m3_step_at(byte, scale);
                }
            }
            let mut weights = WeightMap::new();
            weights.insert(
                format!("{prefix}.weight"),
                mlxcel_core::from_bytes(&bytes, &[rows as i32, cols as i32], dtype::UINT8),
            );
            weights.insert(
                format!("{prefix}{CT_FP8_SCALE_SUFFIX}"),
                mlxcel_core::from_slice_f32(&scales, &[row_blocks as i32, col_blocks as i32]),
            );
            CtFixture {
                weights,
                source,
                expected,
                source_bounds,
                rows,
                cols,
            }
        }
    }
}

/// A divide that leaves the value alone at `scale == 0.0` and never produces a
/// non-finite result, so a degenerate row (all zeros) still has bytes to store.
fn safe_divide(value: f32, scale: f32) -> f32 {
    if scale == 0.0 { value } else { value / scale }
}

/// Assert that the mxfp8 planes decode within the documented bound: the source
/// quantization error (half an E4M3 step) plus the mxfp8 float-quantization
/// bound already asserted by
/// `fp8_block_requantize_round_trip_stays_within_half_an_e4m3_step`.
fn assert_within_bounds(
    source: &[f32],
    expected: &[f32],
    source_bounds: &[f32],
    converted: &WeightMap,
    prefix: &str,
) {
    let packed = converted
        .get(&format!("{prefix}.weight"))
        .expect("packed weight");
    let scales = converted
        .get(&format!("{prefix}.scales"))
        .expect("mxfp8 scales");

    let dequantized = unsafe {
        mlxcel_core::dequantize(
            packed.as_ref().expect("packed"),
            scales.as_ref().expect("scales"),
            std::ptr::null(),
            MXFP8_GROUP_SIZE,
            MXFP8_BITS,
            MXFP8_MODE,
        )
    };
    let recovered = read_f32(&dequantized);
    assert_eq!(recovered.len(), expected.len());

    let group = MXFP8_GROUP_SIZE as usize;
    for (g, (expected, actual)) in expected
        .chunks_exact(group)
        .zip(recovered.chunks_exact(group))
        .enumerate()
    {
        let group_max = expected.iter().fold(0f32, |m, v| m.max(v.abs()));
        let mxfp8_bound = group_max / 16.0;
        for (k, (expected, actual)) in expected.iter().zip(actual).enumerate() {
            let index = g * group + k;
            let source = source[index];
            let source_bound = source_bounds[index] + f32::EPSILON * source.abs();
            let source_error = (expected - source).abs();
            assert!(
                source_error <= source_bound,
                "element {index}: the fixture's own E4M3 quantization moved {source} by \
                 {source_error}, beyond the half-step bound {source_bound}"
            );

            let bound = source_bound + mxfp8_bound + f32::EPSILON * expected.abs();
            let error = (actual - source).abs();
            assert!(
                error <= bound,
                "element {index}: mxfp8 error {error} exceeded {bound} (source {source}, exact \
                 reconstruction {expected}, group max {group_max})"
            );
        }
    }
}

/// The full `RedHatAI/Qwen3.5-4B-FP8-dynamic` config, trimmed to the fields
/// this route reads. Everything else in the real file (`ignore` and the
/// observer settings) does not change the layout decision.
fn dynamic_channel_config() -> Value {
    json!({
        "model_type": "qwen3_5",
        "quantization_config": {
            "config_groups": {
                "group_0": {
                    "format": "float-quantized",
                    "input_activations": {
                        "block_structure": null,
                        "dynamic": true,
                        "num_bits": 8,
                        "strategy": "token",
                        "type": "float",
                    },
                    "targets": ["Linear"],
                    "weights": {
                        "block_structure": null,
                        "dynamic": false,
                        "num_bits": 8,
                        "strategy": "channel",
                        "type": "float",
                    },
                }
            },
            "format": "float-quantized",
            "ignore": [
                "model.visual.blocks.0.attn.qkv",
                "model.language_model.layers.0.linear_attn.in_proj_qkv",
                "lm_head",
                "re:^mtp.*",
            ],
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
        }
    })
}

fn detection_error(config: &Value) -> String {
    compressed_tensors_fp8_quantization(config)
        .expect_err("expected a refused compressed-tensors FP8 layout")
}

#[test]
fn compressed_tensors_config_detection() {
    let detected = compressed_tensors_fp8_quantization(&dynamic_channel_config())
        .expect("detection")
        .expect("a float-quantized channel export is supported");
    assert_eq!(
        detected,
        Fp8CtQuantization {
            layout: Fp8CtScaleLayout::Channel,
            group_size: MXFP8_GROUP_SIZE,
            bits: MXFP8_BITS,
            mode: MXFP8_MODE,
        }
    );
    assert_eq!(
        detected.effective_config(),
        json!({"group_size": 32, "bits": 8, "mode": "mxfp8"})
    );

    // A block export carries the tile in `block_structure`; both the strategy
    // name and the null-block channel form must be distinguished.
    let mut block = dynamic_channel_config();
    block["quantization_config"]["config_groups"]["group_0"]["weights"]["strategy"] =
        json!("block");
    block["quantization_config"]["config_groups"]["group_0"]["weights"]["block_structure"] =
        json!([128, 128]);
    let detected = compressed_tensors_fp8_quantization(&block)
        .expect("detection")
        .expect("a block export is supported");
    assert_eq!(
        detected.layout,
        Fp8CtScaleLayout::Block {
            rows: 128,
            cols: 128
        }
    );

    // The VLM wrappers place the block under text_config on some releases.
    let mut nested = dynamic_channel_config();
    let quant = nested["quantization_config"].take();
    nested
        .as_object_mut()
        .expect("object")
        .remove("quantization_config");
    nested["text_config"] = json!({"model_type": "qwen3_5_text"});
    nested["text_config"]["quantization_config"] = quant;
    assert!(
        compressed_tensors_fp8_quantization(&nested)
            .expect("detection")
            .is_some()
    );

    // Other quantization methods and plain checkpoints are not this route.
    assert_eq!(
        compressed_tensors_fp8_quantization(&json!({"model_type": "qwen3_5"})).expect("detection"),
        None
    );
    assert_eq!(
        compressed_tensors_fp8_quantization(&json!({
            "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}
        }))
        .expect("detection"),
        None
    );

    // A ModelOpt NVFP4 export is `compressed-tensors` on some releases but is
    // a packed 4-bit layout, which the NVFP4 repack route owns.
    let modelopt_nvfp4 = json!({
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "nvfp4-pack-quantized",
            "config_groups": {
                "group_0": {
                    "format": "nvfp4-pack-quantized",
                    "weights": {
                        "type": "float",
                        "num_bits": 4,
                        "strategy": "group",
                        "group_size": 16,
                    }
                }
            }
        }
    });
    assert_eq!(
        compressed_tensors_fp8_quantization(&modelopt_nvfp4).expect("detection"),
        None
    );
    // The same export spelled the ModelOpt way is also not this route.
    assert_eq!(
        compressed_tensors_fp8_quantization(&json!({
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "NVFP4",
                "exclude_modules": ["lm_head"],
            }
        }))
        .expect("detection"),
        None
    );

    // A mixed-precision export keeps the real formats per group and is out of
    // scope: the FP8 group alone is not the layout of the whole checkpoint.
    let mixed = json!({
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "mixed-precision",
            "config_groups": {
                "group_0": {
                    "format": "float-quantized",
                    "weights": {"type": "float", "num_bits": 8, "strategy": "channel"},
                }
            }
        }
    });
    assert_eq!(
        compressed_tensors_fp8_quantization(&mixed).expect("detection"),
        None
    );

    // Per-tensor FP8 has a different sidecar shape and is refused by name.
    let mut per_tensor = dynamic_channel_config();
    per_tensor["quantization_config"]["config_groups"]["group_0"]["weights"]["strategy"] =
        json!("tensor");
    let error = detection_error(&per_tensor);
    assert!(
        error.contains("weights.strategy") && error.contains("tensor"),
        "unhelpful error: {error}"
    );

    // Group quantization likewise pairs a scale with a group, not a channel.
    let mut grouped = dynamic_channel_config();
    grouped["quantization_config"]["config_groups"]["group_0"]["weights"]["strategy"] =
        json!("group");
    let error = detection_error(&grouped);
    assert!(error.contains("group"), "unhelpful error: {error}");

    // Another bit width would decode with the wrong table.
    let mut four_bit = dynamic_channel_config();
    four_bit["quantization_config"]["config_groups"]["group_0"]["weights"]["num_bits"] = json!(4);
    let error = detection_error(&four_bit);
    assert!(
        error.contains("weights.num_bits") && error.contains('4'),
        "unhelpful error: {error}"
    );

    // Dynamic weights compute their scale at run time and store none.
    let mut dynamic = dynamic_channel_config();
    dynamic["quantization_config"]["config_groups"]["group_0"]["weights"]["dynamic"] = json!(true);
    let error = detection_error(&dynamic);
    assert!(
        error.contains("weights.dynamic"),
        "unhelpful error: {error}"
    );

    // A non-float element type is not E4M3.
    let mut integer = dynamic_channel_config();
    integer["quantization_config"]["config_groups"]["group_0"]["weights"]["type"] = json!("int");
    let error = detection_error(&integer);
    assert!(error.contains("weights.type"), "unhelpful error: {error}");

    // A block structure with a zero extent has no tile to pair a scale with.
    let mut zero_block = dynamic_channel_config();
    zero_block["quantization_config"]["config_groups"]["group_0"]["weights"]["block_structure"] =
        json!([0, 128]);
    let error = detection_error(&zero_block);
    assert!(
        error.contains("block_structure"),
        "unhelpful error: {error}"
    );
}

/// A channel-scaled checkpoint must convert exactly to what quantizing the
/// host-expanded product produces, and stay inside the documented error bound.
#[test]
fn compressed_tensors_channel_requantize_matches_direct_path() {
    let prefix = "model.layers.0.mlp.down_proj";
    // 130 rows is not a multiple of any block extent; 96 columns is a multiple
    // of the mxfp8 group size, as every real projection is.
    let CtFixture {
        weights,
        source,
        expected,
        source_bounds,
        rows,
        cols,
    } = ct_fixture(
        prefix,
        130,
        96,
        Fp8CtScaleLayout::Channel,
        false,
        0x51D3_9E11,
    );
    assert!(has_compressed_tensors_fp8_weights(&weights));
    let rows = rows as i32;
    let cols = cols as i32;

    let converted = requantize_compressed_tensors_fp8_weights(weights, Fp8CtScaleLayout::Channel)
        .expect("compressed-tensors requantization");

    assert!(
        !converted.contains_key(&format!("{prefix}{CT_FP8_SCALE_SUFFIX}")),
        "the per-channel scale must be consumed, not left for the model constructor"
    );
    assert!(
        !converted.contains_key(&format!("{prefix}{CT_FP8_INPUT_SCALE_SUFFIX}")),
        "the dynamic activation scale is unused and must be removed"
    );
    assert!(
        !converted.contains_key(&format!("{prefix}.biases")),
        "mxfp8 is a block-float mode and must not gain an affine zero-point plane"
    );

    let packed = converted
        .get(&format!("{prefix}.weight"))
        .expect("packed weight");
    let scales = converted
        .get(&format!("{prefix}.scales"))
        .expect("mxfp8 scales");
    assert_eq!(
        mlxcel_core::array_shape(packed.as_ref().expect("packed")),
        vec![rows, cols / 4]
    );
    assert_eq!(
        mlxcel_core::array_dtype(packed.as_ref().expect("packed")),
        dtype::UINT32
    );
    assert_eq!(
        mlxcel_core::array_shape(scales.as_ref().expect("scales")),
        vec![rows, cols / MXFP8_GROUP_SIZE]
    );

    // Reference: quantize the host-expanded product directly.
    let reference_dense = mlxcel_core::from_slice_f32(&expected, &[rows, cols]);
    let reference = mlxcel_core::quantize_weights_with_mode(
        &reference_dense,
        MXFP8_GROUP_SIZE,
        MXFP8_BITS,
        MXFP8_MODE,
    );
    assert_eq!(
        raw_bytes(packed),
        raw_bytes(&mlxcel_core::quantized_weights_w(&reference)),
        "packed mxfp8 plane diverged from quantize(decode(bytes) * channel_scale)"
    );
    assert_eq!(
        raw_bytes(scales),
        raw_bytes(&mlxcel_core::quantized_weights_scales(&reference)),
        "mxfp8 scale plane diverged from quantize(decode(bytes) * channel_scale)"
    );

    assert_within_bounds(&source, &expected, &source_bounds, &converted, prefix);
}

/// The `[out, 1]` spelling of the channel scale must take the same path as
/// `[out]`: llm-compressor writes the column vector, and a shape mismatch here
/// would be a load failure on the real checkpoint.
#[test]
fn compressed_tensors_channel_scale_accepts_a_column_vector() {
    let prefix = "model.layers.1.self_attn.o_proj";
    let CtFixture {
        weights,
        source,
        expected,
        source_bounds,
        ..
    } = ct_fixture(prefix, 48, 64, Fp8CtScaleLayout::Channel, true, 0x0BAD_F00D);
    let converted = requantize_compressed_tensors_fp8_weights(weights, Fp8CtScaleLayout::Channel)
        .expect("compressed-tensors requantization");
    assert!(converted.contains_key(&format!("{prefix}.scales")));
    assert_within_bounds(&source, &expected, &source_bounds, &converted, prefix);
}

/// A `[64, 64]` block scale on a shape that is not a multiple of 64 exercises
/// the padding and the slice-back on both axes.
#[test]
fn compressed_tensors_block_requantize_pads_partial_blocks() {
    let prefix = "model.layers.2.mlp.gate_proj";
    let layout = Fp8CtScaleLayout::Block { rows: 64, cols: 64 };
    let CtFixture {
        weights,
        source,
        expected,
        source_bounds,
        ..
    } = ct_fixture(prefix, 100, 96, layout, false, 0xFEED_BEEF);
    // 100 rows -> 2 row blocks (128 padded), 96 cols -> 2 column blocks.
    assert_eq!(
        mlxcel_core::array_shape(
            weights[&format!("{prefix}{CT_FP8_SCALE_SUFFIX}")]
                .as_ref()
                .expect("scale")
        ),
        vec![2, 2]
    );

    let converted = requantize_compressed_tensors_fp8_weights(weights, layout)
        .expect("compressed-tensors requantization");

    let packed = converted
        .get(&format!("{prefix}.weight"))
        .expect("packed weight");
    let scales = converted
        .get(&format!("{prefix}.scales"))
        .expect("mxfp8 scales");

    let reference_dense = mlxcel_core::from_slice_f32(&expected, &[100, 96]);
    let reference = mlxcel_core::quantize_weights_with_mode(
        &reference_dense,
        MXFP8_GROUP_SIZE,
        MXFP8_BITS,
        MXFP8_MODE,
    );
    assert_eq!(
        raw_bytes(packed),
        raw_bytes(&mlxcel_core::quantized_weights_w(&reference)),
        "padded block reconstruction diverged from the host reference"
    );
    assert_eq!(
        raw_bytes(scales),
        raw_bytes(&mlxcel_core::quantized_weights_scales(&reference))
    );

    assert_within_bounds(&source, &expected, &source_bounds, &converted, prefix);
}

/// A ModelOpt NVFP4 triplet carries `.weight_scale` too, and must be left
/// entirely to `sanitize.rs`'s repack route.
#[test]
fn compressed_tensors_route_leaves_nvfp4_triplets_untouched() {
    let prefix = "model.layers.0.mlp.up_proj";
    let mut weights = WeightMap::new();
    weights.insert(
        format!("{prefix}.weight"),
        mlxcel_core::from_bytes(&[0x38; 32], &[2, 16], dtype::UINT8),
    );
    weights.insert(
        format!("{prefix}.weight_scale"),
        mlxcel_core::from_slice_f32(&[0.5; 2], &[2, 1]),
    );
    weights.insert(
        format!("{prefix}.weight_scale_2"),
        mlxcel_core::from_slice_f32(&[1.0], &[1]),
    );
    weights.insert(
        format!("{prefix}.input_scale"),
        mlxcel_core::from_slice_f32(&[0.25], &[1]),
    );

    assert!(!has_compressed_tensors_fp8_weights(&weights));

    let converted = requantize_compressed_tensors_fp8_weights(weights, Fp8CtScaleLayout::Channel)
        .expect("an NVFP4 triplet is not this route's input");

    assert!(converted.contains_key(&format!("{prefix}.weight_scale")));
    assert!(converted.contains_key(&format!("{prefix}.weight_scale_2")));
    assert!(converted.contains_key(&format!("{prefix}.input_scale")));
    assert!(!converted.contains_key(&format!("{prefix}.scales")));
    assert_eq!(
        mlxcel_core::array_dtype(
            converted[&format!("{prefix}.weight")]
                .as_ref()
                .expect("weight")
        ),
        dtype::UINT8
    );
}

/// Ignored projections arrive as bf16 with no scale; a bf16 tensor that
/// happens to carry a `.weight_scale` is not an E4M3 weight and must not be
/// requantized as one.
#[test]
fn compressed_tensors_route_skips_predecoded_weights() {
    let prefix = "model.layers.0.linear_attn.out_proj";
    let mut weights = WeightMap::new();
    weights.insert(
        format!("{prefix}.weight"),
        mlxcel_core::ones(&[64, 64], dtype::BFLOAT16),
    );
    weights.insert(
        format!("{prefix}{CT_FP8_SCALE_SUFFIX}"),
        mlxcel_core::from_slice_f32(&[1.0; 64], &[64]),
    );

    let converted = requantize_compressed_tensors_fp8_weights(weights, Fp8CtScaleLayout::Channel)
        .expect("requantization");

    assert_eq!(
        mlxcel_core::array_dtype(
            converted[&format!("{prefix}.weight")]
                .as_ref()
                .expect("weight")
        ),
        dtype::BFLOAT16
    );
    assert!(!converted.contains_key(&format!("{prefix}.scales")));
}

/// A width that is not a multiple of the mxfp8 group size has no
/// representation; refuse it by name rather than letting MLX abort inside the
/// first quantized matmul.
#[test]
fn compressed_tensors_route_rejects_odd_width() {
    let prefix = "model.layers.0.mlp.down_proj";
    let fixture = ct_fixture(
        prefix,
        64,
        100,
        Fp8CtScaleLayout::Channel,
        false,
        0x1234_5678,
    );
    let error = expect_requantize_error(fixture.weights, Fp8CtScaleLayout::Channel);
    assert!(
        error.contains("100") && error.contains("group size"),
        "unhelpful error: {error}"
    );
}

/// A channel scale whose length does not match the weight's output channels is
/// a corrupt pairing; accepting it would mis-scale silently.
#[test]
fn compressed_tensors_route_rejects_a_mismatched_channel_scale() {
    let prefix = "model.layers.0.mlp.gate_proj";
    let fixture = ct_fixture(prefix, 64, 64, Fp8CtScaleLayout::Channel, false, 0x9999);
    let mut weights = fixture.weights;
    weights.insert(
        format!("{prefix}{CT_FP8_SCALE_SUFFIX}"),
        mlxcel_core::from_slice_f32(&[1.0; 8], &[8]),
    );
    let error = expect_requantize_error(weights, Fp8CtScaleLayout::Channel);
    assert!(
        error.contains("channel scale") && error.contains("64"),
        "unhelpful error: {error}"
    );
}

/// A checkpoint with no compressed-tensors scales must come back untouched, so
/// the pre-pass is safe to run on any Qwen3.5 load.
#[test]
fn compressed_tensors_route_is_identity_without_scales() {
    let mut weights = WeightMap::new();
    weights.insert(
        "model.embed_tokens.weight".to_string(),
        mlxcel_core::ones(&[8, 8], dtype::BFLOAT16),
    );
    let converted = requantize_compressed_tensors_fp8_weights(weights, Fp8CtScaleLayout::Channel)
        .expect("requantization");
    assert_eq!(converted.len(), 1);
    assert!(converted.contains_key("model.embed_tokens.weight"));
}
