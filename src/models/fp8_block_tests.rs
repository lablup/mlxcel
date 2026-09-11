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

use super::{
    Fp8BlockQuantization, MXFP8_BITS, MXFP8_GROUP_SIZE, MXFP8_MODE, SUPPORTED_FP8_BLOCK,
    has_block_fp8_weights, merge_fp8_block_quantization, qwen_fp8_block_quantization,
    requantize_block_fp8_weights,
};
use crate::models::sanitize::{f8_e4m3_to_f32, f32_to_f8_e4m3};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use serde_json::json;

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

    fn below(&mut self, bound: u32) -> u32 {
        self.next_u32() % bound
    }
}

/// A bf16 value and its exact f32 widening.
///
/// bf16 is the top 16 bits of an f32, so the widening is exact and the host
/// reference below multiplies by the very same number the device path reads.
fn bf16_value(rng: &mut Rng) -> (u16, f32) {
    // f32 exponent field 120..=130 spans 2^-7 .. 2^3, which brackets the
    // magnitudes real FP8 block scales take (block_amax / 448).
    let exponent = 120 + rng.below(11) as u16;
    let mantissa = rng.below(128) as u16;
    let bits = (exponent << 7) | mantissa;
    (bits, f32::from_bits(u32::from(bits) << 16))
}

/// An E4M3 byte that is not one of the two NaN encodings.
fn e4m3_byte(rng: &mut Rng) -> u8 {
    loop {
        let byte = rng.below(256) as u8;
        if byte != 0x7F && byte != 0xFF {
            return byte;
        }
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

fn raw_bytes(array: &UniquePtr<MlxArray>) -> Vec<u8> {
    let inner = array.as_ref().expect("array handle");
    mlxcel_core::eval(inner);
    mlxcel_core::array_to_raw_bytes(inner)
}

/// `WeightMap` holds `UniquePtr<MlxArray>`, which is not `Debug`, so
/// `Result::expect_err` cannot be used on a requantization result.
fn expect_requantize_error(weights: WeightMap, reason: &str) -> String {
    match requantize_block_fp8_weights(weights, SUPPORTED_FP8_BLOCK) {
        Ok(_) => panic!("expected a failure: {reason}"),
        Err(error) => error,
    }
}

struct BlockFp8Fixture {
    weights: WeightMap,
    /// `decode(bytes) * expanded_scales`, computed on the host.
    expanded: Vec<f32>,
    rows: usize,
    cols: usize,
}

/// Build a synthetic `<prefix>.weight` / `<prefix>.weight_scale_inv` pair plus
/// the host-side reconstruction of what it means.
///
/// The host reconstruction expands each block scale over its whole 128x128
/// tile and multiplies element by element. That is deliberately a different
/// expression of the same arithmetic from the device path under test, which
/// pads to whole blocks, reshapes to `[rb, 128, cb, 128]`, and broadcasts. The
/// two share only the E4M3 decode table.
fn block_fp8_fixture(prefix: &str, rows: usize, cols: usize, seed: u64) -> BlockFp8Fixture {
    let block = SUPPORTED_FP8_BLOCK;
    let row_blocks = rows.div_ceil(block);
    let col_blocks = cols.div_ceil(block);

    let mut rng = Rng::new(seed);
    let bytes: Vec<u8> = (0..rows * cols).map(|_| e4m3_byte(&mut rng)).collect();

    let mut scale_bits = Vec::with_capacity(row_blocks * col_blocks);
    let mut scale_values = Vec::with_capacity(row_blocks * col_blocks);
    for _ in 0..row_blocks * col_blocks {
        let (bits, value) = bf16_value(&mut rng);
        scale_bits.extend_from_slice(&bits.to_le_bytes());
        scale_values.push(value);
    }

    let mut expanded = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let scale = scale_values[(r / block) * col_blocks + (c / block)];
            expanded.push(f8_e4m3_to_f32(bytes[r * cols + c]) * scale);
        }
    }

    let mut weights = WeightMap::new();
    weights.insert(
        format!("{prefix}.weight"),
        mlxcel_core::from_bytes(
            &bytes,
            &[rows as i32, cols as i32],
            mlxcel_core::dtype::UINT8,
        ),
    );
    weights.insert(
        format!("{prefix}.weight_scale_inv"),
        mlxcel_core::from_bytes_f16(&scale_bits, &[row_blocks as i32, col_blocks as i32], true),
    );

    BlockFp8Fixture {
        weights,
        expanded,
        rows,
        cols,
    }
}

/// The reconstruct-then-requantize path must land on exactly the tensors MLX
/// produces from the naive expanded-scale product, on both planes.
///
/// This is the whole correctness claim for the reconstruction formula: block
/// pairing, padding, the broadcast axis order, and the trailing slice are all
/// wrong-answer-shaped rather than crash-shaped, and only a bitwise comparison
/// against an independently written reference catches them.
#[test]
fn fp8_block_requantize_matches_direct_path() {
    // 130x160 is deliberately not block-aligned on either axis: rows need
    // padding to 256 and columns to 256, so the trailing partial blocks and
    // the slice back to the original extent are both exercised. 160 is still a
    // multiple of the mxfp8 group size 32, as every real projection is.
    let fixture = block_fp8_fixture("model.layers.0.mlp.down_proj", 130, 160, 0x51D3_9E11);
    let rows = fixture.rows as i32;
    let cols = fixture.cols as i32;

    assert!(has_block_fp8_weights(&fixture.weights));

    let converted = requantize_block_fp8_weights(fixture.weights, SUPPORTED_FP8_BLOCK)
        .expect("block FP8 requantization");

    assert!(
        !converted.contains_key("model.layers.0.mlp.down_proj.weight_scale_inv"),
        "the block-scale sidecar must be consumed, not left for the model constructor"
    );
    let packed = converted
        .get("model.layers.0.mlp.down_proj.weight")
        .expect("packed weight");
    let scales = converted
        .get("model.layers.0.mlp.down_proj.scales")
        .expect("mxfp8 scales");

    // MLX packs mxfp8 as uint32 words of four 8-bit values, with one uint8
    // E8M0 exponent per group of 32.
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
    assert_eq!(
        mlxcel_core::array_dtype(scales.as_ref().expect("scales")),
        dtype::UINT8
    );
    assert!(
        !converted.contains_key("model.layers.0.mlp.down_proj.biases"),
        "mxfp8 is a block-float mode and must not gain an affine zero-point plane"
    );

    // Reference: quantize the host-expanded product directly.
    let reference_dense = mlxcel_core::from_slice_f32(&fixture.expanded, &[rows, cols]);
    let reference = mlxcel_core::quantize_weights_with_mode(
        &reference_dense,
        MXFP8_GROUP_SIZE,
        MXFP8_BITS,
        MXFP8_MODE,
    );
    let reference_w = mlxcel_core::quantized_weights_w(&reference);
    let reference_scales = mlxcel_core::quantized_weights_scales(&reference);

    assert_eq!(
        raw_bytes(packed),
        raw_bytes(&reference_w),
        "packed mxfp8 plane diverged from quantize(decode(bytes) * expanded_scales)"
    );
    assert_eq!(
        raw_bytes(scales),
        raw_bytes(&reference_scales),
        "mxfp8 scale plane diverged from quantize(decode(bytes) * expanded_scales)"
    );
}

/// Dequantizing the requantized planes must land within half an E4M3 step of
/// the reconstruction, which holds only while every mxfp8 block scale is
/// rounded up.
///
/// The block scale is `amax / 448` encoded as E8M0, a bare power of two.
/// Rounded up, the block maximum scales into `(224, 448]` and every element
/// stays inside the E4M3 range, where it rounds to nearest. E4M3 carries four
/// significant bits, so an element in the top binade (`[256, 448]` after
/// scaling) moves by at most 16 of those units, which is at most 2^-4 of a
/// block maximum that is itself at least 256; an element in a lower binade, or
/// in a block whose maximum scaled below 256, moves by less.
///
/// Rounded to nearest in log2 space instead, the scale lands below
/// `amax / 448` for about half the blocks, their maxima scale past 448 and
/// saturate, and the loss on a block maximum reaches `1 - 2^-1/2`, about 29%.
/// No per-element bound between those two holds, one full E4M3 step
/// (`group_max / 8`) included. Metal and CPU rounded that way until
/// ml-explore/mlx#4353 while CUDA always rounded up. On an MLX pin that
/// predates it, 301 of this fixture's 650 blocks saturate, block 0 among them:
/// its maximum 4.8046875 takes the scale 2^-7 and would scale to 615, so
/// element 4 (4.00390625, scaled to 512.5) comes back as 3.5.
///
/// Same seed and shape as [`fp8_block_requantize_matches_direct_path`]: the
/// padded trailing blocks are part of what is being bounded.
#[test]
fn fp8_block_requantize_round_trip_stays_within_half_an_e4m3_step() {
    let fixture = block_fp8_fixture("model.layers.0.mlp.down_proj", 130, 160, 0x51D3_9E11);
    let converted = requantize_block_fp8_weights(fixture.weights, SUPPORTED_FP8_BLOCK)
        .expect("block FP8 requantization");
    let packed = converted
        .get("model.layers.0.mlp.down_proj.weight")
        .expect("packed weight");
    let scales = converted
        .get("model.layers.0.mlp.down_proj.scales")
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
    assert_eq!(recovered.len(), fixture.expanded.len());

    let group = MXFP8_GROUP_SIZE as usize;
    let exponents = raw_bytes(scales);
    assert_eq!(exponents.len(), fixture.expanded.len() / group);
    let e4m3_max = f8_e4m3_to_f32(0x7E);

    for (g, (block, restored)) in fixture
        .expanded
        .chunks_exact(group)
        .zip(recovered.chunks_exact(group))
        .enumerate()
    {
        let group_max = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        // E8M0 stores a biased exponent and nothing else.
        let exponent = i32::from(exponents[g]) - 127;
        let scale = 2f32.powi(exponent);
        assert!(
            group_max <= e4m3_max * scale,
            "block {g}: E8M0 scale 2^{exponent} is below amax / 448 = {}, so its maximum \
             {group_max} scales to {} and saturates at 448 (the scale was rounded down, \
             see ml-explore/mlx#4353)",
            group_max / e4m3_max,
            group_max / scale
        );

        let bound = group_max / 16.0 + f32::EPSILON;
        for (k, (expected, actual)) in block.iter().zip(restored).enumerate() {
            let error = (expected - actual).abs();
            assert!(
                error <= bound,
                "element {}: mxfp8 error {error} exceeded {bound} (group max {group_max}, \
                 scale 2^{exponent})",
                g * group + k
            );
        }
    }
}

/// A block-aligned tensor takes the no-padding branch, and it must agree with
/// the reference there too. Without this the padded case could pass while the
/// common case silently took a different route.
#[test]
fn fp8_block_requantize_matches_direct_path_when_block_aligned() {
    let fixture = block_fp8_fixture("model.layers.1.self_attn.q_proj", 256, 128, 0x0BAD_F00D);
    let rows = fixture.rows as i32;
    let cols = fixture.cols as i32;

    let converted = requantize_block_fp8_weights(fixture.weights, SUPPORTED_FP8_BLOCK)
        .expect("block FP8 requantization");
    let packed = converted
        .get("model.layers.1.self_attn.q_proj.weight")
        .expect("packed weight");
    let scales = converted
        .get("model.layers.1.self_attn.q_proj.scales")
        .expect("mxfp8 scales");

    let reference_dense = mlxcel_core::from_slice_f32(&fixture.expanded, &[rows, cols]);
    let reference = mlxcel_core::quantize_weights_with_mode(
        &reference_dense,
        MXFP8_GROUP_SIZE,
        MXFP8_BITS,
        MXFP8_MODE,
    );

    assert_eq!(
        raw_bytes(packed),
        raw_bytes(&mlxcel_core::quantized_weights_w(&reference))
    );
    assert_eq!(
        raw_bytes(scales),
        raw_bytes(&mlxcel_core::quantized_weights_scales(&reference))
    );
}

/// The E4M3 decode must run through the byte table rather than any float
/// reinterpretation: every one of the 254 finite encodings has to survive the
/// device LUT lookup exactly.
#[test]
fn fp8_block_requantize_decodes_every_finite_e4m3_byte() {
    // One row of 256 columns holds every byte; the block scale is 1.0 so the
    // reconstruction is the decode alone.
    let bytes: Vec<u8> = (0..=u8::MAX).collect();
    let mut weights = WeightMap::new();
    weights.insert(
        "probe.weight".to_string(),
        mlxcel_core::from_bytes(&bytes, &[1, 256], mlxcel_core::dtype::UINT8),
    );
    // bf16 1.0 is 0x3F80; the row pads to one block and the 256 columns span
    // two, so the sidecar is [1, 2] and both scales are 1.0.
    let unit_scales: Vec<u8> = [0x3F80u16, 0x3F80]
        .iter()
        .flat_map(|bits| bits.to_le_bytes())
        .collect();
    weights.insert(
        "probe.weight_scale_inv".to_string(),
        mlxcel_core::from_bytes_f16(&unit_scales, &[1, 2], true),
    );

    let lut = super::e4m3_lookup_table();
    let restored = super::reconstruct_block_fp8_tensor(
        "probe.weight",
        weights["probe.weight"].as_ref().expect("weight"),
        weights["probe.weight_scale_inv"]
            .as_ref()
            .expect("scale_inv"),
        SUPPORTED_FP8_BLOCK,
        lut.as_ref().expect("lut"),
    )
    .expect("reconstruction");

    let decoded = read_f32(&restored);
    for byte in 0..=u8::MAX {
        let expected = f8_e4m3_to_f32(byte);
        let actual = decoded[byte as usize];
        if expected.is_nan() {
            assert!(actual.is_nan(), "byte {byte:#04x} should decode to NaN");
        } else {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "byte {byte:#04x} decoded to {actual} instead of {expected}"
            );
        }
    }

    // The encoder round-trips every finite decode, so the table is the exact
    // inverse the checkpoint was written with.
    for byte in 0..=u8::MAX {
        let value = f8_e4m3_to_f32(byte);
        if value.is_nan() {
            continue;
        }
        assert_eq!(
            f32_to_f8_e4m3(value),
            byte,
            "byte {byte:#04x} failed to round-trip"
        );
    }
}

/// mxfp8 packs one E8M0 exponent per 32 values along the last axis, so a width
/// that is not a multiple of 32 has no representation. Refuse it by name
/// instead of letting MLX abort inside the first quantized matmul.
#[test]
fn fp8_block_requantize_rejects_odd_width() {
    let fixture = block_fp8_fixture("model.layers.0.mlp.up_proj", 128, 100, 0x1234_5678);
    let error = expect_requantize_error(
        fixture.weights,
        "a width of 100 is not a multiple of the mxfp8 group size",
    );
    assert!(
        error.contains("100") && error.contains("group size"),
        "unhelpful error: {error}"
    );
}

/// A sidecar with no weight, or with a shape that does not tile the weight, is
/// a corrupt pairing. Both mis-scale silently if accepted.
#[test]
fn fp8_block_requantize_rejects_broken_pairings() {
    let mut orphan = WeightMap::new();
    orphan.insert(
        "model.layers.0.mlp.gate_proj.weight_scale_inv".to_string(),
        mlxcel_core::from_bytes_f16(&0x3F80u16.to_le_bytes(), &[1, 1], true),
    );
    let error = expect_requantize_error(orphan, "a sidecar without its weight must fail");
    assert!(error.contains("no matching"), "unhelpful error: {error}");

    let fixture = block_fp8_fixture("model.layers.0.mlp.gate_proj", 256, 256, 0x9999);
    let mut wrong_scale = fixture.weights;
    wrong_scale.insert(
        "model.layers.0.mlp.gate_proj.weight_scale_inv".to_string(),
        mlxcel_core::from_bytes_f16(&[0x80, 0x3F, 0x80, 0x3F], &[1, 2], true),
    );
    let error =
        expect_requantize_error(wrong_scale, "a [1, 2] sidecar cannot tile a 256x256 weight");
    assert!(error.contains("[2, 2]"), "unhelpful error: {error}");
}

/// A weight already decoded to a float dtype has lost the byte-to-block
/// pairing, so accepting it would apply the scales to the wrong numbers.
#[test]
fn fp8_block_requantize_rejects_predecoded_weights() {
    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.self_attn.o_proj.weight".to_string(),
        mlxcel_core::zeros(&[128, 128], dtype::FLOAT16),
    );
    weights.insert(
        "model.layers.0.self_attn.o_proj.weight_scale_inv".to_string(),
        mlxcel_core::from_bytes_f16(&0x3F80u16.to_le_bytes(), &[1, 1], true),
    );
    let error = expect_requantize_error(weights, "a pre-decoded float weight must fail");
    assert!(error.contains("uint8"), "unhelpful error: {error}");
}

/// Tensors without a sidecar stay exactly as they were: norms, embeddings, the
/// gated-delta conv and the whole vision tower arrive dense in these
/// checkpoints, and `UnifiedLinear` decides per tensor from `.scales`.
#[test]
fn fp8_block_requantize_leaves_unpaired_tensors_dense() {
    let fixture = block_fp8_fixture("model.layers.0.mlp.down_proj", 128, 128, 0xFEED);
    let mut weights = fixture.weights;
    weights.insert(
        "model.norm.weight".to_string(),
        mlxcel_core::ones(&[128], dtype::BFLOAT16),
    );
    weights.insert(
        "model.layers.0.linear_attn.conv1d.weight".to_string(),
        mlxcel_core::ones(&[256, 4, 1], dtype::BFLOAT16),
    );

    let converted =
        requantize_block_fp8_weights(weights, SUPPORTED_FP8_BLOCK).expect("requantization");

    assert_eq!(
        mlxcel_core::array_dtype(converted["model.norm.weight"].as_ref().expect("norm")),
        dtype::BFLOAT16
    );
    assert_eq!(
        mlxcel_core::array_shape(
            converted["model.layers.0.linear_attn.conv1d.weight"]
                .as_ref()
                .expect("conv1d")
        ),
        vec![256, 4, 1]
    );
    assert!(!converted.contains_key("model.norm.scales"));
    assert!(!has_block_fp8_weights(&converted));
}

/// A checkpoint with no sidecars must come back untouched, so the pre-pass is
/// safe to run unconditionally on the Qwen3.5 paths.
#[test]
fn fp8_block_requantize_is_identity_without_sidecars() {
    let mut weights = WeightMap::new();
    weights.insert(
        "model.embed_tokens.weight".to_string(),
        mlxcel_core::ones(&[8, 8], dtype::BFLOAT16),
    );
    let converted =
        requantize_block_fp8_weights(weights, SUPPORTED_FP8_BLOCK).expect("requantization");
    assert_eq!(converted.len(), 1);
    assert!(converted.contains_key("model.embed_tokens.weight"));
}

#[test]
fn qwen_fp8_config_detection() {
    let supported = json!({
        "model_type": "qwen3_5",
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        }
    });
    let detected = qwen_fp8_block_quantization(&supported)
        .expect("detection")
        .expect("a [128, 128] fp8 block layout is supported");
    assert_eq!(
        detected,
        Fp8BlockQuantization {
            block_rows: 128,
            block_cols: 128,
            group_size: MXFP8_GROUP_SIZE,
            bits: MXFP8_BITS,
            mode: MXFP8_MODE,
        }
    );
    assert_eq!(
        detected.effective_config(),
        json!({"group_size": 32, "bits": 8, "mode": "mxfp8"})
    );

    // The block extent is what pairs a scale with its elements, so an
    // unvalidated geometry is an error rather than a silent 128 assumption.
    let wrong_block = json!({
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "weight_block_size": [64, 64],
        }
    });
    let error = qwen_fp8_block_quantization(&wrong_block).expect_err("[64, 64] is unsupported");
    assert!(error.contains("weight_block_size"), "unhelpful: {error}");

    // Another quantization method is simply not this path.
    let int4 = json!({"quantization_config": {"quant_method": "int4", "bits": 4}});
    assert_eq!(qwen_fp8_block_quantization(&int4).expect("detection"), None);

    // No quantization metadata at all.
    assert_eq!(
        qwen_fp8_block_quantization(&json!({"model_type": "qwen3_5"})).expect("detection"),
        None
    );

    // Per-tensor FP8 has a different sidecar layout and is refused by name.
    let per_tensor = json!({"quantization_config": {"quant_method": "fp8", "fmt": "e4m3"}});
    let error = qwen_fp8_block_quantization(&per_tensor).expect_err("per-tensor fp8");
    assert!(error.contains("weight_block_size"), "unhelpful: {error}");

    // A non-E4M3 float8 format would decode with the wrong exponent width.
    let e5m2 = json!({
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e5m2",
            "weight_block_size": [128, 128],
        }
    });
    let error = qwen_fp8_block_quantization(&e5m2).expect_err("e5m2");
    assert!(error.contains("E4M3"), "unhelpful: {error}");

    // The VLM wrappers place the block under text_config on some releases.
    let nested = json!({
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "quantization_config": {
                "quant_method": "fp8",
                "fmt": "e4m3",
                "weight_block_size": [128, 128],
            }
        }
    });
    assert!(
        qwen_fp8_block_quantization(&nested)
            .expect("detection")
            .is_some()
    );
}

#[test]
fn merge_fp8_block_quantization_respects_an_existing_block() {
    let detected = Fp8BlockQuantization {
        block_rows: 128,
        block_cols: 128,
        group_size: MXFP8_GROUP_SIZE,
        bits: MXFP8_BITS,
        mode: MXFP8_MODE,
    };

    let mut fresh = json!({"model_type": "qwen3_5_text"});
    assert!(merge_fp8_block_quantization(&mut fresh, detected).expect("merge"));
    assert_eq!(fresh["quantization"]["group_size"], json!(32));
    assert_eq!(fresh["quantization"]["bits"], json!(8));

    // A pre-converted checkpoint already declares its own layout; the FP8
    // pre-pass must not overwrite it.
    let mut converted = json!({"quantization": {"group_size": 64, "bits": 4}});
    assert!(!merge_fp8_block_quantization(&mut converted, detected).expect("merge"));
    assert_eq!(converted["quantization"]["bits"], json!(4));
}
