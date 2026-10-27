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

use super::load_dense_param;
use mlxcel_core::dtype;
use mlxcel_core::weights::WeightMap;

const GROUP: i32 = 64;
const BITS: i32 = 4;

fn dense(shape: &[i32]) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let len = shape.iter().map(|&dim| dim as usize).product();
    let values: Vec<f32> = (0..len)
        .map(|index| ((index as f32 * 0.071).sin() * 0.8) + index as f32 * 0.0001)
        .collect();
    mlxcel_core::from_slice_f32(&values, shape)
}

fn quantized_map_with_mode(
    key: &str,
    source: &mlxcel_core::MlxArray,
    stored_shape: &[i32],
    group_size: i32,
    bits: i32,
    mode: &str,
) -> WeightMap {
    let source_shape = mlxcel_core::array_shape(source);
    let width = source_shape[source_shape.len() - 1];
    let rows: i32 = source_shape[..source_shape.len() - 1].iter().product();
    let flat = mlxcel_core::reshape(source, &[rows, width]);
    let quantized = mlxcel_core::quantize_weights_with_mode(&flat, group_size, bits, mode);
    let packed = mlxcel_core::quantized_weights_w(&quantized);
    let scales = mlxcel_core::quantized_weights_scales(&quantized);
    let has_biases = mlxcel_core::quantized_weights_has_biases(&quantized);
    let packed_last = mlxcel_core::array_shape(&packed)[1];
    let groups = mlxcel_core::array_shape(&scales)[1];
    let mut packed_shape = stored_shape.to_vec();
    *packed_shape.last_mut().expect("non-empty test shape") = packed_last;
    let mut sidecar_shape = stored_shape.to_vec();
    *sidecar_shape.last_mut().expect("non-empty test shape") = groups;

    let prefix = key.strip_suffix(".weight");
    let scales_key = prefix.map_or_else(|| format!("{key}.scales"), |p| format!("{p}.scales"));
    let biases_key = prefix.map_or_else(|| format!("{key}.biases"), |p| format!("{p}.biases"));
    let mut weights = WeightMap::new();
    weights.insert(
        key.to_string(),
        mlxcel_core::reshape(&packed, &packed_shape),
    );
    weights.insert(scales_key, mlxcel_core::reshape(&scales, &sidecar_shape));
    if has_biases {
        let biases = mlxcel_core::quantized_weights_biases(&quantized);
        weights.insert(biases_key, mlxcel_core::reshape(&biases, &sidecar_shape));
    }
    weights
}

fn quantized_map(key: &str, source: &mlxcel_core::MlxArray, stored_shape: &[i32]) -> WeightMap {
    quantized_map_with_mode(key, source, stored_shape, GROUP, BITS, "affine")
}

fn values(array: &mlxcel_core::MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(array)
}

fn assert_close(actual: &mlxcel_core::MlxArray, expected: &mlxcel_core::MlxArray) {
    assert_eq!(
        mlxcel_core::array_shape(actual),
        mlxcel_core::array_shape(expected)
    );
    for (index, (actual, expected)) in values(actual).iter().zip(values(expected)).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-3,
            "entry {index}: actual {actual}, expected {expected}"
        );
    }
}

#[test]
fn dense_parameter_is_copied_bit_identically_and_shape_checked() {
    let source = mlxcel_core::astype(&dense(&[3, 128]), dtype::BFLOAT16);
    let mut weights = WeightMap::new();
    weights.insert("table".into(), source);
    let loaded = load_dense_param(&weights, "table", &[3, 128], None).expect("dense table");
    assert_eq!(mlxcel_core::array_dtype(&loaded), dtype::BFLOAT16);
    assert_eq!(
        mlxcel_core::array_to_raw_bytes(&loaded),
        mlxcel_core::array_to_raw_bytes(&weights["table"])
    );

    let error = load_dense_param(&weights, "table", &[2, 128], None)
        .err()
        .expect("wrong dense shape must fail");
    assert!(
        error.contains("table") && error.contains("expected [2, 128]"),
        "{error}"
    );
}

#[test]
fn quantized_parameters_dequantize_in_one_two_and_three_dimensions() {
    for shape in [&[128][..], &[3, 128], &[2, 3, 128]] {
        let source = dense(shape);
        let weights = quantized_map("table", &source, shape);
        let loaded = load_dense_param(&weights, "table", shape, Some((GROUP, BITS)))
            .expect("quantized table");
        let packed = &weights["table"];
        let scales = &weights["table.scales"];
        let biases = &weights["table.biases"];
        let rows: i32 = shape[..shape.len() - 1].iter().product();
        let flat_packed = mlxcel_core::reshape(
            packed,
            &[rows, mlxcel_core::array_shape(packed)[shape.len() - 1]],
        );
        let flat_scales = mlxcel_core::reshape(
            scales,
            &[rows, mlxcel_core::array_shape(scales)[shape.len() - 1]],
        );
        let flat_biases = mlxcel_core::reshape(
            biases,
            &[rows, mlxcel_core::array_shape(biases)[shape.len() - 1]],
        );
        let reference = unsafe {
            mlxcel_core::dequantize(
                &flat_packed,
                &flat_scales,
                flat_biases.as_ref().expect("non-null") as *const _,
                GROUP,
                BITS,
                "affine",
            )
        };
        let reference = mlxcel_core::reshape(&reference, shape);
        assert_close(&loaded, &reference);
    }
}

#[test]
fn shape_derived_modes_cover_nondefault_affine_mxfp4_mxfp8_and_nvfp4() {
    let shape = [2, 128];
    let source = dense(&shape);
    for (group_size, bits, mode) in [
        (32, 4, "affine"),
        (32, 4, "mxfp4"),
        (32, 8, "mxfp8"),
        (16, 4, "nvfp4"),
    ] {
        let weights = quantized_map_with_mode("table", &source, &shape, group_size, bits, mode);
        assert_eq!(
            weights.contains_key("table.biases"),
            mode == "affine",
            "{mode} bias presence"
        );
        if mode != "affine" {
            assert_eq!(
                mlxcel_core::array_dtype(&weights["table.scales"]),
                dtype::UINT8
            );
        }
        let loaded = load_dense_param(&weights, "table", &shape, None)
            .unwrap_or_else(|error| panic!("supported {mode} table: {error}"));
        let packed_shape = mlxcel_core::array_shape(&weights["table"]);
        let scales_shape = mlxcel_core::array_shape(&weights["table.scales"]);
        let packed = mlxcel_core::reshape(&weights["table"], &[2, packed_shape[1]]);
        let scales = mlxcel_core::reshape(&weights["table.scales"], &[2, scales_shape[1]]);
        let biases = weights
            .get("table.biases")
            .map(|biases| mlxcel_core::reshape(biases, &[2, scales_shape[1]]));
        let bias_ptr = biases.as_ref().map_or(std::ptr::null(), |biases| {
            biases.as_ref().expect("non-null biases") as *const _
        });
        let reference =
            unsafe { mlxcel_core::dequantize(&packed, &scales, bias_ptr, group_size, bits, mode) };
        assert_close(&loaded, &reference);
    }
}

#[test]
fn quantized_layout_errors_are_reported_before_dequantize() {
    let shape = [2, 3, 128];
    let source = dense(&shape);

    let mut missing_weight = quantized_map("table", &source, &shape);
    missing_weight.remove("table");
    let error = load_dense_param(&missing_weight, "table", &shape, None)
        .err()
        .expect("missing packed weight must fail");
    assert_eq!(error, "Weight not found: table");

    let mut wrong_dtype = quantized_map("table", &source, &shape);
    wrong_dtype.insert("table".into(), dense(&[2, 3, 16]));
    let error = load_dense_param(&wrong_dtype, "table", &shape, None)
        .err()
        .expect("non-uint32 packed weight must fail");
    assert!(error.contains("uint32"), "{error}");

    let mut wrong_leading = quantized_map("table", &source, &shape);
    wrong_leading.insert(
        "table.scales".into(),
        mlxcel_core::zeros(&[3, 2, 2], dtype::FLOAT32),
    );
    let error = load_dense_param(&wrong_leading, "table", &shape, None)
        .err()
        .expect("wrong sidecar leading dimensions must fail");
    assert!(error.contains("leading dimensions"), "{error}");

    let mut wrong_bias = quantized_map("table", &source, &shape);
    wrong_bias.insert(
        "table.biases".into(),
        mlxcel_core::zeros(&[2, 3, 1], dtype::FLOAT32),
    );
    let error = load_dense_param(&wrong_bias, "table", &shape, None)
        .err()
        .expect("wrong bias dimensions must fail");
    assert!(error.contains("biases shape and dtype"), "{error}");

    let mut block_float_with_float_scales = quantized_map("table", &source, &shape);
    block_float_with_float_scales.remove("table.biases");
    let error = load_dense_param(&block_float_with_float_scales, "table", &shape, None)
        .err()
        .expect("block-float scales with float dtype must fail");
    assert!(error.contains("uint8"), "{error}");

    let mut wrong_width = quantized_map("table", &source, &shape);
    wrong_width.insert(
        "table".into(),
        mlxcel_core::from_slice_u32(&[0; 2 * 3 * 15], &[2, 3, 15]),
    );
    let error = load_dense_param(&wrong_width, "table", &shape, None)
        .err()
        .expect("wrong logical width must fail");
    assert!(
        error.contains("shape mismatch") || error.contains("layout"),
        "{error}"
    );

    let packed_without_scales = quantized_map("table", &source, &shape);
    let mut packed_without_scales = packed_without_scales;
    packed_without_scales.remove("table.scales");
    packed_without_scales.remove("table.biases");
    let error = load_dense_param(&packed_without_scales, "table", &shape, None)
        .err()
        .expect("uint32 packed table without scales must fail");
    assert!(error.contains("missing its scales sidecar"), "{error}");

    let mut wrong_scale_rank = quantized_map("table", &source, &shape);
    wrong_scale_rank.insert(
        "table.scales".into(),
        mlxcel_core::zeros(&[6, 2], dtype::FLOAT32),
    );
    let error = load_dense_param(&wrong_scale_rank, "table", &shape, None)
        .err()
        .expect("wrong scale rank must fail");
    assert!(error.contains("invalid shape"), "{error}");
}

#[test]
fn invalid_expected_shapes_are_rejected_without_tensor_ops() {
    let weights = WeightMap::new();
    for shape in [&[][..], &[0, 128], &[2, -1, 128]] {
        let error = load_dense_param(&weights, "table", shape, None)
            .err()
            .expect("invalid expected shape must fail");
        assert!(error.contains("invalid expected shape"), "{error}");
    }
}

#[test]
fn dense_integer_parameter_is_rejected_before_tensor_ops() {
    let mut weights = WeightMap::new();
    weights.insert(
        "table".into(),
        mlxcel_core::from_slice_i32(&[1, 2, 3, 4], &[2, 2]),
    );
    let error = load_dense_param(&weights, "table", &[2, 2], None)
        .err()
        .expect("integer dense parameter must fail");
    assert!(error.contains("floating dtype"), "{error}");
}
