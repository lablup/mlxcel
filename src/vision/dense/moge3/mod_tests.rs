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

use serde_json::{Map, Value, json};

use super::config::Moge3Config;
use super::conv_stack::sanitize_conv_transpose_weight;
use super::geometry::{normalized_view_plane_uv, recover_focal_shift};
use super::model::token_grid;
use super::processor::{
    InferOptions, image_to_array, validate_float_image, validate_output_budget,
};
use super::sparse::{
    SparseShape, SubmanifoldConv3d, neighbor_construction_peak_bytes, sparse_pool2x_mean,
    sparse_upsample2x_nearest, submanifold_conv3d_neighbor_map,
};

fn flat_f32(value: &Value, out: &mut Vec<f32>) {
    match value {
        Value::Array(v) => v.iter().for_each(|x| flat_f32(x, out)),
        Value::Number(v) => out.push(v.as_f64().unwrap() as f32),
        _ => panic!("not numeric"),
    }
}
fn flat_i32(value: &Value, out: &mut Vec<i32>) {
    match value {
        Value::Array(v) => v.iter().for_each(|x| flat_i32(x, out)),
        Value::Number(v) => out.push(v.as_i64().unwrap() as i32),
        _ => panic!("not numeric"),
    }
}
fn f32_array(value: &Value, shape: &[i32]) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let mut data = Vec::new();
    flat_f32(value, &mut data);
    mlxcel_core::from_slice_f32(&data, shape)
}
fn i32_array(value: &Value, shape: &[i32]) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let mut data = Vec::new();
    flat_i32(value, &mut data);
    mlxcel_core::from_slice_i32(&data, shape)
}
fn assert_close(actual: &mlxcel_core::MlxArray, expected: &mlxcel_core::MlxArray, tolerance: f64) {
    assert!(
        mlxcel_core::item_bool(&mlxcel_core::allclose(
            actual, expected, tolerance, tolerance
        )),
        "arrays differ: actual {:?}, expected {:?}",
        mlxcel_core::array_shape(actual),
        mlxcel_core::array_shape(expected)
    );
}

#[test]
fn moge3_sparse_primitives_match_authentic_six_voxel_oracle() {
    let oracle: Value = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/moge3/sparse-oracle.json"
    ))
    .unwrap();
    let coords = i32_array(&oracle["coords"], &[6, 4]);
    let feats = f32_array(&oracle["features"], &[6, 2]);
    let shape = SparseShape::new(1, 5, 5, 5).unwrap();
    let neighbors = submanifold_conv3d_neighbor_map(&coords, shape).unwrap();
    let expected_neighbors = i32_array(&oracle["neighbors"], &[6, 27]);
    assert!(mlxcel_core::item_bool(&mlxcel_core::array_equal(
        &neighbors,
        &expected_neighbors,
        false
    )));
    let conv = SubmanifoldConv3d::new(
        f32_array(&oracle["weight"], &[3, 3, 3, 3, 2]),
        f32_array(&oracle["bias"], &[3]),
    )
    .unwrap();
    let (output, reused) = conv
        .forward(&feats, &coords, shape, Some(&neighbors))
        .unwrap();
    assert_close(&output, &f32_array(&oracle["conv"], &[6, 3]), 0.0);
    assert!(mlxcel_core::item_bool(&mlxcel_core::array_equal(
        &reused,
        &expected_neighbors,
        false
    )));
    let pooled = sparse_pool2x_mean(&feats, &coords, shape).unwrap();
    assert_eq!(pooled.shape, SparseShape::new(1, 3, 3, 3).unwrap());
    assert_close(
        &pooled.features,
        &f32_array(&oracle["pool_features"], &[3, 2]),
        0.0,
    );
    assert!(mlxcel_core::item_bool(&mlxcel_core::array_equal(
        &pooled.coords,
        &i32_array(&oracle["pool_coords"], &[3, 4]),
        false
    )));
    assert!(mlxcel_core::item_bool(&mlxcel_core::array_equal(
        &pooled.parent_indices,
        &i32_array(&oracle["parents"], &[6]),
        false
    )));
    let (upsampled, up_coords, up_shape) =
        sparse_upsample2x_nearest(&pooled.features, &pooled.parent_indices, &coords, shape)
            .unwrap();
    assert_close(&upsampled, &f32_array(&oracle["upsample"], &[6, 2]), 0.0);
    assert!(mlxcel_core::item_bool(&mlxcel_core::array_equal(
        &up_coords, &coords, false
    )));
    assert_eq!(up_shape, shape);
}

#[test]
fn moge3_geometry_recovers_known_focal_and_nonzero_shift() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/moge3/geometry-search-oracle.json"
    ))
    .unwrap();
    let uv = normalized_view_plane_uv(16, 12, None).unwrap();
    let u = mlxcel_core::reshape(
        &mlxcel_core::slice(&uv, &[0, 0, 0], &[12, 16, 1]),
        &[12, 16],
    );
    let v = mlxcel_core::reshape(
        &mlxcel_core::slice(&uv, &[0, 0, 1], &[12, 16, 2]),
        &[12, 16],
    );
    let z = mlxcel_core::add(
        &mlxcel_core::from_slice_f32(&[2.0], &[1]),
        &mlxcel_core::add(
            &mlxcel_core::multiply(&u, &mlxcel_core::from_slice_f32(&[0.4], &[1])),
            &mlxcel_core::multiply(&v, &mlxcel_core::from_slice_f32(&[0.3], &[1])),
        ),
    );
    let points = mlxcel_core::concatenate(
        &mlxcel_core::multiply(
            &uv,
            &mlxcel_core::expand_dims(
                &mlxcel_core::divide(&z, &mlxcel_core::from_slice_f32(&[1.7], &[1])),
                -1,
            ),
        ),
        &mlxcel_core::expand_dims(&z, -1),
        -1,
    );
    let points = mlxcel_core::expand_dims(&points, 0);
    let (focal, shift) = recover_focal_shift(&points, None, None).unwrap();
    mlxcel_core::eval(&focal);
    mlxcel_core::eval(&shift);
    assert!(
        (mlxcel_core::item_f32(&focal) - fixture["geometry"]["focal"].as_f64().unwrap() as f32)
            .abs()
            <= 1e-4
    );
    assert!(mlxcel_core::item_f32(&shift).abs() <= 1e-4);
    let shifted = mlxcel_core::subtract(
        &points,
        &mlxcel_core::from_slice_f32(&[0.0, 0.0, 0.35], &[1, 1, 1, 3]),
    );
    let (focal, shift) = recover_focal_shift(&shifted, None, None).unwrap();
    mlxcel_core::eval(&focal);
    mlxcel_core::eval(&shift);
    assert!((mlxcel_core::item_f32(&focal) - 1.7).abs() <= 1e-4);
    assert!((mlxcel_core::item_f32(&shift) - 0.35).abs() <= 1e-4);
    let fixed = mlxcel_core::from_slice_f32(&[1.7], &[1]);
    let (_, shift) = recover_focal_shift(&points, None, Some(&fixed)).unwrap();
    mlxcel_core::eval(&shift);
    assert!(mlxcel_core::item_f32(&shift).abs() <= 1e-4);
    let empty = mlxcel_core::zeros(&[1, 12, 16], mlxcel_core::dtype::BOOL);
    let (focal, shift) = recover_focal_shift(&points, Some(&empty), None).unwrap();
    mlxcel_core::eval(&focal);
    mlxcel_core::eval(&shift);
    assert_eq!(mlxcel_core::item_f32(&focal), 1.0);
    assert_eq!(mlxcel_core::item_f32(&shift), 0.0);
}

#[test]
fn moge3_conv_transpose_sanitizer_is_idempotent() {
    let source = mlxcel_core::reshape(&mlxcel_core::arange_f32(0.0, 48.0, 1.0), &[4, 3, 2, 2]);
    let once = sanitize_conv_transpose_weight(&source, 4, 3).unwrap();
    assert_eq!(mlxcel_core::array_shape(&once), vec![3, 2, 2, 4]);
    let twice = sanitize_conv_transpose_weight(&once, 4, 3).unwrap();
    assert!(mlxcel_core::item_bool(&mlxcel_core::array_equal(
        &once, &twice, false
    )));
}

#[test]
fn moge3_options_validate_ranges_types_and_combinations() {
    let config = Moge3Config::default();
    let default = InferOptions::parse(&config, &Map::new()).unwrap();
    assert_eq!(default.num_tokens, 3600);
    assert_eq!(default.refine_steps, 3);
    let cli = InferOptions::parse(
        &config,
        &serde_json::from_value(json!({"task": "geometry"})).unwrap(),
    )
    .unwrap();
    assert_eq!(cli.num_tokens, default.num_tokens);
    let zero=InferOptions::parse(&config,&serde_json::from_value(json!({"resolution_level":0,"refine_steps":0,"return_per_step":true,"force_projection":false,"apply_mask":false,"fov_x":70.0})).unwrap()).unwrap();
    assert_eq!(zero.num_tokens, 1200);
    assert_eq!(zero.refine_steps, 0);
    assert!(zero.return_per_step);
    let explicit: Map<String, Value> =
        serde_json::from_value(json!({"num_tokens": 3600, "resolution_level": 0})).unwrap();
    assert_eq!(
        InferOptions::parse(&config, &explicit).unwrap().num_tokens,
        3600
    );
    for bad in [
        json!({"task":"pose"}),
        json!({"resolution_level":10}),
        json!({"resolution_level":-1}),
        json!({"num_tokens":1199}),
        json!({"refine_steps":17}),
        json!({"fov_x":0}),
        json!({"fov_x":180}),
        json!({"apply_mask":"yes"}),
        json!({"unknown":1}),
    ] {
        let map: Map<String, Value> = serde_json::from_value(bad).unwrap();
        assert!(InferOptions::parse(&config, &map).is_err());
    }

    let direct = InferOptions {
        num_tokens: usize::MAX,
        force_projection: true,
        apply_mask: true,
        fov_x: None,
        refine_steps: 0,
        return_per_step: false,
    };
    assert!(direct.validate(&config).is_err());
    for fov in [f32::NAN, 0.0, 180.0] {
        let direct = InferOptions {
            num_tokens: 3600,
            fov_x: Some(fov),
            ..default
        };
        assert!(direct.validate(&config).is_err());
    }
    let direct = InferOptions {
        refine_steps: usize::MAX,
        ..default
    };
    assert!(direct.validate(&config).is_err());
    let underflow: Map<String, Value> = serde_json::from_str(r#"{"fov_x":5e-324}"#).unwrap();
    assert!(InferOptions::parse(&config, &underflow).is_err());
}

#[test]
fn moge3_grid_rounds_ties_to_even_and_budgets_retained_outputs() {
    assert_eq!(token_grid(1200, 192.0).unwrap(), (2, 480));
    let options = InferOptions {
        num_tokens: 3600,
        force_projection: true,
        apply_mask: true,
        fov_x: None,
        refine_steps: 3,
        return_per_step: true,
    };
    validate_output_budget(&[1, 480, 640, 3], options).unwrap();
    let excessive = InferOptions {
        refine_steps: 16,
        ..options
    };
    assert!(validate_output_budget(&[1, 4096, 4096, 3], excessive).is_err());
    assert!(
        neighbor_construction_peak_bytes(
            921_600,
            32,
            super::sparse::neighbor_map_bytes(921_600).unwrap(),
        )
        .unwrap()
            <= super::config::MAX_SPARSE_WORKING_BYTES
    );
    assert!(
        neighbor_construction_peak_bytes(
            2_000_000,
            32,
            super::sparse::neighbor_map_bytes(2_000_000).unwrap(),
        )
        .unwrap()
            > super::config::MAX_SPARSE_WORKING_BYTES
    );
}

#[test]
fn moge3_config_rejects_wrong_task_schema_and_group_norm_width() {
    let valid = Moge3Config::default();
    valid.validate().unwrap();

    let mut config = valid.clone();
    config
        .points_head
        .as_mut()
        .unwrap()
        .dim_out
        .as_mut()
        .unwrap()[4] = Some(2);
    assert!(config.validate().is_err());

    let mut config = valid.clone();
    config.scale_head.as_mut().unwrap().dims[3] = 2;
    assert!(config.validate().is_err());

    let mut config = valid.clone();
    config.refiner.as_mut().unwrap().in_channels = 4;
    assert!(config.validate().is_err());

    let mut config = valid;
    config.neck.dim_res_blocks[1] = 100;
    config.neck.res_block_in_norm = "group_norm".into();
    assert!(config.validate().is_err());
}

#[test]
fn moge3_sparse_public_inputs_fail_closed() {
    assert!(SparseShape::new(i64::from(i32::MAX) + 1, 1, 1, 1).is_err());
    let shape = SparseShape::new(1, 2, 2, 2).unwrap();
    let duplicate = mlxcel_core::from_slice_i32(&[0, 0, 0, 0, 0, 0, 0, 0], &[2, 4]);
    assert!(submanifold_conv3d_neighbor_map(&duplicate, shape).is_err());
    let out_of_bounds = mlxcel_core::from_slice_i32(&[0, 2, 0, 0], &[1, 4]);
    assert!(submanifold_conv3d_neighbor_map(&out_of_bounds, shape).is_err());

    let zero_channel_weight = mlxcel_core::zeros(&[1, 3, 3, 3, 0], mlxcel_core::dtype::FLOAT32);
    let bias = mlxcel_core::zeros(&[1], mlxcel_core::dtype::FLOAT32);
    assert!(SubmanifoldConv3d::new(zero_channel_weight, bias).is_err());

    let weight = mlxcel_core::zeros(&[1, 3, 3, 3, 1], mlxcel_core::dtype::FLOAT32);
    let bias = mlxcel_core::zeros(&[1], mlxcel_core::dtype::FLOAT32);
    let conv = SubmanifoldConv3d::new(weight, bias).unwrap();
    let feats = mlxcel_core::ones(&[1, 1], mlxcel_core::dtype::FLOAT32);
    let coords = mlxcel_core::from_slice_i32(&[0, 0, 0, 0], &[1, 4]);
    let invalid_neighbor = mlxcel_core::full_f32(&[1, 27], 1.0, mlxcel_core::dtype::INT32);
    assert!(
        conv.forward(&feats, &coords, shape, Some(&invalid_neighbor))
            .is_err()
    );
    let invalid_parent = mlxcel_core::from_slice_i32(&[1], &[1]);
    assert!(sparse_upsample2x_nearest(&feats, &invalid_parent, &coords, shape).is_err());
}

#[test]
fn moge3_processor_scales_integer_images_and_rejects_invalid_float_ranges() {
    let image =
        image::DynamicImage::ImageRgb8(image::RgbImage::from_raw(1, 1, vec![0, 128, 255]).unwrap());
    let array = image_to_array(&image).unwrap();
    assert_eq!(mlxcel_core::array_shape(&array), vec![1, 1, 1, 3]);
    let middle = mlxcel_core::slice(&array, &[0, 0, 0, 1], &[1, 1, 1, 2]);
    assert!((mlxcel_core::item_f32(&middle) - 128.0 / 255.0).abs() <= f32::EPSILON);

    for values in [
        vec![-0.01, 0.0, 1.0],
        vec![0.0, 1.01, 1.0],
        vec![0.0, f32::NAN, 1.0],
    ] {
        let invalid = mlxcel_core::from_slice_f32(&values, &[1, 1, 3]);
        assert!(validate_float_image(&invalid).is_err());
    }
    let empty = mlxcel_core::zeros(&[0, 1, 3], mlxcel_core::dtype::FLOAT32);
    assert!(validate_float_image(&empty).is_err());
}
