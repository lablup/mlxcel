// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn input(height: usize, width: usize) -> UniquePtr<MlxArray> {
    let values: Vec<f32> = (0..height * width)
        .map(|index| (((index * 37) % 101) as i32 - 50) as f32 / 17.0)
        .collect();
    mlxcel_core::from_slice_f32(&values, &[1, height as i32, width as i32, 1])
}

fn assert_probes(actual: &MlxArray, expected: &[(usize, f32)]) {
    let values = mlxcel_core::utils::array_to_vec_f32(actual);
    for &(index, expected) in expected {
        let actual = values[index];
        assert!(
            (actual - expected).abs() <= 4e-5,
            "index {index}: expected {expected}, got {actual}"
        );
    }
}

fn assert_values_close(left: &MlxArray, right: &MlxArray) {
    let left = mlxcel_core::utils::array_to_vec_f32(left);
    let right = mlxcel_core::utils::array_to_vec_f32(right);
    assert_eq!(left.len(), right.len());
    for (index, (left, right)) in left.into_iter().zip(right).enumerate() {
        assert!(
            (left - right).abs() <= 4e-5,
            "index {index}: expected {right}, got {left}"
        );
    }
}

// Literals were generated once with tests/fixtures/generate_dense_interpolate_fixtures.py,
// which runs torch.nn.functional.interpolate on the fixed inputs below.
#[test]
fn upstream_bilinear_literals_match_in_both_directions() {
    let a = input(7, 9);
    assert_probes(
        &resize_bilinear_nhwc(&a, (14, 5), false, false).unwrap(),
        &[
            (0, -2.07058811),
            (1, 0.658824205),
            (4, 1.71764588),
            (10, -0.747058868),
            (23, 0.888234735),
            (35, 0.397058904),
            (65, 0.200000048),
            (68, 2.4470582),
            (69, 0.423528194),
        ],
    );
    assert_probes(
        &resize_bilinear_nhwc(&a, (14, 5), false, true).unwrap(),
        &[
            (0, -1.69747913),
            (1, -0.172793865),
            (4, 1.3445369),
            (10, -0.692226946),
            (23, 0.2159926),
            (35, 0.324579865),
            (65, -0.0210083164),
            (68, 1.49632359),
            (69, 0.474789947),
        ],
    );

    let b = input(14, 5);
    assert_probes(
        &resize_bilinear_nhwc(&b, (7, 9), false, false).unwrap(),
        &[
            (0, -0.470588207),
            (1, -0.735294104),
            (4, 0.911764741),
            (9, 0.5),
            (21, 0.323529392),
            (31, 0.852941155),
            (58, 0.794117689),
            (61, -0.529412031),
            (62, -0.794117689),
        ],
    );
    assert_probes(
        &resize_bilinear_nhwc(&b, (7, 9), false, true).unwrap(),
        &[
            (0, -0.260504246),
            (1, -0.666666746),
            (4, 0.69747901),
            (9, 0.5),
            (21, 0.323529303),
            (31, 0.852941215),
            (58, 1.00840342),
            (61, -0.456582785),
            (62, -0.579831958),
        ],
    );
}

#[test]
fn upstream_bicubic_size_literals_match_in_both_directions() {
    let a = input(7, 9);
    assert_probes(
        &resize_bicubic_nhwc(&a, Some((14, 5)), None, false).unwrap(),
        &[
            (0, -2.48464966),
            (1, 1.29624152),
            (4, 2.09783149),
            (10, -0.743794441),
            (23, 1.64457595),
            (35, 0.132367209),
            (65, -0.01364756),
            (68, 3.11410022),
            (69, 0.364709646),
        ],
    );
    assert_probes(
        &resize_bicubic_nhwc(&a, Some((14, 5)), None, true).unwrap(),
        &[
            (0, -1.88656294),
            (1, -0.0548147932),
            (4, 1.50378382),
            (10, -0.717688322),
            (23, 0.348062992),
            (35, 0.174739793),
            (65, -0.221480012),
            (68, 1.66238093),
            (69, 0.54034245),
        ],
    );

    let b = input(14, 5);
    assert_probes(
        &resize_bicubic_nhwc(&b, Some((7, 9)), None, false).unwrap(),
        &[
            (0, -0.296773136),
            (1, -0.74775368),
            (4, 1.00551474),
            (9, 0.580065489),
            (21, 0.675555229),
            (31, 0.852941155),
            (58, 0.700367689),
            (61, -0.681985795),
            (62, -0.967932999),
        ],
    );
    assert_probes(
        &resize_bicubic_nhwc(&b, Some((7, 9)), None, true).unwrap(),
        &[
            (0, -0.197107658),
            (1, -0.748206973),
            (4, 0.867831647),
            (9, 0.756815016),
            (21, 0.693133593),
            (31, 1.13143373),
            (58, 1.13635254),
            (61, -0.634362221),
            (62, -0.797623694),
        ],
    );
}

#[test]
fn upstream_bicubic_scale_factor_literals_match_in_both_directions() {
    let a = input(7, 9);
    assert_probes(
        &resize_bicubic_nhwc(&a, None, Some((2.0, 0.6)), false).unwrap(),
        &[
            (0, -2.59783769),
            (1, 1.85225189),
            (4, 0.225638911),
            (10, -0.942854404),
            (23, 0.0419154689),
            (35, 0.404766262),
            (65, 0.353826582),
            (68, 2.09779048),
            (69, -0.69623214),
        ],
    );
    assert_probes(
        &resize_bicubic_nhwc(&a, None, Some((2.0, 0.6)), true).unwrap(),
        &[
            (0, -1.88656294),
            (1, -0.0548147932),
            (4, 1.50378382),
            (10, -0.717688322),
            (23, 0.348062992),
            (35, 0.174739793),
            (65, -0.221480012),
            (68, 1.66238093),
            (69, 0.54034245),
        ],
    );

    let b = input(14, 5);
    assert_probes(
        &resize_bicubic_nhwc(&b, None, Some((0.5, 1.8)), false).unwrap(),
        &[
            (0, -0.296773136),
            (1, -0.74775368),
            (4, 1.00551474),
            (9, 0.580065489),
            (21, 0.675555229),
            (31, 0.852941155),
            (58, 0.700367689),
            (61, -0.681985795),
            (62, -0.967932999),
        ],
    );
    assert_probes(
        &resize_bicubic_nhwc(&b, None, Some((0.5, 1.8)), true).unwrap(),
        &[
            (0, -0.197107658),
            (1, -0.748206973),
            (4, 0.867831647),
            (9, 0.756815016),
            (21, 0.693133593),
            (31, 1.13143373),
            (58, 1.13635254),
            (61, -0.634362221),
            (62, -0.797623694),
        ],
    );
}

#[test]
fn antialiased_bicubic_uses_the_floored_output_size_scale() {
    let source = input(7, 9);
    let from_scale = resize_bicubic_nhwc(&source, None, Some((1.3, 0.72)), true).unwrap();
    let from_size = resize_bicubic_nhwc(&source, Some((9, 6)), None, true).unwrap();
    assert_values_close(&from_scale, &from_size);

    // The non-antialiased upstream path retains the supplied scale, so it is
    // observably different from recomputing the effective 9/7 and 6/9 scales.
    let from_scale = resize_bicubic_nhwc(&source, None, Some((1.3, 0.72)), false).unwrap();
    let from_size = resize_bicubic_nhwc(&source, Some((9, 6)), None, false).unwrap();
    let from_scale = mlxcel_core::utils::array_to_vec_f32(&from_scale);
    let from_size = mlxcel_core::utils::array_to_vec_f32(&from_size);
    assert!(
        from_scale
            .iter()
            .zip(from_size)
            .any(|(left, right)| (left - right).abs() > 1e-4)
    );
}

#[test]
fn identity_size_paths_preserve_dtype_even_for_a_nonunit_scale() {
    let source = mlxcel_core::astype(&input(7, 9), mlxcel_core::dtype::FLOAT16);

    let bilinear = resize_bilinear_nhwc(&source, (7, 9), false, true).unwrap();
    assert_eq!(
        mlxcel_core::array_dtype(&bilinear),
        mlxcel_core::dtype::FLOAT16
    );
    assert_values_close(&bilinear, &source);

    // Both products floor back to the original dimensions. Upstream returns
    // the tensor before consulting the antialias weight path.
    let bicubic = resize_bicubic_nhwc(&source, None, Some((1.01, 1.01)), true).unwrap();
    assert_eq!(
        mlxcel_core::array_dtype(&bicubic),
        mlxcel_core::dtype::FLOAT16
    );
    assert_values_close(&bicubic, &source);
}

#[test]
fn upstream_nearest_literals_match_in_both_directions() {
    assert_probes(
        &resize_nearest_nhwc(&input(7, 9), (14, 5)).unwrap(),
        &[
            (0, -2.94117641),
            (1, -0.764705896),
            (4, 0.411764711),
            (10, -1.17647064),
            (23, -0.411764711),
            (35, 2.35294127),
            (65, 1.70588231),
            (68, 0.70588237),
            (69, -0.882352948),
        ],
    );
    assert_probes(
        &resize_nearest_nhwc(&input(14, 5), (7, 9)).unwrap(),
        &[
            (0, -2.94117641),
            (1, -2.94117641),
            (4, 1.41176474),
            (9, 1.0),
            (21, 1.17647064),
            (31, 1.35294116),
            (58, 1.29411769),
            (61, -2.47058821),
            (62, -0.294117659),
        ],
    );
}
