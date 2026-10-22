// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0

//! Test-only numerical diagnostics for exact upstream stage comparisons.

use mlxcel_core::MlxArray;

pub(crate) type StageMetric = (String, f32, f32);

pub(crate) fn parity_metrics(actual: &MlxArray, expected: &MlxArray) -> (f32, f32) {
    assert_eq!(
        mlxcel_core::array_shape(actual),
        mlxcel_core::array_shape(expected)
    );
    let actual = mlxcel_core::astype(actual, mlxcel_core::dtype::FLOAT32);
    let expected = mlxcel_core::astype(expected, mlxcel_core::dtype::FLOAT32);
    for value in [&actual, &expected] {
        assert_eq!(
            mlxcel_core::item_f32(&mlxcel_core::astype(
                &mlxcel_core::all_all(&mlxcel_core::isfinite(value)),
                mlxcel_core::dtype::FLOAT32,
            )),
            1.0
        );
    }
    let difference = mlxcel_core::subtract(&actual, &expected);
    let max_abs = mlxcel_core::item_f32(&mlxcel_core::max_all(&mlxcel_core::abs(&difference)));
    let rms =
        mlxcel_core::item_f32(&mlxcel_core::mean_all(&mlxcel_core::square(&difference))).sqrt();
    let scale =
        mlxcel_core::item_f32(&mlxcel_core::mean_all(&mlxcel_core::square(&expected))).sqrt();
    (max_abs, rms / scale.max(f32::MIN_POSITIVE))
}
