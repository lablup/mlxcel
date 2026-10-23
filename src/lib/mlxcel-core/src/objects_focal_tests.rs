// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
use crate::*;

#[test]
fn objects_focal_rejects_malformed_inputs_at_both_boundaries() {
    let uv = zeros(&[1, 4, 2], dtype::FLOAT32);
    let xyz = zeros(&[1, 4, 3], dtype::FLOAT32);
    for trace in [false, true] {
        let rejected = |u: &MlxArray, x: &MlxArray| {
            if trace {
                objects_focal_shift_states_for_test(u, x).is_err()
            } else {
                compiled_objects_focal_shift(u, x).is_err()
            }
        };
        for shape in [&[4, 2][..], &[1, 4, 3], &[2, 4, 2], &[1, 3, 2], &[1, 0, 2]] {
            assert!(
                rejected(&zeros(shape, dtype::FLOAT32), &xyz),
                "UV {shape:?}"
            );
        }
        for shape in [&[4, 3][..], &[1, 4, 2], &[2, 4, 3], &[1, 3, 3], &[0, 4, 3]] {
            assert!(
                rejected(&uv, &zeros(shape, dtype::FLOAT32)),
                "XYZ {shape:?}"
            );
        }
        for dtype in [dtype::BFLOAT16, dtype::FLOAT16, dtype::INT32] {
            assert!(rejected(&astype(&uv, dtype), &xyz));
            assert!(rejected(&uv, &astype(&xyz, dtype)));
        }
    }
}

#[test]
fn objects_focal_analytic_initial_state_and_output_contract() {
    // Exactly representable symmetric rays at unit depth: f=1, shift=0,
    // residual/Jacobian/cost/gradient/Hessian all zero at initialization.
    let uv = from_slice_f32(&[1., 0., 0., 1., -1., 0., 0., -1.], &[1, 4, 2]);
    let xyz = from_slice_f32(
        &[1., 0., 1., 0., 1., 1., -1., 0., 1., 0., -1., 1.],
        &[1, 4, 3],
    );
    let states = objects_focal_shift_states_for_test(&uv, &xyz).unwrap();
    assert_eq!(array_shape(&states), [31, 1, 7]);
    assert_eq!(array_dtype(&states), dtype::FLOAT32);
    let values = utils::array_to_vec_f32(&states);
    assert_eq!(&values[..7], &[0., 1e-3, 0., 0., 0., 0., 1.]);
    for state in values[7..].chunks_exact(7) {
        assert_eq!(state[0], 0.);
        assert_eq!(state[1], 1e-3f32 * 5.);
        assert_eq!(&state[2..], &[1., 0., 0., 0., 1.]);
    }
    let output = compiled_objects_focal_shift(&uv, &xyz).unwrap();
    assert_eq!(array_shape(&output), [1, 2]);
    assert_eq!(array_dtype(&output), dtype::FLOAT32);
    assert_eq!(utils::array_to_vec_f32(&output), [1., 0.]);
}
