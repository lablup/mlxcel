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

//! Reductions whose output dtype is not their input dtype.
//!
//! MLX gives `sum` / `prod` over `bool` and the narrow integers a 32-bit
//! output, and `any` / `all` a `bool` output whatever the input is. The CUDA
//! reduce overlay (bf16 accumulation in fp32) took the output pointer from the
//! input type, so each of those wrote elements of the wrong size and returned
//! garbage: `sum(x > 0)` over the last axis of a 3 x 4 mask with 3, 0 and 2
//! true entries per row came back as `[65537, 0, 0]` on a GB10 (the exact
//! numbers depend on what the output buffer held), and `any` / `all` over a
//! multi-byte input wrote past the end of their one-byte-per-element output.
//! The Turbo-4 MLA latent codec counts boundaries with exactly that sum,
//! which is how it was found.
//!
//! Each case runs on the default device and on the CPU stream and must agree
//! exactly, values and dtype; the mask counts are also checked against a host
//! count. The shapes reach the row (simple, one and two rows per block, and
//! looped), column (small, looped and two-pass), all-reduce (single and multi
//! block) and empty-input kernels.
//!
//! Half-precision inputs are reduced natively here, which the production
//! entry points avoid on Metal (`reduce_in_f32` in the bridge: an M5 fault
//! with half-precision reductions). Those cases are left out of a Metal
//! build; the overlay under test is CUDA's.
//!
//!   cargo test -p mlxcel-core --lib reduce_output_dtype_tests

use super::*;
use crate::streams::{DefaultDeviceGuard, lock_default_device};

/// Deterministic values with zeros, positives and negatives mixed, so masks
/// and truthiness differ along every axis.
fn data(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| match (i * 7 + i / 13) % 5 {
            0 | 1 => 0.0,
            2 => 1.5,
            3 => -2.0,
            _ => 3.0,
        })
        .collect()
}

fn host(a: &MlxArray) -> (i32, Vec<i32>, Vec<f32>) {
    let as_f32 = contiguous(&astype(a, dtype::FLOAT32), false);
    eval(&as_f32);
    (
        array_dtype(a),
        array_shape(a),
        crate::utils::array_to_vec_f32(&as_f32),
    )
}

/// `op` on the default device against the same call on the CPU stream.
fn assert_matches_cpu(
    name: &str,
    op: impl Fn() -> UniquePtr<MlxArray>,
) -> (i32, Vec<i32>, Vec<f32>) {
    let device = host(&op());
    let reference = {
        let _cpu = DefaultDeviceGuard::cpu();
        host(&op())
    };
    assert_eq!(device.0, reference.0, "{name}: output dtype");
    assert_eq!(device.1, reference.1, "{name}: output shape");
    assert_eq!(device.2, reference.2, "{name}: values");
    device
}

/// `[8, 32]` and `[24, 16]` put a one-byte input on the CUDA small column
/// reduction (a short column whose stride is a multiple of 16 elements);
/// `[300, 5]` takes the two-pass column reduction, `[64, 4096]` the
/// multi-block all-reduce, and `[1025, 8]` the row reduction that handles two
/// rows per block (1024 output rows or more).
const SHAPES: [&[i32]; 9] = [
    &[3, 4],
    &[8, 32],
    &[24, 16],
    &[5, 300],
    &[300, 5],
    &[1025, 8],
    &[64, 4096],
    &[7, 33, 65],
    &[2, 3, 4, 5],
];

/// Whether to reduce half-precision arrays in their own dtype. Not on Metal,
/// where the bridge routes those reductions through f32 for an M5 fault.
fn native_half_precision_reductions() -> bool {
    !cfg!(feature = "metal")
}

#[test]
fn a_sum_or_product_over_a_mask_counts_on_every_axis() {
    let _device = lock_default_device();
    let zero = full_f32(&[1], 0.0, dtype::FLOAT32);
    for shape in SHAPES {
        let len: i32 = shape.iter().product();
        let values = data(len as usize);
        let x = from_slice_f32(&values, shape);
        let mask = greater(&x, &zero);
        assert_eq!(array_dtype(&mask), dtype::BOOL);
        let true_count = values.iter().filter(|v| **v > 0.0).count() as f32;

        let (dt, _, total) =
            assert_matches_cpu(&format!("sum(bool) all {shape:?}"), || sum_all(&mask));
        assert_eq!(dt, dtype::INT32, "{shape:?}: MLX sums a mask in int32");
        assert_eq!(total, vec![true_count], "{shape:?}: host count");

        for axis in 0..shape.len() as i32 {
            for keepdims in [false, true] {
                let (dt, _, counts) = assert_matches_cpu(
                    &format!("sum(bool) axis {axis} keepdims {keepdims} {shape:?}"),
                    || sum_axis(&mask, axis, keepdims),
                );
                assert_eq!(dt, dtype::INT32, "{shape:?} axis {axis}");
                assert_eq!(
                    counts.iter().sum::<f32>(),
                    true_count,
                    "{shape:?} axis {axis}: the per-slice counts add up to the host count"
                );
            }
            assert_matches_cpu(&format!("prod(bool) axis {axis} {shape:?}"), || {
                prod_axis(&mask, axis, false)
            });
        }
        // The negative axis spelling takes the same kernels.
        assert_matches_cpu(&format!("sum(bool) axis -1 {shape:?}"), || {
            sum_axis(&mask, -1, false)
        });
        assert_matches_cpu(&format!("prod(bool) all {shape:?}"), || prod_all(&mask));
    }
}

#[test]
fn a_sum_over_narrow_integers_matches_the_cpu() {
    let _device = lock_default_device();
    for shape in SHAPES {
        let len: i32 = shape.iter().product();
        let values: Vec<f32> = data(len as usize).iter().map(|v| v.abs().floor()).collect();
        let x = from_slice_f32(&values, shape);
        for narrow in [dtype::UINT8, dtype::UINT16, dtype::INT8, dtype::INT16] {
            let x = astype(&x, narrow);
            assert_matches_cpu(&format!("sum({narrow}) all {shape:?}"), || sum_all(&x));
            for axis in 0..shape.len() as i32 {
                assert_matches_cpu(&format!("sum({narrow}) axis {axis} {shape:?}"), || {
                    sum_axis(&x, axis, false)
                });
            }
        }
    }
}

#[test]
fn any_and_all_over_a_non_bool_input_return_bool() {
    let _device = lock_default_device();
    for shape in SHAPES {
        let len: i32 = shape.iter().product();
        let values = data(len as usize);
        let x = from_slice_f32(&values, shape);
        for input in [
            dtype::FLOAT32,
            dtype::FLOAT16,
            dtype::BFLOAT16,
            dtype::INT64,
            dtype::INT32,
            dtype::UINT8,
            dtype::BOOL,
        ] {
            let half = input == dtype::FLOAT16 || input == dtype::BFLOAT16;
            if half && !native_half_precision_reductions() {
                continue;
            }
            let x = astype(&x, input);
            let (dt, _, any) =
                assert_matches_cpu(&format!("any({input}) all {shape:?}"), || any_all(&x));
            assert_eq!(dt, dtype::BOOL, "any({input}) {shape:?}");
            assert_eq!(
                any,
                vec![1.0],
                "any({input}) {shape:?}: the data has nonzeros"
            );
            let (dt, _, all) =
                assert_matches_cpu(&format!("all({input}) all {shape:?}"), || all_all(&x));
            assert_eq!(dt, dtype::BOOL, "all({input}) {shape:?}");
            assert_eq!(all, vec![0.0], "all({input}) {shape:?}: the data has zeros");
            for axis in 0..shape.len() as i32 {
                for keepdims in [false, true] {
                    let (dt, _, _) = assert_matches_cpu(
                        &format!("any({input}) axis {axis} keepdims {keepdims} {shape:?}"),
                        || any_axis(&x, axis, keepdims),
                    );
                    assert_eq!(dt, dtype::BOOL);
                    let (dt, _, _) = assert_matches_cpu(
                        &format!("all({input}) axis {axis} keepdims {keepdims} {shape:?}"),
                        || all_axis(&x, axis, keepdims),
                    );
                    assert_eq!(dt, dtype::BOOL);
                }
            }
        }
    }
}

/// Several axes reduced at once. MLX merges adjacent reduced axes into one, so
/// `[1, 2]` or `[0, 1]` take the same kernels as a single axis; reduced axes
/// with a kept one between them (`[0, 2]`, `[1, 3]`, `[0, 2]` of four) are
/// what reaches the looped row reduction and the column reductions over more
/// than one axis. The per-axis FFI calls cannot ask for either, so this goes
/// through the probe.
///
/// The f32 comparisons are exact on purpose: the values are multiples of 0.5
/// with small sums, and every reduced slice holds a zero, so a product is 0
/// in any order.
#[test]
fn a_reduction_over_several_axes_at_once_matches_the_cpu() {
    let _device = lock_default_device();
    let zero = full_f32(&[1], 0.0, dtype::FLOAT32);
    const OPS: [(i32, &str); 4] = [(0, "sum"), (1, "prod"), (2, "any"), (3, "all")];
    let cases: [(&[i32], &[&[i32]]); 4] = [
        (&[7, 33, 65], &[&[1, 2], &[0, 1], &[0, 2]]),
        (
            &[2, 3, 4, 5],
            &[
                &[2, 3],
                &[1, 2, 3],
                &[0, 1],
                &[0, 1, 2],
                &[1, 3],
                &[0, 2],
                &[0, 3],
            ],
        ),
        (
            &[6, 5, 40, 32],
            &[&[2, 3], &[0, 1], &[1, 2, 3], &[0, 2], &[1, 3]],
        ),
        (&[40, 3, 40, 4], &[&[0, 2], &[1, 3], &[0, 3]]),
    ];
    for (shape, axis_sets) in cases {
        let len: i32 = shape.iter().product();
        let values = data(len as usize);
        let x = from_slice_f32(&values, shape);
        let mask = greater(&x, &zero);
        let narrow = astype(&mask, dtype::UINT8);
        let true_count = values.iter().filter(|v| **v > 0.0).count() as f32;
        for axes in axis_sets {
            for keepdims in [false, true] {
                for (input, name) in [(&mask, "bool"), (&narrow, "uint8"), (&x, "f32")] {
                    for (op, op_name) in OPS {
                        let label = format!("{op_name}({name}) axes {axes:?} {shape:?}");
                        let (dt, _, got) = assert_matches_cpu(&label, || {
                            reduce_axes_probe_for_test(input, op, axes, keepdims).unwrap()
                        });
                        if op >= 2 {
                            assert_eq!(dt, dtype::BOOL, "{label}");
                        }
                        if op == 0 && name != "f32" {
                            assert_ne!(dt, array_dtype(input), "{label}: MLX widens it");
                            assert_eq!(got.iter().sum::<f32>(), true_count, "{label}");
                        }
                    }
                }
            }
        }
    }
    assert!(reduce_axes_probe_for_test(&x_scalar(), 9, &[0], false).is_err());
}

fn x_scalar() -> UniquePtr<MlxArray> {
    from_slice_f32(&[1.0], &[1])
}

/// A reduction over an empty axis writes the op's identity through the same
/// output type: 0 for a mask sum, 1 for a mask product, `false` for `any`,
/// `true` for `all`.
#[test]
fn a_reduction_over_an_empty_axis_writes_the_identity() {
    let _device = lock_default_device();
    let x = zeros(&[3, 0], dtype::FLOAT32);
    let mask = astype(&x, dtype::BOOL);
    let (dt, shape, sum) = assert_matches_cpu("sum(bool) over an empty axis", || {
        sum_axis(&mask, -1, false)
    });
    assert_eq!((dt, shape, sum), (dtype::INT32, vec![3], vec![0.0; 3]));
    let (dt, _, prod) = assert_matches_cpu("prod(bool) over an empty axis", || {
        prod_axis(&mask, -1, false)
    });
    assert_eq!((dt, prod), (dtype::INT32, vec![1.0; 3]));
    let (dt, _, any) =
        assert_matches_cpu("any(f32) over an empty axis", || any_axis(&x, -1, false));
    assert_eq!((dt, any), (dtype::BOOL, vec![0.0; 3]));
    let (dt, _, all) =
        assert_matches_cpu("all(f32) over an empty axis", || all_axis(&x, -1, false));
    assert_eq!((dt, all), (dtype::BOOL, vec![1.0; 3]));
}

/// Reductions that keep their dtype are unchanged: an f32 sum and a max.
#[test]
fn same_dtype_reductions_are_unchanged() {
    let _device = lock_default_device();
    for shape in SHAPES {
        let len: i32 = shape.iter().product();
        // Small integers, so every partial sum is exact in bf16 and fp32 and
        // the comparison does not depend on the order of accumulation.
        let values: Vec<f32> = (0..len).map(|i| (i % 3) as f32).collect();
        let x = from_slice_f32(&values, shape);
        for axis in 0..shape.len() as i32 {
            let (dt, _, _) = assert_matches_cpu(&format!("sum(f32) axis {axis} {shape:?}"), || {
                sum_axis_native_dtype(&x, axis, false)
            });
            assert_eq!(dt, dtype::FLOAT32);
            let (dt, _, _) = assert_matches_cpu(&format!("max(f32) axis {axis} {shape:?}"), || {
                max_axis(&x, axis, false)
            });
            assert_eq!(dt, dtype::FLOAT32);
        }
    }
}

/// A bf16 sum is the one reduction whose accumulation type (fp32) is not its
/// output type. It still returns bf16, through the row and column kernels and
/// through the fp32 intermediates of the two-pass column reduction
/// (`[300, 5]` over axis 0) and the multi-block all-reduce (`[64, 4096]`).
///
/// The input is a sparse pattern of ones with at most 128 in any reduced
/// slice, so every partial sum is a small integer, exact in bf16 and fp32
/// alike, and the result does not depend on the order of accumulation.
#[test]
fn a_bf16_sum_still_returns_bf16() {
    if !native_half_precision_reductions() {
        return;
    }
    let _device = lock_default_device();
    for shape in SHAPES {
        let len: i32 = shape.iter().product();
        let every = (len as usize / 128 + 1).max(3);
        let values: Vec<f32> = (0..len as usize)
            .map(|i| if i % every == 0 { 1.0 } else { 0.0 })
            .collect();
        let ones = values.iter().sum::<f32>();
        assert!(ones <= 128.0, "{shape:?}: {ones} ones");
        let x = astype(&from_slice_f32(&values, shape), dtype::BFLOAT16);
        for axis in 0..shape.len() as i32 {
            let (dt, _, sums) =
                assert_matches_cpu(&format!("sum(bf16) axis {axis} {shape:?}"), || {
                    sum_axis_native_dtype(&x, axis, false)
                });
            assert_eq!(dt, dtype::BFLOAT16, "{shape:?} axis {axis}");
            assert_eq!(sums.iter().sum::<f32>(), ones, "{shape:?} axis {axis}");
        }
        let all_axes: Vec<i32> = (0..shape.len() as i32).collect();
        let (dt, _, total) = assert_matches_cpu(&format!("sum(bf16) all {shape:?}"), || {
            reduce_axes_probe_for_test(&x, 0, &all_axes, false).unwrap()
        });
        assert_eq!((dt, total), (dtype::BFLOAT16, vec![ones]), "{shape:?}");
    }
}
