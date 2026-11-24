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

//! Row-invariant dense products (issue #3280): every row of an M-row kernel
//! product equals that row's one-row product byte for byte, on both the
//! narrow (split) and the wide variant, the values are a dot product, and
//! the scope routes [`dense`] as documented.

use super::*;
use crate::dtype::{BFLOAT16, FLOAT32};
use crate::utils::array_to_vec_f32;

fn values(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn array(rows: i32, cols: i32, seed: u64, dt: i32) -> UniquePtr<MlxArray> {
    let host = values((rows * cols) as usize, seed);
    crate::astype(&crate::from_slice_f32(&host, &[rows, cols]), dt)
}

fn bytes(x: &MlxArray) -> Vec<u8> {
    crate::array_to_raw_bytes(x)
}

fn check(n: i32, k: i32, dt: i32) {
    let w = array(n, k, 11, dt);
    let x = array(MAX_ROWS, k, 5, dt);
    let wf = array_to_vec_f32(&crate::astype(&w, FLOAT32));
    let xf = array_to_vec_f32(&crate::astype(&x, FLOAT32));
    for rows in [2, 5, 8, MAX_ROWS] {
        let block = crate::utils::slice_axis(&x, 0, 0, rows);
        let full = matvec(&block, &w, dt).expect("kernel runs on Metal");
        for r in 0..rows {
            let one = matvec(&crate::utils::slice_axis(&x, 0, r, r + 1), &w, dt).unwrap();
            let row = crate::utils::slice_axis(&full, 0, r, r + 1);
            assert_eq!(bytes(&row), bytes(&one), "n {n} k {k} rows {rows} row {r}");
        }
        let got = array_to_vec_f32(&crate::astype(&full, FLOAT32));
        let tol = if dt == FLOAT32 { 1e-4 } else { 2e-2 };
        for r in 0..rows as usize {
            for c in 0..n as usize {
                let want: f64 = (0..k as usize)
                    .map(|i| f64::from(xf[r * k as usize + i]) * f64::from(wf[c * k as usize + i]))
                    .sum();
                let got = f64::from(got[r * n as usize + c]);
                assert!(
                    (got - want).abs() <= tol * (1.0 + want.abs()),
                    "n {n} k {k} row {r} col {c}: {got} vs {want}"
                );
            }
        }
    }
}

#[test]
fn every_row_matches_its_one_row_product() {
    if !available() {
        eprintln!("skipping: no Metal kernels on this backend");
        return;
    }
    // Narrow (split K) variant, including K below one 128-wide stride.
    check(24, 64, FLOAT32);
    check(130, 520, BFLOAT16);
    // Wide variant.
    check(600, 256, FLOAT32);
    check(1030, 132, BFLOAT16);
}

#[test]
fn the_contract_refuses_unsupported_shapes() {
    if !available() {
        return;
    }
    let w = array(8, 64, 1, FLOAT32);
    assert!(
        matvec(&array(17, 64, 2, FLOAT32), &w, FLOAT32).is_none(),
        "17 rows"
    );
    assert!(
        matvec(
            &array(1, 62, 2, FLOAT32),
            &array(8, 62, 1, FLOAT32),
            FLOAT32
        )
        .is_none(),
        "K % 4"
    );
    assert!(
        matvec(&array(1, 32, 2, FLOAT32), &w, FLOAT32).is_none(),
        "K mismatch"
    );
}

#[test]
fn dense_is_plain_matmul_outside_a_scope_and_row_invariant_inside() {
    let w = array(40, 64, 3, BFLOAT16);
    let x = array(6, 64, 4, BFLOAT16);
    let plain = crate::matmul(&x, &crate::transpose(&w));
    assert_eq!(bytes(&dense(&x, &w)), bytes(&plain), "no scope: matmul");
    for kernel in [false, true] {
        let row = |r: i32| {
            let _scope = Scope::enter(Mode {
                kernel,
                verify_block: false,
            });
            dense(&crate::utils::slice_axis(&x, 0, r, r + 1), &w)
        };
        let block = {
            let _scope = Scope::enter(Mode {
                kernel,
                verify_block: true,
            });
            dense(&x, &w)
        };
        for r in 0..6 {
            let got = crate::utils::slice_axis(&block, 0, r, r + 1);
            assert_eq!(bytes(&got), bytes(&row(r)), "kernel {kernel} row {r}");
        }
    }
    assert!(mode().is_none(), "scopes restore the previous mode");
}

#[test]
fn per_position_splits_only_inside_a_verify_block() {
    let x = array(4, 8, 9, FLOAT32);
    let x = crate::reshape(&x, &[1, 4, 8]);
    let calls = std::cell::Cell::new(0);
    let f = |x: &MlxArray| {
        calls.set(calls.get() + 1);
        crate::copy(x)
    };
    per_position(&x, f);
    assert_eq!(calls.get(), 1);
    let _scope = Scope::enter(Mode {
        kernel: false,
        verify_block: true,
    });
    let y = per_position(&x, f);
    assert_eq!(calls.get(), 5);
    assert_eq!(bytes(&y), bytes(&x));
}
