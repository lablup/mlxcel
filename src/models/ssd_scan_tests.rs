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

//! Tests for the shared chunked SSD scan (#3073).

use super::{SsdEval, chunked_ssd_scan};
use mlxcel_core::{MlxArray, UniquePtr};

const BATCH: i32 = 2;
const SEQ_LEN: i32 = 11;
const HEADS: i32 = 4;
const HEAD_DIM: i32 = 3;
const GROUPS: i32 = 2;
const STATE_DIM: i32 = 5;

struct Inputs {
    x: UniquePtr<MlxArray>,
    b: UniquePtr<MlxArray>,
    c: UniquePtr<MlxArray>,
    dt: UniquePtr<MlxArray>,
    a_log: UniquePtr<MlxArray>,
    d: UniquePtr<MlxArray>,
    dt_bias: UniquePtr<MlxArray>,
    state: UniquePtr<MlxArray>,
}

fn wave(shape: &[i32], seed: f32) -> UniquePtr<MlxArray> {
    let len: usize = shape.iter().map(|d| *d as usize).product();
    let data: Vec<f32> = (0..len)
        .map(|i| 0.3 * ((i as f32) * 0.37 + seed).sin())
        .collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

fn inputs(dtype: i32) -> Inputs {
    let cast = |a: UniquePtr<MlxArray>| mlxcel_core::astype(&a, dtype);
    Inputs {
        x: cast(wave(&[BATCH, SEQ_LEN, HEADS, HEAD_DIM], 0.1)),
        b: cast(wave(&[BATCH, SEQ_LEN, GROUPS, STATE_DIM], 0.7)),
        c: cast(wave(&[BATCH, SEQ_LEN, GROUPS, STATE_DIM], 1.3)),
        dt: cast(wave(&[BATCH, SEQ_LEN, HEADS], 2.1)),
        a_log: cast(wave(&[HEADS], 0.4)),
        d: cast(wave(&[HEADS], 2.9)),
        dt_bias: cast(wave(&[HEADS], 3.3)),
        state: wave(&[BATCH, HEADS, HEAD_DIM, STATE_DIM], 4.2),
    }
}

fn scan(
    inp: &Inputs,
    with_state: bool,
    chunk: usize,
    policy: SsdEval,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    chunked_ssd_scan(
        &inp.x,
        &inp.a_log,
        &inp.b,
        &inp.c,
        &inp.d,
        &inp.dt,
        &inp.dt_bias,
        with_state.then_some(inp.state.as_ref().unwrap()),
        (0.0, f32::INFINITY),
        chunk,
        policy,
    )
}

fn close(a: &MlxArray, b: &MlxArray, tol: f64) -> bool {
    let a = mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32);
    let b = mlxcel_core::astype(b, mlxcel_core::dtype::FLOAT32);
    let ok = mlxcel_core::allclose(&a, &b, tol, tol);
    mlxcel_core::eval(&ok);
    mlxcel_core::item_bool(&ok)
}

fn identical(a: &MlxArray, b: &MlxArray) -> bool {
    let ok = mlxcel_core::array_equal(a, b, false);
    mlxcel_core::eval(&ok);
    mlxcel_core::item_bool(&ok)
}

/// A chunk shorter than the sequence (with a partial tail) gives the
/// unchunked result, from an empty state and from a prior state, for every
/// evaluation policy, in float32 and float16.
#[test]
fn chunked_scan_matches_unchunked_scan() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    for (dtype, tol) in [
        (mlxcel_core::dtype::FLOAT32, 1e-5),
        (mlxcel_core::dtype::FLOAT16, 2e-3),
    ] {
        let inp = inputs(dtype);
        for with_state in [false, true] {
            let (full_y, full_state) =
                scan(&inp, with_state, SEQ_LEN as usize, SsdEval::EveryChunk);
            assert_eq!(
                mlxcel_core::array_shape(&full_y),
                vec![BATCH, SEQ_LEN, HEADS, HEAD_DIM]
            );
            assert_eq!(mlxcel_core::array_dtype(&full_y), dtype);
            for (chunk, policy) in [
                (3, SsdEval::EveryChunk),
                (3, SsdEval::ScheduleEvery(1)),
                (4, SsdEval::ScheduleEvery(5)),
                (2, SsdEval::ScheduleEvery(usize::MAX)),
            ] {
                let (y, state) = scan(&inp, with_state, chunk, policy);
                let what = format!("dtype {dtype} state {with_state} chunk {chunk} {policy:?}");
                assert!(close(&full_y, &y, tol), "{what}: output differs");
                assert!(
                    close(&full_state, &state, tol),
                    "{what}: final state differs"
                );
            }
        }
    }
}

/// The evaluation policy only decides when the graph is evaluated, never what
/// it computes: every policy gives bit-identical output and state at one
/// chunk length.
#[test]
fn evaluation_policy_does_not_change_the_result() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let inp = inputs(mlxcel_core::dtype::FLOAT16);
    let (ref_y, ref_state) = scan(&inp, true, 3, SsdEval::EveryChunk);
    for policy in [
        SsdEval::ScheduleEvery(1),
        SsdEval::ScheduleEvery(4),
        SsdEval::ScheduleEvery(usize::MAX),
    ] {
        let (y, state) = scan(&inp, true, 3, policy);
        assert!(identical(&ref_y, &y), "{policy:?}: output not identical");
        assert!(
            identical(&ref_state, &state),
            "{policy:?}: state not identical"
        );
    }
}
