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

//! Raon_mHC against closed forms and hand-computed cases.

use super::{MhcRoute, RaonMhc, aggregate, helmert_basis, mix, reduce_streams};
use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

fn guard() -> crate::models::embedding_test_support::MlxTestGuard {
    crate::models::embedding_test_support::mlx_test_guard()
}

fn arr(data: &[f32], shape: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_f32(data, shape)
}

fn values(a: &MlxArray) -> Vec<f32> {
    let a = mlxcel_core::astype(a, dtype::FLOAT32);
    mlxcel_core::eval(&a);
    array_to_vec_f32(&a)
}

fn noise(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(7);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / (1u32 << 24) as f32) - 0.5
        })
        .collect()
}

/// One site's tensors under `site`, with the given scalars and planes.
#[allow(clippy::too_many_arguments)]
fn site_weights(
    n: usize,
    d: usize,
    phi_pre: Vec<f32>,
    phi_post: Vec<f32>,
    phi_z: Vec<f32>,
    alphas: [f32; 3],
    b_pre: Vec<f32>,
    b_post: Vec<f32>,
    b_z: Vec<f32>,
    eta: f32,
) -> WeightMap {
    let m = (n - 1) * (n - 1);
    let w = (n * d) as i32;
    let mut weights = WeightMap::new();
    weights.insert("site.phi_pre".into(), arr(&phi_pre, &[n as i32, w]));
    weights.insert("site.phi_post".into(), arr(&phi_post, &[n as i32, w]));
    weights.insert("site.res.phi_z".into(), arr(&phi_z, &[m as i32, w]));
    weights.insert("site.alpha_pre".into(), arr(&[alphas[0]], &[]));
    weights.insert("site.alpha_post".into(), arr(&[alphas[1]], &[]));
    weights.insert("site.res.alpha_z".into(), arr(&[alphas[2]], &[]));
    weights.insert("site.b_pre".into(), arr(&b_pre, &[n as i32]));
    weights.insert("site.b_post".into(), arr(&b_post, &[n as i32]));
    weights.insert("site.res.b_z".into(), arr(&b_z, &[m as i32]));
    weights.insert("site.res.eta".into(), arr(&[eta], &[]));
    weights
}

#[test]
fn helmert_basis_matches_the_closed_form_for_four_streams() {
    let q = helmert_basis(4);
    let s = |x: f64| x.sqrt();
    #[rustfmt::skip]
    let expected: [f64; 12] = [
        1.0 / s(2.0),  1.0 / s(6.0),  1.0 / s(12.0),
        -1.0 / s(2.0), 1.0 / s(6.0),  1.0 / s(12.0),
        0.0,           -2.0 / s(6.0), 1.0 / s(12.0),
        0.0,           0.0,           -3.0 / s(12.0),
    ];
    for (got, want) in q.iter().zip(expected) {
        assert!((*got as f64 - want).abs() < 1e-7, "{q:?}");
    }
    // Q^T Q = I and Q^T 1 = 0.
    for a in 0..3 {
        let col_sum: f64 = (0..4).map(|r| q[r * 3 + a] as f64).sum();
        assert!(col_sum.abs() < 1e-7);
        for b in 0..3 {
            let dot: f64 = (0..4)
                .map(|r| q[r * 3 + a] as f64 * q[r * 3 + b] as f64)
                .sum();
            assert!((dot - if a == b { 1.0 } else { 0.0 }).abs() < 1e-6);
        }
    }
}

#[test]
fn residual_is_doubly_stochastic_and_non_negative_for_random_z() {
    let _guard = guard();
    let (n, d, b, l) = (4usize, 8usize, 2i32, 3i32);
    let w = n * d;
    for seed in 0..4u32 {
        let weights = site_weights(
            n,
            d,
            noise(n * w, seed + 1),
            noise(n * w, seed + 2),
            noise(9 * w, seed + 3).iter().map(|v| v * 40.0).collect(),
            [1.5, 0.7, 3.0],
            noise(n, seed + 4),
            noise(n, seed + 5),
            noise(9, seed + 6).iter().map(|v| v * 8.0).collect(),
            0.3 * seed as f32 - 0.5,
        );
        let site = RaonMhc::from_weights(&weights, "site", n, d, 1e-5, 2.0, dtype::FLOAT32)
            .expect("site loads");
        let streams = arr(
            &noise((b * l) as usize * w, seed + 9),
            &[b, l, n as i32, d as i32],
        );
        let route = site.route(&streams);
        let h_res = values(&route.h_res);
        for token in 0..(b * l) as usize {
            let m = &h_res[token * 16..(token + 1) * 16];
            for i in 0..4 {
                let row: f32 = (0..4).map(|j| m[i * 4 + j]).sum();
                let col: f32 = (0..4).map(|j| m[j * 4 + i]).sum();
                assert!((row - 1.0).abs() < 1e-5, "row {i} sums to {row}: {m:?}");
                assert!((col - 1.0).abs() < 1e-5, "col {i} sums to {col}: {m:?}");
            }
            assert!(m.iter().all(|&v| v >= -1e-6), "negative entry: {m:?}");
        }
        assert!(
            values(&route.h_pre)
                .iter()
                .all(|&v| (0.0..=1.0).contains(&v))
        );
        assert!(
            values(&route.h_post)
                .iter()
                .all(|&v| (0.0..=2.0).contains(&v))
        );
    }
}

/// n = 2, D = 1, streams (1, 1): every coefficient worked by hand in f64.
#[test]
fn two_stream_route_matches_a_hand_computation() {
    let _guard = guard();
    let weights = site_weights(
        2,
        1,
        vec![1.0, 0.0, 0.0, 1.0],
        vec![0.5, 0.5, -1.0, 0.0],
        vec![1.0, 0.0],
        [2.0, 1.0, 2.0],
        vec![0.0, 0.5],
        vec![0.25, 0.0],
        vec![0.5],
        0.0,
    );
    let eps = 1e-5f64;
    let site =
        RaonMhc::from_weights(&weights, "site", 2, 1, eps as f32, 2.0, dtype::FLOAT32).unwrap();
    let route = site.route(&arr(&[1.0, 1.0], &[1, 1, 2, 1]));
    let inv = 1.0 / (1.0 + eps).sqrt();
    let sig = |x: f64| 1.0 / (1.0 + (-x).exp());
    let pre = [sig(2.0 * inv), sig(2.0 * inv + 0.5)];
    let post = [2.0 * sig(inv + 0.25), 2.0 * sig(-inv)];
    let z = 2.0 * inv + 0.5;
    // Q = [1, -1] / sqrt 2, so E = z [[.5, -.5], [-.5, .5]]; B = [[.75, .25], [.25, .75]].
    let e = [0.5 * z, -0.5 * z, -0.5 * z, 0.5 * z];
    let bm = [0.75, 0.25, 0.25, 0.75];
    let rho = (0..4).map(|i| -e[i] / bm[i]).fold(0.0f64, f64::max);
    let gamma = 0.5;
    let res: Vec<f64> = (0..4)
        .map(|i| {
            let eye = if i == 0 || i == 3 { 1.0 } else { 0.0 };
            (1.0 - gamma) * eye + gamma * (bm[i] + e[i] / (1.0 + rho))
        })
        .collect();
    let close = |got: Vec<f32>, want: &[f64]| {
        for (g, w) in got.iter().zip(want) {
            assert!((*g as f64 - w).abs() < 1e-5, "got {got:?}, want {want:?}");
        }
    };
    close(values(&route.h_pre), &pre);
    close(values(&route.h_post), &post);
    close(values(&route.h_res), &res);
}

#[test]
fn aggregate_and_mix_match_a_hand_computed_two_stream_case() {
    let _guard = guard();
    // streams s0 = [1, 2], s1 = [3, 4].
    let streams = arr(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let route = MhcRoute {
        h_pre: arr(&[0.25, 0.5], &[1, 1, 2]),
        h_post: arr(&[1.0, 0.5], &[1, 1, 2]),
        h_res: arr(&[0.75, 0.25, 0.25, 0.75], &[1, 1, 2, 2]),
    };
    let x = aggregate(&streams, &route.h_pre, dtype::FLOAT32);
    assert_eq!(values(&x), vec![1.75, 2.5]);
    let branch = arr(&[10.0, 20.0], &[1, 1, 2]);
    let out = mix(&streams, &branch, &route, dtype::FLOAT32);
    assert_eq!(mlxcel_core::array_shape(&out), vec![1, 1, 2, 2]);
    assert_eq!(values(&out), vec![11.5, 22.5, 7.5, 13.5]);
    assert_eq!(
        values(&reduce_streams(&out, dtype::FLOAT32)),
        vec![9.5, 18.0]
    );
}
