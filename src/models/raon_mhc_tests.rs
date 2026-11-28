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

fn raw(a: &MlxArray) -> Vec<u8> {
    mlxcel_core::eval(a);
    mlxcel_core::array_to_raw_bytes(a)
}

/// The compiled site graphs against the eager ops on one site: bit for bit
/// on Metal and the CPU; on CUDA and ROCm the fused kernel may contract a
/// multiply and an add into one FMA (issue #3474), so the two may differ by
/// rounding there.
/// Decode exactness does not rest on this: a verify block runs the compiled
/// graphs too (see the next test).
fn assert_same_or_close(got: &MlxArray, want: &MlxArray, what: &str) {
    if !cfg!(any(feature = "cuda", feature = "rocm")) {
        assert_eq!(raw(got), raw(want), "{what}");
        return;
    }
    let f32s = |a: &MlxArray| -> Vec<f32> {
        raw(&mlxcel_core::astype(a, dtype::FLOAT32))
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    // About 8 float32 ulps of the largest value, or one bfloat16 ulp of it.
    let rel = if mlxcel_core::array_dtype(want) == dtype::FLOAT32 {
        1e-6
    } else {
        1.0 / 128.0
    };
    let (got, want) = (f32s(got), f32s(want));
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let worst = got
        .iter()
        .zip(&want)
        .fold(0f32, |m, (g, w)| m.max((g - w).abs()));
    assert!(
        worst <= rel * scale,
        "{what}: compiled and eager differ by {worst:e} (largest value {scale:e})"
    );
}

/// The compiled decode graphs (issue #3458) replay the eager ops: the branch
/// input and the write-back of a single-position step, in bf16 and f32, at
/// one and several rows, at the real stream width.
#[test]
fn compiled_site_equals_the_eager_site_bit_for_bit() {
    let _guard = guard();
    let n = 4usize;
    for (d, seeds) in [(64usize, 0..6u32), (2560, 0..2)] {
        let w = n * d;
        for seed in seeds {
            let weights = site_weights(
                n,
                d,
                noise(n * w, seed + 11),
                noise(n * w, seed + 12),
                noise(9 * w, seed + 13).iter().map(|v| v * 40.0).collect(),
                [1.5, 0.7, 3.0],
                noise(n, seed + 14),
                noise(n, seed + 15),
                noise(9, seed + 16).iter().map(|v| v * 8.0).collect(),
                0.3 * seed as f32 - 0.5,
            );
            for dt in [dtype::BFLOAT16, dtype::FLOAT32] {
                let site =
                    RaonMhc::from_weights(&weights, "site", n, d, 1e-5, 2.0, dt).expect("loads");
                for b in [1i32, 3] {
                    let what = format!("d {d} seed {seed} dtype {dt} rows {b}");
                    let streams = mlxcel_core::astype(
                        &arr(
                            &noise(b as usize * w, seed + 19)
                                .iter()
                                .map(|v| v * 6.0)
                                .collect::<Vec<_>>(),
                            &[b, 1, n as i32, d as i32],
                        ),
                        dt,
                    );
                    let branch = mlxcel_core::astype(
                        &arr(&noise(b as usize * d, seed + 23), &[b, 1, d as i32]),
                        dt,
                    );
                    let (x_eager, eager) = site.enter(&streams, false);
                    let (x_compiled, compiled) = site.enter(&streams, true);
                    assert!(eager.streams_f32.is_some(), "{what}: eager path");
                    assert!(compiled.streams_f32.is_none(), "{what}: compiled path");
                    assert_eq!(mlxcel_core::array_dtype(&x_compiled), dt, "{what}");
                    assert_same_or_close(&x_compiled, &x_eager, &format!("{what}: branch input"));
                    assert_same_or_close(
                        &compiled.h_post,
                        &eager.h_post,
                        &format!("{what}: h_post"),
                    );
                    assert_same_or_close(&compiled.h_res, &eager.h_res, &format!("{what}: h_res"));
                    let out_eager = site.leave(&streams, &branch, &eager);
                    let out_compiled = site.leave(&streams, &branch, &compiled);
                    assert_eq!(mlxcel_core::array_dtype(&out_compiled), dt, "{what}");
                    assert_same_or_close(&out_compiled, &out_eager, &format!("{what}: write-back"));
                }
            }
        }
    }
}

/// With the compiled site on, a single-position step and a speculative
/// verify block (one compiled step per position) take the compiled graphs;
/// a prefill of several positions stays on the eager ops.
#[test]
fn compiled_site_covers_decode_steps_and_verify_blocks_only() {
    let _guard = guard();
    let (n, d) = (4usize, 8usize);
    let w = n * d;
    let weights = site_weights(
        n,
        d,
        noise(n * w, 31),
        noise(n * w, 32),
        noise(9 * w, 33),
        [1.5, 0.7, 3.0],
        noise(n, 34),
        noise(n, 35),
        noise(9, 36),
        0.2,
    );
    let site = RaonMhc::from_weights(&weights, "site", n, d, 1e-5, 2.0, dtype::FLOAT32).unwrap();
    let block = arr(&noise(3 * w, 37), &[1, 3, n as i32, d as i32]);
    assert!(site.enter(&block, true).1.streams_f32.is_some());
    let step = arr(&noise(w, 38), &[1, 1, n as i32, d as i32]);
    assert!(site.enter(&step, true).1.streams_f32.is_none());
    let _scope = mlxcel_core::row_invariant::Scope::enter(mlxcel_core::row_invariant::Mode {
        kernel: false,
        verify_block: true,
    });
    assert!(site.enter(&step, true).1.streams_f32.is_none());
    assert!(site.enter(&block, true).1.streams_f32.is_none());
    assert!(site.enter(&block, false).1.streams_f32.is_some());
}

/// Inside a verify block each position of the compiled site equals the
/// decode step of that position alone, bit for bit, on every backend
/// (issue #3474): the branch input, the coefficients and the write-back.
#[test]
fn a_compiled_verify_block_equals_its_decode_steps_bit_for_bit() {
    let _guard = guard();
    let (n, d, l) = (4usize, 64usize, 5i32);
    let w = n * d;
    let weights = site_weights(
        n,
        d,
        noise(n * w, 41),
        noise(n * w, 42),
        noise(9 * w, 43).iter().map(|v| v * 40.0).collect(),
        [1.5, 0.7, 3.0],
        noise(n, 44),
        noise(n, 45),
        noise(9, 46).iter().map(|v| v * 8.0).collect(),
        0.1,
    );
    for dt in [dtype::BFLOAT16, dtype::FLOAT32] {
        let site = RaonMhc::from_weights(&weights, "site", n, d, 1e-5, 2.0, dt).unwrap();
        let streams = mlxcel_core::astype(
            &arr(
                &noise(l as usize * w, 47)
                    .iter()
                    .map(|v| v * 6.0)
                    .collect::<Vec<_>>(),
                &[1, l, n as i32, d as i32],
            ),
            dt,
        );
        let branch = mlxcel_core::astype(&arr(&noise(l as usize * d, 48), &[1, l, d as i32]), dt);
        let (x, out, h_res) = {
            let _scope =
                mlxcel_core::row_invariant::Scope::enter(mlxcel_core::row_invariant::Mode {
                    kernel: false,
                    verify_block: true,
                });
            let (x, block) = site.enter(&streams, true);
            let out = site.leave(&streams, &branch, &block);
            (x, out, block.h_res)
        };
        for t in 0..l {
            let at = |a: &MlxArray| mlxcel_core::utils::slice_axis(a, 1, t, t + 1);
            let (step_streams, step_branch) = (at(&streams), at(&branch));
            let (step_x, step) = site.enter(&step_streams, true);
            let step_out = site.leave(&step_streams, &step_branch, &step);
            let what = format!("dtype {dt} position {t}");
            assert_eq!(raw(&at(&x)), raw(&step_x), "{what}: branch input");
            assert_eq!(raw(&at(&h_res)), raw(&step.h_res), "{what}: h_res");
            assert_eq!(raw(&at(&out)), raw(&step_out), "{what}: write-back");
        }
    }
}

#[test]
fn decode_fusion_setting_parses_its_values() {
    use super::DecodeFusion;
    assert_eq!(DecodeFusion::parse(None), DecodeFusion::ALL);
    assert_eq!(DecodeFusion::parse(Some("all")), DecodeFusion::ALL);
    assert_eq!(DecodeFusion::parse(Some(" OFF ")), DecodeFusion::OFF);
    let mhc = DecodeFusion::parse(Some("mhc"));
    assert!(mhc.mhc && !mhc.selector);
    let selector = DecodeFusion::parse(Some("selector"));
    assert!(selector.selector && !selector.mhc);
    assert_eq!(
        DecodeFusion::parse(Some("selector, mhc")),
        DecodeFusion::ALL
    );
    // A name that is no fast path keeps the default instead of turning one off.
    assert_eq!(DecodeFusion::parse(Some("mhc,typo")), DecodeFusion::ALL);
    // Malformed values (a misspelled `off`, a trailing comma, a bogus entry)
    // warn and fall back to every fast path, they do not turn one off.
    for value in ["of", "mhc,", "selector,bogus", "", ",", "off,mhc"] {
        assert_eq!(
            DecodeFusion::parse(Some(value)),
            DecodeFusion::ALL,
            "{value:?}"
        );
    }
}

/// The one-row selector product (issue #3458) at the real width in bf16: a
/// decode step's coefficients equal the same row's inside a verify block,
/// bit for bit, in the row-major form and in the reference form, and a
/// product of several rows outside a verify block keeps the plain matmul
/// whatever the form of the one-row product.
#[test]
fn one_row_selector_products_agree_between_a_decode_step_and_a_verify_row() {
    let _guard = guard();
    let (n, d) = (4usize, 2560usize);
    let w = n * d;
    let weights = site_weights(
        n,
        d,
        noise(n * w, 41),
        noise(n * w, 42),
        noise(9 * w, 43).iter().map(|v| v * 40.0).collect(),
        [1.5, 0.7, 3.0],
        noise(n, 44),
        noise(n, 45),
        noise(9, 46).iter().map(|v| v * 8.0).collect(),
        0.1,
    );
    let dt = dtype::BFLOAT16;
    let block = mlxcel_core::astype(
        &arr(
            &noise(4 * w, 47).iter().map(|v| v * 6.0).collect::<Vec<_>>(),
            &[1, 4, n as i32, d as i32],
        ),
        dt,
    );
    let row = |streams: &MlxArray, t: i32| mlxcel_core::utils::slice_axis(streams, 1, t, t + 1);
    let coefficients =
        |route: &MhcRoute| (raw(&route.h_pre), raw(&route.h_post), raw(&route.h_res));
    let mut plain_blocks = Vec::new();
    for row_major in [true, false] {
        let mut site = RaonMhc::from_weights(&weights, "site", n, d, 1e-5, 2.0, dt).unwrap();
        site.set_row_major_single(row_major);
        // Outside a verify block the four rows are one matmul.
        plain_blocks.push(coefficients(&site.route(&block)));
        let steps: Vec<_> = (0..4)
            .map(|t| coefficients(&site.route(&row(&block, t))))
            .collect();
        let _scope = mlxcel_core::row_invariant::Scope::enter(mlxcel_core::row_invariant::Mode {
            kernel: false,
            verify_block: true,
        });
        let verify = site.route(&block);
        for (t, step) in steps.iter().enumerate() {
            let t = t as i32;
            let verify_row = (
                raw(&row(&verify.h_pre, t)),
                raw(&row(&verify.h_post, t)),
                raw(&row(&verify.h_res, t)),
            );
            assert_eq!(&verify_row, step, "row-major {row_major}: verify row {t}");
        }
    }
    assert_eq!(
        plain_blocks[0], plain_blocks[1],
        "a multi-row product is unchanged"
    );
}
