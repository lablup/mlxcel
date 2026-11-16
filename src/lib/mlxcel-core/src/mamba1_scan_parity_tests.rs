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

//! Parity tests for the fused Mamba1 selective-scan kernel (#2005).
//!
//! The kernel walks the recurrence
//! `s_t = exp(dt_t * A) * s_{t-1} + dt_t * x_t * B_t`,
//! `y_t = s_t . C_t + D * x_t` with the state in float32. Two properties are
//! pinned here against a scalar CPU reference:
//!
//! 1. With f32 inputs the kernel reproduces the reference (the only difference
//!    is summation order inside `simd_sum`), for fresh and carried state, a
//!    single step and several, three state widths (up to a full 32 lanes),
//!    and a channel count that is not a multiple of the threadgroup's eight
//!    rows.
//! 2. With bf16 inputs the float32-state kernel (Metal, and ROCm since #2069)
//!    is at least as close to the f32 reference as the graph scan it
//!    replaces, which rounds the state to bf16 every step.
//!
//! The CUDA port (#1981) instead rounds every step exactly as the graph scan
//! does, so on CUDA the kernel is compared with MLX's own graph scan and must
//! match it bit for bit, in f32, f16 and bf16, and for f16 activations with
//! the float32 `A` that Mamba and Jamba keep for f16 (#3125).
//!
//! The tests return early wherever `mamba1_scan_kernel_available()` is false
//! (CPU-only builds). On ROCm the graph-exact CUDA tests skip: ROCm runs the
//! float32-state variant, because its graph scan's `state @ C` (K = N < 32)
//! goes to rocBLAS, whose reduction order no custom kernel reproduces.

use crate::hardware::{GpuBackendKind, gpu_backend_kind};
use crate::utils::slice_axis;
use crate::{MlxArray, UniquePtr, ffi};

fn seeded(len: usize, seed: u64, scale: f32, offset: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((state >> 40) as f32) / ((1u32 << 24) as f32) * 2.0 - 1.0) * scale + offset
        })
        .collect()
}

fn to_vec(arr: &MlxArray) -> Vec<f32> {
    let a = ffi::astype(arr, crate::dtype::FLOAT32);
    ffi::eval(&a);
    ffi::array_to_raw_bytes(&a)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bf16_round(v: f32) -> f32 {
    // Round to nearest even at bf16 precision (keep the top 16 bits).
    let bits = v.to_bits();
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1)) & 0xFFFF_0000;
    f32::from_bits(rounded)
}

struct Case {
    x: Vec<f32>,
    dt: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
    a: Vec<f32>,
    d: Vec<f32>,
    s0: Vec<f32>,
}

fn make_case(batch: usize, seq: usize, dm: usize, n: usize, carried: bool) -> Case {
    Case {
        x: seeded(batch * seq * dm, 1, 1.0, 0.0),
        // dt is a softplus output: positive, mostly below one.
        dt: seeded(batch * seq * dm, 2, 0.25, 0.3),
        b: seeded(batch * seq * n, 3, 1.0, 0.0),
        c: seeded(batch * seq * n, 4, 1.0, 0.0),
        // A = -exp(A_log): negative.
        a: seeded(dm * n, 5, 0.5, -1.0),
        d: seeded(dm, 6, 1.0, 0.0),
        s0: if carried {
            seeded(batch * dm * n, 7, 0.5, 0.0)
        } else {
            vec![0.0; batch * dm * n]
        },
    }
}

/// Scalar reference. `round_state` emulates the graph scan, which keeps every
/// intermediate and the carried state in the activation dtype.
fn reference(
    k: &Case,
    batch: usize,
    seq: usize,
    dm: usize,
    n: usize,
    round_state: Option<fn(f32) -> f32>,
) -> (Vec<f32>, Vec<f32>) {
    let r = |v: f32| round_state.map_or(v, |f| f(v));
    let mut y = vec![0.0f32; batch * seq * dm];
    let mut s = k.s0.clone();
    for bi in 0..batch {
        for t in 0..seq {
            let row = bi * seq + t;
            for di in 0..dm {
                let dt = k.dt[row * dm + di];
                let xv = k.x[row * dm + di];
                let mut acc = 0.0f32;
                for ni in 0..n {
                    let idx = (bi * dm + di) * n + ni;
                    let decay = r((dt * k.a[di * n + ni]).exp());
                    let input = r(r(dt * xv) * k.b[row * n + ni]);
                    s[idx] = r(r(decay * s[idx]) + input);
                    acc += s[idx] * k.c[row * n + ni];
                }
                y[row * dm + di] = r(acc) + r(xv * k.d[di]);
            }
        }
    }
    (y, s)
}

fn run_kernel(
    k: &Case,
    batch: usize,
    seq: usize,
    dm: usize,
    n: usize,
    dtype: i32,
    a_dtype: i32,
) -> (Vec<f32>, Vec<f32>) {
    let arr = |v: &[f32], shape: &[i32]| ffi::astype(&ffi::from_slice_f32(v, shape), dtype);
    let (b, l, d, n_) = (batch as i32, seq as i32, dm as i32, n as i32);
    let x = arr(&k.x, &[b, l, d]);
    let dt = arr(&k.dt, &[b, l, d]);
    let bm = arr(&k.b, &[b, l, n_]);
    let cm = arr(&k.c, &[b, l, n_]);
    let a = ffi::astype(&ffi::from_slice_f32(&k.a, &[d, n_]), a_dtype);
    let dp = arr(&k.d, &[d]);
    let s0 = ffi::from_slice_f32(&k.s0, &[b, d, n_]);
    let mut y: UniquePtr<MlxArray> = UniquePtr::null();
    let mut s: UniquePtr<MlxArray> = UniquePtr::null();
    ffi::mamba1_selective_scan(&x, &dt, &bm, &cm, &a, &dp, &s0, &mut y, &mut s)
        .expect("the caller checked the port, so the launcher must not refuse");
    assert_eq!(ffi::array_shape(&y), vec![b, l, d]);
    assert_eq!(ffi::array_shape(&s), vec![b, d, n_]);
    assert_eq!(ffi::array_dtype(&y), dtype);
    (to_vec(&y), to_vec(&s))
}

fn max_rel(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
    got.iter()
        .zip(want)
        .fold(0.0f32, |m, (g, w)| m.max((g - w).abs()))
        / scale
}

#[test]
fn f32_kernel_matches_scalar_reference() {
    if !ffi::mamba1_scan_kernel_available() {
        return;
    }
    let (batch, dm) = (2, 24);
    // 32 fills a whole warp or wavefront, the widest state the kernel accepts:
    // only there does every step of the 16..1 fold carry data.
    for n in [8, 16, 32] {
        for seq in [1, 7] {
            for carried in [false, true] {
                let k = make_case(batch, seq, dm, n, carried);
                let (want_y, want_s) = reference(&k, batch, seq, dm, n, None);
                let (y, s) = run_kernel(
                    &k,
                    batch,
                    seq,
                    dm,
                    n,
                    crate::dtype::FLOAT32,
                    crate::dtype::FLOAT32,
                );
                let (ey, es) = (max_rel(&y, &want_y), max_rel(&s, &want_s));
                let what = format!("n {n} seq {seq} carried {carried}");
                assert!(ey < 1e-5, "{what}: y rel error {ey}");
                assert!(es < 1e-5, "{what}: state rel error {es}");
            }
        }
    }
}

#[test]
fn bf16_kernel_is_no_less_accurate_than_the_graph_scan() {
    // A property of the float32-state variant (Metal, ROCm). The CUDA port
    // rounds like the graph scan by design;
    // `cuda_kernel_is_bit_identical_to_the_graph_scan` pins that instead.
    if !ffi::mamba1_scan_float_state_kernel_available() {
        return;
    }
    let (batch, seq, dm, n) = (1, 64, 24, 16);
    let mut k = make_case(batch, seq, dm, n, true);
    // Feed both sides the same bf16-representable inputs, so the comparison
    // isolates how the scan itself rounds.
    for v in [&mut k.x, &mut k.dt, &mut k.b, &mut k.c, &mut k.d] {
        v.iter_mut().for_each(|e| *e = bf16_round(*e));
    }
    let (exact_y, _) = reference(&k, batch, seq, dm, n, None);
    let (graph_y, _) = reference(&k, batch, seq, dm, n, Some(bf16_round));
    let (kernel_y, _) = run_kernel(
        &k,
        batch,
        seq,
        dm,
        n,
        crate::dtype::BFLOAT16,
        crate::dtype::FLOAT32,
    );

    let rms = |got: &[f32]| {
        (got.iter()
            .zip(&exact_y)
            .map(|(g, w)| (g - w) * (g - w))
            .sum::<f32>()
            / got.len() as f32)
            .sqrt()
    };
    let (kernel_err, graph_err) = (rms(&kernel_y), rms(&graph_y));
    assert!(
        kernel_err <= graph_err,
        "kernel RMS error {kernel_err} must not exceed the bf16 graph scan's {graph_err}"
    );
}

/// The per-step graph scan Jamba runs without the kernel
/// (`JambaMambaMixer::ssm_step`), built from the same MLX ops in the same
/// order, all inputs in `dtype` except `A` in `a_dtype`. A float32 `A` with
/// f16 inputs takes Jamba's `decay_in_activation_dtype`: `delta * A` and its
/// `exp` in float32, then the decay rounded to `dtype`. `carried` false
/// passes no state, as a fresh sequence does.
#[allow(clippy::too_many_arguments)]
fn graph_scan(
    k: &Case,
    batch: usize,
    seq: usize,
    dm: usize,
    n: usize,
    dtype: i32,
    a_dtype: i32,
    carried: bool,
) -> (Vec<f32>, Vec<f32>) {
    let arr = |v: &[f32], shape: &[i32]| ffi::astype(&ffi::from_slice_f32(v, shape), dtype);
    let (b, l, d, n_) = (batch as i32, seq as i32, dm as i32, n as i32);
    let x = arr(&k.x, &[b, l, d]);
    let delta = arr(&k.dt, &[b, l, d]);
    let bm = arr(&k.b, &[b, l, n_]);
    let cm = arr(&k.c, &[b, l, n_]);
    let a = ffi::astype(&ffi::from_slice_f32(&k.a, &[d, n_]), a_dtype);
    let dp = arr(&k.d, &[d]);

    let delta_x = ffi::reshape(&ffi::multiply(&delta, &x), &[b, l, d, 1]);
    let new_state = ffi::multiply(&delta_x, &ffi::reshape(&bm, &[b, l, 1, n_]));
    let dt_a = ffi::exp(&ffi::multiply(&ffi::reshape(&delta, &[b, l, d, 1]), &a));
    let dt_a = if a_dtype == dtype {
        dt_a
    } else {
        ffi::astype(&dt_a, dtype)
    };

    let mut state = carried.then(|| arr(&k.s0, &[b, d, n_]));
    let mut ys = Vec::with_capacity(seq);
    for t in 0..l {
        let ns_t = ffi::squeeze_axis(&slice_axis(&new_state, 1, t, t + 1), 1);
        let updated = match state {
            Some(ref prev) => {
                let dt_a_t = ffi::squeeze_axis(&slice_axis(&dt_a, 1, t, t + 1), 1);
                ffi::add(&ffi::multiply(prev, &dt_a_t), &ns_t)
            }
            None => ns_t,
        };
        let c_t = ffi::reshape(
            &ffi::squeeze_axis(&slice_axis(&cm, 1, t, t + 1), 1),
            &[b, n_, 1],
        );
        ys.push(ffi::squeeze_axis(&ffi::matmul(&updated, &c_t), -1));
        state = Some(updated);
    }
    let y = crate::ops::stack_owned(&ys, 1);
    let y = ffi::add(&y, &ffi::multiply(&ffi::reshape(&dp, &[1, 1, d]), &x));
    (to_vec(&y), to_vec(state.as_ref().expect("seq >= 1")))
}

/// Issue #1981: the CUDA kernel replaces the per-step graph scan on the
/// serving path, and greedy output must not move. With uniform-dtype inputs
/// it has to reproduce the graph scan exactly, output rows and final state.
#[test]
fn cuda_kernel_is_bit_identical_to_the_graph_scan() {
    if !ffi::mamba1_scan_kernel_available() || gpu_backend_kind() != GpuBackendKind::Cuda {
        return;
    }
    let (batch, dm) = (2, 24);
    for dtype in [
        crate::dtype::BFLOAT16,
        crate::dtype::FLOAT16,
        crate::dtype::FLOAT32,
    ] {
        for n in [8, 16] {
            for seq in [1, 7, 33] {
                for carried in [false, true] {
                    let mut k = make_case(batch, seq, dm, n, carried);
                    // The kernel reads A and the state in the activation
                    // dtype on CUDA; give the graph the same values.
                    let round = |v: &mut Vec<f32>| {
                        let a = ffi::astype(&ffi::from_slice_f32(v, &[v.len() as i32]), dtype);
                        *v = to_vec(&a);
                    };
                    round(&mut k.a);
                    round(&mut k.s0);
                    let what = format!("dtype {dtype} n {n} seq {seq} carried {carried}");
                    let (want_y, want_s) = graph_scan(&k, batch, seq, dm, n, dtype, dtype, carried);
                    let (y, s) = run_kernel(&k, batch, seq, dm, n, dtype, dtype);
                    assert_bit_identical(&what, (&y, &s), (&want_y, &want_s));
                }
            }
        }
    }
}

fn assert_bit_identical(what: &str, got: (&[f32], &[f32]), want: (&[f32], &[f32])) {
    let diff = |got: &[f32], want: &[f32]| {
        got.iter()
            .zip(want)
            .filter(|(g, w)| g.to_bits() != w.to_bits())
            .count()
    };
    assert_eq!(got.0.len(), want.0.len(), "{what}: y length");
    assert_eq!(
        diff(got.0, want.0),
        0,
        "{what}: y differs from the graph scan"
    );
    assert_eq!(
        diff(got.1, want.1),
        0,
        "{what}: state differs from the graph scan"
    );
}

/// Issue #3125: f16 Mamba and Jamba keep `A` in float32, since f16 cannot hold
/// `-exp(A_log)` past `A_log` 11.09. The CUDA kernel takes that pair and must
/// still reproduce the graph scan bit for bit, including an `A` far outside
/// f16's range and `A` values f16 cannot represent exactly.
#[test]
fn cuda_kernel_is_bit_identical_to_the_graph_scan_for_f16_x_with_float32_a() {
    if !ffi::mamba1_scan_kernel_available() || gpu_backend_kind() != GpuBackendKind::Cuda {
        return;
    }
    let (batch, dm) = (2, 24);
    let (f16, f32) = (crate::dtype::FLOAT16, crate::dtype::FLOAT32);
    for n in [8, 16] {
        for seq in [1, 7, 33] {
            for carried in [false, true] {
                let mut k = make_case(batch, seq, dm, n, carried);
                // -exp(19.5): -inf in f16, the value that motivated float32 A.
                k.a[1] = -(19.5f32.exp());
                // The state is f16 on CUDA; give the graph the same values.
                k.s0 = to_vec(&ffi::astype(
                    &ffi::from_slice_f32(&k.s0, &[k.s0.len() as i32]),
                    f16,
                ));
                let x = ffi::astype(&ffi::from_slice_f32(&k.x, &[1]), f16);
                let a = ffi::from_slice_f32(&k.a, &[k.a.len() as i32]);
                assert!(ffi::mamba1_scan_kernel_accepts(&x, &x, &x, &x, &a, &x));
                let what = format!("f16 x, f32 A, n {n} seq {seq} carried {carried}");
                let (want_y, want_s) = graph_scan(&k, batch, seq, dm, n, f16, f32, carried);
                let (y, s) = run_kernel(&k, batch, seq, dm, n, f16, f32);
                assert!(
                    y.iter().chain(&s).all(|v| v.is_finite()),
                    "{what}: output must be finite"
                );
                assert_bit_identical(&what, (&y, &s), (&want_y, &want_s));
            }
        }
    }
}

/// The dtype rule of the graph-exact port, which needs no CUDA device: one
/// floating dtype throughout, or f16 with a float32 `A` (#3125), and no
/// other mix.
#[test]
fn graph_exact_dtype_rule_admits_only_uniform_inputs_and_f16_with_float32_a() {
    use crate::dtype::{BFLOAT16, FLOAT16, FLOAT32, INT32};
    let accepts =
        |x, delta, b, c, a, d| ffi::mamba1_graph_exact_scan_accepts_dtypes(x, delta, b, c, a, d);
    for t in [FLOAT32, FLOAT16, BFLOAT16] {
        assert!(accepts(t, t, t, t, t, t), "uniform {t}");
    }
    assert!(!accepts(INT32, INT32, INT32, INT32, INT32, INT32));
    // The admitted mix.
    assert!(accepts(
        FLOAT16, FLOAT16, FLOAT16, FLOAT16, FLOAT32, FLOAT16
    ));
    // A float32 A with bf16 or float32 elsewhere is not it.
    assert!(!accepts(
        BFLOAT16, BFLOAT16, BFLOAT16, BFLOAT16, FLOAT32, BFLOAT16
    ));
    assert!(!accepts(
        FLOAT32, FLOAT32, FLOAT32, FLOAT32, FLOAT16, FLOAT32
    ));
    assert!(!accepts(
        FLOAT16, FLOAT16, FLOAT16, FLOAT16, BFLOAT16, FLOAT16
    ));
    // Only A may differ, and only that way.
    for i in 0..4 {
        let mut other = [FLOAT16; 4];
        other[i] = FLOAT32;
        let [delta, b, c, d] = other;
        assert!(
            !accepts(FLOAT16, delta, b, c, FLOAT32, d),
            "f16 with float32 A and float32 input {i}"
        );
        assert!(
            !accepts(FLOAT16, delta, b, c, FLOAT16, d),
            "f16 with float32 input {i}"
        );
    }
}

/// The CUDA kernel serves only inputs it can reproduce exactly; anything else
/// keeps the graph scan.
#[test]
fn cuda_kernel_declines_mixed_dtype_inputs() {
    if !ffi::mamba1_scan_kernel_available() || gpu_backend_kind() != GpuBackendKind::Cuda {
        return;
    }
    let bf = |shape: &[i32]| {
        ffi::astype(
            &ffi::from_slice_f32(&vec![0.5; shape.iter().product::<i32>() as usize], shape),
            crate::dtype::BFLOAT16,
        )
    };
    let (x, dt, b, c, a, d) = (
        bf(&[1, 4, 8]),
        bf(&[1, 4, 8]),
        bf(&[1, 4, 16]),
        bf(&[1, 4, 16]),
        bf(&[8, 16]),
        bf(&[8]),
    );
    assert!(ffi::mamba1_scan_kernel_accepts(&x, &dt, &b, &c, &a, &d));
    let a32 = ffi::astype(&a, crate::dtype::FLOAT32);
    assert!(!ffi::mamba1_scan_kernel_accepts(&x, &dt, &b, &c, &a32, &d));
    // f16 with a float32 A is the one admitted mix (#3125); a float32 delta
    // next to it is not.
    let h = |arr: &MlxArray| ffi::astype(arr, crate::dtype::FLOAT16);
    let (xh, dth, bh, ch, dh) = (h(&x), h(&dt), h(&b), h(&c), h(&d));
    assert!(ffi::mamba1_scan_kernel_accepts(
        &xh, &dth, &bh, &ch, &a32, &dh
    ));
    let dt32 = ffi::astype(&dt, crate::dtype::FLOAT32);
    assert!(!ffi::mamba1_scan_kernel_accepts(
        &xh, &dt32, &bh, &ch, &a32, &dh
    ));
    let wide = bf(&[1, 4, 64]);
    assert!(!ffi::mamba1_scan_kernel_accepts(
        &x,
        &dt,
        &wide,
        &wide,
        &bf(&[8, 64]),
        &d
    ));
}

/// Metal and ROCm (#2069) have the float32-state port, so the tests above
/// must not skip there: a backend that lost its port would otherwise pass
/// them by returning early. `MLXCEL_MAMBA1_SCAN_KERNEL=0` is the one
/// legitimate reason for the predicate to be false on those backends.
#[test]
fn float_state_kernel_is_available_where_ported() {
    if !matches!(
        gpu_backend_kind(),
        GpuBackendKind::Metal | GpuBackendKind::Rocm
    ) || std::env::var("MLXCEL_MAMBA1_SCAN_KERNEL").as_deref() == Ok("0")
        || crate::test_support::kernel_ports::skip_for_wave32_only_rocm_port("Mamba1 scan")
    {
        return;
    }
    assert!(ffi::mamba1_scan_kernel_available());
    assert!(ffi::mamba1_scan_float_state_kernel_available());
}
