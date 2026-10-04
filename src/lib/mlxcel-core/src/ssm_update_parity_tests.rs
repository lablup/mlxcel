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

//! Parity tests for the fused single-token Mamba2 SSM update kernel (#2067).
//!
//! `ssm_update_kernel` replaces the ~55-op SSD graph (`ssm_step` in
//! `src/models/granitemoehybrid.rs` and its falcon-h1, plamo-2 and Nemotron-H
//! siblings) for every single-token decode step that has a state. For one
//! token that graph reduces to the recurrence
//!
//! ```text
//! dt'   = clip(softplus(dt + dt_bias), t_min, t_max)
//! s'    = exp(-exp(A_log) * dt') * s + dt' * x * B
//! y     = s' . C + D * x
//! ```
//!
//! with B and C shared by the heads of a group. The reference below composes
//! exactly that from MLX ops in float32, as the kernel computes it (the graph
//! path promotes x, B, C and dt to float32 too, but forms `-exp(A_log)` in
//! `A_log`'s own dtype, so for a bf16 `A_log` it rounds A where the kernel does
//! not). The kernel's output and new state are compared with it at the
//! granite-4.0-h-tiny and Nemotron-H mixer shapes, in f32 and bf16, and at a
//! head size that leaves padded rows in the last threadgroup.
//!
//! Tolerances are normalized RMS and normalized max deviation (both divided by
//! the reference's RMS): f32 1e-5 / 1e-4 and bf16 1.6e-2 / 7e-2, the bf16
//! budget of `fused_norm_parity_tests.rs`. The state is carried in float32 in
//! every case, as the models carry it, so it is held to the f32 budget even
//! when the activations are bf16.
//!
//! These tests return early only on a build with no GPU backend, when the
//! default device is the CPU, or when the `MLXCEL_SSM_KERNEL=0` /
//! `MLXCEL_SSM_CUDA_KERNEL=0` kill switch is set. On
//! Metal, CUDA and ROCm a false `ssm_kernel_available()` is itself a defect
//! (each has a port), so the test fails there instead of skipping.
//!
//! Run on ROCm:
//!   cargo test --release --features rocm -p mlxcel-core --lib \
//!     ssm_update_parity_tests -- --test-threads=1

use super::*;
use crate::hardware::{GpuBackendKind, gpu_backend_kind};

const KILL_SWITCHES: [&str; 2] = ["MLXCEL_SSM_KERNEL", "MLXCEL_SSM_CUDA_KERNEL"];

/// Normalized (RMS, max) deviation budget for values stored in `dtype`.
fn tolerance_for(dtype: i32) -> (f64, f64) {
    if dtype == dtype::FLOAT32 {
        (1e-5, 1e-4)
    } else {
        (1.6e-2, 7e-2)
    }
}

fn kill_switch_set() -> bool {
    KILL_SWITCHES
        .iter()
        .any(|name| std::env::var(name).is_ok_and(|v| v == "0"))
}

/// True when the calling test should return early, after saying why.
fn skip() -> bool {
    if ssm_kernel_available() {
        return false;
    }
    let backend = gpu_backend_kind();
    if backend == GpuBackendKind::None {
        eprintln!("skipping ssm_update_parity_tests: no GPU backend in this build");
        return true;
    }
    if !default_device_is_gpu() {
        eprintln!("skipping ssm_update_parity_tests: the default device is the CPU");
        return true;
    }
    if kill_switch_set() {
        eprintln!("skipping ssm_update_parity_tests: SSM kernel kill switch is set");
        return true;
    }
    panic!(
        "ssm_kernel_available() is false on {backend:?}, which has a port in ssm_ports(); \
         the predicate and the port table disagree"
    );
}

fn flatten_f32(arr: &MlxArray) -> Vec<f32> {
    let a = astype(arr, dtype::FLOAT32);
    eval(&a);
    array_to_raw_bytes(&a)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// RMS and max of `a - b`, both divided by the RMS of `b`.
fn normalized_deviation(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len(), "length mismatch in deviation check");
    let mut diff_sq = 0f64;
    let mut ref_sq = 0f64;
    let mut max_abs = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        assert!(x.is_finite(), "non-finite kernel output {x}");
        let d = f64::from(*x) - f64::from(*y);
        diff_sq += d * d;
        ref_sq += f64::from(*y) * f64::from(*y);
        max_abs = max_abs.max(d.abs());
    }
    let ref_rms = (ref_sq / b.len() as f64).sqrt().max(1e-20);
    (
        (diff_sq / a.len() as f64).sqrt() / ref_rms,
        max_abs / ref_rms,
    )
}

fn assert_within(label: &str, got: &MlxArray, want: &MlxArray, dtype: i32) {
    let (nrms, nmax) = normalized_deviation(&flatten_f32(got), &flatten_f32(want));
    let (rms_tol, max_tol) = tolerance_for(dtype);
    assert!(
        nrms < rms_tol && nmax < max_tol,
        "{label}: normalized rms {nrms:.3e} (tol {rms_tol:.1e}), \
         normalized max {nmax:.3e} (tol {max_tol:.1e})"
    );
}

/// One mixer's single-token step.
struct Shape {
    name: &'static str,
    batch: i32,
    heads: i32,
    head_dim: i32,
    groups: i32,
    state_dim: i32,
    /// The `(min, max)` clip applied to `softplus(dt + dt_bias)`.
    dt_limits: (f32, f32),
}

/// granite-4.0-h-tiny: 48 heads of 64, one group, state 128, with granite's
/// default `time_step_limit` of `(0.001, 100.0)`.
const GRANITE_TINY: Shape = Shape {
    name: "granite-4.0-h-tiny",
    batch: 1,
    heads: 48,
    head_dim: 64,
    groups: 1,
    state_dim: 128,
    dt_limits: (0.001, 100.0),
};

/// Nemotron-3-Nano-30B-A3B: 64 heads of 64 in 8 groups, state 128. Run at
/// batch 2 with a dt window of `(1e-3, 0.1)` (the checkpoint's
/// `time_step_min` / `time_step_max`; its `time_step_limit` is `(0, inf)`),
/// narrow enough to clip, so the group/batch indexing and the clip are both
/// exercised.
const NEMOTRON_H: Shape = Shape {
    name: "nemotron-h",
    batch: 2,
    heads: 64,
    head_dim: 64,
    groups: 8,
    state_dim: 128,
    dt_limits: (1e-3, 0.1),
};

/// A head size that is not a multiple of the threadgroup's 8 rows, so the last
/// threadgroup has rows past `Dh` that must return before touching memory.
const PADDED_ROWS: Shape = Shape {
    name: "head_dim 60",
    batch: 1,
    heads: 4,
    head_dim: 60,
    groups: 2,
    state_dim: 64,
    dt_limits: (0.0, f32::INFINITY),
};

struct Case {
    x: UniquePtr<MlxArray>,
    a_log: UniquePtr<MlxArray>,
    b: UniquePtr<MlxArray>,
    c: UniquePtr<MlxArray>,
    d: UniquePtr<MlxArray>,
    dt: UniquePtr<MlxArray>,
    dt_bias: UniquePtr<MlxArray>,
    state: UniquePtr<MlxArray>,
}

fn normal(shape: &[i32]) -> UniquePtr<MlxArray> {
    unsafe { random_normal(shape, dtype::FLOAT32, std::ptr::null()) }
}

fn uniform(low: f32, high: f32, shape: &[i32]) -> UniquePtr<MlxArray> {
    unsafe { random_uniform(low, high, shape, dtype::FLOAT32, std::ptr::null()) }
}

/// Random single-token inputs. Activations and the per-head parameters are in
/// `act_dtype`, except `A_log`, which is in `a_log_dtype` (Nemotron-H stores it
/// in f32 next to bf16 activations, granite in bf16). The state is f32.
/// Everything is evaluated up front so the kernel and the reference read the
/// same bytes.
fn make_case(s: &Shape, seed: u64, act_dtype: i32, a_log_dtype: i32) -> Case {
    random_seed(seed);
    let cast = |a: UniquePtr<MlxArray>, dt: i32| {
        let out = astype(&a, dt);
        eval(&out);
        out
    };
    // A = exp(A_log) in [1, 16] and dt' around softplus(-2) ~ 0.13 keep the
    // decay exp(-A dt') well inside (0, 1), so the carried state contributes.
    let a_log = cast(log(&uniform(1.0, 16.0, &[s.heads])), a_log_dtype);
    let dt_bias = cast(uniform(-3.0, -1.0, &[s.heads]), act_dtype);
    Case {
        x: cast(normal(&[s.batch, 1, s.heads, s.head_dim]), act_dtype),
        a_log,
        b: cast(normal(&[s.batch, 1, s.groups, s.state_dim]), act_dtype),
        c: cast(normal(&[s.batch, 1, s.groups, s.state_dim]), act_dtype),
        d: cast(uniform(0.5, 1.5, &[s.heads]), act_dtype),
        dt: cast(normal(&[s.batch, 1, s.heads]), act_dtype),
        dt_bias,
        state: cast(
            normal(&[s.batch, s.heads, s.head_dim, s.state_dim]),
            dtype::FLOAT32,
        ),
    }
}

fn run_kernel(s: &Shape, k: &Case) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let mut out = UniquePtr::null();
    let mut state = UniquePtr::null();
    ssm_update_kernel(
        &k.x,
        &k.a_log,
        &k.b,
        &k.c,
        &k.d,
        &k.dt,
        &k.dt_bias,
        &k.state,
        s.dt_limits.0,
        s.dt_limits.1,
        &mut out,
        &mut state,
    )
    .expect("ssm_kernel_available() is true, so the launcher must not refuse");
    eval(&out);
    eval(&state);
    (out, state)
}

/// The single-token SSD step from MLX ops, in float32. Returns `y` as
/// `[batch, 1, heads, head_dim]` and the new state as
/// `[batch, heads, head_dim, state_dim]`.
fn reference(s: &Shape, k: &Case) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let f32_of = |a: &MlxArray| astype(a, dtype::FLOAT32);
    let (b, h, dh, g, ds) = (s.batch, s.heads, s.head_dim, s.groups, s.state_dim);

    let lo = full_f32(&[1], s.dt_limits.0, dtype::FLOAT32);
    let hi = full_f32(&[1], s.dt_limits.1, dtype::FLOAT32);
    let dt = clip(
        &softplus(&add(&f32_of(&k.dt), &f32_of(&k.dt_bias))),
        &lo,
        &hi,
    );
    let dt = reshape(&dt, &[b, h, 1, 1]);

    let a = negative(&exp(&f32_of(&k.a_log)));
    let decay = exp(&multiply(&dt, &reshape(&a, &[1, h, 1, 1])));

    // B and C per head: [b, 1, g, ds] -> [b, g, ds] -> repeat heads/group -> [b, h, 1, ds].
    let per_head = |m: &MlxArray| {
        let m = reshape(&f32_of(m), &[b, g, ds]);
        let m = repeat(&m, h / g, 1);
        reshape(&m, &[b, h, 1, ds])
    };
    let b_h = per_head(&k.b);
    let c_h = per_head(&k.c);

    let x = reshape(&f32_of(&k.x), &[b, h, dh, 1]);
    let dbx = multiply(&multiply(&dt, &x), &b_h);
    let new_state = add(&multiply(&decay, &k.state), &dbx);

    let y = sum_axis(&multiply(&new_state, &c_h), -1, false);
    let d = reshape(&f32_of(&k.d), &[1, h, 1]);
    let x_flat = reshape(&x, &[b, h, dh]);
    let y = add(&y, &multiply(&x_flat, &d));
    let y = reshape(&y, &[b, 1, h, dh]);
    eval(&y);
    eval(&new_state);
    (y, new_state)
}

fn check(s: &Shape, seed: u64, act_dtype: i32, a_log_dtype: i32) {
    let case = make_case(s, seed, act_dtype, a_log_dtype);
    let (out, state) = run_kernel(s, &case);
    let (want_out, want_state) = reference(s, &case);
    let backend = gpu_backend_kind();
    let label = format!(
        "{} act dtype {act_dtype} A_log dtype {a_log_dtype} on {backend:?}",
        s.name
    );
    assert_eq!(array_dtype(&out), act_dtype, "{label}: output dtype");
    assert_eq!(
        array_dtype(&state),
        dtype::FLOAT32,
        "{label}: state dtype follows state_in"
    );
    assert_within(&format!("{label}: y"), &out, &want_out, act_dtype);
    assert_within(
        &format!("{label}: new state"),
        &state,
        &want_state,
        dtype::FLOAT32,
    );
}

#[test]
fn ssm_update_kernel_matches_graph_step_f32() {
    let _guard = crate::test_support::env_lock::env_lock();
    if skip() {
        return;
    }
    check(&GRANITE_TINY, 2067, dtype::FLOAT32, dtype::FLOAT32);
    check(&NEMOTRON_H, 2068, dtype::FLOAT32, dtype::FLOAT32);
    check(&PADDED_ROWS, 2073, dtype::FLOAT32, dtype::FLOAT32);
}

#[test]
fn ssm_update_kernel_matches_graph_step_bf16() {
    let _guard = crate::test_support::env_lock::env_lock();
    if skip() {
        return;
    }
    check(&GRANITE_TINY, 2069, dtype::BFLOAT16, dtype::BFLOAT16);
    check(&NEMOTRON_H, 2070, dtype::BFLOAT16, dtype::FLOAT32);
}

/// The same shape and activation dtype with `A_log` first in bf16 and then in
/// f32, in one process. The kernel's generated signature takes each input's
/// runtime dtype; on a backend whose JIT cache key omits `A_log`'s dtype the
/// second launch would reuse the module compiled for the first and read f32
/// bytes as bf16.
#[test]
fn ssm_update_kernel_keys_on_a_log_dtype() {
    let _guard = crate::test_support::env_lock::env_lock();
    if skip() {
        return;
    }
    check(&GRANITE_TINY, 2071, dtype::BFLOAT16, dtype::BFLOAT16);
    check(&GRANITE_TINY, 2072, dtype::BFLOAT16, dtype::FLOAT32);
}

/// Shapes the kernel cannot index safely are refused on the host, as an error
/// the caller sees, rather than launched: a state width that is not a multiple
/// of 32 (or under 32) would leave state columns unwritten, heads that do not
/// divide into groups would read past B and C, and a per-head parameter of the
/// wrong length would be read past its end.
#[test]
fn ssm_update_kernel_refuses_unsupported_shapes() {
    let _guard = crate::test_support::env_lock::env_lock();
    if skip() {
        return;
    }
    let bad = [
        Shape {
            name: "state 48",
            state_dim: 48,
            ..PADDED_ROWS
        },
        Shape {
            name: "state 16",
            state_dim: 16,
            ..PADDED_ROWS
        },
        Shape {
            name: "4 heads in 3 groups",
            groups: 3,
            ..PADDED_ROWS
        },
        // Valid shape; `D` is given one element too many below.
        Shape {
            name: "D of length heads + 1",
            ..PADDED_ROWS
        },
    ];
    for s in &bad {
        let mut case = make_case(s, 2074, dtype::FLOAT32, dtype::FLOAT32);
        if s.name.starts_with("D of length") {
            case.d = uniform(0.5, 1.5, &[s.heads + 1]);
        }
        let mut out = UniquePtr::null();
        let mut state = UniquePtr::null();
        let result = ssm_update_kernel(
            &case.x,
            &case.a_log,
            &case.b,
            &case.c,
            &case.d,
            &case.dt,
            &case.dt_bias,
            &case.state,
            s.dt_limits.0,
            s.dt_limits.1,
            &mut out,
            &mut state,
        );
        let err = result.expect_err(s.name);
        assert!(
            err.what().contains("unsupported shapes"),
            "{}: unexpected error {}",
            s.name,
            err.what()
        );
    }
}

/// `MLXCEL_SSM_KERNEL=0` and its older alias `MLXCEL_SSM_CUDA_KERNEL=0` each
/// turn the predicate off, on every backend with a port.
#[test]
fn ssm_kernel_kill_switches_turn_predicate_off() {
    let _guard = crate::test_support::env_lock::env_lock();
    let saved: Vec<(&str, Option<String>)> = KILL_SWITCHES
        .iter()
        .map(|name| (*name, std::env::var(name).ok()))
        .collect();
    // SAFETY (every env mutation in this test): the crate-wide env_lock guard
    // is held for the whole window, and every other test in this crate that
    // mutates the environment takes the same lock.
    for name in KILL_SWITCHES {
        unsafe { std::env::remove_var(name) };
    }
    // Checked before restoring the environment, so a failure leaves it as the
    // test found it.
    let result = std::panic::catch_unwind(|| {
        if skip() {
            return;
        }
        for name in KILL_SWITCHES {
            unsafe { std::env::set_var(name, "0") };
            assert!(
                !ssm_kernel_available(),
                "{name}=0 must force the graph path"
            );
            unsafe { std::env::set_var(name, "1") };
            assert!(ssm_kernel_available(), "{name}=1 must leave the kernel on");
            unsafe { std::env::remove_var(name) };
        }
    });
    for (name, value) in saved {
        match value {
            Some(v) => unsafe { std::env::set_var(name, v) },
            None => unsafe { std::env::remove_var(name) },
        }
    }
    if let Err(panic) = result {
        std::panic::resume_unwind(panic);
    }
}
