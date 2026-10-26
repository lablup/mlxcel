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

//! Native sliding-window SDPA route (issue #2621): the route decision as a
//! pure function, and on Metal the native entry against the legacy explicit
//! mask and an independent host reference.
//!
//! Tolerances, fixed before the native arm was run: max abs difference
//! against the f64 host reference of `3e-5` for float32, `2e-3` for float16
//! (about two ulp at the output magnitude) and `1.6e-2` for bfloat16 (two
//! ulp). Keys outside every row's window and keys after a row's position must
//! not move that row at all, which is checked bitwise.

use crate::cache::RotatingKVCache;
use crate::dtype;
use crate::ffi::{self, MlxArray};
use crate::hardware::{GpuBackendKind, gpu_backend_kind};
use crate::window_sdpa::{
    DEFAULT_MODE, MIN_NATIVE_QUERY_LEN, NativeWindowSdpaMode, WindowSdpaExtras, WindowSdpaRoute,
    WindowSdpaShape, causal_window_attention_with_mode, dispatch_counts,
    legacy_causal_window_attention, legacy_causal_window_mask, native_arm_is_fused,
    native_causal_window_attention, native_window_eligibility, parse_mode, plan_with,
    rotating_prefill_key_len,
};
use crate::{UniquePtr, utils};

const SCALE: f32 = 0.125;

fn tolerance(dtype_code: i32) -> f32 {
    match dtype_code {
        dtype::FLOAT32 => 3e-5,
        dtype::FLOAT16 => 2e-3,
        dtype::BFLOAT16 => 1.6e-2,
        other => panic!("no tolerance stated for dtype {other}"),
    }
}

fn eligible_shape() -> WindowSdpaShape {
    WindowSdpaShape {
        batch: 1,
        q_heads: 8,
        kv_heads: 2,
        q_len: 64,
        k_len: 96,
        head_dim: 128,
        value_dim: 128,
        q_dtype: dtype::FLOAT16,
        k_dtype: dtype::FLOAT16,
        v_dtype: dtype::FLOAT16,
    }
}

const WINDOW: i32 = 32;

#[test]
fn parse_mode_reads_off_on_auto_and_defaults() {
    assert_eq!(parse_mode(Some("0")), NativeWindowSdpaMode::Off);
    assert_eq!(parse_mode(Some(" OFF ")), NativeWindowSdpaMode::Off);
    assert_eq!(parse_mode(Some("legacy")), NativeWindowSdpaMode::Off);
    assert_eq!(parse_mode(Some("1")), NativeWindowSdpaMode::On);
    assert_eq!(parse_mode(Some("on")), NativeWindowSdpaMode::On);
    assert_eq!(parse_mode(Some("native")), NativeWindowSdpaMode::On);
    assert_eq!(parse_mode(Some("auto")), NativeWindowSdpaMode::Auto);
    assert_eq!(parse_mode(None), DEFAULT_MODE);
    assert_eq!(parse_mode(Some("")), DEFAULT_MODE);
    assert_eq!(parse_mode(Some("maybe")), DEFAULT_MODE);
}

#[test]
fn eligibility_accepts_the_causal_window_prefill_shape() {
    let shape = eligible_shape();
    assert_eq!(
        native_window_eligibility(
            &shape,
            WINDOW,
            WindowSdpaExtras::default(),
            GpuBackendKind::Metal
        ),
        Ok(())
    );
    for head_dim in [64, 72, 80, 96, 128, 192, 256] {
        let shape = WindowSdpaShape {
            head_dim,
            value_dim: head_dim,
            ..eligible_shape()
        };
        assert_eq!(
            native_window_eligibility(
                &shape,
                WINDOW,
                WindowSdpaExtras::default(),
                GpuBackendKind::Metal
            ),
            Ok(()),
            "head dim {head_dim} is in the full-kernel set"
        );
    }
    for d in [dtype::FLOAT32, dtype::FLOAT16, dtype::BFLOAT16] {
        let shape = WindowSdpaShape {
            q_dtype: d,
            k_dtype: d,
            v_dtype: d,
            ..eligible_shape()
        };
        assert_eq!(
            native_window_eligibility(
                &shape,
                WINDOW,
                WindowSdpaExtras::default(),
                GpuBackendKind::Metal
            ),
            Ok(()),
            "dtype {d} is eligible"
        );
    }
    // MHA as well as GQA.
    let mha = WindowSdpaShape {
        q_heads: 4,
        kv_heads: 4,
        ..eligible_shape()
    };
    assert_eq!(
        native_window_eligibility(
            &mha,
            WINDOW,
            WindowSdpaExtras::default(),
            GpuBackendKind::Metal
        ),
        Ok(())
    );
    // The smallest eligible query length and a query block longer than the
    // window are both in.
    let nine = WindowSdpaShape {
        q_len: MIN_NATIVE_QUERY_LEN,
        ..eligible_shape()
    };
    assert_eq!(
        native_window_eligibility(
            &nine,
            WINDOW,
            WindowSdpaExtras::default(),
            GpuBackendKind::Metal
        ),
        Ok(())
    );
    let long_block = WindowSdpaShape {
        q_len: 96,
        k_len: 96,
        ..eligible_shape()
    };
    assert_eq!(
        native_window_eligibility(
            &long_block,
            WINDOW,
            WindowSdpaExtras::default(),
            GpuBackendKind::Metal
        ),
        Ok(())
    );
    assert_eq!(
        native_window_eligibility(
            &long_block,
            1,
            WindowSdpaExtras::default(),
            GpuBackendKind::Metal
        ),
        Ok(()),
        "W = 1 is a positive window"
    );
}

#[test]
fn eligibility_names_every_exclusion() {
    let base = eligible_shape();
    let plain = WindowSdpaExtras::default();
    let check = |shape: WindowSdpaShape, window: i32, extras: WindowSdpaExtras, backend, want| {
        assert_eq!(
            native_window_eligibility(&shape, window, extras, backend),
            Err(want),
            "expected {want}"
        );
    };
    for backend in [
        GpuBackendKind::Cuda,
        GpuBackendKind::Rocm,
        GpuBackendKind::None,
    ] {
        check(base, WINDOW, plain, backend, "backend_not_metal");
    }
    let metal = GpuBackendKind::Metal;
    check(base, 0, plain, metal, "window_disabled");
    check(base, -1, plain, metal, "window_disabled");
    check(
        base,
        WINDOW,
        WindowSdpaExtras {
            softcap: 50.0,
            ..plain
        },
        metal,
        "softcap",
    );
    check(
        base,
        WINDOW,
        WindowSdpaExtras {
            array_mask: true,
            ..plain
        },
        metal,
        "array_mask",
    );
    check(
        base,
        WINDOW,
        WindowSdpaExtras {
            sinks: true,
            ..plain
        },
        metal,
        "sinks",
    );
    // Decode and small verify blocks stay on their existing routes.
    for q_len in [1, 4, 8] {
        check(
            WindowSdpaShape { q_len, ..base },
            WINDOW,
            plain,
            metal,
            "query_len_le_8",
        );
    }
    check(
        WindowSdpaShape {
            q_len: 97,
            k_len: 96,
            ..base
        },
        WINDOW,
        plain,
        metal,
        "query_longer_than_keys",
    );
    // At or below the window the plain causal path is the right one.
    check(
        WindowSdpaShape {
            q_len: 32,
            k_len: 32,
            ..base
        },
        WINDOW,
        plain,
        metal,
        "keys_within_window",
    );
    check(
        WindowSdpaShape {
            q_len: 16,
            k_len: 24,
            ..base
        },
        WINDOW,
        plain,
        metal,
        "keys_within_window",
    );
    check(
        WindowSdpaShape {
            q_heads: 6,
            kv_heads: 4,
            ..base
        },
        WINDOW,
        plain,
        metal,
        "head_count_ratio",
    );
    check(
        WindowSdpaShape {
            head_dim: 100,
            value_dim: 100,
            ..base
        },
        WINDOW,
        plain,
        metal,
        "head_dim",
    );
    check(
        WindowSdpaShape {
            head_dim: 128,
            value_dim: 64,
            ..base
        },
        WINDOW,
        plain,
        metal,
        "head_dim",
    );
    check(
        WindowSdpaShape {
            q_dtype: dtype::INT8,
            ..base
        },
        WINDOW,
        plain,
        metal,
        "dtype",
    );
}

#[test]
fn plan_follows_the_mode_and_the_fused_verdict() {
    let shape = eligible_shape();
    let plain = WindowSdpaExtras::default();
    let metal = GpuBackendKind::Metal;
    let probe_not_called = |_: &WindowSdpaShape, _: i32| -> bool { panic!("probe must not run") };

    let off = plan_with(
        &shape,
        WINDOW,
        plain,
        NativeWindowSdpaMode::Off,
        metal,
        probe_not_called,
    );
    assert_eq!(off.route, WindowSdpaRoute::Legacy);
    assert_eq!(off.reason, "mode_off");
    assert_eq!(off.fused, None);

    let on = plan_with(
        &shape,
        WINDOW,
        plain,
        NativeWindowSdpaMode::On,
        metal,
        probe_not_called,
    );
    assert!(on.is_native());
    assert_eq!(on.reason, "eligible");

    let auto_fused = plan_with(
        &shape,
        WINDOW,
        plain,
        NativeWindowSdpaMode::Auto,
        metal,
        |_, _| true,
    );
    assert!(auto_fused.is_native());
    assert_eq!(auto_fused.fused, Some(true));

    let auto_graph = plan_with(
        &shape,
        WINDOW,
        plain,
        NativeWindowSdpaMode::Auto,
        metal,
        |_, _| false,
    );
    assert_eq!(auto_graph.route, WindowSdpaRoute::Legacy);
    assert_eq!(auto_graph.reason, "graph_fallback");
    assert_eq!(auto_graph.fused, Some(false));

    // An ineligible call never consults the mode or the probe: forcing the
    // route on cannot drop a softcap, a sink, an extra mask or a backend.
    let softcapped = plan_with(
        &shape,
        WINDOW,
        WindowSdpaExtras {
            softcap: 30.0,
            ..plain
        },
        NativeWindowSdpaMode::On,
        metal,
        probe_not_called,
    );
    assert_eq!(softcapped.route, WindowSdpaRoute::Legacy);
    assert_eq!(softcapped.reason, "softcap");
    let cuda = plan_with(
        &shape,
        WINDOW,
        plain,
        NativeWindowSdpaMode::On,
        GpuBackendKind::Cuda,
        probe_not_called,
    );
    assert_eq!(cuda.reason, "backend_not_metal");
    let decode = plan_with(
        &WindowSdpaShape { q_len: 1, ..shape },
        WINDOW,
        plain,
        NativeWindowSdpaMode::On,
        metal,
        probe_not_called,
    );
    assert_eq!(decode.reason, "query_len_le_8");
}

#[test]
fn rotating_prefill_key_len_follows_the_678_contract() {
    assert_eq!(
        rotating_prefill_key_len(8, 0, 16),
        8,
        "fresh prefill: new keys only"
    );
    assert_eq!(
        rotating_prefill_key_len(8, 5, 16),
        13,
        "short history is kept whole"
    );
    assert_eq!(
        rotating_prefill_key_len(8, 100, 16),
        23,
        "a rolled-over ring keeps window - 1 prior keys"
    );
    assert_eq!(
        rotating_prefill_key_len(8, 100, 1),
        8,
        "W = 1 keeps no prior key"
    );
    assert_eq!(rotating_prefill_key_len(24, 15, 16), 39);
}

// ── Metal: the native entry against the legacy mask and a host reference ──

/// xorshift64* in [-1, 1), deterministic so a failure reproduces exactly.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

fn to_vec_f32(arr: &MlxArray) -> Vec<f32> {
    utils::array_to_vec_f32(arr)
}

fn random_array(rng: &mut Rng, shape: &[i32], dtype_code: i32) -> UniquePtr<MlxArray> {
    let n: usize = shape.iter().map(|d| *d as usize).product();
    let data: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
    let f32_arr = ffi::from_slice_f32(&data, shape);
    if dtype_code == dtype::FLOAT32 {
        f32_arr
    } else {
        ffi::astype(&f32_arr, dtype_code)
    }
}

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    q_len: i32,
    k_len: i32,
    window: i32,
    q_heads: i32,
    kv_heads: i32,
    head_dim: i32,
}

/// Window-causal attention on the host in f64 over the rounded inputs: row
/// `r` of `q_len` queries, aligned to the suffix of `k_len` keys, sees key
/// `c` iff `c <= r + off` and `c >= r + off - window + 1`, `off = k_len -
/// q_len`. GQA repeats each KV head over its query group.
fn host_reference(q: &[f32], k: &[f32], v: &[f32], case: Case, scale: f32) -> Vec<f32> {
    let Case {
        q_len,
        k_len,
        window,
        q_heads,
        kv_heads,
        head_dim,
        ..
    } = case;
    let (q_len, k_len, d) = (q_len as usize, k_len as usize, head_dim as usize);
    let group = (q_heads / kv_heads) as usize;
    let off = k_len as i64 - q_len as i64;
    let mut out = vec![0f32; q_heads as usize * q_len * d];
    for h in 0..q_heads as usize {
        let kv_h = h / group;
        for r in 0..q_len {
            let q_row = &q[(h * q_len + r) * d..(h * q_len + r + 1) * d];
            let lo = (r as i64 + off - window as i64 + 1).max(0) as usize;
            let hi = (r as i64 + off) as usize;
            let mut scores = Vec::with_capacity(hi + 1 - lo);
            for c in lo..=hi {
                let k_row = &k[(kv_h * k_len + c) * d..(kv_h * k_len + c + 1) * d];
                let dot: f64 = q_row
                    .iter()
                    .zip(k_row)
                    .map(|(a, b)| *a as f64 * *b as f64)
                    .sum();
                scores.push(dot * scale as f64);
            }
            let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let denom: f64 = weights.iter().sum();
            let out_row = &mut out[(h * q_len + r) * d..(h * q_len + r + 1) * d];
            for (w, c) in weights.iter().zip(lo..=hi) {
                let v_row = &v[(kv_h * k_len + c) * d..(kv_h * k_len + c + 1) * d];
                for (o, x) in out_row.iter_mut().zip(v_row) {
                    *o += (w / denom * *x as f64) as f32;
                }
            }
        }
    }
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn metal_or_skip() -> bool {
    if gpu_backend_kind() != GpuBackendKind::Metal {
        eprintln!("skipping: the native window route is Metal-only");
        return false;
    }
    true
}

fn run_case(case: Case, dtype_code: i32, seed: u64) {
    let mut rng = Rng::new(seed);
    let q = random_array(
        &mut rng,
        &[1, case.q_heads, case.q_len, case.head_dim],
        dtype_code,
    );
    let k = random_array(
        &mut rng,
        &[1, case.kv_heads, case.k_len, case.head_dim],
        dtype_code,
    );
    let v = random_array(
        &mut rng,
        &[1, case.kv_heads, case.k_len, case.head_dim],
        dtype_code,
    );
    let (qf, kf, vf) = (to_vec_f32(&q), to_vec_f32(&k), to_vec_f32(&v));
    let reference = host_reference(&qf, &kf, &vf, case, SCALE);

    let native = to_vec_f32(&native_causal_window_attention(
        &q,
        &k,
        &v,
        SCALE,
        case.window,
    ));
    let legacy = to_vec_f32(&legacy_causal_window_attention(
        &q,
        &k,
        &v,
        SCALE,
        case.window,
    ));
    let tol = tolerance(dtype_code);
    assert!(
        native.iter().all(|x| x.is_finite()),
        "{}: native output has a non-finite value",
        case.name
    );
    let native_err = max_abs_diff(&native, &reference);
    let legacy_err = max_abs_diff(&legacy, &reference);
    let arms = max_abs_diff(&native, &legacy);
    assert!(
        native_err <= tol,
        "{} dtype {dtype_code}: native vs host reference {native_err} > {tol}",
        case.name
    );
    assert!(
        legacy_err <= tol,
        "{} dtype {dtype_code}: legacy vs host reference {legacy_err} > {tol}",
        case.name
    );
    assert!(
        arms <= tol,
        "{} dtype {dtype_code}: native vs legacy {arms} > {tol}",
        case.name
    );
}

const CASES: [Case; 8] = [
    Case {
        name: "fresh prefill longer than the window, GQA, D=128",
        q_len: 48,
        k_len: 48,
        window: 16,
        q_heads: 8,
        kv_heads: 2,
        head_dim: 128,
    },
    Case {
        name: "continuation after a rolled-over ring (W-1 prior keys)",
        q_len: 24,
        k_len: 39,
        window: 16,
        q_heads: 8,
        kv_heads: 2,
        head_dim: 128,
    },
    Case {
        name: "continuation with a short history",
        q_len: 16,
        k_len: 20,
        window: 16,
        q_heads: 4,
        kv_heads: 4,
        head_dim: 64,
    },
    Case {
        name: "query block longer than the window",
        q_len: 40,
        k_len: 52,
        window: 12,
        q_heads: 4,
        kv_heads: 1,
        head_dim: 64,
    },
    Case {
        name: "W = 1",
        q_len: 16,
        k_len: 24,
        window: 1,
        q_heads: 2,
        kv_heads: 2,
        head_dim: 64,
    },
    Case {
        name: "smallest native query length",
        q_len: 9,
        k_len: 40,
        window: 10,
        q_heads: 8,
        kv_heads: 2,
        head_dim: 128,
    },
    Case {
        name: "Gemma 3 1B shape, D=256, MQA",
        q_len: 48,
        k_len: 48,
        window: 16,
        q_heads: 4,
        kv_heads: 1,
        head_dim: 256,
    },
    Case {
        name: "unaligned lengths past one key block",
        q_len: 37,
        k_len: 70,
        window: 23,
        q_heads: 8,
        kv_heads: 2,
        head_dim: 128,
    },
];

#[test]
fn native_matches_legacy_and_host_reference_in_float32() {
    if !metal_or_skip() {
        return;
    }
    for (i, case) in CASES.iter().enumerate() {
        run_case(*case, dtype::FLOAT32, 0x2621_0001 + i as u64);
    }
}

#[test]
fn native_matches_legacy_and_host_reference_in_float16() {
    if !metal_or_skip() {
        return;
    }
    for (i, case) in CASES.iter().enumerate() {
        run_case(*case, dtype::FLOAT16, 0x2621_0101 + i as u64);
    }
}

#[test]
fn native_matches_legacy_and_host_reference_in_bfloat16() {
    if !metal_or_skip() {
        return;
    }
    for (i, case) in CASES.iter().enumerate() {
        run_case(*case, dtype::BFLOAT16, 0x2621_0201 + i as u64);
    }
}

/// Keys no row can see (older than every window, or after a row's own
/// position) must not move the output bitwise; a key inside a row's window
/// must.
#[test]
fn invisible_keys_cannot_affect_the_native_output() {
    if !metal_or_skip() {
        return;
    }
    let case = Case {
        name: "invisible keys",
        q_len: 16,
        k_len: 40,
        window: 8,
        q_heads: 8,
        kv_heads: 2,
        head_dim: 128,
    };
    let mut rng = Rng::new(0x2621_0301);
    let q = random_array(&mut rng, &[1, 8, 16, 128], dtype::FLOAT16);
    let k = random_array(&mut rng, &[1, 2, 40, 128], dtype::FLOAT16);
    let v = random_array(&mut rng, &[1, 2, 40, 128], dtype::FLOAT16);
    let base = to_vec_f32(&native_causal_window_attention(
        &q,
        &k,
        &v,
        SCALE,
        case.window,
    ));

    // Row 0 sits at logical position 24 and sees keys 17..=24; nothing sees
    // keys 0..17. Replace them with large values.
    let perturb = |arr: &MlxArray, start: i32, stop: i32, value: f32| {
        let shape = ffi::array_shape(arr);
        let patch = ffi::astype(
            &ffi::full_f32(
                &[shape[0], shape[1], stop - start, shape[3]],
                value,
                dtype::FLOAT32,
            ),
            ffi::array_dtype(arr),
        );
        let before = ffi::slice(arr, &[0, 0, 0, 0], &[shape[0], shape[1], start, shape[3]]);
        let after = ffi::slice(
            arr,
            &[0, 0, stop, 0],
            &[shape[0], shape[1], shape[2], shape[3]],
        );
        let head = crate::concatenate(&before, &patch, 2);
        crate::concatenate(&head, &after, 2)
    };
    let k_old = perturb(&k, 0, 17, 7.0);
    let v_old = perturb(&v, 0, 17, -9.0);
    let out = to_vec_f32(&native_causal_window_attention(
        &q,
        &k_old,
        &v_old,
        SCALE,
        case.window,
    ));
    assert_eq!(
        max_abs_diff(&base, &out),
        0.0,
        "keys older than every window changed the output"
    );

    // The last key is visible to the last row only; every other row must be
    // bitwise unchanged when it moves.
    let k_future = perturb(&k, 39, 40, 5.0);
    let v_future = perturb(&v, 39, 40, -5.0);
    let out = to_vec_f32(&native_causal_window_attention(
        &q,
        &k_future,
        &v_future,
        SCALE,
        case.window,
    ));
    let row = 128usize;
    for h in 0..8usize {
        for r in 0..15usize {
            let span = (h * 16 + r) * row..(h * 16 + r + 1) * row;
            assert_eq!(
                max_abs_diff(&base[span.clone()], &out[span]),
                0.0,
                "head {h} row {r} saw the future key"
            );
        }
        let last = (h * 16 + 15) * row..(h * 16 + 16) * row;
        assert!(
            max_abs_diff(&base[last.clone()], &out[last]) > 0.0,
            "head {h}: the last row must see the last key"
        );
    }

    // A key inside row 0's window (key 20) must move row 0.
    let k_in = perturb(&k, 20, 21, 3.0);
    let out = to_vec_f32(&native_causal_window_attention(
        &q,
        &k_in,
        &v,
        SCALE,
        case.window,
    ));
    assert!(
        max_abs_diff(&base[..row], &out[..row]) > 0.0,
        "a visible key must move the row that sees it"
    );
}

/// The mode switch picks the arm, both arms agree, and `causal_attention`'s
/// over-window branch reaches this module.
#[test]
fn causal_attention_over_window_prefill_takes_the_planned_route() {
    if !metal_or_skip() {
        return;
    }
    let mut rng = Rng::new(0x2621_0401);
    let q = random_array(&mut rng, &[1, 8, 48, 128], dtype::FLOAT16);
    let k = random_array(&mut rng, &[1, 2, 48, 128], dtype::FLOAT16);
    let v = random_array(&mut rng, &[1, 2, 48, 128], dtype::FLOAT16);
    let window = 16;
    let before = dispatch_counts();
    let on = to_vec_f32(&causal_window_attention_with_mode(
        &q,
        &k,
        &v,
        SCALE,
        window,
        NativeWindowSdpaMode::On,
    ));
    let off = to_vec_f32(&causal_window_attention_with_mode(
        &q,
        &k,
        &v,
        SCALE,
        window,
        NativeWindowSdpaMode::Off,
    ));
    let after = dispatch_counts();
    assert!(
        after.native > before.native,
        "the forced-on call counts as native"
    );
    assert!(
        after.legacy > before.legacy,
        "the forced-off call counts as legacy"
    );
    let tol = tolerance(dtype::FLOAT16);
    assert!(max_abs_diff(&on, &off) <= tol, "the two modes disagree");

    // The shared dispatcher under the process mode lands within tolerance of
    // both arms, whichever it chose.
    let shared = to_vec_f32(&crate::causal_attention(&q, &k, &v, SCALE, 0.0, window));
    assert!(max_abs_diff(&shared, &off) <= tol);
    let mask = legacy_causal_window_mask(&q, &k, window);
    let explicit = to_vec_f32(&crate::layers::attention(
        &q,
        &k,
        &v,
        SCALE,
        Some(mask.as_ref().unwrap()),
        0.0,
        window,
    ));
    assert_eq!(
        off, explicit,
        "the legacy arm is the pre-#2621 masked call, byte for byte"
    );
}

/// A softcapped over-window prefill keeps the softcap composite with the
/// explicit mask: the native entry has no softcap.
#[test]
fn softcapped_over_window_prefill_keeps_the_mask_route() {
    if !metal_or_skip() {
        return;
    }
    let mut rng = Rng::new(0x2621_0501);
    let q = random_array(&mut rng, &[1, 8, 48, 128], dtype::FLOAT32);
    let k = random_array(&mut rng, &[1, 2, 48, 128], dtype::FLOAT32);
    let v = random_array(&mut rng, &[1, 2, 48, 128], dtype::FLOAT32);
    let (window, softcap) = (16, 30.0);
    let got = to_vec_f32(&crate::causal_attention(&q, &k, &v, SCALE, softcap, window));
    let mask = utils::create_causal_mask_with_window_full(48, 0, Some(window));
    let want = to_vec_f32(&crate::layers::attention(
        &q,
        &k,
        &v,
        SCALE,
        Some(mask.as_ref().unwrap()),
        softcap,
        window,
    ));
    assert_eq!(got, want, "softcap must not reach the native window entry");
}

/// MLX's dispatch verdict on the pinned commit: head dim 128 reaches the full
/// attention kernel with a window; head dim 256 does so on a pre-NAX GPU only
/// for a fresh half-precision prefill of 2048 rows or more.
#[test]
fn fused_probe_matches_the_pinned_dispatch_rules() {
    if !metal_or_skip() {
        return;
    }
    let d128 = WindowSdpaShape {
        q_len: 64,
        k_len: 96,
        ..eligible_shape()
    };
    assert!(
        native_arm_is_fused(&d128, WINDOW),
        "D=128 window prefill is fused"
    );
    assert!(
        !native_arm_is_fused(
            &WindowSdpaShape {
                q_len: 97,
                k_len: 96,
                ..d128
            },
            WINDOW
        ),
        "queries longer than the keys have no fused window kernel"
    );
    let hw = crate::hardware::get_hardware();
    if hw.has_neural_accelerator && hw.macos_supports_na {
        return;
    }
    let gemma = WindowSdpaShape {
        q_heads: 4,
        kv_heads: 1,
        head_dim: 256,
        value_dim: 256,
        ..eligible_shape()
    };
    assert!(
        !native_arm_is_fused(
            &WindowSdpaShape {
                q_len: 1024,
                k_len: 1024,
                ..gemma
            },
            512
        ),
        "D=256 below 2048 rows is the graph fallback on a pre-NAX GPU"
    );
    assert!(
        native_arm_is_fused(
            &WindowSdpaShape {
                q_len: 2048,
                k_len: 2048,
                ..gemma
            },
            512
        ),
        "a fresh 2048-row half-precision D=256 prefill is fused on a pre-NAX GPU"
    );
    assert!(
        !native_arm_is_fused(
            &WindowSdpaShape {
                q_len: 2048,
                k_len: 2559,
                ..gemma
            },
            512
        ),
        "a D=256 continuation chunk is the graph fallback on a pre-NAX GPU"
    );
}

// ── The rotating cache carries the window ──

fn rotating_step(
    rng: &mut Rng,
    len: i32,
    heads: i32,
    kv_heads: i32,
    dim: i32,
) -> (
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
) {
    (
        random_array(rng, &[1, heads, len, dim], dtype::FLOAT32),
        random_array(rng, &[1, kv_heads, len, dim], dtype::FLOAT32),
        random_array(rng, &[1, kv_heads, len, dim], dtype::FLOAT32),
    )
}

/// An unmasked over-window multi-token step through `RotatingKVCache::attend`
/// equals the explicit sliding mask the model used to build, for a fresh
/// prefill past the window, a continuation chunk after the ring rolled over,
/// and the decode steps that follow (which stay on the maskless single-query
/// path); the cache cursors agree throughout.
#[test]
fn rotating_attend_without_a_mask_applies_the_window() {
    if !metal_or_skip() {
        return;
    }
    const WINDOW: i32 = 16;
    const HEADS: i32 = 4;
    const KV: i32 = 2;
    const DIM: i32 = 64;
    let tol = tolerance(dtype::FLOAT32);
    let mut rng = Rng::new(0x2621_0601);
    let mut got = RotatingKVCache::new(WINDOW);
    let mut want = RotatingKVCache::new(WINDOW);

    // Fresh 40-token prefill, longer than the window: every key is kept
    // (#401/#408) and the band is applied over all of them.
    let (q, k, v) = rotating_step(&mut rng, 40, HEADS, KV, DIM);
    let (k2, v2) = (ffi::copy(&k), ffi::copy(&v));
    let a = to_vec_f32(&got.attend(&q, k, v, SCALE, None));
    let (ck, cv) = want.update_and_fetch(k2, v2);
    let mask = utils::create_sliding_window_prefill_mask(40, 0, WINDOW);
    let b = to_vec_f32(&crate::layers::attention(
        &q,
        &ck,
        &cv,
        SCALE,
        Some(mask.as_ref().unwrap()),
        0.0,
        WINDOW,
    ));
    assert!(
        a.iter().all(|x| x.is_finite()),
        "fresh prefill has NaN rows"
    );
    let err = max_abs_diff(&a, &b);
    assert!(err <= tol, "fresh over-window prefill differs: {err}");
    assert_eq!(
        (got.offset, got.visible_len()),
        (want.offset, want.visible_len())
    );

    // A 24-token continuation after the ring rolled over: the cache returns
    // W-1 prior keys plus the new ones (#678) and the mask is sized the same.
    let live = got.visible_len();
    let (q, k, v) = rotating_step(&mut rng, 24, HEADS, KV, DIM);
    let (k2, v2) = (ffi::copy(&k), ffi::copy(&v));
    let a = to_vec_f32(&got.attend(&q, k, v, SCALE, None));
    let (ck, cv) = want.update_and_fetch(k2, v2);
    assert_eq!(
        ffi::array_shape(&ck)[2],
        rotating_prefill_key_len(24, live, WINDOW),
        "the key length the plan assumes is what the ring returned"
    );
    let mask = utils::create_sliding_window_prefill_mask(24, live, WINDOW);
    let b = to_vec_f32(&crate::layers::attention(
        &q,
        &ck,
        &cv,
        SCALE,
        Some(mask.as_ref().unwrap()),
        0.0,
        WINDOW,
    ));
    assert!(a.iter().all(|x| x.is_finite()), "continuation has NaN rows");
    let err = max_abs_diff(&a, &b);
    assert!(err <= tol, "continuation chunk differs: {err}");
    assert_eq!(
        (got.offset, got.visible_len()),
        (want.offset, want.visible_len())
    );

    // Decode: single-query steps keep the existing truncated-window maskless
    // path on both sides.
    for i in 0..20 {
        let (q, k, v) = rotating_step(&mut rng, 1, HEADS, KV, DIM);
        let (k2, v2) = (ffi::copy(&k), ffi::copy(&v));
        let a = to_vec_f32(&got.attend(&q, k, v, SCALE, None));
        let (ck, cv) = want.update_and_fetch(k2, v2);
        assert!(
            ffi::array_shape(&ck)[2] <= WINDOW,
            "decode keys exceed the window"
        );
        let b = to_vec_f32(&crate::layers::attention(
            &q, &ck, &cv, SCALE, None, 0.0, WINDOW,
        ));
        assert_eq!(a, b, "decode step {i} differs");
        assert_eq!(
            (got.offset, got.visible_len()),
            (want.offset, want.visible_len()),
            "cursors at {i}"
        );
    }

    // Trim (speculative rollback) then another multi-token step still lines
    // the band up with what the ring returns.
    got.trim(3);
    want.trim(3);
    let live = got.visible_len();
    let (q, k, v) = rotating_step(&mut rng, 12, HEADS, KV, DIM);
    let (k2, v2) = (ffi::copy(&k), ffi::copy(&v));
    let a = to_vec_f32(&got.attend(&q, k, v, SCALE, None));
    let (ck, cv) = want.update_and_fetch(k2, v2);
    let mask = utils::create_sliding_window_prefill_mask(12, live, WINDOW);
    let b = to_vec_f32(&crate::layers::attention(
        &q,
        &ck,
        &cv,
        SCALE,
        Some(mask.as_ref().unwrap()),
        0.0,
        WINDOW,
    ));
    let err = max_abs_diff(&a, &b);
    assert!(err <= tol, "post-trim chunk differs: {err}");
    assert_eq!(
        (got.offset, got.visible_len()),
        (want.offset, want.visible_len())
    );
}
