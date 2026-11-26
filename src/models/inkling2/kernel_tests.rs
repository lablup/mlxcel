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

//! Kernel-versus-ops parity and gate tests for the Inkling 2 Mini kernels
//! (issue #3359), after mlx-vlm's `test_inkling_kernel_gates.py`,
//! and the sconv cases of `test_speculative.py` (the grouped expert cases are
//! in `kernel_grouped_tests.rs`). Parity is bitwise: bf16 outputs are compared as
//! `u16` views, f32 outputs by their bits. Cases that need the Metal port
//! return early where it is missing (the gate tests then check the refusal).

use mlxcel_core::cache::DecodeLookaheadAppendScope;
use mlxcel_core::streams::{DefaultDeviceGuard, lock_default_device};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use serde_json::json;

use super::super::attention::l2_clip;
use super::super::config::Inkling2Config;
use super::super::conv::{CONV_HISTORY_MARGIN, ShortConv, rewind_history};
use super::*;

pub(super) struct Rng(pub(super) u64);

impl Rng {
    pub(super) fn values(&mut self, count: i32, scale: f32) -> Vec<f32> {
        (0..count)
            .map(|_| {
                self.0 = self
                    .0
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
            })
            .collect()
    }

    pub(super) fn array(&mut self, shape: &[i32], scale: f32, dt: i32) -> UniquePtr<MlxArray> {
        let values = self.values(shape.iter().product(), scale);
        let a = mlxcel_core::from_slice_f32(&values, shape);
        if dt == dtype::FLOAT32 {
            a
        } else {
            mlxcel_core::astype(&a, dt)
        }
    }
}

pub(super) fn metal() -> bool {
    let ok = mlxcel_core::inkling2_kernels::available();
    if !ok {
        eprintln!("skipping: no Inkling 2 Metal port on this backend");
    }
    ok
}

/// The exact bits of `a`, with its shape and dtype.
fn bits(a: &MlxArray) -> (Vec<i32>, i32, Vec<u32>) {
    let dt = mlxcel_core::array_dtype(a);
    let raw: Vec<u32> = match dt {
        d if d == dtype::FLOAT32 => mlxcel_core::utils::array_to_vec_f32(a)
            .iter()
            .map(|v| v.to_bits())
            .collect(),
        _ => {
            let view = mlxcel_core::view(&mlxcel_core::contiguous(a, false), dtype::UINT16);
            mlxcel_core::utils::array_to_vec_f32(&view)
                .iter()
                .map(|v| *v as u32)
                .collect()
        }
    };
    (mlxcel_core::array_shape(a), dt, raw)
}

pub(super) fn assert_bits(a: &MlxArray, b: &MlxArray, what: &str) {
    let (sa, da, va) = bits(a);
    let (sb, db, vb) = bits(b);
    assert_eq!((sa, da), (sb, db), "{what}: shape or dtype");
    let diff = va.iter().zip(&vb).filter(|(x, y)| x != y).count();
    assert_eq!(diff, 0, "{what}: {diff} of {} elements differ", va.len());
}

// ---------------------------------------------------------------- sconv

const C: i32 = 256;

fn sconv(rng: &mut Rng, kernel: i32) -> ShortConv {
    let mut w = WeightMap::new();
    w.insert(
        "s.conv.weight".to_string(),
        rng.array(&[C, kernel, 1], 0.5, dtype::BFLOAT16),
    );
    ShortConv::from_weights(&w, "s", C as usize).unwrap()
}

/// A `[B, L, C]` bf16 input read through transposed strides.
fn strided_x(rng: &mut Rng, b: i32, l: i32) -> UniquePtr<MlxArray> {
    let x = rng.array(&[b, C, l], 1.0, dtype::BFLOAT16);
    mlxcel_core::transpose_axes(&x, &[0, 2, 1])
}

/// A `[B, rows, C]` f32 history that is a slice of a larger buffer.
fn strided_history(rng: &mut Rng, b: i32, rows: i32) -> UniquePtr<MlxArray> {
    let full = rng.array(&[b, rows + 5, C], 1.0, dtype::FLOAT32);
    mlxcel_core::utils::slice_axis(&full, 1, 2, rows + 2)
}

#[test]
fn kill_switch_parses_off_values() {
    for off in ["0", "false", " OFF ", "no"] {
        assert!(!enabled_from(Some(off)), "{off}");
    }
    for on in [None, Some("1"), Some("true"), Some("")] {
        assert!(enabled_from(on), "{on:?}");
    }
}

#[test]
fn short_convolution_matches_the_ops_path_bitwise() {
    let _lock = lock_default_device();
    if !metal() {
        return;
    }
    let mut rng = Rng(11);
    let conv = sconv(&mut rng, 4);
    for (b, l, rows) in [(1, 1, 3), (1, 8, 19), (4, 5, 7), (2, 3, 0)] {
        for output in [dtype::FLOAT32, dtype::BFLOAT16] {
            for residual_dtype in [None, Some(dtype::BFLOAT16), Some(dtype::FLOAT32)] {
                for strided in [false, true] {
                    let x = if strided {
                        strided_x(&mut rng, b, l)
                    } else {
                        rng.array(&[b, l, C], 1.0, dtype::BFLOAT16)
                    };
                    let history = match (rows, strided) {
                        (0, _) => None,
                        (_, true) => Some(strided_history(&mut rng, b, rows)),
                        (_, false) => Some(rng.array(&[b, rows, C], 1.0, dtype::FLOAT32)),
                    };
                    let residual = residual_dtype.map(|dt| {
                        let r = rng.array(&[b, C, l], 2.0, dt);
                        mlxcel_core::transpose_axes(&r, &[0, 2, 1])
                    });
                    let copy =
                        |h: &Option<UniquePtr<MlxArray>>| h.as_ref().map(|h| mlxcel_core::copy(h));
                    let (mut ops_state, mut kernel_state) = (copy(&history), copy(&history));
                    let want = conv.forward_ops(&x, &mut ops_state, residual.as_deref(), output);
                    let got = conv
                        .forward_kernel(&x, &mut kernel_state, residual.as_deref(), output)
                        .expect("the gate admits this case");
                    let what = format!(
                        "B {b} L {l} rows {rows} out {output} residual {residual_dtype:?} strided {strided}"
                    );
                    assert_bits(&got, &want, &what);
                    assert_bits(
                        kernel_state.as_ref().unwrap(),
                        ops_state.as_ref().unwrap(),
                        &format!("{what}: next history"),
                    );
                }
            }
        }
    }
}

#[test]
fn short_convolution_gate_refuses_outside_its_shapes() {
    let _lock = lock_default_device();
    let mut rng = Rng(12);
    let conv = sconv(&mut rng, 4);
    let refused = |x: &MlxArray, conv: &ShortConv| {
        let mut state = None;
        let out = conv.forward_kernel(x, &mut state, None, dtype::FLOAT32);
        out.is_none() && state.is_none()
    };
    // L = 9, B = 5, f16 and f32 inputs, a width that is not a multiple of 128.
    assert!(refused(&rng.array(&[1, 9, C], 1.0, dtype::BFLOAT16), &conv));
    assert!(refused(&rng.array(&[5, 1, C], 1.0, dtype::BFLOAT16), &conv));
    assert!(refused(&rng.array(&[1, 1, C], 1.0, dtype::FLOAT16), &conv));
    assert!(refused(&rng.array(&[1, 1, C], 1.0, dtype::FLOAT32), &conv));
    let mut w = WeightMap::new();
    w.insert(
        "n.conv.weight".to_string(),
        rng.array(&[192, 4, 1], 0.5, dtype::BFLOAT16),
    );
    let narrow = ShortConv::from_weights(&w, "n", 192).unwrap();
    assert!(refused(
        &rng.array(&[1, 1, 192], 1.0, dtype::BFLOAT16),
        &narrow
    ));
    // Kernel size 3.
    assert!(refused(
        &rng.array(&[1, 1, C], 1.0, dtype::BFLOAT16),
        &sconv(&mut rng, 3)
    ));
    // An f16 history.
    let mut f16_state = Some(rng.array(&[1, 3, C], 1.0, dtype::FLOAT16));
    let x = rng.array(&[1, 1, C], 1.0, dtype::BFLOAT16);
    assert!(
        conv.forward_kernel(&x, &mut f16_state, None, dtype::FLOAT32)
            .is_none()
    );
    // Off the GPU.
    let _cpu = DefaultDeviceGuard::cpu();
    assert!(refused(&x, &conv));
}

/// After a speculative block, rewinding to keep 0, 2, 4 or all of its rows
/// leaves the kernel's history equal to the ops path's, and a later step on
/// each agrees bitwise.
#[test]
fn short_convolution_history_rewinds_like_the_ops_path() {
    let _lock = lock_default_device();
    if !metal() {
        return;
    }
    let mut rng = Rng(13);
    let conv = sconv(&mut rng, 4);
    let prefix = rng.array(&[1, 6, C], 1.0, dtype::BFLOAT16);
    let block = rng.array(&[1, 5, C], 1.0, dtype::BFLOAT16);
    let next = rng.array(&[1, 1, C], 1.0, dtype::BFLOAT16);
    for keep in [0, 2, 4, 5] {
        let mut ops = None;
        let mut fused = None;
        let _ = conv.forward_ops(&prefix, &mut ops, None, dtype::FLOAT32);
        let _ = conv.forward_kernel(&prefix, &mut fused, None, dtype::FLOAT32);
        {
            let _speculative = DecodeLookaheadAppendScope::enter();
            let _ = conv.forward_ops(&block, &mut ops, None, dtype::FLOAT32);
            let _ = conv.forward_kernel(&block, &mut fused, None, dtype::FLOAT32);
        }
        let trim = 5 - keep;
        let ops_h = rewind_history(ops.as_ref().unwrap(), trim, 4).unwrap();
        let fused_h = rewind_history(fused.as_ref().unwrap(), trim, 4).unwrap();
        assert_bits(&fused_h, &ops_h, &format!("keep {keep}: history"));
        let (mut ops, mut fused) = (Some(ops_h), Some(fused_h));
        let a = conv.forward_ops(&next, &mut ops, None, dtype::BFLOAT16);
        let b = conv
            .forward_kernel(&next, &mut fused, None, dtype::BFLOAT16)
            .unwrap();
        assert_bits(&b, &a, &format!("keep {keep}: next step"));
        assert_bits(
            fused.as_ref().unwrap(),
            ops.as_ref().unwrap(),
            &format!("keep {keep}: history after the next step"),
        );
    }
}

/// A verify block longer than the margin keeps enough rows to drop it whole.
#[test]
fn a_long_verify_block_can_be_rewound_whole() {
    let _lock = lock_default_device();
    let mut rng = Rng(14);
    let conv = sconv(&mut rng, 4);
    let mut state = None;
    let _ = conv.forward_ops(
        &rng.array(&[1, 30, C], 1.0, dtype::FLOAT32),
        &mut state,
        None,
        dtype::FLOAT32,
    );
    let block = CONV_HISTORY_MARGIN + 9;
    {
        let _speculative = DecodeLookaheadAppendScope::enter();
        let _ = conv.forward_ops(
            &rng.array(&[1, block, C], 1.0, dtype::FLOAT32),
            &mut state,
            None,
            dtype::FLOAT32,
        );
    }
    assert!(rewind_history(state.as_ref().unwrap(), block, 4).is_ok());
}

// ------------------------------------------------------------ head prep

struct Heads {
    q: UniquePtr<MlxArray>,
    k: UniquePtr<MlxArray>,
    v: UniquePtr<MlxArray>,
    r: UniquePtr<MlxArray>,
    q_scale: UniquePtr<MlxArray>,
    r_scale: UniquePtr<MlxArray>,
}

const CLIP: f32 = 8.0;
const REL_CLIP: f32 = 2.0;
const EPS: f32 = 1e-6;

fn heads(rng: &mut Rng, b: i32, l: i32, scale: f32, dim: i32, rel: i32, dt: i32) -> Heads {
    // q and r are views with permuted strides; k and v are row-major.
    let q = rng.array(&[b, 24, l, dim], scale, dt);
    let r = rng.array(&[b, 24, l, rel], scale, dt);
    Heads {
        q: mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]),
        k: rng.array(&[b, l, 8, dim], scale, dt),
        v: rng.array(&[b, l, 8, dim], scale, dt),
        r: mlxcel_core::transpose_axes(&r, &[0, 2, 1, 3]),
        q_scale: rng.array(&[24, dim], 1.0, dtype::FLOAT32),
        r_scale: rng.array(&[24, rel], 1.0, dtype::FLOAT32),
    }
}

fn heads_kernel(h: &Heads) -> Option<[UniquePtr<MlxArray>; 4]> {
    prepare_attention_heads(
        &h.q, &h.k, &h.v, &h.r, &h.q_scale, &h.r_scale, CLIP, REL_CLIP, EPS,
    )
}

fn heads_ops(h: &Heads) -> [UniquePtr<MlxArray>; 4] {
    let v = mlxcel_core::astype(
        &mlxcel_core::fast_rms_norm_no_weight(&mlxcel_core::astype(&h.v, dtype::FLOAT32), EPS),
        dtype::BFLOAT16,
    );
    [
        l2_clip(&h.q, CLIP, Some(&h.q_scale)),
        l2_clip(&h.k, CLIP, None),
        v,
        l2_clip(&h.r, REL_CLIP, Some(&h.r_scale)),
    ]
}

#[test]
fn head_preparation_matches_the_ops_path_bitwise() {
    let _lock = lock_default_device();
    if !metal() {
        return;
    }
    let mut rng = Rng(21);
    for l in [1, 4, 8] {
        for scale in [0.25, 5.0, 1.0] {
            let h = heads(&mut rng, 1, l, scale, 128, 16, dtype::BFLOAT16);
            let got = heads_kernel(&h).expect("the gate admits this case");
            for (name, (a, b)) in ["q", "k", "v", "r"]
                .iter()
                .zip(got.iter().zip(&heads_ops(&h)))
            {
                assert_bits(a, b, &format!("L {l} scale {scale}: {name}"));
            }
        }
    }
}

#[test]
fn head_preparation_propagates_a_nan_like_the_ops_path() {
    let _lock = lock_default_device();
    if !metal() {
        return;
    }
    let mut rng = Rng(22);
    let mut h = heads(&mut rng, 1, 8, 1.0, 128, 16, dtype::BFLOAT16);
    let poison = |x: &MlxArray, at: usize| {
        let mut values = mlxcel_core::utils::array_to_vec_f32(x);
        values[at] = f32::NAN;
        mlxcel_core::astype(
            &mlxcel_core::from_slice_f32(&values, &mlxcel_core::array_shape(x)),
            dtype::BFLOAT16,
        )
    };
    h.q = poison(&h.q, 7 * 24 * 128 + 3 * 128 + 5);
    h.k = poison(&h.k, 2 * 8 * 128 + 70);
    h.v = poison(&h.v, 5 * 8 * 128 + 1);
    h.r = poison(&h.r, 7 * 24 * 16 + 9);
    let got = heads_kernel(&h).unwrap();
    for (name, (a, b)) in ["q", "k", "v", "r"]
        .iter()
        .zip(got.iter().zip(&heads_ops(&h)))
    {
        assert_bits(a, b, &format!("NaN: {name}"));
    }
}

#[test]
fn head_preparation_gate_refuses_outside_its_shapes() {
    let _lock = lock_default_device();
    let mut rng = Rng(23);
    for (b, l, dim, rel, dt) in [
        (2, 1, 128, 16, dtype::BFLOAT16),
        (1, 9, 128, 16, dtype::BFLOAT16),
        (1, 1, 64, 16, dtype::BFLOAT16),
        (1, 1, 128, 8, dtype::BFLOAT16),
        (1, 1, 128, 16, dtype::FLOAT16),
    ] {
        let h = heads(&mut rng, b, l, 1.0, dim, rel, dt);
        assert!(
            heads_kernel(&h).is_none(),
            "B {b} L {l} dim {dim} rel {rel} dtype {dt}"
        );
    }
    let h = heads(&mut rng, 1, 1, 1.0, 128, 16, dtype::BFLOAT16);
    let _cpu = DefaultDeviceGuard::cpu();
    assert!(heads_kernel(&h).is_none(), "CPU");
}

fn text_config(looped: bool) -> Inkling2TextConfig {
    let mut text = json!({
        "model_type": "inkling2_text",
        "hidden_size": 16, "num_hidden_layers": 6, "vocab_size": 32,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
        "swa_num_attention_heads": 2, "swa_num_key_value_heads": 1, "swa_head_dim": 8,
        "sliding_window_size": 4, "local_layer_ids": [0, 1, 3, 4],
        "d_rel": 4, "rel_extent": 8, "sconv_kernel_size": 4, "intermediate_size": 8,
        "n_routed_experts": 4, "num_experts_per_tok": 2, "n_layers_per_attn_res_block": 2
    });
    if looped {
        text["model_type"] = json!("inkling2_loop_text");
        text["loop_config"] =
            json!({"core_start": 2, "core_end": 4, "num_passes": 3, "gate_rank": 2});
    }
    let model_type = if looped { "inkling2_loop" } else { "inkling2" };
    let raw = json!({"model_type": model_type, "text_config": text});
    Inkling2Config::from_json(&raw.to_string())
        .unwrap()
        .text_config
}

#[test]
fn loop_kernels_are_wanted_only_on_a_looped_model_with_kernels_on() {
    assert!(loop_kernels_wanted(&text_config(true), true));
    assert!(!loop_kernels_wanted(&text_config(true), false));
    assert!(!loop_kernels_wanted(&text_config(false), true));
}

/// Issue #3410: the grouped expert matvec serves unlooped Small too, and the
/// kill switch still turns it off.
#[test]
fn grouped_experts_are_wanted_with_or_without_a_loop() {
    assert!(grouped_experts_wanted(&text_config(true), true));
    assert!(grouped_experts_wanted(&text_config(false), true));
    assert!(!grouped_experts_wanted(&text_config(false), false));
}

/// The dispatch counter behind the `RUST_LOG` evidence of #3373 counts a
/// kernel only when it ran, not when its gate refused and the ops path did.
#[test]
fn dispatches_are_counted_only_when_a_kernel_runs() {
    let _lock = lock_default_device();
    if !metal() {
        return;
    }
    let mut rng = Rng(31);
    let conv = sconv(&mut rng, 4);
    let before = dispatch_counts();
    let x = rng.array(&[1, 1, C], 1.0, dtype::BFLOAT16);
    let mut state = None;
    assert!(
        conv.forward_kernel(&x, &mut state, None, dtype::BFLOAT16)
            .is_some()
    );
    let x = rng.array(&[1, 1, C], 1.0, dtype::FLOAT32);
    let mut state = None;
    assert!(
        conv.forward_kernel(&x, &mut state, None, dtype::FLOAT32)
            .is_none()
    );
    assert!(heads_kernel(&heads(&mut rng, 1, 1, 1.0, 128, 16, dtype::BFLOAT16)).is_some());
    assert!(heads_kernel(&heads(&mut rng, 1, 1, 1.0, 64, 16, dtype::BFLOAT16)).is_none());
    let after = dispatch_counts();
    let delta: Vec<u64> = after.iter().zip(before).map(|(a, b)| a - b).collect();
    assert_eq!(
        delta,
        [1, 1, 0],
        "short convolution, head preparation, grouped experts"
    );
}
