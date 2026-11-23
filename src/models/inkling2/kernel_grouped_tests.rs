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

//! Grouped expert kernel parity and gate tests (issue #3359), after mlx-vlm's
//! `test_inkling2_grouped_experts.py`. Helpers are shared with
//! `kernel_tests.rs`.

use mlxcel_core::cache::DecodeLookaheadAppendScope;
use mlxcel_core::streams::{DefaultDeviceGuard, lock_default_device};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::tests::{Rng, assert_bits, metal};
use super::*;
use crate::models::switch_layers::{SwitchGLU, SwitchGluActivation};

const HIDDEN: i32 = 2304;
const INTER: i32 = 1152;
const EXPERTS: i32 = 8;

fn plane(rng: &mut Rng, w: &mut WeightMap, key: &str, out: i32, input: i32, mode: &str) {
    let (gs, bits) = if mode == "nvfp4" { (16, 4) } else { (32, 8) };
    let values = rng.array(&[EXPERTS, out, input], 0.05, dtype::FLOAT32);
    let q = mlxcel_core::quantize_weights_with_mode(&values, gs, bits, mode);
    w.insert(
        format!("{key}.weight"),
        mlxcel_core::quantized_weights_w(&q),
    );
    w.insert(
        format!("{key}.scales"),
        mlxcel_core::quantized_weights_scales(&q),
    );
    if mode == "nvfp4" {
        // Distinct per-expert MLX-native tensor scales (`g / 2688` per row).
        let g: Vec<f32> = (0..EXPERTS)
            .map(|e| 2688.0 * (0.5 + e as f32 / 8.0))
            .collect();
        w.insert(
            format!("{key}.native_global_scale"),
            mlxcel_core::from_slice_f32(&g, &[EXPERTS]),
        );
    }
}

fn experts(mode: &str, seed: u64) -> SwitchGLU {
    let mut rng = Rng(seed);
    let mut w = WeightMap::new();
    plane(&mut rng, &mut w, "e.gate_proj", INTER, HIDDEN, mode);
    plane(&mut rng, &mut w, "e.up_proj", INTER, HIDDEN, mode);
    plane(&mut rng, &mut w, "e.down_proj", HIDDEN, INTER, mode);
    let (gs, bits) = if mode == "nvfp4" { (16, 4) } else { (32, 8) };
    SwitchGLU::from_weights_with_mode(&w, "e", gs, bits, mode)
        .unwrap()
        .with_activation(SwitchGluActivation::SiTU {
            beta: 4.0,
            linear_beta: Some(25.0),
        })
}

/// Expert 5 on every token (two tiles of four), the others in runs of one
/// to three.
fn route_set() -> UniquePtr<MlxArray> {
    let second = [0, 1, 1, 2, 2, 2, 3, 4];
    let ids: Vec<u32> = second.iter().flat_map(|&e| [5, e]).collect();
    mlxcel_core::from_slice_u32(&ids, &[8, 2])
}

#[test]
fn grouped_experts_match_per_token_switch_glu_bitwise() {
    let _lock = lock_default_device();
    if !metal() {
        return;
    }
    let mut rng = Rng(31);
    for (mode, seed) in [("nvfp4", 32), ("mxfp8", 33)] {
        let glu = experts(mode, seed);
        assert!(grouped_experts_supported(&glu), "{mode}");
        let x = rng.array(&[8, HIDDEN], 1.0, dtype::BFLOAT16);
        let indices = route_set();
        let got = grouped_expert_projections(&glu, &x, &indices).expect(mode);
        let per_token: Vec<_> = (0..8)
            .map(|t| {
                glu.forward(
                    &mlxcel_core::utils::slice_axis(&x, 0, t, t + 1),
                    &mlxcel_core::utils::slice_axis(&indices, 0, t, t + 1),
                )
            })
            .collect();
        let refs: Vec<_> = per_token.iter().map(|a| &**a).collect();
        let want = mlxcel_core::concatenate_many(&refs, 0);
        assert_bits(&got, &want, &format!("{mode}: per-token SwitchGLU"));
        assert_bits(
            &got,
            &glu.forward(&x, &indices),
            &format!("{mode}: batched gather"),
        );
    }
}

#[test]
fn grouped_experts_refuse_affine_planes() {
    let mut rng = Rng(34);
    let mut w = WeightMap::new();
    for (name, out, input) in [
        ("gate_proj", 64, 128),
        ("up_proj", 64, 128),
        ("down_proj", 128, 64),
    ] {
        let values = rng.array(&[2, out, input], 0.05, dtype::FLOAT32);
        let q = mlxcel_core::quantize_weights(&values, 64, 4);
        w.insert(
            format!("a.{name}.weight"),
            mlxcel_core::quantized_weights_w(&q),
        );
        w.insert(
            format!("a.{name}.scales"),
            mlxcel_core::quantized_weights_scales(&q),
        );
        w.insert(
            format!("a.{name}.biases"),
            mlxcel_core::quantized_weights_biases(&q),
        );
    }
    let affine = SwitchGLU::from_weights(&w, "a", 64, 4).unwrap();
    assert!(!grouped_experts_supported(&affine));
}

#[test]
fn grouped_experts_run_only_inside_a_short_verify_forward() {
    let _lock = lock_default_device();
    let mut rng = Rng(35);
    let x = |rng: &mut Rng, b: i32, l: i32, dt: i32| rng.array(&[b, l, HIDDEN], 1.0, dt);
    // Not a verify forward: plain decode and prefill never take it.
    assert!(!grouped_experts_apply(
        &x(&mut rng, 1, 4, dtype::BFLOAT16),
        8
    ));
    let _speculative = DecodeLookaheadAppendScope::enter();
    assert_eq!(
        grouped_experts_apply(&x(&mut rng, 1, 4, dtype::BFLOAT16), 8),
        mlxcel_core::inkling2_kernels::available()
    );
    for (b, l, dt, top_k) in [
        (2, 4, dtype::BFLOAT16, 8),
        (1, 1, dtype::BFLOAT16, 8),
        (1, 9, dtype::BFLOAT16, 8),
        (1, 4, dtype::FLOAT16, 8),
        (1, 4, dtype::BFLOAT16, 64),
    ] {
        assert!(
            !grouped_experts_apply(&x(&mut rng, b, l, dt), top_k),
            "B {b} L {l} dtype {dt} top_k {top_k}"
        );
    }
    let probe = x(&mut rng, 1, 4, dtype::BFLOAT16);
    let _cpu = DefaultDeviceGuard::cpu();
    assert!(!grouped_experts_apply(&probe, 8), "CPU");
}
