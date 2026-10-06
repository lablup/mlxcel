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

//! The custom-kernel gates decline on the CPU device of a GPU build
//! (lablup/mlxcel#2069, #2063).
//!
//! Custom kernels (`fast::metal_kernel`, `fast::hip_kernel`) run only on the
//! GPU stream; on a CPU stream their `eval_cpu` throws "Custom kernels only
//! run on GPU". `MLXCEL_DEVICE=cpu` moves the default device to the CPU, so
//! every predicate a model consults before launching one must also answer
//! false there, or the model throws where it should take its graph path.
//! The predicates covered here once read only the port table, which says
//! nothing about the device.
//!
//! A separate test binary, because the test moves the process-global default
//! device: the shared lib test binaries must not see it move. It holds
//! `streams::lock_default_device` for its whole body as well.

use mlxcel_core::layers::{LayerNorm, residual_add3_layer_norm};
use mlxcel_core::streams::{DefaultDeviceGuard, lock_default_device};
use mlxcel_core::{MlxArray, dtype};

fn values(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 37 + seed * 11) % 23) as f32 - 11.0) / 3.0)
        .collect()
}

fn to_f32_vec(arr: &MlxArray) -> Vec<f32> {
    let as_f32 = mlxcel_core::astype(arr, dtype::FLOAT32);
    mlxcel_core::eval(&as_f32);
    mlxcel_core::array_to_raw_bytes(&as_f32)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn custom_kernel_gates_decline_on_the_cpu_device() {
    let _device = lock_default_device();
    let _cpu = DefaultDeviceGuard::cpu();

    assert!(
        !mlxcel_core::fused_xielu_kernel_available(),
        "fused xIELU must take its elementwise fallback on the CPU device"
    );
    assert!(
        !mlxcel_core::fused_add3_layer_norm_available(),
        "residual_add3_layer_norm must run the unfused pair on the CPU device"
    );
    assert!(
        !mlxcel_core::mamba1_scan_float_state_kernel_available(),
        "Mamba must take the graph scan on the CPU device"
    );
    assert!(
        !mlxcel_core::fused_moe_kernels_available(),
        "SwitchGLU must take gather_qmm on the CPU device"
    );
    assert!(
        !mlxcel_core::moe_down_kernel_available(),
        "Nemotron-H must take forward_nonfused on the CPU device"
    );
    assert!(
        !mlxcel_core::fused_moe_relu2_kernels_available(),
        "MLXCEL_FUSED_MOE_RELU2 must decline to gather_qmm on the CPU device"
    );
    assert!(
        !mlxcel_core::fused_add_rms_norm_available(),
        "MLXCEL_FUSED_ADD_RMSNORM=1 must take graph_add_rms_norm on the CPU device"
    );
    assert!(
        !mlxcel_core::fused_rope_qk_append_available(),
        "MLXCEL_FUSED_ROPE_APPEND=1 must take the fast_rope graph on the CPU device"
    );

    // The entry points behind the first two gates run on the CPU instead of
    // throwing. fused_xielu is held to the scalar formula it implements.
    let (alpha_p, alpha_n, beta, eps) = (0.8731f32, 0.6042f32, 0.5f32, -1e-6f32);
    let vals = values(256, 1);
    let x = mlxcel_core::from_slice_f32(&vals, &[1, 2, 128]);
    let out = mlxcel_core::fused_xielu(&x, alpha_p, alpha_n, beta, eps)
        .expect("fused_xielu falls back rather than refusing");
    for (got, &v) in to_f32_vec(&out).iter().zip(&vals) {
        let want = if v > 0.0 {
            alpha_p * v * v + beta * v
        } else {
            (v.min(eps).exp_m1() - v) * alpha_n + beta * v
        };
        assert!(
            (got - want).abs() <= 1e-5 * want.abs().max(1.0),
            "fused_xielu on the CPU at x={v}: got {got}, want {want}"
        );
    }

    let dim = 96;
    let shape = [1, 3, dim];
    let arr = |seed: usize, len: usize, shape: &[i32]| {
        let a = mlxcel_core::from_slice_f32(&values(len, seed), shape);
        mlxcel_core::astype(&a, dtype::BFLOAT16)
    };
    let len = (3 * dim) as usize;
    let (a, b, x) = (
        arr(2, len, &shape),
        arr(3, len, &shape),
        arr(4, len, &shape),
    );
    let norm = LayerNorm::new(
        arr(5, dim as usize, &[dim]),
        Some(arr(6, dim as usize, &[dim])),
        1e-5,
    );
    let (x_new, h) = residual_add3_layer_norm(&a, &b, &x, &norm);
    let x_ref = mlxcel_core::compiled_add3(&a, &b, &x);
    let h_ref = norm.forward(&x_ref);
    assert!(mlxcel_core::item_bool(&mlxcel_core::array_equal(
        &x_new, &x_ref, false
    )));
    assert!(mlxcel_core::item_bool(&mlxcel_core::array_equal(
        &h, &h_ref, false
    )));
}
