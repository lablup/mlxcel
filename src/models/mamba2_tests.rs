// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Regression tests for Mamba2 conv_state contiguous fix.

use mlxcel_core::{dtype, generate::ModelStateSnapshot, utils::slice_axis};

/// Simulate 50 decode steps of conv-state update and assert the stored shape
/// stays at [B=1, k-1=3, channels=8] regardless of how many steps we run.
///
/// Before the fix, each step would store a slice that aliased the growing
/// `padded_input` accumulation graph, leaking memory proportional to step count.
/// After the fix, contiguous() materializes a compact [1, 3, 8] buffer each time.
#[test]
#[ignore = "requires serial MLX execution"]
fn mamba2_conv_state_shape_plateaus_after_50_steps() {
    let batch = 1i32;
    let channels = 8i32;
    let k = 4usize; // conv_kernel_size
    let n_keep = (k - 1) as i32; // = 3
    let expected_shape = vec![batch, n_keep, channels];

    let mut conv_state =
        mlxcel_core::zeros(&[batch, n_keep, channels], mlxcel_core::dtype::FLOAT32);

    for _step in 0..50 {
        let new_token = mlxcel_core::zeros(&[batch, 1, channels], mlxcel_core::dtype::FLOAT32);

        // Build padded_input = concat(conv_state, new_token, axis=1) -> [1, k, channels]
        let padded_input = mlxcel_core::concatenate(&conv_state, &new_token, 1);

        let padded_shape = mlxcel_core::array_shape(&padded_input);
        let len = padded_shape[1] as usize;

        // Apply the fixed conv-state update: slice then contiguous
        let tail = slice_axis(&padded_input, 1, (len - (k - 1)) as i32, len as i32);
        conv_state = mlxcel_core::contiguous(&tail, false);

        mlxcel_core::eval(&conv_state);

        let shape = mlxcel_core::array_shape(&conv_state);
        assert_eq!(
            shape, expected_shape,
            "step {_step}: conv_state shape {shape:?} != expected {expected_shape:?}"
        );
    }
}

/// Verify that Mamba2Cache::new() starts with no conv_state.
#[test]
fn mamba2_cache_new_has_no_state() {
    let cache = super::Mamba2Cache::new();
    assert!(cache.conv_state.is_none());
    assert!(cache.ssm_state.is_none());
}

/// Verify that Mamba2Cache::default() starts with no conv_state.
#[test]
fn mamba2_cache_default_has_no_state() {
    let cache = super::Mamba2Cache::default();
    assert!(cache.conv_state.is_none());
    assert!(cache.ssm_state.is_none());
}

#[test]
fn mamba2_cache_snapshot_restore_round_trips_state_shapes() {
    let mut cache = super::Mamba2Cache::new();
    cache.conv_state = Some(mlxcel_core::zeros(&[1, 3, 8], dtype::FLOAT32));
    cache.ssm_state = Some(mlxcel_core::zeros(&[1, 2, 4, 8], dtype::FLOAT32));

    let mut snapshot = ModelStateSnapshot::new("mamba2", 9);
    cache.snapshot_into(&mut snapshot, "layer0");

    let mut restored = super::Mamba2Cache::new();
    restored.restore_from(&snapshot, "layer0");

    let conv = restored
        .conv_state
        .as_ref()
        .and_then(|a| a.as_ref())
        .expect("conv_state restored");
    let ssm = restored
        .ssm_state
        .as_ref()
        .and_then(|a| a.as_ref())
        .expect("ssm_state restored");
    assert_eq!(mlxcel_core::array_shape(conv), vec![1, 3, 8]);
    assert_eq!(mlxcel_core::array_shape(ssm), vec![1, 2, 4, 8]);
}
