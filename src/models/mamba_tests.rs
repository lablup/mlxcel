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

//! Regression tests for Mamba conv_state contiguous fix.

use mlxcel_core::utils::slice_axis;

/// Simulate 50 decode steps of conv-state update and assert the stored shape
/// stays at [B=1, k-1=3, channels=4] regardless of how many steps we run.
///
/// Before the fix, each step would store a slice that aliased the growing
/// `padded_input` (= concat(prev_state, new_token)) accumulation graph, so the
/// effective live allocation grew linearly with step count.  After the fix,
/// contiguous() materializes a compact [1, 3, 4] buffer each time.
#[test]
#[ignore = "requires serial MLX execution"]
fn mamba_conv_state_shape_plateaus_after_50_steps() {
    let batch = 1i32;
    let channels = 4i32;
    let k = 4usize; // conv_kernel_size
    let n_keep = (k - 1) as i32; // = 3
    let expected_shape = vec![batch, n_keep, channels];

    // Initial conv_state: zeros [1, k-1, channels]
    let mut conv_state =
        mlxcel_core::zeros(&[batch, n_keep, channels], mlxcel_core::dtype::FLOAT32);

    for _step in 0..50 {
        // Simulate: new single-token input [1, 1, channels]
        let new_token = mlxcel_core::zeros(&[batch, 1, channels], mlxcel_core::dtype::FLOAT32);

        // Build padded_input = concat(conv_state, new_token, axis=1) -> [1, k, channels]
        let padded_input = mlxcel_core::concatenate(&conv_state, &new_token, 1);

        let padded_shape = mlxcel_core::array_shape(&padded_input);
        let len = padded_shape[1] as usize;

        // Apply the fixed conv-state update: slice then contiguous
        let tail = slice_axis(&padded_input, 1, (len - (k - 1)) as i32, len as i32);
        conv_state = mlxcel_core::contiguous(&tail, false);

        // Materialize so shape reflects the actual allocation
        mlxcel_core::eval(&conv_state);

        let shape = mlxcel_core::array_shape(&conv_state);
        assert_eq!(
            shape, expected_shape,
            "step {_step}: conv_state shape {shape:?} != expected {expected_shape:?}"
        );
    }
}

/// Verify that MambaCache::new() starts with no conv_state.
#[test]
fn mamba_cache_new_has_no_state() {
    let cache = super::MambaCache::new();
    assert!(cache.conv_state.is_none());
    assert!(cache.ssm_state.is_none());
}

/// Verify that MambaCache::default() starts with no conv_state.
#[test]
fn mamba_cache_default_has_no_state() {
    let cache = super::MambaCache::default();
    assert!(cache.conv_state.is_none());
    assert!(cache.ssm_state.is_none());
}
