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

//! Regression tests for Jamba conv_state contiguous fix.

use mlxcel_core::{dtype, generate::ModelStateSnapshot, layers::KVCache, utils::slice_axis};

/// Simulate 50 decode steps of conv-state update and assert the stored shape
/// stays at [B=1, k-1=3, channels=8] regardless of how many steps we run.
///
/// Before the fix, each step stored a lazy slice aliasing the growing
/// `padded_input` concat graph, leaking memory proportional to step count.
/// After the fix, contiguous() materializes a compact [1, 3, 8] buffer each time.
#[test]
#[ignore = "requires serial MLX execution"]
fn jamba_conv_state_shape_plateaus_after_50_steps() {
    let batch = 1i32;
    let channels = 8i32;
    let k = 4usize; // conv_kernel_size (mamba_d_conv)
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

/// Verify that JambaMambaCache::new() starts with no conv_state or ssm_state.
#[test]
fn jamba_cache_new_has_no_state() {
    let cache = super::JambaMambaCache::new();
    assert!(cache.conv_state.is_none());
    assert!(cache.ssm_state.is_none());
}

/// Verify that JambaMambaCache::default() starts with no state.
#[test]
fn jamba_cache_default_has_no_state() {
    let cache = super::JambaMambaCache::default();
    assert!(cache.conv_state.is_none());
    assert!(cache.ssm_state.is_none());
}

#[test]
fn jamba_mamba_cache_snapshot_restore_round_trips_state_shapes() {
    let mut cache = super::JambaMambaCache::new();
    cache.conv_state = Some(mlxcel_core::zeros(&[1, 3, 8], dtype::FLOAT32));
    cache.ssm_state = Some(mlxcel_core::zeros(&[1, 2, 4, 8], dtype::FLOAT32));

    let mut snapshot = ModelStateSnapshot::new("jamba", 11);
    cache.snapshot_into(&mut snapshot, "layer1.mamba");

    let mut restored = super::JambaMambaCache::new();
    restored.restore_from(&snapshot, "layer1.mamba");

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

#[test]
fn jamba_attention_layer_snapshot_restore_round_trips_kv_state() {
    let mut kv = KVCache::new();
    kv.keys = Some(mlxcel_core::zeros(&[1, 2, 11, 4], dtype::FLOAT32));
    kv.values = Some(mlxcel_core::zeros(&[1, 2, 11, 4], dtype::FLOAT32));
    kv.offset = 11;
    let cache = super::JambaLayerCache::Attention(kv);

    let mut snapshot = ModelStateSnapshot::new("jamba", 11);
    cache.snapshot_into(&mut snapshot, "layer0");

    let mut restored = super::JambaLayerCache::Attention(KVCache::new());
    restored.restore_from(&snapshot, "layer0");

    match restored {
        super::JambaLayerCache::Attention(kv) => {
            let keys = kv
                .keys
                .as_ref()
                .and_then(|a| a.as_ref())
                .expect("keys restored");
            let values = kv
                .values
                .as_ref()
                .and_then(|a| a.as_ref())
                .expect("values restored");
            assert_eq!(kv.offset, 11);
            assert_eq!(mlxcel_core::array_shape(keys), vec![1, 2, 11, 4]);
            assert_eq!(mlxcel_core::array_shape(values), vec![1, 2, 11, 4]);
        }
        super::JambaLayerCache::Mamba(_) => panic!("restored wrong cache variant"),
    }
}
