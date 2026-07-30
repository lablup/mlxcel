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

use mlxcel_core::{dtype, generate::ModelStateSnapshot, utils::slice_axis};

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

#[test]
fn mamba_cache_snapshot_restore_round_trips_state_shapes() {
    let mut cache = super::MambaCache::new();
    cache.conv_state = Some(mlxcel_core::zeros(&[1, 3, 4], dtype::FLOAT32));
    cache.ssm_state = Some(mlxcel_core::zeros(&[1, 2, 4, 8], dtype::FLOAT32));

    let mut snapshot = ModelStateSnapshot::new("mamba", 7);
    cache.snapshot_into(&mut snapshot, "layer0");

    let mut restored = super::MambaCache::new();
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
    assert_eq!(mlxcel_core::array_shape(conv), vec![1, 3, 4]);
    assert_eq!(mlxcel_core::array_shape(ssm), vec![1, 2, 4, 8]);
}

/// Mamba takes its embedding table out of the `WeightMap` by `remove`, under
/// either the `backbone.embeddings` or the `model.embed_tokens` spelling, so it
/// cannot address the single-prefix `UnifiedEmbedding::from_weights` and never
/// reaches `reconcile_quantization_layout`. Before issue #958 the declared pair
/// was stored verbatim on a hand-built `QuantizedEmbedding` and divided by
/// inside `quantized_embedding` at the first token, which crosses the cxx
/// bridge as `UniquePtr<MlxArray>` rather than `Result` and therefore aborts the
/// process instead of failing the load.
///
/// This drives the real `MambaModel::from_weights` rather than the pure bounds
/// helper, so the guard is exercised where a checkpoint actually reaches it. The
/// assertions are on the load result and no forward pass is run, so a regression
/// fails cleanly here rather than aborting the test binary.
#[test]
fn mamba_rejects_quantization_params_that_would_abort_the_embedding_lookup() {
    use super::{MambaConfig, MambaModel};
    use mlxcel_core::weights::WeightMap;

    // `num_hidden_layers: 0` plus tied embeddings reduces the load to the
    // embedding table and the final norm, so the honest case loads all the way
    // through rather than stopping at an unrelated missing weight.
    let config_with = |group_size: i32, bits: i32| -> MambaConfig {
        serde_json::from_value(serde_json::json!({
            "model_type": "mamba",
            "vocab_size": 8,
            "hidden_size": 8,
            "intermediate_size": 16,
            "state_size": 4,
            "num_hidden_layers": 0,
            "conv_kernel": 4,
            "time_step_rank": 1,
            "tie_word_embeddings": true,
            "quantization": { "group_size": group_size, "bits": bits },
        }))
        .expect("test config must parse")
    };

    // Honest affine 4-bit geometry: packed_in * 32 == bits * groups * gs,
    // i.e. 1 * 32 == 4 * 1 * 8. The table is [vocab, packed_in].
    let build_weights = || {
        let mut weights = WeightMap::new();
        weights.insert(
            "backbone.embeddings.weight".to_string(),
            mlxcel_core::from_slice_f32(&[0.0; 8], &[8, 1]),
        );
        weights.insert(
            "backbone.embeddings.scales".to_string(),
            mlxcel_core::from_slice_f32(&[1.0; 8], &[8, 1]),
        );
        weights.insert(
            "backbone.embeddings.biases".to_string(),
            mlxcel_core::from_slice_f32(&[0.0; 8], &[8, 1]),
        );
        weights.insert(
            "backbone.norm_f.weight".to_string(),
            mlxcel_core::from_slice_f32(&[1.0; 8], &[8]),
        );
        weights
    };

    // Positive control first, so a guard that rejects everything quantized
    // cannot pass this test.
    MambaModel::from_weights(config_with(8, 4), build_weights())
        .expect("an honest affine pair must still load a quantized Mamba embedding");

    for (group_size, bits, field) in crate::models::switch_layers::HOSTILE_QUANT_PARAMS {
        let err = MambaModel::from_weights(config_with(group_size, bits), build_weights())
            .err()
            .unwrap_or_else(|| {
                panic!("Mamba must reject group_size {group_size} / bits {bits} at load")
            })
            .to_string();
        assert!(err.contains(field), "unhelpful error: {err}");
    }

    // A float embedding table carries no packing, so the params are irrelevant
    // there and must not be enforced.
    let mut regular = build_weights();
    regular.remove("backbone.embeddings.scales");
    regular.remove("backbone.embeddings.biases");
    MambaModel::from_weights(config_with(0, 0), regular)
        .expect("a float embedding table must not be gated on quantization params");
}
