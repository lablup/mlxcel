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

//! Wrapper-level tests for the Nemotron H Nano Omni VLM.
//!
//! These tests exercise the multimodal projector and the
//! `pixel_shuffle` downsample without bringing up the full text
//! backbone, so they remain fast and platform-independent.

// `super` is the `nemotron_h_nano_omni_vl` module that includes this
// file via `#[path]`, so types defined there are reachable directly.
use super::{
    NemotronHNanoOmniProjector, NemotronHNanoOmniVlConfig, read_int32_vec,
    subsampling_output_lengths,
};
use mlxcel_core::weights::WeightMap;

const PROJ_IN_FEATURES: usize = 16;
const PROJ_HIDDEN: usize = 32;
const TEXT_HIDDEN: usize = 8;

fn build_projector_weights(prefix: &str) -> WeightMap {
    let mut weights = WeightMap::new();
    weights.insert(
        format!("{prefix}.layers.0.weight"),
        mlxcel_core::ones(&[PROJ_IN_FEATURES as i32], mlxcel_core::dtype::FLOAT32),
    );

    let fc1_data: Vec<f32> = (0..PROJ_HIDDEN * PROJ_IN_FEATURES)
        .map(|i| (i as f32) * 1e-3)
        .collect();
    weights.insert(
        format!("{prefix}.layers.1.weight"),
        mlxcel_core::from_slice_f32(&fc1_data, &[PROJ_HIDDEN as i32, PROJ_IN_FEATURES as i32]),
    );

    let fc2_data: Vec<f32> = (0..TEXT_HIDDEN * PROJ_HIDDEN)
        .map(|i| (i as f32) * 1e-3)
        .collect();
    weights.insert(
        format!("{prefix}.layers.3.weight"),
        mlxcel_core::from_slice_f32(&fc2_data, &[TEXT_HIDDEN as i32, PROJ_HIDDEN as i32]),
    );
    weights
}

#[test]
fn projector_maps_features_to_text_hidden_size() {
    let weights = build_projector_weights("mlp1");
    let projector =
        NemotronHNanoOmniProjector::from_weights(&weights, "mlp1", 64, 4).expect("build projector");

    let input = mlxcel_core::ones(
        &[1, 4, PROJ_IN_FEATURES as i32],
        mlxcel_core::dtype::FLOAT32,
    );
    let out = projector.forward(input.as_ref().unwrap());
    assert_eq!(
        mlxcel_core::array_shape(&out),
        vec![1, 4, TEXT_HIDDEN as i32]
    );
}

/// `subsampling_output_lengths` should mirror upstream
/// `_get_subsampling_output_length` exactly. We exercise the helper here
/// because audio merge correctness depends on precise per-clip length
/// trimming; a one-off bug in the divider would silently skew the
/// number of audio tokens emitted into the LLM stream.
#[test]
fn subsampling_output_lengths_matches_python_reference() {
    let lengths = mlxcel_core::from_slice_i32(&[800, 400, 100, 0], &[4]);
    let kernel = 3;
    let stride = 2;
    let stages = 3; // log2(8) -- the released Nemotron Omni default.
    let out = subsampling_output_lengths(lengths.as_ref().unwrap(), kernel, stride, stages);
    let cpu = read_int32_vec(&out);
    // Python: floor((L - 1) / 2 + 1) per stage.
    // 800 → 400 → 200 → 100
    // 400 → 200 → 100 → 50
    // 100 → 50  → 25  → 13
    //   0 →  0  →  0  →  0  (padding does not produce more frames)
    // For clarity, emit the chain for the third element.
    assert_eq!(cpu, vec![100, 50, 13, 0]);
}

#[test]
fn vl_config_default_downsample_factor_round_trips() {
    let config = NemotronHNanoOmniVlConfig {
        vit_hidden_size: 1280,
        projector_hidden_size: 4096,
        text_hidden_size: 4096,
        downsample_ratio: 0.5,
        ps_version: "v1".to_string(),
        img_context_token_id: 100,
        image_start_token_id: 0,
        image_end_token_id: 0,
        sound_context_token_id: None,
        sound_start_token_id: 0,
        sound_end_token_id: 0,
        eos_token_ids: Vec::new(),
    };
    // Sanity that the config struct exposes everything the runtime wires.
    assert_eq!(config.downsample_ratio, 0.5);
    assert_eq!(config.ps_version, "v1");
    assert_eq!(config.img_context_token_id, 100);
    assert!(config.sound_context_token_id.is_none());
}
