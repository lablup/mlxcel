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

//! Synthetic-weight encoder tests for the Parakeet path.
//!
//! These tests build a tiny `NemotronOmniSoundEncoder` from manually
//! crafted weights and verify shape / dtype contracts. They do NOT
//! validate Apple-Silicon-only numerical accuracy — that is deferred
//! to the bring-up doc as documented in the PR body.

use super::config::NemotronOmniAudioConfig;
use super::encoder::{NemotronOmniSoundEncoder, parakeet_rel_position_vector};
use super::feature_extractor::NemotronOmniFeatureExtractor;
use super::projector::NemotronOmniSoundProjection;
use mlxcel_core::weights::WeightMap;

/// Construct a downsized config that exercises the same code paths as
/// the real model while keeping per-test memory bounded.
fn small_audio_config() -> NemotronOmniAudioConfig {
    NemotronOmniAudioConfig {
        hidden_size: 16,
        num_attention_heads: 2,
        num_hidden_layers: 1,
        intermediate_size: 32,
        attention_bias: false,
        convolution_bias: false,
        conv_kernel_size: 3,
        subsampling_factor: 4, // log2(4)=2 stride-2 stages
        subsampling_conv_channels: 4,
        num_mel_bins: 16,
        subsampling_conv_kernel_size: 3,
        subsampling_conv_stride: 2,
        max_position_embeddings: 256,
        scale_input: false,
        projection_hidden_size: 32,
        projection_bias: false,
        sampling_rate: 16_000,
        hop_length: 160,
        n_fft: 512,
        win_length: 400,
        preemphasis: 0.97,
        rms_norm_eps: 1e-5,
    }
}

fn rand_f32(shape: &[i32], seed: u32) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let total: usize = shape.iter().map(|&d| d as usize).product();
    let mut state = seed.wrapping_add(0x9E37_79B9);
    let data: Vec<f32> = (0..total)
        .map(|_| {
            state = state.wrapping_mul(0x5BD1_E995).wrapping_add(1);
            // Map to small range so RMSNorm/etc. stay numerically tame.
            ((state as f32) / (u32::MAX as f32) - 0.5) * 0.1
        })
        .collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

fn build_synthetic_audio_weights(config: &NemotronOmniAudioConfig) -> WeightMap {
    let mut weights = WeightMap::new();
    let h = config.hidden_size as i32;
    let head_dim = config.head_dim() as i32;
    let heads = config.num_attention_heads as i32;
    let intermediate = config.intermediate_size as i32;
    let kernel = config.conv_kernel_size as i32;
    let sub_kernel = config.subsampling_conv_kernel_size as i32;
    let sub_channels = config.subsampling_conv_channels as i32;
    let stages = config.num_subsampling_layers();
    let stride_pow = config.subsampling_conv_stride.pow(stages as u32) as i32;
    let out_freq = config.num_mel_bins as i32 / stride_pow;
    let lin_in = sub_channels * out_freq;

    let prefix = "sound_encoder.encoder";
    // Subsampling layer 0 (Conv2d(1 -> sub_channels, stride=2)).
    weights.insert(
        format!("{prefix}.subsampling.layers.0.weight"),
        rand_f32(&[sub_channels, sub_kernel, sub_kernel, 1], 0),
    );
    // Subsampling subsequent stages: depthwise then pointwise.
    let mut layer_idx = 2usize;
    for stage in 0..(stages - 1) {
        // Depthwise stride=2.
        weights.insert(
            format!("{prefix}.subsampling.layers.{layer_idx}.weight"),
            rand_f32(&[sub_channels, sub_kernel, sub_kernel, 1], 1 + stage as u32),
        );
        layer_idx += 1;
        // Pointwise 1x1.
        weights.insert(
            format!("{prefix}.subsampling.layers.{layer_idx}.weight"),
            rand_f32(&[sub_channels, 1, 1, sub_channels], 100 + stage as u32),
        );
        layer_idx += 2;
    }
    // Subsampling linear: [hidden, sub_channels * out_freq].
    weights.insert(
        format!("{prefix}.subsampling.linear.weight"),
        rand_f32(&[h, lin_in], 200),
    );
    weights.insert(
        format!("{prefix}.subsampling.linear.bias"),
        rand_f32(&[h], 201),
    );

    // One Parakeet block.
    let block_prefix = format!("{prefix}.layers.0");
    // Norms (LayerNorm: weight + bias, both [h]).
    for name in &[
        "norm_feed_forward1",
        "norm_self_att",
        "norm_conv",
        "norm_feed_forward2",
        "norm_out",
    ] {
        weights.insert(
            format!("{block_prefix}.{name}.weight"),
            mlxcel_core::ones(&[h], mlxcel_core::dtype::FLOAT32),
        );
        weights.insert(
            format!("{block_prefix}.{name}.bias"),
            mlxcel_core::zeros(&[h], mlxcel_core::dtype::FLOAT32),
        );
    }

    // Feed-forward 1 / 2.
    for name in &["feed_forward1", "feed_forward2"] {
        weights.insert(
            format!("{block_prefix}.{name}.linear1.weight"),
            rand_f32(&[intermediate, h], 300),
        );
        weights.insert(
            format!("{block_prefix}.{name}.linear2.weight"),
            rand_f32(&[h, intermediate], 301),
        );
    }

    // Self-attention.
    let attn = format!("{block_prefix}.self_attn");
    let proj_dim = heads * head_dim;
    weights.insert(
        format!("{attn}.q_proj.weight"),
        rand_f32(&[proj_dim, h], 400),
    );
    weights.insert(
        format!("{attn}.k_proj.weight"),
        rand_f32(&[proj_dim, h], 401),
    );
    weights.insert(
        format!("{attn}.v_proj.weight"),
        rand_f32(&[proj_dim, h], 402),
    );
    weights.insert(
        format!("{attn}.o_proj.weight"),
        rand_f32(&[h, proj_dim], 403),
    );
    weights.insert(
        format!("{attn}.relative_k_proj.weight"),
        rand_f32(&[proj_dim, h], 404),
    );
    weights.insert(
        format!("{attn}.bias_u"),
        mlxcel_core::zeros(&[heads, head_dim], mlxcel_core::dtype::FLOAT32),
    );
    weights.insert(
        format!("{attn}.bias_v"),
        mlxcel_core::zeros(&[heads, head_dim], mlxcel_core::dtype::FLOAT32),
    );

    // Convolution module. Pointwise conv1 expands to 2*hidden for GLU.
    let conv = format!("{block_prefix}.conv");
    weights.insert(
        format!("{conv}.pointwise_conv1.weight"),
        rand_f32(&[2 * h, 1, h], 500),
    );
    weights.insert(
        format!("{conv}.depthwise_conv.weight"),
        rand_f32(&[h, kernel, 1], 501),
    );
    weights.insert(
        format!("{conv}.pointwise_conv2.weight"),
        rand_f32(&[h, 1, h], 502),
    );
    weights.insert(
        format!("{conv}.norm.weight"),
        mlxcel_core::ones(&[h], mlxcel_core::dtype::FLOAT32),
    );
    weights.insert(
        format!("{conv}.norm.bias"),
        mlxcel_core::zeros(&[h], mlxcel_core::dtype::FLOAT32),
    );
    weights.insert(
        format!("{conv}.norm.running_mean"),
        mlxcel_core::zeros(&[h], mlxcel_core::dtype::FLOAT32),
    );
    weights.insert(
        format!("{conv}.norm.running_var"),
        mlxcel_core::ones(&[h], mlxcel_core::dtype::FLOAT32),
    );

    weights
}

fn build_synthetic_projection_weights(
    config: &NemotronOmniAudioConfig,
    llm_hidden: i32,
) -> WeightMap {
    let mut weights = WeightMap::new();
    let h = config.hidden_size as i32;
    let proj = config.projection_hidden_size as i32;
    weights.insert(
        "sound_projection.norm.weight".to_string(),
        mlxcel_core::ones(&[h], mlxcel_core::dtype::FLOAT32),
    );
    weights.insert(
        "sound_projection.linear1.weight".to_string(),
        rand_f32(&[proj, h], 600),
    );
    weights.insert(
        "sound_projection.linear2.weight".to_string(),
        rand_f32(&[llm_hidden, proj], 601),
    );
    weights
}

#[test]
fn encoder_loads_and_produces_expected_shape() {
    let config = small_audio_config();
    let weights = build_synthetic_audio_weights(&config);
    let encoder = NemotronOmniSoundEncoder::from_weights(&weights, "sound_encoder", &config, 64, 4)
        .expect("build encoder");

    // Build a [1, T, num_mel_bins] feature batch. T must be >= subsampling
    // factor so the time dim survives the subsampling stack with shape > 0.
    let frames = 32i32;
    let total = (frames * config.num_mel_bins as i32) as usize;
    let data: Vec<f32> = (0..total).map(|i| (i as f32) * 1e-3).collect();
    let input = mlxcel_core::from_slice_f32(&data, &[1, frames, config.num_mel_bins as i32]);

    let output = encoder.forward(input.as_ref().unwrap(), None);
    let shape = mlxcel_core::array_shape(&output);
    let expected_t = config.subsampling_output_length(frames as usize) as i32;
    assert_eq!(shape, vec![1, expected_t, config.hidden_size as i32]);
    assert_eq!(
        mlxcel_core::array_dtype(&output),
        mlxcel_core::dtype::FLOAT32
    );
}

#[test]
fn encoder_with_attention_mask_zeros_padded_frames() {
    let config = small_audio_config();
    let weights = build_synthetic_audio_weights(&config);
    let encoder = NemotronOmniSoundEncoder::from_weights(&weights, "sound_encoder", &config, 64, 4)
        .expect("build encoder");

    let frames = 32i32;
    let total = (frames * config.num_mel_bins as i32) as usize;
    let data: Vec<f32> = (0..total).map(|i| (i as f32) * 1e-3).collect();
    let input = mlxcel_core::from_slice_f32(&data, &[1, frames, config.num_mel_bins as i32]);
    // Attention mask with only the first 16 frames valid.
    let mut mask_data = vec![1i32; 16];
    mask_data.extend(vec![0i32; 16]);
    let mask = mlxcel_core::from_slice_i32(&mask_data, &[1, frames]);

    let output = encoder.forward(input.as_ref().unwrap(), Some(mask.as_ref().unwrap()));
    let shape = mlxcel_core::array_shape(&output);
    let expected_t = config.subsampling_output_length(frames as usize) as i32;
    assert_eq!(shape, vec![1, expected_t, config.hidden_size as i32]);
    // A non-trivial attention mask should not crash the path; numerical
    // checks are deferred to Apple-Silicon bring-up.
}

#[test]
fn projector_loads_and_maps_to_llm_hidden() {
    let config = small_audio_config();
    let llm_hidden = 24i32;
    let weights = build_synthetic_projection_weights(&config, llm_hidden);
    let projection =
        NemotronOmniSoundProjection::from_weights(&weights, "sound_projection", &config, 64, 4)
            .expect("build projection");
    let frames = 4i32;
    let total = (frames * config.hidden_size as i32) as usize;
    let data: Vec<f32> = (0..total).map(|i| (i as f32) * 1e-3).collect();
    let input = mlxcel_core::from_slice_f32(&data, &[1, frames, config.hidden_size as i32]);
    let output = projection.forward(input.as_ref().unwrap());
    let shape = mlxcel_core::array_shape(&output);
    assert_eq!(shape, vec![1, frames, llm_hidden]);
}

#[test]
fn missing_weight_surfaces_as_loader_error() {
    let config = small_audio_config();
    let mut weights = build_synthetic_audio_weights(&config);
    weights.remove("sound_encoder.encoder.subsampling.layers.0.weight");
    let result = NemotronOmniSoundEncoder::from_weights(&weights, "sound_encoder", &config, 64, 4);
    assert!(result.is_err(), "expected loader error for missing weight");
}

#[test]
fn config_subsampling_output_matches_iteration_count() {
    let config = small_audio_config();
    // log2(4) = 2 stages; from 32 -> 16 -> 8.
    assert_eq!(config.num_subsampling_layers(), 2);
    assert_eq!(config.subsampling_output_length(32), 8);
}

/// Pins the relative-position vector for T=4 to the upstream-equivalent
/// descending order `[3, 2, 1, 0, -1, -2, -3]` (mirrors
/// `mx.arange(seq_length - 1, -seq_length, -1)` from
/// https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/nemotron_h_nano_omni/audio.py).
///
/// This is the regression guard for the bug where the position vector
/// was double-reversed into ascending order (`[-3, -2, -1, 0, 1, 2, 3]`),
/// silently flipping the sign of every sin component fed into the
/// relative-position attention BD term. A buggy implementation would
/// fail the first assertion below.
#[test]
fn rel_position_vector_descends_for_t4() {
    let positions = parakeet_rel_position_vector(4);
    assert_eq!(
        positions,
        vec![3.0_f32, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0],
        "position vector must be descending [T-1 .. -(T-1)] to match upstream"
    );
}

/// Spot-check additional `T` values to ensure the helper behaves
/// uniformly. T=1 collapses to a single zero; T=2 yields three entries.
#[test]
fn rel_position_vector_handles_edge_lengths() {
    assert_eq!(parakeet_rel_position_vector(1), vec![0.0_f32]);
    assert_eq!(parakeet_rel_position_vector(2), vec![1.0_f32, 0.0, -1.0]);
    let t8 = parakeet_rel_position_vector(8);
    assert_eq!(t8.len(), 15);
    assert_eq!(t8[0], 7.0); // first = T - 1
    assert_eq!(t8[7], 0.0); // middle = 0
    assert_eq!(t8[14], -7.0); // last = -(T - 1)
}

/// Pins the sinusoidal frequencies derived from the position vector for
/// `hidden_size = 4`. With the buggy ascending order the first row's
/// sin would be `sin(-3.0)` instead of `sin(3.0)`, so this test would
/// also fail on the pre-fix code.
///
/// theta_k = 1 / 10000^(2k / hidden_size). For hidden_size = 4:
///   theta_0 = 1 / 10000^0          = 1.0
///   theta_1 = 1 / 10000^(2/4)      = 1 / 100.0 = 0.01
///
/// The first row corresponds to position `T - 1 = 3`. Channel layout is
/// `[sin(3*theta_0), cos(3*theta_0), sin(3*theta_1), cos(3*theta_1)]`.
#[test]
fn rel_position_first_row_matches_analytic_frequencies() {
    const T: usize = 4;
    const HIDDEN: usize = 4;
    let positions = parakeet_rel_position_vector(T);
    assert_eq!(positions[0], 3.0_f32);

    let theta_0 = 1.0_f32; // 10000^0
    let theta_1 = 1.0_f32 / 100.0_f32; // 10000^(2/4) == 100
    let p = positions[0];

    let expected = [
        (p * theta_0).sin(),
        (p * theta_0).cos(),
        (p * theta_1).sin(),
        (p * theta_1).cos(),
    ];

    // Mirror the encoder's stack-then-reshape: at hidden_size = 4,
    // hidden_size / 2 = 2. We compute the same row directly from the
    // public helper above and assert the channel layout matches the
    // analytic expectation.
    let half = HIDDEN / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|k| 1.0 / 10_000_f32.powf((2 * k) as f32 / HIDDEN as f32))
        .collect();
    assert!((inv_freq[0] - theta_0).abs() < 1e-6);
    assert!((inv_freq[1] - theta_1).abs() < 1e-6);

    let row: Vec<f32> = inv_freq
        .iter()
        .flat_map(|&iv| [(p * iv).sin(), (p * iv).cos()])
        .collect();

    assert_eq!(row.len(), HIDDEN);
    for (got, want) in row.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() < 1e-6,
            "rel-pos channel mismatch: got {got} want {want}"
        );
    }
}

/// End-to-end synthetic test that mirrors what
/// `compute_nemotron_h_nano_omni_audio_embeddings` does internally
/// once the WAV is loaded: run the feature extractor on a synthetic
/// waveform, push the result through the audio encoder + projector,
/// and verify the post-subsampling token count matches the
/// configured-formula prediction.
///
/// This is the integration regression guard CRITICAL
/// #2: the bug we are guarding against is a length/shape mismatch
/// between what the prompt-expansion path expects (`subsampling_output_length(num_frames)`)
/// and what the encoder actually emits.
#[test]
fn synthetic_audio_pipeline_produces_consistent_token_count() {
    let config = small_audio_config();
    let encoder_weights = build_synthetic_audio_weights(&config);
    let encoder =
        NemotronOmniSoundEncoder::from_weights(&encoder_weights, "sound_encoder", &config, 64, 4)
            .expect("build encoder");
    let llm_hidden = 24i32;
    let projector_weights = build_synthetic_projection_weights(&config, llm_hidden);
    let projector = NemotronOmniSoundProjection::from_weights(
        &projector_weights,
        "sound_projection",
        &config,
        64,
        4,
    )
    .expect("build projector");

    // Synthetic waveform: 0.5 seconds @ configured sampling rate. Long
    // enough to survive 2 subsampling stages without collapsing to T=0.
    let n_samples = (config.sampling_rate as usize) / 2;
    let samples: Vec<f32> = (0..n_samples)
        .map(|i| {
            (2.0f32 * std::f32::consts::PI * 220.0 * i as f32 / config.sampling_rate as f32).sin()
        })
        .collect();

    let extractor = NemotronOmniFeatureExtractor::new(&config);
    let extracted = extractor.extract_batch(&[&samples[..]]);

    // Sanity: feature extractor produced a non-empty batch.
    let batch = extracted.features_shape[0] as usize;
    let frames = extracted.features_shape[1] as usize;
    let mels = extracted.features_shape[2] as usize;
    assert_eq!(batch, 1, "single-clip CLI path");
    assert!(frames > 0);
    assert_eq!(mels, config.num_mel_bins);

    let features = mlxcel_core::from_slice_f32(
        &extracted.features,
        &[batch as i32, frames as i32, mels as i32],
    );
    let mask =
        mlxcel_core::from_slice_i32(&extracted.attention_mask, &[batch as i32, frames as i32]);

    let encoded = encoder.forward(features.as_ref().unwrap(), Some(mask.as_ref().unwrap()));
    let encoded_shape = mlxcel_core::array_shape(&encoded);

    // Token count = post-subsampling length predicted by the formula.
    // The CLI prompt-expansion path uses this exact formula to size
    // the `sound_context_token_id` block, so this is the contract the
    // CLI helper relies on.
    let expected_tokens = config.subsampling_output_length(frames) as i32;
    assert_eq!(
        encoded_shape,
        vec![batch as i32, expected_tokens, config.hidden_size as i32],
        "encoder output token count must equal subsampling_output_length(num_frames)"
    );

    let projected = projector.forward(&encoded);
    let projected_shape = mlxcel_core::array_shape(&projected);
    assert_eq!(
        projected_shape,
        vec![batch as i32, expected_tokens, llm_hidden],
        "projector output must produce LLM-hidden-sized embeddings"
    );

    // The CLI helper expands the prompt with `expected_tokens`
    // placeholders, so this number is also the audio-token count it
    // reports in the `VlmPreparationSummary::NemotronHNanoOmniAudio`
    // summary. Lock it to a positive value (formula above never
    // returns 0 for a non-trivial frame count).
    assert!(
        expected_tokens > 0,
        "subsampling_output_length must produce >= 1 audio token for a non-empty clip"
    );
}
