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

use super::*;

fn tiny_config() -> Gemma3nAudioConfig {
    Gemma3nAudioConfig {
        vocab_size: 8,
        vocab_offset: 100,
        input_feat_size: 4,
        hidden_size: 4,
        rms_norm_eps: 1e-6,
        gradient_clipping: 1e4,
        conf_attention_chunk_size: 2,
        conf_attention_context_left: 2,
        conf_attention_context_right: 0,
        conf_attention_logit_cap: 5.0,
        conf_num_attention_heads: 2,
        conf_num_hidden_layers: 1,
        conf_conv_kernel_size: 3,
        conf_reduction_factor: 2,
        conf_residual_weight: 0.5,
        sscp_conv_channel_size: vec![2, 2],
        sscp_conv_group_norm_eps: 1e-3,
        sscp_conv_kernel_size: vec![[3, 3], [3, 3]],
        sscp_conv_stride_size: vec![[2, 2], [2, 2]],
    }
}

fn insert(weights: &mut WeightMap, key: &str, shape: &[i32], value: f32) {
    weights.insert(
        key.to_string(),
        mlxcel_core::full_f32(shape, value, mlxcel_core::dtype::FLOAT32),
    );
}

fn values(array: &MlxArray) -> Vec<f32> {
    mlxcel_core::eval(array);
    mlxcel_core::array_to_raw_bytes(array)
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn tiny_weights() -> WeightMap {
    let mut weights = WeightMap::new();
    insert(
        &mut weights,
        "audio.subsample_conv_projection.conv_0.conv.weight",
        &[2, 3, 3, 1],
        0.01,
    );
    insert(
        &mut weights,
        "audio.subsample_conv_projection.conv_0.norm.weight",
        &[2],
        1.0,
    );
    insert(
        &mut weights,
        "audio.subsample_conv_projection.conv_1.conv.weight",
        &[2, 3, 3, 2],
        0.01,
    );
    insert(
        &mut weights,
        "audio.subsample_conv_projection.conv_1.norm.weight",
        &[2],
        1.0,
    );
    insert(
        &mut weights,
        "audio.subsample_conv_projection.input_proj_linear.weight",
        &[4, 2],
        0.01,
    );

    for name in ["ffw_layer_start", "ffw_layer_end"] {
        let prefix = format!("audio.conformer.0.{name}");
        insert(
            &mut weights,
            &format!("{prefix}.pre_layer_norm.weight"),
            &[4],
            1.0,
        );
        insert(
            &mut weights,
            &format!("{prefix}.ffw_layer_1.weight"),
            &[16, 4],
            0.01,
        );
        insert(
            &mut weights,
            &format!("{prefix}.ffw_layer_2.weight"),
            &[4, 16],
            0.01,
        );
        insert(
            &mut weights,
            &format!("{prefix}.post_layer_norm.weight"),
            &[4],
            1.0,
        );
    }

    for name in ["pre_attn_norm", "post_norm"] {
        insert(
            &mut weights,
            &format!("audio.conformer.0.attention.{name}.weight"),
            &[4],
            1.0,
        );
    }
    for name in ["q_proj", "k_proj", "v_proj"] {
        insert(
            &mut weights,
            &format!("audio.conformer.0.attention.attn.{name}.weight"),
            &[4, 4],
            0.01,
        );
    }
    insert(
        &mut weights,
        "audio.conformer.0.attention.attn.per_dim_scale",
        &[2],
        0.0,
    );
    insert(
        &mut weights,
        "audio.conformer.0.attention.attn.relative_position_embedding.pos_proj.weight",
        &[4, 4],
        0.01,
    );
    insert(
        &mut weights,
        "audio.conformer.0.attention.post.weight",
        &[4, 4],
        0.01,
    );

    for name in ["pre_layer_norm", "conv_norm"] {
        insert(
            &mut weights,
            &format!("audio.conformer.0.lconv1d.{name}.weight"),
            &[4],
            1.0,
        );
    }
    insert(
        &mut weights,
        "audio.conformer.0.lconv1d.linear_start.weight",
        &[8, 4],
        0.01,
    );
    insert(
        &mut weights,
        "audio.conformer.0.lconv1d.depthwise_conv1d.weight",
        &[4, 3, 1],
        0.01,
    );
    insert(
        &mut weights,
        "audio.conformer.0.lconv1d.linear_end.weight",
        &[4, 4],
        0.01,
    );
    insert(&mut weights, "audio.conformer.0.norm.weight", &[4], 1.0);
    weights
}

#[test]
fn default_local_attention_contract_rejects_global_substitution() {
    let config = Gemma3nAudioConfig::default();
    assert_eq!(config.conf_attention_chunk_size, 12);
    assert_eq!(config.max_past_horizon(), 12);
    assert_eq!(config.context_size(), 24);
    assert_eq!(config.head_dim(), 192);
    assert_eq!(config.conf_attention_logit_cap, 50.0);

    let encoder =
        Gemma3nAudioEncoder::from_weights(&tiny_weights(), "audio", &tiny_config(), 64, 4).unwrap();
    let causal_mask = encoder.causal_valid_mask();
    mlxcel_core::eval(&causal_mask);
    assert_eq!(
        mlxcel_core::array_to_raw_bytes(&causal_mask),
        vec![1, 1, 0, 0, 1, 1]
    );
}

#[test]
fn invalid_sscp_config_is_rejected_before_loading_weights() {
    let mut config = Gemma3nAudioConfig::default();
    config.sscp_conv_channel_size.pop();
    assert!(config.validate().is_err());
    let config = Gemma3nAudioConfig {
        conf_attention_logit_cap: 0.0,
        ..Gemma3nAudioConfig::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn cumulative_group_norm_does_not_collapse_to_ordinary_group_norm() {
    let norm = CumulativeGroupNorm {
        weight: mlxcel_core::from_slice_f32(&[1.0, 2.0], &[2]),
        eps: 0.0,
    };
    let input = mlxcel_core::from_slice_f32(&[1.0, 3.0, 5.0, 7.0], &[1, 2, 1, 2]);
    let actual = values(&norm.forward(&input));
    let expected = [-1.0, 2.0, 1.0 / 3.0f32.sqrt(), 6.0 / 3.0f32.sqrt()];
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }
}

#[test]
fn sscp_casts_processor_features_to_checkpoint_convolution_dtype() {
    let block = SscpConvBlock {
        conv_weight: mlxcel_core::astype(
            &mlxcel_core::full_f32(&[2, 3, 3, 1], 0.01, mlxcel_core::dtype::FLOAT32),
            mlxcel_core::dtype::BFLOAT16,
        ),
        norm: CumulativeGroupNorm {
            weight: mlxcel_core::ones(&[2], mlxcel_core::dtype::BFLOAT16),
            eps: 1e-3,
        },
        time_padding_after: 2,
        stride_time: 2,
        stride_frequency: 2,
    };
    let samples: Vec<f32> = (0..16).map(|index| index as f32 / 37.0).collect();
    let input = mlxcel_core::from_slice_f32(&samples, &[1, 4, 4, 1]);
    let input_bf16 = mlxcel_core::astype(&input, mlxcel_core::dtype::BFLOAT16);
    let from_processor = values(&block.forward(&input).unwrap());
    let from_checkpoint_dtype = values(&block.forward(&input_bf16).unwrap());
    assert_eq!(from_processor, from_checkpoint_dtype);
}

#[test]
fn tiny_encoder_exercises_sscp_conformer_reduction_and_mask() {
    let encoder =
        Gemma3nAudioEncoder::from_weights(&tiny_weights(), "audio", &tiny_config(), 64, 4).unwrap();
    let mel = mlxcel_core::from_slice_f32(
        &(0..32).map(|index| index as f32 / 32.0).collect::<Vec<_>>(),
        &[1, 8, 4],
    );
    let invalid = mlxcel_core::astype(
        &mlxcel_core::from_slice_i32(&[0, 0, 0, 0, 0, 0, 1, 1], &[1, 8]),
        mlxcel_core::dtype::BOOL,
    );
    let (encoded, mask) = encoder.forward(&mel, &invalid).unwrap();
    assert_eq!(mlxcel_core::array_shape(&encoded), vec![1, 1, 4]);
    assert_eq!(mlxcel_core::array_shape(&mask), vec![1, 1]);
    mlxcel_core::eval(&encoded);
}

#[test]
fn malformed_checkpoint_conv_shape_is_rejected() {
    let mut weights = tiny_weights();
    weights.insert(
        "audio.subsample_conv_projection.conv_0.conv.weight".into(),
        mlxcel_core::zeros(&[2, 1, 3, 3], mlxcel_core::dtype::FLOAT32),
    );
    let error = Gemma3nAudioEncoder::from_weights(&weights, "audio", &tiny_config(), 64, 4)
        .err()
        .expect("shape mismatch must fail");
    assert!(error.contains("expected"));
}

#[test]
fn malformed_checkpoint_projection_shape_is_rejected() {
    let mut weights = tiny_weights();
    weights.insert(
        "audio.conformer.0.ffw_layer_start.ffw_layer_1.weight".into(),
        mlxcel_core::zeros(&[4, 4], mlxcel_core::dtype::FLOAT32),
    );
    let error = Gemma3nAudioEncoder::from_weights(&weights, "audio", &tiny_config(), 64, 4)
        .err()
        .expect("linear shape mismatch must fail");
    assert!(error.contains("expected logical"));
}
