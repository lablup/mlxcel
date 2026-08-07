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

//! Checkpoint-free unit tests for the Florence-2 DaViT backbone: config
//! parsing and validation, the channels-last sanitize pass, and a
//! synthetic-weight forward that pins the stage shape progression.
//!
//! Numerical parity against the mlx-vlm reference lives in
//! `tests/florence2_vision_parity.rs`, which needs the real checkpoint.

use serde_json::{Value, json};

use super::*;

/// The real `Florence-2-base-ft` `vision_config`, including the fields the
/// backbone does not consume.
fn real_vision_config() -> Value {
    json!({
        "model_type": "",
        "depths": [1, 1, 9, 1],
        "dim_embed": [128, 256, 512, 1024],
        "num_heads": [4, 8, 16, 32],
        "num_groups": [4, 8, 16, 32],
        "patch_size": [7, 3, 3, 3],
        "patch_stride": [4, 2, 2, 2],
        "patch_padding": [3, 1, 1, 1],
        "patch_prenorm": [false, true, true, true],
        "window_size": 12,
        "drop_path_rate": 0.1,
        "projection_dim": 768,
        "image_pos_embed": {"type": "learned_abs_2d", "max_pos_embeddings": 50},
        "visual_temporal_embedding": {"type": "COSINE", "max_temporal_embeddings": 100},
        "image_feature_source": ["spatial_avg_pool", "temporal_avg_pool"],
    })
}

#[test]
fn parses_real_shaped_vision_config() {
    let config = Florence2VisionConfig::from_vision_config(&real_vision_config())
        .expect("real-shaped vision config parses");

    assert_eq!(config.num_stages(), 4);
    assert_eq!(config.dim_embed, vec![128, 256, 512, 1024]);
    assert_eq!(config.num_groups, vec![4, 8, 16, 32]);
    assert_eq!(config.patch_prenorm, vec![false, true, true, true]);
    assert_eq!(config.window_size, 12);
    // Absent from the checkpoint, so these come from the defaults.
    assert_eq!(config.in_chans, 3);
    assert!((config.mlp_ratio - 4.0).abs() < 1e-6);
    assert!(config.qkv_bias);
    assert!(config.conv_at_attn);
    assert!(config.conv_at_ffn);
    // Training-only, but still parsed.
    assert!((config.drop_path_rate - 0.1).abs() < 1e-6);

    // The backbone emits dim_embed[-1]; projection_dim belongs to the fusion
    // stage and is retained, not applied.
    assert_eq!(config.output_dim(), 1024);
    assert_eq!(config.projection_dim, 768);
    assert_eq!(
        config.image_feature_source,
        vec![
            "spatial_avg_pool".to_string(),
            "temporal_avg_pool".to_string()
        ]
    );
    assert_eq!(
        config
            .image_pos_embed
            .as_ref()
            .and_then(|v| v.get("max_pos_embeddings"))
            .and_then(Value::as_i64),
        Some(50)
    );
    assert_eq!(
        config
            .visual_temporal_embedding
            .as_ref()
            .and_then(|v| v.get("type"))
            .and_then(Value::as_str),
        Some("COSINE")
    );
}

#[test]
fn accepts_every_davit_model_type_spelling_and_rejects_others() {
    // Empty string: what every real Florence-2 checkpoint actually ships.
    assert!(Florence2VisionConfig::from_vision_config(&real_vision_config()).is_ok());

    // Explicit "davit".
    let mut explicit = real_vision_config();
    explicit["model_type"] = json!("davit");
    assert!(Florence2VisionConfig::from_vision_config(&explicit).is_ok());

    // Absent entirely.
    let mut absent = real_vision_config();
    absent.as_object_mut().unwrap().remove("model_type");
    assert!(Florence2VisionConfig::from_vision_config(&absent).is_ok());

    // Anything else is not a DaViT tower.
    let mut bogus = real_vision_config();
    bogus["model_type"] = json!("siglip");
    let err = Florence2VisionConfig::from_vision_config(&bogus)
        .expect_err("non-DaViT vision model_type must be rejected");
    assert!(
        err.to_string().contains("siglip"),
        "error should name the offending model_type: {err}"
    );
}

#[test]
fn from_model_config_descends_into_vision_config() {
    let full = json!({
        "model_type": "florence2",
        "vision_config": real_vision_config(),
        "text_config": {"d_model": 768},
    });
    let nested = Florence2VisionConfig::from_model_config(&full).expect("descend into vision");
    let bare = Florence2VisionConfig::from_model_config(&real_vision_config())
        .expect("bare vision config accepted");
    assert_eq!(nested.dim_embed, bare.dim_embed);
    assert_eq!(nested.output_dim(), bare.output_dim());
}

#[test]
fn rejects_inconsistent_stage_lists() {
    let mut short = real_vision_config();
    short["num_heads"] = json!([4, 8, 16]);
    let err = Florence2VisionConfig::from_vision_config(&short)
        .expect_err("stage list length mismatch must be rejected");
    assert!(err.to_string().contains("num_heads"), "{err}");

    let mut indivisible = real_vision_config();
    indivisible["num_groups"] = json!([4, 8, 16, 33]);
    let err = Florence2VisionConfig::from_vision_config(&indivisible)
        .expect_err("dim not divisible by num_groups must be rejected");
    assert!(err.to_string().contains("num_groups"), "{err}");

    let mut missing = real_vision_config();
    missing.as_object_mut().unwrap().remove("dim_embed");
    assert!(Florence2VisionConfig::from_vision_config(&missing).is_err());
}

fn seq_array(shape: &[i32], seed: f32) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 0.37 + seed).sin() * 0.1)
        .collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

#[test]
fn sanitize_remaps_conv_weights_and_is_idempotent() {
    let mut weights = WeightMap::new();
    // PyTorch layout (O, I, kH, kW).
    weights.insert("convs.0.proj.weight".into(), seq_array(&[8, 3, 7, 7], 0.0));
    weights.insert("convs.1.proj.weight".into(), seq_array(&[16, 8, 3, 3], 1.0));
    weights.insert(
        "blocks.0.0.spatial_block.conv1.fn.dw.weight".into(),
        seq_array(&[8, 1, 3, 3], 2.0),
    );
    // Non-conv tensors must pass through untouched.
    weights.insert("convs.0.norm.weight".into(), seq_array(&[8], 3.0));
    weights.insert("convs.0.proj.bias".into(), seq_array(&[8], 4.0));
    // Unused buffers are dropped.
    weights.insert("blocks.0.0.position_ids".into(), seq_array(&[4], 5.0));

    let once = sanitize(weights);
    assert!(!once.contains_key("blocks.0.0.position_ids"));
    assert_eq!(
        mlxcel_core::array_shape(&once["convs.0.proj.weight"]),
        vec![8, 7, 7, 3]
    );
    assert_eq!(
        mlxcel_core::array_shape(&once["convs.1.proj.weight"]),
        vec![16, 3, 3, 8]
    );
    assert_eq!(
        mlxcel_core::array_shape(&once["blocks.0.0.spatial_block.conv1.fn.dw.weight"]),
        vec![8, 3, 3, 1]
    );
    assert_eq!(
        mlxcel_core::array_shape(&once["convs.0.norm.weight"]),
        vec![8]
    );

    // Second pass must be a no-op: mlx-community exports already ship the
    // channels-last layout, so `load` must not double-transpose them.
    let expected: Vec<(String, Vec<i32>)> = once
        .iter()
        .map(|(k, v)| (k.clone(), mlxcel_core::array_shape(v)))
        .collect();
    let twice = sanitize(once);
    assert_eq!(twice.len(), expected.len());
    for (key, shape) in expected {
        assert_eq!(
            mlxcel_core::array_shape(&twice[&key]),
            shape,
            "sanitize is not idempotent for {key}"
        );
    }
}

/// Tiny two-stage DaViT with the same structure as the real one: 16x16 input
/// -> 8x8 @ 8 channels -> 4x4 @ 16 channels. `window_size = 6` does not
/// divide either grid, so both stages exercise the window pad / crop path.
fn tiny_config() -> Florence2VisionConfig {
    Florence2VisionConfig::from_vision_config(&json!({
        "model_type": "davit",
        "depths": [1, 1],
        "dim_embed": [8, 16],
        "num_heads": [2, 4],
        "num_groups": [2, 4],
        "patch_size": [3, 3],
        "patch_stride": [2, 2],
        "patch_padding": [1, 1],
        "patch_prenorm": [false, true],
        "window_size": 6,
    }))
    .expect("tiny config parses")
}

fn tiny_weights(config: &Florence2VisionConfig) -> WeightMap {
    let mut weights = WeightMap::new();
    let mut seed = 0.0f32;
    let push = |weights: &mut WeightMap, key: String, shape: &[i32], seed: &mut f32| {
        *seed += 1.0;
        weights.insert(key, seq_array(shape, *seed));
    };

    for stage in 0..config.num_stages() {
        let dim = config.dim_embed[stage];
        let in_chans = if stage == 0 {
            config.in_chans
        } else {
            config.dim_embed[stage - 1]
        };
        let k = config.patch_size[stage];
        // Channels-last, as sanitize would leave it.
        push(
            &mut weights,
            format!("convs.{stage}.proj.weight"),
            &[dim, k, k, in_chans],
            &mut seed,
        );
        push(
            &mut weights,
            format!("convs.{stage}.proj.bias"),
            &[dim],
            &mut seed,
        );
        // pre_norm normalizes the incoming width, post_norm the outgoing one.
        let norm_dim = if config.patch_prenorm[stage] {
            in_chans
        } else {
            dim
        };
        for suffix in ["weight", "bias"] {
            push(
                &mut weights,
                format!("convs.{stage}.norm.{suffix}"),
                &[norm_dim],
                &mut seed,
            );
        }

        for depth in 0..config.depths[stage] {
            let block = format!("blocks.{stage}.{depth}");
            for (half, attn) in [
                ("spatial_block", "window_attn"),
                ("channel_block", "channel_attn"),
            ] {
                for conv in ["conv1", "conv2"] {
                    push(
                        &mut weights,
                        format!("{block}.{half}.{conv}.fn.dw.weight"),
                        &[dim, 3, 3, 1],
                        &mut seed,
                    );
                    push(
                        &mut weights,
                        format!("{block}.{half}.{conv}.fn.dw.bias"),
                        &[dim],
                        &mut seed,
                    );
                }
                for suffix in ["weight", "bias"] {
                    push(
                        &mut weights,
                        format!("{block}.{half}.{attn}.norm.{suffix}"),
                        &[dim],
                        &mut seed,
                    );
                    push(
                        &mut weights,
                        format!("{block}.{half}.ffn.norm.{suffix}"),
                        &[dim],
                        &mut seed,
                    );
                }
                push(
                    &mut weights,
                    format!("{block}.{half}.{attn}.fn.qkv.weight"),
                    &[dim * 3, dim],
                    &mut seed,
                );
                push(
                    &mut weights,
                    format!("{block}.{half}.{attn}.fn.qkv.bias"),
                    &[dim * 3],
                    &mut seed,
                );
                push(
                    &mut weights,
                    format!("{block}.{half}.{attn}.fn.proj.weight"),
                    &[dim, dim],
                    &mut seed,
                );
                push(
                    &mut weights,
                    format!("{block}.{half}.{attn}.fn.proj.bias"),
                    &[dim],
                    &mut seed,
                );
                let hidden = (dim as f32 * config.mlp_ratio) as i32;
                push(
                    &mut weights,
                    format!("{block}.{half}.ffn.fn.net.fc1.weight"),
                    &[hidden, dim],
                    &mut seed,
                );
                push(
                    &mut weights,
                    format!("{block}.{half}.ffn.fn.net.fc1.bias"),
                    &[hidden],
                    &mut seed,
                );
                push(
                    &mut weights,
                    format!("{block}.{half}.ffn.fn.net.fc2.weight"),
                    &[dim, hidden],
                    &mut seed,
                );
                push(
                    &mut weights,
                    format!("{block}.{half}.ffn.fn.net.fc2.bias"),
                    &[dim],
                    &mut seed,
                );
            }
        }
    }
    weights
}

#[test]
fn synthetic_forward_follows_the_stage_shape_progression() {
    let config = tiny_config();
    let weights = tiny_weights(&config);
    let model =
        Florence2DaViT::from_weights(&weights, &config, "").expect("build tiny DaViT backbone");

    let pixels = seq_array(&[1, 3, 16, 16], 0.5);
    let stages = model.forward_stages(&pixels);
    assert_eq!(stages.len(), 2);

    assert_eq!(stages[0].1, (8, 8));
    assert_eq!(mlxcel_core::array_shape(&stages[0].0), vec![1, 64, 8]);
    assert_eq!(stages[1].1, (4, 4));
    assert_eq!(mlxcel_core::array_shape(&stages[1].0), vec![1, 16, 16]);

    let out = Florence2DaViT::forward(&model, &pixels);
    assert_eq!(mlxcel_core::array_shape(&out), vec![1, 16, 16]);
    mlxcel_core::eval(&out);

    // The VisionEncoder trait entry point must agree with the inherent one.
    let via_trait = VisionEncoder::forward(&model, &pixels);
    assert_eq!(
        mlxcel_core::array_shape(&via_trait.hidden_states),
        vec![1, 16, 16]
    );
}

/// `Florence2DaViT` has no `Debug`, so `expect_err` is unavailable.
fn load_error(result: Result<Florence2DaViT, String>, what: &str) -> String {
    match result {
        Ok(_) => panic!("{what}"),
        Err(err) => err,
    }
}

#[test]
fn rejects_hostile_patch_geometry_and_depths() {
    // A zero stride and a negative padding both reach MLX `conv2d`, which
    // throws across an FFI boundary that cannot carry the throw, so the
    // config parse has to be the thing that rejects them.
    let mut zero_stride = real_vision_config();
    zero_stride["patch_stride"] = json!([4, 0, 2, 2]);
    let err = Florence2VisionConfig::from_vision_config(&zero_stride)
        .expect_err("a zero patch_stride must be rejected");
    assert!(err.to_string().contains("patch_stride"), "{err}");

    let mut negative_padding = real_vision_config();
    negative_padding["patch_padding"] = json!([3, 1, -1, 1]);
    let err = Florence2VisionConfig::from_vision_config(&negative_padding)
        .expect_err("a negative patch_padding must be rejected");
    assert!(err.to_string().contains("patch_padding"), "{err}");

    // `from_weights` sizes a Vec from `depths` before looking up any weight.
    let mut absurd_depth = real_vision_config();
    absurd_depth["depths"] = json!([1, 1, i32::MAX, 1]);
    let err = Florence2VisionConfig::from_vision_config(&absurd_depth)
        .expect_err("an absurd depth must not reach Vec::with_capacity");
    assert!(err.to_string().contains("depths"), "{err}");

    let mut zero_channels = real_vision_config();
    zero_channels["in_chans"] = json!(0);
    assert!(Florence2VisionConfig::from_vision_config(&zero_channels).is_err());
}

#[test]
fn from_weights_revalidates_a_hand_built_config() {
    // The struct has public fields, so a caller can bypass the parse-time
    // check entirely. `num_heads` of zero is an integer divide-by-zero in
    // `WindowAttention::from_weights` if it gets that far.
    let mut config = tiny_config();
    config.num_heads = vec![0, 4];
    let weights = tiny_weights(&config);
    let err = load_error(
        Florence2DaViT::from_weights(&weights, &config, ""),
        "a hand-built invalid config must be rejected",
    );
    assert!(err.contains("num_heads"), "{err}");
}

#[test]
fn from_weights_rejects_unsanitized_conv_weights() {
    let config = tiny_config();

    // PyTorch layout (O, I, kH, kW) is what `sanitize` exists to remap. Left
    // unremapped it reaches MLX `conv2d`, which throws at graph build.
    let mut patch_embed = tiny_weights(&config);
    patch_embed.insert("convs.1.proj.weight".into(), seq_array(&[16, 8, 3, 3], 7.0));
    let err = load_error(
        Florence2DaViT::from_weights(&patch_embed, &config, ""),
        "an unsanitized patch-embed weight must fail the load",
    );
    assert!(err.contains("convs.1.proj.weight"), "{err}");
    assert!(err.contains("sanitize"), "{err}");

    let mut depthwise = tiny_weights(&config);
    depthwise.insert(
        "blocks.0.0.spatial_block.conv1.fn.dw.weight".into(),
        seq_array(&[8, 1, 3, 3], 8.0),
    );
    let err = load_error(
        Florence2DaViT::from_weights(&depthwise, &config, ""),
        "an unsanitized depthwise weight must fail the load",
    );
    assert!(err.contains("dw.weight"), "{err}");
    assert!(err.contains("sanitize"), "{err}");
}

#[test]
fn missing_weight_reports_the_key() {
    let config = tiny_config();
    let mut weights = tiny_weights(&config);
    weights.remove("blocks.1.0.channel_block.channel_attn.fn.qkv.weight");
    let err = load_error(
        Florence2DaViT::from_weights(&weights, &config, ""),
        "a missing block weight must fail the load",
    );
    assert!(
        err.contains("channel_attn.fn.qkv.weight"),
        "error should name the missing key: {err}"
    );
}
