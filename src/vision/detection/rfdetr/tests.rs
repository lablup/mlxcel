// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! Focused RF-DETR graph and geometry tests.

use mlxcel_core::weights::WeightMap;

use super::backbone::{window_partition, window_unpartition};
use super::layers::ConvNorm;
use super::model::bbox_reparameterize;
use super::postprocess::DecodedDetection;
use super::predictor::decode_masks;
use super::projector::Projector;
use crate::vision::dense::interpolate::{resize_bicubic_nhwc, resize_bilinear_nhwc};

fn values(array: &mlxcel_core::MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(array, mlxcel_core::dtype::FLOAT32))
}

#[test]
fn window_partition_round_trips_spatial_tokens() {
    let input: Vec<f32> = (0..4 * 4 * 384).map(|value| value as f32).collect();
    let input = mlxcel_core::from_slice_f32(&input, &[1, 16, 384]);
    let windows = window_partition(&input, 1, 4, 4, 2);
    assert_eq!(mlxcel_core::array_shape(&windows), [4, 4, 384]);
    let output = window_unpartition(&windows, 1, 4, 4, 2);
    assert_eq!(mlxcel_core::array_shape(&output), [1, 16, 384]);
    assert_eq!(values(&output), values(&input));
}

#[test]
fn bbox_reparameterization_matches_hand_computation() {
    let delta = mlxcel_core::from_slice_f32(&[0.5, -1.0, 2.0_f32.ln(), 0.5_f32.ln()], &[1, 1, 4]);
    let reference = mlxcel_core::from_slice_f32(&[0.4, 0.3, 0.2, 0.1], &[1, 1, 4]);
    let actual = values(&bbox_reparameterize(&delta, &reference));
    for (actual, expected) in actual.iter().zip([0.5, 0.2, 0.4, 0.05]) {
        assert!((actual - expected).abs() < 1.0e-6, "{actual} != {expected}");
    }
}

fn insert_conv_norm(weights: &mut WeightMap, prefix: &str) {
    for key in ["conv.weight", "bn.weight", "bn.bias"] {
        weights.insert(
            format!("{prefix}.{key}"),
            mlxcel_core::from_slice_f32(&[1.0], &[1]),
        );
    }
}

/// rfdetr's channel LayerNorm defaults to eps 1e-6, and the checkpoint's
/// `ConvX.bn` and the projector's final norm use that default (mlx-vlm#2709).
#[test]
fn projector_norms_use_eps_1e6() {
    let mut weights = WeightMap::new();
    insert_conv_norm(&mut weights, "single");
    let single = ConvNorm::from_weights(&weights, "single", 128, 3).unwrap();
    assert_eq!(single.norm_eps(), 1.0e-6);

    let prefix = "projector.stages.0.0";
    insert_conv_norm(&mut weights, &format!("{prefix}.cv1"));
    insert_conv_norm(&mut weights, &format!("{prefix}.cv2"));
    for index in 0..3 {
        insert_conv_norm(&mut weights, &format!("{prefix}.m.{index}.cv1"));
        insert_conv_norm(&mut weights, &format!("{prefix}.m.{index}.cv2"));
    }
    for key in ["weight", "bias"] {
        weights.insert(
            format!("projector.stages.0.1.{key}"),
            mlxcel_core::from_slice_f32(&[1.0], &[1]),
        );
    }
    let eps = Projector::from_weights(&weights).unwrap().norm_eps();
    // cv1, three bottlenecks of two, cv2, then the final norm.
    assert_eq!(eps.len(), 9);
    assert!(eps.iter().all(|&eps| eps == 1.0e-6), "{eps:?}");
}

/// rfdetr's `PostProcess` upsamples mask logits with
/// `F.interpolate(mode="bilinear", align_corners=False)` and keeps `> 0`
/// (mlx-vlm#2713). The input is chosen so that a cubic resize thresholds to a
/// different 4x4 plane; `[[-1, 1], [1, -1]]` does not separate the two, since
/// both resizes keep its sign pattern.
#[test]
fn mask_resize_is_bilinear_half_pixel() {
    let logits = [-1.0f32, 3.2, 3.2, -1.0];
    let detection = DecodedDetection {
        bbox: [0.0; 4],
        score: 1.0,
        label: 0,
        class_name: String::new(),
        query_index: 0,
    };
    let pred_masks = mlxcel_core::from_slice_f32(&logits, &[1, 1, 2, 2]);
    let masks = decode_masks(&pred_masks, &[detection], 1, 4, 4).unwrap();
    let actual = masks[0].as_ref().expect("mask plane").as_raw().clone();

    let nhwc = mlxcel_core::from_slice_f32(&logits, &[1, 2, 2, 1]);
    let binary = |plane: &[f32]| -> Vec<u8> {
        plane
            .iter()
            .map(|&value| if value > 0.0 { 255 } else { 0 })
            .collect()
    };
    let bilinear = binary(&values(
        &resize_bilinear_nhwc(&nhwc, (4, 4), false, false).unwrap(),
    ));
    let bicubic = binary(&values(
        &resize_bicubic_nhwc(&nhwc, Some((4, 4)), None, false).unwrap(),
    ));
    #[rustfmt::skip]
    let expected: Vec<u8> = [
        0, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 0,
    ]
    .iter()
    .map(|&on| on * 255)
    .collect();
    assert_eq!(bilinear, expected);
    assert_ne!(
        bicubic, bilinear,
        "the input must separate cubic from bilinear"
    );
    assert_eq!(actual, bilinear);
}
