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

//! RF-DETR checkpoint key and convolution-layout conversion.

use std::collections::{BTreeMap, HashMap};

use mlxcel_core::weights::WeightMap;

const RAW_PREFIX: &str = "model.";
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    Raw,
    Converted,
}

#[derive(Debug, Clone)]
enum PlannedAction {
    Keep {
        destination: String,
        transpose_conv: bool,
    },
    Split {
        destinations: [String; 3],
        segment: i32,
    },
}

/// Return whether the full checkpoint is in the raw HuggingFace namespace.
///
/// Mixed maps fail closed because converting them could collide with an
/// already-converted tensor or double-permute a convolution.
pub fn needs_sanitize(weights: &WeightMap) -> Result<bool, String> {
    super::checkpoint::validate_weight_map(weights)?;
    classify_names(weights.keys().map(String::as_str)).map(|layout| layout == Layout::Raw)
}

fn classify_names<'a>(names: impl IntoIterator<Item = &'a str>) -> Result<Layout, String> {
    let mut first_raw: Option<&str> = None;
    let mut first_converted: Option<&str> = None;
    for name in names {
        if name.starts_with(RAW_PREFIX) {
            first_raw = Some(first_raw.map_or(name, |old| old.min(name)));
        } else {
            first_converted = Some(first_converted.map_or(name, |old| old.min(name)));
        }
    }
    match (first_raw, first_converted) {
        (Some(_), None) => Ok(Layout::Raw),
        (None, Some(_)) => Ok(Layout::Converted),
        (Some(raw), Some(converted)) => Err(format!(
            "mixed RF-DETR checkpoint layouts: raw key {raw:?} and converted key {converted:?}"
        )),
        (None, None) => Err("RF-DETR checkpoint contains no tensors".into()),
    }
}

/// Apply the ordered key rewrites from the pinned mlx-vlm sanitizer.
fn rename_key(key: &str) -> String {
    let mut name = key.strip_prefix(RAW_PREFIX).unwrap_or(key).to_string();
    name = name.replace(
        "backbone.0.encoder.encoder.embeddings.",
        "backbone.embeddings.",
    );
    name = name.replace(
        "backbone.0.encoder.encoder.encoder.layer.",
        "backbone.encoder.layers.",
    );
    name = name.replace(
        "backbone.0.encoder.encoder.layernorm.",
        "backbone.layernorm.",
    );
    name = name.replace("backbone.0.projector.", "projector.");
    name = name.replace(".attention.attention.query.", ".attention.q_proj.");
    name = name.replace(".attention.attention.key.", ".attention.k_proj.");
    name = name.replace(".attention.attention.value.", ".attention.v_proj.");
    name = name.replace(".attention.output.dense.", ".attention.o_proj.");
    name = name.replace(".layer_scale1.lambda1", ".layer_scale1");
    name.replace(".layer_scale2.lambda1", ".layer_scale2")
}

fn is_conv_weight(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    name.ends_with(".weight")
        && (lower.contains("conv")
            || name.contains("spatial_features_proj")
            || name.contains("patch_embeddings.projection"))
}

fn reserve_destination(
    destinations: &mut BTreeMap<String, String>,
    destination: &str,
    source: &str,
) -> Result<(), String> {
    if let Some(previous) = destinations.insert(destination.to_string(), source.to_string()) {
        return Err(format!(
            "RF-DETR key collision: {previous:?} and {source:?} both map to {destination:?}"
        ));
    }
    Ok(())
}

fn qkv_destinations(name: &str, needle: &str, suffix: &str) -> [String; 3] {
    let base = name.replace(needle, "");
    ["q_proj", "k_proj", "v_proj"].map(|projection| format!("{base}{projection}.{suffix}"))
}

fn build_raw_plan(
    mut specs: Vec<(String, Vec<i32>)>,
) -> Result<BTreeMap<String, PlannedAction>, String> {
    specs.sort_by(|left, right| left.0.cmp(&right.0));
    let mut destinations = BTreeMap::new();
    let mut plan = BTreeMap::new();

    for (source, shape) in specs {
        let destination = rename_key(&source);
        if destination.contains("mask_token") {
            continue;
        }

        let action = if destination.contains("self_attn.in_proj_weight") {
            let expected_rows = shape.get(1).and_then(|columns| columns.checked_mul(3));
            if shape.len() != 2 || shape[0] <= 0 || shape[1] <= 0 || Some(shape[0]) != expected_rows
            {
                return Err(format!(
                    "invalid RF-DETR fused QKV weight shape for {source:?}: {shape:?}"
                ));
            }
            let split = qkv_destinations(&destination, "in_proj_weight", "weight");
            for target in &split {
                reserve_destination(&mut destinations, target, &source)?;
            }
            PlannedAction::Split {
                destinations: split,
                segment: shape[1],
            }
        } else if destination.contains("self_attn.in_proj_bias") {
            if shape.len() != 1 || shape[0] <= 0 || shape[0] % 3 != 0 {
                return Err(format!(
                    "invalid RF-DETR fused QKV bias shape for {source:?}: {shape:?}"
                ));
            }
            let split = qkv_destinations(&destination, "in_proj_bias", "bias");
            for target in &split {
                reserve_destination(&mut destinations, target, &source)?;
            }
            PlannedAction::Split {
                destinations: split,
                segment: shape[0] / 3,
            }
        } else {
            reserve_destination(&mut destinations, &destination, &source)?;
            let transpose_conv = shape.len() == 4
                && is_conv_weight(&destination)
                && !crate::loading::conv2d_weight_is_channel_last(&shape);
            PlannedAction::Keep {
                destination,
                transpose_conv,
            }
        };
        plan.insert(source, action);
    }
    Ok(plan)
}

fn split_bounds(index: usize, segment: i32) -> Result<(i32, i32), String> {
    let index = i32::try_from(index).map_err(|_| "RF-DETR QKV split index overflow")?;
    let start = index
        .checked_mul(segment)
        .ok_or("RF-DETR QKV split start overflow")?;
    let stop = start
        .checked_add(segment)
        .ok_or("RF-DETR QKV split stop overflow")?;
    Ok((start, stop))
}

/// Sanitize raw RF-DETR tensors, or return an already-converted map unchanged.
pub fn sanitize(weights: WeightMap) -> Result<WeightMap, String> {
    if !needs_sanitize(&weights)? {
        return Ok(weights);
    }

    let specs = weights
        .iter()
        .map(|(name, value)| (name.clone(), mlxcel_core::array_shape(value)))
        .collect();
    let mut plan = build_raw_plan(specs)?;
    let mut output = HashMap::with_capacity(plan.len());

    for (source, value) in weights {
        let Some(action) = plan.remove(&source) else {
            continue;
        };
        match action {
            PlannedAction::Keep {
                destination,
                transpose_conv,
            } => {
                let value = if transpose_conv {
                    mlxcel_core::transpose_axes(&value, &[0, 2, 3, 1])
                } else {
                    value
                };
                output.insert(destination, value);
            }
            PlannedAction::Split {
                destinations,
                segment,
            } => {
                for (index, destination) in destinations.into_iter().enumerate() {
                    let (start, stop) = split_bounds(index, segment)?;
                    output.insert(
                        destination,
                        mlxcel_core::utils::slice_axis(&value, 0, start, stop),
                    );
                }
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signature(weights: &WeightMap) -> Vec<(String, Vec<i32>)> {
        let mut result: Vec<_> = weights
            .iter()
            .map(|(name, value)| (name.clone(), mlxcel_core::array_shape(value)))
            .collect();
        result.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        result
    }

    #[test]
    fn provenance_is_global_and_mixed_layouts_fail() {
        assert_eq!(
            classify_names(["model.backbone.x", "model.transformer.y"]).unwrap(),
            Layout::Raw
        );
        assert_eq!(
            classify_names(["backbone.x", "transformer.y"]).unwrap(),
            Layout::Converted
        );
        assert!(classify_names(["model.backbone.x", "transformer.y"]).is_err());
        assert!(classify_names(std::iter::empty()).is_err());
    }

    #[test]
    fn ordered_renames_match_upstream() {
        assert_eq!(
            rename_key(
                "model.backbone.0.encoder.encoder.encoder.layer.3.attention.attention.query.weight"
            ),
            "backbone.encoder.layers.3.attention.q_proj.weight"
        );
        assert_eq!(
            rename_key("model.backbone.0.projector.stages.0.0.cv1.conv.weight"),
            "projector.stages.0.0.cv1.conv.weight"
        );
        assert_eq!(
            rename_key("model.backbone.0.encoder.encoder.encoder.layer.0.layer_scale1.lambda1"),
            "backbone.encoder.layers.0.layer_scale1"
        );
    }

    #[test]
    fn sanitizer_splits_qkv_transposes_raw_convs_and_is_idempotent() {
        let mut raw = WeightMap::new();
        raw.insert(
            "model.transformer.decoder.layers.0.self_attn.in_proj_weight".into(),
            mlxcel_core::from_slice_f32(
                &(0..12).map(|value| value as f32).collect::<Vec<_>>(),
                &[6, 2],
            ),
        );
        raw.insert(
            "model.transformer.decoder.layers.0.self_attn.in_proj_bias".into(),
            mlxcel_core::from_slice_f32(&[0.0; 6], &[6]),
        );
        raw.insert(
            "model.backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight".into(),
            mlxcel_core::from_slice_f32(&[0.0; 6], &[2, 3, 1, 1]),
        );
        raw.insert(
            "model.backbone.0.encoder.encoder.embeddings.mask_token".into(),
            mlxcel_core::from_slice_f32(&[0.0; 2], &[1, 2]),
        );
        raw.insert(
            "model.some.num_batches_tracked".into(),
            mlxcel_core::from_slice_f32(&[0.0], &[1]),
        );

        let once = sanitize(raw).unwrap();
        assert_eq!(
            mlxcel_core::array_shape(&once["transformer.decoder.layers.0.self_attn.q_proj.weight"]),
            [2, 2]
        );
        assert_eq!(
            mlxcel_core::array_shape(
                &once["backbone.embeddings.patch_embeddings.projection.weight"]
            ),
            [2, 1, 1, 3]
        );
        assert!(!once.keys().any(|name| name.contains("mask_token")));
        assert!(once.contains_key("some.num_batches_tracked"));
        assert!(!needs_sanitize(&once).unwrap());

        let once_signature = signature(&once);
        let twice = sanitize(once).unwrap();
        assert_eq!(signature(&twice), once_signature);
    }

    #[test]
    fn raw_names_with_channel_last_convs_are_not_double_transposed() {
        let plan = build_raw_plan(vec![(
            "model.segmentation_head.spatial_features_proj.weight".into(),
            vec![256, 1, 1, 256],
        )])
        .unwrap();
        assert!(matches!(
            &plan["model.segmentation_head.spatial_features_proj.weight"],
            PlannedAction::Keep {
                transpose_conv: false,
                ..
            }
        ));
    }

    #[test]
    fn malformed_fused_qkv_and_destination_collisions_fail() {
        let malformed = build_raw_plan(vec![(
            "model.transformer.decoder.layers.0.self_attn.in_proj_weight".into(),
            vec![7, 2],
        )])
        .unwrap_err();
        assert!(malformed.contains("fused QKV weight"));

        let collision = build_raw_plan(vec![
            (
                "model.transformer.decoder.layers.0.self_attn.in_proj_weight".into(),
                vec![6, 2],
            ),
            (
                "model.transformer.decoder.layers.0.self_attn.q_proj.weight".into(),
                vec![2, 2],
            ),
        ])
        .unwrap_err();
        assert!(collision.contains("collision"));

        let overflow = build_raw_plan(vec![(
            "model.transformer.decoder.layers.0.self_attn.in_proj_weight".into(),
            vec![i32::MAX, i32::MAX],
        )])
        .unwrap_err();
        assert!(overflow.contains("fused QKV weight"));
        assert!(split_bounds(2, i32::MAX).unwrap_err().contains("overflow"));
    }
}
