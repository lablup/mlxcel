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

//! Raw Paddle/HuggingFace to MLX checkpoint conversion.
//!
//! Raw checkpoints are identified by their `model.` namespace. Converted
//! checkpoints never carry that namespace and are returned unchanged, making
//! sanitization idempotent even for ambiguous convolution shapes.

use std::collections::{BTreeMap, HashMap};

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

const RAW_PREFIX: &str = "model.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    Raw,
    Converted,
}

#[derive(Debug, Clone)]
struct PlannedKey {
    destination: String,
    transpose_conv: bool,
}

/// Determine whether a checkpoint needs the raw-source conversion pipeline.
///
/// A mix of `model.*` and converted keys is rejected: choosing either layout
/// would leave half the graph under the wrong names, and choosing raw would
/// also double-transpose already-converted convolutions.
pub fn needs_sanitize(weights: &WeightMap) -> Result<bool, String> {
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
            "mixed PP-DocLayoutV3 checkpoint layouts: raw key {raw:?} and converted key \
             {converted:?}"
        )),
        (None, None) => Err("PP-DocLayoutV3 checkpoint contains no tensors".into()),
    }
}

/// Apply the ordered upstream renames to a single key.
pub fn rename_key(key: &str) -> String {
    let mut name = replace_prefix(key, RAW_PREFIX, "");
    name = replace_prefix(&name, "backbone.model.", "backbone.");
    name = replace_prefix(&name, "encoder.encoder.", "encoder.aifi.");
    name = rename_indexed_sequence(&name, "encoder_input_proj.");
    name = rename_indexed_sequence(&name, "decoder_input_proj.");
    name = replace_prefix(&name, "enc_output.0.", "enc_output.fc.");
    name = replace_prefix(&name, "enc_output.1.", "enc_output.ln.");
    name = name.replace(".convolution.", ".conv.");
    name = name.replace(".normalization.", ".bn.");
    name.replace(".norm.", ".bn.")
}

fn replace_prefix(value: &str, prefix: &str, replacement: &str) -> String {
    value
        .strip_prefix(prefix)
        .map_or_else(|| value.to_string(), |tail| format!("{replacement}{tail}"))
}

fn rename_indexed_sequence(key: &str, prefix: &str) -> String {
    let Some(rest) = key.strip_prefix(prefix) else {
        return key.to_string();
    };
    let Some((index, tail)) = rest.split_once('.') else {
        return key.to_string();
    };
    if index.is_empty() || !index.bytes().all(|byte| byte.is_ascii_digit()) {
        return key.to_string();
    }
    if let Some(field) = tail.strip_prefix("0.") {
        format!("{prefix}{index}.conv.{field}")
    } else if let Some(field) = tail.strip_prefix("1.") {
        format!("{prefix}{index}.bn.{field}")
    } else {
        key.to_string()
    }
}

fn should_drop_raw(key: &str) -> bool {
    key.ends_with(".num_batches_tracked") || key.starts_with("model.denoising_class_embed")
}

fn build_raw_plan(specs: Vec<(String, usize)>) -> Result<BTreeMap<String, PlannedKey>, String> {
    let mut specs = specs;
    specs.sort_by(|left, right| left.0.cmp(&right.0));
    let mut destinations: BTreeMap<String, String> = BTreeMap::new();
    let mut plan = BTreeMap::new();
    for (source, rank) in specs {
        if should_drop_raw(&source) {
            continue;
        }
        let destination = rename_key(&source);
        if let Some(previous) = destinations.insert(destination.clone(), source.clone()) {
            return Err(format!(
                "PP-DocLayoutV3 key collision: {previous:?} and {source:?} both map to \
                 {destination:?}"
            ));
        }
        let transpose_conv = rank == 4 && destination.ends_with(".conv.weight");
        plan.insert(
            source,
            PlannedKey {
                destination,
                transpose_conv,
            },
        );
    }
    Ok(plan)
}

/// Sanitize a raw checkpoint, or return an already-converted checkpoint as-is.
pub fn sanitize(weights: WeightMap) -> Result<WeightMap, String> {
    if !needs_sanitize(&weights)? {
        return Ok(weights);
    }

    let specs = weights
        .iter()
        .map(|(key, value)| (key.clone(), mlxcel_core::array_shape(value).len()))
        .collect();
    let mut plan = build_raw_plan(specs)?;
    let mut output: WeightMap = HashMap::with_capacity(plan.len());
    for (source, value) in weights {
        let Some(item) = plan.remove(&source) else {
            continue;
        };
        let value = if item.transpose_conv {
            transpose_raw_conv(value)
        } else {
            value
        };
        output.insert(item.destination, value);
    }
    Ok(output)
}

fn transpose_raw_conv(value: UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
    mlxcel_core::transpose_axes(&value, &[0, 2, 3, 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signature(weights: &WeightMap) -> Vec<(String, Vec<i32>)> {
        let mut signature: Vec<_> = weights
            .iter()
            .map(|(name, value)| (name.clone(), mlxcel_core::array_shape(value)))
            .collect();
        signature.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        signature
    }

    #[test]
    fn provenance_is_global_and_rejects_mixed_layouts() {
        assert_eq!(
            classify_names(["model.backbone.x", "model.decoder.y"]).unwrap(),
            Layout::Raw
        );
        assert_eq!(
            classify_names(["backbone.x", "decoder.y"]).unwrap(),
            Layout::Converted
        );
        assert!(
            classify_names(["model.backbone.x", "decoder.y"])
                .unwrap_err()
                .contains("mixed")
        );
        assert!(classify_names(std::iter::empty()).is_err());
    }

    #[test]
    fn ordered_renames_match_the_pinned_converter() {
        assert_eq!(
            rename_key("model.backbone.model.embedder.stem1.convolution.weight"),
            "backbone.embedder.stem1.conv.weight"
        );
        assert_eq!(
            rename_key("model.encoder.encoder.0.layers.0.self_attn.q_proj.weight"),
            "encoder.aifi.0.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            rename_key("model.encoder_input_proj.2.1.running_var"),
            "encoder_input_proj.2.bn.running_var"
        );
        assert_eq!(
            rename_key("model.decoder_input_proj.1.0.weight"),
            "decoder_input_proj.1.conv.weight"
        );
        assert_eq!(
            rename_key("model.enc_output.1.weight"),
            "enc_output.ln.weight"
        );
    }

    #[test]
    fn converted_names_are_rename_fixpoints() {
        for key in [
            "backbone.embedder.stem1.conv.weight",
            "encoder.aifi.0.layers.0.self_attn.q_proj.weight",
            "decoder_input_proj.2.bn.running_mean",
            "enc_output.fc.weight",
        ] {
            assert_eq!(rename_key(key), key);
        }
    }

    #[test]
    fn raw_plan_drops_only_source_auxiliary_tensors_and_marks_rank_four_convs() {
        let plan = build_raw_plan(vec![
            ("model.denoising_class_embed.weight".into(), 2),
            ("model.block.bn.num_batches_tracked".into(), 0),
            ("model.block.conv.weight".into(), 4),
            ("model.block.conv.bias".into(), 1),
            ("model.block.other.weight".into(), 4),
        ])
        .unwrap();
        assert_eq!(plan.len(), 3);
        assert!(plan["model.block.conv.weight"].transpose_conv);
        assert!(!plan["model.block.conv.bias"].transpose_conv);
        assert!(!plan["model.block.other.weight"].transpose_conv);
    }

    #[test]
    fn raw_plan_rejects_rename_collisions() {
        let error = build_raw_plan(vec![
            ("model.enc_output.0.weight".into(), 2),
            ("model.enc_output.fc.weight".into(), 2),
        ])
        .unwrap_err();
        assert!(error.contains("collision"));
        assert!(error.contains("enc_output.fc.weight"));
    }

    #[test]
    fn raw_and_converted_maps_have_identical_shapes_and_sanitize_is_idempotent() {
        let mut raw = WeightMap::new();
        raw.insert(
            "model.backbone.model.block.convolution.weight".into(),
            mlxcel_core::from_slice_f32(&[0.0; 6], &[2, 3, 1, 1]),
        );
        raw.insert(
            "model.backbone.model.block.convolution.bias".into(),
            mlxcel_core::from_slice_f32(&[0.0; 2], &[2]),
        );
        let once = sanitize(raw).unwrap();

        let mut converted = WeightMap::new();
        converted.insert(
            "backbone.block.conv.weight".into(),
            mlxcel_core::from_slice_f32(&[0.0; 6], &[2, 1, 1, 3]),
        );
        converted.insert(
            "backbone.block.conv.bias".into(),
            mlxcel_core::from_slice_f32(&[0.0; 2], &[2]),
        );
        assert_eq!(signature(&once), signature(&converted));

        let once_signature = signature(&once);
        let twice = sanitize(once).unwrap();
        assert_eq!(signature(&twice), once_signature);
        assert!(!needs_sanitize(&twice).unwrap());
    }
}
