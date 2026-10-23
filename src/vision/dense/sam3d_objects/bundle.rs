// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Staged, dtype-preserving bundle loading and verified backbone sharing.
use super::{
    config::Config,
    decoders::{GaussianDecoder, MeshDecoder, StructureDecoder},
    flow::{LatentFlow, StructureFlow},
    vision::ConditionEncoder,
};
use crate::vision::encoders::dinov2::{Dinov2, Dinov2Config};
use anyhow::{Context, Result, ensure};
use mlxcel_core::{
    self as mx, MlxArray, UniquePtr,
    weights::{WeightMap, load_weights_from_dir_index_filtered},
};
use std::path::Path;
const BACKBONES: [&str; 4] = [
    "ss_condition_embedder.module_list.0.backbone.",
    "ss_condition_embedder.module_list.1.backbone.",
    "slat_condition_embedder.module_list.0.backbone.",
    "slat_condition_embedder.module_list.1.backbone.",
];
pub fn array(w: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>> {
    w.get(key)
        .map(|v| mx::copy(v))
        .with_context(|| format!("Objects weight missing: {key}"))
}
pub fn config(path: &Path) -> Result<Config> {
    let c: Config = serde_json::from_str(&std::fs::read_to_string(path.join("config.json"))?)?;
    c.validate()?;
    Ok(c)
}
fn scoped(path: &Path, prefix: &str) -> Result<WeightMap> {
    let w = load_weights_from_dir_index_filtered(path, |n| n.starts_with(prefix))
        .map_err(anyhow::Error::msg)?;
    ensure!(!w.is_empty(), "empty Objects weight scope {prefix}");
    Ok(w.into_iter()
        .map(|(k, v)| (k[prefix.len()..].to_string(), v))
        .collect())
}
fn bf16(w: &WeightMap) -> Result<()> {
    for (name, v) in w {
        ensure!(
            mx::array_dtype(v) == mx::dtype::BFLOAT16,
            "Objects weight {name} must remain BF16"
        );
    }
    Ok(())
}
fn equal(a: &WeightMap, b: &WeightMap) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().all(|(k, x)| {
        b.get(k).is_some_and(|y| {
            if mx::array_shape(x) != mx::array_shape(y) || mx::array_dtype(x) != mx::array_dtype(y)
            {
                return false;
            }
            let same = mx::array_equal(x, y, false);
            mx::eval(&same);
            mx::item_bool(&same)
        })
    })
}
fn dino(w: &WeightMap, c: &Config) -> Result<Dinov2> {
    let dc = Dinov2Config {
        hidden_size: c.dino_hidden_size,
        num_hidden_layers: c.dino_layers,
        num_attention_heads: c.dino_heads,
        image_size: c.dino_image_size,
        num_register_tokens: 4,
        interpolate_offset: 0.1,
        intermediate_layers: Vec::new(),
        ..Dinov2Config::vitl14()
    };
    Ok(Dinov2::from_weights(w, dc)
        .map_err(anyhow::Error::msg)?
        .with_objects_policy())
}
pub struct Model {
    pub backbones: Vec<Dinov2>,
    pub structure_condition: ConditionEncoder,
    pub latent_condition: ConditionEncoder,
    pub structure: StructureFlow,
    pub latent: LatentFlow,
    pub occupancy: StructureDecoder,
    pub gaussian: GaussianDecoder,
    pub gaussian_4: GaussianDecoder,
    pub mesh: MeshDecoder,
}
impl Model {
    pub fn load(path: &Path, c: &Config) -> Result<Self> {
        let schema = super::checkpoint_schema::expected(c)?;
        let reference = scoped(path, BACKBONES[0])?;
        super::checkpoint_schema::validate_loaded(&reference, &schema, BACKBONES[0])?;
        bf16(&reference)?;
        let mut shared = true;
        for prefix in &BACKBONES[1..] {
            let candidate = scoped(path, prefix)?;
            bf16(&candidate)?;
            super::checkpoint_schema::validate_loaded(&candidate, &schema, prefix)?;
            shared &= equal(&reference, &candidate);
        }
        let mut backbones = vec![dino(&reference, c)?];
        drop(reference);
        if !shared {
            for prefix in &BACKBONES[1..] {
                let candidate = scoped(path, prefix)?;
                super::checkpoint_schema::validate_loaded(&candidate, &schema, prefix)?;
                backbones.push(dino(&candidate, c)?);
            }
        }
        let weights = load_weights_from_dir_index_filtered(path, |n| {
            !n.starts_with("depth_model.") && !BACKBONES.iter().any(|p| n.starts_with(p))
        })
        .map_err(anyhow::Error::msg)?;
        bf16(&weights)?;
        super::checkpoint_schema::validate_loaded(&weights, &schema, "")?;
        let allowed = [
            "ss_generator.",
            "slat_generator.",
            "ss_condition_embedder.",
            "slat_condition_embedder.",
            "ss_decoder.",
            "slat_decoder_gs.",
            "slat_decoder_gs_4.",
            "slat_decoder_mesh.",
        ];
        ensure!(
            weights
                .keys()
                .all(|k| allowed.iter().any(|p| k.starts_with(p))),
            "unexpected Objects checkpoint scope"
        );
        Ok(Self {
            backbones,
            structure_condition: ConditionEncoder::from_weights(
                &weights,
                "ss_condition_embedder",
                c,
                true,
            )?,
            latent_condition: ConditionEncoder::from_weights(
                &weights,
                "slat_condition_embedder",
                c,
                false,
            )?,
            structure: StructureFlow::from_weights(&weights, "ss_generator", c)?,
            latent: LatentFlow::from_weights(&weights, "slat_generator", c)?,
            occupancy: StructureDecoder::from_weights(&weights, "ss_decoder", c)?,
            gaussian: GaussianDecoder::from_weights(&weights, "slat_decoder_gs", c)?,
            gaussian_4: GaussianDecoder::from_weights(&weights, "slat_decoder_gs_4", c)?,
            mesh: MeshDecoder::from_weights(&weights, "slat_decoder_mesh", c)?,
        })
    }
}
pub fn estimate_depth(
    path: &Path,
    c: &Config,
    image: &image::DynamicImage,
) -> Result<Option<UniquePtr<MlxArray>>> {
    let Some(depth_config) = c.depth_model.clone() else {
        return Ok(None);
    };
    let weights = scoped(path, "depth_model.")?;
    let schema = super::checkpoint_schema::expected(c)?;
    super::checkpoint_schema::validate_loaded(&weights, &schema, "depth_model.")?;
    let depth = super::super::moge3::Moge3::from_weights(depth_config, weights)?;
    let options = super::super::moge3::InferOptions {
        num_tokens: c
            .depth_num_tokens
            .unwrap_or(depth.config().num_tokens_range[1]),
        force_projection: false,
        apply_mask: false,
        fov_x: None,
        refine_steps: if depth.config().refiner.is_some() {
            3
        } else {
            0
        },
        return_per_step: false,
    };
    let output = depth.estimate_with_options(&super::processing::rgb(image)?, options)?;
    let points = mx::multiply(
        &output.points,
        &mx::from_slice_f32(&[-1.0, -1.0, 1.0], &[3]),
    );
    let mask = mx::expand_dims(&output.mask, -1);
    let points = mx::where_cond(&mask, &points, &mx::from_slice_f32(&[f32::NAN], &[]));
    let points = if mx::array_shape(&points).len() == 4 {
        mx::squeeze_axis(&points, 0)
    } else {
        points
    };
    mx::eval(&points);
    Ok(Some(points))
}
