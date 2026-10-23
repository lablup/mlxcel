// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Shared DINO trunks and point-patch condition encoders.
use super::super::{
    interpolate::{resize_bilinear_nhwc, resize_nearest_nhwc},
    preprocess::normalize_imagenet,
};
use super::{
    bundle::array,
    config::Config,
    layers::{LayerNorm, Linear, layer_norm},
    processing::Inputs,
};
use crate::vision::encoders::dinov2::{Attention, Dinov2, Mlp};
use anyhow::{Context, Result, ensure};
use mlxcel_core::{self as mx, MlxArray, UniquePtr, weights::WeightMap};
use std::collections::BTreeMap;
type Array = UniquePtr<MlxArray>;
pub type FeatureCache = BTreeMap<String, (Array, Array)>;
fn shaped_array(w: &WeightMap, key: &str, shape: &[i32]) -> Result<Array> {
    let value = array(w, key)?;
    ensure!(
        mx::array_shape(&value) == shape,
        "Objects weight {key}: expected {shape:?}, got {:?}",
        mx::array_shape(&value)
    );
    Ok(value)
}

struct PointPatch {
    projection: Linear,
    invalid: Array,
    position: Array,
    window_position: Array,
    cls: Array,
    norm1: LayerNorm,
    attention: Attention,
    norm2: LayerNorm,
    mlp: Mlp,
    size: usize,
    patch: usize,
}
impl PointPatch {
    fn from_weights(w: &WeightMap, p: &str, c: &Config) -> Result<Self> {
        c.validate()?;
        let channels = c.point_channels as i32;
        let grid = (c.point_size / c.point_patch_size) as i32;
        let invalid = shaped_array(w, &format!("{p}.invalid_xyz_token"), &[channels])?;
        let position = shaped_array(w, &format!("{p}.pos_embed"), &[1, channels, grid, grid])?;
        let window_position = shaped_array(
            w,
            &format!("{p}.pos_embed_window"),
            &[1, 1 + c.point_patch_size.pow(2) as i32, channels],
        )?;
        let cls = shaped_array(w, &format!("{p}.cls_token"), &[1, 1, channels])?;
        Ok(Self {
            projection: Linear::from_weights(w, &format!("{p}.point_proj"))?,
            invalid,
            position,
            window_position,
            cls,
            norm1: LayerNorm::from_weights(w, &format!("{p}.blocks.0.norm1"), 1e-6)?,
            attention: Attention::from_weights(
                w,
                &format!("{p}.blocks.0.attn"),
                c.point_channels,
                c.point_heads,
                true,
            )
            .map_err(anyhow::Error::msg)?,
            norm2: LayerNorm::from_weights(w, &format!("{p}.blocks.0.norm2"), 1e-6)?,
            mlp: Mlp::from_weights(
                w,
                &format!("{p}.blocks.0.mlp"),
                c.point_channels,
                c.point_channels * 2,
            )
            .map_err(anyhow::Error::msg)?,
            size: c.point_size,
            patch: c.point_patch_size,
        })
    }
    fn forward(&self, points: &MlxArray) -> Result<Array> {
        let points = resize_nearest_nhwc(points, (self.size, self.size))?;
        let valid = mx::all_axis(&mx::isfinite(&points), -1, true);
        let zero = mx::from_slice_f32(&[0.0], &[]);
        let points = mx::astype(
            &mx::where_cond(&valid, &points, &zero),
            mx::array_dtype(&self.projection.weight),
        );
        let x = self.projection.forward(&points);
        let x = mx::where_cond(&valid, &x, &self.invalid);
        let s = mx::array_shape(&x);
        let p = self.patch as i32;
        let x = mx::reshape(
            &mx::transpose_axes(
                &mx::reshape(&x, &[s[0], s[1] / p, p, s[2] / p, p, s[3]]),
                &[0, 1, 3, 2, 4, 5],
            ),
            &[-1, p * p, s[3]],
        );
        let n = mx::array_shape(&x)[0];
        let mut x = mx::add(
            &mx::concatenate(&mx::broadcast_to(&self.cls, &[n, 1, s[3]]), &x, 1),
            &self.window_position,
        );
        // Each point patch is an independent 65-token attention sequence. Chunk
        // only the batch dimension to honor the shared DINO attention budget.
        let normalized = self.norm1.forward(&x);
        let mut attended = Vec::new();
        for start in (0..n).step_by(512) {
            let end = (start + 512).min(n);
            attended.push(self.attention.forward(&mx::slice(
                &normalized,
                &[start, 0, 0],
                &[end, p * p + 1, s[3]],
            ))?);
        }
        let attended = mx::concatenate_many(&attended.iter().map(|v| &**v).collect::<Vec<_>>(), 0);
        x = mx::add(&x, &attended);
        x = mx::add(&x, &self.mlp.forward_objects(&self.norm2.forward(&x))?);
        let x = mx::reshape(&mx::slice(&x, &[0, 0, 0], &[n, 1, s[3]]), &[s[0], -1, s[3]]);
        Ok(mx::add(
            &x,
            &mx::reshape(
                &mx::transpose_axes(&self.position, &[0, 2, 3, 1]),
                &[1, -1, s[3]],
            ),
        ))
    }
}
struct Projection {
    norm: LayerNorm,
    w1: Linear,
    w2: Linear,
    w3: Linear,
}
impl Projection {
    fn from_weights(w: &WeightMap, p: &str) -> Result<Self> {
        Ok(Self {
            norm: LayerNorm::from_weights(w, &format!("{p}.0"), 1e-5)?,
            w1: Linear::from_weights(w, &format!("{p}.1.w1"))?,
            w2: Linear::from_weights(w, &format!("{p}.1.w2"))?,
            w3: Linear::from_weights(w, &format!("{p}.1.w3"))?,
        })
    }
    fn forward(&self, x: &MlxArray) -> Array {
        let x = self.norm.forward(x);
        self.w2.forward(&mx::multiply(
            &mx::compiled_silu(&self.w1.forward(&x)),
            &self.w3.forward(&x),
        ))
    }
}
pub struct ConditionEncoder {
    projections: Vec<Projection>,
    point: Option<PointPatch>,
    index: Array,
    size: usize,
    point_count: usize,
}
impl ConditionEncoder {
    pub fn from_weights(w: &WeightMap, p: &str, c: &Config, pointmap: bool) -> Result<Self> {
        c.validate()?;
        let index = shaped_array(w, &format!("{p}.idx_emb"), &[3, c.cond_channels as i32])?;
        Ok(Self {
            projections: (0..if pointmap { 3 } else { 2 })
                .map(|i| Projection::from_weights(w, &format!("{p}.projection_nets.{i}")))
                .collect::<Result<_>>()?,
            point: pointmap
                .then(|| PointPatch::from_weights(w, &format!("{p}.module_list.2"), c))
                .transpose()?,
            index,
            size: c.image_size,
            point_count: (c.point_size / c.point_patch_size).pow(2),
        })
    }
    pub fn forward(
        &self,
        backbones: &[Dinov2],
        inputs: &Inputs,
        cache: &mut FeatureCache,
    ) -> Result<Array> {
        let groups = [
            ["image", "rgb_image"],
            ["mask", "rgb_image_mask"],
            ["pointmap", "rgb_pointmap"],
        ];
        let mut tokens = Vec::new();
        for (i, projection) in self.projections.iter().enumerate() {
            for (pos, name) in groups[i].iter().enumerate() {
                if i == 2 && !inputs.contains_key(*name) {
                    tokens.push(mx::zeros(
                        &[1, self.point_count as i32, mx::array_shape(&self.index)[1]],
                        mx::array_dtype(&self.index),
                    ));
                    continue;
                }
                let input = inputs
                    .get(*name)
                    .with_context(|| format!("missing condition input {name}"))?;
                let x = if i < 2 {
                    let shared = backbones.len() == 1;
                    let backbone = &backbones[if shared {
                        0
                    } else {
                        if self.point.is_some() { i } else { i + 2 }
                    }];
                    let cache_key = if shared {
                        (*name).to_string()
                    } else {
                        format!(
                            "{}:{name}",
                            if self.point.is_some() { "ss" } else { "slat" }
                        )
                    };
                    if !cache.contains_key(&cache_key) {
                        let mut x =
                            resize_bilinear_nhwc(input, (self.size, self.size), false, false)?;
                        if mx::array_shape(&x)[3] == 1 {
                            x = mx::concatenate_many(&[&x, &x, &x], -1);
                        }
                        let x = mx::astype(&normalize_imagenet(&x)?, backbone.dtype());
                        let f = backbone.forward_features(&x, None)?;
                        cache.insert(cache_key.clone(), (f.x_norm, f.x_prenorm));
                    }
                    let (norm, prenorm) = cache
                        .get(&cache_key)
                        .context("DINO feature cache missing")?;
                    if self.point.is_some() {
                        let s = mx::array_shape(norm);
                        mx::concatenate(
                            &mx::slice(norm, &[0, 0, 0], &[s[0], 1, s[2]]),
                            &mx::slice(norm, &[0, 5, 0], &s),
                            1,
                        )
                    } else {
                        layer_norm(prenorm, 1e-5)
                    }
                } else {
                    self.point
                        .as_ref()
                        .context("point encoder missing")?
                        .forward(input)?
                };
                let x = projection.forward(&x);
                let c = mx::array_shape(&self.index)[1];
                tokens.push(mx::add(
                    &x,
                    &mx::reshape(
                        &mx::slice(&self.index, &[pos as i32, 0], &[pos as i32 + 1, c]),
                        &[1, 1, c],
                    ),
                ));
            }
        }
        Ok(mx::concatenate_many(
            &tokens.iter().map(|v| &**v).collect::<Vec<_>>(),
            1,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sam3d_objects_condition_constructor_rejects_malformed_rank() {
        let mut weights = WeightMap::new();
        weights.insert("condition.idx_emb".into(), mx::from_slice_f32(&[0.0], &[]));
        let result =
            ConditionEncoder::from_weights(&weights, "condition", &Config::default(), false);
        assert!(
            result
                .err()
                .unwrap()
                .to_string()
                .contains("expected [3, 1024]")
        );
    }
    #[test]
    fn sam3d_objects_point_constructor_rejects_wrong_anchor() {
        let mut weights = WeightMap::new();
        weights.insert(
            "point.invalid_xyz_token".into(),
            mx::zeros(&[512], mx::dtype::FLOAT32),
        );
        weights.insert(
            "point.pos_embed".into(),
            mx::zeros(&[1, 32, 32, 512], mx::dtype::FLOAT32),
        );
        let result = PointPatch::from_weights(&weights, "point", &Config::default());
        assert!(
            result
                .err()
                .unwrap()
                .to_string()
                .contains("expected [1, 512, 32, 32]")
        );
    }
}
