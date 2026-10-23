// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! Complete released RF-DETR inference graph.

use std::path::Path;

use mlxcel_core::layers::Linear;
use mlxcel_core::weights::{WeightMap, load_weights_from_dir};
use mlxcel_core::{MlxArray, UniquePtr};

use super::backbone::WindowedDinov2;
use super::config::RfDetrConfig;
use super::layers::{Activation, Mlp, concat};
use super::projector::Projector;
use super::segmentation::SegmentationHead;
use super::transformer::Transformer;
use super::{checkpoint, sanitize, weights};

pub struct RfDetrOutput {
    pub pred_logits: UniquePtr<MlxArray>,
    pub pred_boxes: UniquePtr<MlxArray>,
    pub pred_masks: Option<UniquePtr<MlxArray>>,
}

pub struct RfDetrModel {
    config: RfDetrConfig,
    weight_dtype: i32,
    backbone: WindowedDinov2,
    projector: Projector,
    transformer: Transformer,
    class_head: Linear,
    box_head: Mlp,
    segmentation: Option<SegmentationHead>,
}

impl RfDetrModel {
    pub fn load<P: AsRef<Path>>(directory: P) -> Result<Self, String> {
        let directory = directory.as_ref();
        let config_path = directory.join("config.json");
        let config_text = crate::vision::detection::read_bounded_metadata(&config_path)?;
        let config = RfDetrConfig::from_json_str(&config_text)?;
        let preflight = checkpoint::inspect(directory)?;
        let raw = load_weights_from_dir(directory)?;
        preflight.validate_loaded(&raw)?;
        let weights = if sanitize::needs_sanitize(&raw)? {
            sanitize::sanitize(raw)?
        } else {
            raw
        };
        Self::from_weights(weights, config)
    }

    pub fn from_weights(weights_map: WeightMap, config: RfDetrConfig) -> Result<Self, String> {
        config.validate()?;
        let weight_dtype = weights::validate(&weights_map, &config)?;
        Ok(Self {
            backbone: WindowedDinov2::from_weights(&weights_map, &config)?,
            projector: Projector::from_weights(&weights_map)?,
            transformer: Transformer::from_weights(&weights_map, &config)?,
            class_head: Linear::from_weights(&weights_map, "class_embed")?,
            box_head: Mlp::from_weights(&weights_map, "bbox_embed", 3, Activation::Relu)?,
            segmentation: config
                .segmentation
                .then(|| SegmentationHead::from_weights(&weights_map, config.seg_num_blocks))
                .transpose()?,
            config,
            weight_dtype,
        })
    }

    pub fn config(&self) -> &RfDetrConfig {
        &self.config
    }

    pub fn weight_dtype(&self) -> i32 {
        self.weight_dtype
    }

    pub fn forward(&self, pixel_values: &MlxArray) -> Result<RfDetrOutput, String> {
        self.forward_with_masks(pixel_values, true)
    }

    pub fn forward_with_masks(
        &self,
        pixel_values: &MlxArray,
        include_masks: bool,
    ) -> Result<RfDetrOutput, String> {
        self.validate_input(pixel_values)?;
        let features = self.backbone.forward(pixel_values)?;
        let memory = self.projector.forward(&features)?;
        let shape = mlxcel_core::array_shape(&memory);
        let spatial = (shape[1], shape[2]);
        let flat = mlxcel_core::reshape(&memory, &[shape[0], shape[1] * shape[2], shape[3]]);
        let (hidden, reference) = self.transformer.forward(&flat, spatial);
        let pred_logits = self.class_head.forward(&hidden);
        let delta = self.box_head.forward(&hidden);
        let pred_boxes = bbox_reparameterize(&delta, &reference);
        let pred_masks = materialize_optional_masks(include_masks, || {
            self.segmentation
                .as_ref()
                .map(|head| {
                    head.forward(
                        &memory,
                        &hidden,
                        (self.config.resolution, self.config.resolution),
                    )
                })
                .transpose()
        })?;
        let finite = [
            mlxcel_core::all_all(&mlxcel_core::isfinite(&pred_logits)),
            mlxcel_core::all_all(&mlxcel_core::isfinite(&pred_boxes)),
        ];
        if !finite.iter().all(|check| mlxcel_core::item_bool(check)) {
            return Err("RF-DETR produced non-finite detection outputs".into());
        }
        if let Some(masks) = &pred_masks
            && !mlxcel_core::item_bool(&mlxcel_core::all_all(&mlxcel_core::isfinite(masks)))
        {
            return Err("RF-DETR produced non-finite mask logits".into());
        }
        Ok(RfDetrOutput {
            pred_logits,
            pred_boxes,
            pred_masks,
        })
    }

    fn validate_input(&self, input: &MlxArray) -> Result<(), String> {
        let shape = mlxcel_core::array_shape(input);
        let resolution = i32::try_from(self.config.resolution)
            .map_err(|_| "RF-DETR resolution exceeds backend limits")?;
        if shape != [1, resolution, resolution, 3] {
            return Err(format!(
                "RF-DETR expects [1,{resolution},{resolution},3] input, got {shape:?}"
            ));
        }
        let dtype = mlxcel_core::array_dtype(input);
        if dtype != self.weight_dtype {
            return Err(format!(
                "RF-DETR input dtype {dtype} does not match checkpoint dtype {}",
                self.weight_dtype
            ));
        }
        if !mlxcel_core::item_bool(&mlxcel_core::all_all(&mlxcel_core::isfinite(input))) {
            return Err("RF-DETR input contains non-finite values".into());
        }
        Ok(())
    }
}

fn materialize_optional_masks<T>(
    include_masks: bool,
    build: impl FnOnce() -> Result<Option<T>, String>,
) -> Result<Option<T>, String> {
    if include_masks { build() } else { Ok(None) }
}

pub(crate) fn bbox_reparameterize(delta: &MlxArray, reference: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(delta);
    let (batch, queries) = (shape[0], shape[1]);
    let delta_xy = mlxcel_core::slice(delta, &[0, 0, 0], &[batch, queries, 2]);
    let delta_wh = mlxcel_core::slice(delta, &[0, 0, 2], &[batch, queries, 4]);
    let reference_xy = mlxcel_core::slice(reference, &[0, 0, 0], &[batch, queries, 2]);
    let reference_wh = mlxcel_core::slice(reference, &[0, 0, 2], &[batch, queries, 4]);
    let pred_xy = mlxcel_core::add(
        &mlxcel_core::multiply(&delta_xy, &reference_wh),
        &reference_xy,
    );
    let pred_wh = mlxcel_core::multiply(&mlxcel_core::exp(&delta_wh), &reference_wh);
    concat(&[pred_xy, pred_wh], -1)
}

#[cfg(test)]
mod tests {
    use super::materialize_optional_masks;

    #[test]
    fn unrequested_masks_skip_the_mask_computation_entirely() {
        let skipped = materialize_optional_masks(false, || -> Result<Option<usize>, String> {
            panic!("mask computation must not run")
        })
        .unwrap();
        assert_eq!(skipped, None);

        let error = materialize_optional_masks(true, || -> Result<Option<usize>, String> {
            Err("invalid mask logits".into())
        })
        .unwrap_err();
        assert_eq!(error, "invalid mask logits");
    }
}
