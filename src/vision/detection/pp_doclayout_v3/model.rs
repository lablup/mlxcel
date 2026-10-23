// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! Full PP-DocLayoutV3 detector graph.

use std::path::Path;

use mlxcel_core::layers::{LayerNorm, Linear};
use mlxcel_core::weights::{WeightMap, load_weights_from_dir};
use mlxcel_core::{MlxArray, UniquePtr};

use crate::vision::detection::rt_detr_v2::hybrid_encoder::load_layer_norm;
use crate::vision::detection::rt_detr_v2::transformer::{
    Mlp, SpatialShape, generate_anchors, inverse_sigmoid,
};

use super::backbone::HgNetV2Backbone;
use super::config::PpDocLayoutV3Config;
use super::decoder::{Decoder, DecoderOutput};
use super::encoder::HybridEncoder;
use super::layers::{Activation, ConvNorm};
use super::mask_box::mask_to_box_coordinate;
use super::{sanitize, weights};

const MAX_INPUT_PIXELS: usize = 2_000_000;
const MAX_FEATURE_TOKENS: usize = 32_000;

pub struct PpDocLayoutV3Output {
    pub logits: UniquePtr<MlxArray>,
    pub pred_boxes: UniquePtr<MlxArray>,
    pub order_logits: UniquePtr<MlxArray>,
}

pub struct PpDocLayoutV3Model {
    config: PpDocLayoutV3Config,
    weight_dtype: i32,
    backbone: HgNetV2Backbone,
    encoder_input: Vec<ConvNorm>,
    encoder: HybridEncoder,
    encoder_output: Linear,
    encoder_output_norm: LayerNorm,
    score_head: Linear,
    box_head: Mlp,
    decoder_input: Vec<ConvNorm>,
    decoder: Decoder,
    decoder_norm: LayerNorm,
    mask_query_head: Mlp,
}

impl PpDocLayoutV3Model {
    pub fn load<P: AsRef<Path>>(directory: P) -> Result<Self, String> {
        let directory = directory.as_ref();
        let config_path = directory.join("config.json");
        let contents = std::fs::read_to_string(&config_path)
            .map_err(|error| format!("failed to read {}: {error}", config_path.display()))?;
        let config = PpDocLayoutV3Config::from_json_str(&contents)?;
        config.validate()?;
        let raw = load_weights_from_dir(directory)?;
        let weights = if sanitize::needs_sanitize(&raw)? {
            sanitize::sanitize(raw)?
        } else {
            raw
        };
        Self::from_weights(weights, config)
    }

    pub fn from_weights(
        weights_map: WeightMap,
        config: PpDocLayoutV3Config,
    ) -> Result<Self, String> {
        config.validate()?;
        let weight_dtype = weights::validate(&weights_map, &config)?;
        let eps = config.batch_norm_eps;
        let activation = Activation::None;
        let mut encoder_input = Vec::with_capacity(config.encoder_in_channels.len());
        for index in 0..config.encoder_in_channels.len() {
            encoder_input.push(ConvNorm::from_weights(
                &weights_map,
                &format!("encoder_input_proj.{index}"),
                1,
                0,
                1,
                activation,
                eps,
            )?);
        }
        let mut decoder_input = Vec::with_capacity(config.decoder_in_channels.len());
        for index in 0..config.decoder_in_channels.len() {
            decoder_input.push(ConvNorm::from_weights(
                &weights_map,
                &format!("decoder_input_proj.{index}"),
                1,
                0,
                1,
                activation,
                eps,
            )?);
        }
        Ok(Self {
            backbone: HgNetV2Backbone::from_weights(&weights_map, &config.backbone_config, eps)?,
            encoder: HybridEncoder::from_weights(&weights_map, &config)?,
            encoder_output: Linear::from_weights(&weights_map, "enc_output.fc")?,
            encoder_output_norm: load_layer_norm(
                &weights_map,
                "enc_output.ln",
                config.layer_norm_eps,
            )?,
            score_head: Linear::from_weights(&weights_map, "enc_score_head")?,
            box_head: Mlp::from_weights(&weights_map, "enc_bbox_head", 3)?,
            decoder: Decoder::from_weights(&weights_map, &config)?,
            decoder_norm: load_layer_norm(&weights_map, "decoder_norm", config.layer_norm_eps)?,
            mask_query_head: Mlp::from_weights(&weights_map, "mask_query_head", 3)?,
            config,
            weight_dtype,
            encoder_input,
            decoder_input,
        })
    }

    pub fn config(&self) -> &PpDocLayoutV3Config {
        &self.config
    }

    pub fn weight_dtype(&self) -> i32 {
        self.weight_dtype
    }

    pub fn forward(&self, pixel_values: &MlxArray) -> Result<PpDocLayoutV3Output, String> {
        self.validate_input(pixel_values)?;
        let pixel_values = mlxcel_core::astype(pixel_values, self.weight_dtype);
        #[cfg(test)]
        super::diagnostics::capture("input.pixel_values", &pixel_values);
        let mut backbone = self.backbone.forward(&pixel_values);
        let stride_four = backbone.remove(0);
        let projected: Vec<UniquePtr<MlxArray>> = self
            .encoder_input
            .iter()
            .zip(&backbone)
            .map(|(projection, feature)| projection.forward(feature))
            .collect();
        #[cfg(test)]
        for (index, feature) in projected.iter().enumerate() {
            super::diagnostics::capture(&format!("encoder.projected{index}"), feature);
        }
        let (pan, mask_feature) = self.encoder.forward(projected, &stride_four)?;
        let sources: Vec<UniquePtr<MlxArray>> = self
            .decoder_input
            .iter()
            .zip(&pan)
            .map(|(projection, feature)| projection.forward(feature))
            .collect();
        #[cfg(test)]
        for (index, source) in sources.iter().enumerate() {
            super::diagnostics::capture(&format!("decoder.source{index}"), source);
        }
        let shapes: Vec<SpatialShape> = sources
            .iter()
            .map(|feature| {
                let shape = mlxcel_core::array_shape(feature);
                (shape[1], shape[2])
            })
            .collect();
        let tokens: i32 = shapes.iter().map(|(height, width)| height * width).sum();
        if tokens as usize > MAX_FEATURE_TOKENS {
            return Err(format!(
                "PP-DocLayoutV3 feature pyramid has {tokens} tokens; limit is {MAX_FEATURE_TOKENS}"
            ));
        }
        let flat = flatten_features(&sources);
        #[cfg(test)]
        super::diagnostics::capture("decoder.flat", &flat);
        let (anchors, valid_mask) = generate_anchors(&shapes);
        let anchors = mlxcel_core::astype(&anchors, mlxcel_core::array_dtype(&flat));
        let valid_mask = mlxcel_core::astype(&valid_mask, mlxcel_core::array_dtype(&flat));
        let memory = mlxcel_core::multiply(&flat, &valid_mask);
        #[cfg(test)]
        super::diagnostics::capture("decoder.memory", &memory);
        let memory = self
            .encoder_output_norm
            .forward(&self.encoder_output.forward(&memory));
        let scores = self.score_head.forward(&memory);
        let coordinates = mlxcel_core::add(&self.box_head.forward(&memory), &anchors);
        #[cfg(test)]
        {
            super::diagnostics::capture("decoder.encoder_scores", &scores);
            super::diagnostics::capture("decoder.encoder_coordinates", &coordinates);
        }
        let top_queries = top_queries(&scores, self.config.num_queries as i32);
        let batch = mlxcel_core::array_shape(&flat)[0];
        let box_indices = mlxcel_core::broadcast_to(
            &mlxcel_core::expand_dims(&top_queries, 2),
            &[batch, self.config.num_queries as i32, 4],
        );
        let mut reference_logits = mlxcel_core::take_along_axis(&coordinates, &box_indices, 1);
        let content_indices = mlxcel_core::broadcast_to(
            &mlxcel_core::expand_dims(&top_queries, 2),
            &[
                batch,
                self.config.num_queries as i32,
                self.config.d_model as i32,
            ],
        );
        let target =
            mlxcel_core::stop_gradient(&mlxcel_core::take_along_axis(&memory, &content_indices, 1));
        if self.config.mask_enhanced {
            let query = self.decoder_norm.forward(&target);
            let mask_query = self.mask_query_head.forward(&query);
            let mask_shape = mlxcel_core::array_shape(&mask_feature);
            let mask_flat = mlxcel_core::reshape(
                &mask_feature,
                &[mask_shape[0], mask_shape[1] * mask_shape[2], mask_shape[3]],
            );
            let mask_flat = mlxcel_core::transpose_axes(&mask_flat, &[0, 2, 1]);
            let masks = mlxcel_core::matmul(&mask_query, &mask_flat);
            let masks = mlxcel_core::reshape(
                &masks,
                &[
                    mask_shape[0],
                    self.config.num_queries as i32,
                    mask_shape[1],
                    mask_shape[2],
                ],
            );
            let mask_boxes = mask_to_box_coordinate(&masks)?;
            reference_logits = inverse_sigmoid(&mask_boxes, 1.0e-5);
        }
        #[cfg(test)]
        {
            super::diagnostics::capture("decoder.target", &target);
            super::diagnostics::capture("decoder.reference_logits", &reference_logits);
        }
        let DecoderOutput {
            logits,
            boxes,
            order_logits,
        } = self.decoder.forward(
            &target,
            &reference_logits,
            &flat,
            &shapes,
            &self.box_head,
            &self.score_head,
            &self.decoder_norm,
        );
        Ok(PpDocLayoutV3Output {
            logits,
            pred_boxes: boxes,
            order_logits,
        })
    }

    fn validate_input(&self, pixel_values: &MlxArray) -> Result<(), String> {
        let shape = mlxcel_core::array_shape(pixel_values);
        if shape.len() != 4 || shape[0] != 1 || shape[3] != 3 {
            return Err(format!(
                "PP-DocLayoutV3 input must have shape [1,H,W,3], got {shape:?}"
            ));
        }
        if shape[1] != self.config.image_size as i32 || shape[2] != self.config.image_size as i32 {
            return Err(format!(
                "PP-DocLayoutV3 input must be {}x{}, got {}x{}",
                self.config.image_size, self.config.image_size, shape[1], shape[2]
            ));
        }
        let pixels = (shape[0] as usize)
            .checked_mul(shape[1] as usize)
            .and_then(|value| value.checked_mul(shape[2] as usize))
            .ok_or_else(|| "PP-DocLayoutV3 input pixel count overflow".to_string())?;
        if pixels > MAX_INPUT_PIXELS {
            return Err(format!(
                "PP-DocLayoutV3 input has {pixels} pixels; limit is {MAX_INPUT_PIXELS}"
            ));
        }
        let dtype = mlxcel_core::array_dtype(pixel_values);
        if !matches!(
            dtype,
            mlxcel_core::dtype::FLOAT16
                | mlxcel_core::dtype::BFLOAT16
                | mlxcel_core::dtype::FLOAT32
        ) {
            return Err(format!(
                "PP-DocLayoutV3 input must be floating point, got dtype {dtype}"
            ));
        }
        let finite = mlxcel_core::all_all(&mlxcel_core::isfinite(pixel_values));
        if !mlxcel_core::item_bool(&finite) {
            return Err("PP-DocLayoutV3 input contains non-finite values".into());
        }
        Ok(())
    }
}

fn flatten_features(features: &[UniquePtr<MlxArray>]) -> UniquePtr<MlxArray> {
    let flatten = |feature: &MlxArray| {
        let shape = mlxcel_core::array_shape(feature);
        mlxcel_core::reshape(feature, &[shape[0], shape[1] * shape[2], shape[3]])
    };
    let mut output = flatten(&features[0]);
    for feature in features.iter().skip(1) {
        output = mlxcel_core::concatenate(&output, &flatten(feature), 1);
    }
    output
}

fn top_queries(scores: &MlxArray, count: i32) -> UniquePtr<MlxArray> {
    let maximum = mlxcel_core::max_axis(scores, -1, false);
    let indices = mlxcel_core::argsort(&mlxcel_core::multiply_scalar(&maximum, -1.0), 1);
    let shape = mlxcel_core::array_shape(&indices);
    mlxcel_core::slice(&indices, &[0, 0], &[shape[0], count])
}
