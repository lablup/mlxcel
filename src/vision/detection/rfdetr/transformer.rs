// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! RF-DETR two-stage query selection and transformer decoder.

use mlxcel_core::layers::{LayerNorm, Linear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::vision::weights::load_dense_param;

use super::config::RfDetrConfig;
use super::decoder::Decoder;
use super::layers::{Activation, Mlp, concat, load_norm};

pub struct Transformer {
    encoder_output: Linear,
    encoder_norm: LayerNorm,
    encoder_class: Linear,
    encoder_box: Mlp,
    decoder: Decoder,
    query: UniquePtr<MlxArray>,
    reference: UniquePtr<MlxArray>,
    queries: i32,
}

impl Transformer {
    pub fn from_weights(weights: &WeightMap, config: &RfDetrConfig) -> Result<Self, String> {
        let stored_queries = config
            .num_queries
            .checked_mul(config.group_detr)
            .ok_or_else(|| "RF-DETR query table row count overflows usize".to_string())?;
        for key in ["query_feat.weight", "refpoint_embed.weight"] {
            if !weights.contains_key(key) {
                return Err(format!("RF-DETR weight not found: {key}"));
            }
        }
        Ok(Self {
            encoder_output: Linear::from_weights(weights, "transformer.enc_output.0")?,
            encoder_norm: load_norm(weights, "transformer.enc_output_norm.0", 1.0e-5)?,
            encoder_class: Linear::from_weights(weights, "transformer.enc_out_class_embed.0")?,
            encoder_box: Mlp::from_weights(
                weights,
                "transformer.enc_out_bbox_embed.0",
                3,
                Activation::Relu,
            )?,
            decoder: Decoder::from_weights(weights, config)?,
            query: load_dense_param(
                weights,
                "query_feat.weight",
                &[stored_queries as i32, config.hidden_dim as i32],
                None,
            )?,
            reference: load_dense_param(
                weights,
                "refpoint_embed.weight",
                &[stored_queries as i32, 4],
                None,
            )?,
            queries: config.num_queries as i32,
        })
    }

    pub fn forward(
        &self,
        memory: &MlxArray,
        spatial: (i32, i32),
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let shape = mlxcel_core::array_shape(memory);
        let batch = shape[0];
        let projected = self
            .encoder_norm
            .forward(&self.encoder_output.forward(memory));
        let logits = self.encoder_class.forward(&projected);
        let delta = self.encoder_box.forward(&projected);
        let proposals = encoder_proposals(spatial);
        let proposal_batch = mlxcel_core::expand_dims(&proposals, 0);
        let delta_xy = mlxcel_core::slice(&delta, &[0, 0, 0], &[batch, shape[1], 2]);
        let delta_wh = mlxcel_core::slice(&delta, &[0, 0, 2], &[batch, shape[1], 4]);
        let proposal_xy = mlxcel_core::slice(&proposal_batch, &[0, 0, 0], &[1, shape[1], 2]);
        let proposal_wh = mlxcel_core::slice(&proposal_batch, &[0, 0, 2], &[1, shape[1], 4]);
        let encoded_xy = mlxcel_core::add(
            &mlxcel_core::multiply(&delta_xy, &proposal_wh),
            &proposal_xy,
        );
        let encoded_wh = mlxcel_core::multiply(&mlxcel_core::exp(&delta_wh), &proposal_wh);
        let encoded = concat(&[encoded_xy, encoded_wh], -1);

        let scores = mlxcel_core::max_axis(&logits, -1, false);
        let partition = mlxcel_core::argpartition(
            &mlxcel_core::multiply_scalar(&scores, -1.0),
            self.queries,
            1,
        );
        let indices = mlxcel_core::stop_gradient(&mlxcel_core::slice(
            &partition,
            &[0, 0],
            &[batch, self.queries],
        ));
        let selected_scores = mlxcel_core::take_along_axis(&scores, &indices, 1);
        let order = mlxcel_core::argsort(&mlxcel_core::multiply_scalar(&selected_scores, -1.0), 1);
        let indices =
            mlxcel_core::stop_gradient(&mlxcel_core::take_along_axis(&indices, &order, 1));
        let gather4 = mlxcel_core::broadcast_to(
            &mlxcel_core::expand_dims(&indices, 2),
            &[batch, self.queries, 4],
        );
        let selected =
            mlxcel_core::stop_gradient(&mlxcel_core::take_along_axis(&encoded, &gather4, 1));
        let learned = mlxcel_core::slice(&self.reference, &[0, 0], &[self.queries, 4]);
        let learned = mlxcel_core::expand_dims(&learned, 0);
        let learned_xy = mlxcel_core::slice(&learned, &[0, 0, 0], &[1, self.queries, 2]);
        let learned_wh = mlxcel_core::slice(&learned, &[0, 0, 2], &[1, self.queries, 4]);
        let selected_xy = mlxcel_core::slice(&selected, &[0, 0, 0], &[batch, self.queries, 2]);
        let selected_wh = mlxcel_core::slice(&selected, &[0, 0, 2], &[batch, self.queries, 4]);
        let reference_xy = mlxcel_core::add(
            &mlxcel_core::multiply(&learned_xy, &selected_wh),
            &selected_xy,
        );
        let reference_wh = mlxcel_core::multiply(&mlxcel_core::exp(&learned_wh), &selected_wh);
        let reference = concat(&[reference_xy, reference_wh], -1);

        let query = mlxcel_core::slice(&self.query, &[0, 0], &[self.queries, 256]);
        let query = mlxcel_core::broadcast_to(
            &mlxcel_core::expand_dims(&query, 0),
            &[batch, self.queries, 256],
        );
        let hidden = self.decoder.forward(&query, memory, &reference, spatial);
        (hidden, reference)
    }
}

fn encoder_proposals(spatial: (i32, i32)) -> UniquePtr<MlxArray> {
    let (height, width) = spatial;
    let y = mlxcel_core::divide_scalar(
        &mlxcel_core::add(
            &mlxcel_core::arange_f32(0.0, height as f32, 1.0),
            &mlxcel_core::full_f32(&[1], 0.5, mlxcel_core::dtype::FLOAT32),
        ),
        height as f32,
    );
    let x = mlxcel_core::divide_scalar(
        &mlxcel_core::add(
            &mlxcel_core::arange_f32(0.0, width as f32, 1.0),
            &mlxcel_core::full_f32(&[1], 0.5, mlxcel_core::dtype::FLOAT32),
        ),
        width as f32,
    );
    let y = mlxcel_core::broadcast_to(&mlxcel_core::reshape(&y, &[height, 1]), &[height, width]);
    let x = mlxcel_core::broadcast_to(&mlxcel_core::reshape(&x, &[1, width]), &[height, width]);
    let size = mlxcel_core::full_f32(&[height, width], 0.05, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::reshape(
        &mlxcel_core::stack_owned(&[x, y, mlxcel_core::copy(&size), size], -1),
        &[height * width, 4],
    )
}
