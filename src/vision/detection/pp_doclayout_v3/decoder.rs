// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! Six-layer PP-DocLayoutV3 deformable decoder.

use mlxcel_core::layers::{LayerNorm, Linear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::vision::detection::rt_detr_v2::hybrid_encoder::{SelfAttention, load_layer_norm};
use crate::vision::detection::rt_detr_v2::transformer::{Mlp, inverse_sigmoid};

use super::config::PpDocLayoutV3Config;
use super::deformable::{MsDeformableAttention, SpatialShape};
use super::layers::Activation;
use super::order::GlobalPointer;

struct DecoderLayer {
    self_attention: SelfAttention,
    self_attention_norm: LayerNorm,
    cross_attention: MsDeformableAttention,
    cross_attention_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_norm: LayerNorm,
    activation: Activation,
}

impl DecoderLayer {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &PpDocLayoutV3Config,
    ) -> Result<Self, String> {
        Ok(Self {
            self_attention: SelfAttention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                config.d_model,
                config.decoder_attention_heads,
            )?,
            self_attention_norm: load_layer_norm(
                weights,
                &format!("{prefix}.self_attn_layer_norm"),
                config.layer_norm_eps,
            )?,
            cross_attention: MsDeformableAttention::from_weights(
                weights,
                &format!("{prefix}.encoder_attn"),
                config,
            )?,
            cross_attention_norm: load_layer_norm(
                weights,
                &format!("{prefix}.encoder_attn_layer_norm"),
                config.layer_norm_eps,
            )?,
            fc1: Linear::from_weights(weights, &format!("{prefix}.fc1"))?,
            fc2: Linear::from_weights(weights, &format!("{prefix}.fc2"))?,
            final_norm: load_layer_norm(
                weights,
                &format!("{prefix}.final_layer_norm"),
                config.layer_norm_eps,
            )?,
            activation: Activation::parse(&config.decoder_activation_function)?,
        })
    }

    fn forward(
        &self,
        index: usize,
        x: &MlxArray,
        position: &MlxArray,
        memory: &MlxArray,
        reference: &MlxArray,
        shapes: &[SpatialShape],
    ) -> UniquePtr<MlxArray> {
        let attention = super::attention::forward_fast(&self.self_attention, x, Some(position));
        #[cfg(test)]
        if index == 0 {
            super::diagnostics::capture("decoder.layer0.self_attention.q", &attention.query);
            super::diagnostics::capture("decoder.layer0.self_attention.key", &attention.key);
            super::diagnostics::capture("decoder.layer0.self_attention.value", &attention.value);
            super::diagnostics::capture("decoder.layer0.self_attention.sdpa", &attention.sdpa);
            super::diagnostics::capture("decoder.layer0.self_attention.merged", &attention.merged);
            super::diagnostics::capture("decoder.layer0.self_attention.output", &attention.output);
        }
        let self_residual = mlxcel_core::add(x, &attention.output);
        #[cfg(test)]
        if index == 0 {
            super::diagnostics::capture("decoder.layer0.self_attention.residual", &self_residual);
        }
        let attended = self.self_attention_norm.forward(&self_residual);
        #[cfg(test)]
        if index == 0 {
            super::diagnostics::capture("decoder.layer0.self_attention.norm", &attended);
        }
        let cross_output = self.cross_attention.forward(
            index,
            &attended,
            reference,
            memory,
            shapes,
            Some(position),
        );
        let cross_residual = mlxcel_core::add(&attended, &cross_output);
        #[cfg(test)]
        if index == 0 {
            super::diagnostics::capture("decoder.layer0.cross.residual", &cross_residual);
        }
        let crossed = self.cross_attention_norm.forward(&cross_residual);
        #[cfg(test)]
        if index == 0 {
            super::diagnostics::capture("decoder.layer0.cross.norm", &crossed);
        }
        let fc1 = self.fc1.forward(&crossed);
        #[cfg(test)]
        if index == 0 {
            super::diagnostics::capture("decoder.layer0.ffn.fc1", &fc1);
        }
        let activated = self.activation.apply(&fc1);
        #[cfg(test)]
        if index == 0 {
            super::diagnostics::capture("decoder.layer0.ffn.activation", &activated);
        }
        let hidden = self.fc2.forward(&activated);
        #[cfg(test)]
        if index == 0 {
            super::diagnostics::capture("decoder.layer0.ffn.fc2", &hidden);
        }
        let final_residual = mlxcel_core::add(&crossed, &hidden);
        #[cfg(test)]
        if index == 0 {
            super::diagnostics::capture("decoder.layer0.ffn.residual", &final_residual);
        }
        self.final_norm.forward(&final_residual)
    }
}

pub struct DecoderOutput {
    pub logits: UniquePtr<MlxArray>,
    pub boxes: UniquePtr<MlxArray>,
    pub order_logits: UniquePtr<MlxArray>,
}

pub struct Decoder {
    layers: Vec<DecoderLayer>,
    query_position: Mlp,
    order_heads: Vec<Linear>,
    global_pointer: GlobalPointer,
}

impl Decoder {
    pub fn from_weights(weights: &WeightMap, config: &PpDocLayoutV3Config) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(config.decoder_layers);
        let mut order_heads = Vec::with_capacity(config.decoder_layers);
        for index in 0..config.decoder_layers {
            layers.push(DecoderLayer::from_weights(
                weights,
                &format!("decoder.layers.{index}"),
                config,
            )?);
            order_heads.push(Linear::from_weights(
                weights,
                &format!("decoder_order_head.{index}"),
            )?);
        }
        Ok(Self {
            layers,
            query_position: Mlp::from_weights(weights, "decoder.query_pos_head", 2)?,
            order_heads,
            global_pointer: GlobalPointer::from_weights(weights, config)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        target: &MlxArray,
        reference_logits: &MlxArray,
        memory: &MlxArray,
        shapes: &[SpatialShape],
        bbox_head: &Mlp,
        class_head: &Linear,
        output_norm: &LayerNorm,
    ) -> DecoderOutput {
        let mut target = mlxcel_core::copy(target);
        let mut reference = mlxcel_core::sigmoid(reference_logits);
        let mut final_logits = mlxcel_core::zeros(&[1], mlxcel_core::dtype::FLOAT32);
        let mut final_boxes = mlxcel_core::zeros(&[1], mlxcel_core::dtype::FLOAT32);
        let mut final_order = mlxcel_core::zeros(&[1], mlxcel_core::dtype::FLOAT32);

        for (index, layer) in self.layers.iter().enumerate() {
            let position = self.query_position.forward(&reference);
            #[cfg(test)]
            if index == 0 {
                super::diagnostics::capture("decoder.layer0.position", &position);
            }
            let level_reference = mlxcel_core::expand_dims(&reference, 2);
            target = layer.forward(index, &target, &position, memory, &level_reference, shapes);
            #[cfg(test)]
            super::diagnostics::capture(&format!("decoder.layer{index}.output"), &target);
            let refined = mlxcel_core::sigmoid(&mlxcel_core::add(
                &bbox_head.forward(&target),
                &inverse_sigmoid(&reference, 1.0e-5),
            ));
            reference = mlxcel_core::stop_gradient(&refined);
            #[cfg(test)]
            super::diagnostics::capture(&format!("decoder.layer{index}.reference"), &reference);
            let output = output_norm.forward(&target);
            final_logits = class_head.forward(&output);
            final_boxes = refined;
            let order_head = self.order_heads[index].forward(&output);
            let final_layer = index + 1 == self.layers.len();
            #[cfg(test)]
            if final_layer {
                super::diagnostics::capture("decoder.layer5.output_norm", &output);
                super::diagnostics::capture("decoder.layer5.order_head", &order_head);
            }
            final_order = self.global_pointer.forward(&order_head, final_layer);
        }
        #[cfg(test)]
        {
            super::diagnostics::capture("output.logits", &final_logits);
            super::diagnostics::capture("output.pred_boxes", &final_boxes);
            super::diagnostics::capture("output.order_logits", &final_order);
        }
        DecoderOutput {
            logits: final_logits,
            boxes: final_boxes,
            order_logits: final_order,
        }
    }
}
