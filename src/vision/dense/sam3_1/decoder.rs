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

//! Multiplex mask decoder used by SAM 3.1 tracking.

use mlxcel_core::layers::Embedding;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::super::sam3::sam_components::{
    Conv2dLayer, ConvTranspose2d, LayerNorm2d, OutputMlp, TwoWayTransformer,
};
use super::config::MultiplexMaskDecoderConfig;

pub(crate) struct DecoderOutput {
    pub masks: UniquePtr<MlxArray>,
    pub iou_scores: UniquePtr<MlxArray>,
    pub sam_tokens: UniquePtr<MlxArray>,
    pub object_scores: UniquePtr<MlxArray>,
}

pub(crate) struct MultiplexMaskDecoder {
    hidden_size: i32,
    multiplex_count: i32,
    masks_per_object: i32,
    multimask_outputs_only: bool,
    use_multimask_token_for_obj_ptr: bool,
    transformer: TwoWayTransformer,
    iou_token: Embedding,
    mask_tokens: Embedding,
    obj_score_token: Embedding,
    hypernetworks: Vec<OutputMlp>,
    iou_head: OutputMlp,
    object_score_head: OutputMlp,
    upscale_conv1: ConvTranspose2d,
    upscale_conv2: ConvTranspose2d,
    upscale_norm: LayerNorm2d,
    conv_s0: Conv2dLayer,
    conv_s1: Conv2dLayer,
}

impl MultiplexMaskDecoder {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &MultiplexMaskDecoderConfig,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let hidden_size = config.hidden_size as i32;
        let masks_per_object = if config.multimask_outputs_only {
            config.num_multimask_outputs as i32
        } else {
            config.num_multimask_outputs as i32 + 1
        };
        let mut hypernetworks = Vec::with_capacity(masks_per_object as usize);
        for index in 0..masks_per_object {
            hypernetworks.push(OutputMlp::from_weights(
                weights,
                &format!("{prefix}.output_hypernetworks_mlps.{index}"),
                group_size,
                bits,
            )?);
        }
        Ok(Self {
            hidden_size,
            multiplex_count: config.multiplex_count as i32,
            masks_per_object,
            multimask_outputs_only: config.multimask_outputs_only,
            use_multimask_token_for_obj_ptr: config.use_multimask_token_for_obj_ptr,
            transformer: TwoWayTransformer::from_weights(
                weights,
                &format!("{prefix}.transformer"),
                hidden_size,
                config.num_attention_heads as i32,
                config.num_hidden_layers,
                config.attention_downsample_rate as i32,
                group_size,
                bits,
            )?,
            iou_token: Embedding::from_weights(weights, &format!("{prefix}.iou_token"))?,
            mask_tokens: Embedding::from_weights(weights, &format!("{prefix}.mask_tokens"))?,
            obj_score_token: Embedding::from_weights(
                weights,
                &format!("{prefix}.obj_score_token"),
            )?,
            hypernetworks,
            iou_head: OutputMlp::from_weights(
                weights,
                &format!("{prefix}.iou_prediction_head"),
                group_size,
                bits,
            )?,
            object_score_head: OutputMlp::from_weights(
                weights,
                &format!("{prefix}.pred_obj_score_head"),
                group_size,
                bits,
            )?,
            upscale_conv1: ConvTranspose2d::from_weights(
                weights,
                &format!("{prefix}.upscale_conv1"),
            )?,
            upscale_conv2: ConvTranspose2d::from_weights(
                weights,
                &format!("{prefix}.upscale_conv2"),
            )?,
            upscale_norm: LayerNorm2d::from_weights(
                weights,
                &format!("{prefix}.upscale_layer_norm"),
            )?,
            conv_s0: Conv2dLayer::from_weights(
                weights,
                &format!("{prefix}.conv_s0"),
                1,
                0,
                1,
                true,
            )?,
            conv_s1: Conv2dLayer::from_weights(
                weights,
                &format!("{prefix}.conv_s1"),
                1,
                0,
                1,
                true,
            )?,
        })
    }

    pub(crate) fn project_high_resolution(
        &self,
        fpn: &[UniquePtr<MlxArray>],
    ) -> Result<[UniquePtr<MlxArray>; 2], String> {
        if fpn.len() < 2 {
            return Err(format!(
                "SAM 3.1 mask decoder needs two high-resolution FPN levels, got {}",
                fpn.len()
            ));
        }
        Ok([self.conv_s0.forward(&fpn[0]), self.conv_s1.forward(&fpn[1])])
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &self,
        image_embeddings: &MlxArray,
        image_pe: &MlxArray,
        high_resolution: &[UniquePtr<MlxArray>; 2],
        extra_per_object: Option<&MlxArray>,
        sparse_prompt: Option<&MlxArray>,
        dense_prompt: Option<&MlxArray>,
        multimask_output: bool,
    ) -> Result<DecoderOutput, String> {
        if self.multimask_outputs_only && !multimask_output {
            return Err("SAM 3.1 propagation decoder requires multimask output".into());
        }
        let shape = mlxcel_core::array_shape(image_embeddings);
        if shape.len() != 3 || shape[2] != self.hidden_size {
            return Err(format!(
                "SAM 3.1 mask decoder expected [B,HW,{}], got {shape:?}",
                self.hidden_size
            ));
        }
        let image_batch = shape[0];
        let batch = sparse_prompt
            .map(mlxcel_core::array_shape)
            .and_then(|shape| shape.first().copied())
            .unwrap_or(image_batch);
        let source = if image_batch == batch {
            mlxcel_core::copy(image_embeddings)
        } else if image_batch == 1 {
            mlxcel_core::broadcast_to(image_embeddings, &[batch, shape[1], shape[2]])
        } else {
            return Err(format!(
                "SAM 3.1 cannot broadcast image batch {image_batch} to prompt batch {batch}"
            ));
        };
        let source = if let Some(dense) = dense_prompt {
            mlxcel_core::add(&source, dense)
        } else {
            source
        };

        let token_batch = |embedding: &Embedding, rows: i32| {
            mlxcel_core::broadcast_to(
                &mlxcel_core::reshape(&embedding.weight, &[1, rows, self.hidden_size]),
                &[batch, rows, self.hidden_size],
            )
        };
        let object_tokens = token_batch(&self.obj_score_token, self.multiplex_count);
        let iou_tokens = token_batch(&self.iou_token, self.multiplex_count);
        let mut mask_tokens = mlxcel_core::reshape(
            &self.mask_tokens.weight,
            &[
                1,
                self.multiplex_count,
                self.masks_per_object,
                self.hidden_size,
            ],
        );
        mask_tokens = mlxcel_core::broadcast_to(
            &mask_tokens,
            &[
                batch,
                self.multiplex_count,
                self.masks_per_object,
                self.hidden_size,
            ],
        );
        if let Some(extra) = extra_per_object {
            mask_tokens = mlxcel_core::add(&mask_tokens, &mlxcel_core::expand_dims(extra, 2));
        }
        let mask_tokens = mlxcel_core::reshape(
            &mask_tokens,
            &[
                batch,
                self.multiplex_count * self.masks_per_object,
                self.hidden_size,
            ],
        );
        let mut tokens =
            mlxcel_core::concatenate_many(&[&object_tokens, &iou_tokens, &mask_tokens], 1);
        if let Some(sparse) = sparse_prompt {
            tokens = mlxcel_core::concatenate(&tokens, sparse, 1);
        }
        let pe = mlxcel_core::broadcast_to(image_pe, &[batch, shape[1], shape[2]]);
        let (transformed, source) = self.transformer.forward(&source, &pe, &tokens);

        let object_token = mlxcel_core::slice(
            &transformed,
            &[0, 0, 0],
            &[batch, self.multiplex_count, self.hidden_size],
        );
        let iou_token = mlxcel_core::slice(
            &transformed,
            &[0, self.multiplex_count, 0],
            &[batch, 2 * self.multiplex_count, self.hidden_size],
        );
        let mask_token = mlxcel_core::slice(
            &transformed,
            &[0, 2 * self.multiplex_count, 0],
            &[
                batch,
                2 * self.multiplex_count + self.multiplex_count * self.masks_per_object,
                self.hidden_size,
            ],
        );
        let mask_token = mlxcel_core::reshape(
            &mask_token,
            &[
                batch,
                self.multiplex_count,
                self.masks_per_object,
                self.hidden_size,
            ],
        );

        let side = (shape[1] as f64).sqrt() as i32;
        if side * side != shape[1] {
            return Err(format!(
                "SAM 3.1 mask decoder token count {} is not square",
                shape[1]
            ));
        }
        let source = mlxcel_core::reshape(&source, &[batch, side, side, self.hidden_size]);
        let upscaled = mlxcel_core::add(&self.upscale_conv1.forward(&source), &high_resolution[1]);
        let upscaled = super::super::sam3::activation::gelu(&self.upscale_norm.forward(&upscaled));
        let upscaled =
            mlxcel_core::add(&self.upscale_conv2.forward(&upscaled), &high_resolution[0]);
        let upscaled = super::super::sam3::activation::gelu(&upscaled);
        let up_shape = mlxcel_core::array_shape(&upscaled);
        let upscaled =
            mlxcel_core::reshape(&upscaled, &[batch, up_shape[1] * up_shape[2], up_shape[3]]);

        let mut hyper = Vec::with_capacity(self.masks_per_object as usize);
        for index in 0..self.masks_per_object {
            let token = mlxcel_core::slice(
                &mask_token,
                &[0, 0, index, 0],
                &[batch, self.multiplex_count, index + 1, self.hidden_size],
            );
            hyper.push(
                self.hypernetworks[index as usize].forward(&mlxcel_core::squeeze_axis(&token, 2)),
            );
        }
        let hyper = mlxcel_core::stack_owned(&hyper, 2);
        let masks = mlxcel_core::matmul(
            &mlxcel_core::reshape(
                &hyper,
                &[
                    batch,
                    self.multiplex_count * self.masks_per_object,
                    up_shape[3],
                ],
            ),
            &mlxcel_core::transpose_axes(&upscaled, &[0, 2, 1]),
        );
        let masks = mlxcel_core::reshape(
            &masks,
            &[
                batch,
                self.multiplex_count,
                self.masks_per_object,
                up_shape[1],
                up_shape[2],
            ],
        );
        let iou_scores = self.iou_head.forward(&iou_token);
        let object_scores = self.object_score_head.forward(&object_token);

        let (start, stop) = if multimask_output {
            let start = i32::from(!self.multimask_outputs_only);
            (start, self.masks_per_object)
        } else {
            (0, 1)
        };
        let masks = mlxcel_core::slice(
            &masks,
            &[0, 0, start, 0, 0],
            &[batch, self.multiplex_count, stop, up_shape[1], up_shape[2]],
        );
        let iou_scores = mlxcel_core::slice(
            &iou_scores,
            &[0, 0, start],
            &[batch, self.multiplex_count, stop],
        );
        let token_start = if multimask_output && self.use_multimask_token_for_obj_ptr {
            start
        } else {
            0
        };
        let token_stop = if multimask_output && self.use_multimask_token_for_obj_ptr {
            stop
        } else {
            1
        };
        let sam_tokens = mlxcel_core::slice(
            &mask_token,
            &[0, 0, token_start, 0],
            &[batch, self.multiplex_count, token_stop, self.hidden_size],
        );
        Ok(DecoderOutput {
            masks,
            iou_scores,
            sam_tokens,
            object_scores,
        })
    }
}
