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

use mlxcel_core::layers::Embedding;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::super::linear::UnifiedLinear;

use super::super::config::TrackerMaskDecoderConfig;
use super::layers::{Conv2dLayer, LayerNorm2d, TwoWayTransformer, load_array};

struct ConvTranspose2d {
    weight: UniquePtr<MlxArray>,
    bias: UniquePtr<MlxArray>,
}

impl ConvTranspose2d {
    fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            weight: load_array(weights, &format!("{prefix}.weight"))?,
            bias: load_array(weights, &format!("{prefix}.bias"))?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let y = mlxcel_core::conv_transpose2d(x, &self.weight, 2, 2, 0, 0, 1, 1, 0, 0, 1);
        mlxcel_core::add(&y, &self.bias)
    }
}

struct OutputMlp {
    proj_in: UnifiedLinear,
    layers: Vec<UnifiedLinear>,
    proj_out: UnifiedLinear,
}

impl OutputMlp {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            proj_in: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.proj_in"),
                group_size,
                bits,
            )?,
            layers: vec![UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.layers.0"),
                group_size,
                bits,
            )?],
            proj_out: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.proj_out"),
                group_size,
                bits,
            )?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let mut x = mlxcel_core::relu(&self.proj_in.forward(x));
        for layer in &self.layers {
            x = mlxcel_core::relu(&layer.forward(&x));
        }
        self.proj_out.forward(&x)
    }
}

/// Outputs of the tracker SAM mask decoder.
pub struct MaskDecoderOutput {
    /// Selected mask logits, `[batch, masks, height, width]`.
    pub masks: UniquePtr<MlxArray>,
    /// IoU estimates corresponding to `masks`.
    pub iou_scores: UniquePtr<MlxArray>,
    /// All transformed special and prompt tokens.
    pub sam_tokens: UniquePtr<MlxArray>,
    /// Per-batch object-presence logit.
    pub object_score: UniquePtr<MlxArray>,
}

/// SAM two-way transformer and hypernetwork mask decoder.
pub struct SamMaskDecoder {
    hidden_size: i32,
    num_mask_tokens: i32,
    transformer: TwoWayTransformer,
    iou_token: Embedding,
    mask_tokens: Embedding,
    obj_score_token: Embedding,
    output_hypernetworks_mlps: Vec<OutputMlp>,
    iou_prediction_head: OutputMlp,
    pred_obj_score_head: OutputMlp,
    upscale_conv1: ConvTranspose2d,
    upscale_conv2: ConvTranspose2d,
    upscale_layer_norm: LayerNorm2d,
    conv_s0: Conv2dLayer,
    conv_s1: Conv2dLayer,
}

impl SamMaskDecoder {
    pub fn from_weights(
        weights: &WeightMap,
        config: &TrackerMaskDecoderConfig,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let hidden_size = config.hidden_size as i32;
        let num_mask_tokens = config.num_multimask_outputs as i32 + 1;
        let mut output_hypernetworks_mlps = Vec::with_capacity(num_mask_tokens as usize);
        for index in 0..num_mask_tokens {
            output_hypernetworks_mlps.push(OutputMlp::from_weights(
                weights,
                &format!("{prefix}.output_hypernetworks_mlps.{index}"),
                group_size,
                bits,
            )?);
        }
        Ok(Self {
            hidden_size,
            num_mask_tokens,
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
            output_hypernetworks_mlps,
            iou_prediction_head: OutputMlp::from_weights(
                weights,
                &format!("{prefix}.iou_prediction_head"),
                group_size,
                bits,
            )?,
            pred_obj_score_head: OutputMlp::from_weights(
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
            upscale_layer_norm: LayerNorm2d::from_weights(
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

    fn token_batch(&self, embedding: &Embedding, batch: i32, rows: i32) -> UniquePtr<MlxArray> {
        mlxcel_core::broadcast_to(
            &mlxcel_core::reshape(&embedding.weight, &[1, rows, self.hidden_size]),
            &[batch, rows, self.hidden_size],
        )
    }

    /// Decode mask logits from image and prompt embeddings.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        image_embeddings: &MlxArray,
        image_pe: &MlxArray,
        sparse_prompt_embeddings: &MlxArray,
        dense_prompt_embeddings: &MlxArray,
        multimask_output: bool,
        high_res_features: Option<&[&MlxArray]>,
    ) -> Result<MaskDecoderOutput, String> {
        let image_shape = mlxcel_core::array_shape(image_embeddings);
        if image_shape.len() != 3 || image_shape[2] != self.hidden_size {
            return Err(format!(
                "SAM3 mask decoder expected [batch, tokens, {}], got {image_shape:?}",
                self.hidden_size
            ));
        }
        let (batch, tokens) = (image_shape[0], image_shape[1]);
        let side = (tokens as f64).sqrt() as i32;
        if side * side != tokens {
            return Err(format!(
                "SAM3 mask decoder image token count must be square, got {tokens}"
            ));
        }
        let special = [
            self.token_batch(&self.iou_token, batch, 1),
            self.token_batch(&self.mask_tokens, batch, self.num_mask_tokens),
            self.token_batch(&self.obj_score_token, batch, 1),
        ];
        let refs: Vec<&MlxArray> = special.iter().map(|value| &**value).collect();
        let special = mlxcel_core::concatenate_many(&refs, 1);
        let all_tokens = mlxcel_core::concatenate(&special, sparse_prompt_embeddings, 1);
        let src = mlxcel_core::add(image_embeddings, dense_prompt_embeddings);
        let (sam_tokens, src) = self.transformer.forward(&src, image_pe, &all_tokens);

        let iou_token_out = mlxcel_core::reshape(
            &mlxcel_core::slice(&sam_tokens, &[0, 0, 0], &[batch, 1, self.hidden_size]),
            &[batch, self.hidden_size],
        );
        let mask_tokens_out = mlxcel_core::slice(
            &sam_tokens,
            &[0, 1, 0],
            &[batch, 1 + self.num_mask_tokens, self.hidden_size],
        );
        let object_token = mlxcel_core::reshape(
            &mlxcel_core::slice(
                &sam_tokens,
                &[0, 1 + self.num_mask_tokens, 0],
                &[batch, 2 + self.num_mask_tokens, self.hidden_size],
            ),
            &[batch, self.hidden_size],
        );

        let src = mlxcel_core::reshape(&src, &[batch, side, side, self.hidden_size]);
        let upscaled = self.upscale_conv1.forward(&src);
        let mut upscaled =
            super::super::activation::gelu(&self.upscale_layer_norm.forward(&upscaled));
        if let Some(features) = high_res_features
            && let Some(feature) = features.first()
        {
            let skip = self.conv_s1.forward(feature);
            if mlxcel_core::array_shape(&skip)[1..3] == mlxcel_core::array_shape(&upscaled)[1..3] {
                upscaled = mlxcel_core::add(&upscaled, &skip);
            }
        }
        let mut upscaled = super::super::activation::gelu(&self.upscale_conv2.forward(&upscaled));
        if let Some(features) = high_res_features
            && let Some(feature) = features.get(1)
        {
            let skip = self.conv_s0.forward(feature);
            if mlxcel_core::array_shape(&skip)[1..3] == mlxcel_core::array_shape(&upscaled)[1..3] {
                upscaled = mlxcel_core::add(&upscaled, &skip);
            }
        }
        let up_shape = mlxcel_core::array_shape(&upscaled);
        let upscaled =
            mlxcel_core::reshape(&upscaled, &[batch, up_shape[1] * up_shape[2], up_shape[3]]);
        let mut masks = Vec::with_capacity(self.num_mask_tokens as usize);
        for index in 0..self.num_mask_tokens {
            let token = mlxcel_core::reshape(
                &mlxcel_core::slice(
                    &mask_tokens_out,
                    &[0, index, 0],
                    &[batch, index + 1, self.hidden_size],
                ),
                &[batch, self.hidden_size],
            );
            let hyper = self.output_hypernetworks_mlps[index as usize].forward(&token);
            let hyper = mlxcel_core::expand_dims(&hyper, 1);
            let mask = mlxcel_core::sum_axis(&mlxcel_core::multiply(&upscaled, &hyper), -1, false);
            masks.push(mlxcel_core::reshape(
                &mask,
                &[batch, 1, up_shape[1], up_shape[2]],
            ));
        }
        let refs: Vec<&MlxArray> = masks.iter().map(|value| &**value).collect();
        let masks = mlxcel_core::concatenate_many(&refs, 1);
        let iou_scores = self.iou_prediction_head.forward(&iou_token_out);
        let object_score = self.pred_obj_score_head.forward(&object_token);
        let (start, stop) = if multimask_output {
            (1, self.num_mask_tokens)
        } else {
            (0, 1)
        };
        Ok(MaskDecoderOutput {
            masks: mlxcel_core::slice(
                &masks,
                &[0, start, 0, 0],
                &[batch, stop, up_shape[1], up_shape[2]],
            ),
            iou_scores: mlxcel_core::slice(&iou_scores, &[0, start], &[batch, stop]),
            sam_tokens,
            object_score,
        })
    }
}
