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

//! Geometry prompt encoder for SAM 3 box prompts.

use std::f32::consts::TAU;

use mlxcel_core::layers::{LayerNorm, UnifiedEmbedding};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::linear::UnifiedLinear;

use super::config::GeometryEncoderConfig;
use super::encoder::{Mlp, MultiheadAttention, load_layer_norm};

struct GeometryEncoderLayer {
    layer_norm1: LayerNorm,
    self_attn: MultiheadAttention,
    layer_norm2: LayerNorm,
    cross_attn: MultiheadAttention,
    layer_norm3: LayerNorm,
    mlp: Mlp,
}

impl GeometryEncoderLayer {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &GeometryEncoderConfig,
        quantization: (i32, i32),
    ) -> Result<Self, String> {
        Ok(Self {
            layer_norm1: load_layer_norm(
                weights,
                &format!("{prefix}.layer_norm1"),
                config.layer_norm_eps,
            )?,
            self_attn: MultiheadAttention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                config.hidden_size,
                config.num_attention_heads,
                quantization,
            )?,
            layer_norm2: load_layer_norm(
                weights,
                &format!("{prefix}.layer_norm2"),
                config.layer_norm_eps,
            )?,
            cross_attn: MultiheadAttention::from_weights(
                weights,
                &format!("{prefix}.cross_attn"),
                config.hidden_size,
                config.num_attention_heads,
                quantization,
            )?,
            layer_norm3: load_layer_norm(
                weights,
                &format!("{prefix}.layer_norm3"),
                config.layer_norm_eps,
            )?,
            mlp: Mlp::from_weights(
                weights,
                &format!("{prefix}.mlp"),
                &config.hidden_act,
                quantization,
            )?,
        })
    }

    fn forward(
        &self,
        prompt: &MlxArray,
        vision: &MlxArray,
        vision_pos: &MlxArray,
        prompt_mask: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let hidden = self.layer_norm1.forward(prompt);
        let prompt = mlxcel_core::add(
            prompt,
            &self
                .self_attn
                .forward(&hidden, &hidden, &hidden, Some(prompt_mask)),
        );
        let hidden = self.layer_norm2.forward(&prompt);
        let vision_key = mlxcel_core::add(vision, vision_pos);
        let prompt = mlxcel_core::add(
            &prompt,
            &self.cross_attn.forward(&hidden, &vision_key, vision, None),
        );
        let hidden = self.layer_norm3.forward(&prompt);
        mlxcel_core::add(&prompt, &self.mlp.forward(&hidden))
    }
}

struct PoolProjection {
    weight: UniquePtr<MlxArray>,
    bias: Option<UniquePtr<MlxArray>>,
}

impl PoolProjection {
    fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        let name = format!("{prefix}.weight");
        let weight = weights
            .get(&name)
            .map(|value| mlxcel_core::copy(value))
            .ok_or_else(|| format!("Weight not found: {name}"))?;
        let bias = weights
            .get(&format!("{prefix}.bias"))
            .map(|value| mlxcel_core::copy(value));
        Ok(Self { weight, bias })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let output = mlxcel_core::conv2d(x, &self.weight, 1, 1, 0, 0, 1, 1, 1);
        match &self.bias {
            Some(bias) => mlxcel_core::add(&output, bias),
            None => output,
        }
    }
}

/// Transformer encoder that combines box coordinates, ROI features and labels.
pub struct GeometryEncoder {
    hidden_size: i32,
    roi_size: usize,
    label_embed: UnifiedEmbedding,
    cls_embed: UnifiedEmbedding,
    boxes_direct_project: UnifiedLinear,
    boxes_pool_project: PoolProjection,
    boxes_pos_enc_project: UnifiedLinear,
    vision_layer_norm: LayerNorm,
    final_proj: UnifiedLinear,
    prompt_layer_norm: LayerNorm,
    layers: Vec<GeometryEncoderLayer>,
    output_layer_norm: LayerNorm,
}

impl GeometryEncoder {
    /// `prefix` is normally `detector_model.geometry_encoder`.
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &GeometryEncoderConfig,
        quantization: (i32, i32),
    ) -> Result<Self, String> {
        let (group_size, bits) = quantization;
        let linear = |name: &str| {
            UnifiedLinear::from_weights(weights, &format!("{prefix}.{name}"), group_size, bits)
        };
        let embedding = |name: &str| {
            UnifiedEmbedding::from_weights(weights, &format!("{prefix}.{name}"), group_size, bits)
        };
        let mut layers = Vec::with_capacity(config.num_layers);
        for index in 0..config.num_layers {
            layers.push(GeometryEncoderLayer::from_weights(
                weights,
                &format!("{prefix}.layers.{index}"),
                config,
                quantization,
            )?);
        }
        Ok(Self {
            hidden_size: config.hidden_size as i32,
            roi_size: config.roi_size,
            label_embed: embedding("label_embed")?,
            cls_embed: embedding("cls_embed")?,
            boxes_direct_project: linear("boxes_direct_project")?,
            boxes_pool_project: PoolProjection::from_weights(
                weights,
                &format!("{prefix}.boxes_pool_project"),
            )?,
            boxes_pos_enc_project: linear("boxes_pos_enc_project")?,
            vision_layer_norm: load_layer_norm(
                weights,
                &format!("{prefix}.vision_layer_norm"),
                config.layer_norm_eps,
            )?,
            final_proj: linear("final_proj")?,
            prompt_layer_norm: load_layer_norm(
                weights,
                &format!("{prefix}.prompt_layer_norm"),
                config.layer_norm_eps,
            )?,
            layers,
            output_layer_norm: load_layer_norm(
                weights,
                &format!("{prefix}.output_layer_norm"),
                config.layer_norm_eps,
            )?,
        })
    }

    /// Encode normalized `cx,cy,w,h` boxes and append the learned CLS token.
    #[must_use]
    pub fn forward(
        &self,
        boxes: &MlxArray,
        box_labels: &MlxArray,
        box_mask: &MlxArray,
        image_features: &MlxArray,
        image_positions: &MlxArray,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let image_shape = mlxcel_core::array_shape(image_features);
        let box_shape = mlxcel_core::array_shape(boxes);
        let (batch, height, width, channels) = (
            image_shape[0],
            image_shape[1],
            image_shape[2],
            image_shape[3],
        );
        let num_boxes = box_shape[1];
        let vision = mlxcel_core::reshape(image_features, &[batch, height * width, channels]);
        let vision_pos = mlxcel_core::reshape(image_positions, &[batch, height * width, channels]);

        let feature_dtype = mlxcel_core::array_dtype(image_features);
        let boxes_f32 = mlxcel_core::astype(boxes, mlxcel_core::dtype::FLOAT32);
        let box_values = read_f32(&boxes_f32);
        let boxes_typed = mlxcel_core::astype(boxes, feature_dtype);
        let mut prompt = self.boxes_direct_project.forward(&boxes_typed);
        let normalized_features = self.vision_layer_norm.forward(image_features);
        let boxes_xyxy = boxes_to_feature_xyxy(&box_values, height, width);
        let pooled = roi_align_values(
            &normalized_features,
            &boxes_xyxy,
            batch,
            num_boxes,
            self.roi_size,
        );
        let pooled = mlxcel_core::reshape(
            &pooled,
            &[
                batch * num_boxes,
                self.roi_size as i32,
                self.roi_size as i32,
                self.hidden_size,
            ],
        );
        let pooled = self.boxes_pool_project.forward(&pooled);
        let pooled = mlxcel_core::reshape(&pooled, &[batch, num_boxes, self.hidden_size]);
        prompt = mlxcel_core::add(&prompt, &pooled);

        let positions = box_position_encoding(&box_values, self.hidden_size as usize / 2);
        let positions =
            mlxcel_core::from_slice_f32(&positions, &[batch, num_boxes, self.hidden_size + 2]);
        let positions = mlxcel_core::astype(&positions, feature_dtype);
        prompt = mlxcel_core::add(&prompt, &self.boxes_pos_enc_project.forward(&positions));
        prompt = mlxcel_core::add(&prompt, &self.label_embed.forward(box_labels));

        let cls_id = mlxcel_core::from_slice_i32(&[0], &[1]);
        let cls = self.cls_embed.forward(&cls_id);
        let cls = mlxcel_core::reshape(&cls, &[1, 1, self.hidden_size]);
        let cls = mlxcel_core::broadcast_to(&cls, &[batch, 1, self.hidden_size]);
        prompt = mlxcel_core::concatenate(&prompt, &cls, 1);
        let cls_mask = mlxcel_core::ones(&[batch, 1], mlxcel_core::array_dtype(box_mask));
        let prompt_mask = mlxcel_core::concatenate(box_mask, &cls_mask, 1);

        prompt = self
            .prompt_layer_norm
            .forward(&self.final_proj.forward(&prompt));
        let valid = mlxcel_core::astype(&prompt_mask, mlxcel_core::array_dtype(&prompt));
        let one = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::array_dtype(&prompt));
        let self_mask = mlxcel_core::multiply_scalar(&mlxcel_core::subtract(&one, &valid), -1e9);
        let self_mask = mlxcel_core::expand_dims_multi(&self_mask, &[1, 2]);
        for layer in &self.layers {
            prompt = layer.forward(&prompt, &vision, &vision_pos, &self_mask);
        }
        (self.output_layer_norm.forward(&prompt), prompt_mask)
    }
}

/// Torchvision-compatible ROI Align with `spatial_scale=1`, `aligned=false`.
#[must_use]
pub fn roi_align(
    features: &MlxArray,
    boxes_xyxy: &MlxArray,
    output_size: usize,
) -> UniquePtr<MlxArray> {
    let feature_shape = mlxcel_core::array_shape(features);
    let box_shape = mlxcel_core::array_shape(boxes_xyxy);
    let values = read_f32(&mlxcel_core::astype(
        boxes_xyxy,
        mlxcel_core::dtype::FLOAT32,
    ));
    roi_align_values(
        features,
        &values,
        feature_shape[0],
        box_shape[1],
        output_size,
    )
}

fn roi_align_values(
    features: &MlxArray,
    boxes: &[f32],
    batch: i32,
    num_boxes: i32,
    output_size: usize,
) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(features);
    let (height, width, channels) = (shape[1], shape[2], shape[3]);
    if num_boxes == 0 {
        return mlxcel_core::zeros(
            &[batch, 0, output_size as i32, output_size as i32, channels],
            mlxcel_core::array_dtype(features),
        );
    }
    let mut batches = Vec::with_capacity(batch as usize);
    for batch_index in 0..batch {
        let feature = mlxcel_core::slice(
            features,
            &[batch_index, 0, 0, 0],
            &[batch_index + 1, height, width, channels],
        );
        let feature = mlxcel_core::reshape(&feature, &[height * width, channels]);
        let mut rows = Vec::with_capacity(num_boxes as usize);
        for box_index in 0..num_boxes {
            let offset = ((batch_index * num_boxes + box_index) * 4) as usize;
            rows.push(sample_roi(
                &feature,
                [
                    boxes[offset],
                    boxes[offset + 1],
                    boxes[offset + 2],
                    boxes[offset + 3],
                ],
                height,
                width,
                channels,
                output_size,
            ));
        }
        batches.push(stack_arrays(&rows, 0));
    }
    stack_arrays(&batches, 0)
}

fn sample_roi(
    feature: &MlxArray,
    roi: [f32; 4],
    height: i32,
    width: i32,
    channels: i32,
    output_size: usize,
) -> UniquePtr<MlxArray> {
    let [x1, y1, x2, y2] = roi;
    let roi_width = (x2 - x1).max(1.0);
    let roi_height = (y2 - y1).max(1.0);
    let bin_width = roi_width / output_size as f32;
    let bin_height = roi_height / output_size as f32;
    let grid_width = bin_width.ceil().max(1.0) as usize;
    let grid_height = bin_height.ceil().max(1.0) as usize;
    let mut ys = Vec::with_capacity(output_size * grid_height);
    let mut xs = Vec::with_capacity(output_size * grid_width);
    for bin in 0..output_size {
        for sample in 0..grid_height {
            ys.push(
                y1 + bin as f32 * bin_height
                    + (sample as f32 + 0.5) * bin_height / grid_height as f32,
            );
        }
        for sample in 0..grid_width {
            xs.push(
                x1 + bin as f32 * bin_width + (sample as f32 + 0.5) * bin_width / grid_width as f32,
            );
        }
    }
    let mut indices = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut weights = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for &y_coord in &ys {
        for &x_coord in &xs {
            let valid = (-1.0..=height as f32).contains(&y_coord)
                && (-1.0..=width as f32).contains(&x_coord);
            let y = y_coord.max(0.0);
            let x = x_coord.max(0.0);
            let y_floor = y.floor() as i32;
            let x_floor = x.floor() as i32;
            let y_edge = y_floor >= height - 1;
            let x_edge = x_floor >= width - 1;
            let yl = if y_edge { height - 1 } else { y_floor };
            let xl = if x_edge { width - 1 } else { x_floor };
            let yh = if y_edge { height - 1 } else { y_floor + 1 };
            let xh = if x_edge { width - 1 } else { x_floor + 1 };
            let ly = if y_edge { 0.0 } else { y - y_floor as f32 };
            let lx = if x_edge { 0.0 } else { x - x_floor as f32 };
            let gate = if valid { 1.0 } else { 0.0 };
            for (slot, index, weight) in [
                (0, yl * width + xl, (1.0 - ly) * (1.0 - lx)),
                (1, yl * width + xh, (1.0 - ly) * lx),
                (2, yh * width + xl, ly * (1.0 - lx)),
                (3, yh * width + xh, ly * lx),
            ] {
                indices[slot].push(index);
                weights[slot].push(weight * gate);
            }
        }
    }
    let samples = (ys.len() * xs.len()) as i32;
    let mut output = mlxcel_core::zeros(&[samples, channels], mlxcel_core::array_dtype(feature));
    for corner in 0..4 {
        let index = mlxcel_core::from_slice_i32(&indices[corner], &[samples]);
        let values = mlxcel_core::take(feature, &index, 0);
        let weight = mlxcel_core::from_slice_f32(&weights[corner], &[samples, 1]);
        let weight = mlxcel_core::astype(&weight, mlxcel_core::array_dtype(feature));
        output = mlxcel_core::add(&output, &mlxcel_core::multiply(&values, &weight));
    }
    let output = mlxcel_core::reshape(
        &output,
        &[
            output_size as i32,
            grid_height as i32,
            output_size as i32,
            grid_width as i32,
            channels,
        ],
    );
    let output = mlxcel_core::mean_axis(&output, 3, false);
    mlxcel_core::mean_axis(&output, 1, false)
}

fn box_position_encoding(boxes: &[f32], num_pos_feats: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(boxes.len() / 4 * (num_pos_feats * 2 + 2));
    for values in boxes.chunks_exact(4) {
        encode_1d(&mut output, values[1], num_pos_feats);
        encode_1d(&mut output, values[0], num_pos_feats);
        output.extend_from_slice(&[values[3], values[2]]);
    }
    output
}

fn boxes_to_feature_xyxy(boxes: &[f32], height: i32, width: i32) -> Vec<f32> {
    let mut output = Vec::with_capacity(boxes.len());
    for values in boxes.chunks_exact(4) {
        let (cx, cy, box_width, box_height) = (values[0], values[1], values[2], values[3]);
        output.extend_from_slice(&[
            (cx - 0.5 * box_width) * width as f32,
            (cy - 0.5 * box_height) * height as f32,
            (cx + 0.5 * box_width) * width as f32,
            (cy + 0.5 * box_height) * height as f32,
        ]);
    }
    output
}

fn encode_1d(output: &mut Vec<f32>, coordinate: f32, dim: usize) {
    for lane in 0..dim {
        let denominator = 10_000.0f32.powf((2 * (lane / 2)) as f32 / dim as f32);
        let angle = coordinate * TAU / denominator;
        output.push(if lane % 2 == 0 {
            angle.sin()
        } else {
            angle.cos()
        });
    }
}

fn read_f32(array: &MlxArray) -> Vec<f32> {
    mlxcel_core::array_to_raw_bytes(array)
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect()
}

fn stack_arrays(arrays: &[UniquePtr<MlxArray>], axis: i32) -> UniquePtr<MlxArray> {
    let pointers: Vec<*const MlxArray> = arrays
        .iter()
        .filter_map(|array| array.as_ref().map(|value| value as *const MlxArray))
        .collect();
    mlxcel_core::stack(&pointers, axis)
}
