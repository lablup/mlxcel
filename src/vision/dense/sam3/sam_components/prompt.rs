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

use super::super::config::PromptEncoderConfig;
use super::layers::{Conv2dLayer, LayerNorm2d, load_array};

/// Coordinate and label tensors for point prompts.
#[derive(Clone, Copy)]
pub struct PointPrompts<'a> {
    /// Pixel coordinates with shape `[batch, points, 2]`.
    pub coordinates: &'a MlxArray,
    /// Labels with shape `[batch, points]`; `-1` denotes padding.
    pub labels: &'a MlxArray,
}

pub(crate) struct PositionEmbedding {
    positional_embedding: UniquePtr<MlxArray>,
}

impl PositionEmbedding {
    pub(crate) fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            positional_embedding: load_array(weights, &format!("{prefix}.positional_embedding"))?,
        })
    }

    pub(crate) fn grid(&self, size: (i32, i32)) -> UniquePtr<MlxArray> {
        let mut coordinates = Vec::with_capacity((size.0 * size.1 * 2) as usize);
        for y in 0..size.0 {
            for x in 0..size.1 {
                coordinates.push(x as f32 / size.1 as f32);
                coordinates.push(y as f32 / size.0 as f32);
            }
        }
        let coordinates = mlxcel_core::from_slice_f32(&coordinates, &[1, size.0 * size.1, 2]);
        mlxcel_core::squeeze_axis(&self.forward(&coordinates), 0)
    }

    pub(crate) fn forward(&self, coordinates: &MlxArray) -> UniquePtr<MlxArray> {
        let centered = mlxcel_core::subtract(
            &mlxcel_core::multiply_scalar(coordinates, 2.0),
            &mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::array_dtype(coordinates)),
        );
        let projected = mlxcel_core::multiply_scalar(
            &mlxcel_core::matmul(&centered, &self.positional_embedding),
            2.0 * std::f32::consts::PI,
        );
        mlxcel_core::concatenate(
            &mlxcel_core::sin(&projected),
            &mlxcel_core::cos(&projected),
            -1,
        )
    }
}

struct MaskEmbedConvs {
    conv1: Conv2dLayer,
    layer_norm1: LayerNorm2d,
    conv2: Conv2dLayer,
    layer_norm2: LayerNorm2d,
    conv3: Conv2dLayer,
}

impl MaskEmbedConvs {
    fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            conv1: Conv2dLayer::from_weights(weights, &format!("{prefix}.conv1"), 2, 0, 1, true)?,
            layer_norm1: LayerNorm2d::from_weights(weights, &format!("{prefix}.layer_norm1"))?,
            conv2: Conv2dLayer::from_weights(weights, &format!("{prefix}.conv2"), 2, 0, 1, true)?,
            layer_norm2: LayerNorm2d::from_weights(weights, &format!("{prefix}.layer_norm2"))?,
            conv3: Conv2dLayer::from_weights(weights, &format!("{prefix}.conv3"), 1, 0, 1, true)?,
        })
    }

    fn forward(&self, masks: &MlxArray) -> UniquePtr<MlxArray> {
        let x = self.conv1.forward(masks);
        let x = super::super::activation::gelu(&self.layer_norm1.forward(&x));
        let x = self.conv2.forward(&x);
        let x = super::super::activation::gelu(&self.layer_norm2.forward(&x));
        let x = self.conv3.forward(&x);
        let shape = mlxcel_core::array_shape(&x);
        mlxcel_core::reshape(&x, &[shape[0], shape[1] * shape[2], shape[3]])
    }
}

/// SAM random-Fourier prompt encoder for points, boxes, and masks.
pub struct SamPromptEncoder {
    embed_dim: i32,
    image_embedding_size: (i32, i32),
    point_embed: Embedding,
    not_a_point_embed: Embedding,
    mask_embed: MaskEmbedConvs,
    no_mask_embed: Embedding,
    shared_embedding: PositionEmbedding,
}

impl SamPromptEncoder {
    pub fn from_weights(
        weights: &WeightMap,
        config: &PromptEncoderConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let image_size = config.image_size as i32;
        let patch_size = config.patch_size as i32;
        if patch_size <= 0 || image_size % patch_size != 0 {
            return Err(format!(
                "SAM3 prompt image size {image_size} must be divisible by patch size {patch_size}"
            ));
        }
        Ok(Self {
            embed_dim: config.hidden_size as i32,
            image_embedding_size: (image_size / patch_size, image_size / patch_size),
            point_embed: Embedding::from_weights(weights, &format!("{prefix}.point_embed"))?,
            not_a_point_embed: Embedding::from_weights(
                weights,
                &format!("{prefix}.not_a_point_embed"),
            )?,
            mask_embed: MaskEmbedConvs::from_weights(weights, &format!("{prefix}.mask_embed"))?,
            no_mask_embed: Embedding::from_weights(weights, &format!("{prefix}.no_mask_embed"))?,
            shared_embedding: PositionEmbedding::from_weights(
                weights,
                &format!("{prefix}.shared_embedding"),
            )?,
        })
    }

    /// Dense random-Fourier encoding with shape `[1, height*width, channels]`.
    #[must_use]
    pub fn dense_position_encoding(&self) -> UniquePtr<MlxArray> {
        mlxcel_core::expand_dims(&self.shared_embedding.grid(self.image_embedding_size), 0)
    }

    fn embed_points(&self, points: PointPrompts<'_>) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(points.coordinates);
        let offset = mlxcel_core::full_f32(&[1], 0.5, mlxcel_core::array_dtype(points.coordinates));
        let coordinates = mlxcel_core::add(points.coordinates, &offset);
        let normalizer = mlxcel_core::from_slice_f32(
            &[
                self.image_embedding_size.1 as f32,
                self.image_embedding_size.0 as f32,
            ],
            &[1, 1, 2],
        );
        let positional = self
            .shared_embedding
            .forward(&mlxcel_core::divide(&coordinates, &normalizer));
        let labels = mlxcel_core::astype(points.labels, mlxcel_core::dtype::INT32);
        let padding = mlxcel_core::equal(
            &labels,
            &mlxcel_core::full_f32(&[1], -1.0, mlxcel_core::dtype::INT32),
        );
        let safe_labels = mlxcel_core::where_cond(
            &padding,
            &mlxcel_core::zeros(&shape[..2], mlxcel_core::dtype::INT32),
            &labels,
        );
        let embedded = mlxcel_core::add(&positional, &self.point_embed.forward(&safe_labels));
        let padding = mlxcel_core::expand_dims(&padding, -1);
        let not_a_point = mlxcel_core::broadcast_to(
            &mlxcel_core::reshape(&self.not_a_point_embed.weight, &[1, 1, self.embed_dim]),
            &[shape[0], shape[1], self.embed_dim],
        );
        mlxcel_core::where_cond(&padding, &not_a_point, &embedded)
    }

    fn embed_boxes(&self, boxes: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(boxes);
        let corners = mlxcel_core::reshape(boxes, &[shape[0], shape[1] * 2, 2]);
        let positional = self.shared_embedding.forward(&corners);
        let mut labels = Vec::with_capacity((shape[1] * 2) as usize);
        for _ in 0..shape[1] {
            labels.extend_from_slice(&[2, 3]);
        }
        let labels = mlxcel_core::from_slice_i32(&labels, &[1, shape[1] * 2]);
        let labels = mlxcel_core::broadcast_to(&labels, &[shape[0], shape[1] * 2]);
        mlxcel_core::add(&positional, &self.point_embed.forward(&labels))
    }

    /// Encode optional prompts into sparse tokens and a dense image-grid term.
    pub fn forward(
        &self,
        points: Option<PointPrompts<'_>>,
        boxes: Option<&MlxArray>,
        masks: Option<&MlxArray>,
    ) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>), String> {
        let batch = points
            .map(|value| mlxcel_core::array_shape(value.coordinates)[0])
            .or_else(|| boxes.map(|value| mlxcel_core::array_shape(value)[0]))
            .or_else(|| masks.map(|value| mlxcel_core::array_shape(value)[0]))
            .unwrap_or(1);
        let mut sparse = mlxcel_core::zeros(
            &[batch, 0, self.embed_dim],
            mlxcel_core::array_dtype(&self.point_embed.weight),
        );
        if let Some(points) = points {
            sparse = mlxcel_core::concatenate(&sparse, &self.embed_points(points), 1);
        }
        if let Some(boxes) = boxes {
            sparse = mlxcel_core::concatenate(&sparse, &self.embed_boxes(boxes), 1);
        }
        let dense = match masks {
            Some(masks) => self.mask_embed.forward(masks),
            None => mlxcel_core::broadcast_to(
                &mlxcel_core::reshape(&self.no_mask_embed.weight, &[1, 1, self.embed_dim]),
                &[
                    batch,
                    self.image_embedding_size.0 * self.image_embedding_size.1,
                    self.embed_dim,
                ],
            ),
        };
        Ok((sparse, dense))
    }
}
