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

//! Sparse 3D U-Net depth refiner for MoGe3.

use anyhow::{Result, ensure};
use mlxcel_core::layers::LayerNorm;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::{MAX_SPARSE_WORKING_BYTES, RefinerConfig};
use super::sparse::{
    MAX_SPARSE_NEIGHBOR_BYTES, SparseShape, SubmanifoldConv3d, neighbor_construction_peak_bytes,
    neighbor_map_bytes, sparse_pool2x_mean, sparse_upsample2x_nearest,
};
use super::weights::{self, Linear};

struct SparseResidualBlock {
    norm: LayerNorm,
    conv1: SubmanifoldConv3d,
    conv2: SubmanifoldConv3d,
    skip: Option<Linear>,
}
impl SparseResidualBlock {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        channels: usize,
        out_channels: usize,
    ) -> Result<Self> {
        let norm_weight = weights::array(weights, &format!("{prefix}.norm1.weight"))?;
        let norm_bias = weights::array(weights, &format!("{prefix}.norm1.bias"))?;
        ensure!(
            mlxcel_core::array_shape(&norm_weight) == [channels as i32]
                && mlxcel_core::array_shape(&norm_bias) == [channels as i32],
            "MoGe3 refiner norm {prefix} must have [{channels}] weight and bias"
        );
        Ok(Self {
            norm: LayerNorm::new(norm_weight, Some(norm_bias), 1e-6),
            conv1: SubmanifoldConv3d::from_weights(
                weights,
                &format!("{prefix}.conv1"),
                channels,
                out_channels,
            )?,
            conv2: SubmanifoldConv3d::from_weights(
                weights,
                &format!("{prefix}.conv2"),
                out_channels,
                out_channels,
            )?,
            skip: (channels != out_channels)
                .then(|| {
                    Linear::from_weights(
                        weights,
                        &format!("{prefix}.skip_connection"),
                        channels,
                        out_channels,
                    )
                })
                .transpose()?,
        })
    }
    fn forward(
        &self,
        feats: &MlxArray,
        coords: &MlxArray,
        shape: SparseShape,
        neighbor: Option<&MlxArray>,
    ) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)> {
        let h = mlxcel_core::utils::silu(&self.norm.forward(feats));
        let (h, map) = self.conv1.forward(&h, coords, shape, neighbor)?;
        let h = mlxcel_core::utils::silu(&h);
        let (h, map) = self.conv2.forward(&h, coords, shape, Some(&map))?;
        let skip = match &self.skip {
            Some(v) => v.forward(feats),
            None => mlxcel_core::copy(feats),
        };
        Ok((mlxcel_core::add(&h, &skip), map))
    }
}
struct PoolDown {
    linear: Linear,
}
struct NearestUp {
    linear: Linear,
}
struct Skip {
    features: UniquePtr<MlxArray>,
    coords: UniquePtr<MlxArray>,
    shape: SparseShape,
}

pub struct Sparse3dUnet {
    encoder_downsample: usize,
    encoder_channels: usize,
    input_channels: usize,
    channels: Vec<usize>,
    input: Linear,
    encoder_fuse: Linear,
    fuse1: Linear,
    fuse2: Linear,
    down_stages: Vec<Vec<SparseResidualBlock>>,
    downsample: Vec<PoolDown>,
    bottleneck: Vec<SparseResidualBlock>,
    upsample: Vec<NearestUp>,
    up_stages: Vec<Vec<SparseResidualBlock>>,
    output: Linear,
}
impl Sparse3dUnet {
    pub fn from_weights(weights: &WeightMap, config: &RefinerConfig) -> Result<Self> {
        config.validate()?;
        let levels = config.model_channels.len();
        let enc_counts = config
            .encoder_blocks_per_level
            .resolve(levels, "encoder_blocks_per_level")?;
        let dec_counts = config
            .decoder_blocks_per_level
            .resolve(levels - 1, "decoder_blocks_per_level")?;
        let mut down_stages = Vec::with_capacity(levels);
        for (level, (&ch, &count)) in config.model_channels.iter().zip(&enc_counts).enumerate() {
            let mut stage = Vec::with_capacity(count);
            for block in 0..count {
                stage.push(SparseResidualBlock::from_weights(
                    weights,
                    &format!("refiner.down_stages.{level}.{block}"),
                    ch,
                    ch,
                )?)
            }
            down_stages.push(stage)
        }
        let mut downsample = Vec::new();
        for i in 0..levels - 1 {
            downsample.push(PoolDown {
                linear: Linear::from_weights(
                    weights,
                    &format!("refiner.downsample_blocks.{i}.linear"),
                    config.model_channels[i],
                    config.model_channels[i + 1],
                )?,
            })
        }
        let last = *config.model_channels.last().expect("levels validated");
        let mut bottleneck = Vec::new();
        for i in 0..config.bottleneck_blocks {
            bottleneck.push(SparseResidualBlock::from_weights(
                weights,
                &format!("refiner.bottleneck_stage.{i}"),
                last,
                last,
            )?)
        }
        let mut upsample = Vec::new();
        let mut up_stages = Vec::new();
        for (i, level) in (1..levels).rev().enumerate() {
            upsample.push(NearestUp {
                linear: Linear::from_weights(
                    weights,
                    &format!("refiner.upsample_blocks.{i}.linear"),
                    config.model_channels[level],
                    config.model_channels[level - 1],
                )?,
            });
            let mut stage = Vec::new();
            for j in 0..dec_counts[i] {
                let ch = config.model_channels[level - 1];
                stage.push(SparseResidualBlock::from_weights(
                    weights,
                    &format!("refiner.up_stages.{i}.{j}"),
                    ch,
                    ch,
                )?)
            }
            up_stages.push(stage)
        }
        Ok(Self {
            encoder_downsample: config.encoder_downsample,
            encoder_channels: config.encoder_channels,
            input_channels: config.in_channels,
            channels: config.model_channels.clone(),
            input: Linear::from_weights(
                weights,
                "refiner.input_proj",
                config.in_channels,
                config.model_channels[0],
            )?,
            encoder_fuse: Linear::from_weights(
                weights,
                "refiner.encoder_fuse",
                config.encoder_channels,
                last,
            )?,
            fuse1: Linear::from_weights(weights, "refiner.fuse_proj.0", last * 2, last)?,
            fuse2: Linear::from_weights(weights, "refiner.fuse_proj.2", last, last)?,
            down_stages,
            downsample,
            bottleneck,
            upsample,
            up_stages,
            output: Linear::from_weights(
                weights,
                "refiner.out_proj",
                config.model_channels[0],
                config.out_channels,
            )?,
        })
    }
    pub fn forward(
        &self,
        features: &MlxArray,
        coords: &MlxArray,
        mut shape: SparseShape,
        encoder: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>> {
        let es = mlxcel_core::array_shape(encoder);
        ensure!(
            es.len() == 4
                && shape.batch() == es[0] as i64
                && shape.height() == es[1] as i64 * self.encoder_downsample as i64
                && shape.width() == es[2] as i64 * self.encoder_downsample as i64
                && es[3] == self.encoder_channels as i32
                && mlxcel_core::array_dtype(encoder) == mlxcel_core::dtype::FLOAT32,
            "MoGe3 refiner and encoder resolutions are inconsistent"
        );
        let input_shape = mlxcel_core::array_shape(features);
        ensure!(
            input_shape.len() == 2
                && input_shape[0] > 0
                && input_shape[1] == self.input_channels as i32
                && mlxcel_core::array_dtype(features) == mlxcel_core::dtype::FLOAT32,
            "MoGe3 refiner features must be non-empty [M,{}]",
            self.input_channels
        );
        ensure!(
            mlxcel_core::array_shape(coords) == [input_shape[0], 4]
                && mlxcel_core::array_dtype(coords) == mlxcel_core::dtype::INT32,
            "MoGe3 refiner coords must be int32 [M,4]"
        );
        let input_rows = input_shape[0] as usize;
        let input_bytes = input_rows
            .checked_mul(self.channels[0])
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| anyhow::anyhow!("MoGe3 sparse input budget overflow"))?;
        ensure!(
            input_bytes <= MAX_SPARSE_WORKING_BYTES,
            "MoGe3 sparse input projection requires {input_bytes} bytes"
        );
        let mut feats = self.input.forward(features);
        let mut coords = mlxcel_core::copy(coords);
        let levels = self.down_stages.len();
        let transitions = self.downsample.len();
        let mut maps: Vec<Option<UniquePtr<MlxArray>>> = (0..levels).map(|_| None).collect();
        let mut parents: Vec<Option<UniquePtr<MlxArray>>> =
            (0..transitions).map(|_| None).collect();
        let mut skips: Vec<Option<Skip>> = (0..transitions).map(|_| None).collect();
        let mut retained_neighbor_bytes = 0usize;
        let mut retained_skip_bytes = 0usize;
        for i in 0..levels {
            let rows = mlxcel_core::array_shape(&coords)[0] as usize;
            let working_bytes = rows
                .checked_mul(self.channels[i])
                .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
                .ok_or_else(|| anyhow::anyhow!("MoGe3 sparse feature budget overflow"))?;
            ensure!(
                working_bytes <= MAX_SPARSE_WORKING_BYTES,
                "MoGe3 sparse level requires {working_bytes} feature bytes"
            );
            let up_uses_level = i < transitions && !self.up_stages[transitions - 1 - i].is_empty();
            let needs_map = !self.down_stages[i].is_empty()
                || (i + 1 == levels && !self.bottleneck.is_empty())
                || up_uses_level;
            if needs_map {
                retained_neighbor_bytes = retained_neighbor_bytes
                    .checked_add(neighbor_map_bytes(rows)?)
                    .ok_or_else(|| anyhow::anyhow!("MoGe3 retained neighbor budget overflow"))?;
                ensure!(
                    retained_neighbor_bytes <= MAX_SPARSE_NEIGHBOR_BYTES,
                    "MoGe3 retained neighbor maps require {retained_neighbor_bytes} bytes"
                );
                ensure!(
                    retained_neighbor_bytes + retained_skip_bytes <= MAX_SPARSE_WORKING_BYTES,
                    "MoGe3 retained sparse state exceeds {MAX_SPARSE_WORKING_BYTES} bytes"
                );
                let retained_with_map = retained_neighbor_bytes
                    .checked_add(retained_skip_bytes)
                    .ok_or_else(|| anyhow::anyhow!("MoGe3 retained sparse budget overflow"))?;
                let construction_peak =
                    neighbor_construction_peak_bytes(rows, self.channels[i], retained_with_map)?;
                ensure!(
                    construction_peak <= MAX_SPARSE_WORKING_BYTES,
                    "MoGe3 sparse neighbor construction requires {construction_peak} bytes"
                );
            }
            let mut map = None;
            for block in &self.down_stages[i] {
                let (next, next_map) = block.forward(&feats, &coords, shape, map.as_deref())?;
                feats = next;
                map = Some(next_map)
            }
            maps[i] = map;
            if i < transitions {
                let skip_bytes = working_bytes
                    .checked_add(
                        rows.checked_mul(5 * std::mem::size_of::<i32>())
                            .ok_or_else(|| anyhow::anyhow!("MoGe3 sparse skip budget overflow"))?,
                    )
                    .ok_or_else(|| anyhow::anyhow!("MoGe3 sparse skip budget overflow"))?;
                retained_skip_bytes = retained_skip_bytes
                    .checked_add(skip_bytes)
                    .ok_or_else(|| anyhow::anyhow!("MoGe3 retained skip budget overflow"))?;
                ensure!(
                    retained_skip_bytes <= MAX_SPARSE_WORKING_BYTES,
                    "MoGe3 retained sparse skips require {retained_skip_bytes} bytes"
                );
                ensure!(
                    retained_neighbor_bytes + retained_skip_bytes <= MAX_SPARSE_WORKING_BYTES,
                    "MoGe3 retained sparse state exceeds {MAX_SPARSE_WORKING_BYTES} bytes"
                );
                skips[i] = Some(Skip {
                    features: mlxcel_core::copy(&feats),
                    coords: mlxcel_core::copy(&coords),
                    shape,
                });
                let pooled = sparse_pool2x_mean(&feats, &coords, shape)?;
                feats = self.downsample[i].linear.forward(&pooled.features);
                coords = pooled.coords;
                shape = pooled.shape;
                parents[i] = Some(pooled.parent_indices)
            }
        }
        let cs = mlxcel_core::array_shape(&coords);
        let b = mlxcel_core::astype(
            &super::sparse::coord_column(&coords, 0),
            mlxcel_core::dtype::INT64,
        );
        let row = mlxcel_core::astype(
            &super::sparse::coord_column(&coords, 1),
            mlxcel_core::dtype::INT64,
        );
        let col = mlxcel_core::astype(
            &super::sparse::coord_column(&coords, 2),
            mlxcel_core::dtype::INT64,
        );
        let index = mlxcel_core::add(
            &mlxcel_core::multiply(
                &mlxcel_core::add(
                    &mlxcel_core::multiply(&b, &mlxcel_core::from_slice_i64(&[es[1] as i64], &[1])),
                    &row,
                ),
                &mlxcel_core::from_slice_i64(&[es[2] as i64], &[1]),
            ),
            &col,
        );
        let enc_flat = mlxcel_core::reshape(encoder, &[es[0] * es[1] * es[2], es[3]]);
        let enc = self
            .encoder_fuse
            .forward(&mlxcel_core::take(&enc_flat, &index, 0));
        let fused = mlxcel_core::concatenate(&feats, &enc, -1);
        feats = self
            .fuse2
            .forward(&mlxcel_core::utils::silu(&self.fuse1.forward(&fused)));
        let mut map = maps[levels - 1].take();
        for block in &self.bottleneck {
            let (next, next_map) = block.forward(&feats, &coords, shape, map.as_deref())?;
            feats = next;
            map = Some(next_map)
        }
        for i in 0..transitions {
            let level = levels - 2 - i;
            let skip = skips[level].take().expect("MoGe3 refiner skip");
            let projected = self.upsample[i].linear.forward(&feats);
            let (up, up_coords, up_shape) = sparse_upsample2x_nearest(
                &projected,
                parents[level].as_deref().expect("MoGe3 parent map"),
                &skip.coords,
                skip.shape,
            )?;
            coords = up_coords;
            shape = up_shape;
            feats = mlxcel_core::add(&up, &skip.features);
            let mut map = maps[level].take();
            for block in &self.up_stages[i] {
                let (next, next_map) = block.forward(&feats, &coords, shape, map.as_deref())?;
                feats = next;
                map = Some(next_map)
            }
        }
        ensure!(cs[0] > 0, "MoGe3 bottleneck coords must be non-empty");
        Ok(self.output.forward(&feats))
    }
}
