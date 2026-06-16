// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! ResNet-50-vd / ResNet-101-vd backbone for RT-DETRv2.
//!
//! Port of the backbone half of
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/rt_detr_v2/vision.py. Returns features at
//! the strides selected by `out_features` (default stages 2/3/4 -> strides
//! 8/16/32). The `vd` variant uses a 3-conv stem + 3x3 stride-2 maxpool and
//! AvgPool-based downsampling shortcuts.

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::BackboneConfig;
use super::layers::{Activation, ConvBn, ConvNorm, avg_pool_2x2, max_pool_3x3_s2_p1};

const BOTTLENECK_EXPANSION: usize = 4;

/// One of the three shortcut variants in a bottleneck block.
enum ShortCut {
    /// Identity skip (stride 1, channels unchanged) — no parameters.
    Identity,
    /// Plain 1x1 conv + BN (stride-1 channel change).
    Conv(ConvBn),
    /// vd downsampling: AvgPool 2x2 stride 2, then 1x1 conv + BN (the conv is
    /// stored under `.proj`).
    AvgPool(ConvBn),
}

impl ShortCut {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            ShortCut::Identity => mlxcel_core::copy(x),
            ShortCut::Conv(c) => c.forward(x),
            ShortCut::AvgPool(c) => {
                let pooled = avg_pool_2x2(x);
                c.forward(&pooled)
            }
        }
    }
}

/// ResNet bottleneck: 1x1 -> 3x3 -> 1x1 (last has no activation) + shortcut +
/// post-add activation.
struct BottleNeck {
    shortcut: ShortCut,
    conv0: ConvNorm,
    conv1: ConvNorm,
    conv2: ConvNorm,
    act: Activation,
}

impl BottleNeck {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        in_c: usize,
        out_c: usize,
        stride: i32,
        downsample_in_bottleneck: bool,
        act: Activation,
        eps: f32,
    ) -> Result<Self, String> {
        let should_apply_shortcut = (in_c != out_c) || (stride != 1);
        // Conv channel dims (out_c / expansion for the reduce convs) are read
        // from the checkpoint weights by key, so we don't size them here; the
        // expansion constant is documented on `BOTTLENECK_EXPANSION`.
        debug_assert_eq!(out_c % BOTTLENECK_EXPANSION, 0);

        let shortcut = if !should_apply_shortcut {
            ShortCut::Identity
        } else if stride == 2 {
            // AvgPoolShortCut: inner 1x1 conv+BN lives under `.shortcut.proj`.
            ShortCut::AvgPool(ConvBn::from_weights(
                weights,
                &format!("{prefix}.shortcut.proj"),
                1,
                eps,
            )?)
        } else {
            ShortCut::Conv(ConvBn::from_weights(
                weights,
                &format!("{prefix}.shortcut"),
                stride,
                eps,
            )?)
        };

        // stride-2 sits on the first 1x1 if downsample_in_bottleneck, else on
        // the middle 3x3.
        let first_stride = if downsample_in_bottleneck { stride } else { 1 };
        let middle_stride = if downsample_in_bottleneck { 1 } else { stride };

        // layer.0: 1x1 (pad 0), layer.1: 3x3 (pad 1), layer.2: 1x1 (pad 0, no act).
        let conv0 = ConvNorm::from_weights(
            weights,
            &format!("{prefix}.layer.0"),
            first_stride,
            0,
            act,
            eps,
        )?;
        let conv1 = ConvNorm::from_weights(
            weights,
            &format!("{prefix}.layer.1"),
            middle_stride,
            1,
            act,
            eps,
        )?;
        let conv2 = ConvNorm::from_weights(
            weights,
            &format!("{prefix}.layer.2"),
            1,
            0,
            Activation::None,
            eps,
        )?;

        Ok(Self {
            shortcut,
            conv0,
            conv1,
            conv2,
            act,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let residual = self.shortcut.forward(x);
        let y = self.conv0.forward(x);
        let y = self.conv1.forward(&y);
        let y = self.conv2.forward(&y);
        let y = mlxcel_core::add(&y, &residual);
        self.act.apply(&y)
    }
}

/// A ResNet stage: `depth` bottleneck blocks. The first block downsamples /
/// projects; the rest are stride-1 identity-or-projection blocks.
struct Stage {
    blocks: Vec<BottleNeck>,
}

impl Stage {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &BackboneConfig,
        in_c: usize,
        out_c: usize,
        stride: i32,
        depth: usize,
        act: Activation,
        eps: f32,
    ) -> Result<Self, String> {
        let mut blocks = Vec::with_capacity(depth);
        // First block: in_c -> out_c at `stride`.
        blocks.push(BottleNeck::from_weights(
            weights,
            &format!("{prefix}.layers.0"),
            in_c,
            out_c,
            stride,
            cfg.downsample_in_bottleneck,
            act,
            eps,
        )?);
        // Remaining blocks: out_c -> out_c, stride 1.
        for i in 1..depth {
            blocks.push(BottleNeck::from_weights(
                weights,
                &format!("{prefix}.layers.{i}"),
                out_c,
                out_c,
                1,
                cfg.downsample_in_bottleneck,
                act,
                eps,
            )?);
        }
        Ok(Self { blocks })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let mut y = mlxcel_core::copy(x);
        for block in &self.blocks {
            y = block.forward(&y);
        }
        y
    }
}

/// Stem: three 3x3 ConvNorm layers (stride 2/1/1) then a 3x3 stride-2 maxpool.
struct Embeddings {
    conv0: ConvNorm,
    conv1: ConvNorm,
    conv2: ConvNorm,
}

impl Embeddings {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &BackboneConfig,
        act: Activation,
        eps: f32,
    ) -> Result<Self, String> {
        // embedder.0 stride 2, embedder.1/2 stride 1, all 3x3 pad 1.
        let conv0 =
            ConvNorm::from_weights(weights, &format!("{prefix}.embedder.0"), 2, 1, act, eps)?;
        let conv1 =
            ConvNorm::from_weights(weights, &format!("{prefix}.embedder.1"), 1, 1, act, eps)?;
        let conv2 =
            ConvNorm::from_weights(weights, &format!("{prefix}.embedder.2"), 1, 1, act, eps)?;
        let _ = cfg;
        Ok(Self {
            conv0,
            conv1,
            conv2,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let y = self.conv0.forward(x);
        let y = self.conv1.forward(&y);
        let y = self.conv2.forward(&y);
        max_pool_3x3_s2_p1(&y)
    }
}

/// ResNet-vd backbone.
pub struct Backbone {
    embedder: Embeddings,
    stages: Vec<Stage>,
    out_stage_indices: Vec<usize>,
}

impl Backbone {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &BackboneConfig,
        eps: f32,
    ) -> Result<Self, String> {
        let act = Activation::parse(&cfg.hidden_act)?;

        let embedder =
            Embeddings::from_weights(weights, &format!("{prefix}.embedder"), cfg, act, eps)?;

        let mut stages = Vec::with_capacity(cfg.depths.len());
        let mut prev_c = cfg.embedding_size;
        for (i, (&out_c, &depth)) in cfg.hidden_sizes.iter().zip(cfg.depths.iter()).enumerate() {
            let stride = if i == 0 {
                if cfg.downsample_in_first_stage { 2 } else { 1 }
            } else {
                2
            };
            stages.push(Stage::from_weights(
                weights,
                &format!("{prefix}.encoder.stages.{i}"),
                cfg,
                prev_c,
                out_c,
                stride,
                depth,
                act,
                eps,
            )?);
            prev_c = out_c;
        }

        Ok(Self {
            embedder,
            stages,
            out_stage_indices: cfg.out_stage_indices(),
        })
    }

    /// Returns the selected feature maps (NHWC), one per `out_features` entry.
    pub fn forward(&self, pixel_values: &MlxArray) -> Vec<UniquePtr<MlxArray>> {
        let mut x = self.embedder.forward(pixel_values);
        let mut all_stages: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            x = stage.forward(&x);
            all_stages.push(mlxcel_core::copy(&x));
        }
        self.out_stage_indices
            .iter()
            .map(|&i| mlxcel_core::copy(&all_stages[i]))
            .collect()
    }
}
