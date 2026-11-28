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

//! The fixed OmniParser `icon_detect` YOLO11 graph.
//!
//! The 23-entry layer table is data from
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/yolo11/yolo11.py#L306-L330;
//! its `from` indices are the predecessor feature index, `-1` meaning the
//! previous layer's output. Detection taps layers 16, 19 and 22 (P3 / P4 / P5).

use std::path::Path;

use mlxcel_core::utils::slice_axis;
use mlxcel_core::weights::load_weights_from_dir;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::Yolo11Config;
use super::layers::{BiasConv, C2Psa, C3k2, Conv, Sppf, upsample2x};
use super::weights::Loader;

enum Layer {
    Conv(Conv),
    C3k2(C3k2),
    Sppf(Sppf),
    C2Psa(C2Psa),
    Upsample,
    Concat([usize; 2]),
}

impl Layer {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Layer::Conv(layer) => layer.forward(x),
            Layer::C3k2(layer) => layer.forward(x),
            Layer::Sppf(layer) => layer.forward(x),
            Layer::C2Psa(layer) => layer.forward(x),
            Layer::Upsample => upsample2x(x),
            Layer::Concat(_) => unreachable!("concat is resolved in the model loop"),
        }
    }
}

struct BoxBranch {
    first: Conv,
    second: Conv,
    project: BiasConv,
}

struct ClassBranch {
    depthwise_first: Conv,
    project_first: Conv,
    depthwise_second: Conv,
    project_second: Conv,
    project: BiasConv,
}

/// Anchor-free DFL detection head.
struct Detect {
    classes: usize,
    reg_max: usize,
    box_branch: Vec<BoxBranch>,
    class_branch: Vec<ClassBranch>,
    dfl: UniquePtr<MlxArray>,
}

impl Detect {
    fn load(loader: &mut Loader, prefix: &str, config: &Yolo11Config) -> Result<Self, String> {
        let classes = config.nc;
        let reg_max = config.reg_max;
        let box_hidden = (config.ch[0] / 4).max(16).max(reg_max * 4);
        // Ultralytics `max(min(256, nc * 400), 256)`. For any nc >= 1 the inner
        // min is already 256, so the classification width is always 256.
        let class_hidden = 256usize;
        let mut box_branch = Vec::with_capacity(config.ch.len());
        let mut class_branch = Vec::with_capacity(config.ch.len());
        for (index, &channels) in config.ch.iter().enumerate() {
            let box_prefix = format!("{prefix}.cv2.{index}");
            box_branch.push(BoxBranch {
                first: Conv::load(
                    loader,
                    &format!("{box_prefix}.0"),
                    channels,
                    box_hidden,
                    3,
                    1,
                    1,
                    true,
                )?,
                second: Conv::load(
                    loader,
                    &format!("{box_prefix}.1"),
                    box_hidden,
                    box_hidden,
                    3,
                    1,
                    1,
                    true,
                )?,
                project: BiasConv::load(
                    loader,
                    &format!("{box_prefix}.2"),
                    box_hidden,
                    reg_max * 4,
                    1,
                )?,
            });

            let class_prefix = format!("{prefix}.cv3.{index}");
            class_branch.push(ClassBranch {
                depthwise_first: Conv::load(
                    loader,
                    &format!("{class_prefix}.0.0"),
                    channels,
                    channels,
                    3,
                    1,
                    channels,
                    true,
                )?,
                project_first: Conv::load(
                    loader,
                    &format!("{class_prefix}.0.1"),
                    channels,
                    class_hidden,
                    1,
                    1,
                    1,
                    true,
                )?,
                depthwise_second: Conv::load(
                    loader,
                    &format!("{class_prefix}.1.0"),
                    class_hidden,
                    class_hidden,
                    3,
                    1,
                    class_hidden,
                    true,
                )?,
                project_second: Conv::load(
                    loader,
                    &format!("{class_prefix}.1.1"),
                    class_hidden,
                    class_hidden,
                    1,
                    1,
                    1,
                    true,
                )?,
                project: BiasConv::load(
                    loader,
                    &format!("{class_prefix}.2"),
                    class_hidden,
                    classes,
                    1,
                )?,
            });
        }
        let dfl_weight = loader.take(&format!("{prefix}.dfl.weight"), &[1, 1, 1, reg_max])?;
        let dfl = mlxcel_core::reshape(&dfl_weight, &[reg_max as i32, 1]);
        Ok(Self {
            classes,
            reg_max,
            box_branch,
            class_branch,
            dfl,
        })
    }

    fn forward(&self, xs: [&MlxArray; 3]) -> Result<UniquePtr<MlxArray>, String> {
        let mut boxes = Vec::with_capacity(3);
        let mut scores = Vec::with_capacity(3);
        let mut anchor_x: Vec<f32> = Vec::new();
        let mut anchor_y: Vec<f32> = Vec::new();
        let mut strides: Vec<f32> = Vec::new();
        let mut batch = 0i32;
        let mut total = 0i32;
        for (index, x) in xs.iter().enumerate() {
            let shape = mlxcel_core::array_shape(x);
            if shape.len() != 4 || shape.iter().any(|&dimension| dimension <= 0) {
                return Err(format!(
                    "YOLO11 detection input {index} must be a nonempty NHWC tensor, got {shape:?}"
                ));
            }
            let (b, h, w) = (shape[0], shape[1], shape[2]);
            if index == 0 {
                batch = b;
            } else if b != batch {
                return Err("YOLO11 detection scales disagree on batch size".into());
            }
            let positions = h * w;
            total += positions;

            let branch = &self.box_branch[index];
            let raw = branch
                .project
                .forward(&branch.second.forward(&branch.first.forward(x)));
            boxes.push(mlxcel_core::transpose_axes(
                &mlxcel_core::reshape(&raw, &[b, positions, (self.reg_max * 4) as i32]),
                &[0, 2, 1],
            ));
            let branch = &self.class_branch[index];
            let raw = branch.project.forward(
                &branch.project_second.forward(
                    &branch.depthwise_second.forward(
                        &branch
                            .project_first
                            .forward(&branch.depthwise_first.forward(x)),
                    ),
                ),
            );
            scores.push(mlxcel_core::transpose_axes(
                &mlxcel_core::reshape(&raw, &[b, positions, self.classes as i32]),
                &[0, 2, 1],
            ));

            for row in 0..h {
                for column in 0..w {
                    anchor_x.push(column as f32 + 0.5);
                    anchor_y.push(row as f32 + 0.5);
                }
            }
            let stride = super::config::DETECT_STRIDES[index] as f32;
            strides.extend(std::iter::repeat_n(stride, positions as usize));
        }

        let box_refs: Vec<&MlxArray> = boxes.iter().map(|value| &**value).collect();
        let score_refs: Vec<&MlxArray> = scores.iter().map(|value| &**value).collect();
        let boxes = mlxcel_core::concatenate_many(&box_refs, -1);
        let scores = mlxcel_core::concatenate_many(&score_refs, -1);

        // The anchor tensor is channel-major `(1, 2, N)`: every x, then every y.
        anchor_x.extend(anchor_y);
        let distances = self.distributional_decode(&boxes, batch, total)?;
        let anchor_points = mlxcel_core::from_slice_f32(&anchor_x, &[1, 2, total]);
        let stride_tensor = mlxcel_core::from_slice_f32(&strides, &[1, 1, total]);
        let left_top = slice_axis(&distances, 1, 0, 2);
        let right_bottom = slice_axis(&distances, 1, 2, 4);
        let top_left = mlxcel_core::subtract(&anchor_points, &left_top);
        let bottom_right = mlxcel_core::add(&anchor_points, &right_bottom);
        let centers =
            mlxcel_core::multiply_scalar(&mlxcel_core::add(&top_left, &bottom_right), 0.5);
        let sizes = mlxcel_core::subtract(&bottom_right, &top_left);
        let decoded = mlxcel_core::multiply(
            &mlxcel_core::concatenate(&centers, &sizes, 1),
            &stride_tensor,
        );
        Ok(mlxcel_core::concatenate(
            &decoded,
            &mlxcel_core::sigmoid(&scores),
            1,
        ))
    }

    /// DFL: `(b, 4 * reg_max, n)` to `(b, 4, n)` distances in grid units.
    fn distributional_decode(
        &self,
        boxes: &MlxArray,
        batch: i32,
        total: i32,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let reg_max = self.reg_max as i32;
        let transposed = mlxcel_core::transpose_axes(boxes, &[0, 2, 1]);
        let flattened = mlxcel_core::reshape(&transposed, &[batch * total, 4, reg_max]);
        let weights = mlxcel_core::softmax(&flattened, -1);
        let summed = mlxcel_core::matmul(&weights, &self.dfl);
        let reshaped = mlxcel_core::reshape(&summed, &[batch, total, 4]);
        Ok(mlxcel_core::transpose_axes(&reshaped, &[0, 2, 1]))
    }
}

pub struct Yolo11Model {
    config: Yolo11Config,
    layers: Vec<Layer>,
    detect: Detect,
}

impl Yolo11Model {
    pub fn load<P: AsRef<Path>>(directory: P) -> Result<Self, String> {
        let directory = directory.as_ref();
        let config_path = directory.join("config.json");
        let contents = std::fs::read_to_string(&config_path)
            .map_err(|error| format!("failed to read {}: {error}", config_path.display()))?;
        let config = Yolo11Config::from_json_str(&contents)?;
        let weights = load_weights_from_dir(directory)?;
        let mut loader = Loader::new(weights)?;
        let mut layers = Vec::with_capacity(23);
        for index in 0..23 {
            layers.push(load_layer(&mut loader, index)?);
        }
        let detect = Detect::load(&mut loader, "model.23", &config)?;
        loader.finish()?;
        Ok(Self {
            config,
            layers,
            detect,
        })
    }

    pub fn config(&self) -> &Yolo11Config {
        &self.config
    }

    /// Run the graph on a `(1, H, W, 3)` NHWC tensor in `[0, 1]`. Returns
    /// `(1, 4 + nc, N)` decoded `xywh` boxes in input pixels plus sigmoid
    /// class scores.
    pub fn forward(&self, input: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let shape = mlxcel_core::array_shape(input);
        if shape.len() != 4 || shape[0] != 1 || shape[3] != 3 {
            return Err(format!(
                "YOLO11 expects a (1, H, W, 3) NHWC input, got {shape:?}"
            ));
        }
        if shape[1] <= 0 || shape[2] <= 0 {
            return Err(format!("YOLO11 input has no pixels: {shape:?}"));
        }
        let mut features: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let output = match layer {
                Layer::Concat([first, second]) => {
                    mlxcel_core::concatenate(&features[*first], &features[*second], 3)
                }
                _ => {
                    let previous: &MlxArray =
                        features.last().map(|feature| &**feature).unwrap_or(input);
                    layer.forward(previous)
                }
            };
            features.push(output);
        }
        self.detect
            .forward([&features[16], &features[19], &features[22]])
    }
}

/// Build one layer from the OmniParser table at `index`.
fn load_layer(loader: &mut Loader, index: usize) -> Result<Layer, String> {
    let prefix = format!("model.{index}");
    Ok(match index {
        0 => Layer::Conv(Conv::load(loader, &prefix, 3, 64, 3, 2, 1, true)?),
        1 => Layer::Conv(Conv::load(loader, &prefix, 64, 128, 3, 2, 1, true)?),
        2 => Layer::C3k2(C3k2::load(loader, &prefix, 128, 256, 1, 0.25, true, 1)?),
        3 => Layer::Conv(Conv::load(loader, &prefix, 256, 256, 3, 2, 1, true)?),
        4 => Layer::C3k2(C3k2::load(loader, &prefix, 256, 512, 1, 0.25, true, 1)?),
        5 => Layer::Conv(Conv::load(loader, &prefix, 512, 512, 3, 2, 1, true)?),
        6 => Layer::C3k2(C3k2::load(loader, &prefix, 512, 512, 1, 0.5, true, 1)?),
        7 => Layer::Conv(Conv::load(loader, &prefix, 512, 512, 3, 2, 1, true)?),
        8 => Layer::C3k2(C3k2::load(loader, &prefix, 512, 512, 1, 0.5, true, 1)?),
        9 => Layer::Sppf(Sppf::load(loader, &prefix, 512, 512, 5)?),
        10 => Layer::C2Psa(C2Psa::load(loader, &prefix, 512, 1, 0.5)?),
        11 => Layer::Upsample,
        12 => Layer::Concat([11, 6]),
        13 => Layer::C3k2(C3k2::load(loader, &prefix, 1024, 512, 1, 0.5, true, 1)?),
        14 => Layer::Upsample,
        15 => Layer::Concat([14, 4]),
        16 => Layer::C3k2(C3k2::load(loader, &prefix, 1024, 256, 1, 0.5, true, 1)?),
        17 => Layer::Conv(Conv::load(loader, &prefix, 256, 256, 3, 2, 1, true)?),
        18 => Layer::Concat([17, 13]),
        19 => Layer::C3k2(C3k2::load(loader, &prefix, 768, 512, 1, 0.5, true, 1)?),
        20 => Layer::Conv(Conv::load(loader, &prefix, 512, 512, 3, 2, 1, true)?),
        21 => Layer::Concat([20, 10]),
        22 => Layer::C3k2(C3k2::load(loader, &prefix, 1024, 512, 1, 0.5, true, 1)?),
        _ => return Err(format!("YOLO11 layer index {index} is out of range")),
    })
}
